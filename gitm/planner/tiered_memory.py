"""Opt-in named memory-resource accounting; no whole-step scheduler.

Inputs describe physical demands or explicitly unknown demands. Logical tensor
volume must not be passed as physical traffic without a lowering assumption.
No existing graph builder or catalogue entry is changed by this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _number(value: float | None, name: str, *, positive: bool = False):
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite or unknown')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'invalid {name}')


@dataclass(frozen=True)
class Provenance:
    kind: str
    artifact: str
    qualification: str = ''

    def __post_init__(self):
        if self.kind not in {'measured', 'compiler-derived', 'source-derived', 'scenario', 'unknown'} or not self.artifact.strip():
            raise ValueError('evidence kind and artifact are required')


@dataclass(frozen=True)
class Bandwidth:
    value: float | None
    bound_kind: str  # point, lower, upper, range, unknown
    rate_kind: str  # peak or sustainable, independent of bound direction
    evidence: Provenance
    upper_value: float | None = None  # range: value is its lower endpoint

    def __post_init__(self):
        if self.bound_kind not in {'point', 'lower', 'upper', 'range', 'unknown'} or self.rate_kind not in {'peak', 'sustainable'}:
            raise ValueError('invalid bandwidth semantics')
        _number(self.value, 'bandwidth', positive=True)
        _number(self.upper_value, 'upper bandwidth', positive=True)
        if self.bound_kind == 'unknown':
            if self.value is not None or self.upper_value is not None:
                raise ValueError('unknown bandwidth cannot carry a numeric value')
        elif self.value is None:
            raise ValueError('known bandwidth requires a value')
        if self.bound_kind == 'range':
            if self.upper_value is None or self.upper_value < self.value:
                raise ValueError('invalid bandwidth interval')
        elif self.upper_value is not None:
            raise ValueError('upper_value belongs only to range bandwidth')


@dataclass(frozen=True)
class MemoryResource:
    resource_id: str
    bandwidth: Bandwidth
    scope: str


@dataclass(frozen=True)
class MemoryTier:
    name: str
    capacity_bytes: float | None
    scope: str
    read_resource_id: str
    write_resource_id: str
    evidence: Provenance

    def __post_init__(self):
        _number(self.capacity_bytes, 'capacity')


@dataclass(frozen=True)
class TransferPath:
    path_id: str
    source_tier: str
    destination_tier: str
    engine_resource_id: str
    # A known service-time lower bound occupied on the transfer engine per event.
    # None means uncharacterized startup, not zero startup.
    startup_floor_s: float | None
    evidence: Provenance

    def __post_init__(self):
        _number(self.startup_floor_s, 'startup floor')


@dataclass(frozen=True)
class TierDemand:
    tier: str
    bytes: float | None
    evidence: Provenance

    def __post_init__(self):
        _number(self.bytes, 'traffic bytes')


@dataclass(frozen=True)
class Transfer:
    event_id: str
    path_id: str
    bytes: float | None
    evidence: Provenance

    def __post_init__(self):
        _number(self.bytes, 'transfer bytes')


@dataclass(frozen=True)
class RegionTraffic:
    region_id: str
    reads: tuple[TierDemand, ...] = ()
    writes: tuple[TierDemand, ...] = ()
    transfers: tuple[Transfer, ...] = ()
    # False when omitted/unknown traffic components may exist.
    coverage_complete: bool = False

    def __post_init__(self):
        if type(self.coverage_complete) is not bool:
            raise ValueError('coverage_complete must be an explicit boolean')


@dataclass(frozen=True)
class Residency:
    allocation_id: str
    tensor_id: str
    tier: str
    live_start: float
    live_end: float
    bytes: float | None
    evidence: Provenance

    def __post_init__(self):
        _number(self.bytes, 'resident bytes')
        _number(self.live_start, 'lifetime start')
        _number(self.live_end, 'lifetime end')
        if self.live_start is None or self.live_end is None or self.live_end <= self.live_start:
            raise ValueError('residency requires a nonempty half-open lifetime')


@dataclass(frozen=True)
class MemoryPlan:
    resources: tuple[MemoryResource, ...]
    tiers: tuple[MemoryTier, ...]
    paths: tuple[TransferPath, ...] = ()

    def validate(self):
        def index(items, attribute):
            out = {}
            for item in items:
                name = getattr(item, attribute)
                if not name or not name.strip() or name in out:
                    raise ValueError('resource, tier and path IDs must be nonempty and unique')
                out[name] = item
            return out
        resources = index(self.resources, 'resource_id')
        tiers = index(self.tiers, 'name')
        paths = index(self.paths, 'path_id')
        for resource in resources.values():
            if not resource.scope.strip():
                raise ValueError('resource scope is required')
        for tier in tiers.values():
            if not tier.scope.strip() or tier.read_resource_id not in resources or tier.write_resource_id not in resources:
                raise ValueError('tier scope and endpoint resources are required')
        for path in paths.values():
            if path.source_tier not in tiers or path.destination_tier not in tiers or path.engine_resource_id not in resources:
                raise ValueError('transfer path references an unknown endpoint or engine')
            if path.source_tier == path.destination_tier:
                raise ValueError('same-tier movement needs a separately justified representation')
            if path.engine_resource_id in {tiers[path.source_tier].read_resource_id, tiers[path.destination_tier].write_resource_id}:
                raise ValueError('transfer engine must not alias an endpoint byte resource')
        return resources, tiers, paths


def memory_resource_bounds(plan: MemoryPlan, region: RegionTraffic) -> dict:
    """Aggregate contention before division; exclude transfer endpoints from maps.

    A transfer charges source read, destination write and engine once each.
    If IDs coincide, their demands add. Consuming reads remain separate events.
    Resource maxima are optimistic service floors, never attainable timings.
    """
    resources, tiers, paths = plan.validate()
    if not region.region_id.strip():
        raise ValueError('region identity is required')
    demands, startup, evidence, unknown = {}, {}, {}, set()

    def charge(resource, count, provenance):
        demands.setdefault(resource, 0)
        startup.setdefault(resource, 0.0)
        evidence.setdefault(resource, []).append(provenance)
        if count is None:
            unknown.add(resource)
        else:
            demands[resource] += count
            _number(demands[resource], 'aggregate resource bytes')

    for direction, entries in [('read_resource_id', region.reads), ('write_resource_id', region.writes)]:
        for entry in entries:
            if entry.tier not in tiers:
                raise ValueError('traffic references unknown tier')
            charge(getattr(tiers[entry.tier], direction), entry.bytes, entry.evidence)
    event_ids = set()
    startup_unknown = set()
    for transfer in region.transfers:
        if not transfer.event_id.strip() or transfer.event_id in event_ids:
            raise ValueError('transfer event IDs must be unique in the region')
        event_ids.add(transfer.event_id)
        if transfer.path_id not in paths:
            raise ValueError('unknown transfer path')
        path = paths[transfer.path_id]
        for resource in (tiers[path.source_tier].read_resource_id,
                         tiers[path.destination_tier].write_resource_id,
                         path.engine_resource_id):
            charge(resource, transfer.bytes, transfer.evidence)
            evidence[resource].append(path.evidence)
        if transfer.bytes is None:
            # An unknown size might be zero, so no startup is guaranteed.
            startup_unknown.add(path.engine_resource_id)
        elif transfer.bytes > 0:
            if path.startup_floor_s is None:
                startup_unknown.add(path.engine_resource_id)
            else:
                startup[path.engine_resource_id] += path.startup_floor_s
                _number(startup[path.engine_resource_id], 'aggregate startup')

    terms = {}
    complete = region.coverage_complete
    known_floor = 0.0
    for resource, count in demands.items():
        rate = resources[resource].bandwidth
        lower, upper, conditional = None, None, None
        if rate.bound_kind == 'upper':
            lower = count / rate.value
        elif rate.bound_kind == 'lower':
            upper = count / rate.value
        elif rate.bound_kind == 'range':
            lower, upper = count / rate.upper_value, count / rate.value
        elif rate.bound_kind == 'point':
            conditional = count / rate.value
        if lower is not None:
            lower += startup[resource]
            known_floor = max(known_floor, lower)
        else:
            known_floor = max(known_floor, startup[resource])
        for value in (lower, upper, conditional):
            _number(value, 'service result')
        # A startup FLOOR cannot be added to the ideal-service UPPER bound.
        full_term = (resource not in unknown and resource not in startup_unknown and lower is not None)
        complete = complete and full_term
        terms[resource] = {
            'known_bytes': count, 'total_bytes': None if resource in unknown else count,
            'startup_floor_s': startup[resource], 'startup_characterized': resource not in startup_unknown,
            'known_demand_service_floor_s': lower,
            'known_demand_ideal_payload_service_upper_s': upper,
            'conditional_payload_service_s': conditional,
            'bound_kind': rate.bound_kind, 'rate_kind': rate.rate_kind,
            'complete_term_floor': full_term,
            'bandwidth_evidence': rate.evidence, 'demand_evidence': tuple(evidence[resource]),
        }
    return {'region_id': region.region_id, 'resources': terms,
            'coverage_complete': region.coverage_complete,
            'known_memory_service_floor_s': known_floor,
            'complete_memory_resource_floor_s': known_floor if complete else None,
            'whole_step_latency_s': None, 'attainable_region_time_s': None}


def validate_residency(plan: MemoryPlan, allocations: tuple[Residency, ...], *, coverage_complete: bool = False) -> dict:
    """Check aggregate capacity over half-open lifetimes, not bank/layout legality."""
    if type(coverage_complete) is not bool:
        raise ValueError('coverage_complete must be an explicit boolean')
    _, tiers, _ = plan.validate()
    ids = set()
    by_tier = {name: [] for name in tiers}
    for allocation in allocations:
        if not allocation.allocation_id.strip() or not allocation.tensor_id.strip() or allocation.allocation_id in ids:
            raise ValueError('allocation identity must be unique and tensor identity nonempty')
        ids.add(allocation.allocation_id)
        if allocation.tier not in tiers:
            raise ValueError('residency references unknown tier')
        by_tier[allocation.tier].append(allocation)
    results = {}
    for name, tier in tiers.items():
        events, unknown = [], False
        for item in by_tier[name]:
            if item.bytes is None:
                unknown = True
            else:
                events.extend([(item.live_start, item.bytes), (item.live_end, -item.bytes)])
        live, peak = 0, 0
        for _, delta in sorted(events):  # releases sort before allocations at a tie
            live += delta
            peak = max(peak, live)
            _number(peak, 'peak residency')
        if tier.capacity_bytes is not None and peak > tier.capacity_bytes:
            fits = False  # known lower occupancy already exceeds capacity
        elif tier.capacity_bytes is None or unknown or not coverage_complete:
            fits = None
        else:
            fits = True
        results[name] = {'known_peak_bytes': peak, 'capacity_bytes': tier.capacity_bytes,
                         'aggregate_fits': fits, 'placement_legal': None,
                         'capacity_evidence': tier.evidence,
                         'allocation_evidence': tuple(a.evidence for a in by_tier[name])}
    return results


def legacy_gpu_prediction(op, flops, bytes_moved, hw, dtype='fp16', *, estimated=False, serial_launches=0):
    """Exact old scalar GPU path; no inferred tiers or automatic family conversion.

    This adapter returns the original prediction object, not a rederived bound.
    Family/compiler plans must explicitly opt into named-resource accounting.
    """
    from gitm.planner.roofline import roofline
    return roofline(op, flops, bytes_moved, hw, dtype, estimated=estimated, serial_launches=serial_launches)
