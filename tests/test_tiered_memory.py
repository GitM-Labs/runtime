"""Named-resource semantics; fixtures are synthetic, not deployment evidence."""
import random
from dataclasses import asdict, replace

import pytest

from gitm.planner.roofline import HardwareSpec, roofline
from gitm.planner.tiered_memory import (
    Bandwidth,
    MemoryPlan,
    MemoryResource,
    MemoryTier,
    Provenance,
    RegionTraffic,
    Residency,
    TierDemand,
    Transfer,
    TransferPath,
    legacy_gpu_prediction,
    memory_resource_bounds,
    validate_residency,
)

EVIDENCE = Provenance('scenario', 'synthetic unit fixture')


def plan_for_rate(kind='upper', value=64e9, upper=None, capacity=64_000_000):
    rate = Bandwidth(value, kind, 'peak', EVIDENCE, upper)
    return MemoryPlan((MemoryResource('bus', rate, 'device'),),
                      (MemoryTier('HBM', capacity, 'device', 'bus', 'bus', EVIDENCE),))


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '64'])
def test_invalid_traffic_rejected(value):
    with pytest.raises(ValueError):
        TierDemand('HBM', value, EVIDENCE)


@pytest.mark.parametrize('coverage', ['false', 'true', 0, 1, None])
def test_coverage_requires_boolean(coverage):
    with pytest.raises(ValueError):
        RegionTraffic('step', coverage_complete=coverage)
    with pytest.raises(ValueError):
        validate_residency(plan_for_rate(), (), coverage_complete=coverage)


@pytest.mark.parametrize(('kind', 'value', 'upper', 'floor', 'service_upper', 'conditional'), [
    ('upper', 64e9, None, .001, None, None),
    ('lower', 64e9, None, None, .001, None),
    ('range', 64e9, 128e9, .0005, .001, None),
    ('point', 64e9, None, None, None, .001),
    ('unknown', None, None, None, None, None),
])
def test_rate_direction(kind, value, upper, floor, service_upper, conditional):
    result = memory_resource_bounds(plan_for_rate(kind, value, upper),
        RegionTraffic('step', reads=(TierDemand('HBM', 64_000_000, EVIDENCE),), coverage_complete=True))
    term = result['resources']['bus']
    for actual, expected in [(term['known_demand_service_floor_s'], floor),
                             (term['known_demand_ideal_payload_service_upper_s'], service_upper),
                             (term['conditional_payload_service_s'], conditional)]:
        assert actual == (None if expected is None else pytest.approx(expected))
    assert result['complete_memory_resource_floor_s'] == (None if floor is None else pytest.approx(floor))
    assert result['whole_step_latency_s'] is None
    assert result['attainable_region_time_s'] is None


def test_unknown_demand_does_not_disappear():
    result = memory_resource_bounds(plan_for_rate(), RegionTraffic('step',
        reads=(TierDemand('HBM', 64_000_000, EVIDENCE), TierDemand('HBM', None, EVIDENCE)), coverage_complete=True))
    assert result['resources']['bus']['known_bytes'] == 64_000_000
    assert result['resources']['bus']['total_bytes'] is None
    assert result['known_memory_service_floor_s'] == pytest.approx(.001)
    assert result['complete_memory_resource_floor_s'] is None


def test_transfer_endpoints_and_engine_contention():
    base = plan_for_rate()
    dma = MemoryResource('dma', Bandwidth(64e9, 'upper', 'peak', EVIDENCE), 'device')
    plan = MemoryPlan(base.resources+(dma,), base.tiers+(MemoryTier('DDR', None, 'device', 'bus', 'bus', EVIDENCE),),
                      (TransferPath('copy', 'DDR', 'HBM', 'dma', 1e-6, EVIDENCE),))
    result = memory_resource_bounds(plan, RegionTraffic('step', transfers=(
        Transfer('a', 'copy', 32_000_000, EVIDENCE), Transfer('b', 'copy', 32_000_000, EVIDENCE)), coverage_complete=True))
    assert result['resources']['bus']['known_bytes'] == 128_000_000
    assert result['resources']['dma']['known_bytes'] == 64_000_000
    assert result['resources']['dma']['known_demand_service_floor_s'] == pytest.approx(.001002)
    assert result['complete_memory_resource_floor_s'] == pytest.approx(.002)
    with pytest.raises(ValueError):
        replace(plan, paths=(replace(plan.paths[0], engine_resource_id='bus'),)).validate()


def test_capacity_preserves_large_integer_bytes():
    plan = plan_for_rate(capacity=2**53)
    allocation = Residency('copy', 'tensor', 'HBM', 0, 1, 2**53+1, EVIDENCE)
    result = validate_residency(plan, (allocation,), coverage_complete=True)['HBM']
    assert result['known_peak_bytes'] == 2**53+1
    assert result['aggregate_fits'] is False


def test_capacity_sweep_matches_independent_reference():
    rng = random.Random(61819)
    for case in range(1000):
        allocations = []
        for index in range(rng.randrange(1, 16)):
            start = rng.randrange(10)
            allocations.append(Residency(str(index), str(index), 'HBM', start,
                start+rng.randrange(1, 6), rng.randrange(0, 1_000_000), EVIDENCE))
        # Independently sum every live allocation at each start coordinate.
        reference_peak = max(sum(a.bytes for a in allocations if a.live_start <= t < a.live_end)
                             for t in {a.live_start for a in allocations})
        capacity = rng.randrange(0, 5_000_000)
        result = validate_residency(plan_for_rate(capacity=capacity), tuple(allocations), coverage_complete=True)['HBM']
        assert result['known_peak_bytes'] == reference_peak, case
        assert result['aggregate_fits'] == (reference_peak <= capacity), case
        assert result['placement_legal'] is None


@pytest.mark.parametrize('dtype', ['fp16', 'bf16', 'fp32', 'fp8', 'mxfp4'])
@pytest.mark.parametrize('launches', [0, 1, 20])
@pytest.mark.parametrize('estimated', [False, True])
def test_legacy_adapter_exact(dtype, launches, estimated):
    hw = HardwareSpec()
    original = roofline('legacy', 1e9, 64e6, hw, dtype, serial_launches=launches, estimated=estimated)
    adapted = legacy_gpu_prediction('legacy', 1e9, 64e6, hw, dtype, serial_launches=launches, estimated=estimated)
    assert asdict(original) == asdict(adapted)
