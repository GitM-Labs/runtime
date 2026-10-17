"""Source-derived memory-plan exercises; not deployed workload predictions.

Run from the repository root: python -m scripts.cross_architecture_memory
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

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
    memory_resource_bounds,
    validate_residency,
)


def build_reports():
    trainium = Provenance('source-derived',
        'https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/guides/architecture/trainium2_arch.html',
        'documented 128x512 two-byte tile; whole-chip 3.0TB/s is an optimistic envelope, not core allocation')
    sn = Provenance('source-derived', 'https://arxiv.org/html/2405.07518v2',
                    'DDR-to-HBM activation mechanism and paper-v2 64GiB HBM; no deployed V3.2 mapping')
    scenario = Provenance('scenario', '64MB activation with ten later HBM reads',
                          'Arshia preferred DDR 100GB/s and HBM 1.6TB/s scenario rates; not measured rates')

    def resource(name, value, kind, scope, evidence):
        return MemoryResource(name, Bandwidth(value, kind, 'peak', evidence), scope)

    tr = MemoryPlan((resource('chip_hbm', 3e12, 'upper', 'chip', trainium),
                     resource('core_sbuf', None, 'unknown', 'physical core', trainium),
                     resource('core_dma', None, 'unknown', 'physical core', trainium)),
        (MemoryTier('HBM', None, 'chip', 'chip_hbm', 'chip_hbm', trainium),
         MemoryTier('SBUF', 28*2**20, 'physical core', 'core_sbuf', 'core_sbuf', trainium)),
        (TransferPath('load', 'HBM', 'SBUF', 'core_dma', None, trainium),))
    spatial = MemoryPlan((resource('ddr', 100e9, 'point', 'socket', scenario),
                          resource('hbm', 1.6e12, 'point', 'socket', scenario),
                          resource('activation_engine', None, 'unknown', 'socket', sn)),
        (MemoryTier('DDR', None, 'socket', 'ddr', 'ddr', sn),
         MemoryTier('HBM', 64*2**30, 'socket', 'hbm', 'hbm', sn)),
        (TransferPath('activate', 'DDR', 'HBM', 'activation_engine', None, sn),))
    return {
        'scope': 'source-derived family mechanisms with documented or selected scenario demands; no measured deployment ceiling',
        'Trainium2': memory_resource_bounds(tr, RegionTraffic('load_tile',
            transfers=(Transfer('tile0', 'load', 128*512*2, trainium),), coverage_complete=True)),
        'Trainium2_capacity': validate_residency(tr,
            (Residency('tile_copy', 'tile', 'SBUF', 0, 1, 128*512*2, trainium),), coverage_complete=True),
        'SN40L_activation': memory_resource_bounds(spatial, RegionTraffic('activate',
            transfers=(Transfer('bank0', 'activate', 64_000_000, scenario),), coverage_complete=True)),
        'SN40L_reuse': memory_resource_bounds(spatial, RegionTraffic('reuse',
            reads=(TierDemand('HBM', 10*64_000_000, scenario),), coverage_complete=True)),
        'SN40L_capacity': validate_residency(spatial,
            (Residency('bank_active', 'bank', 'HBM', 0, 11, 64_000_000, scenario),), coverage_complete=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional JSON destination; otherwise stdout')
    args = parser.parse_args()

    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        raise TypeError(type(value).__name__)

    payload = json.dumps(build_reports(), default=encode, indent=2)+'\n'
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
