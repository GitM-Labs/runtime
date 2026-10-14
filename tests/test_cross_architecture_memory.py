"""Source-derived examples use the installed optional planner API."""
import pytest

from scripts.cross_architecture_memory import build_reports


def test_trainium_named_copy_keeps_unknown_resources():
    report = build_reports()['Trainium2']
    assert {term['total_bytes'] for term in report['resources'].values()} == {131072}
    assert report['resources']['chip_hbm']['known_demand_service_floor_s'] == pytest.approx(131072/3e12)
    assert report['resources']['core_sbuf']['known_demand_service_floor_s'] is None
    assert report['resources']['core_dma']['known_demand_service_floor_s'] is None
    assert report['complete_memory_resource_floor_s'] is None
    assert report['whole_step_latency_s'] is None


def test_spatial_activation_and_reuse_keep_evidence_direction():
    reports = build_reports()
    activate, reuse = reports['SN40L_activation'], reports['SN40L_reuse']
    assert activate['resources']['ddr']['total_bytes'] == 64_000_000
    assert activate['resources']['ddr']['conditional_payload_service_s'] == pytest.approx(.00064)
    assert activate['resources']['hbm']['conditional_payload_service_s'] == pytest.approx(.00004)
    assert set(reuse['resources']) == {'hbm'}
    assert reuse['resources']['hbm']['total_bytes'] == 640_000_000
    assert reuse['resources']['hbm']['conditional_payload_service_s'] == pytest.approx(.0004)
    assert activate['complete_memory_resource_floor_s'] is None
    assert reuse['complete_memory_resource_floor_s'] is None


def test_source_examples_claim_only_aggregate_scenario_fit():
    reports = build_reports()
    assert reports['Trainium2_capacity']['SBUF']['aggregate_fits'] is True
    assert reports['Trainium2_capacity']['HBM']['aggregate_fits'] is None
    assert reports['SN40L_capacity']['HBM']['aggregate_fits'] is True
    assert reports['SN40L_capacity']['DDR']['aggregate_fits'] is None
    assert reports['SN40L_capacity']['HBM']['placement_legal'] is None
