"""
End-to-end integration tests for CyberReason Pipeline Service.
"""

import pytest
from src.services.pipeline_service import CyberReasonPipelineService


@pytest.fixture
def service():
    return CyberReasonPipelineService()


def test_hardware_and_dataset_audit(service):
    audit = service.run_hardware_and_dataset_audit()
    assert audit["status"] == "READY_FOR_AIR_GAPPED_TRAINING"
    assert audit["sft_vram_requirement_gb"] == 7.80
    assert audit["dpo_vram_requirement_gb"] == 11.45
    assert audit["dataset_summary"]["total_mitre_techniques"] >= 8
    assert audit["dataset_summary"]["total_sft_instruction_pairs"] >= 8


def test_execute_lightweight_finetuning_simulation(service):
    res = service.execute_lightweight_finetuning_simulation()
    assert res["status"] == "COMPLETED"
    assert "adapter_summary" in res
    assert len(res["sft_training_history"]) == 3
    assert len(res["dpo_alignment_history"]) == 2


def test_attribute_incident(service):
    log_sample = "svchost_fake.exe requested PROCESS_VM_READ on lsass.exe"
    result = service.attribute_incident(log_sample, mode="finetuned")
    assert result["technique_id"] == "T1003.001"
    assert result["air_gapped_compliant"] is True


def test_evaluate_benchmark_end_to_end(service):
    bench = service.evaluate_benchmark()
    assert bench["target_benchmark_met"] is True
    assert bench["accuracy_improvement_pct"] >= 28.0
    assert "Zero CVE Telemetry Leaks" in bench["air_gap_validation"]
