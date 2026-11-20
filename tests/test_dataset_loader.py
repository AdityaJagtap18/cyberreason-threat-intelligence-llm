"""
Unit tests for CyberReasonDatasetLoader.
"""

import pytest
from datasets import DatasetDict
from src.core.dataset_loader import CyberReasonDatasetLoader


@pytest.fixture
def dataset_loader():
    return CyberReasonDatasetLoader()


def test_load_mitre_ttps(dataset_loader):
    data = dataset_loader.load_mitre_ttps()
    assert isinstance(data, list)
    assert len(data) >= 8
    first = data[0]
    assert "technique_id" in first
    assert "name" in first
    assert "tactic" in first
    assert "indicators" in first
    assert "containment" in first


def test_load_cve_instructions(dataset_loader):
    cve_data = dataset_loader.load_cve_instructions()
    assert isinstance(cve_data, list)
    assert len(cve_data) >= 8
    for item in cve_data:
        assert "instruction" in item
        assert "input" in item
        assert "output" in item
        assert len(item["output"]) > 20


def test_load_dpo_pairs(dataset_loader):
    dpo_data = dataset_loader.load_dpo_pairs()
    assert isinstance(dpo_data, list)
    assert len(dpo_data) >= 5
    for item in dpo_data:
        assert "prompt" in item
        assert "chosen" in item
        assert "rejected" in item
        assert len(item["chosen"]) > len(item["rejected"])


def test_get_sft_hf_dataset(dataset_loader):
    hf_dict = dataset_loader.get_sft_hf_dataset(val_ratio=0.25)
    assert isinstance(hf_dict, DatasetDict)
    assert "train" in hf_dict
    assert "validation" in hf_dict
    assert len(hf_dict["train"]) > 0
    sample = hf_dict["train"][0]
    assert "text" in sample
    assert "<|im_start|>system" in sample["text"]


def test_get_dpo_hf_dataset(dataset_loader):
    hf_dict = dataset_loader.get_dpo_hf_dataset(val_ratio=0.2)
    assert isinstance(hf_dict, DatasetDict)
    assert "train" in hf_dict
    assert "validation" in hf_dict
    assert "chosen" in hf_dict["train"].column_names
    assert "rejected" in hf_dict["train"].column_names


def test_get_summary_stats(dataset_loader):
    stats = dataset_loader.get_summary_stats()
    assert stats["total_mitre_techniques"] >= 8
    assert stats["unique_tactics_covered"] >= 4
    assert stats["total_sft_instruction_pairs"] >= 8
    assert "100% On-Premises" in stats["air_gapped_compliance"]
