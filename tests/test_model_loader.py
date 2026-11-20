"""
Unit tests for CyberReasonModelLoader.
"""

import pytest
import torch
from src.core.model_loader import CyberReasonModelLoader, DummyCyberLLM


@pytest.fixture
def model_loader():
    return CyberReasonModelLoader(device="cpu")


def test_get_hardware_info(model_loader):
    info = model_loader.get_hardware_info()
    assert "device" in info
    assert "cuda_available" in info
    assert "total_ram_gb" in info
    assert info["total_ram_gb"] > 0


def test_create_synthetic_model_and_tokenizer(model_loader):
    model, tokenizer = model_loader.create_synthetic_model_and_tokenizer(vocab_size=500, hidden_size=32)
    assert isinstance(model, torch.nn.Module)
    assert hasattr(tokenizer, "pad_token_id")

    # Test tokenization
    res = tokenizer("Alert: suspicious powershell execution")
    assert "input_ids" in res
    assert "attention_mask" in res
    assert res["input_ids"].shape[0] == 1


def test_count_parameters(model_loader):
    model, _ = model_loader.create_synthetic_model_and_tokenizer(vocab_size=200, hidden_size=16)
    param_counts = model_loader.count_parameters(model)
    assert "total_parameters" in param_counts
    assert "trainable_parameters" in param_counts
    assert param_counts["total_parameters"] > 0
    assert param_counts["trainable_percentage"] > 0.0
