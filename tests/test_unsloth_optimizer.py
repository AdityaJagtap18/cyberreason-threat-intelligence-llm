"""
Unit tests for UnslothMemoryOptimizer.
"""

import pytest
from src.core.unsloth_optimizer import UnslothMemoryOptimizer


@pytest.fixture
def optimizer():
    return UnslothMemoryOptimizer()


def test_calculate_sft_memory(optimizer):
    sft = optimizer.calculate_sft_memory()
    assert sft["base_model_4bit_gb"] == 5.20
    assert sft["lora_adapter_gb"] == 0.25
    assert sft["unsloth_activations_gb"] == 2.35
    assert sft["unsloth_peak_vram_gb"] == 7.80  # Exact blueprint formula: 5.2 + 0.25 + 2.35 = 7.8GB
    assert sft["fits_12gb_gpu"] is True
    assert sft["fits_google_colab_t4"] is True
    assert sft["vram_reduction_pct"] > 40.0


def test_calculate_dpo_memory(optimizer):
    dpo = optimizer.calculate_dpo_memory()
    assert dpo["active_model_4bit_gb"] == 5.20
    assert dpo["reference_model_4bit_gb"] == 5.20
    assert dpo["unsloth_dpo_peak_vram_gb"] == 11.45  # Blueprint formula: 5.2 + 5.2 + 1.05 = 11.45GB
    assert dpo["fits_12gb_gpu"] is True  # 11.45GB fits within 12GB VRAM
    assert dpo["fits_16gb_gpu"] is True
    assert dpo["vram_saved_gb"] > 15.0


def test_get_triton_kernel_benchmarks(optimizer):
    benchmarks = optimizer.get_triton_kernel_benchmarks()
    assert len(benchmarks) >= 4
    for b in benchmarks:
        assert "kernel_name" in b
        assert "speedup_ratio" in b
        assert b["speedup_ratio"] > 1.5
        assert b["unsloth_triton_ms"] < b["standard_pytorch_ms"]
