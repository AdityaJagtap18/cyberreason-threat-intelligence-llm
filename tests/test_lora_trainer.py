"""
Unit tests for CyberReasonLoRATrainer.
"""

import os
import pytest
import torch
from src.core.model_loader import CyberReasonModelLoader
from src.core.lora_trainer import CyberReasonLoRATrainer


@pytest.fixture
def trainer_setup():
    loader = CyberReasonModelLoader(device="cpu")
    model, tokenizer = loader.create_synthetic_model_and_tokenizer(vocab_size=300, hidden_size=32)
    trainer = CyberReasonLoRATrainer(
        model=model,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"]
    )
    return trainer, tokenizer


def test_parameter_summary(trainer_setup):
    trainer, _ = trainer_setup
    summary = trainer.get_parameter_summary()
    assert summary["rank"] == 8
    assert summary["alpha"] == 16
    assert summary["scaling_factor"] == 2.0
    assert summary["all_params"] > 0
    assert summary["trainable_params"] > 0


def test_train_step(trainer_setup):
    trainer, tokenizer = trainer_setup
    input_ids = torch.randint(1, 200, (2, 8))
    labels = input_ids.clone()
    loss = trainer.train_step(input_ids, labels)
    assert isinstance(loss, float)
    assert loss > 0.0


def test_train_synthetic_epochs(trainer_setup):
    trainer, _ = trainer_setup
    history = trainer.train_synthetic_epochs(num_steps=3)
    assert len(history) == 3
    assert history[0]["step"] == 1
    assert "loss" in history[0]
    assert history[0]["loss"] > 0.0


def test_save_adapter(trainer_setup, tmp_path):
    trainer, _ = trainer_setup
    out_dir = str(tmp_path / "adapter_test")
    cfg_path = trainer.save_adapter(out_dir)
    assert os.path.exists(cfg_path)
