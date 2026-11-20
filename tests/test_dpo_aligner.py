"""
Unit tests for CyberReasonDPOAligner.
"""

import pytest
import torch
from src.core.model_loader import CyberReasonModelLoader
from src.core.dpo_aligner import CyberReasonDPOAligner


@pytest.fixture
def dpo_setup():
    loader = CyberReasonModelLoader(device="cpu")
    model, tokenizer = loader.create_synthetic_model_and_tokenizer(vocab_size=300, hidden_size=32)
    aligner = CyberReasonDPOAligner(
        policy_model=model,
        beta=0.1,
        learning_rate=1e-4
    )
    return aligner, tokenizer


def test_compute_log_probs(dpo_setup):
    aligner, _ = dpo_setup
    input_ids = torch.randint(1, 200, (2, 8))
    logps = aligner.compute_log_probs(aligner.policy_model, input_ids)
    assert logps.shape == (2,)
    assert torch.all(logps <= 0.0)  # Log probabilities are non-positive


def test_compute_dpo_loss(dpo_setup):
    aligner, _ = dpo_setup
    p_chosen = torch.tensor([-5.0, -4.5])
    p_rejected = torch.tensor([-12.0, -11.0])
    ref_chosen = torch.tensor([-5.0, -4.5])
    ref_rejected = torch.tensor([-5.0, -4.5])

    loss, chosen_r, rej_r = aligner.compute_dpo_loss(p_chosen, p_rejected, ref_chosen, ref_rejected)
    assert loss.item() > 0.0
    assert chosen_r.item() > rej_r.item()  # Chosen response should yield higher implicit reward


def test_dpo_step(dpo_setup):
    aligner, _ = dpo_setup
    chosen = torch.randint(1, 200, (2, 8))
    rejected = torch.randint(1, 200, (2, 8))

    step_res = aligner.dpo_step(chosen, rejected)
    assert "dpo_loss" in step_res
    assert "chosen_reward" in step_res
    assert "rejected_reward" in step_res
    assert "reward_margin" in step_res


def test_simulate_preference_alignment(dpo_setup):
    aligner, _ = dpo_setup
    history = aligner.simulate_preference_alignment(steps=3)
    assert len(history) == 3
    assert history[-1]["step"] == 3
