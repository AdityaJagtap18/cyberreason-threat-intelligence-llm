"""
PRJ-07: CyberReason - Direct Preference Optimization (DPO) Aligner
Implements DPO mathematical objective and alignment mechanics using TRL principles
to align the model toward high-rigor, air-gapped threat attribution.
"""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CyberReasonDPOAligner:
    """
    Direct Preference Optimization (DPO) engine.
    Aligns cyber threat reasoning to prefer rigorous, air-gapped containment
    over insecure recommendations (e.g. leaking telemetry to public SaaS).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        learning_rate: float = 5e-6
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model if reference_model is not None else policy_model
        self.beta = beta
        self.learning_rate = learning_rate

        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        if not trainable_params:
            for p in self.policy_model.parameters():
                p.requires_grad = True
            trainable_params = list(self.policy_model.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes per-sequence log probabilities under the given model.
        Sums the log likelihoods of the tokens.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # Clamp shift labels to logits vocab size
        shift_labels = torch.clamp(shift_labels, 0, shift_logits.size(-1) - 1)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather token log probabilities
        token_log_probs = torch.gather(
            log_probs, dim=2, index=shift_labels.unsqueeze(2)
        ).squeeze(2)

        # Mask padding if provided
        if attention_mask is not None:
            mask = attention_mask[:, 1:].contiguous()
            token_log_probs = token_log_probs * mask

        return token_log_probs.sum(dim=-1)

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates standard DPO loss:
        L_DPO = -E[ log sigma( beta * log(pi(yw)/ref(yw)) - beta * log(pi(yl)/ref(yl)) ) ]
        Returns:
            dpo_loss, chosen_rewards, rejected_rewards
        """
        # Implicit rewards
        chosen_logratios = policy_chosen_logps - ref_chosen_logps
        rejected_logratios = policy_rejected_logps - ref_rejected_logps

        logits = self.beta * (chosen_logratios - rejected_logratios)
        losses = -F.logsigmoid(logits)

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()

        return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()

    def dpo_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor
    ) -> Dict[str, float]:
        """Execute a single DPO optimization step."""
        self.policy_model.train()
        self.optimizer.zero_grad()

        policy_chosen_logps = self.compute_log_probs(self.policy_model, chosen_ids)
        policy_rejected_logps = self.compute_log_probs(self.policy_model, rejected_ids)

        with torch.no_grad():
            ref_chosen_logps = self.compute_log_probs(self.reference_model, chosen_ids)
            ref_rejected_logps = self.compute_log_probs(self.reference_model, rejected_ids)

        loss, chosen_reward, rejected_reward = self.compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )

        loss.backward()
        self.optimizer.step()

        reward_margin = float((chosen_reward - rejected_reward).item())
        return {
            "dpo_loss": round(float(loss.item()), 4),
            "chosen_reward": round(float(chosen_reward.item()), 4),
            "rejected_reward": round(float(rejected_reward.item()), 4),
            "reward_margin": round(reward_margin, 4)
        }

    def simulate_preference_alignment(self, steps: int = 4) -> List[Dict[str, Any]]:
        """Simulate preference optimization trajectory on synthetic vectors."""
        device = next(self.policy_model.parameters()).device
        vocab_size = getattr(self.policy_model, "vocab_size", 300)

        history = []
        for step in range(1, steps + 1):
            chosen = torch.randint(1, min(vocab_size - 1, 100), (2, 12), device=device)
            rejected = torch.randint(1, min(vocab_size - 1, 100), (2, 12), device=device)

            step_res = self.dpo_step(chosen, rejected)
            step_res["step"] = step
            history.append(step_res)

        return history
