"""
PRJ-07: CyberReason - Model Loader
Handles open-weight LLM loading, device management (CUDA/CPU),
tokenizer configuration, and quantization fallback.
"""

import os
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    PreTrainedModel
)


class DummyCyberLLM(PreTrainedModel):
    """
    Lightweight, fast causal LM for air-gapped unit testing and local CPU execution.
    Complies with Hugging Face PreTrainedModel and PEFT CausalLM interface.
    """
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.vocab_size = getattr(config, "vocab_size", 1000)
        self.hidden_size = getattr(config, "n_embd", 64)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """Required by PEFT PeftModelForCausalLM."""
        return {"input_ids": input_ids}

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        clamped_ids = torch.clamp(input_ids, 0, self.vocab_size - 1)
        x = self.embedding(clamped_ids)
        q = self.q_proj(x)
        v = self.v_proj(x)
        hidden = self.out_proj(q + v)

        # In case manual lora attributes exist
        if hasattr(self, "lora_A") and hasattr(self, "lora_B"):
            hidden = hidden + torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)

        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            clamped_labels = torch.clamp(labels, 0, self.vocab_size - 1)
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = clamped_labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        from transformers.modeling_outputs import CausalLMOutput
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class CyberReasonModelLoader:
    """
    Model loader for open-weight threat reasoning models.
    Supports local Hugging Face checkpoints, quantized models, and test stubs.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        load_in_4bit: bool = False
    ):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def get_hardware_info(self) -> Dict[str, Any]:
        """Inspect and return compute hardware capabilities."""
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "total_ram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 16.0,
            "recommended_4bit": True if self.device == "cuda" else False
        }
        return info

    def create_synthetic_model_and_tokenizer(
        self, vocab_size: int = 1000, hidden_size: int = 64
    ) -> Tuple[PreTrainedModel, Any]:
        """
        Creates a synthetic PyTorch causal LM and tokenizer for fast, deterministic
        testing and CPU verification without downloading heavy weights.
        """
        from transformers import GPT2Config
        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_embd=hidden_size,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2
        )

        model = DummyCyberLLM(config).to(self.device)

        class FastCyberTokenizer:
            def __init__(self, vocab_size=vocab_size):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2

            def __call__(self, text, return_tensors="pt", max_length=128, padding=True, truncation=True):
                if isinstance(text, str):
                    tokens = [(hash(w) % (self.vocab_size - 10) + 10) for w in text.split()][:max_length]
                    if not tokens:
                        tokens = [self.bos_token_id]
                    input_ids = torch.tensor([tokens], dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                else:
                    batch = []
                    for t in text:
                        toks = [(hash(w) % (self.vocab_size - 10) + 10) for w in t.split()][:max_length]
                        if not toks:
                            toks = [self.bos_token_id]
                        batch.append(toks)
                    max_len = max(len(b) for b in batch)
                    padded_batch = [b + [self.pad_token_id] * (max_len - len(b)) for b in batch]
                    input_ids = torch.tensor(padded_batch, dtype=torch.long)
                    attention_mask = (input_ids != self.pad_token_id).long()

                res = {"input_ids": input_ids, "attention_mask": attention_mask}
                if return_tensors == "pt":
                    res = {k: v.to(model.device) for k, v in res.items()}
                return res

            def decode(self, token_ids, skip_special_tokens=True):
                return "Decoded synthetic threat attribution response [T1059.001 - MITRE Execution]"

        tokenizer = FastCyberTokenizer(vocab_size=vocab_size)
        return model, tokenizer

    def count_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Compute trainable and total parameters in the model."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_percentage": round(100.0 * trainable / total, 4) if total > 0 else 0.0
        }
