"""
PRJ-07: CyberReason - Pipeline Service
High-level service orchestrating dataset staging, LoRA SFT execution,
DPO alignment, Unsloth VRAM calculation, and threat reasoning attribution.
"""

from typing import Dict, Any, Optional
from src.core.dataset_loader import CyberReasonDatasetLoader
from src.core.model_loader import CyberReasonModelLoader
from src.core.lora_trainer import CyberReasonLoRATrainer
from src.core.dpo_aligner import CyberReasonDPOAligner
from src.core.unsloth_optimizer import UnslothMemoryOptimizer
from src.core.threat_reasoner import AirGappedThreatReasoner


class CyberReasonPipelineService:
    """
    Unified end-to-end pipeline service for air-gapped threat intelligence reasoning.
    """

    def __init__(self, base_data_dir: Optional[str] = None):
        self.dataset_loader = CyberReasonDatasetLoader(base_data_dir=base_data_dir)
        self.model_loader = CyberReasonModelLoader(device="cpu")
        self.unsloth_optimizer = UnslothMemoryOptimizer()
        self.threat_reasoner = AirGappedThreatReasoner(dataset_loader=self.dataset_loader)

    def run_hardware_and_dataset_audit(self) -> Dict[str, Any]:
        """Audit compute environment and dataset coverage."""
        hw = self.model_loader.get_hardware_info()
        stats = self.dataset_loader.get_summary_stats()
        sft_mem = self.unsloth_optimizer.calculate_sft_memory()
        dpo_mem = self.unsloth_optimizer.calculate_dpo_memory()

        return {
            "hardware": hw,
            "dataset_summary": stats,
            "sft_vram_requirement_gb": sft_mem["unsloth_peak_vram_gb"],
            "dpo_vram_requirement_gb": dpo_mem["unsloth_dpo_peak_vram_gb"],
            "colab_t4_compatible": sft_mem["fits_google_colab_t4"],
            "status": "READY_FOR_AIR_GAPPED_TRAINING"
        }

    def execute_lightweight_finetuning_simulation(self) -> Dict[str, Any]:
        """Run a fast synthetic LoRA + DPO training step simulation for tests and demo."""
        model, tokenizer = self.model_loader.create_synthetic_model_and_tokenizer()
        lora_trainer = CyberReasonLoRATrainer(model, r=16, lora_alpha=32)
        sft_history = lora_trainer.train_synthetic_epochs(num_steps=3)

        dpo_aligner = CyberReasonDPOAligner(policy_model=lora_trainer.peft_model, beta=0.1)
        dpo_history = dpo_aligner.simulate_preference_alignment(steps=2)

        return {
            "adapter_summary": lora_trainer.get_parameter_summary(),
            "sft_training_history": sft_history,
            "dpo_alignment_history": dpo_history,
            "status": "COMPLETED"
        }

    def attribute_incident(self, telemetry: str, mode: str = "finetuned") -> Dict[str, Any]:
        """Attribute security incident telemetry log."""
        return self.threat_reasoner.attribute_threat_telemetry(telemetry, mode=mode)

    def evaluate_benchmark(self) -> Dict[str, Any]:
        """Run the full benchmark evaluation against baseline."""
        return self.threat_reasoner.run_benchmark_evaluation()
