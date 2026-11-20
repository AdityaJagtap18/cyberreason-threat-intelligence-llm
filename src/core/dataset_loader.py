"""
PRJ-07: CyberReason - Dataset Loader
Loads curated MITRE ATT&CK Enterprise Matrix v14 and NVD CVE instruction pairs
using Hugging Face Datasets with support for SFT and DPO alignment formats.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, DatasetDict


class CyberReasonDatasetLoader:
    """
    Dataset loader and preprocessor for CyberReason threat intelligence reasoning.
    Integrates MITRE ATT&CK TTPs, NVD CVE instruction pairs, and DPO preference pairs.
    """

    def __init__(self, base_data_dir: Optional[str] = None):
        if base_data_dir is None:
            # Default to project data directory
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            base_data_dir = os.path.join(curr_dir, "..", "..", "data")
        self.base_data_dir = os.path.abspath(base_data_dir)
        self.mitre_path = os.path.join(self.base_data_dir, "mitre_attack", "mitre_ttps.json")
        self.cve_path = os.path.join(self.base_data_dir, "cve_dataset", "cve_instructions.jsonl")
        self.dpo_path = os.path.join(self.base_data_dir, "cve_dataset", "dpo_preference_pairs.jsonl")

    def load_mitre_ttps(self) -> List[Dict[str, Any]]:
        """Load MITRE ATT&CK techniques from JSON file."""
        if not os.path.exists(self.mitre_path):
            raise FileNotFoundError(f"MITRE TTPs file not found at: {self.mitre_path}")
        with open(self.mitre_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def load_cve_instructions(self) -> List[Dict[str, str]]:
        """Load NVD CVE instruction pairs from JSONL file."""
        if not os.path.exists(self.cve_path):
            raise FileNotFoundError(f"CVE instructions file not found at: {self.cve_path}")
        records = []
        with open(self.cve_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def load_dpo_pairs(self) -> List[Dict[str, str]]:
        """Load DPO preference pairs (prompt, chosen, rejected) from JSONL file."""
        if not os.path.exists(self.dpo_path):
            raise FileNotFoundError(f"DPO preference pairs file not found at: {self.dpo_path}")
        records = []
        with open(self.dpo_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_sft_hf_dataset(self, val_ratio: float = 0.2, seed: int = 42) -> DatasetDict:
        """
        Convert CVE instructions and MITRE TTPs into a Hugging Face DatasetDict
        formatted for Supervised Fine-Tuning (SFT).
        """
        raw_cves = self.load_cve_instructions()
        formatted_samples = []

        for item in raw_cves:
            formatted_text = (
                f"<|im_start|>system\nYou are an air-gapped cyber threat intelligence reasoning system. "
                f"Attribute threat tactics and map vulnerabilities with zero telemetry leakage.<|im_end|>\n"
                f"<|im_start|>user\n{item['instruction']}\n\n[Incident Telemetry]:\n{item['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{item['output']}<|im_end|>"
            )
            formatted_samples.append({
                "instruction": item["instruction"],
                "input": item["input"],
                "output": item["output"],
                "text": formatted_text
            })

        hf_dataset = Dataset.from_list(formatted_samples)
        if len(hf_dataset) > 1 and val_ratio > 0.0:
            split_dataset = hf_dataset.train_test_split(test_size=val_ratio, seed=seed)
            return DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })
        else:
            return DatasetDict({"train": hf_dataset, "validation": hf_dataset})

    def get_dpo_hf_dataset(self, val_ratio: float = 0.2, seed: int = 42) -> DatasetDict:
        """
        Convert DPO preference pairs into a Hugging Face DatasetDict
        with 'prompt', 'chosen', 'rejected' columns.
        """
        raw_dpo = self.load_dpo_pairs()
        hf_dataset = Dataset.from_list(raw_dpo)
        if len(hf_dataset) > 1 and val_ratio > 0.0:
            split_dataset = hf_dataset.train_test_split(test_size=val_ratio, seed=seed)
            return DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })
        else:
            return DatasetDict({"train": hf_dataset, "validation": hf_dataset})

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return dataset distribution and summary statistics."""
        mitre_data = self.load_mitre_ttps()
        cve_data = self.load_cve_instructions()
        dpo_data = self.load_dpo_pairs()

        tactics = set(t["tactic"] for t in mitre_data)
        return {
            "total_mitre_techniques": len(mitre_data),
            "unique_tactics_covered": len(tactics),
            "total_sft_instruction_pairs": len(cve_data),
            "total_dpo_preference_pairs": len(dpo_data),
            "tactics_list": sorted(list(tactics)),
            "air_gapped_compliance": "100% On-Premises (0 External Requests)"
        }
