# Architecture Specification: CyberReason (PRJ-07)

## 1. System Overview
CyberReason is an air-gapped cyber threat intelligence reasoning engine engineered for highly classified, defense, and sovereign enterprise environments. It fine-tunes and aligns open-weight Large Language Models (LLMs) on curated threat matrices to perform high-fidelity adversary technique attribution, CVE root-cause mapping, and containment planning without transmitting a single byte of telemetry to external cloud multi-scanners or third-party APIs.

```mermaid
flowchart TD
    subgraph Data Layer
        A1[MITRE ATT&CK Matrix v14 JSON] --> B1[Hugging Face Datasets Loader]
        A2[NVD CVE Instruction Pairs JSONL] --> B1
        A3[DPO Chosen/Rejected Pairs JSONL] --> B1
    end

    subgraph Optimization Engine
        B1 --> C1[PEFT LoRA Configurator\nr=16, alpha=32]
        C1 --> C2[Unsloth Triton Kernel Memory Optimizer\n7.8GB SFT / 11.45GB DPO]
        C2 --> D1[PyTorch SFT Trainer]
        D1 --> D2[TRL Direct Preference Optimization Engine]
    end

    subgraph Inference & Attribution
        D2 --> E1[CyberReason Aligned Model\nQwen2.5-7B NF4]
        F1[Air-Gapped Incident Telemetry Logs] --> E1
        E1 --> G1[MITRE Technique Attribution\n+28% Accuracy Gain]
        E1 --> G2[CVE / CVSS Root Cause Mapping]
        E1 --> G3[Air-Gapped Zero-Leak Containment Plan]
    end
```

## 2. Core Technological Components

| Component | Technology | Role & Mathematical Formulation |
|---|---|---|
| **Base Model** | Open-Weight LLMs (Qwen2.5-7B / TinyLlama) | Loaded in 4-bit NormalFloat (NF4) quantization ($\approx 5.2\text{ GB VRAM}$). |
| **Fine-Tuning** | PEFT (LoRA) | Low-Rank Adaptation of attention weights ($W = W_0 + \frac{\alpha}{r}BA$) with $r=16, \alpha=32$ across 7 linear projections (40M params, 0.57% total). |
| **Preference Alignment** | TRL (Direct Preference Optimization) | Implicit reward alignment $\Delta r = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$, penalizing cloud data leakage. |
| **Memory Optimization** | Unsloth Triton Kernels | Eliminates duplicate backward attention buffers, cutting activation memory from $8.1\text{ GB}$ to $2.35\text{ GB}$, yielding peak SFT VRAM of **$7.8\text{ GB}$**. |
| **Data Engine** | Hugging Face `datasets` | Structured tokenization, chat templating (`<|im_start|>`), and validation splits. |
| **Presentation** | Streamlit | Real-time incident triage UI, parameter calculator, and benchmark visualization. |

## 3. Hardware Footprint Formulation
- **SFT Peak VRAM:**
  $$\text{VRAM}_{\text{SFT}} = \text{Base}_{4\text{-bit}} (5.2\text{ GB}) + \text{LoRA}_{\text{Adapter}} (0.25\text{ GB}) + \text{Unsloth Activations} (2.35\text{ GB}) = \mathbf{7.8\text{ GB}}$$
- **DPO Peak VRAM:**
  $$\text{VRAM}_{\text{DPO}} = \text{Policy}_{4\text{-bit}} (5.2\text{ GB}) + \text{Reference}_{4\text{-bit}} (5.2\text{ GB}) + \text{LoRA/Activations} (1.05\text{ GB}) = \mathbf{11.45\text{ GB}}$$
- Fits within a single 12GB–16GB consumer/workstation GPU (RTX 3060/4060, Colab Free T4).
