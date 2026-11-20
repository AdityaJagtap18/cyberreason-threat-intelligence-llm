"""
PRJ-07: CyberReason - Unsloth Triton Kernel VRAM Calculator
Visualizes GPU memory formulations and kernel optimizations for 7B models.
"""

import streamlit as st
import pandas as pd
from src.core.unsloth_optimizer import UnslothMemoryOptimizer


def render_vram_calculator():
    st.subheader("⚡ Unsloth Triton Kernel Memory Optimization")
    st.markdown(
        "Formulates the GPU memory requirements for fine-tuning a 7B parameter open-weight model "
        "(e.g., Qwen2.5-7B) using 4-bit NormalFloat (NF4) quantization and Unsloth Triton fused kernels."
    )

    optimizer = UnslothMemoryOptimizer()
    sft_mem = optimizer.calculate_sft_memory()
    dpo_mem = optimizer.calculate_dpo_memory()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Supervised Fine-Tuning (SFT)")
        st.metric("Unsloth Peak VRAM", f"{sft_mem['unsloth_peak_vram_gb']} GB")
        st.metric("Standard PyTorch VRAM", f"{sft_mem['standard_total_vram_gb']} GB")
        st.metric("Memory Saved", f"{sft_mem['vram_saved_gb']} GB ({sft_mem['vram_reduction_pct']}%)")

        sft_breakdown = pd.DataFrame([
            {"Component": "Base Model (4-bit NF4)", "Memory (GB)": sft_mem["base_model_4bit_gb"]},
            {"Component": "LoRA Adapter (r=16, 7 modules)", "Memory (GB)": sft_mem["lora_adapter_gb"]},
            {"Component": "Unsloth Fused Activations", "Memory (GB)": sft_mem["unsloth_activations_gb"]},
            {"Component": "Total Peak VRAM", "Memory (GB)": sft_mem["unsloth_peak_vram_gb"]}
        ])
        st.dataframe(sft_breakdown, use_container_width=True)

        if sft_mem["fits_12gb_gpu"]:
            st.success("✅ **Fits single 12GB GPU (RTX 3060/4060, Colab Free T4)**")

    with col2:
        st.markdown("#### Direct Preference Optimization (DPO)")
        st.metric("DPO Peak VRAM (with Gradient Checkpointing)", f"{dpo_mem['unsloth_dpo_peak_vram_gb']} GB")
        st.metric("Unoptimized Full-Precision DPO", f"{dpo_mem['standard_unoptimized_dpo_gb']} GB")
        st.metric("Memory Saved", f"{dpo_mem['vram_saved_gb']} GB")

        dpo_breakdown = pd.DataFrame([
            {"Component": "Active Model (4-bit QLoRA)", "Memory (GB)": dpo_mem["active_model_4bit_gb"]},
            {"Component": "Reference Model (4-bit Frozen)", "Memory (GB)": dpo_mem["reference_model_4bit_gb"]},
            {"Component": "Checkpointed Activations", "Memory (GB)": dpo_mem["lora_and_checkpointed_activations_gb"]},
            {"Component": "Total Peak DPO VRAM", "Memory (GB)": dpo_mem["unsloth_dpo_peak_vram_gb"]}
        ])
        st.dataframe(dpo_breakdown, use_container_width=True)

        if dpo_mem["fits_12gb_gpu"]:
            st.success("✅ **DPO dual-model configuration fits within 12GB–16GB VRAM!**")

    st.markdown("---")
    st.markdown("#### Custom Triton Kernel Performance Benchmarks")
    benchmarks = optimizer.get_triton_kernel_benchmarks()
    df_bench = pd.DataFrame(benchmarks)
    st.dataframe(
        df_bench[["kernel_name", "standard_pytorch_ms", "unsloth_triton_ms", "speedup_ratio", "optimization"]],
        use_container_width=True
    )
