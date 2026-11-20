"""
PRJ-07: CyberReason - LoRA Configuration & SFT Panel
Visualizes Parameter-Efficient Fine-Tuning hyperparameters, trainable parameter ratios,
and simulated loss convergence.
"""

import streamlit as st
import pandas as pd
from src.core.lora_trainer import CyberReasonLoRATrainer
from src.core.model_loader import CyberReasonModelLoader


def render_lora_config_panel():
    st.subheader("⚙️ Parameter-Efficient Fine-Tuning (PEFT / LoRA)")
    st.markdown(
        "Fine-tunes open-weight LLMs (Qwen2.5-7B / TinyLlama) on cybersecurity TTPs and CVE datasets "
        "by freezing base weights and training low-rank decomposition matrices ($r=16, \\alpha=32$)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Hyperparameter Settings")
        rank = st.slider("LoRA Rank (r)", min_value=4, max_value=64, value=16, step=4)
        alpha = st.slider("LoRA Alpha (α)", min_value=8, max_value=128, value=32, step=8)
        dropout = st.slider("LoRA Dropout", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
        target_mods = st.multiselect(
            "Target Attention Modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            default=["q_proj", "v_proj"]
        )

    with col2:
        st.markdown("#### Parameter Efficiency Breakdown (7B Model)")
        # Mathematical estimation for Qwen2.5-7B
        base_params = 7_000_000_000
        lora_params = len(target_mods) * (2 * 4096 * rank * 28)  # 28 transformer layers
        trainable_pct = round(100.0 * lora_params / base_params, 4)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Base Parameters", "7.00 Billion")
            st.metric("Trainable Parameters", f"{lora_params / 1e6:.2f} Million")
        with m2:
            st.metric("Trainable Ratio", f"{trainable_pct:.3f}%")
            st.metric("Scaling Factor (α/r)", f"{alpha / rank:.2f}")

        st.info("💡 By tuning only ~0.3% of parameters, GPU memory and checkpoint sizes are reduced by >90%.")

    st.markdown("---")
    st.markdown("#### Supervised Fine-Tuning (SFT) Execution")
    if st.button("🚀 Run LoRA SFT Training Simulation (3 Epochs)", key="btn_run_sft"):
        with st.spinner("Executing forward/backward optimization steps with PyTorch AdamW..."):
            loader = CyberReasonModelLoader(device="cpu")
            model, _ = loader.create_synthetic_model_and_tokenizer(vocab_size=300, hidden_size=32)
            trainer = CyberReasonLoRATrainer(
                model=model,
                r=rank,
                lora_alpha=alpha,
                target_modules=target_mods
            )
            history = trainer.train_synthetic_epochs(num_steps=5)

            df_hist = pd.DataFrame(history)
            st.success("✅ LoRA SFT Epochs Completed Successfully!")
            st.line_chart(df_hist.set_index("step")["loss"])
            st.dataframe(df_hist, use_container_width=True)
