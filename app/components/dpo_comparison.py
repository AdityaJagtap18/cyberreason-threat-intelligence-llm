"""
PRJ-07: CyberReason - Direct Preference Optimization (DPO) Component
Visualizes the alignment contrast between chosen (air-gapped) and rejected (cloud leak)
threat intelligence responses with implicit reward margins.
"""

import streamlit as st
import pandas as pd
from src.core.dpo_aligner import CyberReasonDPOAligner
from src.core.model_loader import CyberReasonModelLoader


def render_dpo_comparison():
    st.subheader("⚖️ Direct Preference Optimization (DPO) Alignment")
    st.markdown(
        "Alters the open-weight LLM policy $\\pi_\\theta$ using preference pairs $(x, y_w, y_l)$ "
        "without a separate reward model. Enforces **strict air-gapped zero-data-leakage behaviors**."
    )

    st.latex(
        r"\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = "
        r"-\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( "
        r"\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - "
        r"\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]"
    )

    col1, col2 = st.columns(2)
    with col1:
        beta = st.slider("DPO Temperature Parameter (β)", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
    with col2:
        lr = st.selectbox("DPO Learning Rate", [1e-6, 5e-6, 1e-5], index=1)

    st.markdown("#### Sample Preference Alignment Contrast")
    sample_options = [
        "JNDI LDAP Egress Attempt (Log4Shell)",
        "Obfuscated PowerShell Execution (Spearphishing Macro)",
        "LSASS Memory Access (Mimikatz / Credential Dumping)"
    ]
    selected_sample = st.selectbox("Select Threat Scenario", sample_options)

    if "JNDI" in selected_sample:
        c1, c2 = st.columns(2)
        with c1:
            st.success("🏆 **Chosen Response (Air-Gapped Rigor)**")
            st.markdown(
                "- **CVE ID**: CVE-2021-44228 (Log4Shell)\n"
                "- **Technique**: T1190 (Exploit Public-Facing Application)\n"
                "- **Containment**: Sever egress firewall routing; set `LOG4J_FORMAT_MSG_NO_LOOKUPS=true`; "
                "extract RAM capture offline. **Zero external SaaS telemetry leaks.**"
            )
        with c2:
            st.error("🚫 **Rejected Response (Data Leak Risk)**")
            st.markdown(
                "- Suggests uploading raw HTTP logs and payloads to public cloud scanners.\n"
                "- Misses MITRE technique attribution.\n"
                "- Leaves egress firewall open."
            )

    st.markdown("---")
    st.markdown("#### Preference Optimization Step Simulation")
    if st.button("🔄 Run DPO Alignment Optimization Step", key="btn_run_dpo"):
        with st.spinner("Calculating log probabilities and implicit reward margins..."):
            loader = CyberReasonModelLoader(device="cpu")
            model, _ = loader.create_synthetic_model_and_tokenizer(vocab_size=300, hidden_size=32)
            aligner = CyberReasonDPOAligner(policy_model=model, beta=beta, learning_rate=lr)
            history = aligner.simulate_preference_alignment(steps=3)

            df_dpo = pd.DataFrame(history)
            st.success("✅ DPO Preference Alignment Step Completed!")
            st.dataframe(df_dpo, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("DPO Loss", f"{history[-1]['dpo_loss']:.4f}")
            with m2:
                st.metric("Chosen Reward", f"{history[-1]['chosen_reward']:.4f}")
            with m3:
                st.metric("Reward Margin (Δr)", f"{history[-1]['reward_margin']:.4f}")
