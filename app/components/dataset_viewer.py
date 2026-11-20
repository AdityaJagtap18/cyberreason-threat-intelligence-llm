"""
PRJ-07: CyberReason - Streamlit Dataset Viewer Component
Renders MITRE ATT&CK Enterprise Matrix v14 TTPs and NVD CVE instruction pairs.
"""

import streamlit as st
import pandas as pd
from src.core.dataset_loader import CyberReasonDatasetLoader


def render_dataset_viewer(dataset_loader: CyberReasonDatasetLoader):
    st.subheader("🛡️ Curated Threat Intelligence & Vulnerability Datasets")
    st.markdown(
        "Explore curated datasets for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO). "
        "Strictly bounded to air-gapped threat attribution with zero telemetry egress."
    )

    stats = dataset_loader.get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MITRE Techniques", stats["total_mitre_techniques"])
    with col2:
        st.metric("Tactics Covered", stats["unique_tactics_covered"])
    with col3:
        st.metric("SFT Instructions", stats["total_sft_instruction_pairs"])
    with col4:
        st.metric("DPO Pairs", stats["total_dpo_preference_pairs"])

    tab1, tab2, tab3 = st.tabs(["MITRE ATT&CK Matrix", "NVD CVE Instruction Pairs", "DPO Preference Pairs"])

    with tab1:
        mitre_data = dataset_loader.load_mitre_ttps()
        df_mitre = pd.DataFrame(mitre_data)
        st.dataframe(
            df_mitre[["technique_id", "name", "tactic", "description", "containment"]],
            use_container_width=True
        )

    with tab2:
        cve_data = dataset_loader.load_cve_instructions()
        df_cve = pd.DataFrame(cve_data)
        st.dataframe(df_cve[["instruction", "input", "output"]], use_container_width=True)

    with tab3:
        dpo_data = dataset_loader.load_dpo_pairs()
        for idx, item in enumerate(dpo_data):
            with st.expander(f"Preference Pair #{idx+1}: {item['prompt'][:60]}..."):
                st.markdown(f"**Prompt:** {item['prompt']}")
                c1, c2 = st.columns(2)
                with c1:
                    st.success("✅ **Chosen (Air-Gapped & High-Rigor)**")
                    st.markdown(item["chosen"])
                with c2:
                    st.error("❌ **Rejected (Insecure Cloud Leak)**")
                    st.markdown(item["rejected"])
