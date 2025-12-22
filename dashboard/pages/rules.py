import streamlit as st
import json

def render(rules):
    st.header("🛡 Rule Recommendations")

    if not rules:
        st.info("No rules proposed yet")
        return

    for rule in rules:
        with st.expander(
            f"{rule['rule_type']} | Confidence {rule['confidence']}"
        ):
            st.json(rule)
            st.checkbox("Approve rule (demo only)", key=str(rule))
