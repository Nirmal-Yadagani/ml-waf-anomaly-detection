import streamlit as st
import pandas as pd

def render(ml_features, baseline):
    st.header("📊 Baseline vs Current Traffic")

    feature = st.selectbox(
        "Select feature",
        list(baseline.keys())
    )

    st.line_chart(ml_features[feature])

    b = baseline[feature]
    st.markdown(
        f"""
        **Baseline stats**
        - Mean: `{b['mean']:.2f}`
        - p95: `{b['p95']:.2f}`
        - p99: `{b['p99']:.2f}`
        """
    )
