import streamlit as st
import pandas as pd

def render(results):
    st.header("🚨 Detected Anomalies")

    anomalies = results[results["is_anomaly"]]

    if anomalies.empty:
        st.success("No anomalies detected")
        return

    st.metric("Total Anomalies", len(anomalies))

    for ts, row in anomalies.iterrows():
        with st.expander(f"⏱ {ts} | Score: {row['final_score']:.2f}"):
            for reason in row["explanations"]:
                st.markdown(f"- {reason}")
