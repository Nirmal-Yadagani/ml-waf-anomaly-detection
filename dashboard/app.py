# dashboard/app.py

import streamlit as st
import pickle

st.set_page_config(
    page_title="ML-WAF Anomaly Detection",
    layout="wide",
)

@st.cache_data
def load_data():
    with open("dashboard/data/context.pkl", "rb") as f:
        context = pickle.load(f)
    with open("dashboard/data/ml_features.pkl", "rb") as f:
        ml_features = pickle.load(f)
    with open("dashboard/data/results.pkl", "rb") as f:
        results = pickle.load(f)
    with open("dashboard/data/rules.pkl", "rb") as f:
        rules = pickle.load(f)
    with open("dashboard/data/baseline.pkl", "rb") as f:
        baseline = pickle.load(f)
    return context, ml_features, results, rules, baseline


context, ml_features, results, rules, baseline = load_data()

st.title("🔐 ML-Enabled WAF Anomaly Detection")

st.sidebar.success("Pipeline running on Nginx traffic")

st.write(
    """
    This dashboard shows **behavioral anomaly detection**, 
    **explainable alerts**, and **automated security rule recommendations**
    generated from live web traffic.
    """
)

from pages.anomalies import render as anomalies_page
from pages.baselines import render as baseline_page
from pages.rules import render as rules_page

page = st.sidebar.radio(
    "Navigation",
    ["Anomalies", "Baselines", "Rules"]
)

if page == "Anomalies":
    anomalies_page(results)

elif page == "Baselines":
    baseline_page(ml_features, baseline)

elif page == "Rules":
    rules_page(rules)

