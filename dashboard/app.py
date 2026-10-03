"""Entrypoint / router. st.navigation keeps the sidebar page titles readable (Streamlit would
otherwise label the home page after this file's name, "app"). Run with
`streamlit run dashboard/app.py` (or `make dashboard`)."""
import streamlit as st

st.set_page_config(page_title="TabPFN-3.5 · Railway Lab", page_icon="🚆", layout="wide")

pages = [
    st.Page("pages/0_Overview.py", title="Overview", icon="🚆", default=True),
    st.Page("pages/1_Berlin_Explore.py", title="Berlin — Explore", icon="📊", url_path="berlin-explore"),
    st.Page("pages/2_Finnish_Explore.py", title="Finnish — Explore", icon="🇫🇮", url_path="finnish-explore"),
    st.Page("pages/3_Live_Prediction.py", title="Live Prediction (Berlin)", icon="🔮", url_path="predict"),
    st.Page("pages/4_Benchmark_Berlin.py", title="Benchmark: Berlin baseline", icon="⚖️", url_path="bench-berlin"),
    st.Page("pages/5_Benchmark_Deutsche_Bahn.py", title="Benchmark: Deutsche Bahn vs. XGBoost", icon="🚉", url_path="bench-db"),
]
st.navigation(pages).run()
