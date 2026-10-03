"""TabPFN-3.5 vs. the historical-average baseline, on Berlin U-Bahn data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tabpfn_lab.config import RESULTS_DIR  # noqa: E402

st.title("⚖️ Benchmark — Berlin U-Bahn: TabPFN-3.5 vs. baseline")
metrics_path = RESULTS_DIR / "metrics.json"
if not metrics_path.exists():
    st.warning("No benchmark yet — run `make train` (or `python scripts/train_and_eval.py`) "
               "to generate `results/metrics.json`.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
c = metrics["classification_overcrowding_risk"]
r = metrics["regression_expected_flow"]
lat = metrics["latency_ms"]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Overcrowding-risk classification (ROC-AUC, higher is better)**")
    st.bar_chart(pd.DataFrame({
        "ROC-AUC": [c["tabpfn"]["roc_auc"], c["baseline_historical_rate"]["roc_auc"]],
    }, index=["TabPFN-3.5", "Historical-rate baseline"]))
with col2:
    st.markdown("**Expected-flow regression (MAE, lower is better)**")
    st.bar_chart(pd.DataFrame({
        "MAE (passengers)": [r["tabpfn"]["mae"], r["baseline_historical_mean"]["mae"]],
    }, index=["TabPFN-3.5", "Historical-mean baseline"]))
    st.caption(f"{r['mae_improvement_pct']}% lower MAE than the baseline" if r["mae_improvement_pct"] else "")

st.markdown("**Latency**")
st.json(lat)
st.caption(f"Benchmarked on {metrics['n_test_rows']:,} chronologically held-out rows "
           f"(test split starts {metrics['cutoff_date']}), {metrics['n_train_sample']:,}-row train sample.")
st.info(
    "Honest caveat: on this dataset TabPFN-3.5 roughly **ties** this baseline rather than "
    "beating it — the simulated flow is strongly periodic, so a one-line station × hour × "
    "weekday lookup is already close to optimal. See the Deutsche Bahn benchmark page for a "
    "harder, bigger, real-world benchmark against an actual tuned model.", icon="ℹ️")
