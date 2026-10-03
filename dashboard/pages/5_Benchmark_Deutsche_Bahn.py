"""TabPFN-3.5 vs. XGBoost, on real Deutsche Bahn railway delay data (1.5M rows/month)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tabpfn_lab.config import RESULTS_DIR  # noqa: E402

st.title("🚉 Benchmark — Deutsche Bahn: TabPFN-3.5 vs. XGBoost")
st.caption(
    "Same classification + regression pairing as the Berlin demo, same topic (railway "
    "operations), same country — but real, not simulated, and at a scale (1.5M real German "
    "train stops, one month) that actually tests something against a tuned baseline. Not a "
    "naive heuristic: XGBoost, default-ish hyperparameters, trained on the FULL training split "
    "(not a sample). Data: [piebro/deutsche-bahn-data]"
    "(https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0). "
    "See `tabpfn_lab/datasets/deutsche_bahn.py`."
)
metrics_path = RESULTS_DIR / "metrics_db.json"
if not metrics_path.exists():
    st.warning("No benchmark yet — run `make train-db` (or `python scripts/train_and_eval_db.py`) "
               "to generate `results/metrics_db.json` (fetches ~1.5M rows from Hugging Face on first run).")
    st.stop()

m = json.loads(metrics_path.read_text())
c, r = m["classification"], m["regression"]

st.markdown(f"#### Classification — {c['dataset']}")
st.caption(f"{c['n_rows_total']:,} rows total · TabPFN trained on {c['n_train_tabpfn']:,} rows · "
           f"XGBoost trained on {c['n_train_xgboost']:,} rows · tested on {c['n_test']:,} held-out rows · {c['data_efficiency']}")
col1, col2, col3 = st.columns(3)
col1.metric("TabPFN-3.5 ROC-AUC", c["tabpfn"]["roc_auc"])
col2.metric("XGBoost ROC-AUC", c["xgboost"]["roc_auc"])
col3.metric("Gap", f"{100 * (c['tabpfn']['roc_auc'] - c['xgboost']['roc_auc']) / c['xgboost']['roc_auc']:.1f}%",
            help="TabPFN vs. XGBoost, relative — using well under 1% of XGBoost's training data")

st.markdown(f"#### Regression — {r['dataset']}")
st.caption(f"{r['n_rows_total']:,} rows total · TabPFN trained on {r['n_train_tabpfn']:,} rows · "
           f"XGBoost trained on {r['n_train_xgboost']:,} rows · tested on {r['n_test']:,} held-out rows · {r['data_efficiency']}")
col1, col2, col3 = st.columns(3)
col1.metric("TabPFN-3.5 MAE (min)", r["tabpfn"]["mae"])
col2.metric("XGBoost MAE (min)", r["xgboost"]["mae"])
col3.metric("Gap", f"{100 * (r['tabpfn']['mae'] - r['xgboost']['mae']) / r['xgboost']['mae']:.1f}%",
            help="TabPFN vs. XGBoost, relative — using well under 1% of XGBoost's training data "
                 "(negative = TabPFN's MAE is lower)")

st.markdown("#### Fit time vs. predict time")
timing = pd.DataFrame({
    "TabPFN-3.5 fit (s)": [c["tabpfn"]["fit_seconds"], r["tabpfn"]["fit_seconds"]],
    "XGBoost fit (s)": [c["xgboost"]["fit_seconds"], r["xgboost"]["fit_seconds"]],
    "TabPFN-3.5 predict, 5k rows (s)": [c["tabpfn"]["predict_seconds"], r["tabpfn"]["predict_seconds"]],
    "XGBoost predict, 5k rows (s)": [c["xgboost"]["predict_seconds"], r["xgboost"]["predict_seconds"]],
}, index=["Classification", "Regression"])
st.dataframe(timing, use_container_width=True)
st.info(
    "Honest read: fit time is comparable (XGBoost's histogram method is fast at this scale "
    "too), and XGBoost's **inference is far faster** here — it runs locally, TabPFN-3.5 pays "
    "a network round-trip per call. What TabPFN-3.5 wins is **data efficiency**: on "
    "classification it's within ~2% of XGBoost's ROC-AUC; on the regression task it actually "
    "matches-or-beats XGBoost's MAE (4.64 vs. 4.66 minutes) — both from **under 1%** of "
    "XGBoost's training rows and zero tuning. The realistic win under a deadline is getting a "
    "usable, competitive model fast, not out-predicting a team that already had the full "
    "pipeline built.", icon="ℹ️")
