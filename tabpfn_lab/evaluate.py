"""The benchmark behind the hackathon pitch: TabPFN-3.5 vs. a historical-average
baseline, on a chronologically held-out test sample (no date overlap with
training) — accuracy AND latency side by side, because "fast but mediocre" and
"accurate but slow to build" are both the wrong answer for a control-room tool.

Run: `python scripts/train_and_eval.py` — writes `results/metrics.json`.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .baselines import (
    fit_classification_baseline,
    fit_regression_baseline,
    predict_classification_baseline,
    predict_regression_baseline,
)
from .config import RESULTS_DIR
from .datasets.berlin import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, encode_categoricals, stratified_subsample
from .models import FittedModels

TEST_SAMPLE_SIZE = 2_000


def run_benchmark(table: pd.DataFrame, train_pool: pd.DataFrame, test_pool: pd.DataFrame,
                   models: FittedModels, seed: int = 1) -> dict:
    test_df = stratified_subsample(test_pool, TEST_SAMPLE_SIZE, "overcrowded", seed=seed)
    X_test = encode_categoricals(test_df[FEATURE_COLUMNS], reference=models.train_sample[FEATURE_COLUMNS])
    y_class = test_df["overcrowded"].to_numpy()
    y_reg = test_df["passengers"].to_numpy(float)

    # --- TabPFN: accuracy + end-to-end batch latency ---
    t0 = time.perf_counter()
    proba = models.classifier.predict_proba(X_test)[:, 1]
    clf_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    expected = models.regressor.predict(X_test)
    reg_seconds = time.perf_counter() - t0

    # --- baselines: fit (groupby, near-zero cost) + predict ---
    t0 = time.perf_counter()
    rate_table = fit_classification_baseline(train_pool)
    mean_table = fit_regression_baseline(train_pool)
    baseline_fit_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    base_proba = predict_classification_baseline(rate_table, test_df, float(train_pool["overcrowded"].mean())).to_numpy()
    base_expected = predict_regression_baseline(mean_table, test_df, float(train_pool["passengers"].mean())).to_numpy()
    baseline_predict_seconds = time.perf_counter() - t0

    mae_tabpfn = float(np.mean(np.abs(y_reg - expected)))
    mae_baseline = float(np.mean(np.abs(y_reg - base_expected)))

    return {
        "n_test_rows": int(len(test_df)),
        "n_train_sample": int(len(models.train_sample)),
        "cutoff_date": str(models.cutoff_date),
        "classification_overcrowding_risk": {
            "tabpfn": {"roc_auc": round(float(roc_auc_score(y_class, proba)), 3),
                       "pr_auc": round(float(average_precision_score(y_class, proba)), 3)},
            "baseline_historical_rate": {"roc_auc": round(float(roc_auc_score(y_class, base_proba)), 3),
                                          "pr_auc": round(float(average_precision_score(y_class, base_proba)), 3)},
        },
        "regression_expected_flow": {
            "tabpfn": {"mae": round(mae_tabpfn, 2)},
            "baseline_historical_mean": {"mae": round(mae_baseline, 2)},
            "mae_improvement_pct": round(100 * (1 - mae_tabpfn / mae_baseline), 1) if mae_baseline else None,
        },
        "latency_ms": {
            "tabpfn_fit_once_for_both_tasks": round(models.fit_seconds * 1000, 1),
            "tabpfn_classify_batch": round(clf_seconds * 1000, 1),
            "tabpfn_classify_per_row": round(clf_seconds * 1000 / len(test_df), 2),
            "tabpfn_regress_batch": round(reg_seconds * 1000, 1),
            "tabpfn_regress_per_row": round(reg_seconds * 1000 / len(test_df), 2),
            "baseline_fit_both_tasks": round(baseline_fit_seconds * 1000, 1),
            "baseline_predict_batch_both_tasks": round(baseline_predict_seconds * 1000, 1),
            "note": ("Baseline fit/predict is near-instant groupby arithmetic — it will always "
                     "win a raw speed race. The comparison that matters for a critical use case "
                     "is TabPFN's one-shot 'fit on today's feature table, no training pipeline, "
                     "no per-task model to maintain' turnaround vs. the engineering cost of "
                     "building and keeping a tuned classical model like that baseline (or "
                     "XGBoost/CatBoost) current as the schedule, weather, and events change."),
        },
    }


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    import json

    path = path or (RESULTS_DIR / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str))
    return path
