"""Benchmark: TabPFN-3.5 (fit on a small in-context sample) vs. XGBoost (fit on the full
chronological training split) on real Deutsche Bahn railway delay data (`datasets/deutsche_bahn.py`).

Same question `tabpfn_lab/evaluate.py` asks on the Berlin data, but on a dataset big and hard
enough for the answer to mean something: does TabPFN-3.5 reach a standard tuned
gradient-boosted model's accuracy using a couple-percent sample of the data and one `.fit()`
call? Run:

    python scripts/train_and_eval_db.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import TABPFN_MODEL_PATH, authenticate
from .datasets.deutsche_bahn import (
    CATEGORICAL,
    FEATURES,
    chronological_split,
    encode_categoricals,
    load_classification_table,
    load_regression_table,
)
from .xgboost_baseline import fit_classifier as fit_xgb_classifier
from .xgboost_baseline import fit_regressor as fit_xgb_regressor

TABPFN_SAMPLE_SIZE = 10_000  # TabPFN-3.5's in-context sweet spot; XGBoost gets the full train split
TEST_SAMPLE_SIZE = 5_000


def _stratified_sample(df: pd.DataFrame, n: int, label_col: str, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    frac_positive = df[label_col].mean()
    n_pos = min(int(round(n * max(frac_positive, 0.1))), int((df[label_col] == 1).sum()))
    n_neg = min(n - n_pos, int((df[label_col] == 0).sum()))
    pos = df[df[label_col] == 1].sample(n_pos, random_state=seed)
    neg = df[df[label_col] == 0].sample(n_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)


def run_classification_benchmark(seed: int = 0) -> dict:
    df = load_classification_table()
    train, test, cutoff = chronological_split(df)
    test = test.sample(min(TEST_SAMPLE_SIZE, len(test)), random_state=seed).reset_index(drop=True)
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]

    authenticate()
    from tabpfn_client import TabPFNClassifier

    tabpfn_sample = _stratified_sample(train, TABPFN_SAMPLE_SIZE, "delayed", seed)
    X_tabpfn_train = encode_categoricals(tabpfn_sample[FEATURES], CATEGORICAL)
    X_test_tabpfn = encode_categoricals(test[FEATURES], CATEGORICAL, reference=tabpfn_sample[FEATURES])

    t0 = time.perf_counter()
    tabpfn_clf = TabPFNClassifier(model_path=TABPFN_MODEL_PATH, categorical_features_indices=cat_idx)
    tabpfn_clf.fit(X_tabpfn_train, tabpfn_sample["delayed"])
    tabpfn_fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    tabpfn_proba = tabpfn_clf.predict_proba(X_test_tabpfn)[:, 1]
    tabpfn_predict_s = time.perf_counter() - t0

    X_xgb_train = encode_categoricals(train[FEATURES], CATEGORICAL)
    X_test_xgb = encode_categoricals(test[FEATURES], CATEGORICAL, reference=train[FEATURES])
    xgb_clf, xgb_fit_s = fit_xgb_classifier(X_xgb_train, train["delayed"])
    t0 = time.perf_counter()
    xgb_proba = xgb_clf.predict_proba(X_test_xgb)[:, 1]
    xgb_predict_s = time.perf_counter() - t0

    y_test = test["delayed"].to_numpy()
    return {
        "dataset": "Deutsche Bahn delays, 2025-06 (huggingface.co/datasets/piebro/deutsche-bahn-data)",
        "task": "classification: delayed (>=5 min) vs. on time",
        "cutoff_date": str(cutoff), "n_rows_total": int(len(df)),
        "n_train_tabpfn": int(len(tabpfn_sample)), "n_train_xgboost": int(len(train)), "n_test": int(len(test)),
        "tabpfn": {"roc_auc": round(float(roc_auc_score(y_test, tabpfn_proba)), 3),
                   "pr_auc": round(float(average_precision_score(y_test, tabpfn_proba)), 3),
                   "fit_seconds": round(tabpfn_fit_s, 2), "predict_seconds": round(tabpfn_predict_s, 2)},
        "xgboost": {"roc_auc": round(float(roc_auc_score(y_test, xgb_proba)), 3),
                    "pr_auc": round(float(average_precision_score(y_test, xgb_proba)), 3),
                    "fit_seconds": round(xgb_fit_s, 2), "predict_seconds": round(xgb_predict_s, 2)},
        "data_efficiency": f"TabPFN used {len(tabpfn_sample):,}/{len(train):,} rows "
                            f"({100 * len(tabpfn_sample) / len(train):.1f}% of XGBoost's training data)",
    }


def run_regression_benchmark(seed: int = 1) -> dict:
    df = load_regression_table()
    train, test, cutoff = chronological_split(df)
    test = test.sample(min(TEST_SAMPLE_SIZE, len(test)), random_state=seed).reset_index(drop=True)
    cat_idx = [FEATURES.index(c) for c in CATEGORICAL]

    authenticate()
    from tabpfn_client import TabPFNRegressor

    tabpfn_sample = train.sample(min(TABPFN_SAMPLE_SIZE, len(train)), random_state=seed)
    X_tabpfn_train = encode_categoricals(tabpfn_sample[FEATURES], CATEGORICAL)
    X_test_tabpfn = encode_categoricals(test[FEATURES], CATEGORICAL, reference=tabpfn_sample[FEATURES])

    t0 = time.perf_counter()
    tabpfn_reg = TabPFNRegressor(model_path=TABPFN_MODEL_PATH, categorical_features_indices=cat_idx)
    tabpfn_reg.fit(X_tabpfn_train, tabpfn_sample["delay_in_min"])
    tabpfn_fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    tabpfn_pred = tabpfn_reg.predict(X_test_tabpfn)
    tabpfn_predict_s = time.perf_counter() - t0

    X_xgb_train = encode_categoricals(train[FEATURES], CATEGORICAL)
    X_test_xgb = encode_categoricals(test[FEATURES], CATEGORICAL, reference=train[FEATURES])
    xgb_reg, xgb_fit_s = fit_xgb_regressor(X_xgb_train, train["delay_in_min"])
    t0 = time.perf_counter()
    xgb_pred = xgb_reg.predict(X_test_xgb)
    xgb_predict_s = time.perf_counter() - t0

    y_test = test["delay_in_min"].to_numpy(float)
    mae_tabpfn = float(np.mean(np.abs(y_test - tabpfn_pred)))
    mae_xgb = float(np.mean(np.abs(y_test - xgb_pred)))
    return {
        "dataset": "Deutsche Bahn delays, 2025-06 (huggingface.co/datasets/piebro/deutsche-bahn-data)",
        "task": "regression: delay_in_min (clipped to [-15, 120])",
        "cutoff_date": str(cutoff), "n_rows_total": int(len(df)),
        "n_train_tabpfn": int(len(tabpfn_sample)), "n_train_xgboost": int(len(train)), "n_test": int(len(test)),
        "tabpfn": {"mae": round(mae_tabpfn, 2),
                   "rmse": round(float(np.sqrt(np.mean((y_test - tabpfn_pred) ** 2))), 2),
                   "fit_seconds": round(tabpfn_fit_s, 2), "predict_seconds": round(tabpfn_predict_s, 2)},
        "xgboost": {"mae": round(mae_xgb, 2),
                    "rmse": round(float(np.sqrt(np.mean((y_test - xgb_pred) ** 2))), 2),
                    "fit_seconds": round(xgb_fit_s, 2), "predict_seconds": round(xgb_predict_s, 2)},
        "data_efficiency": f"TabPFN used {len(tabpfn_sample):,}/{len(train):,} rows "
                            f"({100 * len(tabpfn_sample) / len(train):.2f}% of XGBoost's training data)",
    }


def save_metrics(metrics: dict, path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=str))
