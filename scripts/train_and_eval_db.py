#!/usr/bin/env python
"""Benchmark TabPFN-3.5 vs. XGBoost on real Deutsche Bahn railway delay data (~2M rows/month,
see tabpfn_lab/db_data.py). Writes results/metrics_db.json.

    python scripts/train_and_eval_db.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabpfn_lab.config import RESULTS_DIR
from tabpfn_lab.evaluate_db import run_classification_benchmark, run_regression_benchmark, save_metrics


def main() -> None:
    print("=== Classification: Deutsche Bahn delays (>=5 min) — TabPFN-3.5 vs. XGBoost ===")
    clf_metrics = run_classification_benchmark()
    print(f"  TabPFN-3.5 ({clf_metrics['n_train_tabpfn']:,} rows, {clf_metrics['tabpfn']['fit_seconds']}s fit): "
          f"ROC-AUC {clf_metrics['tabpfn']['roc_auc']}, PR-AUC {clf_metrics['tabpfn']['pr_auc']}")
    print(f"  XGBoost    ({clf_metrics['n_train_xgboost']:,} rows, {clf_metrics['xgboost']['fit_seconds']}s fit): "
          f"ROC-AUC {clf_metrics['xgboost']['roc_auc']}, PR-AUC {clf_metrics['xgboost']['pr_auc']}")
    print(f"  {clf_metrics['data_efficiency']}")

    print("\n=== Regression: Deutsche Bahn delay minutes — TabPFN-3.5 vs. XGBoost ===")
    reg_metrics = run_regression_benchmark()
    print(f"  TabPFN-3.5 ({reg_metrics['n_train_tabpfn']:,} rows, {reg_metrics['tabpfn']['fit_seconds']}s fit): "
          f"MAE {reg_metrics['tabpfn']['mae']}, RMSE {reg_metrics['tabpfn']['rmse']}")
    print(f"  XGBoost    ({reg_metrics['n_train_xgboost']:,} rows, {reg_metrics['xgboost']['fit_seconds']}s fit): "
          f"MAE {reg_metrics['xgboost']['mae']}, RMSE {reg_metrics['xgboost']['rmse']}")
    print(f"  {reg_metrics['data_efficiency']}")

    path = RESULTS_DIR / "metrics_db.json"
    save_metrics({"classification": clf_metrics, "regression": reg_metrics}, path)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
