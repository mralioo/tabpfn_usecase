#!/usr/bin/env python
"""Fit TabPFN-3.5 on the Berlin U-Bahn feature table, benchmark it against the
historical-average baseline, and write `results/metrics.json`.

    python scripts/train_and_eval.py [--force]

--force re-fits even if a cached model exists in results/model_cache/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabpfn_lab.config import BERLIN_DATA_DIR
from tabpfn_lab.datasets.berlin import build_feature_table, chronological_split, discover_dataset_dir
from tabpfn_lab.evaluate import run_benchmark, save_metrics
from tabpfn_lab.models import load_or_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-fit even if a cached model exists")
    args = parser.parse_args()

    folder = discover_dataset_dir(BERLIN_DATA_DIR)
    print(f"Building feature table from: {folder}")
    table = build_feature_table(folder)
    print(f"Feature table: {table.shape[0]:,} rows, overcrowded rate = {table['overcrowded'].mean():.1%}")

    train_pool, test_pool, cutoff = chronological_split(table)
    print(f"Chronological split: train < {cutoff} ({len(train_pool):,} rows), "
          f"test >= {cutoff} ({len(test_pool):,} rows)")

    print("Fitting TabPFN-3.5 classifier + regressor (or loading from cache)...")
    models = load_or_fit(train_pool, cutoff, force=args.force)
    print(f"  status={models.status}  fit_time={models.fit_seconds:.2f}s  "
          f"train_sample={len(models.train_sample):,} rows")

    print("Running benchmark (TabPFN-3.5 vs. historical-average baseline)...")
    metrics = run_benchmark(table, train_pool, test_pool, models)
    path = save_metrics(metrics)
    print(f"\nWrote {path}")

    c = metrics["classification_overcrowding_risk"]
    r = metrics["regression_expected_flow"]
    print(f"\nOvercrowding risk (ROC-AUC): TabPFN {c['tabpfn']['roc_auc']}  vs  baseline {c['baseline_historical_rate']['roc_auc']}")
    print(f"Expected flow (MAE):         TabPFN {r['tabpfn']['mae']}  vs  baseline {r['baseline_historical_mean']['mae']}"
          f"  ({r['mae_improvement_pct']}% improvement)")


if __name__ == "__main__":
    main()
