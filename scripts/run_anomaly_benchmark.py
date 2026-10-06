#!/usr/bin/env python
"""Anomaly early-warning benchmark: normalise the Berlin flows, score every model on both
folds (Alstom hold-out + Sep 1–21 backtest), write `results/anomaly/`.

    python scripts/run_anomaly_benchmark.py [--offline]

--offline skips the TabPFN-3.5 API calls (XGBoost + baselines only, ~30 s).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabpfn_lab import anomaly  # noqa: E402
from tabpfn_lab.config import load_dotenv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="skip TabPFN-3.5 (no API calls)")
    args = parser.parse_args()
    load_dotenv()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    metrics = anomaly.run_and_cache(live_tabpfn=not args.offline)
    for fold, m in [*metrics["folds"].items(), ("pooled", metrics["pooled"])]:
        print(f"\n{fold}: {m['n_test_rows']:,} test station-hours, surge rate {m['test_surge_rate']:.1%}")
        print(f"  {'model':<26} {'PR-AUC':>7} {'event':>7} {'closure':>8} {'alarm P':>8}")
        for name, s in m["models"].items():
            closure = s.get("closure_drop_recall")
            print(f"  {name:<26} {s['pr_auc']:>7.3f} {s['recall_event']:>7.2f} "
                  f"{'—' if closure is None else f'{closure:.2f}':>8} {s['alarm_precision']:>8.3f}")


if __name__ == "__main__":
    main()
