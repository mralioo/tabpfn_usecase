# Notebooks

[marimo](https://marimo.io/) notebooks — plain `.py` files, git-diffable, reactive. Open them with
`make notebooks`, or one at a time:

```bash
.venv/bin/marimo edit notebooks/02_anomaly_early_warning.py
```

| Notebook | Question | Notes |
| --- | --- | --- |
| `00_data_quality_and_eda.py` | What's in the Alstom files, and is it clean? | Shapes, gaps, duplicate station names, timestamp alignment, closure overlaps, distributions. No models. |
| `01_demand_forecasting_and_overcrowding.py` | Why does "flow ≥ the station's own p90" fail as a target? | The motivation. Lookup baseline vs XGBoost vs TabPFN-3.5 tie (ROC-AUC 0.855 / 0.846 / 0.853), because that target is mostly the daily clock. Makes live TabPFN-3.5 calls. |
| `02_anomaly_early_warning.py` | When will a station run off its normal pattern, why, and what should the operator do? | **The use case.** Oscillation → robust normalisation (z) → drivers (events, closures, weather) → model comparison on 2 folds (PR curves, recall by driver, XGBoost learning curve) → scenario replay → alarm feed → network outlook and line corridor → live TabPFN-3.5 what-if → the control-room agent. Reads `results/anomaly/` (`make anomaly`). The live cells sit behind buttons, so opening the notebook costs no API quota. |
