# Exploration notebooks

[marimo](https://marimo.io/) notebooks — plain `.py` files, git-diffable, reactive (change a
cell, every dependent cell recomputes). No Jupyter, no `.ipynb`.

Run `make notebooks` (opens a browser to pick one) or edit a single file directly:

```bash
.venv/bin/marimo edit notebooks/01_demand_forecasting_and_overcrowding.py
```

Each notebook loads the real data, does its own cleaning/preprocessing, and fits real models —
nothing here is a canned/pre-computed result. `01`–`03` make real `tabpfn_client` API calls
(needs `TABPFN_API_TOKEN` in `.env`; each falls back gracefully or tells you plainly if it's
unavailable) and fit a real local XGBoost model on first run (cached to
`results/model_cache/xgboost_berlin.pkl` after that — same cache the Closure Impact Lab
webapp uses, so the two are never comparing different models by accident).

| Notebook | Question | New vs. existing code |
| --- | --- | --- |
| `00_data_quality_and_eda.py` | What's actually in these files, and is it clean? | Pure EDA over the existing loaders — no modeling |
| `01_demand_forecasting_and_overcrowding.py` | Will this station be overcrowded? How many passengers are expected? | Benchmarks the exact use case `mcp_server/server.py` already serves — ROC/calibration/residuals the production tools don't show |
| `02_closure_impact_and_network_resilience.py` | What happens if this station/line closes? | Same engine as `webapp/` (`tabpfn_lab/closure_impact.py`), run systematically across every real recorded closure instead of one at a time, plus pure graph-theory resilience (articulation points, betweenness) |
| `03_energy_consumption_forecasting.py` | How much energy will this line draw? | **New** — `energy_consumption_*.csv` wasn't joined to anything else in this repo before this notebook |
| `04_anomaly_early_warning.py` | When will a station run off its normal pattern (surge or collapse), and why? | **The refined use case.** Removes the daily/weekly oscillation (robust station × day type × hour profile, z-score), attributes anomalies to events/closures/weather, and benchmarks profile baseline vs XGBoost vs TabPFN-3.5 on two folds plus an XGBoost learning curve. Scenario replay and alarm feed match the webapp's Early-Warning Desk. Reads `results/anomaly/` (`make anomaly`) |

First run of `01`–`03` is slow (feature-table build + model fits, no caching yet); after that,
XGBoost is instant and only the live TabPFN-3.5 calls take real network time (seconds, not
milliseconds — that latency is itself part of what these notebooks are measuring).
