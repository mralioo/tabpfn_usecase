# Project module structure

Two parts: what exists now (this section is accurate as of the Berlin/Finnish exploration work),
and what's planned next — a scientific, same-methodology comparison of TabPFN-3.5 against other
models across all three datasets, then a showcase built on top of it. Not built yet; this is the
plan to review before it is.

## Current structure

```
data/
  SOURCE.md                          overview + provenance of all three datasets
  Berlin_Ubahn_Alstom_data/          committed (small CSVs) — training dataset/, testing dataset/
  DB_data/                           gitignored — mirrored here after the first Hugging Face fetch
  Finnish_Railway_Operations_data/   gitignored — 96 monthly parquet files, placed locally (2.75 GB)

tabpfn_lab/                          the core package — no Streamlit, no agent framework, no MCP
  config.py                          paths (BERLIN_DATA_DIR, DB_DATA_DIR, FINNISH_DATA_DIR, RESULTS_DIR), env/token loading
  datasets/                          one loader module per dataset, shared naming convention (not yet a strict
                                      shared interface — see "Planned" below):
    berlin.py                         raw CSV loading + build_feature_table() + chronological_split()
    deutsche_bahn.py                  HF fetch + load_classification_table()/load_regression_table() + chronological_split()
    finnish.py                        local parquet loading + the same two load_*_table() + chronological_split()
  models.py                          Berlin-specific TabPFN-3.5 fit/predict wrapper (used by the MCP server + dashboard)
  baselines.py                       Berlin's naive historical-average baseline
  xgboost_baseline.py                generic XGBoost fit wrapper (classification + regression) — not Berlin-specific
  evaluate.py                        Berlin: TabPFN-3.5 vs. historical baseline -> results/metrics.json
  evaluate_db.py                     Deutsche Bahn: TabPFN-3.5 vs. XGBoost -> results/metrics_db.json

mcp_server/server.py                 FastMCP tools (Berlin only, for now): predict_overcrowding_risk, predict_expected_flow, …
agent/harness.py, cli.py             single-agent tool-calling loop over the MCP server (Berlin only, for now)

dashboard/
  app.py                             router (st.navigation)
  pages/
    0_Overview.py
    1_Berlin_Explore.py              data exploration — network map, flow, events, weather, closures, energy
    2_Finnish_Explore.py             data exploration — delay distribution, weather relationship, time patterns, station map, multi-year drift
    3_Live_Prediction.py             ask the Berlin TabPFN-3.5 models directly
    4_Benchmark_Berlin.py            TabPFN-3.5 vs. historical baseline, Berlin
    5_Benchmark_Deutsche_Bahn.py     TabPFN-3.5 vs. XGBoost, Deutsche Bahn

scripts/train_and_eval.py, train_and_eval_db.py    the two benchmark entrypoints Make targets call
```

**Why `datasets/berlin.py` etc. aren't a strict shared interface yet**: the three datasets
genuinely differ in shape — Berlin is wide-format station-flow series needing a melt step,
Deutsche Bahn and Finnish are already long/per-stop but with different column names, different
target-derivation rules (Berlin's `overcrowded` is a station-relative percentile; the other two's
`delayed` is an absolute 5-minute threshold), and Finnish needs a `months: list[str]` argument the
other two don't. Forcing a common `Dataset` class now would mean either a lowest-common-denominator
interface that hides those real differences, or an abstraction with three exceptions on day one.
The "Planned" section below is where that gets resolved — once the comparison code actually needs
it, informed by what it actually needs.

## Planned: scientific comparison phase

**Goal**: compare TabPFN-3.5 against other models (XGBoost already built; LightGBM/CatBoost/a
linear baseline are natural next additions) on classification AND regression, across all three
datasets, with one methodology — same train/test protocol, same metrics, same report format —
so the numbers are actually comparable to each other, not three one-off scripts that happen to
print similar-looking tables (`evaluate.py` and `evaluate_db.py` today are exactly that: similar
but hand-duplicated).

```
tabpfn_lab/
  datasets/
    base.py              a light Protocol/ABC: load_classification_table(), load_regression_table(),
                          FEATURES, CATEGORICAL, chronological_split() — written FROM the three existing
                          modules (matching what they already do), not imposed on them speculatively
    berlin.py             conforms to it (near-zero change — already has every piece)
    deutsche_bahn.py      conforms to it (near-zero change)
    finnish.py             conforms to it (near-zero change)
  models/
    base.py               fit(X, y) -> (model, fit_seconds); predict(model, X) -> (y_pred, predict_seconds) —
                           the shape xgboost_baseline.py and models.py already informally share
    tabpfn_model.py        generic TabPFN-3.5 classifier/regressor wrapper (today's models.py, minus the
                            Berlin-specific MCP-server caching concerns, which stay in mcp_server/)
    xgboost_model.py        today's xgboost_baseline.py, renamed for symmetry
    (lightgbm_model.py, catboost_model.py, linear_model.py — added if/when the comparison wants them)
  evaluation/
    runner.py              run_comparison(dataset, models, task) -> one metrics dict per (dataset, model, task) —
                            the generalization of evaluate.py / evaluate_db.py's near-duplicate logic
    report.py              turns a list of runner results into the comparison table/chart the showcase needs
                            (accuracy, fit time, predict time, data-efficiency — same shape already used in
                            the two Benchmark_* dashboard pages, generalized to N datasets × M models)
```

**Why this order, not a bigger rewrite now**: `datasets/base.py` and `models/base.py` get written
by looking at what `berlin.py`, `deutsche_bahn.py`, `finnish.py`, `models.py` and
`xgboost_baseline.py` actually already do — three to five concrete implementations before the
abstraction, not after a guess. `evaluation/runner.py` replaces the hand-duplicated
`evaluate.py`/`evaluate_db.py` pattern once a third near-duplicate (Finnish) would otherwise get
written; two duplicates is an acceptable amount of repetition to postpone that generalization
until it pays for itself.

## Planned: the showcase

Once `evaluation/report.py` exists, the natural output is a comparison dashboard page (or a
`scripts/full_comparison.py` writing one `results/comparison.json` the dashboard reads) showing,
per dataset: TabPFN-3.5 vs. every other model's accuracy, fit time, predict time, and
data-efficiency (rows TabPFN needed vs. rows the other model needed), the same shape the two
existing Benchmark_* pages already show for one model each — generalized, not reinvented.
