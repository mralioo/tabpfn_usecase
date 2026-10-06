# NextMove-TabPFN

**A minimal agent that hands crowd-risk and demand questions to TabPFN-3.5 over MCP, benchmarked
against a naive baseline on accuracy *and* speed — a deep dive into one model, for the
[TabPFN-3.5 hackathon](https://platform.priorlabs.ai/hackathon-3.5).**

> *The same way an agent hands over to a weather tool if it needs a temperature, it should hand
> over to a specialised model for predictions.* — the framing this hackathon asks for, and the
> exact pattern this repo demonstrates.

## Where this came from

This project started from **[NextMove](https://github.com/mralioo/NextMove)**, a four-role
multi-agent "AI team for the control room" (Dispatcher → Analyst ⇄ Inspector → Writer) built for
the InnoTrans 2026 hackathon on this same Berlin U-Bahn dataset. NextMove used `tabpfn-client`
(TabPFN v3.5) as one tool among many, behind a much larger pipeline. Working on it is what made the
author genuinely curious about how good tabular foundation models have gotten — this repo pulls
that one thread on its own: **just** TabPFN-3.5, prediction and classification, wrapped in the
smallest agent harness that still demonstrates the tool-call hand-off pattern. It is not a rerun of
NextMove's architecture.

## The use case

Berlin U-Bahn station-level passenger flow, 15-minute resolution. Two TabPFN-3.5 models fit on one
shared tabular feature table (time-of-day, weekday, weather, daily event load, closure state,
station context):

| Model | Task | Critical-use-case framing |
| --- | --- | --- |
| `TabPFNClassifier` | Will this station be at/above its own historical 90th-percentile flow in this 15-min slot? | **Early-warning** signal an operator acts on *before* a platform overfills — add staff, hold a train. |
| `TabPFNRegressor` | Expected passenger count at this station/timestamp | Operational planning — staffing, train frequency. |

Both are benchmarked (`tabpfn_lab/evaluate.py`) against a **historical-average baseline**
(station × weekend-flag × hour lookup table) on a **chronologically held-out** test split — no
date overlap with training. Honest result: on this dataset TabPFN-3.5 roughly **ties** that
baseline (ROC-AUC 0.853 vs. 0.855; MAE 94.1 vs. 92.1) — the simulated flow is strongly periodic,
so a one-line lookup table is already close to optimal, and no model clearly beats it. That's a
real finding, not a weak result to hide: it's also *why* the refined use case and the second
benchmark below exist.

## The refined use case: anomaly early warning

The p90 target above is mostly the clock (rush hours, night gap, weekends): over 80% of hourly
log-flow variance is explained by a per-(station, day type, hour) profile alone. An operator does
not need a model for that. They need to know when a station will run **off its normal pattern**
because of a concert, a closure or the weather, early enough to act. `tabpfn_lab/anomaly.py`:

1. **Normalise**: hourly flow per station, robust normal profile (median of `log1p(flow)` per
   station × day type × hour, training weeks only), anomaly score
   `z = residual / MAD(station, hour)`. `surge` = z ≥ 2, `drop` = z ≤ −2.
2. **Attribute**: context known a day ahead. Events are mapped to the nearest U-Bahn station
   (curated `VENUE_TO_STATION`) and ticket tiers deduplicated. Closures are resolved on the
   network graph (line suspensions → stations on the segment). Weather is included as well.
3. **Forecast** with five models on the same features: profile baseline (clock only), XGBoost
   (clock only), XGBoost (context, full history), XGBoost (context, same 10k rows as TabPFN),
   TabPFN-3.5 (10k-row in-context sample that includes every event/closure hour).
4. **Evaluate** on two expanding-window folds: Alstom's own hold-out (`*_rest`, Sep 22–30,
   InnoTrans week) and a Sep 1–21 backtest. Metrics are PR-AUC for surges, recall inside a 2%
   alarm budget (overall, event-driven, closure collapses) and z error.

**Real numbers** (`make anomaly`, both folds pooled, 100,200 test station-hours, surge rate 2.1%):

| Model | Training rows | PR-AUC | Event surges caught | Closure collapses forecast |
| --- | --- | --- | --- | --- |
| Profile baseline (clock only) | all | 0.049 | 14% | 0% |
| XGBoost, clock only | all | 0.074 | 16% | 0% |
| XGBoost, context | 10k (same as TabPFN) | 0.068 | 39% | 92% |
| **TabPFN-3.5, context** | **10k** | **0.089** | **58%** | **100%** |
| XGBoost, context | 277k–347k (full) | 0.110 | 76% | 92% |

Reading it plainly:
- **Context is what makes early warning possible.** Clock-only models never forecast a closure
  collapse and miss most event surges.
- **At equal data, TabPFN-3.5 wins** (+31% PR-AUC, ~1.5× the event surges). The XGBoost learning
  curve needs **~50k rows (5×)** to match TabPFN-3.5 at 10k.
- **With the full history (28–35× more rows), XGBoost wins.** TabPFN-3.5 is the tool for
  short-history situations (new station, venue or line; a new question that needs an answer
  today, no pipeline). A trained GBM is the tool once months of labelled data exist. Both use
  the same feature table.
- Most |z| ≥ 2 hours in this simulated data have no known driver (noise). That caps precision
  for every model. Line suspensions barely move flow and there is no spill-over to neighbouring
  stations; a real network would show both.
- Latency: TabPFN-3.5 fits in ~5 s and scores ~1–2 s per 1,000 station-hours over the API.
  That suits a day-ahead plan (168 stations × 20 h ≈ 3,400 rows), not a sub-second loop.

Walk-through: `notebooks/04_anomaly_early_warning.py`. Operator view: the **Early-Warning Desk**
(`make dashboard-closures`, http://127.0.0.1:8000/).

## A bigger, harder benchmark: TabPFN-3.5 vs. XGBoost on real railway data

The Berlin dataset turned out too easy for a baseline comparison to be interesting. So
`tabpfn_lab/db_data.py` + `tabpfn_lab/evaluate_db.py` run the same classification + regression
pairing — same topic (railway operations), same country (Germany) — on a large, real, public
dataset this time, against a real tuned model, **XGBoost**, not a heuristic:

**[Deutsche Bahn delay data](https://huggingface.co/datasets/piebro/deutsche-bahn-data)**
(CC BY 4.0, via the [piebro/deutsche-bahn-data](https://github.com/piebro/deutsche-bahn-data)
project, built from DB's own public timetable/delay feeds) — one month (June 2025) of real German
train stops: **1.5M rows**, 107 major stations, every train type from S-Bahn to ICE. Fetched via
`huggingface_hub.hf_hub_download` (no auth for this public dataset, reproducible with one
function call). Two targets from the same `delay_in_min` column (mirroring both the Berlin pair
above and the convention used in published railway-delay research, e.g. the FI-TW paper's 5-minute
threshold): classification = delayed ≥5 min vs. on time, regression = delay in minutes (outlier
-clipped to [-15, 120]).

TabPFN-3.5 is fit on a 10,000-row in-context sample; XGBoost is fit on the **full** chronological
training split (~1.2M rows) — deliberately not a fair fight on data volume, because that asymmetry
*is* the point being tested.

**Real numbers** (`python scripts/train_and_eval_db.py`, run 2026-10-01, chronological split —
last days of June held out):

| | TabPFN-3.5 (10k rows) | XGBoost (~1.2M rows) | Gap |
| --- | --- | --- | --- |
| Classification ROC-AUC | 0.734 | 0.751 | −2.3% relative |
| Regression MAE, minutes | **4.64** | 4.66 | **TabPFN slightly lower** |
| Fit time | 4.3s / 4.2s | 5.9s / 5.7s | comparable (XGBoost not slower here) |
| Predict time, 5k rows | 2.6s / 2.7s | 0.01s / 0.01s | **XGBoost much faster** (local compute vs. a network API call per batch) |

Reported plainly, including where XGBoost wins: **inference latency is not where TabPFN-3.5 wins
here** — it pays a network round-trip XGBoost doesn't. What it does win is **data efficiency**:
on classification it's within ~2% of XGBoost's ROC-AUC, and on the regression task it actually
**matches-or-beats** XGBoost's MAE — both from **under 1% of the training data**
(10,000 of ~1.2 million rows) and zero feature/hyperparameter tuning. The realistic pitch isn't
"faster than XGBoost at inference" — it's "a usable, competitive model from a `.fit()` call on a
small sample, when you don't have the data volume, time, or tuning budget to build the XGBoost
pipeline in the first place."

## Architecture

```
data/                  Berlin_Ubahn_Alstom_data/ (committed) · DB_data/ · Finnish_Railway_Operations_data/ (both gitignored)
   │                   — see data/SOURCE.md for provenance of all three
   ▼
tabpfn_lab/
   ├── datasets/       one loader module per dataset: berlin.py, deutsche_bahn.py, finnish.py
   ├── models.py, baselines.py, xgboost_baseline.py       TabPFN-3.5 / baseline / XGBoost wrappers
   ├── evaluate.py, evaluate_db.py                         benchmark runners
   ├── mcp_server/     FastMCP server exposing the two predict_* tools (+ resolve_station, describe_dataset)
   ├── agent/          one LLM tool-calling loop (LiteLLM) over the MCP server — picks the tool by use case
   ├── anomaly.py      anomaly early warning: normal profile + z, event/closure/weather context,
   │                   profile baseline / XGBoost / TabPFN-3.5 on two folds -> results/anomaly/
   ├── closure_impact.py   what-if engine behind webapp/ — real BFS + XGBoost + TabPFN-3.5 scoring
   ├── dashboard/      Streamlit, multi-page: Overview · Berlin Explore · Finnish Explore ·
   │                   Live Prediction · Benchmark (Berlin) · Benchmark (Deutsche Bahn)
   ├── webapp/         Starlette app, two pages: Early-Warning Desk (/) — anomaly map, timeline,
   │                   scenarios, model scoreboard, live TabPFN-3.5 re-score; Closure Impact Lab
   │                   (/closures) — map + chat what-if for hypothetical station/line closures
   └── notebooks/      marimo notebooks, one per use case — data EDA, demand/overcrowding,
                        closure impact + network resilience, energy forecasting, anomaly early warning
```

Two separate dashboards, two different jobs: `dashboard/` (Streamlit) is for *exploring* the
datasets and benchmarks; `webapp/` is the operator view. Its home page, the **Early-Warning
Desk**, replays the test weeks hour by hour: stations coloured by forecast or actual anomaly z,
alarm rings from the selected model, a timeline of alarms vs. actual surges, scenario cards
(events, closures, rain) with each model's verdict and a suggested action, and a one-click live
TabPFN-3.5 re-score. It reads `results/anomaly/` (`make anomaly`). The **Closure Impact Lab**
(`/closures`) is the live *what-if* tool: pick a real closure scenario and it runs XGBoost and
TabPFN-3.5 inference on the affected stations side by side. `make dashboard-streamlit` vs.
`make dashboard-closures`.

`notebooks/` (`make notebooks`) is where the use cases themselves got worked out: each marimo
notebook loads real data, does its own cleaning/preprocessing, and benchmarks TabPFN-3.5 against
XGBoost and a baseline for one specific question an operator (or an LLM agent) might ask — see
[`notebooks/README.md`](notebooks/README.md) for what each one covers.

No multi-agent pipeline, no knowledge graph, no ground-truth verification loop — see
[`ROADMAP.md`](ROADMAP.md)'s "explicitly out of scope" section for why, and NextMove if you want
that fuller system. Full module-by-module layout, including the planned next phase (a scientific,
same-methodology comparison of TabPFN-3.5 against other models across all three datasets):
[`docs/MODULES.md`](docs/MODULES.md).

## Data exploration

Before any modeling decision, two dashboard pages just look at the data: **Berlin — Explore**
(network map + fragmentation risk, passenger flow, events, weather correlation, closures, energy)
and **Finnish — Explore** (delay distribution, weather relationship, time-of-day/weekday
patterns, station map, and a multi-year punctuality-drift check across the dataset's 2018-2025
span — punctuality measurably got worse over the years, which is why the loaders all use a
chronological, not random, train/test split). The Finnish dataset (biggest of the three, real,
multi-year — see `data/SOURCE.md`) is exploration-only for now; see `docs/MODULES.md` for the
plan to bring it into the same benchmark structure as Berlin and Deutsche Bahn.

## Quickstart

```bash
git clone <this-repo-url> && cd nextmove-tabpfn
make install    # venv + pip install -r requirements.txt

cp .env.example .env
# edit .env: TABPFN_API_TOKEN (free, https://platform.priorlabs.ai/account/api-keys)
#            AGENT_LITELLM_MODEL + OPENAI_API_KEY (or another LiteLLM-supported provider)

make test                # data/feature smoke tests, no API calls needed
make dashboard-streamlit  # streamlit: Berlin/Finnish exploration + live prediction + both benchmarks
make anomaly              # anomaly early warning: baseline / XGBoost / TabPFN-3.5 on 2 folds -> results/anomaly/ (~4 min)
make anomaly-offline      # same without TabPFN-3.5 API calls (~30 s)
make dashboard-closures   # webapp: Early-Warning Desk (/) + Closure Impact Lab (/closures) -> http://127.0.0.1:8000
make notebooks            # marimo exploration notebooks: one per use case, loads real data + fits real models
make cli                  # interactive agent: ask a crowd-risk or demand question
make train                # Berlin: TabPFN-3.5 vs. historical baseline -> results/metrics.json
make train-db             # Deutsche Bahn: TabPFN-3.5 vs. XGBoost -> results/metrics_db.json
make help                 # list every target
```

(No `make`? The underlying commands are plain `python -m venv .venv`, `pip install -r
requirements.txt`, `pytest tests/`, `streamlit run dashboard/app.py`,
`uvicorn webapp.server:app --port 8000`, `marimo edit notebooks/`, `python agent/cli.py`,
`python scripts/train_and_eval.py`, `python scripts/train_and_eval_db.py`,
`python scripts/run_anomaly_benchmark.py` — see `Makefile`.)

Example CLI questions:
- `"will Alexanderplatz be overcrowded at 2026-07-15 08:00:00?"` → agent calls `resolve_station`,
  then routes to `predict_overcrowding_risk`
- `"how many passengers are expected at U Rudow (Berlin) on 2026-08-01 17:30:00?"` → routes to
  `predict_expected_flow`

## Data

`data/Berlin_Ubahn_Alstom_data/` — generated by **Alstom's data team** for the **InnoTrans 2026
hackathon**, reused here unmodified and committed to the repo. Full attribution and schema:
[`data/SOURCE.md`](data/SOURCE.md) (all three datasets),
[`data/Berlin_Ubahn_Alstom_data/dataset_schema.md`](data/Berlin_Ubahn_Alstom_data/dataset_schema.md).

The Deutsche Bahn and Finnish datasets are **not** committed to this repo. Deutsche Bahn is
fetched on demand from Hugging Face (see `tabpfn_lab/datasets/deutsche_bahn.py`) — a public,
stable, citable source (CC BY 4.0) satisfying the hackathon's "available at a public URL" rule
without a multi-GB parquet file in version control. The Finnish dataset was placed locally from
Kaggle (see `data/SOURCE.md` for the reproducibility caveat this one has that the other two
don't).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE). Applies to the code in this repository; see
[`data/SOURCE.md`](data/SOURCE.md) for the data's own provenance.
