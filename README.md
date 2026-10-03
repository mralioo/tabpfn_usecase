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
real finding, not a weak result to hide: it's also *why* the second benchmark below exists.

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
   ├── closure_impact.py   what-if engine behind webapp/ — real BFS + XGBoost + TabPFN-3.5 scoring
   ├── dashboard/      Streamlit, multi-page: Overview · Berlin Explore · Finnish Explore ·
   │                   Live Prediction · Benchmark (Berlin) · Benchmark (Deutsche Bahn)
   ├── webapp/         Closure Impact Lab — Starlette app: U-Bahn map + operator chat + live
   │                   XGBoost-vs-TabPFN-3.5 comparison for hypothetical station/line closures
   └── notebooks/      marimo notebooks, one per use case — data EDA, demand/overcrowding,
                        closure impact + network resilience, energy forecasting
```

Two separate dashboards, two different jobs: `dashboard/` (Streamlit) is for *exploring* the
datasets and benchmarks; `webapp/` (Closure Impact Lab) is a live *what-if* tool — pick a real
closure scenario from the Berlin dataset, and it runs genuine XGBoost and TabPFN-3.5 inference
(same `tabpfn_lab` code the MCP server uses) on the affected stations, side by side, with an
agent-trace panel showing each MCP/model call as it happens. `make dashboard-streamlit` vs.
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
make dashboard-closures   # Closure Impact Lab: live XGBoost vs TabPFN-3.5 what-if map -> http://127.0.0.1:8000
make notebooks            # marimo exploration notebooks: one per use case, loads real data + fits real models
make cli                  # interactive agent: ask a crowd-risk or demand question
make train                # Berlin: TabPFN-3.5 vs. historical baseline -> results/metrics.json
make train-db             # Deutsche Bahn: TabPFN-3.5 vs. XGBoost -> results/metrics_db.json
make help                 # list every target
```

(No `make`? The underlying commands are plain `python -m venv .venv`, `pip install -r
requirements.txt`, `pytest tests/`, `streamlit run dashboard/app.py`,
`uvicorn webapp.server:app --port 8000`, `marimo edit notebooks/`, `python agent/cli.py`,
`python scripts/train_and_eval.py`, `python scripts/train_and_eval_db.py` — see `Makefile`.)

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
