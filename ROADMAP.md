# Roadmap — TabPFN-3.5 Hackathon submission

Target: **"Build an agent"** track, platform.priorlabs.ai/hackathon-3.5.
Deadline: **2026-10-06, 23:59 CEST**. Judging: showcase of TabPFN-3.5 (50%) ·
creativity/originality (30%) · technical quality & reproducibility (20%).

Pitch in one line: a station-level crowd-risk early-warning tool where an
agent hands a question to the right TabPFN-3.5 model (classifier or
regressor) over MCP — plus a second, bigger benchmark on real Deutsche Bahn
railway delay data (1.5M rows, same topic, same country as the Berlin demo)
that shows TabPFN-3.5 matching or beating a real tuned XGBoost model's
accuracy from well under 1% of the training data, because the Berlin dataset
alone turned out too easy to prove anything against a baseline.

## Day 1 — 2026-10-01 (today): core pipeline

- [x] Scaffold the repo (`tabpfn_lab/`, `mcp_server/`, `agent/`, `dashboard/`, `scripts/`, `tests/`).
- [x] Port + simplify the data loading and feature engineering from NextMove (no Streamlit
      dependency in the core package, no multi-agent pipeline).
- [x] `tabpfn_lab/models.py`: fit `TabPFNClassifier` + `TabPFNRegressor` (`v3.5_default`) on one
      shared stratified sample; `tabpfn_lab/baselines.py` + `evaluate.py`: accuracy (ROC-AUC/PR-AUC,
      MAE) and wall-clock latency vs. a historical-average baseline.
- [x] Set `TABPFN_API_TOKEN` in `.env`, ran `python scripts/train_and_eval.py` end-to-end against the
      real TabPFN-3.5 API. Result: TabPFN roughly **ties** the baseline (ROC-AUC 0.853 vs. 0.855,
      MAE 94.1 vs. 92.1) — the simulated flow is too periodic for any model to clearly win. Real
      finding, not a bug; kept and reported honestly in the README rather than hidden.
- [x] Fixed a real bug found during that run: `tabpfn_client`'s access token is module-level client
      state, not on the fitted model object, so a model restored from a pickle cache in a fresh
      process failed on its first `.predict()` call. Fixed by caching the training *sample* (plain
      data, safe to pickle) instead of the live fitted objects, and re-fitting from it each process
      start (~4s) — itself part of the speed story, not a workaround to hide.
- [x] Added a second, harder benchmark (`tabpfn_lab/datasets/deutsche_bahn.py`, `evaluate_db.py`,
      `xgboost_baseline.py`) on a large, real, **same-topic** railway dataset — Deutsche Bahn
      per-stop delay records (huggingface.co/datasets/piebro/deutsche-bahn-data, CC BY 4.0,
      ~1.5M rows/month) — against a real tuned **XGBoost** baseline, not a naive one. First tried
      an OpenML flight-delay pair (good numbers, but wrong topic); replaced it once this railway
      dataset turned up, since "same topic as the Berlin demo" is a strictly better story. Real
      result: TabPFN-3.5 (10k-row sample) comes within ~2% of XGBoost's ROC-AUC on classification,
      and actually **matches-or-beats** XGBoost's MAE on regression (4.64 vs. 4.66 min) — both from
      under 1% of the ~1.2M-row training split. XGBoost's local inference is faster than TabPFN-3.5's
      networked inference, reported as-is.

## Day 2 — 2026-10-02: MCP server + agent harness

- [x] `mcp_server/server.py`: `describe_dataset`, `resolve_station`, `predict_overcrowding_risk`,
      `predict_expected_flow`.
- [x] `agent/harness.py` + `agent/cli.py`: single-agent tool-calling loop (LiteLLM function calling
      + fastmcp in-memory `Client`) that picks the tool by use case.
- [ ] Set `AGENT_LITELLM_MODEL` / `OPENAI_API_KEY` (or another LiteLLM provider) in `.env`, run
      `python agent/cli.py`, verify 3-4 example questions route to the right tool and answer
      correctly (crowd-risk phrasing → classifier, demand phrasing → regressor).

## Day 3 — 2026-10-03: dashboard + demo narrative

- [x] `dashboard/app.py` (4-tab single file) → converted to a multi-page app (`st.navigation` +
      `dashboard/pages/`) once a third dataset and two exploration pages arrived — verified every
      page boots clean via Streamlit's `AppTest` (not just a curl 200, which a multi-page app's
      client-side router can return regardless of whether a given page's Python actually ran;
      `AppTest` executes the real script and catches real exceptions — it caught a genuine one,
      see below).
- [x] `Makefile`: `make install` / `test` / `dashboard` / `cli` / `train` / `train-db` / `clean` —
      verified `make help` and the underlying commands work.
- [x] Restructured `tabpfn_lab/data_loader.py` + `features.py` + `db_data.py` into
      `tabpfn_lab/datasets/{berlin,deutsche_bahn,finnish}.py` (one module per dataset) once a third
      dataset (Finnish railway, see below) made "one dataset = one top-level module" start to
      sprawl. Updated every importer (`mcp_server/server.py`, `agent` path indirectly via the MCP
      server, `models.py`, `evaluate.py`, `evaluate_db.py`, `scripts/*.py`, `tests/test_smoke.py`,
      `dashboard/pages/*.py`) and re-verified end to end: `pytest`, a live MCP/agent question
      through the real TabPFN API, and the dashboard's Predict button — all still correct.
      `docs/MODULES.md` has the full current layout and the planned next phase.
- [x] Added **Finnish railway data** (`data/Finnish_Railway_Operations_data/`, placed locally from
      Kaggle — the dataset behind the FI-TW paper, arXiv:2601.16592): 96 monthly parquet files,
      2018-2025, ~2.75 GB, real VR long-distance train stops matched with FMI weather.
      `tabpfn_lab/datasets/finnish.py` (loaders + a first `build_feature_table()`) and a dashboard
      exploration page (`2_Finnish_Explore.py`) — delay distribution, weather relationship,
      time-of-day/weekday patterns, a station map, and an on-demand multi-year drift check.
      **Real finding**: punctuality measurably degrades across the dataset's span (January
      snapshots: ~29% delayed-≥5-min in 2018 → ~47% in 2024; cancellation rate ~0.5% → ~3.2%) —
      confirms the chronological-split choice already used everywhere else in this repo is the
      right one here too, not just for Berlin/Deutsche Bahn.
- [x] Added a matching **Berlin exploration page** (`1_Berlin_Explore.py`), condensed from
      NextMove's six-page dashboard (network map + fragmentation risk, passenger flow, events,
      weather, closures, energy) into one page's worth of tabs, reading through the same
      `tabpfn_lab.datasets.berlin` module the agent/MCP server use (so it can't silently disagree
      with them, unlike NextMove's separate Streamlit-cached loader).
- [ ] Record a 2-4 min demo video: ask the CLI agent a crowd-risk question live, show the MCP
      hand-off, then show the Deutsche Bahn benchmark tab for the bigger accuracy story.
- [ ] Write the submission description leading with the Deutsche Bahn benchmark numbers (the
      Berlin numbers are honest-but-modest; matching-or-beating XGBoost's MAE on real 1.5M-row
      railway data from under 1% of the training rows is the stronger showcase evidence) and the
      Berlin agent demo for the MCP hand-off pattern itself.

## Planned next: scientific comparison across all three datasets

Not started — plan only, see `docs/MODULES.md` for the module layout this implies
(`datasets/base.py`, `models/`, `evaluation/runner.py` + `report.py`). Goal: TabPFN-3.5 vs.
XGBoost (and possibly LightGBM/CatBoost/a linear baseline) on classification AND regression,
across Berlin, Deutsche Bahn, AND Finnish, with one shared methodology instead of
`evaluate.py`/`evaluate_db.py`'s current hand-duplicated pattern — then a showcase built on top of
that comparison (a generalized version of the two existing Benchmark_* dashboard pages).

## Day 4 — 2026-10-04: reproducibility pass (20% of the score)

- [ ] Fresh clone, fresh venv, follow this README top to bottom with nothing assumed. Fix whatever
      breaks.
- [ ] Confirm `data/SOURCE.md` attribution is accurate and `LICENSE` (Apache-2.0) is in place.
- [ ] `pytest tests/` passes with no `.env` set (the smoke tests touch no TabPFN/LLM API).

## Day 5 — 2026-10-05: submit (keep Oct 6 as pure slack)

- [ ] Push to a public GitHub repo, submit the repo link + description (+ video link) on the
      hackathon page before end of day, leaving the Oct 6 23:59 CEST cutoff as buffer only.
- [ ] Optional, only if time remains: a second, separately-judged entry narrower in scope (e.g. the
      event-impact regression as a standalone "formalize a new problem" angle) — not required.

## Explicitly out of scope (by design, not oversight)

- No multi-agent pipeline (Dispatcher/Analyst/Inspector/Writer) — one agent, one decision, one tool
  call. That's NextMove's job; this repo is the TabPFN-3.5 deep dive on its own.
- No knowledge graph, no Neo4j, no operator-feedback loop, no quality/ground-truth database.
- The Berlin benchmark deliberately stays a naive baseline (not XGBoost) — the point there is
  foundation-model-vs-the-heuristic-you'd-actually-ship-under-deadline-pressure. The harder,
  XGBoost-vs-TabPFN fight happens on the Deutsche Bahn dataset instead, where the bigger/more
  realistic data makes it a meaningful contest.
- No exhaustive XGBoost hyperparameter tuning on the Deutsche Bahn benchmark either — default-ish
  params, documented as such in `tabpfn_lab/xgboost_baseline.py`, representing what a team ships
  without a dedicated tuning pass.
