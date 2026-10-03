.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help install test dashboard dashboard-streamlit dashboard-closures notebooks cli train train-db clean

help: ## Show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

install: $(VENV)/bin/activate ## Create the venv and install requirements.txt

test: install ## Run the data/feature smoke tests (no TabPFN/LLM API calls, no .env needed)
	$(PY) -m pytest tests/ -q

dashboard: dashboard-streamlit ## Alias for dashboard-streamlit (back-compat)

dashboard-streamlit: install ## Streamlit: Berlin/Finnish exploration, live prediction, both benchmarks
	$(PY) -m streamlit run dashboard/app.py

dashboard-closures: install ## Closure Impact Lab: live XGBoost vs TabPFN-3.5 what-if map + chat -> http://127.0.0.1:8000
	$(PY) -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000

notebooks: install ## Open the marimo exploration notebooks (data EDA, demand/overcrowding, closures, energy)
	$(PY) -m marimo edit notebooks/

cli: install ## Run the interactive agent CLI (ask a crowd-risk or demand question)
	$(PY) agent/cli.py

train: install ## Fit TabPFN-3.5 on the Berlin U-Bahn data, benchmark vs. the historical baseline -> results/metrics.json
	$(PY) scripts/train_and_eval.py

train-db: install ## Benchmark TabPFN-3.5 vs. XGBoost on the Deutsche Bahn railway data -> results/metrics_db.json
	$(PY) scripts/train_and_eval_db.py

clean: ## Remove the venv and __pycache__ directories (keeps .env, data/, results/)
	rm -rf $(VENV)
	find . -type d -name __pycache__ -not -path "./.venv*/*" -exec rm -rf {} +
