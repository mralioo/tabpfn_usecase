.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help install test anomaly anomaly-offline app agent mcp notebooks train-db clean

help: ## Show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

install: $(VENV)/bin/activate ## Create the venv and install requirements.txt

test: install ## Engine, operations and MCP tests (no TabPFN/LLM API calls)
	$(PY) -m pytest tests/ -q

anomaly: install ## Benchmark baseline / XGBoost / TabPFN-3.5 on 2 folds -> results/anomaly/ (~4 min, live API)
	$(PY) scripts/run_anomaly_benchmark.py

anomaly-offline: install ## Same without TabPFN-3.5 API calls (XGBoost + baselines, ~30 s)
	$(PY) scripts/run_anomaly_benchmark.py --offline

app: install ## Webapp: Early-Warning Desk -> http://127.0.0.1:8000 · control-room agent with charts -> /agent
	$(PY) -m uvicorn webapp.server:app --host 127.0.0.1 --port 8000

agent: install ## Control-room agent in the terminal (LLM -> MCP -> TabPFN-3.5)
	$(PY) agent/cli.py

mcp: install ## Run the MCP server over stdio (for Claude Desktop or any MCP client)
	$(PY) mcp_server/server.py

notebooks: install ## Open the marimo notebooks (EDA, why p90 fails, anomaly early warning)
	$(PY) -m marimo edit notebooks/

train-db: install ## Supporting benchmark: TabPFN-3.5 vs XGBoost on real Deutsche Bahn delays -> results/metrics_db.json
	$(PY) scripts/train_and_eval_db.py

clean: ## Remove the venv and __pycache__ directories (keeps .env, data/, results/)
	rm -rf $(VENV)
	find . -type d -name __pycache__ -not -path "./.venv*/*" -exec rm -rf {} +
