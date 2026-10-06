"""tabpfn_lab — the core package.

- `datasets/berlin.py`: Berlin U-Bahn loaders (flows, stations, graph, events, weather, closures).
- `anomaly.py`: anomaly early-warning engine — normal profile, z-score, context features, models
  (profile baseline, XGBoost, TabPFN-3.5), two-fold benchmark -> `results/anomaly/`.
- `operations.py`: operator queries over the engine (network outlook, line corridor, station
  forecast, explain, live what-if) — used by the MCP server, the webapp and the notebooks.
- `datasets/deutsche_bahn.py` + `evaluate_db.py`: the supporting big-data benchmark.

No agent framework, MCP or web code in here — those live one layer up (`agent/`, `mcp_server/`,
`webapp/`) and import from this package.
"""
