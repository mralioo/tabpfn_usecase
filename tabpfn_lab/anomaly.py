"""Anomaly-aware early warning for the Berlin U-Bahn: predict when a station will run
*off its normal pattern* (surge or collapse), and why.

Why this framing. Raw passenger flow is dominated by the daily/weekly oscillation
(rush hours, night gap, weekends). A model that predicts raw flow or "flow >= the
station's p90" mostly learns that clock — and so does a lookup table, which is why the
earlier benchmark (`tabpfn_lab/evaluate.py`) ties TabPFN with a groupby. An operator
does not need to be told that 08:00 is busy. They need to be told when 08:00 will be
*busier than a normal 08:00* — because a concert lets out, it pours, or a station is
shut — so they can add trains, staff the platform, or reroute.

Pipeline:
  1. Hourly flow per station (15-min counts are spiky; operators plan by the hour).
  2. Normal profile: robust median of log1p(flow) per (station, day type, hour), fit on
     the training split only.
  3. Anomaly score z = (log1p(flow) - normal) / MAD of that station-hour — a robust
     z-score of the residual once the oscillation is removed. `surge` = z >= SURGE_Z.
  4. Context features an operator knows ahead of time: event schedule mapped to the
     nearest U-Bahn station, planned closures (station + line suspensions on the graph),
     weather forecast, plus station descriptors.
  5. Models: historical-profile baseline (no context — predicts "normal"), XGBoost on
     the full history, XGBoost on the same small sample TabPFN sees, TabPFN-3.5.

Split: Alstom's own files — `*_pre_innotrans` (Jun 10 – Sep 21) trains, `*_rest`
(Sep 22 – Sep 30, InnoTrans week) tests. A second backtest fold (train < Sep 1, test
Sep 1–21) adds closures and rain the short hold-out week lacks. No shuffling, and the
normal profile is refit per fold so no test hour leaks into "normal".

Artifacts land in `results/anomaly/` (see `run_and_cache`), which the marimo notebook
`notebooks/04_anomaly_early_warning.py` and the webapp dashboard read.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .config import RESULTS_DIR, TABPFN_MODEL_PATH
from .datasets.berlin import (
    build_graph,
    discover_dataset_dir,
    load_closures,
    load_events,
    load_flows,
    load_stations,
    load_weather,
    station_cols,
)

ANOMALY_DIR = RESULTS_DIR / "anomaly"
DATA_END = pd.Timestamp("2026-10-01")  # flows end 2026-10-01 00:45 — drop that partial day
# Two expanding-window folds. Training is always everything before the test window, and the
# normal profile is refit per fold, so neither fold sees its own test hours.
FOLDS = {
    "holdout": {"start": pd.Timestamp("2026-09-22"), "end": DATA_END,
                "label": "Alstom hold-out · Sep 22–30 (InnoTrans week)"},
    "backtest": {"start": pd.Timestamp("2026-09-01"), "end": pd.Timestamp("2026-09-22"),
                 "label": "Backtest · Sep 1–21"},
}
TEST_START = FOLDS["holdout"]["start"]  # first day of Alstom's `*_rest` split
SURGE_Z = 2.0
DROP_Z = -2.0
MIN_SCALE = 0.15  # floor on the per-(station, hour) MAD so very regular hours don't explode z
TABPFN_CONTEXT_ROWS = 10_000
ALARM_BUDGET = 0.02  # operator capacity: act on the top 2% of station-hours

DAYTYPE_NAMES = {0: "weekday", 1: "Saturday", 2: "Sunday"}

# Events have an address but no station key (see dataset_schema.md). Curated map from an
# address/venue keyword to the nearest U-Bahn station (walking egress). Order matters:
# the first match wins, so specific house numbers come before the bare street name.
# Venues with no U-Bahn station within walking distance map to None and only count
# towards the city-wide event total.
VENUE_TO_STATION: list[tuple[str, str | None]] = [
    ("Marlene-Dietrich-Platz", "Potsdamer Platz"),
    ("Herbert-von-Karajan", "Potsdamer Platz"),
    ("Metzer Stra", "Senefelderplatz"),
    ("Sömmeringstra", "Mierendorffplatz"),
    ("Uber-Platz", "Warschauer Str."),
    ("Am Wriezener Bahnhof", "Warschauer Str."),
    ("Revaler Stra", "Warschauer Str."),
    ("Gendarmenmarkt", "Hausvogteiplatz"),
    ("Lenaustra", "Schönleinstr."),
    ("Budapester Stra", "Zoologischer Garten"),
    ("Kantstra", "Zoologischer Garten"),
    ("Schönhauser Allee 36", "Eberswalder Str."),
    ("Olympischer Platz", "Olympia-Stadion"),
    ("Friedrich-Friesen-Allee", "Olympia-Stadion"),
    ("Am Glockenturm", "Olympia-Stadion"),
    ("Hasenheide", "Hermannplatz"),
    ("Tempelhofer Damm", "Platz der Luftbrücke"),
    ("Columbiadamm", "Platz der Luftbrücke"),
    ("Platz der Luftbrücke", "Platz der Luftbrücke"),
    ("Märkisches Ufer", "Märkisches Museum"),
    ("Hermannstraße 146", "Boddinstr."),
    ("Skalitzer Str. 134", "Kottbusser Tor"),
    ("Skalitzer Stra", "Görlitzer Bahnhof"),
    ("Cuvrystra", "Schlesisches Tor"),
    ("Schlesisches Tor", "Schlesisches Tor"),
    ("Nollendorfplatz", "Nollendorfplatz"),
    ("Messedamm", "Kaiserdamm"),
    ("Oudenarder", "Seestr."),
    ("Möckernstra", "Möckernbrücke"),
    ("Mockernstra", "Möckernbrücke"),
    ("Invalidenstra", "Naturkundemuseum"),
    ("Obentrautstra", "Mehringdamm"),
    ("Am Juliusturm", "Zitadelle"),
    ("Treptower Str", "Neukölln"),
    ("Franklinstra", "Ernst-Reuter-Platz"),
    ("Friedrichstra", "Friedrichstr."),
    ("Holzmarktstra", "Jannowitzbrücke"),
    ("Karl-Marx-Straße 141", "Karl-Marx-Str."),
    ("Kurt-Schumacher-Damm", "Kurt-Schumacher-Platz"),
    ("Schnellerstra", None),
    ("Eichenstra", None),
    ("Paul-Heyse-Stra", None),
    ("Straße am FEZ", None),
    ("Lilli-Henoch", None),
    ("Wilhelm-Kabus", None),
]

PROFILE_FEATURES = [
    "hour", "daytype", "n_lines", "is_interchange", "primary_line", "normal_level", "station_scale",
]
CONTEXT_FEATURES = [
    "temp", "prcp", "prcp_3h", "wspd", "cldc", "coco",
    "event_ingress_att", "event_ongoing_att", "event_egress_att", "event_nearby_egress_att",
    "city_event_att_day",
    "station_closed", "line_suspended_here", "neighbor_closed", "network_closures_active",
]
FEATURES = PROFILE_FEATURES + CONTEXT_FEATURES
CATEGORICAL = ["primary_line"]

DRIVER_LABELS = {
    "event": "Event egress / ingress",
    "closure": "Closure (station, line or neighbour)",
    "rain": "Rain (>= 1 mm/h)",
    "heat": "Heat (>= 28 °C)",
    "none": "No known driver",
}


def _folder() -> str:
    return discover_dataset_dir()


def _short(name: str) -> str:
    return name.replace(" (Berlin)", "")


# ------------------------------------------------------------------------------ hourly flows
@lru_cache(maxsize=1)
def hourly_flows() -> pd.DataFrame:
    """Long table: ts, station, passengers (hourly sum of the 15-min counts). Keeps only the
    service hours present in the data (00 and 05–23) and drops duplicate flow columns."""
    flows = load_flows(_folder())
    cols = [c for c in station_cols(flows) if not c.endswith(".1")]
    h = flows.set_index("timestamp")[cols].resample("1h").sum(min_count=1)
    h = h[(h.index < DATA_END) & h.index.hour.isin([0, *range(5, 24)])].dropna(how="all")
    long = h.stack().rename("passengers").reset_index()
    long.columns = ["ts", "station", "passengers"]
    long["hour"] = long["ts"].dt.hour
    dow = long["ts"].dt.dayofweek
    long["daytype"] = np.select([dow < 5, dow == 5], [0, 1], 2)
    return long


# ----------------------------------------------------------------------- normal profile + z
def add_normal_profile(long: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp) -> pd.DataFrame:
    """Split at `test_start`, fit the robust normal profile on the training rows only, and
    score every row with the anomaly z. Rows at/after `test_end` are dropped."""
    out = long[long["ts"] < test_end].copy()
    out["split"] = np.where(out["ts"] >= test_start, "test", "train")
    out["log_p"] = np.log1p(out["passengers"])
    train = out[out["split"] == "train"]
    profile = train.groupby(["station", "daytype", "hour"])["log_p"].median().rename("normal_level")
    out = out.join(profile, on=["station", "daytype", "hour"])
    out["residual"] = out["log_p"] - out["normal_level"]
    # Spread per (station, hour): quiet hours (00:00, early morning) are far noisier in log
    # space than rush hours, and one spread per station turns every quiet hour into a "surge".
    resid_train = out.loc[out["split"] == "train", ["station", "hour", "residual"]]
    scale = (resid_train.groupby(["station", "hour"])["residual"]
             .apply(lambda r: 1.4826 * np.median(np.abs(r - np.median(r))))
             .clip(lower=MIN_SCALE).rename("station_scale"))
    out = out.join(scale, on=["station", "hour"])
    out["z"] = (out["residual"] / out["station_scale"]).clip(-10, 10)
    out["normal_passengers"] = np.expm1(out["normal_level"])
    out["surge"] = (out["z"] >= SURGE_Z).astype(int)
    out["drop"] = (out["z"] <= DROP_Z).astype(int)
    return out


# ------------------------------------------------------------------------- station metadata
@lru_cache(maxsize=1)
def station_meta() -> pd.DataFrame:
    stations = load_stations(_folder())
    meta = (stations.groupby("station_name")
            .agg(lines=("u_bahn_lines", lambda s: ",".join(sorted({x.strip() for v in s for x in v.split(",")}))),
                 ids=("station_id", lambda s: sorted(set(s))),
                 lon=("longitude", "mean"), lat=("latitude", "mean"))
            .reset_index().rename(columns={"station_name": "station"}))
    meta["primary_line"] = meta["lines"].str.split(",").str[0]
    meta["n_lines"] = meta["lines"].str.split(",").apply(len)
    meta["is_interchange"] = (meta["n_lines"] > 1).astype(int)
    return meta


@lru_cache(maxsize=1)
def station_graph() -> nx.Graph:
    """Station-name graph (interchange stations have one VBB id per line; collapse them)."""
    g_ids = build_graph(_folder())
    id_to_name = {i: row.station for row in station_meta().itertuples() for i in row.ids}
    g = nx.Graph()
    for a, b in g_ids.edges():
        na, nb = id_to_name.get(a), id_to_name.get(b)
        if na and nb and na != nb:
            g.add_edge(na, nb)
    return g


def find_station(fragment: str, names: list[str]) -> str | None:
    frag = fragment.strip().lower()
    exact = [n for n in names if _short(n).lower().removeprefix("s+u ").removeprefix("u ") == frag]
    if exact:
        return exact[0]
    hits = [n for n in names if frag in n.lower()]
    return min(hits, key=len) if hits else None


def venue_station(venue: object, address: object, names: list[str]) -> str | None:
    key = f"{venue if isinstance(venue, str) else ''} {address if isinstance(address, str) else ''}"
    for kw, st in VENUE_TO_STATION:
        if kw in key:
            return find_station(st, names) if st else None
    return None


# ------------------------------------------------------------------------- context features
def _to_local_naive(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert("Europe/Berlin").dt.tz_localize(None)


@lru_cache(maxsize=1)
def event_table() -> pd.DataFrame:
    """Events with parsed local times and the mapped U-Bahn station (None if no nearby station)."""
    ev = load_events(_folder()).copy()
    names = sorted(hourly_flows()["station"].unique())
    ev["start"] = _to_local_naive(ev["began_local"])
    ev["end"] = _to_local_naive(ev["estimated_end_local"])
    ev["end"] = ev["end"].where(ev["end"] > ev["start"], ev["start"] + pd.Timedelta(hours=3))
    ev["station"] = [venue_station(v, a, names) for v, a in zip(ev["venue_name"], ev["address"])]
    # Ticket tiers ("... | VIP Ticket", "... | Box seat") are listed as separate events with the
    # same crowd — keep one row per show, otherwise attendance is counted several times.
    ev["show"] = ev["event_name"].str.split("|").str[0].str.replace(r"\s+", " ", regex=True).str.strip()
    ev = (ev.dropna(subset=["start"]).sort_values("estimated_attendance", ascending=False)
          .drop_duplicates(["show", "start", "address"]))
    return ev.sort_values("start").reset_index(drop=True)


@lru_cache(maxsize=1)
def closure_table() -> pd.DataFrame:
    """Closures with the set of stations they shut. A line suspension shuts the shortest
    path between its two endpoints on the stations that line serves."""
    cl = load_closures(_folder()).copy()
    names = sorted(hourly_flows()["station"].unique())
    g = station_graph()
    meta = station_meta().set_index("station")
    closed_sets = []
    for _, c in cl.iterrows():
        if c["closure_type"] == "Station closure":
            st = find_station(str(c["affected_segment"]), names)
            closed_sets.append([st] if st else [])
        elif c["closure_type"] == "Line suspension":
            a, b = [find_station(x, names) for x in str(c["affected_segment"]).split("↔")]
            on_line = [n for n in g.nodes if c["affected_line"] in meta.loc[n, "lines"].split(",")]
            sub = g.subgraph(on_line) if on_line else g
            try:
                closed_sets.append(nx.shortest_path(sub, a, b))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                closed_sets.append([s for s in (a, b) if s])
        else:
            closed_sets.append([])
    cl["closed_stations"] = closed_sets
    return cl


def _hourly_weather() -> pd.DataFrame:
    w = load_weather(_folder()).set_index("timestamp")
    hw = w.resample("1h").agg({"temp": "mean", "prcp": "sum", "wspd": "mean", "cldc": "mean", "coco": "max"})
    hw["prcp_3h"] = hw["prcp"].rolling(3, min_periods=1).sum()
    return hw.reset_index().rename(columns={"timestamp": "ts"})


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(station_meta()[["station", "primary_line", "n_lines", "is_interchange", "lines"]],
                   on="station", how="left")
    out = out.merge(_hourly_weather(), on="ts", how="left")
    for c in ["temp", "prcp", "prcp_3h", "wspd", "cldc", "coco"]:
        out[c] = out[c].interpolate(limit_direction="both")

    key = pd.MultiIndex.from_arrays([out["station"], out["ts"]])
    hour = pd.Timedelta(hours=1)
    g = station_graph()

    # --- events: per-station attendance arriving / inside / leaving, plus neighbours leaving
    ev = event_table()
    ingress, ongoing, egress, near = ({} for _ in range(4))
    for e in ev.dropna(subset=["station"]).itertuples():
        s0, e0, att = e.start.floor("h"), e.end.floor("h"), float(e.estimated_attendance)
        for t in pd.date_range(s0 - hour, s0, freq="h"):
            ingress[(e.station, t)] = ingress.get((e.station, t), 0.0) + att
        for t in pd.date_range(s0 + hour, e0 - hour, freq="h"):
            ongoing[(e.station, t)] = ongoing.get((e.station, t), 0.0) + att
        for t in pd.date_range(e0, e0 + hour, freq="h"):
            egress[(e.station, t)] = egress.get((e.station, t), 0.0) + att
            for nb in g.neighbors(e.station) if e.station in g else []:
                near[(nb, t)] = near.get((nb, t), 0.0) + att
    for col, d in [("event_ingress_att", ingress), ("event_ongoing_att", ongoing),
                   ("event_egress_att", egress), ("event_nearby_egress_att", near)]:
        out[col] = pd.Series(d, dtype=float).reindex(key).fillna(0.0).to_numpy()
    ev_day = ev.groupby(ev["start"].dt.date)["estimated_attendance"].sum()
    out["city_event_att_day"] = out["ts"].dt.date.map(ev_day).fillna(0.0).astype(float)

    # --- closures: hour overlaps [when, end)
    closed, suspended, nbr_closed = set(), set(), set()
    active = pd.Series(0, index=out.index)
    ts = out["ts"]
    for c in closure_table().itertuples():
        hours = pd.date_range(c.when.floor("h"), (c.end - pd.Timedelta(seconds=1)).floor("h"), freq="h")
        active = active + ((ts + hour > c.when) & (ts < c.end)).astype(int)
        target = closed if c.closure_type == "Station closure" else suspended
        for st in c.closed_stations:
            for t in hours:
                target.add((st, t))
                for nb in g.neighbors(st) if st in g else []:
                    if nb not in c.closed_stations:
                        nbr_closed.add((nb, t))
    pairs = list(zip(out["station"], out["ts"]))
    out["station_closed"] = [int(p in closed) for p in pairs]
    out["line_suspended_here"] = [int(p in suspended) for p in pairs]
    out["neighbor_closed"] = [int(p in nbr_closed) for p in pairs]
    out["network_closures_active"] = active.to_numpy()

    out["driver"] = np.select(
        [out["event_ingress_att"] + out["event_egress_att"] + out["event_ongoing_att"] + out["event_nearby_egress_att"] > 0,
         out["station_closed"] + out["line_suspended_here"] + out["neighbor_closed"] > 0,
         out["prcp"] >= 1.0,
         out["temp"] >= 28.0],
        ["event", "closure", "rain", "heat"], "none")
    return out


@lru_cache(maxsize=1)
def context_table() -> pd.DataFrame:
    """Hourly flows + context features (fold-independent)."""
    return add_context(hourly_flows())


@lru_cache(maxsize=4)
def anomaly_table(fold: str = "holdout") -> pd.DataFrame:
    """The hourly station table for one fold: flows, normal profile, z, context, driver tag."""
    f = FOLDS[fold]
    return add_normal_profile(context_table(), f["start"], f["end"])


def encode(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    lines = sorted(station_meta()["primary_line"].unique())
    for col in CATEGORICAL:
        out[col] = pd.Categorical(out[col], categories=lines).codes
    return out.astype(float)


# ------------------------------------------------------------------------------- sampling
def context_sample(train: pd.DataFrame, n: int = TABPFN_CONTEXT_ROWS, seed: int = 0) -> pd.DataFrame:
    """In-context training set for TabPFN (and the small-data XGBoost twin).
    TabPFN learns from the rows it is shown, so the rare situations the operator cares about
    must be in the context: every event and closure hour goes in (a few thousand rows), then
    up to 20% of the budget of rain/heat hours, and the rest is a surge-stratified random
    draw of ordinary hours."""
    rng = np.random.default_rng(seed)
    rare = train[train["driver"].isin(["event", "closure"])]
    if len(rare) > n // 2:
        rare = rare.sample(n // 2, random_state=seed)
    weather = train[train["driver"].isin(["rain", "heat"])]
    weather = weather.sample(min(len(weather), n // 5), random_state=seed)
    pick_flag = pd.concat([rare, weather])
    rest = train.drop(pick_flag.index)
    rest = rest[rest["driver"] == "none"]
    n_rest = n - len(pick_flag)
    pos, neg = rest[rest["surge"] == 1], rest[rest["surge"] == 0]
    n_pos = min(len(pos), int(round(n_rest * 0.15)))
    sample = pd.concat([pick_flag, pos.sample(n_pos, random_state=seed),
                        neg.sample(n_rest - n_pos, random_state=seed)])
    return sample.iloc[rng.permutation(len(sample))]


# --------------------------------------------------------------------------------- models
def fit_profile_baseline(train: pd.DataFrame) -> pd.DataFrame:
    """No-context baseline: 'tomorrow looks like a normal day'. z forecast = 0; surge
    probability = historical surge rate of that (station, day type, hour)."""
    return (train.groupby(["station", "daytype", "hour"])["surge"].mean()
            .rename("p_surge").reset_index())


def predict_profile_baseline(table: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    p = rows[["station", "daytype", "hour"]].merge(table, on=["station", "daytype", "hour"], how="left")["p_surge"]
    return pd.DataFrame({"p_surge": p.fillna(0.0).to_numpy(), "z_hat": np.zeros(len(rows))}, index=rows.index)


def fit_xgboost(train: pd.DataFrame, features: list[str]) -> dict:
    from xgboost import XGBClassifier, XGBRegressor

    from .xgboost_baseline import DEFAULT_PARAMS

    # n_jobs=-1 oversubscribes on many-core hosts (42 s vs 0.2 s for 10k rows on a 22-core WSL box)
    params = {**DEFAULT_PARAMS, "n_jobs": 8}
    X, t0 = encode(train[features]), time.perf_counter()
    clf = XGBClassifier(**params, eval_metric="logloss").fit(X, train["surge"])
    reg = XGBRegressor(**params).fit(X, train["z"])
    return {"clf": clf, "reg": reg, "features": features, "fit_s": time.perf_counter() - t0, "n_train": len(train)}


def predict_xgboost(bundle: dict, rows: pd.DataFrame) -> pd.DataFrame:
    X, t0 = encode(rows[bundle["features"]]), time.perf_counter()
    out = pd.DataFrame({"p_surge": bundle["clf"].predict_proba(X)[:, 1], "z_hat": bundle["reg"].predict(X)},
                       index=rows.index)
    out.attrs["predict_s"] = time.perf_counter() - t0
    return out


def fit_tabpfn(sample: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

    from .config import authenticate

    authenticate()
    cat_idx = [features.index(c) for c in CATEGORICAL if c in features]
    X, t0 = encode(sample[features]), time.perf_counter()
    clf = TabPFNClassifier(model_path=TABPFN_MODEL_PATH, categorical_features_indices=cat_idx).fit(X, sample["surge"])
    reg = TabPFNRegressor(model_path=TABPFN_MODEL_PATH, categorical_features_indices=cat_idx).fit(X, sample["z"])
    return {"clf": clf, "reg": reg, "features": features, "fit_s": time.perf_counter() - t0, "n_train": len(sample)}


def predict_tabpfn(bundle: dict, rows: pd.DataFrame, chunk: int = 5_000) -> pd.DataFrame:
    X, t0 = encode(rows[bundle["features"]]), time.perf_counter()
    p, z = [], []
    for i in range(0, len(X), chunk):
        part = X.iloc[i:i + chunk]
        p.append(bundle["clf"].predict_proba(part)[:, 1])
        z.append(bundle["reg"].predict(part))
    out = pd.DataFrame({"p_surge": np.concatenate(p), "z_hat": np.concatenate(z)}, index=rows.index)
    out.attrs["predict_s"] = time.perf_counter() - t0
    return out


LEARNING_CURVE_SIZES = [5_000, 10_000, 25_000, 50_000, 100_000, 200_000]


def learning_curve(fold: str, sizes: list[int] = LEARNING_CURVE_SIZES) -> list[dict]:
    """XGBoost (context features) trained on growing slices of the fold's history, sampled the
    same way as TabPFN's context: how many rows does XGBoost need to match TabPFN-3.5 at 10k?"""
    table = anomaly_table(fold)
    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    out = []
    for n in [*[m for m in sizes if m < len(train)], len(train)]:
        rows = train if n == len(train) else context_sample(train, n=n)
        b = fit_xgboost(rows, FEATURES)
        sc = evaluate(test, {"x": predict_xgboost(b, test)}).loc["x"]
        out.append({"n_train": int(n), "pr_auc": _clean(sc["pr_auc"]), "recall_event": _clean(sc["recall_event"]),
                    "alarm_precision": _clean(sc["alarm_precision"]), "fit_s": round(b["fit_s"], 2)})
    return out


# ----------------------------------------------------------------------------- evaluation
def evaluate(test: pd.DataFrame, preds: dict[str, pd.DataFrame], budget: float = ALARM_BUDGET) -> pd.DataFrame:
    """One row per model: surge ranking quality, alarm-budget precision/recall, z error,
    and recall per driver (event / closure / rain / heat / none) inside the alarm budget."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y, rows = test["surge"].to_numpy(), []
    k = max(1, int(round(budget * len(test))))
    for name, p in preds.items():
        score = p["p_surge"].to_numpy()
        flagged = np.zeros(len(test), bool)
        flagged[np.argsort(-score, kind="stable")[:k]] = True
        tp = int((flagged & (y == 1)).sum())
        z_err = np.abs(test["z"].to_numpy() - p["z_hat"].to_numpy())
        pass_hat = np.expm1(test["normal_level"] + p["z_hat"] * test["station_scale"]).clip(lower=0)
        row = {
            "model": name,
            "roc_auc": roc_auc_score(y, score) if len(set(score)) > 1 else 0.5,
            "pr_auc": average_precision_score(y, score),
            "alarm_precision": tp / k,
            "alarm_recall": tp / max(1, int(y.sum())),
            "z_mae": float(z_err.mean()),
            "z_mae_context": float(z_err[test["driver"].to_numpy() != "none"].mean()),
            "passenger_mae": float(np.abs(test["passengers"] - pass_hat).mean()),
        }
        drivers = test["driver"].to_numpy()
        explained = (drivers != "none") & (y == 1)
        row["recall_explained"] = float(flagged[explained].mean()) if explained.any() else np.nan
        for d in ["event", "closure", "rain", "heat"]:
            m = (drivers == d) & (y == 1)
            row[f"recall_{d}"] = float(flagged[m].mean()) if m.any() else np.nan
        # collapses an operator can act on: a station that is actually shut and drops
        shut = (test["station_closed"].to_numpy() == 1) & (test["drop"].to_numpy() == 1)
        row["closure_drop_recall"] = float((p["z_hat"].to_numpy()[shut] <= DROP_Z).mean()) if shut.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ------------------------------------------------------------------------------ scenarios
def build_scenarios(test: pd.DataFrame, pred_cols: list[str], fold: str) -> list[dict]:
    """Operator stories inside one fold's test window: each mapped event, each closure, each
    rain spell. Every scenario carries the station-hour series (actual, normal, z) plus each
    model's forecast so the notebook and the dashboard replay exactly the same numbers."""
    lo_f, hi_f = FOLDS[fold]["start"], FOLDS[fold]["end"]
    out: list[dict] = []
    ev = event_table()
    ev = ev[(ev["start"] >= lo_f) & (ev["start"] < hi_f) & ev["station"].notna()]
    for e in ev.sort_values("estimated_attendance", ascending=False).drop_duplicates(["station", "start"]).itertuples():
        venue = e.venue_name if isinstance(e.venue_name, str) else e.address
        out.append(_scenario(test, pred_cols, fold, kind="event", station=e.station,
                             lo=e.start - pd.Timedelta(hours=4), hi=e.end + pd.Timedelta(hours=4),
                             title=f"{e.show} — {_short(e.station)}",
                             cause=f"{e.segment if isinstance(e.segment, str) else 'Event'} at {venue}, "
                                   f"~{int(e.estimated_attendance):,} attendees, {e.start:%a %d %b %H:%M}–{e.end:%H:%M}",
                             attendance=int(e.estimated_attendance)))
    cl = closure_table()
    for c in cl[(cl["when"] >= lo_f) & (cl["when"] < hi_f)].itertuples():
        if not c.closed_stations:
            continue
        out.append(_scenario(test, pred_cols, fold, kind="closure", station=c.closed_stations[0],
                             lo=c.when - pd.Timedelta(hours=4), hi=c.end + pd.Timedelta(hours=4),
                             title=f"{c.closure_type}: {c.affected_segment}",
                             cause=f"{c.description} ({c.duration}, from {c.when:%a %d %b %H:%M})",
                             extra_stations=c.closed_stations[1:]))
    rain = test.drop_duplicates("ts").set_index("ts")["prcp"]
    spell = rain[rain >= 1.0]
    if len(spell):
        groups = (spell.index.to_series().diff() != pd.Timedelta(hours=1)).cumsum()
        for _, hrs in spell.groupby(groups.to_numpy()):
            lo, hi = hrs.index.min(), hrs.index.max()
            out.append(_scenario(test, pred_cols, fold, kind="rain", station=None,
                                 lo=lo - pd.Timedelta(hours=4), hi=hi + pd.Timedelta(hours=4),
                                 title=f"Rain spell {lo:%a %d %b %H:%M}",
                                 cause=f"{hrs.sum():.1f} mm over {len(hrs)} h (peak {hrs.max():.1f} mm/h)"))
    return [s for s in out if s]


def _scenario(test, pred_cols, fold, kind, station, lo, hi, title, cause, extra_stations=(), attendance=None):
    w = test[(test["ts"] >= lo.floor("h")) & (test["ts"] <= hi.ceil("h"))]
    if station is None:  # network-wide: summed passengers, median z / forecasts across stations
        agg = {"passengers": "sum", "normal_passengers": "sum", "z": "median",
               **{c: "median" for c in pred_cols}}
        w = w.groupby("ts").agg(agg).reset_index()
    else:
        w = w[w["station"] == station]
    if w.empty:
        return None
    series = [{"ts": r.ts.isoformat(), "actual": round(float(r.passengers), 1),
               "normal": round(float(r.normal_passengers), 1), "z": round(float(r.z), 2),
               **{c: round(float(getattr(r, c)), 3) for c in pred_cols}} for r in w.itertuples()]
    peak = max(series, key=lambda s: abs(s["z"]))
    return {"id": f"{fold}-{kind}-{_short(station or 'network')}-{lo:%m%d%H}", "fold": fold, "kind": kind,
            "title": title, "cause": cause, "station": station,
            "station_short": _short(station) if station else "Network (median z)",
            "extra_stations": list(extra_stations), "attendance": attendance,
            "start": lo.isoformat(), "end": hi.isoformat(), "peak": peak, "series": series}


# --------------------------------------------------------------------------- orchestration
def run_fold(fold: str, live_tabpfn: bool = True, log=print) -> tuple[dict, pd.DataFrame, dict]:
    """Fit every model on one fold's training window, score its test window.
    Returns (fold metrics, prediction frame, {model name: fitted bundle})."""
    t0 = time.perf_counter()
    table = anomaly_table(fold)
    train, test = table[table["split"] == "train"], table[table["split"] == "test"]
    sample = context_sample(train)
    log(f"[{fold}] train {len(train):,} rows, test {len(test):,} rows, context sample {len(sample):,}")

    preds, timing, bundles = {}, {}, {}
    base = fit_profile_baseline(train)
    preds["Profile baseline"] = predict_profile_baseline(base, test)
    timing["Profile baseline"] = {"n_train": len(train), "fit_s": 0.0, "predict_s": 0.0}

    for name, rows, feats in [("XGBoost (profile only)", train, PROFILE_FEATURES),
                              ("XGBoost (full history)", train, FEATURES),
                              ("XGBoost (10k sample)", sample, FEATURES)]:
        b = fit_xgboost(rows, feats)
        p = predict_xgboost(b, test)
        preds[name], bundles[name] = p, b
        timing[name] = {"n_train": b["n_train"], "fit_s": b["fit_s"], "predict_s": p.attrs["predict_s"]}
        log(f"[{fold}] {name}: fit {b['fit_s']:.1f}s")

    if live_tabpfn:
        b = fit_tabpfn(sample)
        p = predict_tabpfn(b, test)
        preds["TabPFN-3.5 (10k context)"], bundles["TabPFN-3.5 (10k context)"] = p, b
        timing["TabPFN-3.5 (10k context)"] = {"n_train": b["n_train"], "fit_s": b["fit_s"],
                                              "predict_s": p.attrs["predict_s"]}
        log(f"[{fold}] TabPFN-3.5: fit {b['fit_s']:.1f}s, predict {p.attrs['predict_s']:.1f}s")

    scores = evaluate(test, preds)
    k = max(1, int(round(ALARM_BUDGET * len(test))))
    pred_frame = test[["ts", "station", "passengers", "normal_passengers", "normal_level", "station_scale",
                       "z", "surge", "drop", "driver"] + CONTEXT_FEATURES].copy()
    pred_frame.insert(0, "fold", fold)
    models = {}
    for name, p in preds.items():
        key = _model_key(name)
        pred_frame[f"p_{key}"] = p["p_surge"].to_numpy()
        pred_frame[f"z_hat_{key}"] = p["z_hat"].to_numpy()
        threshold = float(np.sort(p["p_surge"].to_numpy())[::-1][k - 1])
        models[name] = {"key": key, **{m: _clean(v) for m, v in scores.loc[name].items()},
                        **timing[name], "alarm_threshold": threshold}
    metrics = {
        "label": FOLDS[fold]["label"],
        "train": [str(train["ts"].min()), str(train["ts"].max())],
        "test": [str(test["ts"].min()), str(test["ts"].max())],
        "n_train_rows": len(train), "n_test_rows": len(test), "alarm_k": k,
        "test_surge_rate": float(test["surge"].mean()),
        "drivers_in_test": test.groupby("driver")["surge"].agg(["size", "sum"]).rename(
            columns={"size": "hours", "sum": "surges"}).astype(int).to_dict(orient="index"),
        "models": models,
        "xgboost_learning_curve": learning_curve(fold),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return metrics, pred_frame, bundles


def run_and_cache(live_tabpfn: bool = True, out_dir: Path = ANOMALY_DIR, log=print) -> dict:
    """Run both folds, write artifacts to `out_dir`, return the metrics dict:
      metrics.json            per-fold + pooled scores, timings, definitions
      test_predictions.parquet every test station-hour with actual z and each model's forecast
      scenarios.json          operator stories (events, closures, rain) with model forecasts
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    folds, frames, scenarios = {}, [], []
    for fold in FOLDS:
        m, frame, _ = run_fold(fold, live_tabpfn=live_tabpfn, log=log)
        folds[fold], frames = m, [*frames, frame]
        pred_cols = [c for c in frame.columns if c.startswith(("p_", "z_hat_"))]
        scenarios += build_scenarios(frame, pred_cols, fold)
    pred_frame = pd.concat(frames, ignore_index=True)
    pred_frame.to_parquet(out_dir / "test_predictions.parquet")

    key_by_name = {n: v["key"] for n, v in folds["holdout"]["models"].items()}
    pooled = evaluate(pred_frame, {n: pd.DataFrame({"p_surge": pred_frame[f"p_{k}"], "z_hat": pred_frame[f"z_hat_{k}"]})
                                   for n, k in key_by_name.items()})
    metrics = {
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "definitions": {"surge_z": SURGE_Z, "drop_z": DROP_Z, "alarm_budget": ALARM_BUDGET,
                        "tabpfn_context_rows": TABPFN_CONTEXT_ROWS, "min_scale": MIN_SCALE,
                        "profile_features": PROFILE_FEATURES, "context_features": CONTEXT_FEATURES},
        "folds": folds,
        "pooled": {"n_test_rows": len(pred_frame), "test_surge_rate": float(pred_frame["surge"].mean()),
                   "models": {n: {"key": key_by_name[n], **{m: _clean(v) for m, v in pooled.loc[n].items()}}
                              for n in pooled.index}},
        "live_tabpfn": live_tabpfn,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_clean))
    (out_dir / "scenarios.json").write_text(json.dumps(scenarios, indent=1, default=str))
    log(f"wrote {out_dir}: {len(pred_frame):,} predictions, {len(scenarios)} scenarios")
    return metrics


def _model_key(name: str) -> str:
    return {"Profile baseline": "profile", "XGBoost (profile only)": "xgb_profile",
            "XGBoost (full history)": "xgb_full", "XGBoost (10k sample)": "xgb_small",
            "TabPFN-3.5 (10k context)": "tabpfn"}[name]


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, np.integer):
        return int(v)
    return v


def load_cached(out_dir: Path = ANOMALY_DIR) -> dict:
    """Artifacts written by `run_and_cache` (raises FileNotFoundError if not run yet)."""
    return {
        "metrics": json.loads((out_dir / "metrics.json").read_text()),
        "scenarios": json.loads((out_dir / "scenarios.json").read_text()),
        "predictions": pd.read_parquet(out_dir / "test_predictions.parquet"),
    }
