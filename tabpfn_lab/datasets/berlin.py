"""Berlin U-Bahn dataset: raw CSV loading + tabular feature/label engineering, one row per
(station, 15-min timestamp). Data: `data/Berlin_Ubahn_Alstom_data/` — see `data/SOURCE.md`.

Adapted from NextMove's `dashboard/utils/data_loader.py` + `ml/features.py` (same parsing
logic — mojibake repair, mixed-file merge, closure-description regexes — the Streamlit cache
decorators swapped for a plain `functools.lru_cache` so this package has no UI dependency).

Two targets, one feature table:
  - classification target `overcrowded`: 1 if flow >= the station's OWN historical 90th
    percentile (no platform-capacity figure exists in the schema, so "overcrowded" is defined
    relative to each station's normal operating range — the critical-use-case signal: an
    early-warning flag an operator would act on *before* the platform fills up)
  - regression target `passengers`: the raw flow count, for expected-demand forecasting
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import networkx as nx
import pandas as pd

from ..config import BERLIN_DATA_DIR

# Official-ish BVG U-Bahn line colors, used consistently across charts.
LINE_COLORS = {
    "U1": "#52822f", "U2": "#da421e", "U3": "#16683d", "U4": "#f0d722", "U5": "#7e5330",
    "U6": "#8c6dab", "U7": "#528dc8", "U8": "#224f86", "U9": "#f3791d",
}

OVERCROWD_QUANTILE = 0.90

FEATURE_COLUMNS = [
    "hour", "dow", "is_weekend", "month",
    "station_avg_passengers", "n_lines", "is_interchange", "primary_line",
    "temp", "prcp", "wspd", "cldc", "coco",
    "daily_event_count", "daily_event_attendance",
    "in_closure", "network_active_closures",
]
CATEGORICAL_COLUMNS = ["primary_line"]


# --------------------------------------------------------------------------------- raw loading
def discover_dataset_dir(base_dir: Path | str = BERLIN_DATA_DIR) -> str:
    """The one folder that holds every dataset file. Training and testing CSVs live in
    separate subfolders here (`training dataset/`, `testing dataset/`) but the testing
    split reuses the training split's station/line/connection files, so loaders below
    search the whole tree under `base_dir`, not a single subfolder."""
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"data dir not found: {base}")
    return str(base)


def _fix_mojibake(name: str) -> str:
    """'U KurfÃ¼rstenstr.' (UTF-8 read as latin-1 by the exporter) -> 'U Kurfürstenstr.'"""
    for enc in ("cp1252", "latin-1"):
        try:
            return name.encode(enc).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return name


def _read_one(path: Path, names: list[str] | None, expect: str | None, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", **kw)
    if expect and names and expect not in df.columns:  # header-less file (events test split): reuse training's columns
        df = pd.read_csv(path, encoding="utf-8-sig", header=None, names=names, **kw)
        for c in df.select_dtypes("object").columns:
            df[c] = df[c].map(lambda v: _fix_mojibake(v) if isinstance(v, str) else v)
    return df.rename(columns={c: _fix_mojibake(c) for c in df.columns if isinstance(c, str)})


def _read_merged(folder: str, pattern: str, expect: str | None = None, **kw) -> pd.DataFrame:
    """Merge every file matching `pattern` under `folder` (training + testing split in one frame)."""
    files = sorted(Path(folder).rglob(pattern))
    if not files:
        raise FileNotFoundError(f"no file matching {pattern} under {folder}")
    names = None
    if expect:
        for f in files:
            cols = list(pd.read_csv(f, encoding="utf-8-sig", nrows=0).columns)
            if expect in cols:
                names = cols
                break
    frames = [_read_one(f, names, expect, **kw) for f in files]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _find_one(folder: str, pattern: str) -> Path:
    matches = sorted(Path(folder).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern} under {folder}")
    return matches[0]


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", dayfirst=False)


@lru_cache(maxsize=4)
def load_stations(folder: str) -> pd.DataFrame:
    df = pd.read_csv(_find_one(folder, "stations_with_ubahn.csv"))
    df["primary_line"] = df["u_bahn_lines"].str.split(",").str[0].str.strip()
    df["n_lines"] = df["u_bahn_lines"].str.split(",").apply(len)
    df["is_interchange"] = df["n_lines"] > 1
    return df


@lru_cache(maxsize=4)
def load_connections(folder: str) -> pd.DataFrame:
    return pd.read_csv(_find_one(folder, "berlin_ubahn_connections.csv"))


@lru_cache(maxsize=4)
def load_lines(folder: str) -> pd.DataFrame:
    return pd.read_csv(_find_one(folder, "berlin_ubahn_lines_used.csv"))


@lru_cache(maxsize=4)
def load_flows(folder: str) -> pd.DataFrame:
    df = _read_merged(folder, "flows*.csv")
    df["timestamp"] = _parse_mixed_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


@lru_cache(maxsize=4)
def load_weather(folder: str) -> pd.DataFrame:
    df = _read_merged(folder, "weather_data*.csv")
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = _parse_mixed_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


@lru_cache(maxsize=4)
def load_events(folder: str) -> pd.DataFrame:
    df = _read_merged(folder, "berlin_events*.csv", expect="event_name") \
        .drop_duplicates(["event_name", "began_local"], keep="last")
    df["began_local"] = pd.to_datetime(df["began_local"], utc=False, format="mixed")
    df["estimated_end_local"] = pd.to_datetime(df["estimated_end_local"], utc=False, format="mixed", errors="coerce")
    df["date"] = df["began_local"].dt.date
    return df


_DURATION_RE = re.compile(r"(?:(\d+)h)?(?:(\d+)min)?")
_LINE_CLOSURE_RE = re.compile(r"Line (U\d+) suspended(?: on a section)? between (.+?) and (.+?) due to (.+)\.")
_STATION_CLOSURE_RE = re.compile(r"Station (.+?) closed due to (.+)\.")


def _parse_duration(text: str) -> pd.Timedelta:
    m = _DURATION_RE.fullmatch(text.strip())
    hours = int(m.group(1)) if m and m.group(1) else 0
    minutes = int(m.group(2)) if m and m.group(2) else 0
    return pd.Timedelta(hours=hours, minutes=minutes)


@lru_cache(maxsize=4)
def load_closures(folder: str) -> pd.DataFrame:
    df = _read_merged(folder, "closures*.csv").drop_duplicates(["when", "description"], keep="last").reset_index(drop=True)
    df["when"] = _parse_mixed_datetime(df["when"])
    df["duration_td"] = df["duration"].apply(_parse_duration)
    df["end"] = df["when"] + df["duration_td"]
    df["duration_hours"] = df["duration_td"].dt.total_seconds() / 3600

    kinds, lines, stations, reasons = [], [], [], []
    for desc in df["description"]:
        line_m = _LINE_CLOSURE_RE.match(desc)
        station_m = _STATION_CLOSURE_RE.match(desc)
        if line_m:
            kinds.append("Line suspension")
            lines.append(line_m.group(1))
            stations.append(f"{line_m.group(2)} ↔ {line_m.group(3)}")
            reasons.append(line_m.group(4))
        elif station_m:
            kinds.append("Station closure")
            lines.append(None)
            stations.append(station_m.group(1))
            reasons.append(station_m.group(2))
        else:
            kinds.append("Other")
            lines.append(None)
            stations.append(None)
            reasons.append(desc)
    df["closure_type"], df["affected_line"] = kinds, lines
    df["affected_segment"], df["reason"] = stations, reasons
    return df


@lru_cache(maxsize=4)
def load_energy(folder: str) -> pd.DataFrame:
    df = _read_merged(folder, "energy_consumption*.csv")
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = _parse_mixed_datetime(df["timestamp"])
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def station_cols(flows: pd.DataFrame) -> list[str]:
    return [c for c in flows.columns if c != "timestamp"]


@lru_cache(maxsize=4)
def flows_long(folder: str) -> pd.DataFrame:
    """Melted flows: timestamp, station_name, passengers — plus calendar parts."""
    flows = load_flows(folder)
    cols = station_cols(flows)
    long = flows.melt(id_vars="timestamp", value_vars=cols, var_name="station_name", value_name="passengers")
    long["date"] = long["timestamp"].dt.date
    long["hour"] = long["timestamp"].dt.hour
    long["dow"] = long["timestamp"].dt.dayofweek
    long["dow_name"] = long["timestamp"].dt.day_name()
    long["is_weekend"] = long["dow"] >= 5
    return long


@lru_cache(maxsize=4)
def station_avg_flow(folder: str) -> pd.DataFrame:
    flows = load_flows(folder)
    cols = station_cols(flows)
    n_days = flows["timestamp"].dt.date.nunique()
    avg_15min = flows[cols].mean().rename("avg_15min_passengers")
    total = flows[cols].sum().rename("total_passengers")
    daily_avg = (total / max(n_days, 1)).rename("avg_daily_passengers")
    peak = flows[cols].max().rename("peak_15min_passengers")
    out = pd.concat([avg_15min, total, daily_avg, peak], axis=1)
    out.index.name = "station_name"
    return out.reset_index()


@lru_cache(maxsize=4)
def hourly_profile(folder: str, station_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    long = flows_long(folder)
    if station_names:
        long = long[long["station_name"].isin(station_names)]
    return long.groupby(["hour", "is_weekend"], as_index=False)["passengers"].mean()


@lru_cache(maxsize=4)
def dow_hour_heatmap(folder: str, station_names: tuple[str, ...] | None = None) -> pd.DataFrame:
    long = flows_long(folder)
    if station_names:
        long = long[long["station_name"].isin(station_names)]
    pivot = long.pivot_table(index="dow_name", columns="hour", values="passengers", aggfunc="mean")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return pivot.reindex([d for d in order if d in pivot.index])


@lru_cache(maxsize=4)
def network_total_flow(folder: str) -> pd.DataFrame:
    flows = load_flows(folder)
    cols = station_cols(flows)
    return pd.DataFrame({"timestamp": flows["timestamp"], "total_passengers": flows[cols].sum(axis=1)})


@lru_cache(maxsize=4)
def build_graph(folder: str) -> nx.Graph:
    conn = load_connections(folder)
    g = nx.Graph()
    g.add_edges_from(conn[["station_id_1", "station_id_2"]].itertuples(index=False, name=None))
    return g


@lru_cache(maxsize=4)
def network_resilience(folder: str) -> pd.DataFrame:
    """Rank stations by fragmentation impact if removed: articulation-point status plus
    betweenness centrality, joined with average daily ridership — pure graph topology, no model."""
    g = build_graph(folder)
    stations = load_stations(folder)
    avg_flow = station_avg_flow(folder)

    articulation = set(nx.articulation_points(g))
    betweenness = nx.betweenness_centrality(g)
    id_to_name = dict(zip(stations["station_id"], stations["station_name"]))

    rows = []
    for node in g.nodes():
        h = g.copy()
        h.remove_node(node)
        components = list(nx.connected_components(h)) if h.number_of_nodes() else []
        rows.append({
            "station_id": node, "station_name": id_to_name.get(node, node),
            "is_articulation_point": node in articulation,
            "betweenness_centrality": betweenness.get(node, 0.0),
            "resulting_components": len(components),
            "largest_fragment_size": max((len(c) for c in components), default=0),
        })
    out = pd.DataFrame(rows).merge(avg_flow[["station_name", "avg_daily_passengers"]], on="station_name", how="left")
    out["fragmentation_score"] = out["betweenness_centrality"] * out["avg_daily_passengers"].fillna(0)
    return out.sort_values("fragmentation_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------- tabular feature table
def build_feature_table(folder: str) -> pd.DataFrame:
    long = flows_long(folder)
    stations = load_stations(folder)
    weather = load_weather(folder)
    events = load_events(folder)
    closures = load_closures(folder)

    thresholds = long.groupby("station_name")["passengers"].quantile(OVERCROWD_QUANTILE)
    long = long.merge(thresholds.rename("station_p90"), on="station_name")
    long["overcrowded"] = (long["passengers"] >= long["station_p90"]).astype(int)

    # Interchange stations appear as one row per line in stations_with_ubahn.csv but as one
    # column in flows.csv — collapse to one metadata row per physical station name first.
    station_meta = (
        stations.groupby("station_name")["u_bahn_lines"]
        .apply(lambda s: ",".join(sorted(set(",".join(s).split(",")))))
        .reset_index()
    )
    station_meta["primary_line"] = station_meta["u_bahn_lines"].str.split(",").str[0]
    station_meta["n_lines"] = station_meta["u_bahn_lines"].str.split(",").apply(len)
    station_meta["is_interchange"] = station_meta["n_lines"] > 1

    station_avg = long.groupby("station_name")["passengers"].mean().rename("station_avg_passengers")
    long = long.merge(station_avg, on="station_name")
    long = long.merge(station_meta, on="station_name", how="left")
    long = long.dropna(subset=["primary_line", "n_lines", "is_interchange"])  # unresolvable duplicate flow columns

    long["month"] = pd.to_datetime(long["timestamp"]).dt.month
    long["is_weekend"] = long["is_weekend"].astype(int)
    long["is_interchange"] = long["is_interchange"].astype(int)

    long = long.merge(weather[["timestamp", "temp", "prcp", "wspd", "cldc", "coco"]], on="timestamp", how="left")

    daily_events = (events.groupby("date")
                     .agg(daily_event_count=("event_name", "count"),
                          daily_event_attendance=("estimated_attendance", "sum"))
                     .reset_index())
    long = long.merge(daily_events, left_on="date", right_on="date", how="left")
    long["daily_event_count"] = long["daily_event_count"].fillna(0)
    long["daily_event_attendance"] = long["daily_event_attendance"].fillna(0)

    long["in_closure"] = 0
    ts = long["timestamp"]
    station_lines = long["u_bahn_lines"].fillna("")
    active_count = pd.Series(0, index=long.index)
    for _, c in closures.iterrows():
        in_window = (ts >= c["when"]) & (ts <= c["end"])
        active_count = active_count + in_window.astype(int)
        if c["closure_type"] == "Station closure" and c["affected_segment"]:
            matches = in_window & long["station_name"].str.contains(str(c["affected_segment"]).strip(), regex=False, case=False)
        elif c["closure_type"] == "Line suspension" and c["affected_line"]:
            matches = in_window & station_lines.apply(
                lambda ls, line=c["affected_line"]: line in [x.strip() for x in ls.split(",")] if ls else False)
        else:
            matches = pd.Series(False, index=long.index)
        long.loc[matches, "in_closure"] = 1
    long["network_active_closures"] = active_count

    keep = ["timestamp", "station_name", "passengers", "overcrowded"] + FEATURE_COLUMNS
    return long[keep].dropna(subset=FEATURE_COLUMNS)


def encode_categoricals(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Label-encode CATEGORICAL_COLUMNS. `reference` fixes the category set so a 1-row
    prediction frame gets the same codes the model was trained with."""
    out = df.copy()
    basis = reference if reference is not None else df
    for col in CATEGORICAL_COLUMNS:
        categories = pd.Categorical(basis[col]).categories
        out[col] = pd.Categorical(out[col], categories=categories).codes
    return out


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """First (1 - test_fraction) of days -> train pool, rest -> test pool. No shuffling: this
    is time-series data, a random split would leak the future into training."""
    dates = sorted(df["timestamp"].dt.date.unique())
    cutoff = dates[int(len(dates) * (1 - test_fraction))]
    train = df[df["timestamp"].dt.date < cutoff]
    test = df[df["timestamp"].dt.date >= cutoff]
    return train, test, cutoff


def stratified_subsample(df: pd.DataFrame, n: int, label_col: str, seed: int = 0) -> pd.DataFrame:
    """TabPFN is an in-context model (best up to ~10k rows, not a big-data model) — sample rather
    than feed the whole ~1.4M-row melted table. Oversamples the positive class a bit so a rare
    label (overcrowding ~10% base rate) isn't drowned out of a few-thousand-row sample."""
    n = min(n, len(df))
    frac_positive = df[label_col].mean()
    n_pos = min(int(round(n * max(frac_positive, 0.15))), int((df[label_col] == 1).sum()))
    n_neg = min(n - n_pos, int((df[label_col] == 0).sum()))
    pos = df[df[label_col] == 1].sample(n_pos, random_state=seed)
    neg = df[df[label_col] == 0].sample(n_neg, random_state=seed)
    return pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
