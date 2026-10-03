"""TabPFN-3.5 model wrappers: fit a classifier + regressor on a stratified
in-context sample, predict fast.

`tabpfn_client`'s `TabPFNClassifier`/`TabPFNRegressor` keep their session as
module-level client state, not on the object itself — pickling a *fitted*
instance across process restarts is fragile (it round-trips only as much as
`sklearn.base.BaseEstimator.get_params()` can recover, and broke on a
`tabpfn_client` version bump during development: see `results/metrics.json`
for one such caught case). So this module caches only the plain-data
stratified *sample* a fit was built on (a DataFrame + a cutoff date — safe to
pickle, nothing API-client-shaped), and re-fits from it on every process
start. That refit is itself part of the pitch: ~4s on this dataset (see
`fit_seconds` below), which is cheap enough to pay on every MCP server / CLI
/ dashboard start rather than engineer a fragile cross-process model cache
around it.

This is the module every other layer (MCP server, dashboard, benchmark) calls
into — one place that knows how to talk to `tabpfn_client`.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR, TABPFN_MODEL_PATH, authenticate
from .datasets.berlin import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, encode_categoricals, stratified_subsample

CACHE_DIR = RESULTS_DIR / "model_cache"
SAMPLE_SIZE = 8_000


@dataclass
class FittedModels:
    classifier: object
    regressor: object
    train_sample: pd.DataFrame  # used as the category reference for encode_categoricals
    cutoff_date: object
    fit_seconds: float
    status: str  # 'fitted' | 'fitted-from-cached-sample'


def _sample_cache_path() -> Path:
    return CACHE_DIR / "train_sample.pkl"


def _fit_from_sample(sample: pd.DataFrame, cutoff_date: object, status: str) -> FittedModels:
    authenticate()
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor

    X = encode_categoricals(sample[FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]

    t0 = time.perf_counter()
    clf = TabPFNClassifier(model_path=TABPFN_MODEL_PATH, categorical_features_indices=cat_idx)
    clf.fit(X, sample["overcrowded"])
    reg = TabPFNRegressor(model_path=TABPFN_MODEL_PATH, categorical_features_indices=cat_idx)
    reg.fit(X, sample["passengers"])
    fit_seconds = time.perf_counter() - t0

    return FittedModels(classifier=clf, regressor=reg, train_sample=sample, cutoff_date=cutoff_date,
                         fit_seconds=fit_seconds, status=status)


def fit_models(train_pool: pd.DataFrame, cutoff_date: object, sample_size: int = SAMPLE_SIZE, seed: int = 0) -> FittedModels:
    """One TabPFNClassifier + one TabPFNRegressor, both fit on the SAME fresh stratified sample
    of the chronological train pool (classification target: `overcrowded`; regression target:
    `passengers`) — one feature table, two use cases, exactly the 'hand the right tool to the
    right prediction' pattern the agent harness picks between. Caches the sample (not the fitted
    objects) so a later process reuses the same rows instead of re-sampling."""
    sample = stratified_subsample(train_pool, sample_size, "overcrowded", seed=seed)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _sample_cache_path().write_bytes(pickle.dumps({"sample": sample, "cutoff_date": cutoff_date}))
    return _fit_from_sample(sample, cutoff_date, status="fitted")


def load_or_fit(train_pool: pd.DataFrame, cutoff_date: object, force: bool = False) -> FittedModels:
    if not force and _sample_cache_path().exists():
        cached = pickle.loads(_sample_cache_path().read_bytes())
        return _fit_from_sample(cached["sample"], cached["cutoff_date"], status="fitted-from-cached-sample")
    return fit_models(train_pool, cutoff_date)


def predict_row(models: FittedModels, row: pd.Series) -> dict:
    """Both predictions for a single (station, timestamp) feature row, with wall-clock latency —
    the number the hackathon pitch ('speed up prediction with high accuracy') rests on."""
    X_row = encode_categoricals(pd.DataFrame([row[FEATURE_COLUMNS]]), reference=models.train_sample[FEATURE_COLUMNS])

    t0 = time.perf_counter()
    proba = float(models.classifier.predict_proba(X_row)[0, 1])
    classify_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    expected = float(models.regressor.predict(X_row)[0])
    regress_seconds = time.perf_counter() - t0

    return {
        "overcrowding_probability": round(proba, 3),
        "predicted_label": "overcrowded" if proba >= 0.5 else "normal",
        "predicted_passengers": round(expected, 1),
        "classify_latency_ms": round(classify_seconds * 1000, 1),
        "regress_latency_ms": round(regress_seconds * 1000, 1),
        "model": f"TabPFNClassifier/TabPFNRegressor({TABPFN_MODEL_PATH})",
    }
