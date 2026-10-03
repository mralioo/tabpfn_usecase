"""The real baseline TabPFN-3.5 has to earn its place against: a standard gradient-boosted
tree model (XGBoost), not a lookup-table heuristic.

Default, lightly-set hyperparameters — not an exhaustively tuned model. That's deliberate:
it represents what a team ships without a dedicated tuning pass, which is the realistic
comparison under a hackathon deadline (or most "we need a model by Friday" situations).
"""
from __future__ import annotations

import time

import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", n_jobs=-1, random_state=0,
)


def fit_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[XGBClassifier, float]:
    model = XGBClassifier(**DEFAULT_PARAMS, eval_metric="logloss")
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0


def fit_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[XGBRegressor, float]:
    model = XGBRegressor(**DEFAULT_PARAMS)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return model, time.perf_counter() - t0
