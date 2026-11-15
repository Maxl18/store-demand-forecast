"""Split weeks, baselines, models, metrics (shared by validation and test).

- test run: run_final_test.py, behind guard.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import (  # noqa: E402
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    load_data,
    select_scope,
)
from features import FEATURES, SALES_FEATURES, has_all_features  # noqa: E402
from guard import require, require_active_ticket  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# - splits: train to 2017-06-30, val 2017-07-01 to 09-30, test 2017-10-01 to 12-31
# - test only via run_final_test.py
# - full Mon to Sun order weeks
VAL_WEEK_MONDAYS = pd.date_range("2017-07-03", "2017-09-18", freq="7D")   # - 12 weeks
TEST_WEEK_MONDAYS = pd.date_range("2017-10-02", "2017-12-25", freq="7D")  # - 13 weeks

SEED = 0

# - library defaults; early stopping off (random holdout ignores time order)
RIDGE_PARAMS = {"alpha": 1.0}
HGB_PARAMS = {
    "learning_rate": 0.1,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "early_stopping": False,
    "random_state": SEED,
}

BASELINES = ["baseline_lag7", "baseline_mean4", "baseline_lag364"]
MODELS = ["ridge", "hgb"]
METHODS = BASELINES + MODELS


def load_scope(until, ticket=None):
    """Load chosen store and items up to `until`.

    - later rows dropped right after reading
    - cut-off after 2017-09-30 needs the guard's ticket
    """
    if until > VAL_END:
        require_active_ticket(ticket, "load_scope with a cut-off after 2017-09-30")
        require(ticket.results_dir == RESULTS.resolve(),
                "the ticket belongs to a different results folder")
    scope = json.loads((RESULTS / "scope.json").read_text())
    df = load_data(ROOT / "train.csv")
    df = select_scope(df, scope["store"], scope["items"])
    df = df.loc[df["date"] <= until, ["date", "item", "sales"]].reset_index(drop=True)
    require(df["date"].max() <= until, "rows after the cut-off date are still present")
    return df, scope


def baseline_forecasts(rows):
    """The three baseline forecasts."""
    return {
        "baseline_lag7": rows["lag_7"],                  # - same weekday last week
        "baseline_mean4": rows["mean_same_weekday_4"],   # - last 4 same weekdays
        "baseline_lag364": rows["lag_364"],              # - same weekday last year
    }


# - one model for all items; item is a feature
def make_ridge():
    """Ridge in a Pipeline: encoder and scaler see training rows only."""
    prep = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ["item", "day_of_week", "month"]),
        ("scale", StandardScaler(), ["day_of_year"] + SALES_FEATURES),
    ])
    return Pipeline([("prep", prep), ("ridge", Ridge(**RIDGE_PARAMS))])


def make_hgb():
    """Gradient boosting; item as a category."""
    return HistGradientBoostingRegressor(categorical_features=["item"], **HGB_PARAMS)


MODEL_MAKERS = {"ridge": make_ridge, "hgb": make_hgb}


def training_rows(feats, train_end):
    """Fit rows: up to train_end, all features present."""
    keep = (feats["date"] <= train_end) & has_all_features(feats)
    return feats.loc[keep]


def fit_model(name, feats, train_end):
    """Fit one model on training rows only."""
    rows = training_rows(feats, train_end)
    require(rows["date"].max() <= train_end, "a training row is dated after the training end")
    model = MODEL_MAKERS[name]()
    model.fit(rows[FEATURES], rows["sales"])
    return model


def predict(model, rows):
    """Model forecast, clipped at 0."""
    return np.clip(model.predict(rows[FEATURES]), 0, None)


def forecast_period(feats, train_end, start, end, methods=METHODS):
    """Fit up to train_end, forecast every (item, day) from start to end.

    - returns (table, models); one column per method, same rows for all
    """
    require(train_end < start, "the forecast period must start after the training period")
    rows = feats.loc[(feats["date"] >= start) & (feats["date"] <= end)]
    # - all features needed: same rows for every method
    require(has_all_features(rows).all(), "some forecast rows are missing features")

    table = rows[["date", "item"]].copy()
    table["actual"] = rows["sales"]
    for name, values in baseline_forecasts(rows).items():
        if name in methods:
            table[name] = values
    models = {}
    for name in MODELS:
        if name in methods:
            models[name] = fit_model(name, feats, train_end)
            table[name] = predict(models[name], rows)
    return table.reset_index(drop=True), models


def mae(actual, forecast):
    """MAE in units."""
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(forecast, float))))


def smape(actual, forecast):
    """SMAPE in percent (0 to 200).

    - per row 2 * |F - A| / (|A| + |F|); both 0 counts as 0
    """
    a = np.asarray(actual, float)
    f = np.asarray(forecast, float)
    denom = np.abs(a) + np.abs(f)
    terms = np.where(denom == 0, 0.0, 2 * np.abs(f - a) / np.where(denom == 0, 1, denom))
    return float(100 * np.mean(terms))


def weekly_totals(table, mondays, methods=METHODS):
    """Weekly sums per item (full Mon to Sun weeks)."""
    week_start = table["date"] - pd.to_timedelta(table["date"].dt.day_of_week, unit="D")
    in_full_week = week_start.isin(mondays)
    weeks = table.loc[in_full_week].assign(week_monday=week_start[in_full_week])
    grouped = weeks.groupby(["item", "week_monday"])
    require((grouped.size() == 7).all(), "a full week must have 7 days")
    return grouped[["actual"] + list(methods)].sum().reset_index()


def score(table, mondays, methods=METHODS):
    """Daily MAE, SMAPE, weekly MAE; overall and per item."""
    weekly = weekly_totals(table, mondays, methods)
    result = {"overall": {}, "per_item": {}}
    for m in methods:
        result["overall"][m] = {
            "mae": mae(table["actual"], table[m]),
            "smape": smape(table["actual"], table[m]),
            "weekly_mae": mae(weekly["actual"], weekly[m]),
        }
    for item in sorted(table["item"].unique()):
        day = table[table["item"] == item]
        week = weekly[weekly["item"] == item]
        result["per_item"][int(item)] = {
            m: {
                "mae": mae(day["actual"], day[m]),
                "smape": smape(day["actual"], day[m]),
                "weekly_mae": mae(week["actual"], week[m]),
            }
            for m in methods
        }
    result["n_daily_rows"] = int(len(table))
    result["n_item_weeks"] = int(len(weekly))
    return result


def metrics_long_table(scores, methods=METHODS):
    """Scores as one flat table."""
    lines = []
    for m in methods:
        lines.append({"scope": "overall", "method": m, **scores["overall"][m]})
    for item, by_method in scores["per_item"].items():
        for m in methods:
            lines.append({"scope": f"item_{item}", "method": m, **by_method[m]})
    return pd.DataFrame(lines)
