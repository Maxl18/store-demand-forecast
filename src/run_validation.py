"""Fit on train (to 2017-06-30), forecast and score validation.

- run: .venv\\Scripts\\python.exe src\\run_validation.py
- test rows dropped right after reading
- records the lowest-MAE model (earlier rule) for disclosure
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, LAG_DAYS, make_features  # noqa: E402
from forecast import (  # noqa: E402
    BASELINES,
    HGB_PARAMS,
    METHODS,
    MODELS,
    RESULTS,
    RIDGE_PARAMS,
    SEED,
    TEST_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    VAL_WEEK_MONDAYS,
    load_scope,
    mae,
    metrics_long_table,
    predict,
    score,
    training_rows,
    forecast_period,
)
from guard import require  # noqa: E402
from headline import load_headline_record  # noqa: E402


def build_validation_predictions():
    """Fit on train, forecast validation; returns (table, feats, models)."""
    df, _ = load_scope(VAL_END)
    require(df["date"].max() < TEST_START, "test-period rows must not be loaded")
    feats = make_features(df)
    table, models = forecast_period(feats, TRAIN_END, VAL_START, VAL_END)
    return table, feats, models


def main():
    table, feats, models = build_validation_predictions()
    train = training_rows(feats, TRAIN_END)

    # - every method forecasts every row
    assert table[METHODS].notna().all().all()
    print(f"Training rows: {len(train)} ({train['date'].min().date()} .. {train['date'].max().date()})")
    print(f"Validation rows scored: {len(table)} "
          f"({table['date'].min().date()} .. {table['date'].max().date()}), same rows for all methods")

    scores = score(table, VAL_WEEK_MONDAYS)

    # - earlier rule (lowest MAE): reported, not used
    overall = scores["overall"]
    chosen = min(MODELS, key=lambda m: overall[m]["mae"])
    best_baseline = min(BASELINES, key=lambda m: overall[m]["mae"])
    gain = overall[best_baseline]["mae"] - overall[chosen]["mae"]
    gain_pct = 100 * gain / overall[best_baseline]["mae"]
    if gain > 0:
        reason = (f"{chosen} has the lowest overall daily MAE of the two models "
                  f"({overall[chosen]['mae']:.3f}) and beats the best baseline "
                  f"{best_baseline} ({overall[best_baseline]['mae']:.3f}) by {gain_pct:.1f}%.")
    else:
        reason = (f"{chosen} has the lower MAE of the two models, but the baseline "
                  f"{best_baseline} beats both models on validation.")

    # - sanity check: validation far below train error would hint at leakage
    train_mae = {m: mae(train["sales"], predict(models[m], train)) for m in MODELS}
    headline = load_headline_record()

    output = {
        "splits": {
            "train": f"{feats['date'].min().date()} .. {TRAIN_END.date()}",
            "validation": f"{VAL_START.date()} .. {VAL_END.date()}",
            "test": "2017-10-01 .. 2017-12-31 (not loaded by this run)",
            "validation_week_mondays": [str(d.date()) for d in VAL_WEEK_MONDAYS],
        },
        "features": FEATURES,
        "lag_days": LAG_DAYS,
        "training_rows": {
            "count": int(len(train)),
            "first_date": str(train["date"].min().date()),
            "last_date": str(train["date"].max().date()),
            "rule": "dates <= 2017-06-30 and every feature present (needs 364 days of history)",
        },
        "smape_definition": "100 * mean(2*|F-A| / (|A|+|F|)); a row where A and F are both 0 counts as 0",
        "forecast_clipping": "model forecasts below 0 are set to 0",
        "overall": overall,
        "per_item": scores["per_item"],
        "n_daily_rows": scores["n_daily_rows"],
        "n_item_weeks": scores["n_item_weeks"],
        "headline_model": headline["headline_model"],
        "headline_rule": headline["rule"],
        "headline_source": "results/headline_model.json (user decision D8; see its disclosure)",
        "lowest_daily_mae_model": chosen,
        "lowest_daily_mae_note": ("the rule fixed before the validation results "
                                  "(lowest overall daily MAE) picks this model; it was replaced "
                                  "by the order-cost rule after the validation results were seen"),
        "lowest_daily_mae_reason": reason,
        "best_baseline": best_baseline,
        "mae_gain_vs_best_baseline": gain,
        "mae_gain_vs_best_baseline_pct": gain_pct,
        "train_in_sample_mae": train_mae,
        "hyperparameters": {"ridge": RIDGE_PARAMS, "hgb": HGB_PARAMS},
        "hyperparameter_search": "none: library defaults (hgb early_stopping turned off)",
        "random_seed": SEED,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "validation_metrics.json").write_text(json.dumps(output, indent=2))
    long_table = metrics_long_table(scores)
    long_table.round(4).to_csv(RESULTS / "validation_metrics.csv", index=False)
    preds = table.copy()
    preds["date"] = preds["date"].dt.date
    preds.round(4).to_csv(RESULTS / "validation_predictions.csv", index=False)

    with pd.option_context("display.width", 120):
        print("\nOverall (daily MAE and SMAPE on all validation days, weekly MAE on 12 full weeks):")
        print(long_table[long_table["scope"] == "overall"].round(3).to_string(index=False))
        print("\nPer item daily MAE:")
        item_rows = long_table[long_table["scope"] != "overall"]
        per_item = item_rows.pivot(index="scope", columns="method", values="mae")
        per_item = per_item.loc[item_rows["scope"].unique(), METHODS]  # - item order
        print(per_item.round(3).to_string())
    print(f"\nIn-sample training MAE: { {m: round(v, 3) for m, v in train_mae.items()} }")
    print(f"Lowest daily MAE: {chosen}. {reason}")
    print(f"Headline model (results/headline_model.json): {headline['headline_model']}; "
          f"{headline['other_model']} is reported next to it.")


if __name__ == "__main__":
    main()
