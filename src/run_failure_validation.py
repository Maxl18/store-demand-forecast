"""Failure analysis on validation predictions.

- run: .venv\\Scripts\\python.exe src\\run_failure_validation.py
- refits on train only; checks rebuilt forecasts match the CSV
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import SIM_SETTING, run_failure_analysis  # noqa: E402
from forecast import BASELINES, MODELS, RESULTS, TEST_START, TRAIN_END  # noqa: E402
from headline import check_headline_consistency, load_headline_record  # noqa: E402
from run_validation import build_validation_predictions  # noqa: E402


def main():
    check_headline_consistency()
    record = load_headline_record()
    headline_model, cost_ratio = record["headline_model"], record["headline_ratio"]

    pred = pd.read_csv(RESULTS / "validation_predictions.csv")
    pred["date"] = pd.to_datetime(pred["date"])
    assert pred["date"].max() < TEST_START, "the prediction table must not contain test-period rows"
    sim = pd.read_csv(RESULTS / "simulation_validation.csv")

    # - rebuild (train only) for train sales and fitted Ridge
    table, feats, models = build_validation_predictions()
    train_sales = feats.loc[feats["date"] <= TRAIN_END, ["date", "item", "sales"]]
    merged = pred.merge(table, on=["date", "item"], suffixes=("", "_rebuilt"))
    assert len(merged) == len(pred) == len(table)
    for col in BASELINES + MODELS:
        diff = np.abs(merged[col] - merged[f"{col}_rebuilt"]).max()
        # - CSV has 4 decimals: small diffs are rounding
        assert diff < 1e-3, f"rebuilt forecasts for {col} differ from the CSV (max diff {diff})"

    summary = run_failure_analysis(
        pred_df=pred,
        period_label="validation",
        model_cols=MODELS,
        baseline_cols=BASELINES,
        out_dir=RESULTS,
        headline_model=headline_model,
        train_sales=train_sales,
        sim_table=sim,
        cost_ratio=cost_ratio,
        sim_setting=SIM_SETTING,
        ridge_pipeline=models["ridge"],
    )

    print("Failure analysis written for period:", summary["period_label"],
          "| headline model:", headline_model, "| cost ratio:", cost_ratio, "| setting:", SIM_SETTING)
    print("Rows:", summary["n_daily_rows"], "| item-weeks:", summary["n_item_weeks"])
    print("Best baseline (daily MAE):", summary["best_baseline_by_overall_daily_mae"])
    for m in MODELS:
        print(m, "daily MAE", summary["overall"][m]["mae"], "bias", summary["overall"][m]["bias"],
              "weekly MAE", summary["overall"][m]["weekly_mae"])
    print("Files are in:", RESULTS)


if __name__ == "__main__":
    main()
