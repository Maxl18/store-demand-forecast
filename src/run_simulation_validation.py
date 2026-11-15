"""Validation orders: tune buffers, cost them.

- run: .venv\\Scripts\\python.exe src\\run_simulation_validation.py
- cheapest buffer per method and ratio on 96 item-weeks, frozen in results/buffers.json
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import (  # noqa: E402
    BASELINES,
    METHODS,
    MODELS,
    RESULTS,
    TEST_START,
    VAL_END,
    VAL_START,
    VAL_WEEK_MONDAYS,
)
from simulate import (  # noqa: E402
    BUFFER_GRID,
    COST_RATIOS,
    HEADLINE_RATIO,
    ROUNDING_RULE,
    TIE_BREAK_RULE,
    best_baselines,
    build_tables,
    check_weeks,
    leave_one_week_out,
    save_buffers,
    tune_all,
    weekly_table,
)
from guard import require  # noqa: E402
from headline import load_headline_model  # noqa: E402

PREDICTIONS_FILE = RESULTS / "validation_predictions.csv"

SIMPLIFICATIONS = [
    "Weekly totals only: running out on a single day is ignored if the week's order covers the week's total.",
    "Leftover stock is not carried into the next week; every week is judged on its own.",
    "Costs (short 3 : over 1, plus 1:1 and 5:1) are assumptions, not measured.",
    "One buffer per method, tuned on only 12 validation weeks (96 item-weeks); the tuned "
    "validation cost is optimistic (see leave_one_week_out_check).",
]


def load_validation_predictions(path=PREDICTIONS_FILE, tries=10):
    """Read validation forecasts; check complete.

    - retries: run_validation.py may be rewriting the file
    """
    scope = json.loads((RESULTS / "scope.json").read_text())
    expected_rows = len(pd.date_range(VAL_START, VAL_END)) * len(scope["items"])  # - 92 x 8 = 736
    for _ in range(tries):
        try:
            daily = pd.read_csv(path)
            daily["date"] = pd.to_datetime(daily["date"])
            if len(daily) == expected_rows and daily[["actual"] + METHODS].notna().all().all():
                break
        except (OSError, ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError(f"could not read a complete {path}")

    # - validation days only, no test rows
    require(daily["date"].min() == VAL_START and daily["date"].max() == VAL_END,
            "validation predictions must cover exactly 2017-07-01 .. 2017-09-30")
    require(daily["date"].max() < TEST_START, "validation predictions contain test-period rows")
    require(sorted(daily["item"].unique()) == sorted(scope["items"]),
            "validation predictions do not have exactly the scope items")
    return daily


def main():
    daily = load_validation_predictions()
    check_weeks(VAL_WEEK_MONDAYS, "validation")
    weekly = weekly_table(daily, VAL_WEEK_MONDAYS)
    require(len(weekly) == 96, "expected 12 weeks x 8 items")
    headline_model = load_headline_model()
    print(f"Order weeks: {len(VAL_WEEK_MONDAYS)} (Mondays {VAL_WEEK_MONDAYS[0].date()} .. "
          f"{VAL_WEEK_MONDAYS[-1].date()}), item-weeks: {len(weekly)}, headline model: {headline_model}")

    # - tune and freeze buffers
    buffers, curves = tune_all(weekly)
    best_baseline = best_baselines(weekly, buffers)
    save_buffers(buffers, best_baseline, headline_model, weekly)
    edge = [(r, m, b) for r, by_m in buffers.items() for m, b in by_m.items()
            if b in (BUFFER_GRID[0], BUFFER_GRID[-1])]

    result, long_table = build_tables(weekly, buffers, best_baseline, headline_model)

    # - optimism check: leave one week out
    lowo = {}
    for ratio, (c_short, c_over) in COST_RATIOS.items():
        lowo[ratio] = {}
        for m in METHODS:
            lowo[ratio][m] = {
                "tuned_cost_in_sample": result["results"][ratio]["tuned_buffer"][m]["total_cost"],
                "leave_one_week_out_cost": leave_one_week_out(weekly, m, c_short, c_over),
            }

    result.update({
        "period": f"{VAL_START.date()} .. {VAL_END.date()} (validation)",
        "forecasts_source": "results/validation_predictions.csv",
        "buffer_grid": BUFFER_GRID,
        "rounding_rule": ROUNDING_RULE,
        "tie_break_rule": TIE_BREAK_RULE,
        "buffers_at_grid_edge": edge,
        "simplifications": SIMPLIFICATIONS,
        "leave_one_week_out_check": lowo,
        "cost_curve_by_buffer": {HEADLINE_RATIO: {m: {str(b): c for b, c in curves[HEADLINE_RATIO][m].items()}
                                                  for m in METHODS}},
    })
    (RESULTS / "simulation_validation.json").write_text(json.dumps(result, indent=2))
    long_table.round(4).to_csv(RESULTS / "simulation_validation.csv", index=False)

    print("\nTotal cost on 96 validation item-weeks (tuned buffer / buffer 0):")
    for ratio in COST_RATIOS:
        print(f"  cost ratio {ratio}, best baseline (tuned) = {best_baseline[ratio]['tuned_buffer']}")
        for m in METHODS:
            t = result["results"][ratio]["tuned_buffer"][m]
            z = result["results"][ratio]["zero_buffer"][m]
            print(f"    {m:16s} buffer {t['buffer']:+.2f}  cost {t['total_cost']:6d}  "
                  f"| buffer 0 cost {z['total_cost']:6d}  | leave-one-week-out {lowo[ratio][m]['leave_one_week_out_cost']:6d}")
    print("\nPer item cost, cost ratio 3:1, tuned buffers:")
    per_item = pd.DataFrame({m: result["results"][HEADLINE_RATIO]["tuned_buffer"][m]["cost_per_item"]
                             for m in METHODS})
    print(per_item.to_string())
    print("\nModel minus best baseline (negative = model cheaper), bootstrap over 12 weeks:")
    for ratio in COST_RATIOS:
        for setting in ("tuned_buffer", "zero_buffer"):
            for m in MODELS:
                u = result["uncertainty"][ratio][setting][m]
                print(f"  {ratio} {setting:12s} {m:5s} vs {u['baseline']:15s}: {u['total_difference']:+7.0f} "
                      f"({u['pct_difference']:+.1f}%), 95% CI [{u['ci95_low']:+.0f}, {u['ci95_high']:+.0f}], "
                      f"weeks won {u['weeks_a_cheaper']}/{u['n_weeks']}, "
                      f"item-weeks won {u['item_weeks_model_cheaper']}/{len(weekly)}")
    print(f"\n{headline_model} minus the other model, tuned buffers (negative = {headline_model} cheaper):")
    for ratio in COST_RATIOS:
        u = result["headline_vs_other_model"][ratio]["tuned_buffer"]
        print(f"  {ratio} vs {u['model_b']}: {u['total_difference']:+.0f}, 95% CI "
              f"[{u['ci95_low']:+.0f}, {u['ci95_high']:+.0f}], weeks {headline_model} cheaper "
              f"{u['weeks_a_cheaper']}/{u['n_weeks']}")
    print(f"\nBuffers at the edge of the grid: {edge if edge else 'none'}")
    print(f"Frozen buffers written to {RESULTS / 'buffers.json'}")


if __name__ == "__main__":
    main()
