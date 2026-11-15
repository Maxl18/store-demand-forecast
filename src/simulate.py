"""Weekly orders from daily forecasts, and their cost.

- order = sum of 7 daily forecasts x (1 + buffer), whole units
- cost = C_SHORT per unit short + C_OVER per unit left over
- simplified: weekly totals only, no carry-over, assumed costs, buffers from 12 validation weeks
- test period only reads the frozen buffers
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import (  # noqa: E402
    BASELINES,
    METHODS,
    MODELS,
    RESULTS,
    TEST_END,
    TEST_START,
    TEST_WEEK_MONDAYS,
    VAL_END,
    VAL_START,
    VAL_WEEK_MONDAYS,
    weekly_totals,
)
from guard import require, require_active_ticket  # noqa: E402

# - cost assumption, not measured: short = 3, left over = 1
C_SHORT = 3
C_OVER = 1

# - three ratios (short : over); 3:1 is the headline
COST_RATIOS = {"1:1": (1, 1), "3:1": (C_SHORT, C_OVER), "5:1": (5, 1)}
HEADLINE_RATIO = "3:1"

# - buffers -20% to +40%, 1-point steps
# - negatives from a validation finding (mean4 over-forecasts after July), not test data
BUFFER_GRID = [round(step / 100, 2) for step in range(-20, 41)]

ROUNDING_RULE = (
    "order = (sum of the 7 daily forecasts) x (1 + buffer), rounded to the nearest whole "
    "unit, an exact half rounded up (120.5 -> 121). Computed as floor(x + 0.5 + 1e-9): the "
    "tiny 1e-9 stops computer rounding noise (120.49999999999999) from turning a true half down."
)
TIE_BREAK_RULE = (
    "lowest total cost on the 12 validation weeks; if several buffers tie, the one closest "
    "to 0; if still tied, the smaller one"
)

# - bootstrap over weeks, fixed seed
N_BOOT = 10_000
BOOT_SEED = 0

BUFFERS_FILE = RESULTS / "buffers.json"

# - asserted: catches wrong date ranges
EXPECTED_WEEKS = {
    "validation": (12, pd.Timestamp("2017-07-03"), pd.Timestamp("2017-09-18")),
    "test": (13, pd.Timestamp("2017-10-02"), pd.Timestamp("2017-12-25")),
}

SETTINGS = ["tuned_buffer", "zero_buffer"]


def round_order(x):
    """Nearest whole unit, half up (ROUNDING_RULE)."""
    return np.floor(np.asarray(x, dtype=float) + 0.5 + 1e-9).astype(int)


def order_quantity(weekly_forecast, buffer):
    """Weekly forecast x (1 + buffer), rounded, at least 0."""
    raw = np.asarray(weekly_forecast, dtype=float) * (1 + buffer)
    return np.maximum(round_order(raw), 0)


def shortfall(actual, order):
    """Units short (demand above order)."""
    return np.maximum(np.asarray(actual) - np.asarray(order), 0)


def leftover(actual, order):
    """Units left over (order above demand)."""
    return np.maximum(np.asarray(order) - np.asarray(actual), 0)


def order_cost(actual, order, c_short, c_over):
    """c_short x short + c_over x left over."""
    return c_short * shortfall(actual, order) + c_over * leftover(actual, order)


def check_weeks(mondays, period):
    """Check week count and first and last Monday."""
    count, first, last = EXPECTED_WEEKS[period]
    mondays = pd.DatetimeIndex(mondays)
    require(len(mondays) == count, f"{period}: expected {count} weeks, got {len(mondays)}")
    require(mondays[0] == first and mondays[-1] == last, f"{period}: wrong first or last Monday")
    require((mondays.day_of_week == 0).all(), "every order week must start on a Monday")


def weekly_table(daily, mondays, methods=METHODS):
    """Weekly sums of actual and forecasts (full Mon to Sun weeks).

    - partial weeks dropped for every method
    """
    weekly = weekly_totals(daily, mondays, methods)
    found = pd.DatetimeIndex(sorted(weekly["week_monday"].unique()))
    require(found.equals(pd.DatetimeIndex(mondays)), "every order week must be present")
    n_items = daily["item"].nunique()
    require(len(weekly) == n_items * len(mondays), "every item needs every week")
    require(weekly[["actual"] + list(methods)].notna().all().all(), "missing forecasts")
    return weekly.sort_values(["item", "week_monday"]).reset_index(drop=True)


def simulate_method(weekly, method, buffer, c_short, c_over):
    """Per item-week: forecast, order, short, over, cost."""
    order = order_quantity(weekly[method], buffer)
    out = weekly[["item", "week_monday", "actual"]].copy()
    out["forecast"] = weekly[method].to_numpy()
    out["buffer"] = buffer
    out["order"] = order
    out["short"] = shortfall(out["actual"], order)
    out["over"] = leftover(out["actual"], order)
    out["cost"] = order_cost(out["actual"], order, c_short, c_over)
    return out


def total_cost(weekly, method, buffer, c_short, c_over):
    """Total cost of one method."""
    return int(simulate_method(weekly, method, buffer, c_short, c_over)["cost"].sum())


def assert_validation_only(weekly):
    """Stop unless all weeks are validation weeks."""
    weeks = pd.DatetimeIndex(sorted(weekly["week_monday"].unique()))
    require(weeks.isin(VAL_WEEK_MONDAYS).all(), "buffer tuning may only use validation weeks")
    require(weeks.min() >= VAL_START and (weeks.max() + pd.Timedelta(days=6)) <= VAL_END,
            "buffer tuning weeks must lie inside the validation period")
    require((weeks + pd.Timedelta(days=6)).max() < TEST_START,
            "buffer tuning must not touch the test period")


def tune_buffer(weekly, method, c_short, c_over, grid=BUFFER_GRID):
    """Cheapest buffer on the grid (ties: TIE_BREAK_RULE).

    - returns (best buffer, {buffer: cost})
    """
    assert_validation_only(weekly)
    costs = {b: total_cost(weekly, method, b, c_short, c_over) for b in grid}
    best = min(grid, key=lambda b: (costs[b], abs(b), b))
    return best, costs


def tune_all(weekly, methods=METHODS):
    """Buffer per method and ratio; returns (buffers, curves)."""
    buffers, curves = {}, {}
    for ratio, (c_short, c_over) in COST_RATIOS.items():
        buffers[ratio], curves[ratio] = {}, {}
        for m in methods:
            buffers[ratio][m], curves[ratio][m] = tune_buffer(weekly, m, c_short, c_over)
    return buffers, curves


def best_baselines(weekly, buffers):
    """Cheapest baseline per ratio (tuned and zero buffer).

    - picked on validation, frozen for the test
    """
    best = {}
    for ratio, (c_short, c_over) in COST_RATIOS.items():
        tuned = {b: total_cost(weekly, b, buffers[ratio][b], c_short, c_over) for b in BASELINES}
        zero = {b: total_cost(weekly, b, 0.0, c_short, c_over) for b in BASELINES}
        best[ratio] = {
            "tuned_buffer": min(BASELINES, key=lambda b: (tuned[b], BASELINES.index(b))),
            "zero_buffer": min(BASELINES, key=lambda b: (zero[b], BASELINES.index(b))),
        }
    return best


def leave_one_week_out(weekly, method, c_short, c_over):
    """Optimism check: tune on 11 weeks, cost the held-out week.

    - check only; frozen buffers use all 12 weeks
    """
    total = 0
    for week in sorted(weekly["week_monday"].unique()):
        held_out = weekly["week_monday"] == week
        b, _ = tune_buffer(weekly.loc[~held_out], method, c_short, c_over)
        total += total_cost(weekly.loc[held_out], method, b, c_short, c_over)
    return int(total)


def paired_bootstrap(cost_a_by_week, cost_b_by_week, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap over weeks: cost A minus B (negative = A cheaper).

    - 95% interval = middle 95% of resampled totals
    """
    a = np.asarray(cost_a_by_week, dtype=float)
    b = np.asarray(cost_b_by_week, dtype=float)
    require(a.shape == b.shape, "paired bootstrap needs the same weeks for both methods")
    diff = a - b
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    totals = diff[picks].sum(axis=1)
    return {
        "total_difference": float(diff.sum()),
        "ci95_low": float(np.percentile(totals, 2.5)),
        "ci95_high": float(np.percentile(totals, 97.5)),
        "share_of_resamples_a_cheaper": float(np.mean(totals < 0)),
        "weeks_a_cheaper": int((diff < 0).sum()),
        "weeks_b_cheaper": int((diff > 0).sum()),
        "weeks_tied": int((diff == 0).sum()),
        "n_weeks": int(len(diff)),
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


def _summary(sim):
    """Totals for simulated item-weeks."""
    return {
        "total_cost": int(sim["cost"].sum()),
        "short_units": int(sim["short"].sum()),
        "over_units": int(sim["over"].sum()),
        "item_weeks": int(len(sim)),
        "item_weeks_short": int((sim["short"] > 0).sum()),
    }


def build_tables(weekly, buffers, best_baseline, headline_model):
    """All cost tables for one period with given buffers (no tuning).

    - returns (result dict, long table)
    """
    sims = {}  # - (ratio, setting, method) -> item-weeks
    for ratio, (c_short, c_over) in COST_RATIOS.items():
        for setting in SETTINGS:
            for m in METHODS:
                b = buffers[ratio][m] if setting == "tuned_buffer" else 0.0
                sims[(ratio, setting, m)] = simulate_method(weekly, m, b, c_short, c_over)

    # - same item-weeks for every method
    first = next(iter(sims.values()))[["item", "week_monday"]]
    require(all(s[["item", "week_monday"]].equals(first) for s in sims.values()),
            "methods must be costed on identical item-weeks")

    lines, results = [], {}
    for (ratio, setting, m), sim in sims.items():
        block = results.setdefault(ratio, {}).setdefault(setting, {})
        per_item = {int(i): _summary(part) for i, part in sim.groupby("item")}
        block[m] = {"buffer": float(sim["buffer"].iloc[0]), **_summary(sim),
                    "cost_per_item": {i: s["total_cost"] for i, s in per_item.items()}}
        base = {"cost_ratio": ratio, "setting": setting, "method": m,
                "buffer": float(sim["buffer"].iloc[0])}
        lines.append({**base, "scope": "overall", **_summary(sim)})
        for i, s in per_item.items():
            lines.append({**base, "scope": f"item_{i}", **s})
    long_table = pd.DataFrame(lines)

    # - cost and % vs best baseline
    sensitivity = []
    for ratio in COST_RATIOS:
        for setting in SETTINGS:
            ref = results[ratio][setting][best_baseline[ratio][setting]]["total_cost"]
            for m in METHODS:
                r = results[ratio][setting][m]
                sensitivity.append({
                    "cost_ratio": ratio, "setting": setting, "method": m,
                    "buffer": r["buffer"], "total_cost": r["total_cost"],
                    "best_baseline": best_baseline[ratio][setting],
                    "pct_vs_best_baseline": 100 * (r["total_cost"] - ref) / ref,
                })

    # - each model vs best baseline, paired by week
    uncertainty = {}
    for ratio in COST_RATIOS:
        for setting in SETTINGS:
            bb = best_baseline[ratio][setting]
            base_sim = sims[(ratio, setting, bb)]
            base_week = base_sim.groupby("week_monday")["cost"].sum()
            for m in MODELS:
                model_sim = sims[(ratio, setting, m)]
                model_week = model_sim.groupby("week_monday")["cost"].sum()
                require(model_week.index.equals(base_week.index), "model and baseline weeks differ")
                res = paired_bootstrap(model_week.to_numpy(), base_week.to_numpy())
                item_diff = model_sim["cost"].to_numpy() - base_sim["cost"].to_numpy()
                res.update({
                    "model": m, "baseline": bb,
                    "model_total": int(model_week.sum()), "baseline_total": int(base_week.sum()),
                    "pct_difference": 100 * res["total_difference"] / float(base_week.sum()),
                    "item_weeks_model_cheaper": int((item_diff < 0).sum()),
                    "item_weeks_baseline_cheaper": int((item_diff > 0).sum()),
                    "item_weeks_tied": int((item_diff == 0).sum()),
                })
                uncertainty.setdefault(ratio, {}).setdefault(setting, {})[m] = res

    # - headline vs other model
    model_vs_model = {}
    for ratio in COST_RATIOS:
        for setting in SETTINGS:
            head_week = sims[(ratio, setting, headline_model)].groupby("week_monday")["cost"].sum()
            for m in MODELS:
                if m == headline_model:
                    continue
                other_week = sims[(ratio, setting, m)].groupby("week_monday")["cost"].sum()
                res = paired_bootstrap(head_week.to_numpy(), other_week.to_numpy())
                res.update({"model_a": headline_model, "model_b": m})
                model_vs_model.setdefault(ratio, {})[setting] = res

    # - items where a baseline beats the model (tuned buffers)
    simple_rule_wins_by_model = {}
    for model in MODELS:
        by_ratio = {}
        for ratio in COST_RATIOS:
            tuned = results[ratio]["tuned_buffer"]
            wins = []
            for i, model_cost in tuned[model]["cost_per_item"].items():
                cheaper = {b: tuned[b]["cost_per_item"][i] for b in BASELINES
                           if tuned[b]["cost_per_item"][i] < model_cost}
                if cheaper:
                    wins.append({"item": i, "model_cost": model_cost, "cheaper_baselines": cheaper})
            by_ratio[ratio] = wins
        simple_rule_wins_by_model[model] = by_ratio
    simple_rule_wins = simple_rule_wins_by_model[headline_model]

    result = {
        "n_item_weeks": int(len(weekly)),
        "n_weeks": int(weekly["week_monday"].nunique()),
        "week_mondays": [str(d.date()) for d in sorted(weekly["week_monday"].unique())],
        "items": sorted(int(i) for i in weekly["item"].unique()),
        "cost_ratios": {r: {"c_short": cs, "c_over": co} for r, (cs, co) in COST_RATIOS.items()},
        "headline_ratio": HEADLINE_RATIO,
        "headline_model": headline_model,
        "best_baseline": best_baseline,
        "results": results,
        "sensitivity": sensitivity,
        "uncertainty": uncertainty,
        "headline_vs_other_model": model_vs_model,
        "simple_rule_wins": simple_rule_wins,
        "simple_rule_wins_by_model": simple_rule_wins_by_model,
    }
    return result, long_table


def save_buffers(buffers, best_baseline, headline_model, weekly, path=BUFFERS_FILE):
    """Save tuned buffers and their rules."""
    content = {
        "frozen": True,
        "note": "Tuned on validation only. The test run reads these values and never re-tunes.",
        "buffers": buffers,
        "best_baseline": best_baseline,
        "headline_model": headline_model,
        "headline_ratio": HEADLINE_RATIO,
        "cost_ratios": {r: {"c_short": cs, "c_over": co} for r, (cs, co) in COST_RATIOS.items()},
        "grid": BUFFER_GRID,
        "rounding_rule": ROUNDING_RULE,
        "tie_break_rule": TIE_BREAK_RULE,
        "tuned_on": {
            "period": f"{VAL_START.date()} .. {VAL_END.date()} (validation)",
            "week_mondays": [str(d.date()) for d in sorted(weekly["week_monday"].unique())],
            "n_item_weeks": int(len(weekly)),
            "forecasts": "results/validation_predictions.csv (models fitted on train only)",
        },
    }
    path.write_text(json.dumps(content, indent=2))
    return content


def load_buffers(path=BUFFERS_FILE):
    """Read frozen buffers; check rules unchanged."""
    content = json.loads(Path(path).read_text())
    require(content["frozen"] is True, "buffers file is not marked frozen")
    require(content["grid"] == BUFFER_GRID, "buffer grid changed since the buffers were frozen")
    require(content["rounding_rule"] == ROUNDING_RULE, "rounding rule changed")
    saved_ratios = {r: (v["c_short"], v["c_over"]) for r, v in content["cost_ratios"].items()}
    require(saved_ratios == COST_RATIOS, "cost ratios changed since the buffers were frozen")
    for ratio in COST_RATIOS:
        require(set(content["buffers"][ratio]) == set(METHODS), "a method is missing a buffer")
        require(all(b in BUFFER_GRID for b in content["buffers"][ratio].values()),
                "a frozen buffer is not on the grid")
    require(content["tuned_on"]["week_mondays"] == [str(d.date()) for d in VAL_WEEK_MONDAYS],
            "the buffers were not tuned on the 12 validation weeks")
    return content


def simulate_test_period(test_daily, ticket):
    """Test-period cost tables with frozen buffers.

    - needs the guard's ticket; no tuning, no writes
    """
    require_active_ticket(ticket, "simulate_test_period")
    frozen = load_buffers(ticket.buffers_path)
    daily = test_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    require(daily["date"].min() >= TEST_START and daily["date"].max() <= TEST_END,
            "simulate_test_period expects test-period forecasts only")
    items = sorted(int(i) for i in daily["item"].unique())
    require(items == ticket.scope_items,
            f"test input items {items} differ from scope.json items {ticket.scope_items}")
    expected = pd.MultiIndex.from_product([ticket.scope_items, pd.date_range(TEST_START, TEST_END)])
    got = pd.MultiIndex.from_arrays([daily["item"].astype(int), daily["date"]]).sort_values()
    require(got.equals(expected), "test input must have exactly one row per item and test day")
    check_weeks(TEST_WEEK_MONDAYS, "test")
    weekly = weekly_table(daily, TEST_WEEK_MONDAYS)
    result, long_table = build_tables(
        weekly, frozen["buffers"], frozen["best_baseline"], frozen["headline_model"])
    result["buffers_source"] = "results/buffers.json (frozen on validation)"
    return result, long_table
