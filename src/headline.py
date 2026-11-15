"""Headline model record (results/headline_model.json): read, build, check.

- headline = boosting ("hgb"), lowest validation order cost at 3:1, before any test data
- earlier rule (lowest daily MAE) picked Ridge; changed after seeing validation results
- Ridge always reported next to it
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import BASELINES, METHODS, MODELS, RESULTS  # noqa: E402
from guard import require  # noqa: E402

HEADLINE_FILE_NAME = "headline_model.json"

RULE = ("lowest total order cost on the 96 validation item-weeks at the default cost ratio 3:1 "
        "(short 3 : over 1), each model with its own buffer tuned on validation")
EARLIER_RULE = ("lowest overall daily MAE on validation (fixed in advance in task 03); "
                "it picked ridge (6.516 vs hgb 6.630)")
DISCLOSURE = [
    "Ridge is marginally ahead on daily MAE on validation (6.52 vs 6.63 units per day).",
    "The selection rule was changed from daily MAE to order cost after seeing the validation "
    "results. Only validation data was used for this; the test period (2017-10-01 onward) was "
    "not read, predicted or scored when the decision was made.",
    "Both models are reported side by side everywhere; Ridge is never dropped from a table.",
    "The validation order costs (including hgb minus Ridge, -294) are in-sample: the buffers were "
    "tuned and scored on the same 12 weeks, so they are optimistic. The test run is the honest "
    "comparison.",
]


def headline_path(results_dir=RESULTS):
    return Path(results_dir) / HEADLINE_FILE_NAME


def load_headline_record(results_dir=RESULTS):
    record = json.loads(headline_path(results_dir).read_text())
    require(record.get("frozen") is True, "headline_model.json must be marked frozen")
    require(record.get("headline_model") in MODELS,
            f"headline_model.json: unknown model {record.get('headline_model')!r}")
    require(record.get("other_model") in MODELS and record["other_model"] != record["headline_model"],
            "headline_model.json: other_model must be the second model")
    return record


def load_headline_model(results_dir=RESULTS):
    return load_headline_record(results_dir)["headline_model"]


def _model_vs_model_cost(results_dir, model_a, model_b, ratio="3:1"):
    """Validation cost model_a minus model_b, bootstrap by week."""
    import pandas as pd  # - local imports keep the checks light
    from forecast import VAL_WEEK_MONDAYS
    from simulate import COST_RATIOS, paired_bootstrap, simulate_method, weekly_table

    daily = pd.read_csv(Path(results_dir) / "validation_predictions.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    weekly = weekly_table(daily, VAL_WEEK_MONDAYS)
    buffers = json.loads((Path(results_dir) / "buffers.json").read_text())["buffers"]
    c_short, c_over = COST_RATIOS[ratio]
    week = {m: simulate_method(weekly, m, buffers[ratio][m], c_short, c_over)
            .groupby("week_monday")["cost"].sum() for m in (model_a, model_b)}
    res = paired_bootstrap(week[model_a].to_numpy(), week[model_b].to_numpy())
    res.update({"model_a": model_a, "model_b": model_b})
    return res


def build_headline_record(results_dir=RESULTS, headline="hgb", decided_on="2026-11-14"):
    """Build the record from validation result files only."""
    results_dir = Path(results_dir)
    val = json.loads((results_dir / "validation_metrics.json").read_text())
    sim = json.loads((results_dir / "simulation_validation.json").read_text())
    buffers = json.loads((results_dir / "buffers.json").read_text())
    other = [m for m in MODELS if m != headline][0]

    tuned_cost = {r: {m: sim["results"][r]["tuned_buffer"][m]["total_cost"] for m in METHODS}
                  for r in sim["results"]}
    zero_cost = {r: {m: sim["results"][r]["zero_buffer"][m]["total_cost"] for m in METHODS}
                 for r in sim["results"]}
    rule_pick = min(MODELS, key=lambda m: tuned_cost["3:1"][m])
    require(rule_pick == headline, f"the order-cost rule picks {rule_pick}, not {headline}")

    overall = val["overall"]
    best_by_metric = {k: min(BASELINES, key=lambda b: (overall[b][k], BASELINES.index(b)))
                      for k in ("mae", "smape", "weekly_mae")}

    return {
        "frozen": True,
        "headline_model": headline,
        "headline_model_name": "gradient boosting (sklearn HistGradientBoostingRegressor)",
        "other_model": other,
        "other_model_note": "reported next to the headline model in every table",
        "rule": RULE,
        "decided_by": "user decision D8 in PLAN.md, before any test-period data was read",
        "decided_on": decided_on,
        "earlier_rule_replaced": EARLIER_RULE,
        "disclosure": DISCLOSURE,
        "validation_numbers": {
            "source": "results/validation_metrics.json and results/simulation_validation.json",
            "order_cost_tuned_buffer": tuned_cost,
            "order_cost_zero_buffer": zero_cost,
            "tuned_buffers": buffers["buffers"],
            "daily_mae": {m: overall[m]["mae"] for m in METHODS},
            "smape": {m: overall[m]["smape"] for m in METHODS},
            "weekly_mae": {m: overall[m]["weekly_mae"] for m in METHODS},
            "headline_minus_other_model_cost_3to1_tuned": _model_vs_model_cost(
                results_dir, headline, other),
        },
        "comparison_baselines_fixed_from_validation": {
            "note": ("Fixed on validation before the test run, so the test tables cannot pick the "
                     "comparison after seeing test results. All baselines are still reported."),
            "order_cost": buffers["best_baseline"],
            "daily_mae": best_by_metric["mae"],
            "smape": best_by_metric["smape"],
            "weekly_mae": best_by_metric["weekly_mae"],
        },
        "headline_ratio": "3:1",
    }


def write_headline_record(results_dir=RESULTS, **kwargs):
    record = build_headline_record(results_dir, **kwargs)
    headline_path(results_dir).write_text(json.dumps(record, indent=2))
    return record


def _find_key(obj, key, where=""):
    """Yield (location, value) for each `key` in nested JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{where}.{k}" if where else k
            if k == key:
                yield here, v
            yield from _find_key(v, key, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _find_key(v, key, f"{where}[{i}]")


def _same(a, b):
    """Equal, floats within 1e-9."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
    return a == b


def headline_problems(results_dir=RESULTS):
    """List of problems ([] = all agree)."""
    results_dir = Path(results_dir)
    record = load_headline_record(results_dir)
    head = record["headline_model"]
    problems = []

    # - every results file must name the same headline model
    for name in ["buffers.json", "validation_metrics.json", "simulation_validation.json",
                 "failure_validation_summary.json"]:
        path = results_dir / name
        if not path.exists():
            continue
        content = json.loads(path.read_text())
        for where, value in _find_key(content, "headline_model"):
            if value != head:
                problems.append(f"{name} {where} = {value!r}, but headline_model.json says {head!r}")
        for where, value in _find_key(content, "chosen_model"):
            if value != head:
                problems.append(f"{name} {where} = {value!r} (old field), but the headline is {head!r}")
    sim_path = results_dir / "simulation_validation.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())
        for ratio, by_setting in sim.get("headline_vs_other_model", {}).items():
            for setting, res in by_setting.items():
                if res.get("model_a") != head:
                    problems.append(f"simulation_validation.json headline_vs_other_model {ratio} "
                                    f"{setting} compares {res.get('model_a')!r} first, not {head!r}")

    # - record must match validation results; rule must pick the same model
    try:
        fresh = build_headline_record(results_dir, headline=head, decided_on=record["decided_on"])
    except Exception as error:  # - rule picks another model, or a file is missing
        problems.append(f"cannot rebuild the headline record from the validation results: {error}")
    else:
        for key in ["validation_numbers", "comparison_baselines_fixed_from_validation", "rule",
                    "other_model", "headline_ratio"]:
            if not _same(fresh[key], record[key]):
                problems.append(f"headline_model.json field {key!r} no longer matches the "
                                f"validation results files")
    return problems


def check_headline_consistency(results_dir=RESULTS):
    """Stop if any file disagrees on the headline model."""
    problems = headline_problems(results_dir)
    require(not problems, "Headline model check failed:\n  - " + "\n  - ".join(problems))
    return load_headline_model(results_dir)
