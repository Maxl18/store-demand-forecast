"""Order simulation tests (plain Python).

- run: .venv\\Scripts\\python.exe src\\test_simulation.py
- rounding, costs, buffer search: small hand-worked examples
- test-period function: made-up data in a temp folder only
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

if not __debug__:
    raise SystemExit("Run the tests without -O: python -O would switch off their checks.")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import guard  # noqa: E402
import headline  # noqa: E402
import simulate  # noqa: E402
from forecast import METHODS, RESULTS, TEST_WEEK_MONDAYS, VAL_WEEK_MONDAYS  # noqa: E402
from run_simulation_validation import load_validation_predictions  # noqa: E402
from simulate import (  # noqa: E402
    BUFFER_GRID,
    COST_RATIOS,
    order_cost,
    order_quantity,
    paired_bootstrap,
    round_order,
    simulate_method,
    tune_buffer,
    weekly_table,
)

PASSED = []


def check(name, condition):
    assert condition, f"FAILED: {name}"
    PASSED.append(name)
    print(f"  ok  {name}")


def test_rounding_and_order():
    print("1. Rounding and order quantity")
    check("100.4 rounds to 100", round_order(100.4) == 100)
    check("100.5 rounds up to 101 (half up)", round_order(100.5) == 101)
    check("100.6 rounds to 101", round_order(100.6) == 101)
    # - exact sum 175.5, float sum 175.49999999999997: half up -> 176
    seven = [22.7265, 31.1963, 31.7219, 19.0705, 29.2478, 38.7728, 2.7642]
    check("float noise does not turn an exact half down (175.5 -> 176)",
          order_quantity(np.sum(seven), 0.0) == 176)
    check("forecast 100, buffer 0 -> order 100", order_quantity(100, 0.0) == 100)
    check("forecast 100, buffer +10% -> 110", order_quantity(100, 0.10) == 110)
    check("forecast 100, buffer +15% -> 115 (100*1.15 is 114.999... in float)",
          order_quantity(100, 0.15) == 115)
    check("forecast 99, buffer +5% -> 103.95 -> 104", order_quantity(99, 0.05) == 104)
    check("forecast 110, buffer +5% -> 115.5 -> 116", order_quantity(110, 0.05) == 116)
    check("forecast 200, buffer -10% -> 180", order_quantity(200, -0.10) == 180)
    check("forecast 0 -> order 0", order_quantity(0, 0.20) == 0)
    # - 7 x 10.2 = 71.4 -> 71; +10%: 78.54 -> 79
    check("7 x 10.2 = 71.4 -> 71", order_quantity(np.sum([10.2] * 7), 0.0) == 71)
    check("71.4 x 1.10 = 78.54 -> 79", order_quantity(71.4, 0.10) == 79)
    check("works on arrays", list(order_quantity([100, 99, 0], 0.10)) == [110, 109, 0])


def test_cost():
    print("2. Cost of one order")
    # - actual 100, order 90: 10 short
    check("3:1, 10 short -> 30", order_cost(100, 90, 3, 1) == 30)
    check("5:1, 10 short -> 50", order_cost(100, 90, 5, 1) == 50)
    check("1:1, 10 short -> 10", order_cost(100, 90, 1, 1) == 10)
    # - actual 100, order 110: 10 over
    check("3:1, 10 over -> 10", order_cost(100, 110, 3, 1) == 10)
    check("5:1, 10 over -> 10", order_cost(100, 110, 5, 1) == 10)
    check("exact order costs 0", order_cost(100, 100, 3, 1) == 0)
    check("works on arrays", list(order_cost([100, 100, 100], [90, 100, 110], 3, 1)) == [30, 0, 10])

    # - made-up: 2 items x 2 validation weeks
    days = pd.date_range("2017-07-03", "2017-07-16")  # - Mon 07-03 to Sun 07-16
    rows = []
    for item, actual, fc in [(1, 10, 9.5), (2, 20, 21.0)]:
        for d in days:
            rows.append({"date": d, "item": item, "actual": actual, "m": fc})
    daily = pd.DataFrame(rows)
    weekly = weekly_table(daily, VAL_WEEK_MONDAYS[:2], methods=["m"])
    # - item 1: actual 70, fc 66.5; +10% = 73.15 -> 73, 3 over -> 3
    # - item 2: actual 140, fc 147; +10% = 161.7 -> 162, 22 over -> 22
    sim = simulate_method(weekly, "m", 0.10, 3, 1)
    check("tiny table: weekly actual totals 70 and 140",
          list(sim["actual"]) == [70, 70, 140, 140])
    check("tiny table: orders 73 and 162", list(sim["order"]) == [73, 73, 162, 162])
    check("tiny table: total cost 2 x (3 + 22) = 50", sim["cost"].sum() == 50)
    # - buffer 0: 66.5 -> 67, 3 short -> 9; 147, 7 over -> 7
    sim0 = simulate_method(weekly, "m", 0.0, 3, 1)
    check("tiny table, buffer 0: orders 67 and 147", list(sim0["order"]) == [67, 67, 147, 147])
    check("tiny table, buffer 0: total cost 2 x (9 + 7) = 32", sim0["cost"].sum() == 32)


def test_real_item_week():
    print("3. One real validation item-week, worked out by hand")
    daily = load_validation_predictions()
    weekly = weekly_table(daily, VAL_WEEK_MONDAYS)
    row = weekly[(weekly["item"] == 5) & (weekly["week_monday"] == "2017-07-03")]
    one = row.reset_index(drop=True)
    # - item 5, week of 2017-07-03, validation_predictions.csv:
    #   actual : 18 + 23 + 22 + 26 + 21 + 25 + 23 = 158
    #   lag7   : 16 + 16 + 14 + 23 + 22 + 26 + 26 = 143 -> 15 short -> 45 at 3:1
    #   mean4  : 20.25 + 23 + 18.25 + 22.25 + 25.75 + 24 + 23.75 = 157.25 -> 157 -> 1 short -> 3
    #   lag364 : 20 + 21 + 25 + 27 + 21 + 24 + 31 = 169 -> 11 over -> 11
    #   ridge  : 20.0293 + 24.0282 + 22.9375 + 26.4155 + 27.8355 + 29.4869 + 32.0651
    #            = 182.798 -> 183 -> 25 over -> 25
    #   hgb    : 164.6695 -> 165 -> 7 over -> 7
    expected = {"baseline_lag7": (143, 45), "baseline_mean4": (157, 3),
                "baseline_lag364": (169, 11), "ridge": (183, 25), "hgb": (165, 7)}
    check("item 5 week 07-03: actual 158", int(one["actual"].iloc[0]) == 158)
    for m, (order, cost) in expected.items():
        s = simulate_method(one, m, 0.0, 3, 1)
        check(f"item 5 week 07-03, {m}, buffer 0: order {order}, cost {cost}",
              s["order"].iloc[0] == order and s["cost"].iloc[0] == cost)
    # - +10%: lag7 157.3 -> 157, 1 short -> 3; ridge 201.08 -> 201, 43 over -> 43
    check("lag7 +10%: order 157, cost 3",
          simulate_method(one, "baseline_lag7", 0.10, 3, 1)["cost"].iloc[0] == 3)
    check("ridge +10%: order 201, cost 43",
          simulate_method(one, "ridge", 0.10, 3, 1)[["order", "cost"]].iloc[0].tolist() == [201, 43])


def test_tuning():
    print("4. Buffer search")
    mondays = VAL_WEEK_MONDAYS[:2]
    weekly = pd.DataFrame({"item": [1, 1], "week_monday": mondays,
                           "actual": [110, 110], "m": [100.0, 100.0]})
    best, costs = tune_buffer(weekly, "m", 3, 1, grid=[0.0, 0.05, 0.10, 0.15])
    # - 0: 10 short x 3 x 2 = 60; 0.05: 30; 0.10: exact, 0; 0.15: 5 over, 10
    check("costs per buffer 60, 30, 0, 10", [costs[b] for b in [0.0, 0.05, 0.10, 0.15]] == [60, 30, 0, 10])
    check("picks +10%", best == 0.10)
    # - ties (1:1): 5 short, 5 over; -5%, 0, +5% all cost 10
    tie = pd.DataFrame({"item": [1, 1], "week_monday": mondays,
                        "actual": [105, 95], "m": [100.0, 100.0]})
    best, costs = tune_buffer(tie, "m", 1, 1, grid=[-0.05, 0.0, 0.05])
    check("tie: all three cost 10", set(costs.values()) == {10})
    check("tie broken towards 0", best == 0.0)
    best, _ = tune_buffer(tie, "m", 1, 1, grid=[0.05, -0.05])
    check("tie between -5% and +5% broken towards the smaller", best == -0.05)
    # - real search = brute force over the grid
    daily = load_validation_predictions()
    real = weekly_table(daily, VAL_WEEK_MONDAYS)
    for m in ["baseline_lag364", "ridge"]:
        best, costs = tune_buffer(real, m, 3, 1)
        brute = min(costs.values())
        check(f"{m}: chosen buffer has the minimum cost of the grid", costs[best] == brute)
    check("grid is -20% .. +40% in 1-point steps (61 values)",
          len(BUFFER_GRID) == 61 and BUFFER_GRID[0] == -0.2 and BUFFER_GRID[-1] == 0.4)
    check("default cost ratio is 3:1", COST_RATIOS[simulate.HEADLINE_RATIO] == (3, 1))


def test_weeks():
    print("5. Week counts")
    check("12 validation weeks, Mondays 2017-07-03 .. 2017-09-18",
          len(VAL_WEEK_MONDAYS) == 12 and str(VAL_WEEK_MONDAYS[0].date()) == "2017-07-03"
          and str(VAL_WEEK_MONDAYS[-1].date()) == "2017-09-18")
    check("13 test weeks, Mondays 2017-10-02 .. 2017-12-25",
          len(TEST_WEEK_MONDAYS) == 13 and str(TEST_WEEK_MONDAYS[0].date()) == "2017-10-02"
          and str(TEST_WEEK_MONDAYS[-1].date()) == "2017-12-25")
    simulate.check_weeks(VAL_WEEK_MONDAYS, "validation")
    simulate.check_weeks(TEST_WEEK_MONDAYS, "test")
    try:
        simulate.check_weeks(VAL_WEEK_MONDAYS[:11], "validation")
        wrong = False
    except AssertionError:
        wrong = True
    check("check_weeks catches a missing week", wrong)

    daily = load_validation_predictions()
    weekly = weekly_table(daily, VAL_WEEK_MONDAYS)
    check("validation: 96 item-weeks (12 x 8)", len(weekly) == 96)
    check("partial days dropped: 84 of 92 days used per item",
          weekly["actual"].sum() == daily.loc[(daily["date"] >= "2017-07-03")
                                              & (daily["date"] <= "2017-09-24"), "actual"].sum())
    # - partial days (07-01, 07-02, 09-25 to 09-30) change nothing
    edited = daily.copy()
    partial = (edited["date"] < "2017-07-03") | (edited["date"] > "2017-09-24")
    edited.loc[partial, METHODS + ["actual"]] += 1000
    check("partial-week days do not enter any order", weekly_table(edited, VAL_WEEK_MONDAYS).equals(weekly))


def test_validation_only():
    print("6. Buffers are chosen from validation weeks only")
    bad = pd.DataFrame({"item": [1, 1], "week_monday": pd.to_datetime(["2017-09-18", "2017-10-02"]),
                        "actual": [100, 100], "m": [100.0, 100.0]})
    try:
        tune_buffer(bad, "m", 3, 1)
        refused = False
    except AssertionError:
        refused = True
    check("tuning refuses a table that contains a test-period week", refused)
    train_week = bad.assign(week_monday=pd.to_datetime(["2017-06-26", "2017-07-03"]))
    try:
        tune_buffer(train_week, "m", 3, 1)
        refused = False
    except AssertionError:
        refused = True
    check("tuning refuses the week that straddles train and validation", refused)

    path = RESULTS / "buffers.json"
    if path.exists():
        frozen = simulate.load_buffers(path)
        check("buffers.json: tuned on the 12 validation Mondays",
              frozen["tuned_on"]["week_mondays"] == [str(d.date()) for d in VAL_WEEK_MONDAYS])
        check("buffers.json: 96 item-weeks", frozen["tuned_on"]["n_item_weeks"] == 96)
        check("buffers.json: every buffer is inside the grid",
              all(b in BUFFER_GRID for r in frozen["buffers"].values() for b in r.values()))
        # - re-tuning gives the frozen values
        daily = load_validation_predictions()
        buffers, _ = simulate.tune_all(weekly_table(daily, VAL_WEEK_MONDAYS))
        check("buffers.json equals a fresh tuning on validation", buffers == frozen["buffers"])
    else:
        print("  (results/buffers.json not found: run run_simulation_validation.py first)")


def fake_test_daily(seed=0):
    """Made-up test-period forecasts (no real test data)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2017-10-01", "2017-12-31")
    rows = []
    for item in [5, 15, 16, 20, 25, 31, 33, 42]:
        level = rng.uniform(20, 100)
        for d in dates:
            actual = int(rng.poisson(level))
            rows.append({"date": d, "item": item, "actual": actual,
                         **{m: level * rng.uniform(0.8, 1.2) for m in METHODS}})
    return pd.DataFrame(rows)


def _refuses(function, *args):
    """True if the call fails a check."""
    try:
        function(*args)
    except AssertionError:
        return True
    return False


def test_test_period_function():
    print("7. Test-period function uses the frozen buffers (made-up data only)")
    fake = fake_test_daily()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # - known distinct buffers, real scope file
        buffers = {r: {m: BUFFER_GRID[10 + 5 * i + j] for i, m in enumerate(METHODS)}
                   for j, r in enumerate(COST_RATIOS)}
        best = {r: {"tuned_buffer": "baseline_lag364", "zero_buffer": "baseline_mean4"} for r in COST_RATIOS}
        val_weekly = pd.DataFrame({"week_monday": VAL_WEEK_MONDAYS.repeat(8), "item": 0})
        path = tmp / "buffers.json"
        simulate.save_buffers(buffers, best, "hgb", val_weekly, path=path)
        shutil.copy(RESULTS / "scope.json", tmp / "scope.json")

        # - no real ticket: refuses
        check("test function refuses to run without a ticket",
              _refuses(simulate.simulate_test_period, fake, None))
        check("a ticket cannot be made without a 'started' line in the run log",
              _refuses(guard.FinalRunTicket, tmp, 1))

        # - synthetic run in temp folder: log "started", get ticket
        guard.append_log(tmp, {"event": "started", "run": 1, "note": "synthetic check in test_simulation"})
        ticket = guard.FinalRunTicket(tmp, 1)

        # - any tuning call stops the run
        real_tune = simulate.tune_buffer
        simulate.tune_buffer = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("tuning called"))
        try:
            result, long_table = simulate.simulate_test_period(fake, ticket)
        finally:
            simulate.tune_buffer = real_tune

        used = {r: {m: result["results"][r]["tuned_buffer"][m]["buffer"] for m in METHODS}
                for r in COST_RATIOS}
        check("test function uses exactly the frozen buffers", used == buffers)
        check("test function: zero-buffer rows use buffer 0",
              all(result["results"][r]["zero_buffer"][m]["buffer"] == 0.0
                  for r in COST_RATIOS for m in METHODS))
        check("test function: 13 weeks, 104 item-weeks",
              result["n_weeks"] == 13 and result["n_item_weeks"] == 104)
        check("test function: best baseline taken from the frozen file", result["best_baseline"] == best)
        check("test function: long table has 3 ratios x 2 settings x 5 methods x 9 scopes",
              len(long_table) == 3 * 2 * 5 * 9)

        # - refuses: non-test dates, missing or extra item, changed rules
        val_like = fake.assign(date=fake["date"] - pd.Timedelta(days=92))
        check("test function refuses non-test dates", _refuses(simulate.simulate_test_period, val_like, ticket))
        missing = fake[fake["item"] != 42]
        check("test function refuses input with an item missing",
              _refuses(simulate.simulate_test_period, missing, ticket))
        extra = pd.concat([fake, fake[fake["item"] == 42].assign(item=99)])
        check("test function refuses input with an extra item",
              _refuses(simulate.simulate_test_period, extra, ticket))
        original = path.read_text()
        changed = json.loads(original)
        changed["cost_ratios"]["3:1"]["c_short"] = 4
        path.write_text(json.dumps(changed))
        check("test function refuses a buffers file with different costs",
              _refuses(simulate.simulate_test_period, fake, ticket))
        path.write_text(original)
        check("with the original buffers file it runs again", simulate.simulate_test_period(fake, ticket))

        # - closed run: ticket dead
        guard.append_log(tmp, {"event": "finished", "run": 1})
        check("a ticket stops working once its run is closed in the log",
              _refuses(simulate.simulate_test_period, fake, ticket))


def test_headline_consistency():
    print("9. One headline model everywhere")
    head = headline.check_headline_consistency()
    check(f"every results file names the headline model of headline_model.json ({head})", True)
    check("headline is gradient boosting (the recorded decision)", head == "hgb")
    record = headline.load_headline_record()
    check("Ridge is kept as the second model", record["other_model"] == "ridge")
    check("the order-cost rule still picks the headline on validation",
          min(("ridge", "hgb"), key=lambda m: record["validation_numbers"]["order_cost_tuned_buffer"]["3:1"][m])
          == head)


def test_bootstrap():
    print("8. Paired bootstrap")
    a = [10, 20, 30, 40]
    same = paired_bootstrap(a, a)
    check("identical costs: difference 0, interval [0, 0], all weeks tied",
          same["total_difference"] == 0 and same["ci95_low"] == 0 and same["ci95_high"] == 0
          and same["weeks_tied"] == 4)
    b = [12, 25, 29, 44]  # - diffs -2, -5, +1, -4: total -10, A cheaper 3 weeks
    r1, r2 = paired_bootstrap(a, b), paired_bootstrap(a, b)
    check("total difference -10, A cheaper in 3 of 4 weeks",
          r1["total_difference"] == -10 and r1["weeks_a_cheaper"] == 3 and r1["weeks_b_cheaper"] == 1)
    check("fixed seed: same interval twice", r1 == r2)
    check("interval inside the possible range [-20, +4]", -20 <= r1["ci95_low"] <= r1["ci95_high"] <= 4)


FINAL_RUN_FILES = ["final_test_log.json", "test_predictions.csv", "test_metrics.json",
                   "simulation_test.json", "simulation_test.csv"]


def results_snapshot():
    """Hashes of the real run log and test outputs."""
    return {name: guard.sha256_file(RESULTS / name) if (RESULTS / name).exists() else None
            for name in FINAL_RUN_FILES}


if __name__ == "__main__":
    before = results_snapshot()
    test_rounding_and_order()
    test_cost()
    test_real_item_week()
    test_tuning()
    test_weeks()
    test_validation_only()
    test_test_period_function()
    test_bootstrap()
    test_headline_consistency()
    check("these tests did not touch the real run log or any real test output",
          results_snapshot() == before)
    print(f"\nALL {len(PASSED)} CHECKS PASSED")
