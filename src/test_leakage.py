"""Leakage tests (plain Python); ends with "ALL CHECKS PASSED".

- run: .venv\\Scripts\\python.exe src\\test_leakage.py
- change sales from a target Monday on: that week's features and forecasts must not move
- negative controls: must catch leaky features
- test period never loaded; first test week checked on made-up sales
"""

import inspect
import sys
from pathlib import Path

if not __debug__:
    raise SystemExit("Run the tests without -O: python -O would switch off their checks.")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from features import FEATURES, LAG_DAYS, MIN_LAG, has_all_features, make_features  # noqa: E402
from forecast import (  # noqa: E402
    METHODS,
    TEST_END,
    TEST_START,
    TEST_WEEK_MONDAYS,
    TRAIN_END,
    VAL_END,
    VAL_START,
    VAL_WEEK_MONDAYS,
    fit_model,
    forecast_period,
    load_scope,
    score,
    training_rows,
)
from run_validation import build_validation_predictions  # noqa: E402

DAY = pd.Timedelta(days=1)

# - weeks across training, both split edges and validation
TEST_MONDAYS = pd.to_datetime([
    "2014-01-06", "2015-06-15", "2016-12-26",
    "2017-06-26", "2017-07-03", "2017-08-14", "2017-09-18", "2017-09-25",
])


def ok(message):
    print(f"PASS  {message}")


def week_part(feats, monday, columns):
    """Features of one Mon to Sun week, fixed order."""
    keep = (feats["date"] >= monday) & (feats["date"] <= monday + 6 * DAY)
    part = feats.loc[keep, ["item", "date"] + [c for c in columns if c not in ("item", "date")]]
    return part.sort_values(["item", "date"]).reset_index(drop=True)


def change_future(df, monday, how, seed=0):
    """Copy with all sales from `monday` on changed."""
    out = df.copy()
    future = out["date"] >= monday
    out["sales"] = out["sales"].astype(float)
    if how == "plus1000":
        out.loc[future, "sales"] += 1000
    elif how == "shuffle":
        rng = np.random.default_rng(seed)
        out.loc[future, "sales"] = rng.permutation(out.loc[future, "sales"].to_numpy())
    elif how == "delete":
        out.loc[future, "sales"] = np.nan
    return out


def check_splits():
    assert TRAIN_END == pd.Timestamp("2017-06-30")
    assert VAL_START == TRAIN_END + DAY and VAL_END == pd.Timestamp("2017-09-30")
    assert TEST_START == VAL_END + DAY and TEST_END == pd.Timestamp("2017-12-31")
    names = {d: d.day_name() for d in [TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END]}
    assert [names[d] for d in names] == ["Friday", "Saturday", "Saturday", "Sunday", "Sunday"]
    ok(f"split dates and weekdays: { {str(d.date()): n for d, n in names.items()} }")

    for mondays, start, end, count, first, last in [
        (VAL_WEEK_MONDAYS, VAL_START, VAL_END, 12, "2017-07-03", "2017-09-18"),
        (TEST_WEEK_MONDAYS, TEST_START, TEST_END, 13, "2017-10-02", "2017-12-25"),
    ]:
        # - Mondays of full weeks in the period
        expected = [d for d in pd.date_range(start, end) if d.day_of_week == 0 and d + 6 * DAY <= end]
        assert list(mondays) == expected and len(mondays) == count
        assert str(mondays[0].date()) == first and str(mondays[-1].date()) == last
        ok(f"{count} full weeks {first} .. {(mondays[-1] + 6 * DAY).date()}")

    # - past 2017-09-30 without the guard's ticket: must refuse before reading
    try:
        load_scope(TEST_END)
        refused = False
    except AssertionError as error:  # - CheckFailed is an AssertionError
        refused = "test period" in str(error)
    assert refused, "load_scope read the test period without a ticket"
    ok("load_scope refuses any cut-off after 2017-09-30 without the guarded run's ticket")

    df, _ = load_scope(VAL_END)
    assert df["date"].max() == VAL_END, "validation data must stop at 2017-09-30"
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)]
    assert train["date"].max() < val["date"].min()
    ok(f"loaded data {df['date'].min().date()} .. {df['date'].max().date()} (no test rows); "
       f"train {train['date'].min().date()} .. {train['date'].max().date()}, "
       f"validation {val['date'].min().date()} .. {val['date'].max().date()}")
    return df


# - changing the target week and later must not change its features
def check_future_invariance(df, feature_fn=make_features, columns=FEATURES, mondays=TEST_MONDAYS):
    base = feature_fn(df)
    for monday in mondays:
        before = week_part(base, monday, columns)
        # - week of 2017-09-25: 6 days (Sunday is in the test period)
        n_days = 6 if monday == pd.Timestamp("2017-09-25") else 7
        assert len(before) == n_days * df["item"].nunique()
        assert sorted(before["date"].dt.day_of_week.unique()) == list(range(n_days))
        for how in ["plus1000", "shuffle", "delete"]:
            after = week_part(feature_fn(change_future(df, monday, how)), monday, columns)
            pd.testing.assert_frame_equal(before, after, check_dtype=False, check_exact=True,
                                          obj=f"features of week {monday.date()} after '{how}'")
        # - control: a change in the week before must show
        control = df.copy()
        last_week = (control["date"] >= monday - 7 * DAY) & (control["date"] < monday)
        control.loc[last_week, "sales"] += 1000
        after = week_part(feature_fn(control), monday, columns)
        assert not before.equals(after), "control failed: features ignore the week before Monday"
    ok(f"features unchanged for {len(mondays)} weeks (all 7 weekdays) when sales from Monday "
       f"on are +1000, shuffled or deleted; a change in the week before Monday is detected")


def latest_date_read(df_one_item, monday, columns, look_back=400):
    """Per feature: latest date whose sales change that week's value.

    - changes one date at a time, Sunday back to `look_back` days
    """
    base = week_part(make_features(df_one_item), monday, columns)
    latest = {c: None for c in columns}
    for day in pd.date_range(monday - look_back * DAY, monday + 6 * DAY)[::-1]:
        changed = df_one_item.copy()
        changed.loc[changed["date"] == day, "sales"] += 1000
        after = week_part(make_features(changed), monday, columns)
        for c in columns:
            if latest[c] is None and not np.array_equal(base[c].to_numpy(), after[c].to_numpy(),
                                                        equal_nan=True):
                latest[c] = day
    return latest


def check_latest_dates(df):
    item = df["item"].iloc[0]
    one_item = df[df["item"] == item].reset_index(drop=True)
    monday = VAL_WEEK_MONDAYS[0]
    # - made-up sales: check the first test week without real test data
    rng = np.random.default_rng(1)
    fake_dates = pd.date_range("2015-01-01", TEST_END)
    fake = pd.DataFrame({"date": fake_dates, "item": 1,
                         "sales": rng.poisson(50, len(fake_dates)).astype(float)})
    for label, data, mon in [("first validation week, real data", one_item, monday),
                             ("first test week, made-up data", fake, TEST_WEEK_MONDAYS[0])]:
        latest = latest_date_read(data, mon, FEATURES)
        print(f"      latest sales date read by each feature, {label} (Monday {mon.date()}):")
        for c in FEATURES:
            shown = "no sales used" if latest[c] is None else f"{latest[c].date()} (Monday - {(mon - latest[c]).days} days)"
            print(f"        {c:22s} {shown}")
            assert latest[c] is None or latest[c] < mon, f"{c} reads sales on or after Monday"
        assert latest["lag_7"] == mon - DAY  # - Sunday's lag_7 reads the Sunday before
    ok("every feature reads only sales dated before the target Monday")


def check_items_separate(df):
    items = sorted(df["item"].unique())
    base = make_features(df)
    changed = df.copy()
    changed.loc[changed["item"] == items[0], "sales"] += 1000
    after = make_features(changed)
    others = base["item"] != items[0]
    pd.testing.assert_frame_equal(base.loc[others, FEATURES], after.loc[others, FEATURES])
    ok(f"changing item {items[0]}'s sales leaves the other {len(items) - 1} items' features unchanged")


# - full pipeline forecasts must not see the target week or later
def check_forecast_invariance(df):
    base, _ = forecast_period(make_features(df), TRAIN_END, VAL_START, VAL_END)
    for monday in [VAL_WEEK_MONDAYS[0], VAL_WEEK_MONDAYS[6], VAL_WEEK_MONDAYS[-1]]:
        in_week = (base["date"] >= monday) & (base["date"] <= monday + 6 * DAY)
        for how in ["plus1000", "shuffle"]:
            new, _ = forecast_period(make_features(change_future(df, monday, how)),
                                     TRAIN_END, VAL_START, VAL_END)
            pd.testing.assert_frame_equal(base.loc[in_week, METHODS], new.loc[in_week, METHODS],
                                          check_exact=True, obj=f"forecasts week {monday.date()} '{how}'")
    ok("forecasts of all 5 methods for 3 validation weeks are identical when sales from "
       "Monday on are +1000 or shuffled (models refitted each time)")


def check_fit_on_train_only(df):
    feats = make_features(df)
    train = training_rows(feats, TRAIN_END)
    assert train["date"].max() == TRAIN_END
    assert train["date"].min() == df["date"].min() + 364 * DAY, "first training row needs 364 days of history"
    assert has_all_features(train).all()

    # - full, validation removed, validation +1000: fits must be identical
    no_val = make_features(df[df["date"] <= TRAIN_END])
    bumped = make_features(change_future(df, VAL_START, "plus1000"))
    ridge = [fit_model("ridge", f, TRAIN_END) for f in (feats, no_val, bumped)]
    hgb = [fit_model("hgb", f, TRAIN_END) for f in (feats, no_val, bumped)]

    scaler = ridge[0].named_steps["prep"].named_transformers_["scale"]
    scaled_cols = list(ridge[0].named_steps["prep"].transformers_[1][2])
    # - scaler means = training-row means
    assert np.allclose(scaler.mean_, train[scaled_cols].mean().to_numpy(), rtol=0, atol=1e-9)
    assert scaler.n_samples_seen_ == len(train)
    for other in ridge[1:]:
        s = other.named_steps["prep"].named_transformers_["scale"]
        enc0 = ridge[0].named_steps["prep"].named_transformers_["onehot"]
        enc = other.named_steps["prep"].named_transformers_["onehot"]
        assert np.array_equal(s.mean_, scaler.mean_) and np.array_equal(s.scale_, scaler.scale_)
        assert all(np.array_equal(a, b) for a, b in zip(enc.categories_, enc0.categories_))
        assert np.array_equal(other.named_steps["ridge"].coef_, ridge[0].named_steps["ridge"].coef_)
    probe = feats[(feats["date"] >= VAL_START)][FEATURES]
    for other in hgb[1:]:
        assert np.array_equal(other.predict(probe), hgb[0].predict(probe))
    ok(f"scaler fitted on the {len(train)} training rows only ({train['date'].min().date()} .. "
       f"{train['date'].max().date()}); scaler, encoder and both models are identical when "
       f"validation rows are removed or changed")


def check_same_rows():
    table, _, _ = build_validation_predictions()
    counts = {m: int(table[m].notna().sum()) for m in METHODS}
    print(f"      scored rows per method: {counts}")
    expected = pd.MultiIndex.from_product([sorted(table["item"].unique()),
                                           pd.date_range(VAL_START, VAL_END)])
    got = pd.MultiIndex.from_frame(table[["item", "date"]]).sort_values()
    assert got.equals(expected), "scored rows are not the full validation grid"
    assert len(set(counts.values())) == 1 and counts[METHODS[0]] == len(table) == 92 * 8
    assert table["date"].min() == VAL_START and table["date"].max() == VAL_END
    scores = score(table, VAL_WEEK_MONDAYS)
    assert scores["n_item_weeks"] == 12 * 8
    ok(f"all {len(METHODS)} methods scored on the same {len(table)} rows "
       f"(92 days x 8 items) and the same 96 item-weeks")


def check_code_patterns():
    code = inspect.getsource(features.make_features) + inspect.getsource(features.rolling_mean_lag7)
    for pattern in ["shift(-", "center=True", ".ewm(", "pct_change", ".diff(", "expanding("]:
        assert pattern not in code, f"risky pattern in feature code: {pattern}"
    assert MIN_LAG >= 7 and min(LAG_DAYS) >= MIN_LAG
    ok(f"no risky patterns in the feature code; smallest lag = {min(LAG_DAYS)} days")


# - negative controls: must fail on leaky features (only here, never in features.py)
def check_test_catches_leaks(df):
    def with_leak(name, make_column):
        def feature_fn(data):
            out = make_features(data)
            out[name] = out.groupby("item")["sales"].transform(make_column)
            return out
        return feature_fn

    leaks = {
        "lag_1 (yesterday)": with_leak("leak", lambda s: s.shift(1)),
        "lag_6 (leaks on Sundays only)": with_leak("leak", lambda s: s.shift(6)),
        "7-day mean including today": with_leak("leak", lambda s: s.rolling(7).mean()),
    }
    for label, feature_fn in leaks.items():
        try:
            check_future_invariance(df, feature_fn, FEATURES + ["leak"], mondays=TEST_MONDAYS[4:5])
        except AssertionError as error:
            # - right reason: a changed feature value
            assert "features of week" in str(error), f"unexpected failure: {error}"
            print(f"      caught leaky feature: {label}")
            continue
        raise AssertionError(f"the leakage test did NOT catch the leaky feature {label}")
    ok("negative controls: the perturbation test fails on 3 deliberately leaky features")


def main():
    df = check_splits()
    check_test_catches_leaks(df)
    check_future_invariance(df)
    check_latest_dates(df)
    check_items_separate(df)
    check_forecast_invariance(df)
    check_fit_on_train_only(df)
    check_same_rows()
    check_code_patterns()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
