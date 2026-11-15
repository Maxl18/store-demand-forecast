"""Features for the weekly order forecast.

- leakage rule: order on Monday for Mon to Sun, so sales features look 7+ days back
- calendar features are known in advance
"""

import pandas as pd

# - all lags 7+ days: none reaches the target week
LAG_DAYS = [7, 14, 21, 28, 364]

# - checked by the leakage test
MIN_LAG = 7

# - FEATURES order = model column order
ID_FEATURES = ["item"]
CALENDAR_FEATURES = ["day_of_week", "month", "day_of_year"]
SALES_FEATURES = [f"lag_{lag}" for lag in LAG_DAYS] + [
    "mean_same_weekday_4",  # - mean of lag_7 to lag_28
    "mean_28_lag7",         # - 28-day mean ending 7 days ago
]
FEATURES = ID_FEATURES + CALENDAR_FEATURES + SALES_FEATURES


def check_daily_grid(df):
    """Error unless one row per item-day, no gaps.

    - shift(n) equals n days only on a full daily grid
    """
    for item, part in df.groupby("item"):
        steps = part["date"].diff().dropna()
        if not (steps == pd.Timedelta(days=1)).all():
            raise ValueError(f"Item {item}: dates are not one row per day without gaps")


def rolling_mean_lag7(sales, window):
    """Mean of `window` days ending at t-7 (one item)."""
    return sales.shift(MIN_LAG).rolling(window, min_periods=window).mean()


def make_features(df):
    """One row per (item, date): sales target plus features.

    - NaN where history is too short
    """
    out = df[["date", "item", "sales"]].sort_values(["item", "date"]).reset_index(drop=True)
    check_daily_grid(out)
    by_item = out.groupby("item")["sales"]

    # - 364 days = same weekday last year
    for lag in LAG_DAYS:
        out[f"lag_{lag}"] = by_item.shift(lag)

    # - any lag missing: mean missing
    same_weekday = [f"lag_{lag}" for lag in (7, 14, 21, 28)]
    out["mean_same_weekday_4"] = out[same_weekday].mean(axis=1, skipna=False)

    # - days t-34 to t-7
    out["mean_28_lag7"] = by_item.transform(lambda s: rolling_mean_lag7(s, 28))

    # - known in advance; Monday = 0
    out["day_of_week"] = out["date"].dt.day_of_week
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.day_of_year

    return out[["date", "item", "sales"] + CALENDAR_FEATURES + SALES_FEATURES]


def has_all_features(feats):
    """True where every feature exists.

    - all methods use it: same rows for every method
    """
    return feats[FEATURES].notna().all(axis=1)
