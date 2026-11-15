"""Load sales data; keep one store and chosen items."""

import pandas as pd

# - fixed split dates, ends inclusive
TRAIN_END = pd.Timestamp("2017-06-30")
VAL_START = pd.Timestamp("2017-07-01")
VAL_END = pd.Timestamp("2017-09-30")
TEST_START = pd.Timestamp("2017-10-01")
TEST_END = pd.Timestamp("2017-12-31")


def load_data(path):
    """Read, parse dates, sort by time.

    - errors on wrong columns, NaN, duplicates, negative sales
    """
    df = pd.read_csv(path)

    expected = ["date", "store", "item", "sales"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns {list(df.columns)}, expected {expected}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    if df.isna().any().any():
        raise ValueError("The data contains missing values")
    if df.duplicated(["store", "item", "date"]).any():
        raise ValueError("The data contains duplicate (store, item, date) rows")
    if (df["sales"] < 0).any():
        raise ValueError("The data contains negative sales")

    return df


def select_scope(df, store, items):
    """Keep one store and the chosen items (returns a copy)."""
    keep = (df["store"] == store) & (df["item"].isin(items))
    return df.loc[keep].copy().reset_index(drop=True)
