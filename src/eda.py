"""Explore the data; choose store and items (scope).

- run: .venv\\Scripts\\python.exe src\\eda.py
- cut at 2017-09-30 right after loading: no test sales
- scope from training period only (to 2017-06-30)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # - files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# - rule fixed before looking at numbers; training data only
STORE = 1
N_ITEMS = 8
SELECTION_RULE = (
    "Store 1. Rank the 50 items by mean daily sales in the training period "
    "(lowest to highest). Take 8 items evenly spaced along that ranking: "
    "rank positions round(linspace(0, 49, 8)) = 0, 7, 14, 21, 28, 35, 42, 49 "
    "(0 = lowest volume, 49 = highest volume)."
)
SELECTION_PERIOD = "2013-01-01 to 2017-06-30 (training period only)"

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
HIGHLIGHT = "#d95f02"
NEUTRAL = "#9aa5b1"
checks = {}  # - saved to eda_checks.json


def eta_squared(values, groups):
    """Variance share explained by groups (0 to 1)."""
    values = pd.Series(np.asarray(values, dtype=float))
    groups = pd.Series(np.asarray(groups))
    total = ((values - values.mean()) ** 2).sum()
    group_means = values.groupby(groups).transform("mean")
    between = ((group_means - values.mean()) ** 2).sum()
    return float(between / total) if total > 0 else float("nan")


df = load_data(ROOT / "train.csv")
# - drop test period at once
df = df[df["date"] <= VAL_END].reset_index(drop=True)
if (df["date"] > VAL_END).any() or df["date"].max() > VAL_END:
    raise ValueError(f"Rows after {VAL_END.date()} are still present: the test period must stay unseen")
print(f"Loaded {len(df):,} rows up to {df['date'].max().date()}, "
      f"{df['store'].nunique()} stores, {df['item'].nunique()} items")

all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
series_counts = df.groupby(["store", "item"])["date"].nunique()
checks["data_quality"] = {
    "rows": int(len(df)),
    "date_min": str(df["date"].min().date()),
    "date_max": str(df["date"].max().date()),
    "calendar_days_in_range": int(len(all_dates)),
    "n_series": int(len(series_counts)),
    "series_with_missing_dates": int((series_counts != len(all_dates)).sum()),
    "total_missing_dates": int(len(all_dates) * len(series_counts) - len(df)),
    "duplicate_keys": int(df.duplicated(["store", "item", "date"]).sum()),
    "missing_values": int(df.isna().sum().sum()),
    "negative_sales_rows": int((df["sales"] < 0).sum()),
    "zero_sales_rows": int((df["sales"] == 0).sum()),
    "sales_min": int(df["sales"].min()),
    "sales_max": int(df["sales"].max()),
    "all_sales_whole_numbers": bool(pd.api.types.is_integer_dtype(df["sales"])),
}
zero_rows = df[df["sales"] == 0][["date", "store", "item"]].copy()
zero_rows["date"] = zero_rows["date"].dt.strftime("%Y-%m-%d")
checks["data_quality"]["zero_sales_detail"] = zero_rows.to_dict("records")
print("Data quality:", json.dumps(checks["data_quality"], indent=2))

# - choose scope on training period only
train = df[df["date"] <= TRAIN_END]
store_train = train[train["store"] == STORE]

item_mean = store_train.groupby("item")["sales"].mean().sort_values()  # - low to high
ranked_items = list(item_mean.index)
positions = np.round(np.linspace(0, len(ranked_items) - 1, N_ITEMS)).astype(int)
chosen_items = sorted(int(ranked_items[p]) for p in positions)
chosen_by_rank = [int(ranked_items[p]) for p in positions]  # - low to high volume

print("Chosen store:", STORE)
print("Rank positions:", positions.tolist())
print("Chosen items (low to high volume):", chosen_by_rank)

scope = {
    "store": STORE,
    "items": chosen_items,
    "selection_rule": SELECTION_RULE,
    "selection_period": SELECTION_PERIOD,
}
with open(RESULTS / "scope.json", "w") as f:
    json.dump(scope, f, indent=2)

# - item summary, training period
g = store_train.groupby("item")["sales"]
summary = pd.DataFrame(
    {
        "mean": g.mean(),
        "std": g.std(),
        "min": g.min(),
        "max": g.max(),
        "share_zero_days": g.apply(lambda s: float((s == 0).mean())),
    }
)
summary["cv"] = summary["std"] / summary["mean"]  # - coefficient of variation
summary["volume_rank_low_to_high"] = summary["mean"].rank(method="first").astype(int) - 1
summary["chosen"] = summary.index.isin(chosen_items)
summary = summary.sort_values("mean").round(4)
summary.index.name = "item"
summary.to_csv(RESULTS / "eda_item_summary.csv")
print(summary.loc[chosen_by_rank].to_string())

# - patterns in scope, training period
tr = select_scope(train, STORE, chosen_items)
tr["weekday"] = tr["date"].dt.day_of_week
tr["month"] = tr["date"].dt.month
tr["year"] = tr["date"].dt.year

season_rows = []
for item, d in tr.groupby("item"):
    wk = d.groupby("weekday")["sales"].mean()
    mo = d.groupby("month")["sales"].mean()
    season_rows.append(
        {
            "item": int(item),
            "weekday_max_over_min": float(wk.max() / wk.min()),
            "best_weekday": WEEKDAYS[int(wk.idxmax())],
            "worst_weekday": WEEKDAYS[int(wk.idxmin())],
            "month_max_over_min": float(mo.max() / mo.min()),
            "best_month": MONTHS[int(mo.idxmax()) - 1],
            "worst_month": MONTHS[int(mo.idxmin()) - 1],
            "weekday_variance_share": eta_squared(d["sales"], d["weekday"]),
            "month_variance_share": eta_squared(d["sales"], d["month"]),
        }
    )
checks["seasonality_per_item"] = season_rows

daily_total = tr.groupby("date")["sales"].sum().reset_index()
daily_total["weekday"] = daily_total["date"].dt.day_of_week
daily_total["month"] = daily_total["date"].dt.month
daily_total["year"] = daily_total["date"].dt.year
wk_tot = daily_total.groupby("weekday")["sales"].mean()
mo_tot = daily_total.groupby("month")["sales"].mean()
checks["seasonality_chosen_total"] = {
    "mean_by_weekday": {WEEKDAYS[i]: float(v) for i, v in wk_tot.items()},
    "weekday_max_over_min": float(wk_tot.max() / wk_tot.min()),
    "mean_by_month": {MONTHS[i - 1]: float(v) for i, v in mo_tot.items()},
    "month_max_over_min": float(mo_tot.max() / mo_tot.min()),
    "weekday_variance_share": eta_squared(daily_total["sales"], daily_total["weekday"]),
    "month_variance_share": eta_squared(daily_total["sales"], daily_total["month"]),
    "year_variance_share": eta_squared(daily_total["sales"], daily_total["year"]),
}
# - Jan to Jun: 5 years, Jul to Dec: 4; full-year cross-check below
checks["seasonality_chosen_total"]["month_note"] = (
    "Month figures use all training days: Jan-Jun cover 5 years (2013-2017), Jul-Dec cover "
    "4 years (2013-2016). With a rising trend this slightly biases the month ratio and the "
    "month variance share. See seasonality_chosen_total_full_years_2013_2016 for the same "
    "numbers on equal years."
)
full_years = daily_total[daily_total["year"] <= 2016]
mo_full = full_years.groupby("month")["sales"].mean()
checks["seasonality_chosen_total_full_years_2013_2016"] = {
    "mean_by_month": {MONTHS[i - 1]: float(v) for i, v in mo_full.items()},
    "month_max_over_min": float(mo_full.max() / mo_full.min()),
    "month_variance_share": eta_squared(full_years["sales"], full_years["month"]),
}

# - trend: yearly growth (2013 to 2016; Jan to Jun 2017 vs 2016)
year_tot = daily_total[daily_total["year"] <= 2016].groupby("year")["sales"].sum()
h1 = daily_total[daily_total["date"].dt.month <= 6].groupby("year")["sales"].sum()
checks["trend_chosen_total"] = {
    "total_sales_by_full_year": {int(y): int(v) for y, v in year_tot.items()},
    "yoy_growth_full_years": {
        f"{y}_vs_{y - 1}": float(year_tot[y] / year_tot[y - 1] - 1) for y in year_tot.index[1:]
    },
    "jan_jun_total_by_year": {int(y): int(v) for y, v in h1.items()},
    "yoy_growth_jan_jun_2017_vs_2016": float(h1[2017] / h1[2016] - 1),
}
# - per-item growth, Jan to Jun 2017 vs 2016
item_h1 = (
    tr[tr["month"] <= 6].groupby(["item", "year"])["sales"].sum().unstack("year")
)
checks["trend_per_item_jan_jun_2017_vs_2016"] = {
    int(i): float(item_h1.loc[i, 2017] / item_h1.loc[i, 2016] - 1) for i in item_h1.index
}
# - context: store 1 all items, all stores (train only)
s1 = store_train.assign(year=store_train["date"].dt.year)
s1y = s1[s1["year"] <= 2016].groupby("year")["sales"].sum()
checks["trend_store1_all_items"] = {
    f"{y}_vs_{y - 1}": float(s1y[y] / s1y[y - 1] - 1) for y in s1y.index[1:]
}
allt = train.assign(year=train["date"].dt.year)
ally = allt[allt["year"] <= 2016].groupby("year")["sales"].sum()
checks["trend_all_stores_all_items"] = {
    f"{y}_vs_{y - 1}": float(ally[y] / ally[y - 1] - 1) for y in ally.index[1:]
}

# - outliers: |z| > 4 vs centred 28-day mean x weekday factor
# - descriptive only, feeds no decision
outlier_rows = []
disp_rows = []
for item, d in store_train.groupby("item"):
    d = d.sort_values("date").copy()
    level = d["sales"].rolling(28, center=True, min_periods=14).mean()
    wk_factor = d.groupby(d["date"].dt.day_of_week)["sales"].mean() / d["sales"].mean()
    expected = level * d["date"].dt.day_of_week.map(wk_factor)
    resid = d["sales"] - expected
    z = (resid - resid.mean()) / resid.std()
    flagged = d[np.abs(z) > 4]
    for _, r in flagged.iterrows():
        outlier_rows.append(
            {"item": int(item), "date": str(r["date"].date()), "sales": int(r["sales"]),
             "z_score": float(z.loc[r.name])}
        )
    # - about 1 = pure counting noise (Poisson)
    disp_rows.append(float(resid.var() / expected.mean()))
checks["outliers_store1_train"] = {
    "rule": "abs(z) > 4 where z standardises (sales - 28-day centred mean * weekday factor)",
    "n_series_checked": int(store_train["item"].nunique()),
    "n_rows_checked": int(len(store_train)),
    "n_flagged": len(outlier_rows),
    "flagged_rows": sorted(outlier_rows, key=lambda r: -abs(r["z_score"]))[:10],
}
checks["noise_dispersion_store1_train"] = {
    "note": "residual variance / mean expected sales per item; about 1 means pure counting noise",
    "min": float(np.min(disp_rows)),
    "median": float(np.median(disp_rows)),
    "max": float(np.max(disp_rows)),
}

# - synthetic data? regularity hints, not proof
profiles = (
    store_train.assign(wd=store_train["date"].dt.day_of_week)
    .groupby(["item", "wd"])["sales"].mean()
    .unstack("wd")
)
profiles = profiles.div(profiles.mean(axis=1), axis=0)
corr = np.corrcoef(profiles.values)
off_diag = corr[~np.eye(len(corr), dtype=bool)]
store_wd = (
    train.assign(wd=train["date"].dt.day_of_week)
    .groupby(["store", "wd"])["sales"].mean()
    .unstack("wd")
)
store_wd = store_wd.div(store_wd.mean(axis=1), axis=0)
daily_by_store = train.groupby(["date", "store"])["sales"].sum().unstack("store")
store_corr = daily_by_store.corr().values
store_off = store_corr[~np.eye(len(store_corr), dtype=bool)]
checks["regularity_train"] = {
    "weekday_profile_corr_between_items_store1": {
        "min": float(off_diag.min()),
        "median": float(np.median(off_diag)),
    },
    "sunday_factor_by_store": {int(s): float(v) for s, v in store_wd[6].items()},
    "store_daily_total_corr_between_stores": {
        "min": float(store_off.min()),
        "median": float(np.median(store_off)),
        "max": float(store_off.max()),
    },
}

# - calendar only, no test sales
def full_weeks(start, end):
    """Mondays of full Mon to Sun weeks inside [start, end]."""
    mondays = pd.date_range(start, end, freq="W-MON")
    return [m for m in mondays if m + pd.Timedelta(days=6) <= end]


weeks = {}
for name, (s, e) in {"validation": (VAL_START, VAL_END), "test": (TEST_START, TEST_END)}.items():
    mons = full_weeks(s, e)
    covered = {d for m in mons for d in pd.date_range(m, periods=7)}
    n_days = (e - s).days + 1
    weeks[name] = {
        "period": f"{s.date()} to {e.date()} ({n_days} days, first day is a {s.day_name()}, "
                  f"last day is a {e.day_name()})",
        "n_full_weeks": len(mons),
        "first_monday": str(mons[0].date()),
        "last_monday": str(mons[-1].date()),
        "last_sunday_of_last_full_week": str((mons[-1] + pd.Timedelta(days=6)).date()),
        "days_in_full_weeks": len(covered),
        "days_left_over_outside_full_weeks": n_days - len(covered),
    }
checks["full_weeks"] = weeks
print("Full weeks:", json.dumps(weeks, indent=2))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

fig, ax = plt.subplots(figsize=(11, 5))
order = item_mean.index.tolist()
colors = [HIGHLIGHT if i in chosen_items else NEUTRAL for i in order]
ax.bar(range(len(order)), item_mean.values, color=colors)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([str(i) for i in order], rotation=90, fontsize=8)
ax.set_xlabel("Item number (sorted from lowest to highest mean sales)")
ax.set_ylabel("Mean daily sales (units)")
ax.set_title(f"Store {STORE}: mean daily sales per item, training period "
             f"(orange = the {N_ITEMS} chosen items)")
fig.tight_layout()
fig.savefig(FIGURES / "eda_item_means.png", dpi=150)
plt.close(fig)

# - divide by item mean: one scale for all items
LINE_COLORS = ["#1f3a93", "#0072b2", "#009e73", "#4d9221", "#8c564b", "#d55e00", "#a6197d", "#000000"]
fig, ax = plt.subplots(figsize=(9, 5))
for k, item in enumerate(chosen_by_rank):
    d = tr[tr["item"] == item].groupby("weekday")["sales"].mean()
    ax.plot(WEEKDAYS, (d / tr[tr["item"] == item]["sales"].mean()).values, marker="o",
            color=LINE_COLORS[k % len(LINE_COLORS)],
            label=f"item {item} (mean {item_mean[item]:.1f})")
ax.axhline(1.0, color="black", lw=0.8, ls="--")
ax.set_xlabel("Day of the week")
ax.set_ylabel("Sales relative to the item's own average (1.0 = average day)")
ax.set_title(f"Store {STORE}: weekday pattern of the chosen items, training period")
ax.legend(fontsize=8, ncol=2, title="Legend sorted low to high volume", title_fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "eda_weekly_pattern.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for k, item in enumerate(chosen_by_rank):
    d = tr[tr["item"] == item].groupby("month")["sales"].mean()
    ax.plot(MONTHS, (d / tr[tr["item"] == item]["sales"].mean()).values, marker="o",
            color=LINE_COLORS[k % len(LINE_COLORS)],
            label=f"item {item} (mean {item_mean[item]:.1f})")
ax.axhline(1.0, color="black", lw=0.8, ls="--")
ax.set_xlabel("Month of the year")
ax.set_ylabel("Sales relative to the item's own average (1.0 = average month)")
ax.set_title(f"Store {STORE}: monthly pattern of the chosen items, training period "
             "(Jan-Jun: 5 years, Jul-Dec: 4 years)")
ax.legend(fontsize=8, ncol=2, title="Legend sorted low to high volume", title_fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "eda_yearly_pattern.png", dpi=150)
plt.close(fig)

# - test period = empty hatched band, no test sales
one_day = pd.Timedelta(days=1)
example_items = [chosen_by_rank[0], chosen_by_rank[N_ITEMS // 2], chosen_by_rank[-1]]
labels = ["low volume", "mid volume", "high volume"]
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, item, lab in zip(axes, example_items, labels):
    d = select_scope(df, STORE, [item]).set_index("date")["sales"]
    ax.plot(d.index, d.values, color=NEUTRAL, lw=0.5, label="daily sales")
    ax.plot(d.index, d.rolling(28, center=True).mean(), color="#1b4f72", lw=1.5,
            label="28-day average")
    # - +1 day: shade includes 2017-09-30
    ax.axvspan(VAL_START, VAL_END + one_day, color="#f4d03f", alpha=0.35, label="validation")
    ax.axvspan(TEST_START, TEST_END + one_day, facecolor="none", edgecolor="#7f8c8d",
               hatch="//", lw=0, label="test (not shown)")
    ax.set_xlim(d.index.min(), TEST_END + one_day)
    ax.set_ylabel("Units sold")
    ax.set_title(f"Store {STORE}, item {item} ({lab})", loc="left", fontsize=10)
    ax.text(TEST_START + (TEST_END - TEST_START) / 2, 0.5,
            "test period: not shown, kept unseen", transform=ax.get_xaxis_transform(),
            rotation=90, ha="center", va="center", fontsize=8, color="#34495e",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
axes[-1].set_xlabel("Date")
axes[0].legend(ncol=4, fontsize=8, loc="upper left")
fig.suptitle("Daily sales up to 2017-09-30, validation period shaded "
             "(test period kept unseen)")
fig.tight_layout()
fig.savefig(FIGURES / "eda_series_examples.png", dpi=150)
plt.close(fig)

checks["scope"] = scope
checks["chosen_items_low_to_high_with_train_mean"] = {
    int(i): float(item_mean[i]) for i in chosen_by_rank
}
with open(RESULTS / "eda_checks.json", "w") as f:
    json.dump(checks, f, indent=2)

print("Done. Wrote scope.json, eda_item_summary.csv, eda_checks.json and 4 figures.")
