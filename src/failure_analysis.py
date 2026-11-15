"""Failure analysis: where and why forecasts miss.

- any period: table of date, item, actual, one column per method
- fixed before the test run; same code for validation and test
- error = forecast minus actual (positive bias = too high)
- each "why" is a checkable number, not a proven cause
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # - files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.special import gammaln  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# - project metrics: numbers match other tables
from forecast import BASELINES, MODELS, TEST_END, TEST_START, mae, smape, weekly_totals  # noqa: E402

# - tuned buffers (the setting that picked the headline model)
SIM_SETTING = "tuned_buffer"

N_WORST = 10          # - worst days listed per model
SEED = 0              # - fixed seed
N_SIM = 2000          # - pure-noise simulations
N_PERM = 5000         # - shuffles for pattern tests
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class FailureAnalysisInputError(ValueError):
    """Bad or missing input (explicit raise, survives python -O)."""


def _need(condition, message):
    """Raise FailureAnalysisInputError(message) unless condition."""
    if not condition:
        raise FailureAnalysisInputError(message)


# - colour-blind friendly
COLOR_ACTUAL = "#000000"
COLOR_MODEL = ["#0072B2", "#D55E00"]                    # - first, second model
COLOR_BASELINE = ["#009E73", "#CC79A7", "#7f7f7f"]      # - baselines in given order


def _round(x, digits=4):
    """Round for JSON (plain float or None)."""
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, digits)


def _poisson_mad(lam):
    """Mean absolute miss of a Poisson count around `lam`.

    - floor even a perfect forecast cannot beat
    - 2 * exp(-lam) * lam^(k+1) / k!, k = floor(lam)
    """
    lam = np.maximum(np.asarray(lam, float), 1e-9)
    k = np.floor(lam)
    log_val = np.log(2) - lam + (k + 1) * np.log(lam) - gammaln(k + 1)
    return np.exp(log_val)


def _poisson_tail(lam, t):
    """P(Poisson count at least t from its mean)."""
    lam = np.asarray(lam, float)
    t = np.asarray(t, float)
    lower = np.where(lam - t >= 0, stats.poisson.cdf(np.floor(lam - t + 1e-9), lam), 0.0)
    upper = stats.poisson.sf(np.ceil(lam + t - 1e-9) - 1, lam)
    return lower + upper


def _spearman(x, y):
    """Spearman rho and p (None if not computable)."""
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": None, "p": None, "n": int(len(x))}
    res = stats.spearmanr(x, y)
    return {"rho": _round(res.statistic), "p": _round(res.pvalue), "n": int(len(x))}


def _full_week_mondays(df):
    """Mondays of full Mon to Sun weeks in the table."""
    week_start = df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit="D")
    counts = df.assign(week=week_start).groupby(["item", "week"]).size().unstack("item")
    full = counts.eq(7).all(axis=1)
    return pd.DatetimeIndex(full[full].index)


def _label_order(values):
    """Display order: weekdays, then numbers, else alphabetical.

    - output only; the test keeps alphabetical order (same p-values)
    """
    values = list(values)
    if all(v in WEEKDAYS for v in values):
        return sorted(values, key=WEEKDAYS.index)
    if all(str(v).lstrip("-").isdigit() for v in values):
        return sorted(values, key=lambda v: int(v))
    return sorted(values)


def _chi_square_pattern(labels, picked_mask, rng):
    """Shuffle test: are picked rows bunched on some labels?

    - p below 0.05 = more bunched than chance
    """
    labels = pd.Series(np.asarray(labels))
    cats = sorted(labels.unique())
    codes = pd.Categorical(labels, categories=cats).codes
    k = int(picked_mask.sum())
    share = np.bincount(codes, minlength=len(cats)) / len(codes)
    expected = share * k
    observed = np.bincount(codes[picked_mask], minlength=len(cats))

    def chi(counts):
        return float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-9)))

    obs_stat = chi(observed)
    sims = np.empty(N_PERM)
    for i in range(N_PERM):
        pick = rng.choice(len(codes), size=k, replace=False)
        sims[i] = chi(np.bincount(codes[pick], minlength=len(cats)))
    p_value = float((np.sum(sims >= obs_stat) + 1) / (N_PERM + 1))
    show = [cats.index(c) for c in _label_order(cats)]   # - display order only
    return {
        "labels": [str(cats[i]) for i in show],
        "observed": [int(observed[i]) for i in show],
        "expected": [_round(expected[i], 2) for i in show],
        "p_value": _round(p_value),
    }


def _overall_table(df, weekly, methods):
    """MAE, SMAPE, bias, weekly MAE, noise floor per method."""
    out = {}
    for m in methods:
        err = df[m] - df["actual"]
        w_err = weekly[m] - weekly["actual"]
        # - noise floor uses the forecast as the true mean: a guide only
        out[m] = {
            "mae": _round(mae(df["actual"], df[m])),
            "smape_pct": _round(smape(df["actual"], df[m])),
            "bias": _round(err.mean()),
            "weekly_mae": _round(mae(weekly["actual"], weekly[m])),
            "weekly_bias": _round(w_err.mean()),
            "noise_floor_mae": _round(_poisson_mad(df[m]).mean()),
            "noise_floor_weekly_mae": _round(_poisson_mad(weekly[m]).mean()),
            # - 1.0 = all counting noise; 2.0 = half
            "error_variance_vs_noise": _round(np.mean((err**2) / np.maximum(df[m], 1e-9))),
        }
    return out


def _worst_days_table(df, models, baselines):
    rows = []
    for m in models:
        err = df[m] - df["actual"]
        top = err.abs().sort_values(ascending=False, kind="mergesort").head(N_WORST).index
        for rank, idx in enumerate(top, start=1):
            r = df.loc[idx]
            row = {
                "model": m,
                "rank": rank,
                "date": r["date"].strftime("%Y-%m-%d"),
                "weekday": WEEKDAYS[int(r["date"].day_of_week)],
                "item": int(r["item"]),
                "actual": _round(r["actual"], 2),
                "predicted": _round(r[m], 2),
                "error": _round(err.loc[idx], 2),
                "abs_error": _round(abs(err.loc[idx]), 2),
                # - error / sqrt(forecast); about 1 normal, 3+ unusual
                "noise_units": _round(err.loc[idx] / np.sqrt(max(r[m], 1e-9)), 2),
            }
            for b in baselines:
                row[b] = _round(r[b], 2)
            rows.append(row)
    return pd.DataFrame(rows)


def _by_item_table(df, weekly, methods, models, baselines, sim_costs):
    rows = []
    means = df.groupby("item")["actual"].mean().sort_values()
    for rank, (item, mean_sales) in enumerate(means.items(), start=1):
        d = df[df["item"] == item]
        w = weekly[weekly["item"] == item]
        row = {"item": int(item), "volume_rank_low_to_high": rank, "mean_sales": _round(mean_sales, 2)}
        for m in methods:
            e = d[m] - d["actual"]
            row[f"{m}_mae"] = _round(mae(d["actual"], d[m]))
            row[f"{m}_smape_pct"] = _round(smape(d["actual"], d[m]))
            row[f"{m}_bias"] = _round(e.mean())
            row[f"{m}_mae_rel_pct"] = _round(100 * mae(d["actual"], d[m]) / mean_sales, 2)
            row[f"{m}_weekly_mae"] = _round(mae(w["actual"], w[m]))
            row[f"{m}_weekly_bias"] = _round((w[m] - w["actual"]).mean())
        for m in models:
            row[f"{m}_noise_floor_mae"] = _round(_poisson_mad(d[m]).mean())
            row[f"{m}_noise_floor_weekly_mae"] = _round(_poisson_mad(w[m]).mean())
        daily = {m: row[f"{m}_mae"] for m in methods}
        wk = {m: row[f"{m}_weekly_mae"] for m in methods}
        row["best_method_daily_mae"] = min(daily, key=daily.get)
        row["best_baseline_daily_mae"] = min(baselines, key=lambda b: daily[b])
        row["best_method_weekly_mae"] = min(wk, key=wk.get)
        row["best_baseline_weekly_mae"] = min(baselines, key=lambda b: wk[b])
        if sim_costs is not None and int(item) in sim_costs.index:
            cost = sim_costs.loc[int(item)]
            for m in methods:
                row[f"{m}_order_cost"] = _round(cost[m], 1)
            cm = {m: float(cost[m]) for m in methods}
            row["best_method_order_cost"] = min(cm, key=cm.get)
            row["best_baseline_order_cost"] = min(baselines, key=lambda b: cm[b])
        rows.append(row)
    return pd.DataFrame(rows)


def _by_group_table(df, key, key_name, models, best_baseline):
    """MAE and bias per weekday or month."""
    cols = list(models) + [best_baseline]
    rows = []
    for k, d in df.groupby(key, sort=True):
        row = {key_name: k, "n_rows": int(len(d)), "mean_sales": _round(d["actual"].mean(), 2)}
        for m in cols:
            e = d[m] - d["actual"]
            row[f"{m}_mae"] = _round(e.abs().mean())
            row[f"{m}_bias"] = _round(e.mean())
            row[f"{m}_bias_pct"] = _round(100 * e.sum() / d["actual"].sum(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# - "why" checks: plain numbers for summary.json
def _noise_check(df, models, baselines):
    """Biggest errors: just counting noise? Do they bunch?"""
    rng = np.random.default_rng(SEED)
    out = {}
    n = len(df)
    for m in models:
        err = (df[m] - df["actual"]).to_numpy()
        lam = np.maximum(df[m].to_numpy(), 1e-9)
        abs_err = np.abs(err)
        order = np.argsort(-abs_err, kind="mergesort")
        tenth = abs_err[order[N_WORST - 1]]

        # - simulate: forecast = true mean, sales = Poisson noise
        sim_max = np.empty(N_SIM)
        sim_tenth = np.empty(N_SIM)
        for i in range(N_SIM):
            draw = rng.poisson(lam)
            a = np.sort(np.abs(draw - lam))[::-1]
            sim_max[i] = a[0]
            sim_tenth[i] = a[N_WORST - 1]
        # - days this large expected from noise
        expected_count = float(_poisson_tail(lam, np.full(n, tenth)).sum())
        z = err / np.sqrt(lam)

        # - worst days bunched by weekday, item, month?
        weekday = df["date"].dt.day_of_week.map(lambda i: WEEKDAYS[i]).to_numpy()
        month = df["date"].dt.strftime("%Y-%m").to_numpy()
        item = df["item"].astype(str).to_numpy()
        top_abs = np.zeros(n, bool)
        top_abs[order[:N_WORST]] = True
        k_big = max(N_WORST, int(round(0.05 * n)))          # - top 5% of days
        top_big_abs = np.zeros(n, bool)
        top_big_abs[order[:k_big]] = True
        top_big_z = np.zeros(n, bool)
        top_big_z[np.argsort(-np.abs(z), kind="mergesort")[:k_big]] = True

        out[m] = {
            "max_abs_error": _round(abs_err.max(), 2),
            "tenth_largest_abs_error": _round(tenth, 2),
            "noise_only_max_abs_error_mean": _round(sim_max.mean(), 2),
            "noise_only_max_abs_error_p95": _round(np.percentile(sim_max, 95), 2),
            "noise_only_tenth_largest_mean": _round(sim_tenth.mean(), 2),
            "noise_only_tenth_largest_p95": _round(np.percentile(sim_tenth, 95), 2),
            "noise_only_expected_days_at_least_tenth_largest": _round(expected_count, 2),
            "share_sim_tenth_largest_below_observed": _round(np.mean(sim_tenth < tenth)),
            "noise_units_std": _round(z.std()),
            "days_noise_units_over_3": int((np.abs(z) > 3).sum()),
            "days_noise_units_over_3_expected_pure_noise": _round(
                _poisson_tail(lam, 3 * np.sqrt(lam)).sum(), 2),
            "k_top_share_tests": int(k_big),
            "pattern_top10_by_abs_error": {
                "weekday": _chi_square_pattern(weekday, top_abs, rng),
                "item": _chi_square_pattern(item, top_abs, rng),
                "month": _chi_square_pattern(month, top_abs, rng),
            },
            "pattern_top5pct_by_abs_error": {
                "weekday": _chi_square_pattern(weekday, top_big_abs, rng),
                "item": _chi_square_pattern(item, top_big_abs, rng),
                "month": _chi_square_pattern(month, top_big_abs, rng),
            },
            "pattern_top5pct_by_noise_units": {
                "weekday": _chi_square_pattern(weekday, top_big_z, rng),
                "item": _chi_square_pattern(item, top_big_z, rng),
                "month": _chi_square_pattern(month, top_big_z, rng),
            },
            "top10_share_over_forecast": _round(np.mean(err[order[:N_WORST]] > 0)),
        }
    return out


def _error_relationships(df, weekly, models, baselines, methods):
    """Same bad days across methods? Errors repeating?"""
    err = pd.DataFrame({m: df[m] - df["actual"] for m in methods})
    w_err = pd.DataFrame({m: weekly[m] - weekly["actual"] for m in methods})
    out = {
        "daily_error_correlation": err.corr().round(3).to_dict(),
        "weekly_error_correlation": w_err.corr().round(3).to_dict(),
    }

    # - model-specific error would make models disagree about as much as they err
    if len(models) >= 2:
        a, b = models[0], models[1]
        out["model_disagreement_std"] = _round((df[a] - df[b]).std())
        out["model_error_std"] = {m: _round(err[m].std()) for m in models}

    # - worst-day overlap between methods
    worst = {m: set(err[m].abs().sort_values(ascending=False, kind="mergesort").head(N_WORST).index)
             for m in methods}
    wide = {m: set(err[m].abs().sort_values(ascending=False, kind="mergesort").head(3 * N_WORST).index)
            for m in methods}
    out["worst10_overlap_with_other_methods_top30"] = {
        m: {o: int(len(worst[m] & wide[o])) for o in methods if o != m} for m in models}

    # - on model's worst days: rules also wrong, same direction? = hard for everyone
    same_dir, avg_abs = {}, {}
    for m in models:
        idx = list(worst[m])
        same_dir[m] = int(sum(all(np.sign(err.loc[i, b]) == np.sign(err.loc[i, m]) for b in baselines)
                              for i in idx))
        avg_abs[m] = {o: _round(err.loc[idx, o].abs().mean(), 2) for o in methods}
    out["worst10_days_all_baselines_also_wrong_same_direction"] = same_dir
    out["worst10_days_mean_abs_error_by_method"] = avg_abs
    if len(models) >= 2:
        out["worst10_days_shared_by_first_two_models"] = int(len(worst[models[0]] & worst[models[1]]))

    # - date effect: all items miss the same way on a date?
    for m in models:
        piv = (df[m] - df["actual"]).groupby([df["date"], df["item"]]).first().unstack("item")
        corr = piv.corr().to_numpy()
        iu = np.triu_indices_from(corr, k=1)
        out.setdefault("cross_item_same_day_error_correlation_mean", {})[m] = _round(corr[iu].mean())
    # - pure noise gives about 0
    # - errors repeating day to day or week to week?
    for m in models:
        d = df.sort_values(["item", "date"]).assign(e=lambda x: x[m] - x["actual"])
        lag1 = d.groupby("item")["e"].apply(lambda s: s.autocorr(1)).mean()
        wk = w_err[m].groupby(weekly["item"]).apply(lambda s: s.autocorr(1)).mean()
        out.setdefault("daily_error_autocorr_lag1_mean_over_items", {})[m] = _round(lag1)
        out.setdefault("weekly_error_autocorr_lag1_mean_over_items", {})[m] = _round(wk)
    return out


def _calendar_scale_check(df, train_sales, models, ridge_pipeline):
    """Fixed-unit calendar effect vs items of different size.

    - Ridge adds the same units to every item; a real % effect misfits small and big items
    - checks: training swing, Ridge calendar coefficients, monthly bias vs volume
    """
    out = {}
    item_mean = df.groupby("item")["actual"].mean()
    if train_sales is not None:
        t = train_sales.copy()
        t["month"] = t["date"].dt.month
        t["wd"] = t["date"].dt.day_of_week
        base = t.groupby("item")["sales"].mean()
        month_prof = t.groupby(["item", "month"])["sales"].mean().unstack()
        wd_prof = t.groupby(["item", "wd"])["sales"].mean().unstack()
        month_swing = month_prof.max(axis=1) - month_prof.min(axis=1)
        wd_swing = wd_prof.max(axis=1) - wd_prof.min(axis=1)
        out["training_swing"] = {
            "month_swing_units": {int(i): _round(v, 2) for i, v in month_swing.items()},
            "month_swing_pct_of_mean": {int(i): _round(100 * month_swing[i] / base[i], 1) for i in base.index},
            "weekday_swing_units": {int(i): _round(v, 2) for i, v in wd_swing.items()},
            "weekday_swing_pct_of_mean": {int(i): _round(100 * wd_swing[i] / base[i], 1) for i in base.index},
            "month_swing_units_max_over_min_item": _round(month_swing.max() / month_swing.min(), 2),
            "month_swing_pct_max_over_min_item": _round(
                (month_swing / base).max() / (month_swing / base).min(), 2),
            "weekday_swing_units_max_over_min_item": _round(wd_swing.max() / wd_swing.min(), 2),
            "weekday_swing_pct_max_over_min_item": _round(
                (wd_swing / base).max() / (wd_swing / base).min(), 2),
            "volume_max_over_min_item": _round(base.max() / base.min(), 2),
        }
    if ridge_pipeline is not None:
        # - shrunk coefficients: only max minus min means anything
        prep = ridge_pipeline.named_steps["prep"]
        names = list(prep.get_feature_names_out())
        coefs = ridge_pipeline.named_steps["ridge"].coef_
        res = {}
        for prefix, label in [("onehot__month_", "month"), ("onehot__day_of_week_", "weekday")]:
            vals = np.array([c for n, c in zip(names, coefs) if n.startswith(prefix)])
            res[f"ridge_{label}_coefficient_swing_units"] = _round(vals.max() - vals.min(), 2)
        out["ridge_coefficients"] = res

    # - bias vs volume, per month and model
    d = df.assign(month=df["date"].dt.strftime("%Y-%m"))
    bias_vs_volume = {}
    for m in models:
        per_month = {}
        for mon, g in d.groupby("month"):
            by_item = g.groupby("item").apply(
                lambda x: pd.Series({"units": (x[m] - x["actual"]).mean(),
                                     "pct": 100 * (x[m] - x["actual"]).sum() / x["actual"].sum()}),
                include_groups=False)
            vol = item_mean.loc[by_item.index]
            per_month[mon] = {
                "bias_pct_vs_volume_spearman": _spearman(vol.to_numpy(), by_item["pct"].to_numpy()),
                "bias_units_vs_volume_spearman": _spearman(vol.to_numpy(), by_item["units"].to_numpy()),
                "bias_pct_by_item": {int(i): _round(v, 2) for i, v in by_item["pct"].items()},
                "bias_units_by_item": {int(i): _round(v, 2) for i, v in by_item["units"].items()},
            }
        bias_vs_volume[m] = per_month
    out["bias_vs_volume_by_month"] = bias_vs_volume
    return out


def _ceiling_check(df, train_sales, models, baselines, best_baseline):
    """Do trees stop short of levels unseen in training?

    - tree forecast = average of training sales, never above the training max
    - per item: training max vs forecasts and actuals; level vs best earlier year
    """
    if train_sales is None:
        return None
    out = {"per_item": {}, "level_bins": {}}
    months = sorted(df["date"].dt.month.unique())
    t = train_sales.assign(year=train_sales["date"].dt.year, month=train_sales["date"].dt.month)
    item_mean = df.groupby("item")["actual"].mean()
    rel_level, bias_pct = {}, {m: {} for m in models}
    for item, d in df.groupby("item"):
        hist = t[t["item"] == item]
        same_months = hist[hist["month"].isin(months)].groupby("year")["sales"].mean()
        best_prior = float(same_months.max()) if len(same_months) else np.nan
        row = {
            "train_max": int(hist["sales"].max()),
            "train_p99": _round(hist["sales"].quantile(0.99), 1),
            "actual_max_in_period": _round(d["actual"].max(), 1),
            "days_actual_above_train_max": int((d["actual"] > hist["sales"].max()).sum()),
            "period_mean_actual": _round(d["actual"].mean(), 2),
            "best_earlier_year_same_months_mean": _round(best_prior, 2),
            "period_level_vs_best_earlier_year_pct": _round(100 * (d["actual"].mean() / best_prior - 1), 1),
        }
        rel_level[int(item)] = 100 * (d["actual"].mean() / best_prior - 1)
        for m in models:
            row[f"{m}_max_forecast"] = _round(d[m].max(), 1)
            row[f"{m}_bias_pct"] = _round(100 * (d[m] - d["actual"]).sum() / d["actual"].sum(), 2)
            bias_pct[m][int(item)] = 100 * (d[m] - d["actual"]).sum() / d["actual"].sum()
        out["per_item"][int(item)] = row
    ids = sorted(rel_level)
    out["spearman_level_vs_best_year_and_bias_pct"] = {
        m: _spearman(np.array([rel_level[i] for i in ids]), np.array([bias_pct[m][i] for i in ids]))
        for m in models}

    # - bias by level (best baseline) vs the item's training history
    q = np.array([(train_sales.loc[train_sales["item"] == i, "sales"].to_numpy() <= v).mean()
                  for i, v in zip(df["item"], df[best_baseline])])
    bins = pd.cut(q, [0, 0.5, 0.9, 1.0001], labels=["bottom half", "middle (50-90%)", "top 10%"],
                  include_lowest=True)
    top_item = int(item_mean.idxmax())
    for label, mask in [("all_items", np.ones(len(df), bool)), (f"top_volume_item_{top_item}", (df["item"] == top_item).to_numpy())]:
        res = {}
        for b in ["bottom half", "middle (50-90%)", "top 10%"]:
            sel = mask & (np.asarray(bins) == b)
            row = {"n_days": int(sel.sum())}
            for m in models:
                row[f"{m}_bias_pct"] = _round(
                    100 * (df.loc[sel, m] - df.loc[sel, "actual"]).sum() / df.loc[sel, "actual"].sum(), 2
                ) if sel.sum() else None
            res[b] = row
        out["level_bins"][label] = res
    out["top_volume_item"] = top_item
    return out


def _simple_rule_wins(by_item, models, baselines, headline, sim_costs_available):
    """Every case where a simple rule beats a model."""
    metrics = [("daily_mae", "_mae"), ("weekly_mae", "_weekly_mae")]
    if sim_costs_available:
        metrics.append(("order_cost", "_order_cost"))
    out = {}
    for m in models:
        out[m] = {}
        for name, suffix in metrics:
            wins = []
            for _, r in by_item.iterrows():
                best_b = min(baselines, key=lambda b: r[f"{b}{suffix}"])
                if r[f"{best_b}{suffix}"] < r[f"{m}{suffix}"]:
                    wins.append({
                        "item": int(r["item"]),
                        "best_baseline": best_b,
                        "baseline_value": _round(r[f"{best_b}{suffix}"], 2),
                        "model_value": _round(r[f"{m}{suffix}"], 2),
                        "items_mean_sales": _round(r["mean_sales"], 1),
                    })
            out[m][name] = {"n_items_baseline_wins": len(wins), "n_items": int(len(by_item)), "cases": wins}
    out["headline_model"] = headline
    return out


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _line_color(m, models, baselines):
    if m in models:
        return COLOR_MODEL[models.index(m) % len(COLOR_MODEL)]
    return COLOR_BASELINE[baselines.index(m) % len(COLOR_BASELINE)]


def _pretty(m):
    return {"ridge": "Ridge", "hgb": "Boosting", "baseline_lag7": "Rule: same day last week",
            "baseline_mean4": "Rule: mean of last 4 same weekdays",
            "baseline_lag364": "Rule: same day last year"}.get(m, m)


def _legend_below(fig, ax, ncol, fontsize=9):
    """Shared legend below the plots."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=fontsize)


def _pick_worst_item_week(weekly, focus):
    """Fixed rule: item-week with the largest weekly miss.

    - ties: lower item, then earlier week
    """
    ordered = weekly.assign(abs_err=(weekly[focus] - weekly["actual"]).abs()).sort_values(
        ["abs_err", "item", "week_monday"], ascending=[False, True, True], kind="mergesort")
    row = ordered.iloc[0]
    return int(row["item"]), pd.Timestamp(row["week_monday"])


def _pick_rule_beats_item(weekly, baselines, focus):
    """Fixed rule: item where a simple rule beats the model most (weekly MAE).

    - returns (item, model MAE, rule MAE) or None
    """
    best = None
    for item, w in weekly.groupby("item"):
        head = mae(w["actual"], w[focus])
        rule = min(mae(w["actual"], w[b]) for b in baselines)
        if head - rule > 0 and (best is None or head - rule > best[1] - best[2]):
            best = (int(item), head, rule)
    return best


def _fig_example(df, weekly, models, baselines, focus, period_label, fig_dir, file_name,
                 item, week_monday=None):
    """Forecast vs actual for one item: daily (top), weekly (bottom).

    - week_monday shaded if given; returns the title numbers
    """
    d = df[df["item"] == item].sort_values("date")
    w = weekly[weekly["item"] == item].sort_values("week_monday").reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 9.6))
    ax = axes[0]
    ax.plot(d["date"], d["actual"], color=COLOR_ACTUAL, lw=1.8, label="Actual sales")
    for m in baselines + models:
        ax.plot(d["date"], d[m], color=_line_color(m, models, baselines),
                lw=1.8 if m in models else 1.0, ls="-" if m in models else "--",
                label=_pretty(m), alpha=0.95 if m in models else 0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold per day")
    ax.set_title(f"Item {item}: daily forecasts vs actual sales ({period_label})")

    ax2 = axes[1]
    x = np.arange(len(w))
    ax2.plot(x, w["actual"], color=COLOR_ACTUAL, lw=2.2, marker="o", label="Actual sales")
    for m in baselines + models:
        ax2.plot(x, w[m], color=_line_color(m, models, baselines), marker="o", ms=4,
                 lw=1.8 if m in models else 1.0, ls="-" if m in models else "--", label=_pretty(m))
    ax2.set_xticks(x)
    ax2.set_xticklabels([t.strftime("%d %b") for t in w["week_monday"]], rotation=45)
    ax2.set_xlabel("Week starting (Monday)")
    ax2.set_ylabel("Units sold per week")

    info = {"model": focus, "item": int(item)}
    if week_monday is not None:
        r = w[w["week_monday"] == week_monday].iloc[0]
        miss = float(r[focus] - r["actual"])
        ax.axvspan(week_monday - pd.Timedelta(hours=12), week_monday + pd.Timedelta(days=6, hours=12),
                   color="#999999", alpha=0.25, lw=0)
        pos = int(w.index[w["week_monday"] == week_monday][0])
        ax2.axvspan(pos - 0.4, pos + 0.4, color="#999999", alpha=0.25, lw=0)
        ax2.set_title(
            f"Item {item}: weekly totals. Biggest weekly miss of {_pretty(focus)} (shaded week of "
            f"{week_monday.strftime('%d %b')}): forecast {r[focus]:.0f}, actual {r['actual']:.0f}, "
            f"off by {miss:+.0f} units ({100 * miss / r['actual']:+.0f}%)", fontsize=9)
        info.update({"week_monday": week_monday.strftime("%Y-%m-%d"), "weekly_actual": _round(r["actual"], 1),
                     "weekly_forecast": _round(r[focus], 1), "weekly_error": _round(miss, 1),
                     "weekly_error_pct": _round(100 * miss / r["actual"], 1),
                     "baseline_weekly_forecasts": {b: _round(r[b], 1) for b in baselines}})
    else:
        head = mae(w["actual"], w[focus])
        rule = min(mae(w["actual"], w[b]) for b in baselines)
        ax2.set_title(f"Item {item}: weekly totals (what the shop owner orders). {_pretty(focus)} misses by "
                      f"{head:.1f} units a week on average, the best simple rule by {rule:.1f}", fontsize=9)
        info.update({"model_weekly_mae": _round(head), "best_rule_weekly_mae": _round(rule)})
    _legend_below(fig, ax, ncol=3)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(fig_dir / file_name, dpi=130)
    plt.close(fig)
    return info


def _fig_by_item(by_item, models, baselines, best_baseline, period_label, fig_dir):
    """Error and bias per item, low to high volume."""
    items = by_item["item"].tolist()
    x = np.arange(len(items))
    labels = [f"{i}\n({v:.0f}/day)" for i, v in zip(items, by_item["mean_sales"])]
    cols = models + [best_baseline]
    width = 0.8 / len(cols)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10))
    for ax, suffix, ylabel, title in [
        (axes[0], "_mae_rel_pct", "Daily MAE as % of the item's mean sales",
         f"Daily forecast error relative to item size ({period_label}), lower is better"),
        (axes[1], "_weekly_bias", "Weekly bias (forecast minus actual), units per week",
         f"Weekly bias by item ({period_label}): above 0 = too many units forecast, below 0 = too few"),
    ]:
        for j, m in enumerate(cols):
            ax.bar(x + (j - (len(cols) - 1) / 2) * width, by_item[f"{m}{suffix}"], width,
                   color=_line_color(m, models, baselines), label=_pretty(m))
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
    axes[1].axhline(0, color="black", lw=0.8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Item (ordered by mean daily sales, low to high; mean units per day in brackets)")
    _legend_below(fig, axes[0], ncol=len(cols))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fig_dir / f"failure_{period_label}_by_item.png", dpi=130)
    plt.close(fig)


def _fig_bias(df, summary_scale, models, baselines, by_item, period_label, fig_dir):
    """Figure: fixed units vs percent.

    - training swing per item (units, %), then bias per item and month
    """
    ts = summary_scale.get("training_swing")
    bvm = summary_scale["bias_vs_volume_by_month"]
    months = sorted(next(iter(bvm.values())).keys())
    vol = by_item.set_index("item")["mean_sales"]
    items = list(vol.index)
    n_panels = len(models) + (2 if ts else 0)
    n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.6 * n_cols, 5.6 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for extra in axes[n_panels:]:
        extra.axis("off")
    xs = [vol[i] for i in items]
    k = 0
    if ts:
        ax = axes[0]
        ax.plot(xs, [ts["month_swing_units"][i] for i in items], "o-", color="#0072B2",
                label="Month swing (best month minus worst month)")
        ax.plot(xs, [ts["weekday_swing_units"][i] for i in items], "s-", color="#009E73",
                label="Weekday swing (best day minus worst day)")
        for i in items:
            ax.annotate(str(i), (vol[i], ts["month_swing_units"][i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
        ax.set_xlabel("Item size: mean units sold per day in the forecast period")
        ax.set_ylabel("Calendar swing in the training data (units per day)")
        ax.set_title("In UNITS the swing grows with item size\n"
                     f"(biggest item / smallest item: month swing x{ts['month_swing_units_max_over_min_item']}, "
                     f"volume x{ts['volume_max_over_min_item']})", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=8)

        ax = axes[1]
        m_pct = [ts["month_swing_pct_of_mean"][i] for i in items]
        w_pct = [ts["weekday_swing_pct_of_mean"][i] for i in items]
        ax.plot(xs, m_pct, "o-", color="#0072B2", label="Month swing, % of the item's mean sales")
        ax.plot(xs, w_pct, "s-", color="#009E73", label="Weekday swing, % of the item's mean sales")
        for i, y in zip(items, m_pct):
            ax.annotate(str(i), (vol[i], y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_ylim(0, max(m_pct + w_pct) * 1.25)
        ax.set_xlabel("Item size: mean units sold per day in the forecast period")
        ax.set_ylabel("Calendar swing in the training data (% of the item's mean)")
        ax.set_title("In PERCENT the swing is almost the same for every item\n"
                     f"(biggest / smallest item: month x{ts['month_swing_pct_max_over_min_item']}, "
                     f"weekday x{ts['weekday_swing_pct_max_over_min_item']}), a percent effect", fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=8)
        k = 2
    cmap = plt.get_cmap("viridis")
    for j_model, m in enumerate(models):
        ax = axes[k + j_model]
        for j, mon in enumerate(months):
            pct = bvm[m][mon]["bias_pct_by_item"]
            ax.plot(xs, [pct[i] for i in items], "o-",
                    color=cmap(j / max(len(months) - 1, 1) * 0.85), label=mon)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Item size: mean units sold per day in the forecast period")
        ax.set_ylabel(f"{_pretty(m)} bias, % of the item's sales")
        ax.set_title(f"{_pretty(m)} bias by item and month ({period_label}), above 0 = forecast too high", fontsize=9)
        ax.legend(title="Month", loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=min(len(months), 4), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"failure_{period_label}_bias.png", dpi=130)
    plt.close(fig)


def _fig_noise(df, models, period_label, fig_dir):
    """Standardised errors vs pure counting noise."""
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(SEED)
    bins = np.linspace(-5, 5, 41)
    for ax, m in zip(axes, models):
        lam = np.maximum(df[m].to_numpy(), 1e-9)
        z = (df[m] - df["actual"]).to_numpy() / np.sqrt(lam)
        sim = (rng.poisson(np.tile(lam, 20)) - np.tile(lam, 20)) / np.sqrt(np.tile(lam, 20))
        ax.hist(np.clip(z, -5, 5), bins=bins, density=True, color=_line_color(m, models, []), alpha=0.75,
                label=f"{_pretty(m)}: actual errors")
        ax.hist(np.clip(-sim, -5, 5), bins=bins, density=True, histtype="step", color="black", lw=1.6,
                label="Pure counting noise only")
        ax.set_xlabel("Error in noise units (forecast minus actual, divided by sqrt of forecast)")
        ax.set_title(f"{_pretty(m)} ({period_label})", fontsize=10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)
    axes[0].set_ylabel("Share of days (density)")
    fig.suptitle("Are the errors bigger than counting noise alone? Wider than the black line = yes", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / f"failure_{period_label}_noise.png", dpi=130)
    plt.close(fig)


def _fig_ceiling(df, train_sales, models, ceiling, period_label, fig_dir):
    item = ceiling["top_volume_item"]
    d = df[df["item"] == item].sort_values("date")
    info = ceiling["per_item"][item]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(d["date"], d["actual"], color=COLOR_ACTUAL, lw=1.6, label="Actual sales")
    for m in models:
        ax.plot(d["date"], d[m], color=_line_color(m, models, []), lw=1.5, label=f"{_pretty(m)} forecast")
    ax.axhline(info["train_max"], color="#CC79A7", ls="--", lw=1.4,
               label=f"Highest daily sales in training ({info['train_max']})")
    ax.axhline(info["train_p99"], color="#7f7f7f", ls=":", lw=1.4,
               label=f"99th percentile of training sales ({info['train_p99']})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Units sold per day")
    ax.set_title(f"Item {item} (highest volume): forecasts against the largest values seen in training ({period_label})",
                 fontsize=10)
    _legend_below(fig, ax, ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(fig_dir / f"failure_{period_label}_ceiling.png", dpi=130)
    plt.close(fig)


def _check_inputs(pred_df, period_label, models, baselines, headline_model, train_sales,
                  sim_table, cost_ratio, sim_setting):
    """Stop on missing or mismatched inputs.

    - checks shape and dates only, not values
    """
    _need(isinstance(period_label, str) and period_label.strip() != "" and period_label.replace("_", "").isalnum(),
          f"period_label must be a short word such as 'validation' or 'test', got {period_label!r}")
    _need(len(models) >= 1 and len(baselines) >= 1, "model_cols and baseline_cols must not be empty")
    _need(len(set(models + baselines)) == len(models + baselines),
          "model_cols and baseline_cols must not repeat a column")
    _need(headline_model in models,
          f"headline_model {headline_model!r} must be one of model_cols {models}")
    _need(pred_df is not None, "pred_df is required")
    need_cols = ["date", "item", "actual"] + models + baselines
    missing = [c for c in need_cols if c not in pred_df.columns]
    _need(not missing, f"pred_df is missing the columns {missing}")
    _need(len(pred_df) > 0, "pred_df is empty")
    _need(not pred_df[need_cols].isna().any().any(), "pred_df has missing values")
    dates = pd.to_datetime(pred_df["date"])
    _need(not pred_df.assign(date=dates).duplicated(["date", "item"]).any(),
          "pred_df has more than one row for the same date and item")
    per_item = pred_df.groupby("item").size()
    _need(per_item.nunique() == 1, "every item must have the same number of days in pred_df")

    # - label must match the dates
    if period_label == "test":
        _need(dates.min() >= TEST_START and dates.max() <= TEST_END,
              f"period_label 'test' needs dates from {TEST_START.date()} to {TEST_END.date()}, "
              f"got {dates.min().date()} .. {dates.max().date()}")
    if period_label == "validation":
        _need(dates.max() < TEST_START,
              f"period_label 'validation' must not contain dates from {TEST_START.date()} on")

    # - train sales: for swing and ceiling checks
    _need(train_sales is not None,
          "train_sales is required (columns date, item, sales of the data the models were fitted on)")
    _need(all(c in train_sales.columns for c in ["date", "item", "sales"]),
          "train_sales needs the columns date, item and sales")
    t_dates = pd.to_datetime(train_sales["date"])
    _need(len(train_sales) > 0, "train_sales is empty")
    _need(t_dates.max() < dates.min(),
          f"train_sales must end before the period starts (it ends {t_dates.max().date()}, "
          f"the period starts {dates.min().date()})")
    _need(set(pred_df["item"].unique()) <= set(train_sales["item"].unique()),
          "train_sales does not contain every item of pred_df")

    # - simulation table: for order cost per item
    _need(sim_table is not None,
          "sim_table is required (the long order-cost table: cost_ratio, setting, method, scope, total_cost)")
    sim_cols = ["cost_ratio", "setting", "method", "scope", "total_cost", "item_weeks"]
    miss_sim = [c for c in sim_cols if c not in sim_table.columns]
    _need(not miss_sim, f"sim_table is missing the columns {miss_sim}")
    _need(isinstance(cost_ratio, str) and cost_ratio != "",
          "cost_ratio is required, for example '3:1' (the ratio of the headline comparison)")
    _need(isinstance(sim_setting, str) and sim_setting != "",
          "sim_setting is required, for example 'tuned_buffer'")
    _need(cost_ratio in set(sim_table["cost_ratio"]),
          f"cost_ratio {cost_ratio!r} is not in sim_table (it has {sorted(set(sim_table['cost_ratio']))})")
    _need(sim_setting in set(sim_table["setting"]),
          f"sim_setting {sim_setting!r} is not in sim_table (it has {sorted(set(sim_table['setting']))})")


def _order_costs_per_item(sim_table, cost_ratio, sim_setting, methods, items, n_item_weeks_per_item):
    """Order cost per item and method, checked."""
    s = sim_table[(sim_table["cost_ratio"] == cost_ratio) & (sim_table["setting"] == sim_setting)]
    by_item = s[s["scope"].str.startswith("item_")].copy()
    by_item["item"] = by_item["scope"].str.replace("item_", "", regex=False).astype(int)
    costs = by_item.pivot(index="item", columns="method", values="total_cost")
    miss_methods = [m for m in methods if m not in costs.columns]
    _need(not miss_methods, f"sim_table has no order cost for the methods {miss_methods} "
                            f"(ratio {cost_ratio}, setting {sim_setting})")
    _need(sorted(costs.index) == sorted(int(i) for i in items),
          f"sim_table covers the items {sorted(costs.index)} but pred_df has {sorted(int(i) for i in items)}")
    _need(not costs[methods].isna().any().any(), "sim_table has missing order costs")
    # - same weeks as the predictions
    weeks = by_item.groupby("item")["item_weeks"].first()
    _need((weeks == n_item_weeks_per_item).all(),
          f"sim_table has {sorted(set(weeks))} weeks per item but pred_df has {n_item_weeks_per_item} full weeks")
    # - item costs must sum to 'overall'
    overall = s[s["scope"] == "overall"].set_index("method")["total_cost"]
    for m in methods:
        _need(m in overall.index and abs(float(overall[m]) - float(costs[m].sum())) < 1e-6,
              f"sim_table: the item costs of {m} do not add up to its overall cost")
    return costs[methods]


def run_failure_analysis(pred_df, period_label, model_cols, baseline_cols, out_dir,
                         headline_model, train_sales, sim_table, cost_ratio, sim_setting,
                         ridge_pipeline=None):
    """Where and why forecasts fail, any period; returns the summary (also saved).

    - all args required except ridge_pipeline
    - train_sales must end before the period
    """
    models, baselines = list(model_cols), list(baseline_cols)
    methods = baselines + models
    _check_inputs(pred_df, period_label, models, baselines, headline_model, train_sales,
                  sim_table, cost_ratio, sim_setting)
    headline = headline_model

    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pred_df[["date", "item", "actual"] + methods].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    train_sales = train_sales[["date", "item", "sales"]].copy()
    train_sales["date"] = pd.to_datetime(train_sales["date"])

    mondays = _full_week_mondays(df)
    _need(len(mondays) >= 1, "the period has no full Monday-to-Sunday week")
    weekly = weekly_totals(df, mondays, methods)   # - same weekly definition as elsewhere

    # - overall scores; best baseline = lowest daily MAE
    overall = _overall_table(df, weekly, methods)
    best_baseline = min(baselines, key=lambda b: overall[b]["mae"])

    # - order cost per item, same ratio and setting
    sim_costs = _order_costs_per_item(sim_table, cost_ratio, sim_setting, methods,
                                      df["item"].unique(), len(mondays))

    worst = _worst_days_table(df, models, baselines)
    by_item = _by_item_table(df, weekly, methods, models, baselines, sim_costs)
    weekday_df = df.assign(weekday=df["date"].dt.day_of_week.map(lambda i: f"{i + 1} {WEEKDAYS[i]}"))
    month_df = df.assign(month=df["date"].dt.strftime("%Y-%m"))
    by_weekday = _by_group_table(weekday_df, "weekday", "weekday", models, best_baseline)
    by_month = _by_group_table(month_df, "month", "month", models, best_baseline)

    # - bias per item and month
    rows = []
    for (mon, item), g in month_df.groupby(["month", "item"]):
        for m in models:
            e = g[m] - g["actual"]
            rows.append({"month": mon, "item": int(item), "model": m, "mean_sales": _round(g["actual"].mean(), 2),
                         "bias": _round(e.mean()), "bias_pct": _round(100 * e.sum() / g["actual"].sum(), 2),
                         "mae": _round(e.abs().mean())})
    by_item_month = pd.DataFrame(rows)

    prefix = f"failure_{period_label}_"
    worst.to_csv(out_dir / f"{prefix}worst_days.csv", index=False)
    by_item.to_csv(out_dir / f"{prefix}by_item.csv", index=False)
    by_weekday.to_csv(out_dir / f"{prefix}by_weekday.csv", index=False)
    by_month.to_csv(out_dir / f"{prefix}by_month.csv", index=False)
    by_item_month.to_csv(out_dir / f"{prefix}by_item_month.csv", index=False)

    # - "why" checks
    noise = _noise_check(df, models, baselines)
    relations = _error_relationships(df, weekly, models, baselines, methods)
    scale = _calendar_scale_check(df, train_sales, models, ridge_pipeline)
    ceiling = _ceiling_check(df, train_sales, models, baselines, best_baseline)
    wins = _simple_rule_wins(by_item, models, baselines, headline, True)

    # - examples by fixed rules, not picked from plots
    _style()
    # - worst item-week per model; headline file is plain "example"
    examples = {}
    for m in [headline] + [x for x in models if x != headline]:
        item, week = _pick_worst_item_week(weekly, m)
        name = f"{prefix}example.png" if m == headline else f"{prefix}example_{m}.png"
        examples[m] = {"file": name, **_fig_example(df, weekly, models, baselines, m, period_label,
                                                    fig_dir, name, item, week)}
    # - item where a rule beats each model most (if any)
    rule_examples = {}
    for m in models:
        pick = _pick_rule_beats_item(weekly, baselines, m)
        if pick is None:
            rule_examples[m] = None
            continue
        name = f"{prefix}rule_beats_{m}.png"
        rule_examples[m] = {"file": name, **_fig_example(df, weekly, models, baselines, m, period_label,
                                                         fig_dir, name, pick[0])}
    _fig_by_item(by_item, models, baselines, best_baseline, period_label, fig_dir)
    _fig_bias(df, scale, models, baselines, by_item, period_label, fig_dir)
    _fig_noise(df, models, period_label, fig_dir)
    _fig_ceiling(df, train_sales, models, ceiling, period_label, fig_dir)

    summary = {
        "period_label": period_label,
        "date_range": [df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d")],
        "n_daily_rows": int(len(df)),
        "n_items": int(df["item"].nunique()),
        "n_full_weeks": int(len(mondays)),
        "n_item_weeks": int(len(weekly)),
        "sign_convention": "error = forecast minus actual; positive = over-forecast",
        "models": models,
        "baselines": baselines,
        "headline_model": headline,
        "best_baseline_by_overall_daily_mae": best_baseline,
        "cost_ratio_used_for_order_cost": cost_ratio,
        "sim_setting_used_for_order_cost": sim_setting,
        "train_sales_range": [train_sales["date"].min().strftime("%Y-%m-%d"),
                              train_sales["date"].max().strftime("%Y-%m-%d")],
        "ridge_calendar_coefficients_available": ridge_pipeline is not None,
        "overall": overall,
        "example_figure": examples[headline],
        "example_figures_other_models": {m: v for m, v in examples.items() if m != headline},
        # - None = no rule beat that model
        "rule_beats_model_figures": rule_examples,
        "noise_check": noise,
        "error_relationships": relations,
        "calendar_scale_check": scale,
        "ceiling_check": ceiling,
        "simple_rule_wins": wins,
        "by_item": by_item.to_dict("records"),
        "order_cost_total_sum_of_items": {m: _round(sim_costs[m].sum(), 1) for m in methods},
        "by_item_ranking": by_item[["item", "mean_sales", "best_method_daily_mae", "best_method_weekly_mae",
                                    "best_method_order_cost"]].to_dict("records"),
        "by_weekday_models_vs_best_baseline": by_weekday.to_dict("records"),
        "by_month_models_vs_best_baseline": by_month.to_dict("records"),
        "worst_days_files": f"{prefix}worst_days.csv",
    }
    with open(out_dir / f"{prefix}summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return summary


def run_failure_analysis_for_test(final_result, train_sales, headline_model, cost_ratio, sim_setting,
                                  out_dir, ridge_pipeline=None):
    """Same analysis for the test period, from run_final_test(confirm=True).

    - loads no data; checks the dict, then calls run_failure_analysis
    """
    _need(isinstance(final_result, dict), "final_result must be the dictionary returned by run_final_test")
    missing = [k for k in ["metrics", "costs", "predictions", "cost_table"] if k not in final_result]
    _need(not missing, f"final_result is missing the keys {missing}")
    pred, metrics, costs = final_result["predictions"], final_result["metrics"], final_result["costs"]
    for where, name in [(metrics, "metrics"), (costs, "costs")]:
        _need(isinstance(where, dict) and where.get("headline_model") == headline_model,
              f"final_result['{name}'] names the headline model "
              f"{where.get('headline_model') if isinstance(where, dict) else None!r}, "
              f"but headline_model is {headline_model!r}")
    _need(len(pred) == metrics.get("n_daily_rows"),
          f"final_result['predictions'] has {len(pred)} rows but the metrics say {metrics.get('n_daily_rows')}")
    return run_failure_analysis(
        pred_df=pred, period_label="test", model_cols=MODELS, baseline_cols=BASELINES, out_dir=out_dir,
        headline_model=headline_model, train_sales=train_sales, sim_table=final_result["cost_table"],
        cost_ratio=cost_ratio, sim_setting=sim_setting, ridge_pipeline=ridge_pipeline)
