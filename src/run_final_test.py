"""Only entry point for the final test (2017-10-01 to 2017-12-31).

- guard: fingerprints frozen inputs before reading test data; refuses (logged) if changed
- first finished run sets the reference outputs
- later runs must recompute identical bytes, else output_mismatch
- use run_final_test(confirm=True) or `python src/run_final_test.py --confirm`
"""

import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import make_features  # noqa: E402
from forecast import (  # noqa: E402
    METHODS,
    MODELS,
    RESULTS,
    ROOT,
    TEST_END,
    TEST_START,
    TEST_WEEK_MONDAYS,
    VAL_END,
    forecast_period,
    load_scope,
    score,
)
from guard import (  # noqa: E402
    CheckFailed,
    FinalRunTicket,
    append_log,
    last_entry_fingerprint,
    log_path,
    read_log,
    require,
    run_is_open,
    sha256_bytes,
    sha256_file,
)
from headline import check_headline_consistency, load_headline_record  # noqa: E402
from simulate import HEADLINE_RATIO, load_buffers, simulate_test_period  # noqa: E402

# - everything that decides the test numbers; fingerprinted every run
FROZEN_INPUTS = [
    "results/scope.json",
    "results/buffers.json",
    "results/headline_model.json",
    "results/validation_metrics.json",
    "results/validation_predictions.csv",
    "src/data_prep.py",
    "src/features.py",
    "src/forecast.py",
    "src/simulate.py",
    "src/guard.py",
    "src/headline.py",
    "src/run_final_test.py",
    "src/failure_analysis.py",
    "src/run_failure_validation.py",
    "src/run_failure_test.py",
    "train.csv",                          # - hashed only, not parsed here
    "requirements.txt",
]

# - written by the first finished run, checked on later runs
OUTPUT_FILES = ["test_predictions.csv", "test_metrics.json", "simulation_test.json", "simulation_test.csv"]

# - added to errors: no quiet fix and re-run
CRASH_RULE = ("CRASH RULE: if the first real run failed because of a bug in the code, do not patch the "
              "code and run again quietly. Stop and decide openly what to do. Any code change made "
              "after the test data was read must be disclosed in the README, and the run log is kept.")


def hash_frozen_inputs(root=ROOT):
    """SHA-256 per frozen input (None if missing)."""
    hashes = {}
    for rel in FROZEN_INPUTS:
        path = Path(root) / rel
        hashes[rel] = sha256_file(path) if path.exists() else None
    return hashes


def library_versions():
    import numpy
    import pandas
    import sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}


def _changed_keys(reference, current):
    return [k for k in sorted(set(reference) | set(current)) if reference.get(k) != current.get(k)]


def _first(entries, event):
    return next((e for e in entries if e["event"] == event), None)


def changed_since_first_run():
    """(changed inputs, changed outputs) vs the log; read only, no test data."""
    entries = read_log(RESULTS)
    started, reference = _first(entries, "started"), _first(entries, "finished")
    if started is None:
        return [], []
    inputs = _changed_keys(started["frozen_inputs"], hash_frozen_inputs())
    outputs = []
    if reference is not None:
        disk = {n: (sha256_file(RESULTS / n) if (RESULTS / n).exists() else None) for n in OUTPUT_FILES}
        outputs = _changed_keys(reference["outputs"], disk)
    return inputs, outputs


def status(results_dir=RESULTS):
    """Run log summary; read only, no test data."""
    entries = read_log(results_dir)
    if not entries:
        return "No final test run yet (no run log)."
    lines = [f"{e['seq']:3d}  {e['time']}  {e['event']:15s} run {e.get('run', '-')}" for e in entries]
    lines.append(f"fingerprint of the last log entry (record it outside the project): "
                 f"{last_entry_fingerprint(results_dir)}")
    return "\n".join(lines)


def _versus(value, baseline_value):
    diff = value - baseline_value
    return {"difference": diff, "pct_difference": 100 * diff / baseline_value}


def _compute(ticket):
    """Refit on train + validation; forecast, score and cost the test period.

    - returns (files {name: bytes}, payload)
    """
    record = load_headline_record()
    head, other = record["headline_model"], record["other_model"]
    fixed = record["comparison_baselines_fixed_from_validation"]

    # - test dates need the ticket
    df, _ = load_scope(TEST_END, ticket=ticket)
    items = sorted(int(i) for i in df["item"].unique())
    require(items == ticket.scope_items, f"loaded items {items} differ from scope.json")
    require(df["date"].max() == TEST_END, "the data must reach 2017-12-31")

    # - refit up to 2017-09-30, forecast test days
    feats = make_features(df)
    table, _ = forecast_period(feats, VAL_END, TEST_START, TEST_END)
    require(len(table) == 92 * len(items), "expected 92 test days for every item")
    require(table[METHODS].notna().all().all(), "every method needs a forecast on every test row")

    scores = score(table, TEST_WEEK_MONDAYS)
    metric_keys = {"daily_mae": "mae", "smape": "smape", "weekly_mae": "weekly_mae"}
    versus_fixed = {}
    for label, key in metric_keys.items():
        base = fixed[label]
        base_value = scores["overall"][base][key]
        versus_fixed[label] = {"baseline": base, "baseline_value": base_value,
                               **{m: {"value": scores["overall"][m][key],
                                      **_versus(scores["overall"][m][key], base_value)} for m in MODELS}}
    metrics = {
        "period": f"{TEST_START.date()} .. {TEST_END.date()} (test)",
        "fitted_on": f"start of data .. {VAL_END.date()} (train + validation), frozen settings",
        "headline_model": head,
        "other_model": other,
        "headline_source": "results/headline_model.json",
        "overall": scores["overall"],
        "per_item": scores["per_item"],
        "n_daily_rows": scores["n_daily_rows"],
        "n_item_weeks": scores["n_item_weeks"],
        "models_vs_baseline_fixed_from_validation": versus_fixed,
        "note": ("Comparison baselines were fixed on validation before this run "
                 "(headline_model.json); every baseline is still reported in 'overall'."),
    }

    # - order costs: frozen buffers and no buffer, all ratios
    costs, cost_table = simulate_test_period(table, ticket)
    require(costs["headline_model"] == head, "buffers.json and headline_model.json disagree")
    costs["period"] = f"{TEST_START.date()} .. {TEST_END.date()} (test)"
    costs["headline_vs_best_baseline"] = {
        "note": (f"best baseline fixed on validation (buffers.json); ratio {HEADLINE_RATIO}, "
                 "frozen tuned buffers; negative = model cheaper"),
        head: costs["uncertainty"][HEADLINE_RATIO]["tuned_buffer"][head],
        other: costs["uncertainty"][HEADLINE_RATIO]["tuned_buffer"][other],
    }

    # - exact bytes for re-run comparison
    preds = table.copy()
    preds["date"] = preds["date"].dt.strftime("%Y-%m-%d")
    files = {
        "test_predictions.csv": preds.to_csv(index=False, lineterminator="\n").encode("utf-8"),
        "test_metrics.json": json.dumps(metrics, indent=2).encode("utf-8"),
        "simulation_test.json": json.dumps(costs, indent=2).encode("utf-8"),
        "simulation_test.csv": cost_table.to_csv(index=False, lineterminator="\n").encode("utf-8"),
    }
    payload = {"predictions": table, "metrics": metrics, "costs": costs, "cost_table": cost_table}
    return files, payload


def run_final_test(confirm=False):
    """Run the final test, or recompute and verify on later runs.

    - returns predictions, metrics, costs, cost_table, run info, log path
    """
    if confirm is not True:
        raise RuntimeError("run_final_test reads the test period. Call it as run_final_test(confirm=True).")

    entries = read_log(RESULTS)
    present = [n for n in OUTPUT_FILES if (RESULTS / n).exists()]
    if not entries and present:
        raise CheckFailed(f"test outputs {present} exist but the run log {log_path(RESULTS)} is missing "
                          f"or empty: the log was deleted. Not running.")

    # - checks before any test data is read
    versions = library_versions()
    started = [e for e in entries if e["event"] == "started"]
    finished = [e for e in entries if e["event"] == "finished"]
    problems = []
    # - cut log: outputs exist, no "finished"
    if present and not finished:
        problems.append(f"test outputs {present} exist but the log has no 'finished' run: the log was "
                        f"cut, or the outputs were put there by hand, or a run crashed while saving "
                        f"them. A person must decide what happened ({CRASH_RULE})")
    # - all finished runs must have the same output fingerprints
    if finished and any(e["outputs"] != finished[0]["outputs"] for e in finished):
        problems.append("the log holds 'finished' runs with different output fingerprints: the log was edited")
    inputs = hash_frozen_inputs()
    missing = [rel for rel, h in inputs.items() if h is None]
    if missing:
        problems.append(f"frozen inputs missing: {', '.join(missing)}")
    try:
        check_headline_consistency()
        load_buffers()
    except Exception as error:  # - failed pre-check = logged refusal
        problems.append(f"{type(error).__name__}: {error}")
    if started:
        first = started[0]
        changed = _changed_keys(first["frozen_inputs"], inputs)
        if changed:
            problems.append(f"frozen inputs changed since the first test run (run 1, started "
                            f"{first['time']}): {', '.join(changed)}. {CRASH_RULE}")
    if problems:
        append_log(RESULTS, {"event": "refused", "reasons": problems, "frozen_inputs": inputs,
                             "library_versions": versions,
                             "note": "refused before reading any test-period data"})
        raise CheckFailed("Final test run REFUSED (nothing from the test period was read; the attempt "
                          "is logged):\n  - " + "\n  - ".join(problems))

    # - started but never ended (killed): close as "abandoned", ticket dead
    for e in started:
        if run_is_open(read_log(RESULTS), e["run"]):
            append_log(RESULTS, {"event": "abandoned", "run": e["run"],
                                 "note": "started but no end was logged (process killed?); "
                                         "test-period data may have been read in that run"})

    run_number = len(started) + 1
    append_log(RESULTS, {"event": "started", "run": run_number, "first_run": run_number == 1,
                         "frozen_inputs": inputs, "library_versions": versions,
                         "headline_model": load_headline_record()["headline_model"]})
    ticket = FinalRunTicket(RESULTS, run_number)
    try:
        files, payload = _compute(ticket)
        new_hashes = {n: sha256_bytes(b) for n, b in files.items()}
        reference = _first(entries, "finished")
        if reference is None:
            # - first finished run = reference; never overwrites (would hide a cut log)
            existing = [n for n in OUTPUT_FILES if (RESULTS / n).exists()]
            require(not existing, f"outputs {existing} already exist but the log has no 'finished' "
                                  f"run: the log was cut. Nothing was overwritten.")
            # - temp files then rename: no half-written outputs
            for name, data in files.items():
                (RESULTS / (name + ".tmp")).write_bytes(data)
            for name in files:
                os.replace(RESULTS / (name + ".tmp"), RESULTS / name)
            append_log(RESULTS, {"event": "finished", "run": run_number, "outputs": new_hashes,
                                 "verified_against_run": None,
                                 "note": "first completed run; its outputs are the reference"})
        else:
            recomputed_differs = _changed_keys(reference["outputs"], new_hashes)
            on_disk = {n: sha256_file(RESULTS / n) for n in OUTPUT_FILES if (RESULTS / n).exists()}
            edited_on_disk = [n for n, h in on_disk.items() if h != reference["outputs"].get(n)]
            if recomputed_differs or edited_on_disk:
                ref_versions = next(e for e in entries if e["event"] == "started"
                                    and e["run"] == reference["run"])["library_versions"]
                append_log(RESULTS, {"event": "output_mismatch", "run": run_number,
                                     "recomputed_outputs_differ": recomputed_differs,
                                     "files_on_disk_differ": edited_on_disk, "outputs": new_hashes,
                                     "library_versions_reference_run": ref_versions,
                                     "library_versions_this_run": versions,
                                     "note": "reference files were left untouched"})
                raise CheckFailed(
                    f"Final test RECOMPUTE MISMATCH (logged): recomputed outputs that differ from run "
                    f"{reference['run']}: {recomputed_differs or 'none'}; files on disk that differ from "
                    f"the logged fingerprints: {edited_on_disk or 'none'}. Library versions then "
                    f"{ref_versions}, now {versions}. The first-run files were not overwritten.")
            restored = [n for n in OUTPUT_FILES if n not in on_disk]
            for name in restored:  # - restore deleted outputs, identical bytes
                (RESULTS / name).write_bytes(files[name])
            append_log(RESULTS, {"event": "finished", "run": run_number, "outputs": new_hashes,
                                 "verified_against_run": reference["run"], "restored_files": restored,
                                 "note": "deterministic recompute; outputs identical to the reference run"})
    except BaseException as error:
        if run_is_open(read_log(RESULTS), run_number):
            append_log(RESULTS, {"event": "failed", "run": run_number,
                                 "error": f"{type(error).__name__}: {error}"[:2000],
                                 "rule": CRASH_RULE})
        if not finished:  # - no finished run yet: crash rule
            error.add_note(CRASH_RULE)
        raise
    anchor = last_entry_fingerprint(RESULTS)
    print(f"Final test run {run_number} logged. Fingerprint of the last log entry "
          f"(keep a copy outside the project): {anchor}")
    payload.update({"run_number": run_number, "first_completed_run": reference is None,
                    "log": str(log_path(RESULTS)), "log_last_entry_sha256": anchor})
    return payload


if __name__ == "__main__":
    if "--confirm" in sys.argv[1:]:
        result = run_final_test(confirm=True)
        print(f"Run {result['run_number']} finished "
              f"({'first completed run' if result['first_completed_run'] else 'verified recompute'}).")
    print(status())
