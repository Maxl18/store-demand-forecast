"""Failure analysis on the test period.

- input: result of run_final_test(confirm=True); never calls it itself
- no fitted Ridge there, so the Ridge calendar check is skipped
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import SIM_SETTING, run_failure_analysis_for_test  # noqa: E402
from forecast import RESULTS, VAL_END, load_scope  # noqa: E402
from headline import check_headline_consistency, load_headline_record  # noqa: E402


def run_failure_test(final_result, out_dir=RESULTS, train_sales=None):
    """Test-period failure analysis; returns the summary (also saved as JSON)."""
    check_headline_consistency()
    record = load_headline_record()
    if train_sales is None:
        train_sales, _ = load_scope(VAL_END)     # - up to 2017-09-30, no test rows
    return run_failure_analysis_for_test(
        final_result=final_result,
        train_sales=train_sales,
        headline_model=record["headline_model"],
        cost_ratio=record["headline_ratio"],
        sim_setting=SIM_SETTING,
        out_dir=out_dir,
    )
