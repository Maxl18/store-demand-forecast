"""Guard: safety checks and append-only run log for the test period.

- require(): unlike assert, survives python -O
- each log entry stores the previous entry's SHA-256: edits and deletions show
- test data needs a FinalRunTicket, valid only while a run is open in the log
"""

import datetime
import hashlib
import json
import os
from pathlib import Path

LOG_NAME = "final_test_log.json"
LOG_VERSION = 1

# - these close a run and end its ticket
CLOSING_EVENTS = {"finished", "failed", "output_mismatch", "abandoned"}


class CheckFailed(AssertionError):
    """Failed safety check (explicit raise, survives python -O)."""


def require(condition, message):
    """Raise CheckFailed(message) unless condition."""
    if not condition:
        raise CheckFailed(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest().upper()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def _entry_fingerprint(entry):
    """SHA-256 of one entry (sorted keys)."""
    return sha256_bytes(json.dumps(entry, sort_keys=True).encode("utf-8"))


def log_path(results_dir):
    return Path(results_dir) / LOG_NAME


def read_log(results_dir):
    """Log entries ([] if no log).

    - checks the fingerprint chain; edited or deleted entries stop the program
    """
    path = log_path(results_dir)
    if not path.exists():
        return []
    content = json.loads(path.read_text(encoding="utf-8"))
    require(content.get("log_version") == LOG_VERSION, f"{path}: unknown log format")
    entries = content["entries"]
    previous = None
    for i, entry in enumerate(entries):
        require(entry.get("seq") == i + 1, f"{path}: entry {i + 1} has the wrong number; the log was edited")
        expected = None if previous is None else _entry_fingerprint(previous)
        require(entry.get("previous_entry_sha256") == expected,
                f"{path}: entry {i + 1} does not match the entry before it; the log was edited "
                f"or an entry was deleted")
        previous = entry
    return entries


def append_log(results_dir, entry):
    """Append one entry; returns it."""
    entries = read_log(results_dir)
    full = {
        "seq": len(entries) + 1,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        **entry,
        "previous_entry_sha256": _entry_fingerprint(entries[-1]) if entries else None,
    }
    content = {
        "log_version": LOG_VERSION,
        "note": ("Append-only log of every attempt to run the final test. Entries are never "
                 "deleted or edited; each one stores the fingerprint of the one before it."),
        "entries": entries + [full],
    }
    path = log_path(results_dir)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(content, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # - atomic: no half-written log
    return full


def last_entry_fingerprint(results_dir):
    """SHA-256 of the newest entry (None if no log).

    - a log rewritten from scratch is caught only by comparing this value
    """
    entries = read_log(results_dir)
    return _entry_fingerprint(entries[-1]) if entries else None


def run_is_open(entries, run_number):
    """True if the run is started and not closed."""
    started = any(e["event"] == "started" and e.get("run") == run_number for e in entries)
    closed = any(e["event"] in CLOSING_EVENTS and e.get("run") == run_number for e in entries)
    return started and not closed


class FinalRunTicket:
    """Permission for one test run, tied to a results folder.

    - created only while the run is open in that folder's log
    """

    def __init__(self, results_dir, run_number):
        self.results_dir = Path(results_dir).resolve()
        self.run_number = run_number
        require(run_is_open(read_log(self.results_dir), run_number),
                f"no open 'started' entry for run {run_number} in {log_path(self.results_dir)}; "
                f"a ticket can only be created by run_final_test after it has logged the start")

    @property
    def buffers_path(self):
        return self.results_dir / "buffers.json"

    @property
    def scope_items(self):
        scope = json.loads((self.results_dir / "scope.json").read_text())
        return sorted(int(i) for i in scope["items"])


def require_active_ticket(ticket, what):
    """Stop unless `ticket` is valid and its run still open."""
    require(isinstance(ticket, FinalRunTicket),
            f"{what} touches the test period (2017-10-01 onward). Only the guarded entry point "
            f"run_final_test.run_final_test(confirm=True) may do that.")
    require(run_is_open(read_log(ticket.results_dir), ticket.run_number),
            f"{what}: the ticket for run {ticket.run_number} is not open in the run log")
