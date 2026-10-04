"""Shared TabPFN credit-logging helper (Canon Detector / FEAT-001).

Every TabPFN-calling script in this project wraps each fit / predict / generate
call in :func:`log_call` so credit usage is tracked in ONE place. TabPFN billing
is flat (~10,000 credits per fit+predict call regardless of row count), so each
wrapped block is expected to show a step-change in usage, and a per-row predict
loop would show up here as many expensive calls.

The source of truth for usage is ``tabpfn_client.get_api_usage()``, which (in
tabpfn_client 0.6.1) returns a human-readable string of the form::

    "Currently, you have used 950000 of the allowed limit of 20000000 credits. ..."

We parse the first integer as the credits-used counter and log the before/after
delta per wrapped call to ``results/credit_log.csv``.

Run modes
---------
``python src/credits.py --selftest``
    Perform ONE wrapped ``get_api_usage()`` call (loading the token via
    :mod:`src.env`), print the current usage + running tally, and append a row
    to ``results/credit_log.csv``.

No unit tests: ``--selftest`` IS the verification (per project policy).
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import tabpfn_client

# results/ lives at the worktree root, one level up from src/.
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_CREDIT_LOG = _RESULTS_DIR / "credit_log.csv"
_LOG_COLUMNS = ["ts", "label", "before", "after", "delta", "running_total"]


def get_usage() -> int:
    """Return current credits-used as an int by parsing ``get_api_usage()``.

    tabpfn_client.get_api_usage() returns a sentence like
    "Currently, you have used 950000 of the allowed limit of 20000000 credits."
    We extract the first integer (credits used so far). Raises if the response
    cannot be parsed, so callers fail loudly rather than logging bogus deltas.
    """
    raw = tabpfn_client.get_api_usage()
    match = re.search(r"used\s+([\d,]+)", str(raw))
    if match is None:
        # Fall back to the first integer anywhere in the string.
        match = re.search(r"(\d[\d,]*)", str(raw))
    if match is None:
        raise RuntimeError(
            f"Could not parse credits from get_api_usage() response: {raw!r}"
        )
    return int(match.group(1).replace(",", ""))


def _append_log_row(row: dict) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not _CREDIT_LOG.exists()
    with _CREDIT_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@contextmanager
def log_call(label: str) -> Iterator[None]:
    """Context manager wrapping a single TabPFN call for credit accounting.

    Records usage before and after the wrapped block, prints the delta and a
    running tally, and appends a row to ``results/credit_log.csv`` with columns
    ``ts,label,before,after,delta,running_total``. ``running_total`` is the
    post-call absolute credits-used figure (the authoritative cumulative count),
    so the log doubles as an audit trail of total spend.

    Usage::

        with log_call("axis1.fit"):
            clf.fit(X_train, y_train)
    """
    before = get_usage()
    try:
        yield
    finally:
        after = get_usage()
        delta = after - before
        _append_log_row(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "label": label,
                "before": before,
                "after": after,
                "delta": delta,
                "running_total": after,
            }
        )
        print(
            f"[credits] {label}: +{delta} this call "
            f"(before={before}, after={after}, total_used={after})"
        )


def _selftest() -> int:
    """One wrapped get_api_usage() call; print tally; append a log row."""
    # Local import to keep src.credits importable without env side effects.
    try:
        from src.env import load_token
    except ImportError:
        # Allow running as `python src/credits.py` from the worktree root.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.env import load_token

    load_token()
    with log_call("credits.selftest"):
        current = get_usage()
    print(f"[credits] selftest OK — current credits used: {current}")
    print(f"[credits] log file: {_CREDIT_LOG}")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
    print("Run with --selftest to perform a wrapped usage check.")
