"""A1.6 — text-vs-stats disagreement (Canon Detector / FEAT-002).

Fits two probes on the SAME stratified split:

* **stats probe** — the 13 structural features only (powerstats + encoded
  substance categoricals).
* **text probe** — the 300 TF-IDF features of about/abilities prose only.

Then finds the characters where the two probes' argmax universes disagree most
(ranked by combined confidence of the disagreement). This is the finding only the
text-capable setup produces: a character whose *numbers* read like one universe
but whose *story* reads like another — a built-in crossover signal.

Call budget
-----------
Exactly FOUR logged TabPFN calls: two fits (stats, text) + two batched
``predict_proba`` over the FULL dataset (never per-row). Both probes are scored on
every character so the disagreement table can rank the whole roster.

Output
------
``results/text_vs_stats.csv`` — character, publisher, stats_argmax, stats_conf,
text_argmax, text_conf, disagree (bool), disagreement_score — sorted so the
strongest disagreements are on top.

Run::

    python src/axis1_text_vs_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.axis1_classifier import (  # noqa: E402
    TABPFN_MODEL_PATH,
    TABPFN_N_ESTIMATORS,
    _build_matrices,
)
from src.credits import log_call  # noqa: E402
from src.dataprep import RANDOM_STATE  # noqa: E402
from src.env import load_token  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_CSV_PATH = _RESULTS_DIR / "text_vs_stats.csv"


def _fit_score(X_train, y_train, X_all, label):
    from tabpfn_client import TabPFNClassifier

    clf = TabPFNClassifier(
        model_path=TABPFN_MODEL_PATH,
        n_estimators=TABPFN_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    with log_call(f"axis1_tvs.fit.{label}"):
        clf.fit(X_train, y_train)
    with log_call(f"axis1_tvs.predict_proba.{label}"):
        proba = clf.predict_proba(X_all)  # ONE batched call over all rows
    classes = np.array([str(c) for c in clf.classes_])
    return proba, classes


def main() -> int:
    load_token()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A1.6 — text-vs-stats disagreement (two probes, same split)")
    print("=" * 70)

    data = _build_matrices()
    frame = data["frame"]
    X_struct_all = data["X_struct_all"]
    X_text_all = data["X_text_all"]
    y = data["y"]
    train_idx = data["train_idx"]
    y_train = y.to_numpy()[train_idx]

    names = frame.df["name"].astype(str).to_numpy()
    pubs = frame.df["publisher"].astype(str).to_numpy()

    # --- Stats-only probe ----------------------------------------------------
    stats_proba, stats_classes = _fit_score(
        X_struct_all[train_idx], y_train, X_struct_all, "stats"
    )
    # --- Text-only probe -----------------------------------------------------
    text_proba, text_classes = _fit_score(
        X_text_all[train_idx], y_train, X_text_all, "text"
    )

    stats_argmax = stats_classes[stats_proba.argmax(axis=1)]
    stats_conf = stats_proba.max(axis=1)
    text_argmax = text_classes[text_proba.argmax(axis=1)]
    text_conf = text_proba.max(axis=1)

    disagree = stats_argmax != text_argmax
    # Rank disagreements by how confidently each probe asserts its (different) call.
    disagreement_score = np.where(disagree, stats_conf * text_conf, 0.0)

    out = pd.DataFrame(
        {
            "character": names,
            "publisher": pubs,
            "stats_argmax": stats_argmax,
            "stats_conf": stats_conf.round(4),
            "text_argmax": text_argmax,
            "text_conf": text_conf.round(4),
            "disagree": disagree,
            "disagreement_score": disagreement_score.round(4),
        }
    ).sort_values(
        ["disagree", "disagreement_score"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)

    out.to_csv(_CSV_PATH, index=False)
    n_disagree = int(disagree.sum())
    print(f"Total characters: {len(out)}  |  disagreements: {n_disagree} "
          f"({n_disagree / len(out):.1%})")
    print(f"Wrote text-vs-stats table -> {_CSV_PATH}")

    print("\nTop 15 stats-vs-story disagreements (numbers say X, prose says Y):")
    for _, r in out.head(15).iterrows():
        print(
            f"  {r['character']:>26} ({r['publisher']:>10}): "
            f"stats->{r['stats_argmax']:<10} ({r['stats_conf']:.2f})  "
            f"text->{r['text_argmax']:<10} ({r['text_conf']:.2f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
