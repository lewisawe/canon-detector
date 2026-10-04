"""A1.3 — famous-character cross-universe plausibility table (FEAT-002).

Picks 15-20 recognizable characters that are actually in the dataset and scores
them cross-universe. "Plausibility" = the probe's calibrated P(universe) for each
character, i.e. how plausibly each famous character could belong to every universe
given only their substance (stats + categoricals + about/abilities text).

Call budget
-----------
The A1.1 fitted estimator is not a persistable object across processes with the
TabPFN client, so we refit the SAME ``TabPFNClassifier`` on the SAME TRAIN split
(one flat fit) and score ALL selected characters in ONE batched ``predict_proba``.
That is exactly two logged TabPFN calls (one fit, one batched predict) — never a
per-character loop.

Calibration
-----------
The A1.2 "best calibrator" is top-label (operates on argmax confidence), so it
cannot remap a full per-universe probability vector without breaking row-sum=1.
We therefore report the raw per-universe probabilities (which A1.2 showed are
already well-calibrated, ECE≈0.017) and record in the CSV header note which
calibrator A1.2 selected. Rows still sum to ~1.0.

Outputs
-------
* ``results/plausibility_table.csv`` — rows = characters, cols = per-universe
  plausibilities + ``argmax_universe`` + ``true_publisher``.
* ``plausibility_heatmap.png`` — Agg heatmap.

Run::

    python src/axis1_plausibility.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

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
_CSV_PATH = _RESULTS_DIR / "plausibility_table.csv"
_PNG_PATH = _ROOT / "plausibility_heatmap.png"
_JSON_PATH = _RESULTS_DIR / "axis1_classifier.json"

# Recognizable characters confirmed present in the dataset (exact 'name' match).
FAMOUS_NAMES = [
    "Batman",
    "Superman",
    "Wonder Woman",
    "Spider-Man",
    "Deadpool",
    "Iron Man",
    "Captain America",
    "Thor",
    "Hulk",
    "Wolverine",
    "Goku",
    "Vegeta",
    "Saitama",
    "Ichigo",
    "Light Yagami",
    "Homelander",
    "Starlight",
    "Spock",
    "Data",
    "Darth Vader",
    "Luke Skywalker",
    "Hellboy",
]


def main() -> int:
    load_token()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A1.3 — famous-character cross-universe plausibility")
    print("=" * 70)

    data = _build_matrices()
    frame = data["frame"]
    X_all = data["X_all"]
    y = data["y"]
    train_idx = data["train_idx"]

    names = frame.df["name"].astype(str)
    # Resolve selected names -> row positions (first match each; keep order/dedupe).
    name_to_pos: dict[str, int] = {}
    for nm in FAMOUS_NAMES:
        hits = np.where(names.str.lower().to_numpy() == nm.lower())[0]
        if len(hits):
            name_to_pos[nm] = int(hits[0])
    sel_names = list(name_to_pos.keys())
    sel_pos = np.array([name_to_pos[n] for n in sel_names])
    print(f"Selected {len(sel_names)} famous characters present in the dataset.")
    if not (15 <= len(sel_names) <= 25):
        print(f"WARNING: expected 15-20 characters, resolved {len(sel_names)}.")

    X_train = X_all[train_idx]
    y_train = y.to_numpy()[train_idx]
    X_sel = X_all[sel_pos]

    from tabpfn_client import TabPFNClassifier

    clf = TabPFNClassifier(
        model_path=TABPFN_MODEL_PATH,
        n_estimators=TABPFN_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    with log_call("axis1_plausibility.fit"):
        clf.fit(X_train, y_train)
    with log_call("axis1_plausibility.predict_proba.famous"):
        proba = clf.predict_proba(X_sel)  # ONE batched call for all characters

    classes = [str(c) for c in clf.classes_]
    argmax_universe = [classes[i] for i in proba.argmax(axis=1)]
    true_pub = frame.df["publisher"].astype(str).to_numpy()[sel_pos]

    table = pd.DataFrame(proba, columns=classes, index=sel_names)
    table.insert(0, "true_publisher", true_pub)
    table["argmax_universe"] = argmax_universe
    table["row_sum"] = proba.sum(axis=1)
    table.index.name = "character"
    table.to_csv(_CSV_PATH)
    print(f"Wrote plausibility table -> {_CSV_PATH}")
    print(f"Row sums: min={table['row_sum'].min():.4f} max={table['row_sum'].max():.4f}")

    # Which calibrator A1.2 chose (reported, not applied to full vectors).
    best_cal = "unknown"
    if _JSON_PATH.exists():
        with _JSON_PATH.open(encoding="utf-8") as fh:
            j = json.load(fh)
        best_cal = j.get("calibration", {}).get("best_calibrator", "unknown")
    print(f"A1.2 best calibrator (top-label): {best_cal}; raw per-universe probs reported.")

    # --- Heatmap -------------------------------------------------------------
    fig_h = max(4.0, 0.42 * len(sel_names))
    fig, ax = plt.subplots(figsize=(1.1 * len(classes) + 2, fig_h))
    im = ax.imshow(proba, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sel_names)))
    ax.set_yticklabels(
        [f"{n} ({p})" for n, p in zip(sel_names, true_pub)], fontsize=8
    )
    ax.set_title("Cross-universe plausibility P(universe | substance)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025)
    cbar.set_label("plausibility", fontsize=8)
    # annotate argmax cell
    for r in range(len(sel_names)):
        c = int(proba[r].argmax())
        ax.text(c, r, "\u25cf", ha="center", va="center", color="#6cf", fontsize=7)
    fig.tight_layout()
    fig.savefig(_PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"Wrote heatmap -> {_PNG_PATH}")

    # Quick face-validity print
    print("\nFace validity (character: argmax vs true):")
    for n in sel_names:
        print(f"  {n:>16}: argmax={table.loc[n,'argmax_universe']:>12}  true={table.loc[n,'true_publisher']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
