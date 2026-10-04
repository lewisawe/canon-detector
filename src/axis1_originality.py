"""A1.4 — originality ranking (Canon Detector / FEAT-002).

``originality = 1 - max_u P(u)``: a character the probe is *unsure* about (low max
probability, spread across universes) sits between archetypes = original; a
character the probe nails with near-certainty is archetypal for its universe.

Reuses the TRAIN probabilities persisted by A1.1 (``results/_axis1_test_probs.npz``)
so NO new TabPFN call is made — the ranking covers every TRAIN character already
scored by the gate model.

Outputs
-------
* ``results/originality_ranking.csv`` — name, publisher, max_prob, originality,
  argmax_universe (sorted most-original first).
* ``originality_ranking.png`` — Agg top/bottom-N bar chart.

Run::

    python src/axis1_originality.py
"""

from __future__ import annotations

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

from src.dataprep import load_frame  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_PROBS_PATH = _RESULTS_DIR / "_axis1_test_probs.npz"
_CSV_PATH = _RESULTS_DIR / "originality_ranking.csv"
_PNG_PATH = _ROOT / "originality_ranking.png"

TOP_N = 15  # for the bar chart extremes


def main() -> int:
    if not _PROBS_PATH.exists():
        raise FileNotFoundError(
            f"{_PROBS_PATH} missing — run src/axis1_classifier.py (A1.1) first."
        )
    data = np.load(_PROBS_PATH, allow_pickle=True)
    train_proba = data["train_proba"]
    train_idx = data["train_idx"]
    classes = data["tabpfn_classes"].astype(str)

    print("=" * 70)
    print("A1.4 — originality ranking (reuses A1.1 TRAIN probs; no TabPFN call)")
    print("=" * 70)

    # Map train rows back to names/publisher via the canonical frame.
    frame = load_frame()
    names = frame.df["name"].astype(str).to_numpy()
    pubs = frame.df["publisher"].astype(str).to_numpy()

    max_prob = train_proba.max(axis=1)
    originality = 1.0 - max_prob
    argmax_universe = classes[train_proba.argmax(axis=1)]

    rank = pd.DataFrame(
        {
            "name": names[train_idx],
            "publisher": pubs[train_idx],
            "max_prob": max_prob,
            "originality": originality,
            "argmax_universe": argmax_universe,
        }
    ).sort_values("originality", ascending=False, kind="stable").reset_index(drop=True)

    assert rank["originality"].between(0.0, 1.0).all(), "originality out of [0,1]"
    rank.to_csv(_CSV_PATH, index=False)
    print(f"Wrote originality ranking ({len(rank)} chars) -> {_CSV_PATH}")
    print(f"originality range: [{rank['originality'].min():.4f}, {rank['originality'].max():.4f}]")

    print("\nMost ORIGINAL (between archetypes):")
    for _, r in rank.head(10).iterrows():
        print(f"  {r['originality']:.3f}  {r['name']:>26} ({r['publisher']}) ~ {r['argmax_universe']}")
    print("Most ARCHETYPAL (probe is certain):")
    for _, r in rank.tail(10).iloc[::-1].iterrows():
        print(f"  {r['originality']:.3f}  {r['name']:>26} ({r['publisher']}) ~ {r['argmax_universe']}")

    # --- Bar chart: top-N original + top-N archetypal ------------------------
    top = rank.head(TOP_N)
    bot = rank.tail(TOP_N).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.42 * TOP_N + 1.5))

    ax1.barh(range(len(top)), top["originality"], color="#48c")
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels([f"{n} ({p})" for n, p in zip(top["name"], top["publisher"])], fontsize=7)
    ax1.invert_yaxis()
    ax1.set_xlabel("originality = 1 - max P(u)")
    ax1.set_title(f"Top {TOP_N} most ORIGINAL")
    ax1.set_xlim(0, 1)

    ax2.barh(range(len(bot)), bot["originality"], color="#c66")
    ax2.set_yticks(range(len(bot)))
    ax2.set_yticklabels([f"{n} ({p})" for n, p in zip(bot["name"], bot["publisher"])], fontsize=7)
    ax2.invert_yaxis()
    ax2.set_xlabel("originality = 1 - max P(u)")
    ax2.set_title(f"Top {TOP_N} most ARCHETYPAL")
    ax2.set_xlim(0, 1)

    fig.suptitle("Axis-1 originality (TRAIN characters)")
    fig.tight_layout()
    fig.savefig(_PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"Wrote bar chart -> {_PNG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
