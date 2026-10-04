"""A2.4 — novelty + coherence validation of generated characters (FEAT-003).

Every generated character (agent winners + gap character) is checked on two axes:

* **Novelty** — Euclidean nearest-neighbour distance, in standardized structural
  feature space, from the generated character to the nearest REAL training
  character. A distance of ~0 would mean the "generated" character is a copy; we
  assert none is an exact duplicate and report min/median NN distance.
* **Coherence** — are the stats sane? Each of the 6 powerstats must be within
  [0, 100], the total within a plausible band, and no absurd single-stat spike
  with everything else floored. A character failing these is flagged
  ``coherent=False`` (kept in the CSV, not deleted — the flag is the finding).

Standardization uses the REAL characters' mean/std so the NN distance is in
comparable units across features (otherwise the powerstats would dominate the
raw-space distance). No TabPFN call is needed here (pure geometry — free).

Output
------
Adds ``nn_distance``, ``nn_neighbour`` (name of closest real char), ``novel`` and
``coherent`` columns to ``results/generated_characters.csv``.

Run::

    python src/axis2_validate.py

No unit tests (project policy): running this script IS the verification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataprep import KEEP_NUMERIC, load_frame  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_GEN_CSV = _RESULTS_DIR / "generated_characters.csv"

# Coherence bounds.
STAT_MIN, STAT_MAX = 0, 100
TOTAL_MIN, TOTAL_MAX = 60, 600          # 6 stats * [10..100] plausible band
NOVELTY_MIN_DISTANCE = 1e-6             # below this = effectively a copy


def main() -> int:
    if not _GEN_CSV.exists():
        raise FileNotFoundError(
            f"{_GEN_CSV} missing — run src/axis2_agent.py (A2.2) first."
        )
    print("=" * 70)
    print("A2.4 — novelty (NN distance) + coherence validation")
    print("=" * 70)

    frame = load_frame()
    struct_cols = list(frame.X_struct.columns)
    X_real = frame.X_struct.to_numpy(dtype=np.float64)
    real_names = frame.df["name"].astype(str).to_numpy()

    scaler = StandardScaler().fit(X_real)
    X_real_std = scaler.transform(X_real)

    gen = pd.read_csv(_GEN_CSV)
    X_gen = gen[struct_cols].to_numpy(dtype=np.float64)
    X_gen_std = scaler.transform(X_gen)

    # --- Novelty: NN distance to the nearest REAL character ------------------
    d2 = ((X_gen_std[:, None, :] - X_real_std[None, :, :]) ** 2).sum(axis=2)
    nn_idx = d2.argmin(axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(X_gen_std)), nn_idx])
    nn_name = real_names[nn_idx]

    gen["nn_distance"] = nn_dist.round(4)
    gen["nn_neighbour"] = nn_name
    gen["novel"] = nn_dist > NOVELTY_MIN_DISTANCE

    # --- Coherence: sane stat ranges -----------------------------------------
    stats = gen[KEEP_NUMERIC].to_numpy(dtype=float)
    in_range = ((stats >= STAT_MIN) & (stats <= STAT_MAX)).all(axis=1)
    totals = stats.sum(axis=1)
    total_ok = (totals >= TOTAL_MIN) & (totals <= TOTAL_MAX)
    # Absurd combo: one stat maxed (>=95) while every other stat is floored (<=5).
    maxed = (stats >= 95).sum(axis=1)
    floored = (stats <= 5).sum(axis=1)
    not_absurd = ~((maxed >= 1) & (floored >= 5))
    gen["coherent"] = in_range & total_ok & not_absurd

    gen.to_csv(_GEN_CSV, index=False)

    # --- Report + assertions -------------------------------------------------
    print(f"Generated characters: {len(gen)}")
    print(f"NN distance (standardized): min={nn_dist.min():.4f}  "
          f"median={np.median(nn_dist):.4f}  max={nn_dist.max():.4f}")
    assert (nn_dist > NOVELTY_MIN_DISTANCE).all(), (
        "A generated character is an EXACT duplicate of a training row!"
    )
    print("Novelty assertion OK — no generated character is an exact training copy.")
    print(f"Coherent: {int(gen['coherent'].sum())}/{len(gen)}")
    print()
    for _, r in gen.iterrows():
        stat_vals = [int(r[c]) for c in KEEP_NUMERIC]
        print(
            f"  [{str(r['objective']):>14}] stats={stat_vals} total={sum(stat_vals)}  "
            f"NN={r['nn_distance']:.3f} (nearest real: {r['nn_neighbour']})  "
            f"novel={bool(r['novel'])} coherent={bool(r['coherent'])}"
        )
    print(f"\nUpdated {_GEN_CSV} with nn_distance, nn_neighbour, novel, coherent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
