"""A2.3 — sparse-region gap finder (FEAT-003).

"Which archetypes has fiction NOT made yet?" We project the structural character
space to 2-D, find the sparsest regions (far from all real characters / lowest
local density), then target synthetic generation INTO those gaps and keep the
candidate that lands closest to a gap. The result is a character occupying a part
of power/role space the training roster leaves empty.

Projection method (documented adaptation)
------------------------------------------
``tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel.get_embeddings`` is
**NotImplementedError** in extensions 0.6.3 (removed during the TabPFN refactor —
confirmed live). Per the plan's fallback, we use a dependency-free **PCA(2)**
projection of the standardized 13-D structural matrix. Gaps are found on a 2-D
grid over the PCA plane: a cell is a gap target if it is far from every real
character (high nearest-neighbour distance) and in a low-density neighbourhood.

Generation into the gap
-----------------------
We draw a batch of synthetic structural characters (ONE generation call, reusing
A2.1), project them with the SAME scaler+PCA, and keep the one closest to the
chosen gap target. It is appended to ``results/generated_characters.csv`` tagged
``source=gap`` with its stats and the gap distance (no Axis-1 score needed — the
gap character is defined geometrically, not by universe plausibility, so no extra
TabPFN scoring call is spent).

Output
------
* ``character_space_map.png`` (Agg) — training characters + agent-generated
  winners + the flagged gap target(s) + the gap character, on the PCA plane.
* appends the gap character row(s) to ``results/generated_characters.csv``.

Run::

    python src/axis2_gapfinder.py

No unit tests (project policy): running this script IS the verification.
"""

from __future__ import annotations

import os

os.environ["USE_TABPFN_LOCAL"] = "false"  # API backend (see axis2_generate)

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; MUST precede pyplot import.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.axis2_generate import generate_structural_batch  # noqa: E402
from src.dataprep import RANDOM_STATE, load_frame  # noqa: E402
from src.env import load_token  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_GEN_CSV = _RESULTS_DIR / "generated_characters.csv"
_PNG_PATH = _ROOT / "character_space_map.png"

GAP_POOL_SIZE = 60   # synthetic candidates to draw (one generation call)
GRID = 24            # resolution of the PCA-plane grid used to score gaps
N_GAPS = 1           # how many gap targets to flag + fill
DENSITY_RADIUS = 0.6  # neighbourhood radius (in standardized PCA units) for density


def _structural_matrix(frame):
    return frame.X_struct.to_numpy(dtype=np.float64), list(frame.X_struct.columns)


def _find_gap_targets(train_xy: np.ndarray, n_gaps: int):
    """Return the n_gaps grid points that are emptiest (far from all real chars).

    Score each grid cell by (distance to nearest real character) weighted down by
    local density, then take the top-n distinct cells. Gaps near the data cloud
    are preferred over the far empty corners by only considering grid cells inside
    the data's bounding box (so a gap is a hole WITHIN the space, not outside it).
    """
    xmin, ymin = train_xy.min(axis=0)
    xmax, ymax = train_xy.max(axis=0)
    xs = np.linspace(xmin, xmax, GRID)
    ys = np.linspace(ymin, ymax, GRID)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    # Distance from each grid cell to the nearest real character.
    d2 = ((grid[:, None, :] - train_xy[None, :, :]) ** 2).sum(axis=2)
    nn_dist = np.sqrt(d2.min(axis=1))
    # Local density: number of real chars within DENSITY_RADIUS.
    density = (np.sqrt(d2) < DENSITY_RADIUS).sum(axis=1)

    # Gap score: far from nearest char AND low density. Exclude cells with 0
    # neighbours within a wide radius (those are outside-the-cloud corners).
    wide = (np.sqrt(d2) < DENSITY_RADIUS * 3).sum(axis=1)
    eligible = wide > 0
    score = np.where(eligible, nn_dist / (1.0 + density), -np.inf)

    order = np.argsort(score)[::-1]
    chosen = []
    for idx in order:
        if score[idx] == -np.inf:
            break
        pt = grid[idx]
        if all(np.hypot(*(pt - grid[c])) > (xmax - xmin) / GRID * 2 for c in chosen):
            chosen.append(idx)
        if len(chosen) >= n_gaps:
            break
    return grid[chosen], nn_dist, grid


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_token()
    print("=" * 70)
    print("A2.3 — sparse-region gap finder (PCA projection; embeddings N/A in 0.6.3)")
    print("=" * 70)

    frame = load_frame()
    X_struct, struct_cols = _structural_matrix(frame)

    # Standardize + PCA(2) on real characters; reuse the transform for everything.
    scaler = StandardScaler().fit(X_struct)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaler.transform(X_struct))
    train_xy = pca.transform(scaler.transform(X_struct))
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.round(3).tolist()}")

    gap_targets, _nn_dist, _grid = _find_gap_targets(train_xy, N_GAPS)
    print(f"Flagged {len(gap_targets)} gap target(s) in PCA space: "
          f"{[tuple(np.round(p, 2)) for p in gap_targets]}")

    # --- Generate candidates and pick those closest to the gap target(s) ------
    df_cand, _m, _f, _c = generate_structural_batch(
        n_samples=GAP_POOL_SIZE, frame=frame, label="axis2_gap.gen"
    )
    cand = df_cand[struct_cols].to_numpy(dtype=np.float64)
    cand_xy = pca.transform(scaler.transform(cand))

    gap_rows = []
    gap_char_xy = []
    for gi, target in enumerate(gap_targets):
        d = np.hypot(cand_xy[:, 0] - target[0], cand_xy[:, 1] - target[1])
        j = int(d.argmin())
        gap_char_xy.append(cand_xy[j])
        rec = {struct_cols[i]: int(round(cand[j, i])) for i in range(len(struct_cols))}
        rec["entropy"] = ""
        rec["originality"] = ""
        rec["argmax_universe"] = ""
        rec["objective"] = f"fill_gap_{gi}"
        rec["iteration"] = -1
        rec["source"] = "gap"
        rec["gap_distance"] = float(d[j])
        gap_rows.append(rec)
        print(f"  gap {gi}: nearest candidate at PCA {tuple(np.round(cand_xy[j], 2))} "
              f"(dist {d[j]:.3f}) -> stats "
              f"{[int(round(cand[j,i])) for i in range(6)]}")

    # --- Append gap character(s) to the generated CSV ------------------------
    gap_df = pd.DataFrame(gap_rows)
    if _GEN_CSV.exists():
        existing = pd.read_csv(_GEN_CSV)
        combined = pd.concat([existing, gap_df], ignore_index=True)
    else:
        combined = gap_df
    combined.to_csv(_GEN_CSV, index=False)
    print(f"Appended {len(gap_df)} gap character(s) (source=gap) -> {_GEN_CSV}")

    # --- Plot the character space map ----------------------------------------
    agent_xy = None
    if _GEN_CSV.exists():
        gen = pd.read_csv(_GEN_CSV)
        agent_mask = gen["source"] == "agent"
        if agent_mask.any():
            agent_struct = gen.loc[agent_mask, struct_cols].to_numpy(dtype=np.float64)
            agent_xy = pca.transform(scaler.transform(agent_struct))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(train_xy[:, 0], train_xy[:, 1], s=8, c="#bbb", alpha=0.6,
               label=f"real characters (n={len(train_xy)})")
    if agent_xy is not None and len(agent_xy):
        ax.scatter(agent_xy[:, 0], agent_xy[:, 1], s=120, marker="*",
                   c="#1b9e77", edgecolor="k", label="agent-generated winners")
    gap_targets = np.asarray(gap_targets)
    ax.scatter(gap_targets[:, 0], gap_targets[:, 1], s=200, marker="X",
               c="#d95f02", edgecolor="k", label="flagged gap target")
    gcxy = np.asarray(gap_char_xy)
    ax.scatter(gcxy[:, 0], gcxy[:, 1], s=140, marker="P", c="#7570b3",
               edgecolor="k", label="gap character (generated)")
    ax.set_xlabel("PCA-1 (structural)")
    ax.set_ylabel("PCA-2 (structural)")
    ax.set_title("Character space map — real roster, generated winners, and a gap")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(_PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"Wrote character space map -> {_PNG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
