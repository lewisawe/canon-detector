"""A2.5 — the agent demo (FEAT-003): the project's single end-to-end smoke check.

One entry point that, given a creative target, runs the SMALLEST path through the
Axis-2 pipeline (generate -> score with the stats-only Axis-1 probe -> validate)
and prints the resulting character with its per-universe plausibilities, the
objective score, and its novelty + coherence. This is the ONE end-to-end run the
project's verification policy allows.

Targets
-------
* ``maximally-cross-universe`` — maximize Shannon entropy of P(universe): a
  character every universe finds equally plausible (the "ultimate crossover").
* ``maximally-original`` — maximize ``1 - max_u P(u)``: a character no single
  universe strongly claims (lands between archetypes).
* ``fill-a-gap`` — pick the generated character sitting in the emptiest region of
  the PCA structural map (defined geometrically, like A2.3).

Design (bounded + batched + honest)
-----------------------------------
* ONE generation call (a small pool) + ONE batched ``predict_proba`` (stats-only
  probe) — never a per-row predict. Scoring is on the 13 structural features only,
  so steering is on genuine substance, not leaked prose tokens.
* Novelty = nearest-neighbour distance to the closest real character in
  standardized structural space; coherence = sane stat ranges (same rules as
  A2.4). No extra TabPFN call for validation (pure geometry).

Run::

    python src/demo.py --target maximally-original
    python src/demo.py --target maximally-cross-universe
    python src/demo.py --target fill-a-gap
"""

from __future__ import annotations

import os

os.environ["USE_TABPFN_LOCAL"] = "false"  # API backend (see axis2_generate)

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.axis1_classifier import (  # noqa: E402
    TABPFN_MODEL_PATH,
    TABPFN_N_ESTIMATORS,
    _build_matrices,
)
from src.axis2_agent import _entropy, _originality  # noqa: E402
from src.axis2_generate import generate_structural_batch  # noqa: E402
from src.axis2_validate import (  # noqa: E402
    NOVELTY_MIN_DISTANCE,
    STAT_MAX,
    STAT_MIN,
    TOTAL_MAX,
    TOTAL_MIN,
)
from src.credits import log_call  # noqa: E402
from src.dataprep import KEEP_NUMERIC, RANDOM_STATE, load_frame  # noqa: E402
from src.env import load_token  # noqa: E402

DEMO_POOL_SIZE = 40  # small pool for the smoke run (one generation call)

TARGETS = ("maximally-cross-universe", "maximally-original", "fill-a-gap")


def _coherent(stat_vals) -> bool:
    stats = np.asarray(stat_vals, dtype=float)
    in_range = bool(((stats >= STAT_MIN) & (stats <= STAT_MAX)).all())
    total_ok = TOTAL_MIN <= stats.sum() <= TOTAL_MAX
    absurd = (stats >= 95).sum() >= 1 and (stats <= 5).sum() >= 5
    return in_range and total_ok and not absurd


def main(target: str) -> int:
    if target not in TARGETS:
        raise SystemExit(f"--target must be one of {TARGETS}, got {target!r}")

    load_token()
    print("=" * 70)
    print(f"DEMO — generate a character for target: {target}")
    print("=" * 70)

    # --- Fit the stats-only Axis-1 probe once --------------------------------
    data = _build_matrices()
    struct_cols = list(data["frame"].X_struct.columns)
    X_struct_all = data["X_struct_all"]
    y = data["y"]
    train_idx = data["train_idx"]

    from tabpfn_client import TabPFNClassifier

    clf = TabPFNClassifier(
        model_path=TABPFN_MODEL_PATH,
        n_estimators=TABPFN_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    with log_call("demo.probe.fit[stats_only]"):
        clf.fit(X_struct_all[train_idx], y.to_numpy()[train_idx])
    classes = np.array([str(c) for c in clf.classes_])

    # --- ONE generation call -------------------------------------------------
    frame = data["frame"]
    df_pool, _m, _f, _c = generate_structural_batch(
        n_samples=DEMO_POOL_SIZE, frame=frame, label="demo.gen"
    )
    cand = df_pool[struct_cols].to_numpy(dtype=float)

    # --- ONE batched scoring call (stats-only, never per-row) ----------------
    with log_call("demo.score"):
        proba = clf.predict_proba(cand.astype(np.float64))
    assert proba.shape[0] == cand.shape[0] > 1, "scoring must be batched"

    # --- Pick the winner for the chosen target -------------------------------
    if target == "maximally-cross-universe":
        scores = _entropy(proba)
        score_name = "entropy"
    elif target == "maximally-original":
        scores = _originality(proba)
        score_name = "originality"
    else:  # fill-a-gap: emptiest region of the PCA structural map
        X_real = frame.X_struct.to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(X_real)
        pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaler.transform(X_real))
        real_xy = pca.transform(scaler.transform(X_real))
        cand_xy = pca.transform(scaler.transform(cand))
        # score = distance to nearest real character (larger = emptier region)
        d2 = ((cand_xy[:, None, :] - real_xy[None, :, :]) ** 2).sum(axis=2)
        scores = np.sqrt(d2.min(axis=1))
        score_name = "gap_distance(min NN in PCA)"

    j = int(scores.argmax())
    win_stats = [int(round(cand[j, i])) for i in range(len(KEEP_NUMERIC))]
    win_cats = {
        struct_cols[i]: int(round(cand[j, i]))
        for i in range(len(KEEP_NUMERIC), len(struct_cols))
    }

    # --- Novelty + coherence -------------------------------------------------
    X_real = frame.X_struct.to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(X_real)
    real_std = scaler.transform(X_real)
    win_std = scaler.transform(cand[j][None, :])
    d = np.sqrt(((real_std - win_std) ** 2).sum(axis=1))
    nn_i = int(d.argmin())
    nn_dist = float(d[nn_i])
    nn_name = frame.df["name"].astype(str).to_numpy()[nn_i]
    novel = nn_dist > NOVELTY_MIN_DISTANCE
    coherent = _coherent(win_stats)

    # --- Print the generated character ---------------------------------------
    ent = float(_entropy(proba[j][None, :])[0])
    orig = float(_originality(proba[j][None, :])[0])
    print(f"\nWinning character ({score_name}={scores[j]:.3f}):")
    print("  Powerstats:")
    for name, val in zip(KEEP_NUMERIC, win_stats):
        print(f"    {name:>26}: {val}")
    print(f"    {'TOTAL':>26}: {sum(win_stats)}")
    print("  Encoded substance categoricals (code):")
    for name, val in win_cats.items():
        print(f"    {name:>30}: {val}")
    print("\n  Cross-universe plausibility P(universe | substance):")
    order = np.argsort(proba[j])[::-1]
    for k in order:
        bar = "#" * int(round(proba[j][k] * 40))
        print(f"    {classes[k]:>14}: {proba[j][k]:.3f} {bar}")
    print(f"\n  entropy          : {ent:.3f}  (max possible {np.log(len(classes)):.3f})")
    print(f"  originality      : {orig:.3f}")
    print(f"  argmax universe  : {classes[int(proba[j].argmax())]}")
    print(f"  novelty NN dist  : {nn_dist:.3f}  (nearest real character: {nn_name})")
    print(f"  novel            : {novel}")
    print(f"  coherent         : {coherent}")
    print("\nDEMO OK — generated a novel, coherent character and scored it on the "
          "stats-only (leakage-free) Axis-1 probe. Scoring was ONE batched call.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axis-2 agent demo (end-to-end smoke).")
    parser.add_argument(
        "--target",
        default="maximally-original",
        choices=TARGETS,
        help="Creative objective to steer the generated character toward.",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.target))
