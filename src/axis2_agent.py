"""A2.2 — bounded generate-score-refine character agent (FEAT-003).

The creative payoff of Axis 2. Given the synthetic-character generator (A2.1) and
the fitted Axis-1 universe probe (FEAT-002), run a BOUNDED loop that steers random
synthetic characters toward two opposite creative extremes:

* **max cross-universe entropy** — the "ultimate crossover": a character every
  universe finds equally plausible (flat P(universe), high Shannon entropy).
* **max originality** ``1 - max_u P(u)`` — "lands in empty space": a character no
  single universe strongly claims.

Loop design (bounded, batched, auditable)
-----------------------------------------
* The expensive, per-column generation (A2.1) is done ONCE up front into a large
  candidate pool (``POOL_SIZE``), plus at most ONE targeted top-up generation, so
  the metered generation cost is bounded regardless of iteration count.
* Each of ``MAX_ITERS`` (<=5) iterations scores ALL current candidates in exactly
  ONE batched ``predict_proba`` over the full candidate matrix — NEVER a per-row
  predict. ``assert`` guards enforce this.
* Steering between iterations is done WITHOUT new API calls: we keep the top
  candidates for each objective and breed the next generation by jittering /
  recombining those winners in structural feature space (deterministic,
  ``random_state=42``). This "resample around top performers" is the plan's
  steering, realised cheaply so the loop stays within budget.
* Stops early at convergence (best objective value stops improving) or at
  ``MAX_ITERS``.

Scoring synthetic structural rows with a STATS-ONLY probe (leakage-free)
-----------------------------------------------------------------------
The full Axis-1 probe reads TF-IDF of about/abilities prose, which contains
literal universe tokens (self-declaration leakage; see
``axis1_classifier`` leakage ablation). Synthetic characters have no prose, so a
text-trained probe would score them against an all-zero text block — ill-defined
and leaky-adjacent. The agent therefore scores with a **stats-only probe** fit on
the 13 STRUCTURAL features only (powerstats + encoded substance categoricals).
Steering is then provably driven by genuine substance signal, matching the
honest masked/stats-only story reported by Axis 1. Held-out stats-only accuracy
is ~0.90 (vs 0.67 majority), so the plausibility signal is real.

Call budget
-----------
<= 2 generation calls (initial pool + optional top-up; each is internally
~n_features metered column-passes, see A2.1) and <= MAX_ITERS (5) batched scoring
calls. Credits are logged per call. The agent asserts total SCORING calls <= 5
and that scoring is always batched (candidate count > 1 per call).

Outputs
-------
``results/generated_characters.csv`` — the winning characters for BOTH objectives
with per-universe plausibilities, ``entropy``, ``originality``, ``objective``,
``iteration`` and ``source`` columns.

Run::

    python src/axis2_agent.py

No unit tests (project policy): running this script IS the verification.
"""

from __future__ import annotations

import os

# Force the API backend before any tabpfn_extensions import (see axis2_generate).
os.environ["USE_TABPFN_LOCAL"] = "false"

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
from src.axis2_generate import (  # noqa: E402
    KEEP_NUMERIC,
    POWERSTAT_MAX,
    POWERSTAT_MIN,
    generate_structural_batch,
)
from src.credits import log_call  # noqa: E402
from src.dataprep import KEEP_CATEGORICAL, RANDOM_STATE, load_frame  # noqa: E402
from src.env import load_token  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_GEN_CSV = _RESULTS_DIR / "generated_characters.csv"

POOL_SIZE = 60          # initial candidate pool (one generation call)
MAX_ITERS = 5           # hard cap on scoring iterations
TOP_K = 8               # winners kept per objective each iteration to breed from
JITTER_STD = 6.0        # structural jitter std (powerstat scale) when breeding
CONVERGE_EPS = 1e-3     # stop if best objective improves by less than this


class ScoreCallCounter:
    """Guards: counts batched scoring calls and forbids per-row scoring."""

    def __init__(self) -> None:
        self.n_calls = 0

    def record(self, n_rows: int) -> None:
        self.n_calls += 1
        assert n_rows > 1, (
            f"Scoring call #{self.n_calls} had {n_rows} row(s) — scoring MUST be "
            "batched over many candidates, never per-row."
        )
        assert self.n_calls <= MAX_ITERS, (
            f"Scoring call budget exceeded: {self.n_calls} > {MAX_ITERS}."
        )


def _fit_axis1_probe():
    """Fit the STATS-ONLY Axis-1 probe once (train split).

    Fits on the 13 structural features ONLY (no TF-IDF text), so scoring of
    synthetic characters is driven by genuine substance, not leaked prose tokens.
    Returns (clf, data, classes).
    """
    data = _build_matrices()
    from tabpfn_client import TabPFNClassifier

    clf = TabPFNClassifier(
        model_path=TABPFN_MODEL_PATH,
        n_estimators=TABPFN_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    X_struct = data["X_struct_all"]
    y = data["y"]
    train_idx = data["train_idx"]
    with log_call("axis2_agent.probe.fit[stats_only]"):
        clf.fit(X_struct[train_idx], y.to_numpy()[train_idx])
    classes = np.array([str(c) for c in clf.classes_])
    return clf, data, classes


def _score(clf, counter: ScoreCallCounter, struct: np.ndarray, label: str):
    """ONE batched predict_proba over all candidate structural rows (stats-only)."""
    X = struct.astype(np.float64)
    counter.record(X.shape[0])
    with log_call(label):
        proba = clf.predict_proba(X)
    return proba


def _entropy(proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _originality(proba: np.ndarray) -> np.ndarray:
    return 1.0 - proba.max(axis=1)


def _breed(winners: np.ndarray, n_children: int, cat_idx: list[int], rng) -> np.ndarray:
    """Create the next candidate batch by jittering/recombining winners.

    Numeric columns get Gaussian jitter (clipped to 0-100); categorical columns
    are inherited from a random winner (so codes stay valid). Deterministic given
    the seeded ``rng``.
    """
    n_feat = winners.shape[1]
    num_idx = [i for i in range(n_feat) if i not in cat_idx]
    children = np.empty((n_children, n_feat), dtype=float)
    for k in range(n_children):
        a, b = rng.integers(0, len(winners), size=2)
        base = winners[a].copy()
        # Recombine: half the numeric genes from the second parent.
        for i in num_idx:
            if rng.random() < 0.5:
                base[i] = winners[b][i]
            base[i] += rng.normal(0.0, JITTER_STD)
        # Categorical genes inherited whole from one parent (valid codes).
        for i in cat_idx:
            base[i] = winners[a][i] if rng.random() < 0.5 else winners[b][i]
        children[k] = base
    # Clip numeric (powerstat) columns to sane range; round all to ints.
    children[:, num_idx] = np.clip(children[:, num_idx], POWERSTAT_MIN, POWERSTAT_MAX)
    return np.rint(children).astype(int).astype(float)


def main() -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_token()
    rng = np.random.default_rng(RANDOM_STATE)

    print("=" * 70)
    print("A2.2 — generate-score-refine agent (two objectives, bounded + batched)")
    print("=" * 70)

    # --- Fit the Axis-1 scoring probe once (stats-only) ----------------------
    clf, data, classes = _fit_axis1_probe()
    struct_cols = list(data["frame"].X_struct.columns)
    cat_idx = [i for i, c in enumerate(struct_cols) if c in KEEP_CATEGORICAL]
    print(f"Axis-1 STATS-ONLY probe fit. classes={list(classes)}  n_features={len(struct_cols)}")

    # --- Initial candidate pool: ONE generation call -------------------------
    frame = data["frame"]
    df_pool, _gen_model, _frame, _cols = generate_structural_batch(
        n_samples=POOL_SIZE, frame=frame, label="axis2_agent.gen.pool"
    )
    pool = df_pool[struct_cols].to_numpy(dtype=float)
    print(f"Initial pool: {pool.shape[0]} candidates x {pool.shape[1]} features")

    counter = ScoreCallCounter()
    # Track the best candidate seen for each objective across all iterations.
    best = {
        "entropy": {"val": -np.inf, "row": None, "proba": None, "iter": -1},
        "originality": {"val": -np.inf, "row": None, "proba": None, "iter": -1},
    }

    # Two subpopulations that evolve toward their OWN objective, so the two
    # winners are genuinely different characters. Both halves are scored together
    # in ONE batched predict_proba per iteration (concatenated), so the scoring
    # call budget stays <= MAX_ITERS.
    half = POOL_SIZE // 2
    subpop = {"entropy": pool[:half].copy(), "originality": pool[half:].copy()}

    def _objective_values(obj_name, proba):
        return _entropy(proba) if obj_name == "entropy" else _originality(proba)

    for it in range(MAX_ITERS):
        ent_pop = subpop["entropy"]
        orig_pop = subpop["originality"]
        stacked = np.vstack([ent_pop, orig_pop])
        proba_all = _score(clf, counter, stacked, f"axis2_agent.score.iter{it}")
        proba = {
            "entropy": proba_all[: len(ent_pop)],
            "originality": proba_all[len(ent_pop):],
        }

        ent = _entropy(proba["entropy"])
        orig = _originality(proba["originality"])
        print(
            f"  iter {it}: n={len(stacked)}  "
            f"entropy-pop[max={ent.max():.3f} mean={ent.mean():.3f}]  "
            f"originality-pop[max={orig.max():.3f} mean={orig.mean():.3f}]"
        )

        improved = False
        for obj_name in ("entropy", "originality"):
            vals = _objective_values(obj_name, proba[obj_name])
            j = int(vals.argmax())
            if vals[j] > best[obj_name]["val"] + CONVERGE_EPS:
                best[obj_name] = {
                    "val": float(vals[j]),
                    "row": subpop[obj_name][j].copy(),
                    "proba": proba[obj_name][j].copy(),
                    "iter": it,
                }
                improved = True

        if it == MAX_ITERS - 1:
            break
        if not improved:
            print(f"  converged at iter {it} (no objective improved).")
            break

        # --- Steer: breed each subpopulation toward ITS OWN objective ---------
        # Breeding (jitter + recombine winners) is FREE — no extra generation
        # call — so the loop steers without re-paying the per-column generation
        # cost. One generation call total (the initial pool) per the budget
        # guardrail; we do NOT regenerate full-dataset batches each iteration.
        for obj_name in ("entropy", "originality"):
            vals = _objective_values(obj_name, proba[obj_name])
            top = subpop[obj_name][np.argsort(vals)[::-1][:TOP_K]]
            subpop[obj_name] = _breed(top, len(subpop[obj_name]), cat_idx, rng)

    # --- Assemble winners for BOTH objectives --------------------------------
    rows = []
    for obj_name in ("entropy", "originality"):
        b = best[obj_name]
        rec = {struct_cols[i]: int(round(b["row"][i])) for i in range(len(struct_cols))}
        for ci, cls in enumerate(classes):
            rec[f"P[{cls}]"] = float(b["proba"][ci])
        rec["entropy"] = float(_entropy(b["proba"][None, :])[0])
        rec["originality"] = float(_originality(b["proba"][None, :])[0])
        rec["argmax_universe"] = str(classes[int(b["proba"].argmax())])
        rec["objective"] = f"max_{obj_name}"
        rec["iteration"] = int(b["iter"])
        rec["source"] = "agent"
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(_GEN_CSV, index=False)
    print(f"\nWrote {len(out)} objective winners -> {_GEN_CSV}")
    print(f"Scoring calls used: {counter.n_calls} (<= {MAX_ITERS}; all batched)")
    for _, r in out.iterrows():
        print(
            f"  [{r['objective']:>16}] entropy={r['entropy']:.3f} "
            f"originality={r['originality']:.3f} argmax={r['argmax_universe']} "
            f"(iter {r['iteration']})"
        )
    print(
        "\nNote: generation is per-column via the extensions API (see A2.1); "
        "the credit log shows the real column-pass cost. Scoring is 1 batched "
        "predict_proba per iteration, never per-row."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
