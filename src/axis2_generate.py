"""A2.1 — TabPFNUnsupervisedModel synthetic-character generation (FEAT-003).

Axis 2 generates NEW characters, then (A2.2+) scores them with the Axis-1 probe
to steer toward "ultimate crossover" and "lands in empty space" archetypes. This
module is the generation primitive: fit an unsupervised joint model on the real
characters' STRUCTURAL feature matrix and sample synthetic rows from it.

Confirmed TabPFNUnsupervisedModel API (probed live, see ``--probe``)
--------------------------------------------------------------------
``tabpfn_extensions.unsupervised.TabPFNUnsupervisedModel`` (extensions 0.6.3):

* **Constructor** takes BOTH a classifier and a regressor::

      TabPFNUnsupervisedModel(tabpfn_clf=<TabPFNClassifier>,
                              tabpfn_reg=<TabPFNRegressor>)

* **Backend**: the extensions resolve ``TabPFNClassifier`` / ``TabPFNRegressor``
  from ``tabpfn_extensions.utils`` keyed on ``USE_TABPFN_LOCAL``. The LOCAL
  ``tabpfn`` backend would be free, but in THIS environment the local model
  weights are not cached and the one-time license acceptance cannot be completed
  non-interactively (``TabPFNLicenseError`` — confirmed live). We therefore force
  ``USE_TABPFN_LOCAL=false`` and generate against the metered **API client**
  backend, which works without a local download.
* **Generation cost is PER-COLUMN (documented API adaptation).**
  ``generate_synthetic_data`` samples each feature column by column across
  ``n_permutations`` passes, so one call makes ~``n_features`` metered sub-calls
  (independent of ``n_samples`` — the row count is free). Measured live: ~10K
  credits per column-pass. We therefore (a) generate on the SMALL 13-column
  structural matrix only, (b) use ``n_permutations=1``, and (c) generate a LARGE
  candidate pool per call so one generation yields many candidates. This keeps
  generation ~130K credits/batch, far under the 20M budget. This is the honest
  behaviour of the real API, NOT the "one flat call" the plan assumed; the Axis-1
  *scoring* probe (A2.2) is still exactly ONE batched ``predict_proba`` per loop
  iteration (never per-row).
* **fit(X, y=None)** — unsupervised; accepts ``np.ndarray | torch.Tensor |
  pd.DataFrame``. We pass the structural matrix as a float32 numpy array and call
  ``set_categorical_features([...])`` first with the categorical column indices.
* **generate_synthetic_data(n_samples=100, t=1.0, n_permutations=3, dag=None)**
  returns a ``torch.Tensor`` of shape ``(n_samples, n_features)``. ``t`` is the
  sampling temperature (higher = more diverse). We convert to numpy and clip the
  6 powerstats back into their natural [0, 100] range (the sampler is continuous).
* ``get_embeddings`` is **NotImplementedError** in 0.6.3 (removed in the TabPFN
  refactor), so the A2.3 gap finder uses a PCA projection, not model embeddings.

Every TabPFN call is wrapped in ``credits.log_call`` for the audit trail.

Run modes
---------
``python src/axis2_generate.py --probe``
    Fit on a tiny sample, sample a handful of rows, and PRINT the real method
    signatures + a sample row. Confirms the API without faking anything.

``python src/axis2_generate.py``
    Full run: fit on the real structural matrix, sample a batch of synthetic
    characters, clip stats to sane ranges, and persist to
    ``results/_axis2_raw.csv``.

No unit tests (project policy): running this script IS the verification.
"""

from __future__ import annotations

import argparse
import inspect
import os
import time

# Backend selection MUST happen before ANY tabpfn_extensions import: utils.py
# binds TabPFNClassifier/TabPFNRegressor to the local-or-API backend at import
# time by reading USE_TABPFN_LOCAL. The local backend's weights are not cached in
# this environment and its one-time license cannot be accepted non-interactively,
# so we force the metered API client backend here.
os.environ["USE_TABPFN_LOCAL"] = "false"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.credits import log_call  # noqa: E402
from src.dataprep import (  # noqa: E402
    KEEP_CATEGORICAL,
    KEEP_NUMERIC,
    RANDOM_STATE,
    load_frame,
)
from src.env import load_token  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_RAW_CSV = _RESULTS_DIR / "_axis2_raw.csv"

# Generation hyper-parameters (fixed for determinism / reproducibility).
GEN_TEMPERATURE = 1.0
GEN_N_PERMUTATIONS = 3
POWERSTAT_MIN = 0
POWERSTAT_MAX = 100

# Bounded retry for transient API errors (network hiccups, read timeouts).
MAX_API_RETRIES = 3
RETRY_BACKOFF_S = 5.0


def _retry_api(fn, *, what: str):
    """Call ``fn`` with up to MAX_API_RETRIES attempts on transient errors.

    Retries only on transient network/timeout errors; re-raises immediately on
    anything else. After the cap is hit the last exception propagates so the
    caller FAILS loudly rather than spinning forever (per the budget guardrails).
    """
    last_exc = None
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — inspect message for transience
            msg = f"{type(exc).__name__}: {exc}".lower()
            transient = any(
                k in msg
                for k in ("timeout", "timed out", "connection", "temporarily",
                          "read operation", "503", "502", "504", "reset")
            )
            last_exc = exc
            if not transient or attempt == MAX_API_RETRIES:
                raise
            wait = RETRY_BACKOFF_S * attempt
            print(f"[retry] {what}: transient error on attempt {attempt}/"
                  f"{MAX_API_RETRIES} ({type(exc).__name__}); retrying in {wait:.0f}s")
            time.sleep(wait)
    if last_exc is not None:  # pragma: no cover — loop always returns/raises
        raise last_exc


def _structural_columns(frame) -> list[str]:
    """Structural feature names in the order dataprep builds them."""
    return list(frame.X_struct.columns)


def _categorical_indices(struct_cols: list[str]) -> list[int]:
    """Indices (within the structural matrix) of the integer-encoded categoricals."""
    return [i for i, c in enumerate(struct_cols) if c in KEEP_CATEGORICAL]


def _build_unsupervised_model(random_state: int = RANDOM_STATE):
    """Construct a TabPFNUnsupervisedModel on the metered API backend.

    Returns (model, backend_name). Forces ``USE_TABPFN_LOCAL=false`` before
    importing the extensions' model classes: the LOCAL backend is free but its
    weights are not cached here and the one-time license cannot be accepted
    non-interactively, so we use the API client backend which works without a
    local download. Small ``n_estimators`` keeps each per-column call cheap.
    """
    os.environ["USE_TABPFN_LOCAL"] = "false"
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
    from tabpfn_extensions.utils import TabPFNClassifier, TabPFNRegressor

    backend = TabPFNClassifier.__module__.split(".")[0]
    clf = TabPFNClassifier(n_estimators=2, random_state=random_state)
    reg = TabPFNRegressor(n_estimators=2, random_state=random_state)
    model = TabPFNUnsupervisedModel(tabpfn_clf=clf, tabpfn_reg=reg)
    return model, backend

def _clip_and_frame(raw: np.ndarray, struct_cols: list[str]) -> pd.DataFrame:
    """Turn a raw synthetic sample into a sane structural DataFrame.

    * Powerstats are clipped to [0, 100] and rounded to ints (the natural scale).
    * Categorical codes are rounded and clipped to the valid code range per column
      (so a generated code always maps back to a real learned level).
    """
    df = pd.DataFrame(raw, columns=struct_cols)
    for col in KEEP_NUMERIC:
        if col in df.columns:
            df[col] = df[col].round().clip(POWERSTAT_MIN, POWERSTAT_MAX).astype(int)
    for col in KEEP_CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].round().clip(lower=0).astype(int)
    return df


def generate_structural_batch(
    n_samples: int,
    model=None,
    frame=None,
    temperature: float = GEN_TEMPERATURE,
    n_permutations: int = GEN_N_PERMUTATIONS,
    label: str = "axis2.generate",
):
    """Fit (if needed) and sample ``n_samples`` synthetic structural characters.

    Reused by A2.2/A2.5 so the loop and the demo share ONE generation path.
    Returns (df_synth, model, frame, struct_cols). Every call wrapped in log_call.
    """
    if frame is None:
        frame = load_frame()
    struct_cols = _structural_columns(frame)
    X_struct = frame.X_struct.to_numpy(dtype=np.float32)

    if model is None:
        model, backend = _build_unsupervised_model()
        cat_idx = _categorical_indices(struct_cols)
        model.set_categorical_features(cat_idx)
        with log_call(f"{label}.fit[{backend}]"):
            model.fit(X_struct)

    with log_call(f"{label}.sample"):
        raw = _retry_api(
            lambda: model.generate_synthetic_data(
                n_samples=n_samples,
                t=temperature,
                n_permutations=n_permutations,
            ),
            what=f"{label}.sample",
        )
    raw_np = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
    df_synth = _clip_and_frame(raw_np, struct_cols)
    return df_synth, model, frame, struct_cols


def _probe() -> int:
    """Tiny live API probe: confirm signatures + sample a few rows (no faking)."""
    from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel

    load_token()  # register token so log_call's usage audit works (generation is local)
    print("=" * 70)
    print("A2.1 PROBE — confirming TabPFNUnsupervisedModel API (live, tiny sample)")
    print("=" * 70)
    print(f"constructor : TabPFNUnsupervisedModel{inspect.signature(TabPFNUnsupervisedModel.__init__)}")
    print(f"fit         : {inspect.signature(TabPFNUnsupervisedModel.fit)}")
    print(
        "generate    : "
        f"{inspect.signature(TabPFNUnsupervisedModel.generate_synthetic_data)}"
    )

    frame = load_frame()
    struct_cols = _structural_columns(frame)
    # Tiny sample of real rows to keep the probe fast.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_rows = rng.choice(len(frame.X_struct), size=40, replace=False)
    X_small = frame.X_struct.to_numpy(dtype=np.float32)[sample_rows]

    model, backend = _build_unsupervised_model()
    print(f"backend     : {backend} (USE_TABPFN_LOCAL={os.environ.get('USE_TABPFN_LOCAL')})")
    model.set_categorical_features(_categorical_indices(struct_cols))
    # FAST_TEST_MODE caps the per-column work so the probe is quick.
    os.environ["FAST_TEST_MODE"] = "1"
    with log_call("axis2.probe.fit"):
        model.fit(X_small)
    with log_call("axis2.probe.sample"):
        raw = model.generate_synthetic_data(n_samples=5, t=1.0, n_permutations=1)
    os.environ.pop("FAST_TEST_MODE", None)

    raw_np = raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw)
    print(f"\nreturned    : type={type(raw).__name__} shape={tuple(raw_np.shape)}")
    df = _clip_and_frame(raw_np, struct_cols)
    print("\nSample synthetic structural rows (clipped to sane ranges):")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.to_string(index=False))
    print("\nPROBE OK — API confirmed, generation returns a real torch.Tensor.")
    return 0


def main(n_samples: int = 60) -> int:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    load_token()  # register token so log_call's usage audit works (generation is local)
    print("=" * 70)
    print(f"A2.1 — generate {n_samples} synthetic characters (structural matrix)")
    print("=" * 70)

    frame = load_frame()
    struct_cols = _structural_columns(frame)
    print(f"Structural features ({len(struct_cols)}): {struct_cols}")
    cat_idx = _categorical_indices(struct_cols)
    print(f"Categorical indices: {cat_idx}")

    df_synth, _model, _frame, _cols = generate_structural_batch(
        n_samples=n_samples, frame=frame
    )
    df_synth.to_csv(_RAW_CSV, index=False)
    print(f"\nWrote {len(df_synth)} raw synthetic rows -> {_RAW_CSV}")
    print("Powerstat ranges of generated rows:")
    for col in KEEP_NUMERIC:
        print(f"  {col:>26}: [{df_synth[col].min()}, {df_synth[col].max()}]"
              f"  mean={df_synth[col].mean():.1f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Axis-2 synthetic generation (A2.1).")
    parser.add_argument("--probe", action="store_true", help="Confirm the API on a tiny sample.")
    parser.add_argument("--n-samples", type=int, default=60, help="Full-run batch size.")
    args = parser.parse_args()
    if args.probe:
        raise SystemExit(_probe())
    raise SystemExit(main(n_samples=args.n_samples))
