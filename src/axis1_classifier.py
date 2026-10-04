"""A1.1 — Axis-1 universe classifier + legitimacy GATE (Canon Detector / FEAT-002).

THE RIGOR GATE. If TabPFN cannot predict a character's universe from *substance*
(powerstats + encoded categoricals + TF-IDF of about/abilities) meaningfully above
chance, nothing downstream (plausibility, originality, generation) is meaningful.

What this does
--------------
1. Build the canonical frame (``src.dataprep.load_frame``) and a single stratified
   train/test split (``random_state=42``, ``stratify=y``, ``test_size≈0.25``). The
   split mask is also used so the structural categorical encoding and the TF-IDF
   block are both fit on TRAIN only (split-safe, no test leakage).
2. Fit ONE ``TabPFNClassifier(model_path="v3.5_default", n_estimators=8)`` on the
   TRAIN structural+text matrix (ONE fit call, wrapped in ``credits.log_call``).
3. ONE batched ``predict_proba`` on the FULL test set (never a per-row loop) and
   ONE batched ``predict_proba`` on the FULL train set (reused by A1.4 originality).
4. Report held-out accuracy, balanced accuracy, macro + per-class one-vs-rest AUC,
   and the confusion matrix. Train a LightGBM baseline on the SAME features/split
   as the comparison gate.
5. Write ``results/axis1_classifier.json`` (both models' metrics, split sizes,
   feature-block info, credit delta) and persist test/train probabilities +
   split indices to ``results/_axis1_test_probs.npz`` so A1.2/A1.3/A1.4 need no
   extra TabPFN call.

Honesty rules
-------------
* Metrics are REAL held-out numbers. Never fabricate.
* AUC is one-vs-rest, macro-averaged over classes that have at least one positive
  AND one negative in the test set; classes too tiny to score are listed as
  ``null`` in ``auc_per_class`` and excluded from the macro average (documented).
* TabPFN is called exactly twice here (one fit, two batched predicts share the
  same fitted estimator — predict_proba is cheap/flat; the fit is the billed step).

Run::

    python src/axis1_classifier.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Make `src` importable whether run as `python src/axis1_classifier.py` or imported.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.credits import get_usage, log_call  # noqa: E402
from src.dataprep import RANDOM_STATE, load_frame  # noqa: E402
from src.env import load_token  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_JSON_PATH = _RESULTS_DIR / "axis1_classifier.json"
_PROBS_PATH = _RESULTS_DIR / "_axis1_test_probs.npz"

TEST_SIZE = 0.25
TABPFN_MODEL_PATH = "v3.5_default"
TABPFN_N_ESTIMATORS = 8


def _build_matrices():
    """Return the split-safe feature matrices, labels, and split bookkeeping.

    Builds a stratified train/test split FIRST, then fits the structural
    categorical encoding and the TF-IDF text block on TRAIN rows only.
    """
    # First pass with all-rows mask just to get y and row count.
    base = load_frame()
    y = base.y
    n = len(y)
    idx = np.arange(n)

    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y.to_numpy(),
    )

    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True

    # Re-build the frame with a real train mask so categorical vocab is TRAIN-only.
    frame = load_frame(train_mask=train_mask)

    # Fit TF-IDF on TRAIN rows only, then transform all rows (split-safe IDF).
    frame.text_block.fit(train_idx)
    X_text_all = frame.text_block.transform(idx)

    X_struct_all = frame.X_struct.to_numpy(dtype=np.float64)
    X_all = np.hstack([X_struct_all, X_text_all])

    feature_names = frame.feature_names
    assert X_all.shape[1] == len(feature_names), (
        f"feature matrix width {X_all.shape[1]} != feature_names {len(feature_names)}"
    )

    return {
        "frame": frame,
        "X_all": X_all,
        "X_struct_all": X_struct_all,
        "X_text_all": X_text_all,
        "y": y,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "n_struct": X_struct_all.shape[1],
        "n_text": X_text_all.shape[1],
        "feature_names": feature_names,
    }


def _per_class_ovr_auc(y_true, proba, classes) -> dict[str, float | None]:
    """One-vs-rest AUC per class on held-out data.

    A class is scored only if the test set contains at least one positive and one
    negative example for it; otherwise AUC is undefined -> reported as None.
    """
    out: dict[str, float | None] = {}
    y_true = np.asarray(y_true)
    for i, cls in enumerate(classes):
        pos = (y_true == cls)
        n_pos = int(pos.sum())
        if n_pos == 0 or n_pos == len(y_true):
            out[str(cls)] = None
            continue
        try:
            out[str(cls)] = float(roc_auc_score(pos.astype(int), proba[:, i]))
        except ValueError:
            out[str(cls)] = None
    return out


def _macro_auc(per_class: dict[str, float | None]) -> float | None:
    vals = [v for v in per_class.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    load_token()
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("A1.1 — Axis-1 universe classifier (THE GATE)")
    print("=" * 70)

    data = _build_matrices()
    X_all = data["X_all"]
    y = data["y"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    classes = sorted(y.unique())

    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train = y.to_numpy()[train_idx]
    y_test = y.to_numpy()[test_idx]

    print(
        f"Split: train={len(train_idx)}  test={len(test_idx)}  "
        f"(test_size={TEST_SIZE}, stratified, random_state={RANDOM_STATE})"
    )
    print(
        f"Features: {X_all.shape[1]} "
        f"(structural={data['n_struct']}, tfidf={data['n_text']})"
    )
    print(f"Classes ({len(classes)}): {classes}")

    usage_before = get_usage()

    # --- TabPFN: ONE fit, then batched predicts (shared fitted estimator) ----
    from tabpfn_client import TabPFNClassifier

    clf = TabPFNClassifier(
        model_path=TABPFN_MODEL_PATH,
        n_estimators=TABPFN_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    with log_call("axis1.fit"):
        clf.fit(X_train, y_train)

    with log_call("axis1.predict_proba.test"):
        test_proba = clf.predict_proba(X_test)
    with log_call("axis1.predict_proba.train"):
        train_proba = clf.predict_proba(X_train)

    usage_after = get_usage()
    credit_delta = usage_after - usage_before

    # TabPFN class order (predict_proba columns align to clf.classes_).
    tabpfn_classes = [str(c) for c in clf.classes_]

    test_pred = np.array(tabpfn_classes)[test_proba.argmax(axis=1)]
    acc = float(accuracy_score(y_test, test_pred))
    bal_acc = float(balanced_accuracy_score(y_test, test_pred))
    per_class_auc = _per_class_ovr_auc(y_test, test_proba, tabpfn_classes)
    auc_macro = _macro_auc(per_class_auc)
    cm = confusion_matrix(y_test, test_pred, labels=tabpfn_classes).tolist()

    n_classes = len(classes)
    chance_acc = float(pd.Series(y_test).value_counts(normalize=True).max())  # majority-class

    print()
    print("-- TabPFN (held-out) --")
    print(f"  accuracy          : {acc:.4f}  (majority-class baseline {chance_acc:.4f})")
    print(f"  balanced_accuracy : {bal_acc:.4f}  (chance {1 / n_classes:.4f})")
    print(f"  auc_macro (OVR)   : {auc_macro}")
    for c in tabpfn_classes:
        print(f"    AUC[{c:>16}] = {per_class_auc[c]}")

    # --- LightGBM baseline on the SAME features/split ------------------------
    import lightgbm as lgb

    gbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    gbm.fit(X_train, y_train)
    gbm_classes = [str(c) for c in gbm.classes_]
    gbm_test_proba = gbm.predict_proba(X_test)
    gbm_pred = np.array(gbm_classes)[gbm_test_proba.argmax(axis=1)]
    gbm_acc = float(accuracy_score(y_test, gbm_pred))
    gbm_bal_acc = float(balanced_accuracy_score(y_test, gbm_pred))
    gbm_per_class_auc = _per_class_ovr_auc(y_test, gbm_test_proba, gbm_classes)
    gbm_auc_macro = _macro_auc(gbm_per_class_auc)

    print()
    print("-- LightGBM baseline (held-out, same split/features) --")
    print(f"  gbm_accuracy      : {gbm_acc:.4f}")
    print(f"  gbm_balanced_acc  : {gbm_bal_acc:.4f}")
    print(f"  gbm_auc_macro     : {gbm_auc_macro}")

    # --- GATE verdict (reported, not faked) ----------------------------------
    gate_pass = (auc_macro is not None) and (auc_macro > 0.5) and (acc > chance_acc)
    print()
    print(
        f"GATE: TabPFN macro-AUC={auc_macro} vs chance 0.5; "
        f"accuracy={acc:.4f} vs majority {chance_acc:.4f} -> "
        + ("ABOVE CHANCE (gate holds)" if gate_pass else "NOT above chance (gate FAILS)")
    )

    # --- Persist probs + split for downstream (no extra TabPFN call) ---------
    np.savez(
        _PROBS_PATH,
        test_proba=test_proba,
        train_proba=train_proba,
        test_idx=test_idx,
        train_idx=train_idx,
        y_test=y_test,
        y_train=y_train,
        tabpfn_classes=np.array(tabpfn_classes, dtype=object),
    )
    print(f"Persisted probs/split -> {_PROBS_PATH}")

    # --- Write metrics JSON --------------------------------------------------
    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {
            "name": "TabPFNClassifier",
            "model_path": TABPFN_MODEL_PATH,
            "n_estimators": TABPFN_N_ESTIMATORS,
            "random_state": RANDOM_STATE,
        },
        "split": {
            "test_size": TEST_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "feature_blocks": {
            "n_structural": int(data["n_struct"]),
            "n_tfidf": int(data["n_text"]),
            "n_total": int(X_all.shape[1]),
            "structural_feature_names": data["feature_names"][: data["n_struct"]],
        },
        "classes": tabpfn_classes,
        "chance": {
            "majority_class_accuracy": chance_acc,
            "uniform_balanced_accuracy": 1.0 / n_classes,
            "auc_chance": 0.5,
        },
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "auc_macro": auc_macro,
        "auc_per_class": per_class_auc,
        "confusion_matrix": {"labels": tabpfn_classes, "matrix": cm},
        "gbm_accuracy": gbm_acc,
        "gbm_balanced_accuracy": gbm_bal_acc,
        "gbm_auc_macro": gbm_auc_macro,
        "gbm_auc_per_class": gbm_per_class_auc,
        "gate_pass": bool(gate_pass),
        "credit_delta": int(credit_delta),
        "credit_usage_after": int(usage_after),
    }
    with _JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Wrote metrics -> {_JSON_PATH}")
    print(f"Credit delta this run: +{credit_delta}  (total used {usage_after})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
