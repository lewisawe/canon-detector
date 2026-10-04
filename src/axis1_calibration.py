"""A1.2 — reliability, ECE, and (re)calibration (Canon Detector / FEAT-002).

Reuses the held-out probabilities persisted by A1.1 (``results/_axis1_test_probs.npz``)
so NO new TabPFN call is made. Computes a reliability curve and the Expected
Calibration Error (ECE) on held-out data for the multi-class probe. If the raw
probabilities are miscalibrated, fits isotonic and Platt (sigmoid) calibrators on
a calibration slice of the held-out set and reports raw-vs-calibrated ECE.

Design notes
------------
* ECE is computed on the **confidence of the predicted (argmax) class** against
  whether that prediction was correct — the standard top-label reliability notion
  for multi-class classifiers, binned into ``N_BINS`` equal-width confidence bins.
* Recalibration here is evaluated honestly by splitting the held-out test set into
  a calibration half and an evaluation half (``random_state=42``). Isotonic/Platt
  are fit on the calibration half's top-label confidences and ECE is reported on
  the evaluation half, so the calibrated ECE is not computed on the fitting data.
* Merges ``ece_raw`` / ``ece_isotonic`` / ``ece_platt`` into
  ``results/axis1_classifier.json`` without clobbering A1.1's keys.
* Saves ``reliability_curve.png`` with the Agg (headless) backend.

Run::

    python src/axis1_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; MUST precede pyplot import.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.dataprep import RANDOM_STATE  # noqa: E402

_RESULTS_DIR = _ROOT / "results"
_JSON_PATH = _RESULTS_DIR / "axis1_classifier.json"
_PROBS_PATH = _RESULTS_DIR / "_axis1_test_probs.npz"
_PNG_PATH = _ROOT / "reliability_curve.png"

N_BINS = 10


def _top_label_confidence(proba: np.ndarray, classes: np.ndarray, y_true: np.ndarray):
    """Return (confidence, correct) for the argmax (top-label) prediction."""
    pred_idx = proba.argmax(axis=1)
    conf = proba[np.arange(len(proba)), pred_idx]
    pred_label = classes[pred_idx]
    correct = (pred_label == y_true).astype(float)
    return conf, correct


def _ece_and_curve(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS):
    """Expected Calibration Error (equal-width bins) + reliability curve points."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    curve_conf, curve_acc, curve_count = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        # last bin inclusive of 1.0
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(conf[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
        curve_conf.append(bin_conf)
        curve_acc.append(bin_acc)
        curve_count.append(count)
    return float(ece), (curve_conf, curve_acc, curve_count)


def main() -> int:
    if not _PROBS_PATH.exists():
        raise FileNotFoundError(
            f"{_PROBS_PATH} missing — run src/axis1_classifier.py (A1.1) first."
        )
    data = np.load(_PROBS_PATH, allow_pickle=True)
    test_proba = data["test_proba"]
    y_test = data["y_test"].astype(str)
    classes = data["tabpfn_classes"].astype(str)

    print("=" * 70)
    print("A1.2 — reliability + ECE + recalibration (NO new TabPFN call)")
    print("=" * 70)

    # --- Raw top-label calibration on the full held-out set ------------------
    conf_raw, correct_raw = _top_label_confidence(test_proba, classes, y_test)
    ece_raw, curve_raw = _ece_and_curve(conf_raw, correct_raw)
    print(f"Raw held-out top-label ECE ({N_BINS} bins): {ece_raw:.4f}")

    # --- Honest recalibration: calib half fits, eval half scores -------------
    n = len(conf_raw)
    idx = np.arange(n)
    calib_idx, eval_idx = train_test_split(
        idx, test_size=0.5, random_state=RANDOM_STATE
    )
    conf_calib, correct_calib = conf_raw[calib_idx], correct_raw[calib_idx]
    conf_eval, correct_eval = conf_raw[eval_idx], correct_raw[eval_idx]

    ece_eval_raw, _ = _ece_and_curve(conf_eval, correct_eval)

    # Isotonic on confidence -> P(correct).
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(conf_calib, correct_calib)
    conf_iso = iso.predict(conf_eval)
    ece_iso, curve_iso = _ece_and_curve(conf_iso, correct_eval)

    # Platt / sigmoid on confidence -> P(correct).
    platt = LogisticRegression()
    platt.fit(conf_calib.reshape(-1, 1), correct_calib.astype(int))
    conf_platt = platt.predict_proba(conf_eval.reshape(-1, 1))[:, 1]
    ece_platt, _ = _ece_and_curve(conf_platt, correct_eval)

    print(f"Eval-half raw ECE     : {ece_eval_raw:.4f}")
    print(f"Eval-half isotonic ECE: {ece_iso:.4f}")
    print(f"Eval-half Platt ECE   : {ece_platt:.4f}")

    best_name = min(
        [("raw", ece_eval_raw), ("isotonic", ece_iso), ("platt", ece_platt)],
        key=lambda kv: kv[1],
    )[0]
    print(f"Best calibrator by eval-half ECE: {best_name}")

    # --- Reliability curve plot (raw full held-out + isotonic eval) ----------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    cc, ca, _cnt = curve_raw
    ax.plot(cc, ca, "o-", color="#c44", label=f"raw (ECE={ece_raw:.3f})")
    ic, ia, _ic = curve_iso
    if ic:
        ax.plot(ic, ia, "s-", color="#48c", label=f"isotonic eval (ECE={ece_iso:.3f})")
    ax.set_xlabel("mean predicted confidence (top label)")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Axis-1 reliability curve (held-out)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(_PNG_PATH, dpi=120)
    plt.close(fig)
    print(f"Wrote reliability curve -> {_PNG_PATH}")

    # --- Merge into axis1_classifier.json (don't clobber A1.1) ---------------
    existing = {}
    if _JSON_PATH.exists():
        with _JSON_PATH.open(encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["calibration"] = {
        "n_bins": N_BINS,
        "method": "top-label (argmax-confidence vs correctness), equal-width bins",
        "ece_raw": ece_raw,
        "ece_eval_raw": ece_eval_raw,
        "ece_isotonic": ece_iso,
        "ece_platt": ece_platt,
        "best_calibrator": best_name,
        "recalibration_eval": "test set split 50/50 (random_state=42); "
        "calibrators fit on calib half, ECE reported on eval half",
    }
    # Convenience top-level keys the acceptance criteria name explicitly.
    existing["ece_raw"] = ece_raw
    existing["ece_isotonic"] = ece_iso
    existing["ece_platt"] = ece_platt
    with _JSON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    print(f"Merged calibration metrics -> {_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
