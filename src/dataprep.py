"""Canonical data-prep for Canon Detector (FEAT-001).

ONE place that loads the read-only source CSV, applies the KEEP/DROP leakage
decisions (from the task's ``context.json`` ``data_feature_decisions``), and
hands every downstream Axis script the same honest modeling frame. Downstream
scripts import :func:`load_frame` rather than re-reading or re-cleaning the CSV.

What it builds
--------------
* **Structural numeric matrix** ``X_struct`` — the 6 powerstats (clean int64, no
  nulls) plus substance categoricals integer-encoded. High-cardinality
  categoricals (species, race, power_type, power_source, combat_style) are capped
  to the top-K levels (``TOP_K_LEVELS`` = 20) with the rest mapped to ``OTHER``
  and missing to ``MISSING``; low-cardinality ones (alignment, gender) are encoded
  directly. Category->code maps are learned on TRAIN only and applied to any
  frame, so the encoding is split-safe (unseen levels fall into ``OTHER``).
* **Split-safe TF-IDF text block** — a :class:`TextBlock` wrapping a
  ``TfidfVectorizer(max_features=300, min_df=5, stop_words="english")`` over
  ``about + " " + abilities``. ``fit(train_idx)`` fits on TRAIN rows only;
  ``transform(idx)`` featurizes any rows. This enables the A1.6 text-vs-stats
  ablation as two numeric feature blocks feeding the same classifier.
* **Label vector** ``y`` from ``publisher``.
* **Feature names** and the **dropped-column list with reasons**.

Leakage guard
-------------
:func:`load_frame` FAILS LOUDLY (raises :class:`LeakageError`) if any drop-leak
column (publisher, source, first_appearance.*, voice_acting.*,
native_name/name/full_name/id, relationships.*, origin places,
basic_info.nationality, personality.alignment, powerstats.total) survives into
the structural feature names.

Run mode
--------
``python src/dataprep.py --report``
    Print X shape, the KEEP and DROP lists, class counts, and write
    ``results/class_balance.csv``. No TabPFN calls (free). This IS the
    verification (per project policy, no unit tests).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Paths -----------------------------------------------------------------
# Read-only source CSV, referenced by absolute path. NEVER copied into worktree.
DATA_CSV = Path(
    "/home/sierra/Desktop/projects/hackathons/tabpfn-3-5/"
    "data/char-ds/data/dataset_characters.csv"
)
_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
_CLASS_BALANCE_CSV = _RESULTS_DIR / "class_balance.csv"

# --- Modeling constants ----------------------------------------------------
LABEL_COL = "publisher"
RANDOM_STATE = 42
TOP_K_LEVELS = 20  # high-cardinality cap before mapping the tail to OTHER
_OTHER = "OTHER"
_MISSING = "MISSING"

# TF-IDF config (fit on TRAIN only; see TextBlock).
TFIDF_MAX_FEATURES = 300
TFIDF_MIN_DF = 5
TEXT_SOURCE_COLS = ["about", "abilities"]

# --- KEEP lists (character substance only) ---------------------------------
KEEP_NUMERIC = [
    "powerstats.combat",
    "powerstats.durability",
    "powerstats.intelligence",
    "powerstats.power",
    "powerstats.speed",
    "powerstats.strength",
]
# Low-cardinality categoricals: encode directly.
KEEP_CATEGORICAL_LOW = [
    "character.alignment",
    "basic_info.gender",
]
# High-cardinality categoricals: top-K then OTHER.
KEEP_CATEGORICAL_HIGH = [
    "basic_info.species",
    "basic_info.race",
    "power_attributes.power_type",
    "power_attributes.power_source",
    "power_attributes.combat_style",
]
KEEP_CATEGORICAL = KEEP_CATEGORICAL_LOW + KEEP_CATEGORICAL_HIGH
KEEP_TEXT = list(TEXT_SOURCE_COLS)


# --- DROP list with reasons ------------------------------------------------
# Patterns ending in ".*" match any column with that prefix.
DROP_REASONS: dict[str, str] = {
    "publisher": "the label",
    "source": "near-identical copy of label (anime/marvel/dc...)",
    "first_appearance.*": "which medium column is populated reveals universe",
    "voice_acting.*": "dub/VO metadata encodes anime vs western",
    "native_name": "character identity = direct label lookup",
    "name": "character identity = direct label lookup",
    "full_name": "character identity = direct label lookup",
    "id": "character identity = direct label lookup",
    "relationships.*": "named allies/enemies leak universe by proper noun; sparse/noisy",
    "origin.place_of_birth": "universe-specific fictional place leaks universe",
    "origin.place": "universe-specific fictional place leaks universe",
    "origin.base": "universe-specific fictional place leaks universe",
    "origin.base_of_operations": "universe-specific fictional place leaks universe",
    "origin.current": "universe-specific fictional place leaks universe",
    "origin.current_location": "universe-specific fictional place leaks universe",
    "misc.headquarters": "universe-specific fictional place leaks universe",
    "basic_info.nationality": "real-world geo, 36 non-null, leak-prone",
    "personality.alignment": "1 non-null (empty)",
    "powerstats.total": "deterministic sum of the 6 stats (redundant)",
}

# Explicit leak columns/patterns the guard must prove are absent from features.
# (A superset-of-concern check: any of these appearing in feature names is fatal.)
_LEAK_PATTERNS = [
    "publisher",
    "source",
    "first_appearance.*",
    "voice_acting.*",
    "native_name",
    "name",
    "full_name",
    "id",
    "relationships.*",
    "origin.place_of_birth",
    "origin.place",
    "origin.base",
    "origin.base_of_operations",
    "origin.current",
    "origin.current_location",
    "misc.headquarters",
    "basic_info.nationality",
    "personality.alignment",
    "powerstats.total",
]


class LeakageError(RuntimeError):
    """Raised when a known label-leaking column survives into feature names."""


def _matches_pattern(column: str, pattern: str) -> bool:
    """True if column equals pattern, or (for 'prefix.*') shares the prefix."""
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return column == prefix or column.startswith(prefix + ".")
    return column == pattern


def _is_leak_column(column: str) -> bool:
    return any(_matches_pattern(column, pat) for pat in _LEAK_PATTERNS)


@dataclass
class TextBlock:
    """Split-safe TF-IDF featurizer over about+abilities.

    Fit the vectorizer on TRAIN rows only, then transform any row index set. Keeps
    the fit/transform boundary explicit so no test-set text leaks into the IDF.
    """

    corpus: pd.Series  # full-length (per-row) joined text, aligned to the frame
    vectorizer: TfidfVectorizer | None = None
    feature_names: list[str] = field(default_factory=list)

    def fit(self, train_index: np.ndarray | pd.Index) -> "TextBlock":
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            stop_words="english",
        )
        self.vectorizer.fit(self.corpus.loc[train_index].tolist())
        self.feature_names = [
            f"tfidf::{tok}" for tok in self.vectorizer.get_feature_names_out()
        ]
        return self

    def transform(self, index: np.ndarray | pd.Index) -> np.ndarray:
        if self.vectorizer is None:
            raise RuntimeError("TextBlock.transform called before fit().")
        return self.vectorizer.transform(self.corpus.loc[index].tolist()).toarray()

    def fit_transform(self, train_index: np.ndarray | pd.Index) -> np.ndarray:
        return self.fit(train_index).transform(train_index)


@dataclass
class Frame:
    """The canonical modeling frame returned by :func:`load_frame`.

    Attributes
    ----------
    df : the raw loaded DataFrame (full 229 cols, read-only use only).
    X_struct : structural numeric matrix (powerstats + encoded categoricals),
        aligned row-for-row with ``df`` / ``y`` (index 0..n-1).
    struct_feature_names : column names of ``X_struct``.
    text_block : split-safe :class:`TextBlock` (NOT yet fit; downstream calls
        ``fit(train_idx)`` / ``transform(idx)`` after choosing the split).
    y : label vector from 'publisher'.
    dropped : list of (column, reason) for every dropped column actually present.
    cat_maps : learned category->code maps per categorical (for reference/debug).
    """

    df: pd.DataFrame
    X_struct: pd.DataFrame
    struct_feature_names: list[str]
    text_block: TextBlock
    y: pd.Series
    dropped: list[tuple[str, str]]
    cat_maps: dict[str, dict[str, int]]

    @property
    def feature_names(self) -> list[str]:
        """Structural + (post-fit) text feature names.

        Text names are only populated after ``text_block.fit`` has run; before a
        split is chosen this returns the structural names only.
        """
        return list(self.struct_feature_names) + list(self.text_block.feature_names)


def _encode_categorical(
    series: pd.Series,
    high_cardinality: bool,
    train_mask: np.ndarray,
) -> tuple[pd.Series, dict[str, int]]:
    """Integer-encode one categorical column, split-safe.

    The level vocabulary (and, for high-cardinality columns, the top-K selection)
    is learned from TRAIN rows only. Unseen/tail levels map to OTHER; missing maps
    to MISSING. Returns the encoded integer series (full length) and the
    value->code map.
    """
    filled = series.astype("object").where(series.notna(), _MISSING).astype(str)
    train_values = filled[train_mask]

    if high_cardinality:
        counts = train_values[train_values != _MISSING].value_counts()
        top_levels = list(counts.head(TOP_K_LEVELS).index)
        vocab = [_MISSING] + top_levels + [_OTHER]
    else:
        levels = sorted(v for v in train_values.unique() if v != _MISSING)
        vocab = [_MISSING] + levels + [_OTHER]

    code_map = {level: i for i, level in enumerate(vocab)}
    other_code = code_map[_OTHER]
    allowed = set(vocab)
    mapped = filled.map(lambda v: code_map.get(v, other_code) if v in allowed else other_code)
    return mapped.astype(np.int64), code_map


def load_frame(csv_path: Path | str = DATA_CSV, train_mask: np.ndarray | None = None) -> Frame:
    """Load the CSV, apply KEEP/DROP, and return the canonical :class:`Frame`.

    Parameters
    ----------
    csv_path : absolute path to the read-only source CSV.
    train_mask : optional boolean array over all rows marking the TRAIN subset
        used to learn categorical vocabularies. If None, ALL rows are treated as
        TRAIN for encoding (fine for the ``--report`` summary); downstream Axis
        scripts should pass a real train mask so the structural encoding is
        split-safe. The TF-IDF :class:`TextBlock` is always fit later with an
        explicit train index, independent of this.

    Raises
    ------
    LeakageError
        If any known leak column survives into the structural feature names.
    FileNotFoundError
        If the CSV is missing at ``csv_path``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Source CSV not found (read-only): {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)

    if LABEL_COL not in df.columns:
        raise RuntimeError(f"Label column '{LABEL_COL}' not found in {csv_path}.")
    y = df[LABEL_COL].astype(str)

    n_rows = len(df)
    if train_mask is None:
        train_mask = np.ones(n_rows, dtype=bool)
    else:
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape[0] != n_rows:
            raise ValueError(
                f"train_mask length {train_mask.shape[0]} != n_rows {n_rows}"
            )

    # --- Structural numeric matrix -----------------------------------------
    struct_cols: dict[str, pd.Series] = {}
    cat_maps: dict[str, dict[str, int]] = {}

    # Powerstats: present, clean int64, no nulls. Keep as-is.
    for col in KEEP_NUMERIC:
        if col not in df.columns:
            raise RuntimeError(f"Expected numeric KEEP column missing: {col}")
        struct_cols[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int64)

    # Categoricals: integer-encode (low direct, high top-K+OTHER), split-safe.
    for col in KEEP_CATEGORICAL:
        if col not in df.columns:
            raise RuntimeError(f"Expected categorical KEEP column missing: {col}")
        high = col in KEEP_CATEGORICAL_HIGH
        encoded, code_map = _encode_categorical(df[col], high, train_mask)
        struct_cols[col] = encoded
        cat_maps[col] = code_map

    X_struct = pd.DataFrame(struct_cols, index=df.index)
    struct_feature_names = list(X_struct.columns)

    # --- Leakage guard (fail loudly) ---------------------------------------
    offenders = [c for c in struct_feature_names if _is_leak_column(c)]
    if offenders:
        raise LeakageError(
            "Leak column(s) survived into structural feature names: "
            f"{offenders}. Fix the KEEP/DROP lists in dataprep.py."
        )

    # --- Text block (split-safe; fit later by caller) ----------------------
    for col in TEXT_SOURCE_COLS:
        if col not in df.columns:
            raise RuntimeError(f"Expected text KEEP column missing: {col}")
    joined_text = (
        df["about"].fillna("").astype(str) + " " + df["abilities"].fillna("").astype(str)
    )
    text_block = TextBlock(corpus=joined_text)

    # --- Dropped-column list with reasons (only those actually present) -----
    dropped: list[tuple[str, str]] = []
    for pattern, reason in DROP_REASONS.items():
        if pattern.endswith(".*"):
            matched = sorted(c for c in df.columns if _matches_pattern(c, pattern))
            for c in matched:
                dropped.append((c, reason))
        elif pattern in df.columns:
            dropped.append((pattern, reason))

    return Frame(
        df=df,
        X_struct=X_struct,
        struct_feature_names=struct_feature_names,
        text_block=text_block,
        y=y,
        dropped=dropped,
        cat_maps=cat_maps,
    )


def _report() -> int:
    """Print shape, KEEP/DROP lists, class counts; write class_balance.csv."""
    frame = load_frame()
    n_rows, n_struct = frame.X_struct.shape

    print("=" * 70)
    print("Canon Detector — dataprep --report")
    print("=" * 70)
    print(f"Source CSV: {DATA_CSV}")
    print(f"Rows: {n_rows}  |  structural features: {n_struct}")
    print(
        f"Text block: TF-IDF over {TEXT_SOURCE_COLS} "
        f"(max_features={TFIDF_MAX_FEATURES}, min_df={TFIDF_MIN_DF}, "
        "stop_words='english'; fit on TRAIN only downstream)"
    )
    print()

    print("KEEP — numeric (6 powerstats):")
    for c in KEEP_NUMERIC:
        print(f"  + {c}")
    print("KEEP — categoricals (low-cardinality, direct encode):")
    for c in KEEP_CATEGORICAL_LOW:
        print(f"  + {c}  (levels={frame.df[c].nunique(dropna=True)})")
    print(f"KEEP — categoricals (high-cardinality, top-{TOP_K_LEVELS}+OTHER):")
    for c in KEEP_CATEGORICAL_HIGH:
        print(f"  + {c}  (levels={frame.df[c].nunique(dropna=True)})")
    print("KEEP — text (TF-IDF source):")
    for c in KEEP_TEXT:
        print(f"  + {c}")
    print()

    print(f"DROP — {len(frame.dropped)} columns (leak/low-signal), with reasons:")
    for col, reason in frame.dropped:
        print(f"  - {col}: {reason}")
    print()

    # Leakage guard proof (already enforced in load_frame; restate here).
    offenders = [c for c in frame.struct_feature_names if _is_leak_column(c)]
    print(
        "Leakage guard: "
        + ("PASS — no leak column in features." if not offenders else f"FAIL — {offenders}")
    )
    print()

    print(f"Class counts ('{LABEL_COL}'):")
    counts = frame.y.value_counts()
    for cls, n in counts.items():
        print(f"  {cls:>16}: {n:>5}  ({n / n_rows:6.2%})")
    print()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    balance = (
        counts.rename_axis(LABEL_COL)
        .reset_index(name="count")
        .assign(fraction=lambda d: (d["count"] / n_rows).round(6))
    )
    balance.to_csv(_CLASS_BALANCE_CSV, index=False)
    print(f"Wrote class balance -> {_CLASS_BALANCE_CSV}")

    assert n_rows == 2264, f"Expected 2264 rows, got {n_rows}"
    print("Row-count assertion OK (2264).")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canon Detector data prep.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print shape/KEEP/DROP/class counts and write results/class_balance.csv.",
    )
    args = parser.parse_args()
    if args.report:
        raise SystemExit(_report())
    parser.print_help()
