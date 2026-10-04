# Canon Detector

A TabPFN-3.5 hackathon entry that treats comic/anime characters as points in a
"universe space" and does two things with it:

- **Axis 1: a calibrated cross-universe plausibility probe (the rigor gate).**
  Can a model tell which universe a character belongs to from their *substance*
  (powerstats, species/race/alignment, and the prose of `about`/`abilities`), and are its probabilities trustworthy? This is the gate: if the probe can't
  separate universes honestly, nothing downstream means anything.
- **Axis 2: a generate-score-refine character agent (the creative payoff).**
  Use TabPFN's unsupervised model to invent brand-new characters, then steer them
  with the Axis-1 probe toward two creative extremes: the "ultimate crossover"
  (every universe finds it equally plausible) and "lands in empty space" (no
  universe claims it), plus a gap character sitting where the real roster is
  emptiest. Every invented character is validated as novel and coherent.

Everything is standalone Python under `src/`, writing metrics/tables to
`results/` and plots to the repo root. No server, no build step.

---

## Method

### Data and the modeling frame (`src/dataprep.py`)

2,264 characters, label = `publisher` (10 "universes": anime 1524, marvel 356,
dc 232, the_boys 42, star_trek 36, image_comics 36, star_wars 24, and three tiny
classes ≤5). One `load_frame()` builds the honest modeling frame every script
reuses:

- **Structural block (13 features):** the 6 powerstats + 7 substance categoricals
  (alignment, gender, species, race, power type/source, combat style). High-
  cardinality categoricals are capped to the top-20 levels (rest → `OTHER`),
  learned on TRAIN only so the encoding is split-safe.
- **Text block (300 features):** TF-IDF of `about + abilities`, fit on TRAIN only.
- Optional **universe-token mask** (see leakage guard) redacts giveaway words from
  the prose before TF-IDF.

### Axis 1: the gate (`src/axis1_*.py`)

Stratified 75/25 split (`random_state=42`). ONE `TabPFNClassifier(model_path=
"v3.5_default", n_estimators=8)` fit, then batched `predict_proba` (never per-row).
Reported on held-out data: accuracy, balanced accuracy, macro + per-class
one-vs-rest AUC, confusion matrix, with a LightGBM baseline on the identical
split/features. Calibration (ECE + reliability curve + isotonic/Platt) reuses the
persisted held-out probabilities so it costs no extra TabPFN call. Downstream:
a famous-character plausibility table, an originality ranking, per-universe
fingerprints, and a text-vs-stats disagreement study.

### Axis 2: the agent (`src/axis2_*.py`, `src/demo.py`)

1. **Generate (`axis2_generate.py`).** `TabPFNUnsupervisedModel` is fit on the
   13-D structural matrix and samples synthetic characters.
2. **Score + refine (`axis2_agent.py`).** A bounded loop (≤5 iterations) scores
   ALL candidates in ONE batched `predict_proba` per iteration with a **stats-only**
   Axis-1 probe, computes two objectives, cross-universe **entropy** and
   **originality** `1 − max_u P(u)`, and breeds the next generation by
   jittering/recombining the top performers (free; no extra generation call).
   Two subpopulations evolve separately so the two winners are genuinely
   different characters.
3. **Gap finder (`axis2_gapfinder.py`).** PCA(2) projection of the structural
   space; flag the emptiest region and generate a character into it.
4. **Validate (`axis2_validate.py`).** Nearest-neighbour distance to the closest
   real character (novelty: assert no copies) and sane-stat-range checks
   (coherence).
5. **Demo (`src/demo.py`).** One entry point (`--target maximally-cross-universe |
   maximally-original | fill-a-gap`) runs the smallest path and prints the
   resulting character + scores + novelty/coherence. This is the project's single
   end-to-end smoke check.

---

## The leakage guard (and how it was proven)

A universe classifier is only interesting if it reads *character substance*, not a
column that already names the publisher. Two layers of guard:

### 1. Dropped columns (structural leak guard)

`load_frame()` **fails loudly** (`LeakageError`) if any of these survive into the
feature matrix:

| Dropped | Reason |
|---|---|
| `publisher` | the label |
| `source` | near-identical copy of the label (anime/marvel/dc…) |
| `first_appearance.*` (anime, comics, game, manga) | which medium column is populated reveals the universe |
| `voice_acting.*` | dub/VO metadata encodes anime vs western |
| `name`, `full_name`, `id`, `native_name` | character identity = direct label lookup |
| `relationships.*` (122 cols) | named allies/enemies leak the universe by proper noun |
| `origin.place_of_birth/base*/current*`, `misc.headquarters` | in-universe place names leak the universe |
| `basic_info.nationality` | real-world geo, leak-prone |
| `powerstats.total` | deterministic sum of the 6 stats (redundant) |

**Kept** = the 6 powerstats + 7 substance categoricals + `about`/`abilities` prose.
`src/axis1_fingerprints.py` re-asserts 0 of 313 features match any leak pattern and
writes the proof to `results/fingerprints.md`.

### 2. The text self-declaration leak (the honest headline)

The `about`/`abilities` prose literally contains universe tokens like "marvel", "dc",
"anime", "trek", "wars", "heroes", and TF-IDF reads them. So part of "predict the
universe from substance" is really "the description names its own universe." We do
NOT hide this. `src/dataprep.py` has a reproducible **universe-token mask** that
redacts those tokens from the prose before TF-IDF, and `src/axis1_classifier.py`
reports the ablation three ways:

| Probe | Held-out accuracy | What it means |
|---|---|---|
| **leaky** (full text) | **0.9647** | inflated: reads universe words in the prose |
| **masked** (tokens redacted) | **0.9488** | the **honest headline** |
| **stats-only** (no text at all) | **0.8958** | pure structural substance signal |

**Leakage drop (leaky − masked) = 0.0159.** The honest finding is strong: redacting
the giveaway words costs only ~1.6 points, so the probe is overwhelmingly reading
genuine substance, not self-declaration. Macro OVR-AUC stays **0.997** masked. And
**Axis 2 steers with the stats-only probe**, so every generated character is scored
on structural substance alone, never on leaked tokens.

---

## Key findings

- **The gate holds, honestly.** Masked-text accuracy **0.9488** (balanced 0.9312,
  macro-AUC 0.9971); stats-only **0.8958** vs a 0.6731 majority-class baseline. The
  TabPFN probe beats a LightGBM baseline decisively on balanced accuracy
  (0.9543 vs 0.6312 on the leaky features): it handles the tiny universes far
  better.
- **Calibrated.** Top-label ECE **0.0170 raw** (already well-calibrated); isotonic
  trims the held-out eval half to **0.0136**. See `reliability_curve.png`.
- **Plausibility (`results/plausibility_table.csv`, `plausibility_heatmap.png`).**
  22 famous characters scored cross-universe; rows sum to 1.0 and every argmax
  matches the true publisher, face-valid.
- **Originality (`results/originality_ranking.csv`, `originality_ranking.png`).**
  `1 − max_u P(u)` over all 1,698 train characters. Most archetypal are anime
  characters (the dominant class the probe is most certain about); the most
  "original" sit between universes.
- **Text-vs-stats (`results/text_vs_stats.csv`).** Characters whose *numbers* read
  like one universe but whose *prose* reads like another, a built-in crossover
  signal only the text-capable setup surfaces (e.g. DC characters that read
  statistically like Marvel).
- **Generated-character payoff (`results/generated_characters.csv`,
  `character_space_map.png`).** The agent produced two distinct, novel, coherent
  invented characters plus a gap character:
  - **max-entropy ("ultimate crossover"):** entropy **2.279** (max possible 2.303)
    a nearly flat P(universe) across all 10 universes; nearest real character
    Hiro Nakamura (NN dist 3.00).
  - **max-originality ("empty space"):** originality **0.853**; nearest real
    character Sein (NN dist 1.29).
  - **gap character:** sits in the emptiest PCA region; nearest real character
    Deadshot (NN dist 1.34).
  All three are **novel** (no exact training copy) and **coherent** (sane stats).

---

## Honest caveats

- **Anime is 67% of the data.** We keep all publishers as universes (per the
  design) and report per-class one-vs-rest AUC + balanced accuracy so the tiny
  classes are judged fairly, not hidden behind raw accuracy.
- **Tiny classes.** dark_horse / idw_publishing / nbc_heroes have ≤5 members;
  their per-class AUC can read 1.0 on a handful of held-out positives, treat those
  as indicative, not robust.
- **The headline is the MASKED number (0.9488), not 0.9647.** The leaky number is
  shown only to make the ~1.6-point self-declaration leak visible. Stats-only
  (0.8958) is the floor with no text at all.
- **API adaptations (confirmed live, never faked).** `TabPFNUnsupervisedModel`
  needs BOTH a classifier and a regressor; `generate_synthetic_data` samples
  **per column** (one call ≈ `n_features` metered sub-calls, ~10K credits each,
  independent of row count), so we generate on the small 13-column structural
  matrix with `n_permutations=1` and a large candidate pool per call. The local
  `tabpfn` backend (which would be free) is unavailable here: its weights aren't
  cached and the one-time license can't be accepted non-interactively, so
  generation runs on the metered API backend. `get_embeddings` is
  `NotImplementedError` in extensions 0.6.3, so the gap finder uses a PCA
  projection, not model embeddings.
- **SHAP vs fallback.** `tabpfn_extensions.interpretability` needs `shap`, which is
  not installed; fingerprints use the documented LightGBM-gain fallback instead.

---

## Reproduce

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
#    numpy pandas scikit-learn lightgbm tabpfn_client==0.6.1
#    tabpfn_extensions==0.6.3 matplotlib python-dotenv

# 2. Token: put TABPFN_TOKEN=<token> in a .env at the repo root (git-ignored,
#    never committed/printed). src/env.py loads it. Get a key at
#    https://platform.priorlabs.ai

# 3. Data: place dataset_characters.csv under data/ (git-ignored, not
#    redistributed here). Source: the public multi-publisher character dataset
#    github.com/Sidmaz666/character-dataset (2,264 characters, 10 publishers).

# 4. Run order (from the repo root so `src` is importable):
python src/dataprep.py --report            # frame + leakage guard + class balance
python src/axis1_classifier.py             # GATE + leaky/masked/stats-only ablation
python src/axis1_calibration.py            # ECE + reliability curve (no new call)
python src/axis1_plausibility.py           # famous-character table + heatmap
python src/axis1_originality.py            # originality ranking + chart
python src/axis1_fingerprints.py           # per-universe fingerprints + leak proof
python src/axis1_text_vs_stats.py          # text-vs-stats disagreements

python src/axis2_generate.py --probe       # confirm the generation API live
python src/axis2_generate.py               # -> results/_axis2_raw.csv
python src/axis2_agent.py                  # -> results/generated_characters.csv
python src/axis2_gapfinder.py              # -> character_space_map.png + gap row
python src/axis2_validate.py               # adds novelty + coherence columns
python src/demo.py --target maximally-original   # the single end-to-end smoke run
```

Determinism: `random_state=42` throughout. Verification is by running the real
scripts and inspecting their printed metrics / written files; there are no
per-function unit tests (project policy).

---

## Deliverables

- **`src/`**: `credits.py`, `env.py`, `dataprep.py`; `axis1_classifier.py`,
  `axis1_calibration.py`, `axis1_plausibility.py`, `axis1_originality.py`,
  `axis1_fingerprints.py`, `axis1_text_vs_stats.py`; `axis2_generate.py`,
  `axis2_agent.py`, `axis2_gapfinder.py`, `axis2_validate.py`, `demo.py`.
- **`results/`**: `axis1_classifier.json`, `class_balance.csv`,
  `plausibility_table.csv`, `originality_ranking.csv`, `text_vs_stats.csv`,
  `fingerprints.md`, `generated_characters.csv`, `credit_log.csv`
  (+ caches `_axis1_test_probs.npz`, `_axis2_raw.csv`).
- **Plots (Agg/headless):** `reliability_curve.png`, `plausibility_heatmap.png`,
  `originality_ranking.png`, `character_space_map.png`.
- **`requirements.txt`**, this `README.md`.

---

## Credit tally (`results/credit_log.csv`)

TabPFN billing is flat (~10K credits per metered call regardless of row count), so
every call is batched and logged. Observed per-call deltas are only:

| Delta | What |
|---|---|
| 0 | a fit (billed asynchronously / flat) or a local no-op |
| 10,000 | ONE batched `predict_proba` (Axis-1 scoring, agent/demo scoring) |
| ~130,000 | a tiny probe generation (13 columns × 1 permutation on a 40-row sample) |
| ~330,000–340,000 | a full-data generation batch (13 per-column passes) |

There is **no per-row predict loop** anywhere; the agent asserts scoring is
batched (candidate count > 1) and that the loop uses ≤5 scoring calls. The whole
project (including re-runs during development) used ~3.11M credits this session
(sum of the per-call deltas in `results/credit_log.csv`); the account meter went
from 950K before the first call to **4.45M** after the final end-to-end demo
run, well inside the 20M budget. The expensive line items are the per-column
generations; Axis-1 and all scoring are cheap.
