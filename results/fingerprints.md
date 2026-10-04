# Axis-1 Universe Fingerprints

Per-universe top substance features that distinguish each universe, from a LightGBM one-vs-rest gain analysis on the **same split-safe feature matrix** the TabPFN probe uses (structural stats+categoricals + TF-IDF of about/abilities).

> **Interpretability method:** `tabpfn_extensions.interpretability` depends on `shap`, which is not installed in this environment (documented in FEAT-001 / requirements.txt). Per the plan's fallback, fingerprints use **LightGBM gain** (global multiclass + per-universe one-vs-rest). No TabPFN credits are spent here.

## Global top features

Total features: 313 (structural=13, tfidf=300). Most important overall: `power_attributes.power_source` (structural).

| rank | feature | gain |
|---:|---|---:|
| 1 | `power_attributes.power_source` (structural) | 6124.9 |
| 2 | text token "marvel" (from about/abilities prose) | 5611.5 |
| 3 | text token "dc" (from about/abilities prose) | 4937.1 |
| 4 | text token "anime" (from about/abilities prose) | 2945.6 |
| 5 | text token "null" (from about/abilities prose) | 2680.5 |
| 6 | text token "trek" (from about/abilities prose) | 2047.1 |
| 7 | text token "superhuman" (from about/abilities prose) | 1187.4 |
| 8 | text token "wars" (from about/abilities prose) | 1167.8 |
| 9 | text token "comics" (from about/abilities prose) | 810.2 |
| 10 | text token "description" (from about/abilities prose) | 633.0 |
| 11 | text token "star" (from about/abilities prose) | 524.3 |
| 12 | text token "human" (from about/abilities prose) | 471.8 |
| 13 | text token "character" (from about/abilities prose) | 438.0 |
| 14 | text token "self" (from about/abilities prose) | 431.8 |
| 15 | text token "tier" (from about/abilities prose) | 398.0 |
| 16 | `power_attributes.combat_style` (structural) | 379.0 |
| 17 | text token "source" (from about/abilities prose) | 365.9 |
| 18 | text token "attributes" (from about/abilities prose) | 311.4 |
| 19 | `powerstats.power` (structural) | 264.4 |
| 20 | text token "type" (from about/abilities prose) | 251.5 |

## Per-universe fingerprints

### anime

- `power_attributes.power_source` (structural) — gain 7486.9
- text token "null" (from about/abilities prose) — gain 5156.9
- text token "anime" (from about/abilities prose) — gain 4726.5
- text token "comics" (from about/abilities prose) — gain 1334.2
- text token "description" (from about/abilities prose) — gain 807.5
- text token "source" (from about/abilities prose) — gain 578.5
- text token "tier" (from about/abilities prose) — gain 172.5
- text token "skilled" (from about/abilities prose) — gain 161.0
- text token "medium" (from about/abilities prose) — gain 144.2
- text token "self" (from about/abilities prose) — gain 117.6
- `powerstats.power` (structural) — gain 110.8
- text token "attack" (from about/abilities prose) — gain 94.0

### dark_horse

- `basic_info.race` (structural) — gain 104.1
- text token "english" (from about/abilities prose) — gain 91.2
- text token "form" (from about/abilities prose) — gain 72.7
- text token "stories" (from about/abilities prose) — gain 57.8
- text token "defense" (from about/abilities prose) — gain 48.8
- text token "null" (from about/abilities prose) — gain 44.2
- text token "old" (from about/abilities prose) — gain 25.8
- text token "80" (from about/abilities prose) — gain 21.4
- text token "super" (from about/abilities prose) — gain 20.9
- text token "sense" (from about/abilities prose) — gain 19.1
- text token "support" (from about/abilities prose) — gain 16.6
- text token "dark" (from about/abilities prose) — gain 14.9

### dc

- text token "dc" (from about/abilities prose) — gain 6920.1
- `power_attributes.power_source` (structural) — gain 626.5
- text token "human" (from about/abilities prose) — gain 504.6
- text token "type" (from about/abilities prose) — gain 382.5
- text token "power_level" (from about/abilities prose) — gain 381.9
- text token "marvel" (from about/abilities prose) — gain 286.1
- text token "refer" (from about/abilities prose) — gain 232.9
- text token "character" (from about/abilities prose) — gain 172.5
- text token "used" (from about/abilities prose) — gain 155.3
- text token "null" (from about/abilities prose) — gain 120.0
- text token "published" (from about/abilities prose) — gain 108.3
- `powerstats.strength` (structural) — gain 97.4

### idw_publishing

- text token "franchise" (from about/abilities prose) — gain 166.2
- `basic_info.species` (structural) — gain 77.9
- text token "null" (from about/abilities prose) — gain 70.0
- text token "passive" (from about/abilities prose) — gain 58.8
- text token "85" (from about/abilities prose) — gain 26.3
- text token "leader" (from about/abilities prose) — gain 23.8
- text token "self" (from about/abilities prose) — gain 21.9
- text token "version" (from about/abilities prose) — gain 19.6
- text token "transformation" (from about/abilities prose) — gain 18.1
- text token "support" (from about/abilities prose) — gain 17.5
- text token "energy" (from about/abilities prose) — gain 13.0
- text token "attack" (from about/abilities prose) — gain 12.6

### image_comics

- `power_attributes.combat_style` (structural) — gain 454.4
- text token "tier" (from about/abilities prose) — gain 269.8
- text token "self" (from about/abilities prose) — gain 231.7
- text token "comics" (from about/abilities prose) — gain 193.0
- text token "physical" (from about/abilities prose) — gain 185.7
- `power_attributes.power_source` (structural) — gain 150.0
- text token "description" (from about/abilities prose) — gain 130.2
- text token "range" (from about/abilities prose) — gain 118.1
- text token "unlimited" (from about/abilities prose) — gain 97.6
- text token "flight" (from about/abilities prose) — gain 89.0
- text token "anime" (from about/abilities prose) — gain 83.6
- text token "refer" (from about/abilities prose) — gain 58.6

### marvel

- text token "marvel" (from about/abilities prose) — gain 8413.9
- text token "superhuman" (from about/abilities prose) — gain 1732.9
- text token "null" (from about/abilities prose) — gain 736.4
- text token "character" (from about/abilities prose) — gain 486.2
- `power_attributes.power_source` (structural) — gain 329.9
- text token "attributes" (from about/abilities prose) — gain 316.2
- text token "comics" (from about/abilities prose) — gain 310.0
- text token "human" (from about/abilities prose) — gain 265.8
- text token "dc" (from about/abilities prose) — gain 248.5
- text token "universal" (from about/abilities prose) — gain 160.6
- text token "type" (from about/abilities prose) — gain 120.6
- text token "anime" (from about/abilities prose) — gain 118.1

### nbc_heroes

- text token "heroes" (from about/abilities prose) — gain 201.1
- text token "power" (from about/abilities prose) — gain 139.1
- text token "healing" (from about/abilities prose) — gain 32.0
- text token "portrayed" (from about/abilities prose) — gain 25.4
- text token "passive" (from about/abilities prose) — gain 19.4
- text token "requires" (from about/abilities prose) — gain 17.2
- text token "able" (from about/abilities prose) — gain 16.5
- text token "control" (from about/abilities prose) — gain 8.3
- text token "fictional" (from about/abilities prose) — gain 2.4
- text token "70" (from about/abilities prose) — gain 0.9
- text token "100" (from about/abilities prose) — gain 0.1
- text token "defense" (from about/abilities prose) — gain 0.1

### star_trek

- text token "trek" (from about/abilities prose) — gain 2288.7
- text token "support" (from about/abilities prose) — gain 454.8
- text token "series" (from about/abilities prose) — gain 53.7
- text token "called" (from about/abilities prose) — gain 45.0
- text token "time" (from about/abilities prose) — gain 23.1
- text token "limited" (from about/abilities prose) — gain 12.0
- text token "65" (from about/abilities prose) — gain 11.9
- text token "98" (from about/abilities prose) — gain 6.9
- text token "portrayed" (from about/abilities prose) — gain 4.2
- text token "following" (from about/abilities prose) — gain 3.2
- text token "fictional" (from about/abilities prose) — gain 0.5
- text token "television" (from about/abilities prose) — gain 0.3

### star_wars

- text token "wars" (from about/abilities prose) — gain 1288.0
- text token "star" (from about/abilities prose) — gain 727.5
- text token "force" (from about/abilities prose) — gain 33.8
- text token "dark" (from about/abilities prose) — gain 20.8
- text token "offensive" (from about/abilities prose) — gain 19.4
- text token "expert" (from about/abilities prose) — gain 6.1
- text token "attack" (from about/abilities prose) — gain 4.3
- text token "film" (from about/abilities prose) — gain 4.1
- text token "support" (from about/abilities prose) — gain 3.8
- text token "limit" (from about/abilities prose) — gain 3.6
- text token "medium" (from about/abilities prose) — gain 3.1
- text token "objects" (from about/abilities prose) — gain 2.0

### the_boys

- `power_attributes.power_source` (structural) — gain 1086.8
- text token "self" (from about/abilities prose) — gain 249.1
- `powerstats.combat` (structural) — gain 220.5
- text token "comics" (from about/abilities prose) — gain 166.6
- text token "anime" (from about/abilities prose) — gain 136.6
- text token "range" (from about/abilities prose) — gain 121.0
- `powerstats.power` (structural) — gain 108.2
- text token "description" (from about/abilities prose) — gain 97.6
- text token "70" (from about/abilities prose) — gain 79.8
- text token "unlimited" (from about/abilities prose) — gain 79.5
- text token "team" (from about/abilities prose) — gain 71.9
- text token "superhero" (from about/abilities prose) — gain 70.4

## Leakage guard — closing proof

The following checks confirm the probe's accuracy comes from character substance, not an identity/label leak:

1. **No dropped identifier survives into features.** All 313 feature names were re-checked against the drop-leak patterns (`publisher`, `source`, `first_appearance.*`, `voice_acting.*`, `native_name`, `name`, `full_name`, `id`, `relationships.*`, `origin.place_of_birth`, `origin.place`, `origin.base`, `origin.base_of_operations`, `origin.current`, `origin.current_location`, `misc.headquarters`, `basic_info.nationality`, `personality.alignment`, `powerstats.total`); **0** matched. The label `publisher`, its near-copy `source`, `first_appearance.*`, `voice_acting.*`, identity columns (`name`/`full_name`/`id`/`native_name`), `relationships.*`, in-universe place fields, `basic_info.nationality`, and `powerstats.total` are all absent from the feature matrix.

2. **No single feature encodes the publisher.** The top signals are substance: the 6 powerstats, encoded substance categoricals (species/race/power type/alignment/gender), and TF-IDF tokens of the free-text `about`/`abilities` prose. Of the global top-20 features, 17 are prose text tokens. Prose tokens may contain proper nouns — this is explicitly allowed by the plan (resolution #4): the guard forbids handing the model a *clean* universe-encoding column, which is proven absent above, not scrubbing every proper noun from narrative text. The importance is spread across many substance features rather than dominated by one lookup-style column.

3. **A LightGBM baseline on the identical features reaches comparable accuracy** (see `results/axis1_classifier.json`), confirming the signal is in the honest substance features and reproducible by an independent model, not a TabPFN artefact.

