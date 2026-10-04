# Data

This project uses a public multi-publisher character dataset. The raw CSV is
**not redistributed here** (it belongs to its original author); fetch it yourself.

## Get the dataset

```bash
# from the repo root
git clone --depth 1 https://github.com/Sidmaz666/character-dataset.git data/char-ds
cp data/char-ds/data/dataset_characters.csv data/dataset_characters.csv
```

That's it. `src/dataprep.py` reads `data/dataset_characters.csv` by default.

## What it is

2,264 characters across 10 "universes" (publishers):

| publisher | count |
|---|---|
| anime | 1524 |
| marvel | 356 |
| dc | 232 |
| the_boys | 42 |
| star_trek | 36 |
| image_comics | 36 |
| star_wars | 24 |
| dark_horse | 5 |
| idw_publishing | 5 |
| nbc_heroes | 4 |

Every character has complete `powerstats.*` (combat, durability, intelligence,
power, speed, strength) and text fields (`about`, `abilities`). The modeling
frame (`src/dataprep.py`) keeps only character *substance* and drops every
column that would leak the publisher label — see the leakage guard in the top
level README.

Credit: dataset by github.com/Sidmaz666/character-dataset.
