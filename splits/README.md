# Open-Set Writer-ID Splits — CVL & Historical-WI

Deterministic, leakage-free **open-set** splits for the two page-level datasets in the paper, the
builders that generate them, and a convention-agnostic verifier. Every experiment in
[`../cvl-hwi-writerid-method/`](../cvl-hwi-writerid-method/) reads these CSVs via `--splits-dir`.

## What "open-set" means here

Standard writer-ID splits assume every test writer was already seen at enrollment (closed-set).
Real deployments face **unknown** writers. We cast each dataset into a **three-group protocol**
(after Dr. Christlein's design):

- **Known** — writers enrolled *and* queried. Each contributes **3 enrollment (train) + 1 val + 1 test** page. These are the writers a system must identify.
- **Pseudo-unknown** — writers that appear **only in `val`**, all pages labelled `-1`. Used to tune the open-set rejection threshold *without* touching test.
- **Unknown** — writers that appear **only in `test`**, all pages labelled `-1`. Never seen at training, enrollment, or threshold-tuning — the true open-set challenge.

`writer_id = -1` marks any unknown page (pseudo or true), and **no page appears in more than one split**.

## How the splits are built

Each writer's **role is decided from its page count** — no manual assignment — then a fixed seed
splits the ambiguous pool. Both builders are fully deterministic (`--seed 0`).

**CVL** (`build_cvl_splits.py`) — cropped, handwriting-only pages; writers have 5 or 7 pages:
- **7-page** writers (those with both English *and* German samples) → **Unknown** (all pages → `test.csv`, `-1`).
- **5-page** writers → **Known / Pseudo-unknown**: `--pseudo-unknown-frac 0.05` → ⌊0.05 × 283⌋ = **14** Pseudo-unknown (all pages → `val.csv`, `-1`); the remaining **269** are **Known** with a **3 / 1 / 1** train/val/test page split.

**Historical-WI** (`build_hwi_splits.py`) — ICDAR-2017 binarized (ScriptNet):
- **5-page** writers (the 720 ScriptNet-test writers) → **Known**, **3 / 1 / 1** split.
- **fewer-page** writers (the 394 ICDAR-train writers, ~3 pages each) → unknown pool: `--pseudo-unknown-frac 0.30` → ⌊0.30 × 394⌋ = **118** Pseudo-unknown (→ `val.csv`, `-1`); the remaining **276** → **Unknown** (→ `test.csv`, `-1`).

Writers with an unexpected page count are skipped with a warning (none occur in either dataset).

## The resulting splits

### CVL (`cvl_splits/`) — 269 Known · 14 Pseudo-unknown · 27 Unknown writers

| Split | Known pages | Unknown (`-1`) pages | Total |
|-------|------------:|---------------------:|------:|
| train | 807  (269 × 3) | 0                     | **807**  |
| val   | 269  (269 × 1) | 70  (14 pseudo × 5)   | **339**  |
| test  | 269  (269 × 1) | 189 (27 unknown × 7)  | **458**  |

### Historical-WI (`hwi_splits/`) — 720 Known · 118 Pseudo-unknown · 276 Unknown writers

| Split | Known pages | Unknown (`-1`) pages | Total |
|-------|------------:|---------------------:|------:|
| train | 2160 (720 × 3) | 0                      | **2160** |
| val   | 720  (720 × 1) | 354 (118 pseudo × 3)   | **1074** |
| test  | 720  (720 × 1) | 828 (276 unknown × 3)  | **1548** |

## File format

Each `*_splits/` folder holds:

```
train.csv / val.csv / test.csv     # columns: image_id, image_path, writer_id   (writer_id = -1 → unknown)
writers_known.txt                  # <prefixed_id>\t<original_id>   per Known writer
writers_pseudo_unknown.txt         # ... Pseudo-unknown writers
writers_unknown.txt                # ... Unknown writers
```

Known writer IDs are prefixed (`C####` for CVL, `T####` for HWI); unknowns are `-1`. `image_path`
is relative to the dataset's image root (the method resolves it via `--data-dir`).

## (Re)build and verify

```bash
# rebuild — needs the raw images (see ../cvl-hwi-writerid-method/README.md#data)
python build_cvl_splits.py --images-root /path/to/cvl --out-root cvl_splits --pseudo-unknown-frac 0.05 --seed 0
python build_hwi_splits.py --images-root /path/to/hwi --out-root hwi_splits --pseudo-unknown-frac 0.30 --seed 0

# verify structural integrity (no page leakage, identical known set across splits, 3/1/1, disjoint pools)
python verify_splits.py cvl_splits hwi_splits
```

`verify_splits.py` is **convention-agnostic**: it auto-detects the known population and the
per-writer allocation and asserts the open-set invariants, so it works unchanged on any dataset's
splits folder (it reports `10/10 passed` on both sets here).
