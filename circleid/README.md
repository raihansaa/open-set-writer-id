# CircleID Writer-Identification — Reproduction Archive

Curated, paper-aligned source archive for the **CircleID** half of the WML 2026 paper
*"Cross-Dataset Transfer in Open-Set Writer Identification: From Hand-Drawn Circles to
Handwritten Pages"* — accepted at [ICDAR WML 2026](https://icdarwml2026.midasoc.org/); see the [repository root](../README.md) for the full project.

This folder isolates **only** the code that the paper actually references to reproduce the
CircleID private-leaderboard result, organised exactly as the paper narrates it:
**from the competition-day submission (47.02%) to the final post-deadline configuration
(65.166%)** — paper §4 / **Table 2**.

Every script here corresponds to a **Table 2 row** (or the §4.2 sub-prototype alternative).
Experimental dead-ends that the paper does **not** mention (HDBSCAN, k-reciprocal, ProSub,
SGR, VLAD-refit, multi-seed vote, TTA, power-norm, sub-center ArcFace, RMD, …) are **not**
copied here — they remain in the original `fresh_start/` tree (see §6).

> The CVL and Historical-WI transfer experiments (paper §5–§7) live in the sibling [`../cvl-hwi-writerid-method/`](../cvl-hwi-writerid-method/) and
> [`../splits/`](../splits/) folders of this repository — this `circleid/` folder is the hand-drawn-circles half only.

The scripts are the originals with a single **path-standardization** change: their data and
embedding folders now resolve from the `CIRCLEID_DATA` / `CIRCLEID_EMB` environment variables
(script-relative defaults otherwise), so they run from any directory. See [MANIFEST.md](MANIFEST.md)
for the per-file table.

**What ships in this folder:** the **14 reproduction scripts** (`precompetition/` + `post_competition/`) plus this README and `MANIFEST.md` — code and docs only (~0.4 MB). The competition **data**, trained **embeddings / checkpoints**, and reference submission CSVs are **not redistributed**; §4 lists where each lives and how to regenerate it. 

---

## 1. What the pipeline is (paper §4.1)

A single **embedding trunk** at inference, trained with auxiliary branches — a polar
encoder + pen-contrastive/pen-classification heads + outlier exposure — that shape the
trunk during training but are dropped at test time (see §5; paper §4.1, §5.3):

```
224x224 input ─► DINOv2 ViT-B/14-reg (FROZEN)
                   + LoRA (r=16, α=32) on last 6 blocks
                 ─► NetVLAD (K=64, ink-foreground tokens) ⊕ CLS token
                 ─► proj → 512-d, L2-normalised embedding  e
                 ─► single-prototype ArcFace (s=30, margin 0→0.3)
                 ─► open-set rule (PCA-64 Mahalanobis → cluster-OOD cascade)
```

Competition submission used **RGB** input; the post-deadline gains came from the
**skeleton distance-transform** input + a **cluster-level open-set rejection** cascade.

---

## 2. The reproduction chain (paper Table 2)

Each row is one cumulative refinement. **Score = OS-Top-1 on the CircleID private Kaggle
leaderboard** (42.81% of the test set are unseen writers; random ≈ 0.30).

| # | Paper change | Score | Script (this archive) |
|--:|---|--:|---|
| 1 | DINOv2+LoRA+NetVLAD+ArcFace+PCA-Maha — **RGB final submission** | **47.02** | `precompetition/train_v4.py` → `precompetition/submit_v4_advanced.py` |
| 2 | + Skeleton distance-transform input | 52.1 | `post_competition/train.py --use-skeleton` → `post_competition/submit.py` |
| 3 | + Two-seed ensemble (seeds 42 + 137) | 52.1 | `post_competition/train.py` (both seeds) |
| 4 | + Per-writer min-Mahalanobis refinement | 52.1 | `post_competition/perwriter_maha_submit.py` |
| 5 | + Cluster all test (K=44), per-cluster Maha | 58.6 | `post_competition/cluster_ood_submit.py` |
| 6 | + Keep top-6 clusters, reject rest as −1 | 62.3 | `post_competition/cluster_ood_variants.py` |
| 7 | + One writer per cluster (centroid→nearest proto) | 63.2 | `post_competition/cluster_writer_assign.py` |
| 8 | + Per-sample gating (T=0.10) | **63.220** | `post_competition/cluster_gated_assign.py` |
| 9 | + PCA-Mahalanobis / sub-prototype re-scoring *(matched, reverted)* | 64.996 | `post_competition/cluster_pca_maha.py` · `cluster_subproto.py` |
| 10 | + **K-sweep tuning (K=60, keep-7)** | **64.996** | `post_competition/cluster_k_sweep.py` |
| 11 | + Positive–negative prototype gate (m=0) | 65.045 | `post_competition/cluster_posthoc_3experiments.py` |
| 12 | + **CV-calibrated 7-feature rejection scorer** | **65.166** | `post_competition/cluster_cv_calibrated_scorer.py` |

**Headline:** Row 12 = **0.65166 private** > published winner I-Signing **0.648**.
Competition-day rank: **9 / 113** at 47.02% (paper footnote 1).
The two biggest jumps are **skeleton-DT (Row 2, +5.1pp)** and the
**cluster K-sweep (Rows 5→10, 52.1→64.996 — the largest single gain)**.

---

## 3. Folder layout

```
circleid/
├── README.md                 ← this file (Table 2 map + reproduction modes)
├── MANIFEST.md               ← every script: Table-2 row, score, role
├── precompetition/           ← Row 1, the RGB 47.02% competition submission
│   ├── train_v4.py
│   └── submit_v4_advanced.py
└── post_competition/         ← Rows 2–12, the winning path → 65.166%  (12 scripts)
    ├── train.py                       (Rows 2–3; --use-skeleton = Row 2)
    ├── submit.py                       (Row 2 baseline, max-cos unk70 = 0.521)
    ├── perwriter_maha_submit.py        (Row 4)
    ├── cluster_ood_submit.py           (Row 5)
    ├── cluster_ood_variants.py         (Row 6)
    ├── cluster_writer_assign.py        (Row 7)
    ├── cluster_gated_assign.py         (Row 8  → 0.63220)
    ├── cluster_pca_maha.py             (Row 9, PCA-Maha re-scoring — reverted)
    ├── cluster_subproto.py             (Row 9, sub-prototype re-scoring — reverted)
    ├── cluster_k_sweep.py              (Row 10 → 0.64996)
    ├── cluster_posthoc_3experiments.py (Row 11 → 0.65045, PN gate)
    └── cluster_cv_calibrated_scorer.py (Row 12 → 0.65166  ★ FINAL)
```

---

## 4. How to reproduce the scores

The post-hoc scripts (Rows 4–12) are **pure** (numpy / pandas / scikit-learn): they read the
**frozen trunk embeddings** + the data **CSVs** (not the images) and re-emit the submission
CSVs **deterministically** (fixed seeds, deterministic `AgglomerativeClustering`). Because only
their data/embedding-path lines changed, they produce identical predictions.

### Data + embeddings — two environment variables

Every script resolves two directories, each overridable by an environment variable (with script-relative defaults otherwise):

- **`CIRCLEID_DATA`** — folder holding `train.csv` / `additional_train.csv` / `test.csv` (+ `images/`). Default: `circleid/icdar-2026-circleid-writer-identification/`.
- **`CIRCLEID_EMB`** — folder holding `embeddings_seed_42.npz` + `embeddings_seed_137.npz`. Default: the script's own `outputs_skeleton/` (post-competition scripts) or `outputs_v4/` (Row 1).

Set them once and run any script from any directory:

```bash
export CIRCLEID_DATA=/path/to/icdar-2026-circleid-writer-identification
export CIRCLEID_EMB=/path/to/skeleton-embeddings   # folder with embeddings_seed_42.npz + _137.npz
python post_competition/cluster_cv_calibrated_scorer.py   # Row 12, the final 0.65166
```

### Inputs the post-hoc chain needs (obtained separately; not in this repo)

| Input | Size |
|---|---|
| Data CSVs (train / additional_train / test) | ~2 MB |
| Skeleton trunk embeddings (seeds 42, 137) | ~195 MB |
| Skeleton checkpoints (for re-extraction only) | ~925 MB |
| RGB (v4) embeddings — Row 1 | ~195 MB |
| Circle images (for full retrain only) | ~510 MB |

> NPZ schema: `train_emb (34650,512)` per-patch, `train_labels (34650,)`,
> `test_emb (5905,512)` pooled, `test_cosine (5905,44)`. Embedding dim is **512** (projection
> head), not raw 768-d DINOv2.

### Mode A — post-hoc cascade, CPU, seconds (recommended)

With `CIRCLEID_DATA` and `CIRCLEID_EMB` set (see above), run any post-hoc script from any
directory — each reads the frozen embeddings + CSVs and writes its submission CSV under its own folder:

```bash
cd post_competition     # the post-hoc scripts (Rows 4-12) live here
python cluster_gated_assign.py            # Row 8  → submissions_gated_assign/gated_T0100.csv   (0.63220)
python cluster_k_sweep.py                 # Row 10 → submissions_k_sweep/K060_keep07_gated.csv  (0.64996)
python cluster_posthoc_3experiments.py    # Row 11 → submissions_posthoc_3exp/03_pn_gate_m000.csv (0.65045)
python cluster_cv_calibrated_scorer.py    # Row 12 → submissions_cv_scorer/A_reject_pk_lt_025.csv (0.65166)
```

Each run rewrites its `submissions_*/` CSVs; diff against the preserved reference CSVs in §2 to
confirm an exact match. This archive is the **index** of which script produces which Table-2 row;
each script writes its submission CSV next to itself — no `fresh_start/` layout needed.

### Mode B — full retrain from images (GPU, ~5–8 h on RTX 4060)

```bash
# Row 1 — RGB competition trunk (47.02%)
python precompetition/train_v4.py
python precompetition/submit_v4_advanced.py      # → submission_*_pca64_unk70pct.csv

# Row 2 — skeleton-DT trunk (52.1%), two seeds
python post_competition/train.py --use-skeleton --seed 42
python post_competition/train.py --use-skeleton --seed 137
python post_competition/submit.py                 # max-cos unk70 baseline = 0.521

# Rows 4–12 — post-hoc cascade on the freshly-extracted embeddings (as in Mode A)
```

> The freshly-extracted embeddings land in each trainer's output folder; point `CIRCLEID_EMB` at
> them (and `CIRCLEID_DATA` at the competition data), and the Rows 4-12 scripts pick them up from
> any directory — no fixed layout required.

---

## 5. Key hyper-parameters (paper §4.1)

- **Backbone:** DINOv2 ViT-B/14-reg, frozen. **LoRA** r=16, α=32 on QKV of the last **6** blocks.
- **Aggregation:** NetVLAD K=64, k-means init, ink-foreground tokens only (thr 0.1), ⊕ CLS → 512-d L2.
- **Loss:** single-prototype ArcFace, s=30, margin ramp 0→0.3 over 15 epochs; + pen-contrastive /
  pen-classification aux heads + outlier exposure on additional-train unknowns.
- **Input (Row 2):** skeleton-DT = Otsu binarise → `cv2.distanceTransform` → 3-channel tile.
- **Optim:** AdamW (wd 0.01), LoRA lr 5e-5 / heads lr 5e-4, 40 epochs, batch 48 (grad-accum 2 ≈ 96),
  5-ep warmup + cosine, AMP, EMA 0.999, two-seed ensemble.
- **Base OOD rule:** PCA-64 Mahalanobis (per-writer means, pooled Ledoit–Wolf covariance),
  fixed-ratio rank cut — reject farthest 70%.
- **Final OOD rule:** Agglomerative (cosine, average linkage), K=60, keep top-7 clusters,
  per-sample gating T=0.10, PN-gate (m=0), CV-calibrated 7-feature LR scorer (reject if p_known < 0.25).

---

## 6. Not included (intentionally)

- **Experiments not named in the paper.** ~36 documented post-hoc dead-ends (HDBSCAN,
  k-reciprocal, ProSub, SGR, VLAD-refit, multi-seed vote, 3-seed, TTA, adaptive gating,
  iterative-proto, OOD anchors, linkage probe, exemplar-SVM, softmax-max, transductive,
  fine-tune variants, power-norm, sub-center-ArcFace + morph, CV-recover, RMD scorer,
  ensemble vote) and the `verify_embeddings.py` sanity utility — all remain in the original
  `fresh_start/` directory. The WML paper does not reference them, so they are out of scope here.
- **CVL / Historical-WI transfer pipeline** (paper §5–§7): now in the sibling [`../cvl-hwi-writerid-method/`](../cvl-hwi-writerid-method/) and [`../splits/`](../splits/) folders (this folder is circles-only).
- **Large binary artifacts** (`.npz`, `.pt`, the image dataset, reference CSVs): left in place;
  §4 lists exactly where each lives. This archive is scripts + docs only (~0.4 MB).
