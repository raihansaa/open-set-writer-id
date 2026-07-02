# MANIFEST — every script in this archive

All 14 scripts are the originals with a single **path-standardization** change — their data and
embedding folders resolve from the `CIRCLEID_DATA` / `CIRCLEID_EMB` environment variables
(script-relative defaults) so they run from any directory. Every script maps to a **WML paper Table 2 row**. "Score" = CircleID private
leaderboard OS-Top-1 (random ≈ 0.30; published winner 0.648).



---

## precompetition/ — Table 2 Row 1 (RGB, 47.02%)

| File | Role |
|---|---|
| `train_v4.py` | RGB DINOv2+LoRA+NetVLAD+ArcFace trunk; trains seeds 42/137 → `outputs_v4/embeddings_seed_*.npz`. |
| `submit_v4_advanced.py` | 6 OOD scorers on v4 embeddings; **PCA-64 Mahalanobis unk70% = 0.470** is the competition submission. |

---

## post_competition/ — Table 2 Rows 2–12 (→ 65.166%)

| File | Row | Score | Role |
|---|--:|--:|---|
| `train.py` | 2–3 | 52.1 | Trunk training. `--use-skeleton` ⇒ skeleton-DT input (Row 2). Two seeds 42/137 (Row 3). Writes `outputs_skeleton/embeddings_seed_*.npz`. |
| `submit.py` | 2 | 52.1 | Trunk baseline submission (max-cos unk70 = **0.521**) + 6-method OOD ablation. |
| `perwriter_maha_submit.py` | 4 | 52.1 | Per-writer Mahalanobis, Ledoit–Wolf shrinkage (≈0 over baseline). |
| `cluster_ood_submit.py` | 5 | 58.6 | **Cluster-level OOD** — Agglomerative K=44, score centroids by Mahalanobis, keep top-X%. The +6.5pp jump. |
| `cluster_ood_variants.py` | 6 | 62.3 | Variant sweep (keep-N, cosine vs Maha centroid scoring, PCA, KMeans) → keep top-6. |
| `cluster_writer_assign.py` | 7 | 63.2 | One writer per cluster (centroid → nearest train prototype) vs per-sample argmax. |
| `cluster_gated_assign.py` | 8 | **63.220** | Intra-cluster gating: per-sample argmax if top1−top2 margin ≥ T, else cluster-level. T=0.10 → `gated_T0100.csv`. |
| `cluster_pca_maha.py` | 9 | 64.996 | PCA-Mahalanobis re-scoring — matched K-sweep, **no gain, reverted** (paper §4.2). |
| `cluster_subproto.py` | 9 | 64.996 | Sub-prototype re-scoring (the "sub-prototype /" co-mention in §4.2) — also matched & reverted. |
| `cluster_k_sweep.py` | 10 | **64.996** | K∈{44…90} × keep-N sweep. **K=60 keep-7 gated** = `K060_keep07_gated.csv` — beats the 0.648 winner. Largest single gain. |
| `cluster_posthoc_3experiments.py` | 11 | 65.045 | The **positive–negative prototype gate (m=0)** = `03_pn_gate_m000.csv` is the keeper. |
| `cluster_cv_calibrated_scorer.py` | 12 | **65.166 ★** | 5-fold leave-writers-out CV trains a 7-feature LR; reject if p_known < 0.25 → `A_reject_pk_lt_025.csv`. **Final headline.** |

---

## Not included (intentionally)

- **Experiments not referenced by the WML paper** — ~36 documented post-hoc dead-ends
  (HDBSCAN, k-reciprocal, ProSub, SGR, VLAD-refit, multi-seed vote, 3-seed, TTA, adaptive
  gating, iterative-proto, OOD anchors, linkage probe, exemplar-SVM, softmax-max, transductive,
  fine-tune variants, power-norm, sub-center-ArcFace + morph, CV-recover, RMD scorer, ensemble
  vote) plus the `verify_embeddings.py` sanity utility. All remain in the original `fresh_start/`
  directory. The paper has no failed-methods figure (only Figs 1–4), so these are out of scope.
- **CVL / Historical-WI transfer pipeline** (paper §5–§7): the sibling `../cvl-hwi-writerid-method/` + `../splits/` folders — a different
  codebase (page-level patches, real validation labels, three OOD rules A/B/C).
- **Large binary artifacts** (embeddings `.npz`, checkpoints `.pt`, the image dataset, the
  reference submission CSVs): left in place in the main tree. README §4 lists where each lives.
  This archive is scripts + docs only (~0.4 MB).
