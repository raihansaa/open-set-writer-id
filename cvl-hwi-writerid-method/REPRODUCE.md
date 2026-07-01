# Reproducing the paper results

Step-by-step reproduction of the CVL and Historical-WI open-set results
(Table 3 / Figure 4). Run all commands from inside `cvl-hwi-writerid-method/` (the code does a local `import patches`, so it must run from there).

## 0. Environment + data

```bash
pip install -r requirements.txt
```

Place the downloaded images so the `image_path` column in the split CSVs resolves
(see [README](README.md#data)):

```
cvl/cvl/<page>.tif            # CVL cropped (handwriting-only) pages
hwi/hwi_test/<page>.jpg       # Historical-WI binarized, ScriptNet test partition
hwi/hwi_train/<page>.jpg      # Historical-WI binarized, ICDAR-2017 train partition
```

The split manifests live in `../splits/cvl_splits/` and `../splits/hwi_splits/`
(train/val/test `.csv` + `writers_*.txt`) and define the Known / Pseudo-unknown / Unknown protocol.

## 1. Train + extract embeddings (three seeds)

The headline config is the transferable subset: DINOv2 ViT-B/14 (frozen) + LoRA-12
+ NetVLAD-64 + multi-prototype ArcFace (K=2) + hard-negative mining, augmentation off.

```bash
for SEED in 42 7 137; do
  # CVL — 40 epochs
  python train_writerid.py --data-dir cvl --splits-dir ../splits/cvl_splits \
    --out-dir runs/cvl_seed${SEED} \
    --aggregator netvlad --vlad-clusters 64 --lora-blocks 12 \
    --multi-proto-k 2 --hard-neg-weight 0.2 --epochs 40 --no-aug --seed ${SEED}

  # Historical-WI — 50 epochs
  python train_writerid.py --data-dir hwi --splits-dir ../splits/hwi_splits \
    --out-dir runs/hwi_seed${SEED} \
    --aggregator netvlad --vlad-clusters 64 --lora-blocks 12 \
    --multi-proto-k 2 --hard-neg-weight 0.2 --epochs 50 --no-aug --seed ${SEED}
done
```

Each run writes `runs/<...>/embeddings_seed<SEED>.npz`. On a single RTX 5060 (8 GB)
expect roughly 45–100 min per run.

*Vanilla baseline row of Table 3:* use `--lora-blocks 6 --multi-proto-k 1`.

## 2. Post-hoc refinement, then open-set rejection (rules A / B / C)

First refine the page descriptors — **required** (this is what lifts CVL AUROC ~0.55→0.85):
```bash
python posthoc_refine.py --emb runs/cvl_seed42/embeddings_seed42.npz --aggregate gmp --rerank   # CVL
python posthoc_refine.py --emb runs/hwi_seed42/embeddings_seed42.npz --aggregate mean            # HWI
```
This writes `embeddings_seed42_refined.npz`. Run the OOD rules on the **refined** file:
```bash
python submit_writerid.py --emb runs/cvl_seed42/embeddings_seed42_refined.npz \
  --out-dir runs/cvl_seed42/submissions
```

Runs the rejection method zoo and tunes every threshold on the validation
Pseudo-unknown pool only:
- **Rule A** — per-writer multi-prototype Mahalanobis (`baseline_maha`, `mp*`, `pca*`)
- **Rule B** — cluster K-sweep with keep-N (`ksweep_*`)
- **Rule C** — joint train+test agglomerative clustering (the CVL-headline rule), run separately:
  `python cluster_joint.py runs/cvl_seed42/embeddings_seed42_refined.npz`

## 3. Evaluate

```bash
python eval_metrics.py --emb runs/cvl_seed42/embeddings_seed42.npz \
  --sub-dir runs/cvl_seed42/submissions
```

Reports OS-Top1, Known-Top1, and unknown-rejection AUROC per method. Average the
three seeds for the per-method mean ± std, and ensemble the three seeds for the
headline row.

## Expected results (three-seed ensemble)

| Dataset       | Best OOD rule               | OS-Top1 | Known-Top1 | AUROC |
|---------------|-----------------------------|---------|------------|-------|
| CVL           | Joint train+test clustering | 74.67%  | 63.20%     | 85.3% |
| Historical-WI | Per-writer MP-Mahalanobis   | 54.13%  | 13.06%     | 57.3% |

Small deviations (≈ ±1 pp on CVL, ±1.5 pp on HWI) across machines/library versions
are expected; the three-seed mean should match within the reported std.

## Notes

- No PCA-whitening and no SSL warm-start are used (off-the-shelf DINOv2).
- Augmentation is disabled in the final config (`--no-aug`); enabling it lowers CVL
  OS-Top1 by ≈ 2 pp.
- The flags `--lambda-supcon`, `--aggregator attention`, `--pk-sampler`, `--init-from`
  are follow-up work, disabled by default — do not set them to reproduce the paper.
