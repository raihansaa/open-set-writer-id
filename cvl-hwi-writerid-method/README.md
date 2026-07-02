# Cross-Dataset Transfer in Open-Set Writer Identification

Code for the **CVL** and **Historical-WI** experiments in:

> **Cross-Dataset Transfer in Open-Set Writer Identification: From Hand-Drawn Circles to Handwritten Pages**
> Accepted at the 6th ICDAR Workshop on Machine Learning ([WML 2026](https://icdarwml2026.midasoc.org/)), Vienna, Austria, 3 Sept 2026 — [paper link: TODO]

This repository releases the **transferable-subset pipeline**: a frozen **DINOv2 ViT-B/14** backbone
with **LoRA**, **NetVLAD** aggregation, a **multi-prototype ArcFace** head with **hard-negative
mining**, and three open-set rejection rules — (A) per-writer multi-prototype Mahalanobis,
(B) cluster K-sweep, and (C) joint train+test clustering — evaluated on CVL and Historical-WI under a
CircleID-style open-set protocol (Known / Pseudo-unknown / Unknown writer pools).

> The ICDAR 2026 CircleID competition code is released **separately** and is not part of this repository.

## Headline results (open-set Top-1, three-seed ensemble)

| Dataset       | Best OOD rule                 | OS-Top1    | Known-Top1 | AUROC |
|---------------|-------------------------------|------------|------------|-------|
| CVL           | Joint train+test clustering   | **74.67%** | 63.20%     | 85.3% |
| Historical-WI | Per-writer MP-Mahalanobis     | **54.13%** | 13.06%     | 57.3% |

## Repository structure

```
train_writerid.py     # backbone training + page-embedding extraction (DINOv2+LoRA+NetVLAD+MP-ArcFace)
patches.py            # ink-anchored 224x224 patch sampler
posthoc_refine.py     # post-hoc page-descriptor refinement (CVL: gmp+rerank, HWI: mean)
submit_writerid.py    # open-set rejection rules A (Mahalanobis) + B (cluster K-sweep)
cluster_joint.py      # open-set rule C (joint train+test clustering; CVL-headline rule)
eval_metrics.py       # OS-Top1 / Known-Top1 / unknown-rejection AUROC
../splits/cvl_splits/ # CVL open-set split manifests (writers + train/val/test CSVs)  [sibling folder]
../splits/hwi_splits/ # Historical-WI open-set split manifests                        [sibling folder]
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate      # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Python ≥ 3.10 and a CUDA-capable GPU are recommended. The DINOv2 backbone is fetched automatically
via `torch.hub` on first run (internet required once).

## Data

The raw datasets are **not redistributed** here (licensing); only our split manifests are provided.
Download each dataset from its official source and place the images so that `data_dir / image_path`
(the `image_path` column in the CSVs) resolves:

- **CVL** [Kleber et al., ICDAR 2013] — place page images under `cvl/cvl/` (e.g. `cvl/cvl/0052-1-cropped.tif`).
  Download: [CVL Database — TU Wien CVL](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/) · mirror: [Zenodo 1492267](https://zenodo.org/records/1492267). Use the cropped (handwriting-only) images.
- **Historical-WI** [Fiel et al., ICDAR 2017] — binarized ScriptNet version; place images under
  `hwi/hwi_test/` and `hwi/hwi_train/` (e.g. `hwi/hwi_test/1-IMG_MAX_9964.jpg`).
  Download: [Historical-WI — Zenodo 1324999](https://zenodo.org/records/1324999) (use the **binarized** version) · [ScriptNet competition](https://scriptnet.iit.demokritos.gr/competitions/6/).

The split CSVs (`image_id, image_path, writer_id`; `writer_id = -1` marks unknown) and the
`writers_known/pseudo_unknown/unknown.txt` files live in `../splits/cvl_splits/` and
`../splits/hwi_splits/`, define the three-group open-set protocol, and are ready to use as-is.

## Reproducing the results

Run from inside `cvl-hwi-writerid-method/`. Example with seed 42; repeat for your three seeds and ensemble.

**1 — Train + extract embeddings**

```bash
# CVL (40 epochs)
python train_writerid.py --data-dir cvl --splits-dir ../splits/cvl_splits --out-dir runs/cvl_seed42 \
  --aggregator netvlad --vlad-clusters 64 --lora-blocks 12 \
  --multi-proto-k 2 --hard-neg-weight 0.2 --epochs 40 --no-aug --seed 42

# Historical-WI (50 epochs)
python train_writerid.py --data-dir hwi --splits-dir ../splits/hwi_splits --out-dir runs/hwi_seed42 \
  --aggregator netvlad --vlad-clusters 64 --lora-blocks 12 \
  --multi-proto-k 2 --hard-neg-weight 0.2 --epochs 50 --no-aug --seed 42
```

This writes `runs/<...>/embeddings_seed42.npz` (train/val/test page embeddings + cosines).
*Vanilla baseline* (Table 3): use `--lora-blocks 6 --multi-proto-k 1`.

**2 — Open-set rejection (rules A/B/C)**

```bash
python submit_writerid.py --emb runs/cvl_seed42/embeddings_seed42.npz --out-dir runs/cvl_seed42/submissions
```

Runs the rejection method zoo — per-writer MP-Mahalanobis (rule A), cluster K-sweep (rule B), and
joint train+test clustering (rule C) — selecting each threshold on the validation Pseudo-unknown pool.

**3 — Evaluate**

```bash
python eval_metrics.py --emb runs/cvl_seed42/embeddings_seed42.npz --sub-dir runs/cvl_seed42/submissions
```

Reports OS-Top1, Known-Top1, and unknown-rejection AUROC per method.

## Canonical configuration (note on extra flags)

The commands above reproduce the paper. `train_writerid.py` also exposes flags from our follow-up
(journal) work that are **disabled by default and were not used in this paper** — leave them at their
defaults:

- `--lambda-supcon` — SupCon auxiliary loss (default 0, off)
- `--aggregator attention` — gated attention-MIL pooling (paper uses `netvlad`)
- `--pk-sampler` — P×K writer-stratified batches (paper uses standard shuffling)
- `--init-from` — SSL warm-start checkpoint (paper uses off-the-shelf DINOv2)

## Hardware

All experiments run on a single consumer **NVIDIA RTX 5060 (8 GB)** using mixed-precision (fp16) and
gradient checkpointing at 224×224, batch size 32.

## Citation

```bibtex
@inproceedings{raihan2026crossdataset,
  title     = {Cross-Dataset Transfer in Open-Set Writer Identification:
               From Hand-Drawn Circles to Handwritten Pages},
  author    = {Raihan, Md and Gorges, Thomas and H\"uttner, Lukas and Christlein, Vincent},
  booktitle = {6th ICDAR Workshop on Machine Learning (WML)},
  address   = {Vienna, Austria},
  year      = {2026}
}
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
