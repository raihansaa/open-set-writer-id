# Open-Set Writer Identification — From Hand-Drawn Circles to Handwritten Pages

Code, open-set data splits, and reproduction scripts for the WML 2026 paper:

> **Cross-Dataset Transfer in Open-Set Writer Identification: From Hand-Drawn Circles to Handwritten Pages**
> Md Raihan, Thomas Gorges, Lukas Hüttner, Vincent Christlein
> *Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany*
> Accepted at the [6th ICDAR Workshop on Machine Learning (WML 2026)](https://icdarwml2026.midasoc.org/) — Vienna, Austria, 3 September 2026.

## Overview

Writer identification is usually studied as a **closed-set** retrieval problem. This work
tackles the harder, more realistic **open-set** setting — query pages may come from writers
never seen at enrollment — under a **few-shot** budget of only **3 enrollment pages per
writer**, and asks how well a *single* recipe **transfers across datasets**: from the
hand-drawn circles of the ICDAR **CircleID** competition to full handwritten pages in
**CVL** (modern) and **Historical-WI** (historical).

The method is a frozen **DINOv2 ViT-B/14** backbone with **LoRA**, **NetVLAD** aggregation,
and a **multi-prototype ArcFace** head with hard-negative mining, followed by one of three
**open-set rejection rules** — (A) per-writer multi-prototype Mahalanobis, (B) cluster
K-sweep, (C) joint train+test clustering. Every dataset is cast into the same three-group
protocol of **Known / Pseudo-unknown / Unknown** writers.

## Headline results

**Handwritten pages** — open-set Top-1, three-seed ensemble (paper Table 3):

| Dataset       | Best rejection rule         |    OS-Top1 | Known-Top1 |  AUROC |
|---------------|-----------------------------|-----------:|-----------:|-------:|
| CVL           | Joint train+test clustering | **74.67%** |     63.20% |  85.3% |
| Historical-WI | Per-writer MP-Mahalanobis   | **54.13%** |     13.06% |  57.3% |

**Hand-drawn circles** — CircleID private leaderboard (paper Table 2): **65.17%**, up from
the 47.02% competition-day submission.

## Repository structure

| Folder | Contents |
|--------|----------|
| [`cvl-hwi-writerid-method/`](cvl-hwi-writerid-method/) | The open-set writer-ID method for **CVL** and **Historical-WI**: DINOv2 + LoRA + NetVLAD + multi-prototype ArcFace, plus the three rejection rules. Training, embedding extraction, post-hoc refinement, and evaluation. See its [README](cvl-hwi-writerid-method/README.md) and [REPRODUCE](cvl-hwi-writerid-method/REPRODUCE.md). |
| [`splits/`](splits/) | The **open-set split protocol**: deterministic builders (`build_cvl_splits.py`, `build_hwi_splits.py`), a convention-agnostic verifier (`verify_splits.py`), and the ready-to-use split manifests (`cvl_splits/`, `hwi_splits/`) that define the Known / Pseudo-unknown / Unknown pools. |
| [`circleid/`](circleid/) | Reproduction archive for the **CircleID** (hand-drawn circles) half of the paper — the code behind each row of Table 2, from the competition-day submission to the final configuration. See its [README](circleid/README.md). |

## Quickstart

The datasets are **not redistributed** here (licensing) — download them from their official
sources (links in the method [README](cvl-hwi-writerid-method/README.md#data)) and use the
split manifests provided in [`splits/`](splits/). Then:

```bash
cd cvl-hwi-writerid-method
pip install -r requirements.txt
# train + extract → refine → reject → evaluate; full commands in REPRODUCE.md
```

- **Verify the open-set splits** at any time:
  `python splits/verify_splits.py splits/cvl_splits splits/hwi_splits`
- **Full step-by-step reproduction:** [`cvl-hwi-writerid-method/REPRODUCE.md`](cvl-hwi-writerid-method/REPRODUCE.md)

## Citation

If you use this code or the open-set CVL / Historical-WI protocols, please cite:

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

A machine-readable [`CITATION.cff`](cvl-hwi-writerid-method/CITATION.cff) is also provided.

## License

Released under the [MIT License](cvl-hwi-writerid-method/LICENSE).
