#!/usr/bin/env python3
"""
Intra-cluster confidence gating.

Hybrid between per-sample argmax and cluster-level argmax. Within each known cluster:
    if per-sample top1-top2 margin >= T  -> use per-sample argmax
    else                                  -> use cluster-level argmax

Rationale: per-sample argmax knows better when a sample is CONFIDENTLY one
writer (high top1-top2 margin). Cluster-level argmax knows better when a
sample is ambiguous (low margin). The hybrid lets each sample's decision
depend on its own confidence.

Sweeps margin threshold T to find the cross-over point.
"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = Path("./submissions_gated_assign")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
K = 44
N_KEEP = 6
MARGIN_THRESHOLDS = (0.005, 0.010, 0.020, 0.030, 0.050, 0.075, 0.100, 0.150, 0.200)


def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def load_average_seeds(seeds):
    bundles = [np.load(EMB_DIR / f"embeddings_seed_{s}.npz") for s in seeds]
    train_emb = np.mean([l2norm(b["train_emb"]) for b in bundles], axis=0)
    test_emb = np.mean([l2norm(b["test_emb"]) for b in bundles], axis=0)
    train_emb = l2norm(train_emb)
    test_emb = l2norm(test_emb)
    train_labels = bundles[0]["train_labels"]
    test_cosine = np.mean([b["test_cosine"] for b in bundles], axis=0)
    return train_emb, train_labels, test_emb, test_cosine


def emit(name, preds_str_list, df_test, n_test):
    n_unk = sum(1 for p in preds_str_list if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds_str_list})
    fname = f"{name}.csv"
    sub.to_csv(OUT_DIR / fname, index=False)
    return {"name": name, "n_unknown": n_unk, "frac_unknown": n_unk / n_test}


def main():
    print("Loading + averaging seeds...")
    train_emb, train_labels, test_emb, test_cosine = load_average_seeds(SEEDS)

    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_add = pd.read_csv(DATA_DIR / "additional_train.csv")
    df_add_known = df_add[df_add["writer_id"] != "-1"]
    df_all = pd.concat([df_train, df_add_known], ignore_index=True)
    writers = sorted(df_all["writer_id"].unique())
    idx2writer = {i: w for i, w in enumerate(writers)}
    n_writers = len(writers)
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    n_test = len(df_test)

    print(f"  Writers: {n_writers} | test: {n_test}")

    # Per-sample writer prediction from test_cosine
    pred_per_sample = test_cosine.argmax(axis=1)

    # Per-sample top1-top2 MARGIN (how confident this sample is in its writer)
    sorted_cos = np.sort(test_cosine, axis=1)
    margin_per_sample = sorted_cos[:, -1] - sorted_cos[:, -2]
    print(f"  margin range: [{margin_per_sample.min():.4f}, {margin_per_sample.max():.4f}]")
    print(f"  margin percentiles: p10={np.percentile(margin_per_sample, 10):.4f} "
          f"p50={np.percentile(margin_per_sample, 50):.4f} "
          f"p90={np.percentile(margin_per_sample, 90):.4f}")

    # Train-writer prototypes
    proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        proto[w] = train_emb[train_labels == w].mean(axis=0)
    proto_l2 = l2norm(proto)

    # K=44 cluster test (winner config)
    print(f"\nClustering K={K} Agglomerative cosine...")
    clu = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    ).fit_predict(test_emb)
    centroids = l2norm(np.stack([test_emb[clu == c].mean(axis=0) for c in range(K)]))
    centroid_proto_sim = centroids @ proto_l2.T
    clu_knownness = centroid_proto_sim.max(axis=1)
    clu_writer = centroid_proto_sim.argmax(axis=1)
    sort_order = np.argsort(-clu_knownness)
    keep_set = set(sort_order[:N_KEEP].tolist())

    # Diagnostic: per known cluster, how do per-sample argmax distribute?
    print(f"\nKnown clusters (top {N_KEEP}):")
    for c in sort_order[:N_KEEP]:
        mask = clu == c
        size = mask.sum()
        cluster_w = int(clu_writer[c])
        # Within cluster: how many samples have per-sample argmax == cluster_writer
        agree = (pred_per_sample[mask] == cluster_w).mean()
        # Confidence distribution
        median_margin = np.median(margin_per_sample[mask])
        print(f"  cluster {c:3d}: size={size:4d} cluster_w={idx2writer[cluster_w]:<5} "
              f"per-sample agree={agree:.3f} median_margin={median_margin:.4f}")

    rows = []

    # ── Baseline 1: pure cluster-level ──
    preds_clust = [idx2writer[int(clu_writer[clu[i]])] if int(clu[i]) in keep_set else "-1"
                   for i in range(n_test)]
    rows.append(emit("baseline_cluster_keepN06", preds_clust, df_test, n_test))

    # ── Baseline 2: pure per-sample  ──
    preds_persample = [idx2writer[pred_per_sample[i]] if int(clu[i]) in keep_set else "-1"
                       for i in range(n_test)]
    rows.append(emit("baseline_persample_keepN06", preds_persample, df_test, n_test))

    # ── Gated variants: per-sample if margin >= T, else cluster-level ──
    for T in MARGIN_THRESHOLDS:
        preds = []
        switches_to_persample = 0
        for i in range(n_test):
            if int(clu[i]) not in keep_set:
                preds.append("-1")
                continue
            if margin_per_sample[i] >= T:
                preds.append(idx2writer[pred_per_sample[i]])
                switches_to_persample += 1
            else:
                preds.append(idx2writer[int(clu_writer[clu[i]])])
        tag = f"gated_T{int(T*1000):04d}"
        r = emit(tag, preds, df_test, n_test)
        r["T"] = T
        r["n_persample_switches"] = switches_to_persample
        rows.append(r)
        print(f"  {tag}: T={T:.3f}  per-sample switches={switches_to_persample}/{n_test - r['n_unknown']} known samples")

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. {len(rows)} submissions in {OUT_DIR.resolve()}")
    print("\nSubmit in priority order:")
    print("  1. baseline_cluster_keepN06.csv  (sanity = 0.632)")
    print("  2. gated_T0050.csv               (T=0.050, moderate gating)")
    print("  3. gated_T0030.csv               (T=0.030, more aggressive per-sample override)")
    print("  4. gated_T0100.csv               (T=0.100, conservative per-sample override)")


if __name__ == "__main__":
    main()
