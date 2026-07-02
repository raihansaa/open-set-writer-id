#!/usr/bin/env python3
"""
cluster-level single-writer assignment.

This script tests three assignment strategies at the current optimum
(K=44 Agglomerative cosine, keep top N clusters):

    A. Per-sample argmax (current 0.623 method, reference)
    B. Cluster-level: argmax cosine of cluster centroid to writer prototypes
       (all samples in cluster get that one writer)
    C. Cluster-level majority vote: each cluster gets the MOST COMMON
       per-sample argmax (mode over its samples)

Tested at n_keep in {5, 6, 7} to find new optimum after the assignment change.
"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = Path("./submissions_cluster_writer_assign")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
K = 44
N_KEEP_VALUES = (5, 6, 7, 8)


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
    print("Loading + averaging seeds 42 and 137...")
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

    # Train-writer prototypes (mean of L2-normed embeddings per writer, re-normalized)
    proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        proto[w] = train_emb[train_labels == w].mean(axis=0)
    proto_l2 = l2norm(proto)

    # Cluster test embeddings 
    print(f"\nClustering test set with K={K} Agglomerative cosine linkage...")
    clu = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    ).fit_predict(test_emb)
    centroids = l2norm(np.stack([test_emb[clu == c].mean(axis=0) for c in range(K)]))

    # Cluster-level scoring: max cosine to any prototype
    centroid_proto_sim = centroids @ proto_l2.T          # (K, n_writers)
    clu_knownness = centroid_proto_sim.max(axis=1)
    clu_writer_centroid = centroid_proto_sim.argmax(axis=1)
    sort_order = np.argsort(-clu_knownness)

    print("Cluster knownness scores (top 10):")
    for i, c in enumerate(sort_order[:10]):
        size = (clu == c).sum()
        print(f"  cluster {c:3d}: score={clu_knownness[c]:.4f} "
              f"size={size:4d} writer_by_centroid={idx2writer[clu_writer_centroid[c]]}")

    # Per-cluster majority writer via per-sample argmax
    clu_writer_majority = np.zeros(K, dtype=np.int64)
    for c in range(K):
        mask = clu == c
        if mask.sum() == 0:
            continue
        counts = Counter(pred_per_sample[mask])
        clu_writer_majority[c] = counts.most_common(1)[0][0]

    # Agreement between centroid-writer and majority-writer per cluster
    agree = (clu_writer_centroid == clu_writer_majority).mean()
    print(f"\nAgreement (centroid-writer vs majority-writer per cluster): {agree:.4f}")

    rows = []

    for n_keep in N_KEEP_VALUES:
        keep_clusters = set(sort_order[:n_keep].tolist())

        # ── A: Per-sample argmax  ──
        preds_A = [idx2writer[pred_per_sample[i]] if int(clu[i]) in keep_clusters else "-1"
                   for i in range(n_test)]
        rows.append(emit(f"A_persample_keepN{n_keep:02d}", preds_A, df_test, n_test))

        # ── B: Cluster-level argmax of CENTROID cosine to writer prototype ──
        preds_B = [idx2writer[int(clu_writer_centroid[clu[i]])]
                   if int(clu[i]) in keep_clusters else "-1"
                   for i in range(n_test)]
        rows.append(emit(f"B_centroid_keepN{n_keep:02d}", preds_B, df_test, n_test))

        # ── C: Cluster-level MAJORITY vote of per-sample argmax ──
        preds_C = [idx2writer[int(clu_writer_majority[clu[i]])]
                   if int(clu[i]) in keep_clusters else "-1"
                   for i in range(n_test)]
        rows.append(emit(f"C_majority_keepN{n_keep:02d}", preds_C, df_test, n_test))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. {len(rows)} submissions in {OUT_DIR.resolve()}")
    print("\nReference: A_persample_keepN06 should reproduce 0.623.")
    print("Primary candidates to submit:")
    print("  1. A_persample_keepN06.csv     (sanity, should be 0.623)")
    print("  2. B_centroid_keepN06.csv      (Codex's #1: cluster-level by centroid)")
    print("  3. C_majority_keepN06.csv      (alternative: majority vote)")
    print("  4. B_centroid_keepN07.csv      (test if n_keep optimum shifts)")


if __name__ == "__main__":
    main()
