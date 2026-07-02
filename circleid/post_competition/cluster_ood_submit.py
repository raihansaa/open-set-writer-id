#!/usr/bin/env python3
"""
Cluster-level OOD scoring on the existing skeleton-DT trunk.

Rationale: CircleID test has ~118 images per writer. If same-writer images
cluster together in embedding space, cluster-level OOD decision lets ~118
samples vote together (reducing per-sample noise). Decision is whole-cluster
known/unknown — all samples in a cluster inherit the verdict.

Algorithm:
    1. Cluster test embeddings into K clusters (Agglomerative, cosine linkage).
    2. For each cluster: centroid = mean of L2-normed embeddings in cluster.
    3. Score centroid = -min(Mahalanobis distance to any writer prototype).
    4. Sort clusters by score; top X% by knownness become "known clusters".
    5. Known cluster -> assign argmax writer per sample (using per-sample cosine).
       Unknown cluster -> all samples get "-1".
"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = Path("./submissions_cluster_ood")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
K_VALUES = (44, 60, 80, 100, 150)
PCA_DIMS = (0, 64)             
KEEP_PCT = (25, 35, 45, 55)     #


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


def emit(name, preds_str_list, df_test, n_test, out_dir=OUT_DIR):
    n_unk = sum(1 for p in preds_str_list if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds_str_list})
    fname = f"{name}.csv"
    sub.to_csv(out_dir / fname, index=False)
    print(f"  {fname}: {n_unk}/{n_test} unknown ({100*n_unk/n_test:.1f}%)")
    return {"name": name, "n_unknown": n_unk, "frac_unknown": n_unk / n_test}


def build_proto_and_cov(train_emb, train_labels, n_writers):
    """Per-writer prototype mean + pooled Ledoit-Wolf covariance."""
    D = train_emb.shape[1]
    proto = np.zeros((n_writers, D), dtype=np.float32)
    for w in range(n_writers):
        proto[w] = train_emb[train_labels == w].mean(axis=0)
    residuals = train_emb - proto[train_labels]
    lw = LedoitWolf().fit(residuals)
    try:
        cov_inv = np.linalg.inv(lw.covariance_).astype(np.float32)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(lw.covariance_).astype(np.float32)
    return proto, cov_inv


def min_maha_to_protos(x, proto, cov_inv):
    """For each row of x, return min Mahalanobis distance to any prototype."""
    min_d = np.full(len(x), np.inf, dtype=np.float32)
    for w in range(len(proto)):
        diff = x - proto[w]
        d = np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))
        min_d = np.minimum(min_d, d)
    return min_d


def main():
    print(f"Loading + averaging seeds {SEEDS} from {EMB_DIR.resolve()} ...")
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

    pred_per_sample = test_cosine.argmax(axis=1)

    rows = []

    # Baseline reproduction for sanity
    print("\n[baseline] max-cos unk70 (must reproduce ~0.521)")
    known0 = test_cosine.max(axis=1)
    thresh = np.percentile(known0, 70)
    preds_base = [idx2writer[pred_per_sample[i]] if known0[i] >= thresh else "-1"
                  for i in range(n_test)]
    rows.append(emit("baseline_unk70", preds_base, df_test, n_test))

    # Cluster-level OOD sweeps
    for pca_dim in PCA_DIMS:
        if pca_dim == 0:
            train_e = train_emb
            test_e = test_emb
        else:
            pca = PCA(n_components=pca_dim, random_state=42).fit(train_emb)
            train_e = pca.transform(train_emb).astype(np.float32)
            test_e = pca.transform(test_emb).astype(np.float32)
            print(f"\n  PCA-{pca_dim}: var explained {pca.explained_variance_ratio_.sum():.4f}")

        proto, cov_inv = build_proto_and_cov(train_e, train_labels, n_writers)

        for K in K_VALUES:
            print(f"\n[cluster] PCA={pca_dim}d, K={K} ...")
            clu = AgglomerativeClustering(
                n_clusters=K, metric="cosine", linkage="average"
            ).fit_predict(test_e)
            centroids = np.stack([test_e[clu == c].mean(axis=0) for c in range(K)])
            # Score each cluster centroid by min-Mahalanobis to writer protos
            clu_score = -min_maha_to_protos(centroids, proto, cov_inv)
            sizes = np.bincount(clu, minlength=K)
            print(f"  cluster sizes: min={sizes.min()} median={int(np.median(sizes))} max={sizes.max()}")
            print(f"  cluster scores range: [{clu_score.min():.2f}, {clu_score.max():.2f}]")

            # Sweep over how many clusters to keep as known
            for keep_pct in KEEP_PCT:
                n_keep = max(1, int(round(K * keep_pct / 100)))
                keep_clusters = set(np.argsort(-clu_score)[:n_keep])  # top-N by knownness
                preds = []
                for i in range(n_test):
                    if int(clu[i]) in keep_clusters:
                        preds.append(idx2writer[pred_per_sample[i]])
                    else:
                        preds.append("-1")
                tag = f"clu_pca{pca_dim:03d}_K{K:03d}_keep{keep_pct:02d}"
                rows.append(emit(tag, preds, df_test, n_test))

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. Submissions in {OUT_DIR.resolve()}")
    print("\nSubmit in priority order:")
    print("  1. baseline_unk70.csv               (sanity, should reproduce ~0.521)")
    print("  2. clu_pca000_K050_keep35.csv       (full-d, K~writer count, ~65% rejection)")
    print("  3. clu_pca064_K080_keep45.csv       (PCA-64, mid K, ~55% rejection)")
    print("  4. The cluster output whose n_unknown is closest to ~4133 (matching the 0.521 rate)")


if __name__ == "__main__":
    main()
