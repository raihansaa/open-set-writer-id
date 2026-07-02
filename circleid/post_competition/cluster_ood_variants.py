#!/usr/bin/env python3
"""
Variant explorations on the cluster-level OOD breakthrough.

Tests four orthogonal axes around the current optimum:

    1. Smaller cluster counts (6, 7) at K=42, 44
       -- Does peak go below 8 clusters?
    2. Cosine cluster-centroid scoring (instead of Mahalanobis)
       -- Simpler scorer, no covariance estimation
    3. PCA-128 + Mahalanobis (more conservative than PCA-64 which lost)
    4. KMeans clustering (instead of Agglomerative)
       -- Different cluster shapes
"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = Path("./submissions_cluster_variants")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]


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


def build_proto_and_cov(train_emb, train_labels, n_writers):
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
    min_d = np.full(len(x), np.inf, dtype=np.float32)
    for w in range(len(proto)):
        diff = x - proto[w]
        d = np.sqrt(np.maximum(np.sum(diff @ cov_inv * diff, axis=1), 0))
        min_d = np.minimum(min_d, d)
    return min_d


def max_cosine_to_protos(x, proto_l2):
    """Max cosine to any prototype (proto_l2 must be L2-normed)."""
    sim = x @ proto_l2.T            # (n_centroids, n_writers)
    return sim.max(axis=1)


def emit(name, preds_str_list, df_test, n_test):
    n_unk = sum(1 for p in preds_str_list if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds_str_list})
    fname = f"{name}.csv"
    sub.to_csv(OUT_DIR / fname, index=False)
    return {"name": name, "n_unknown": n_unk, "frac_unknown": n_unk / n_test}


def cluster_partition(test_emb, K, algo="agg"):
    if algo == "agg":
        return AgglomerativeClustering(
            n_clusters=K, metric="cosine", linkage="average"
        ).fit_predict(test_emb)
    elif algo == "kmeans":
        return KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(test_emb)
    raise ValueError(algo)


def predict_from_clusters(clu, K, clu_score, n_keep, pred_per_sample, idx2writer, n_test):
    keep_clusters = set(np.argsort(-clu_score)[:n_keep])
    return [idx2writer[pred_per_sample[i]] if int(clu[i]) in keep_clusters else "-1"
            for i in range(n_test)]


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
    pred_per_sample = test_cosine.argmax(axis=1)

    print(f"  Writers: {n_writers} | test: {n_test}")

    # Build proto + cov in full-d
    proto_full, cov_inv_full = build_proto_and_cov(train_emb, train_labels, n_writers)
    proto_full_l2 = l2norm(proto_full)

    # PCA-128 versions
    pca128 = PCA(n_components=128, random_state=42).fit(train_emb)
    train_pca = pca128.transform(train_emb).astype(np.float32)
    test_pca = pca128.transform(test_emb).astype(np.float32)
    proto_p128, cov_p128 = build_proto_and_cov(train_pca, train_labels, n_writers)

    rows = []

    # ── A. Smaller cluster counts at K=42, 44 (Agglomerative, full-d Maha) ─
    print("\n[A] Smaller cluster counts (6, 7) via Agglomerative + full-d Maha")
    for K in (42, 44):
        clu = cluster_partition(test_emb, K, "agg")
        centroids = np.stack([test_emb[clu == c].mean(axis=0) for c in range(K)])
        clu_score = -min_maha_to_protos(centroids, proto_full, cov_inv_full)
        for n_keep in (5, 6, 7):
            preds = predict_from_clusters(clu, K, clu_score, n_keep, pred_per_sample, idx2writer, n_test)
            tag = f"agg_K{K:03d}_keepN{n_keep:02d}_mahaFull"
            r = emit(tag, preds, df_test, n_test)
            r["K"] = K; r["n_keep"] = n_keep; r["scorer"] = "mahaFull"; r["algo"] = "agg"
            rows.append(r)
            print(f"  {tag}: {r['n_unknown']}/{n_test} unknown ({100*r['frac_unknown']:.1f}%)")

    # ── B. Cosine scorer at K=44, keep 8 ──
    print("\n[B] Cosine scorer (vs Mahalanobis) at K=44 keep_N in {7, 8, 9}")
    K = 44
    clu = cluster_partition(test_emb, K, "agg")
    centroids = np.stack([test_emb[clu == c].mean(axis=0) for c in range(K)])
    centroids_l2 = l2norm(centroids)
    clu_score_cos = max_cosine_to_protos(centroids_l2, proto_full_l2)
    for n_keep in (7, 8, 9):
        preds = predict_from_clusters(clu, K, clu_score_cos, n_keep, pred_per_sample, idx2writer, n_test)
        tag = f"agg_K{K:03d}_keepN{n_keep:02d}_cosScore"
        r = emit(tag, preds, df_test, n_test)
        r["K"] = K; r["n_keep"] = n_keep; r["scorer"] = "cosine"; r["algo"] = "agg"
        rows.append(r)
        print(f"  {tag}: {r['n_unknown']}/{n_test} unknown ({100*r['frac_unknown']:.1f}%)")

    # ── C. PCA-128 + Mahalanobis at K=44 ──
    print("\n[C] PCA-128 + Maha cluster scoring at K=44 keep_N in {7, 8, 9}")
    centroids_pca = np.stack([test_pca[clu == c].mean(axis=0) for c in range(K)])
    clu_score_p128 = -min_maha_to_protos(centroids_pca, proto_p128, cov_p128)
    for n_keep in (7, 8, 9):
        preds = predict_from_clusters(clu, K, clu_score_p128, n_keep, pred_per_sample, idx2writer, n_test)
        tag = f"agg_K{K:03d}_keepN{n_keep:02d}_mahaPca128"
        r = emit(tag, preds, df_test, n_test)
        r["K"] = K; r["n_keep"] = n_keep; r["scorer"] = "mahaPca128"; r["algo"] = "agg"
        rows.append(r)
        print(f"  {tag}: {r['n_unknown']}/{n_test} unknown ({100*r['frac_unknown']:.1f}%)")

    # ── D. KMeans at K=44, full-d Maha ──
    print("\n[D] KMeans clustering (vs Agglomerative) at K=44 keep_N in {7, 8, 9}")
    clu_km = cluster_partition(test_emb, K, "kmeans")
    centroids_km = np.stack([test_emb[clu_km == c].mean(axis=0) for c in range(K)])
    clu_score_km = -min_maha_to_protos(centroids_km, proto_full, cov_inv_full)
    for n_keep in (7, 8, 9):
        preds = predict_from_clusters(clu_km, K, clu_score_km, n_keep, pred_per_sample, idx2writer, n_test)
        tag = f"km_K{K:03d}_keepN{n_keep:02d}_mahaFull"
        r = emit(tag, preds, df_test, n_test)
        r["K"] = K; r["n_keep"] = n_keep; r["scorer"] = "mahaFull"; r["algo"] = "kmeans"
        rows.append(r)
        print(f"  {tag}: {r['n_unknown']}/{n_test} unknown ({100*r['frac_unknown']:.1f}%)")

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. {len(rows)} submissions in {OUT_DIR.resolve()}")
    print("\nPriority candidates to submit (closest to current best K044_keepN8 = 0.604):")
    print("  1. agg_K044_keepN07_mahaFull.csv  -- does 7 clusters beat 8?")
    print("  2. agg_K044_keepN08_cosScore.csv  -- does cosine match/beat Maha?")
    print("  3. agg_K044_keepN08_mahaPca128.csv -- conservative PCA at the best keep")
    print("  4. km_K044_keepN08_mahaFull.csv  -- does KMeans match Agglomerative?")


if __name__ == "__main__":
    main()
