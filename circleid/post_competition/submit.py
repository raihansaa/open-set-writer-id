#!/usr/bin/env python3

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))


import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs"))  
OUTPUT_DIR = Path("./submissions_advanced")
OUTPUT_DIR_CLUSTER = Path("./submissions_cluster")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_CLUSTER.mkdir(parents=True, exist_ok=True)

# ── Load data ──
d = np.load(EMB_DIR / "embeddings_seed_42.npz")
train_emb = d["train_emb"]       # (34650, 512)
train_labels = d["train_labels"]  # (34650,)
test_emb = d["test_emb"]         # (5905, 512)
test_cosine = d["test_cosine"]   # (5905, 44)

df_train = pd.read_csv(DATA_DIR / "train.csv")
df_add = pd.read_csv(DATA_DIR / "additional_train.csv")
df_add_known = df_add[df_add["writer_id"] != "-1"]
df_all = pd.concat([df_train, df_add_known], ignore_index=True)
writers = sorted(df_all["writer_id"].unique())
idx2writer = {i: w for i, w in enumerate(writers)}
n_writers = len(writers)

df_test = pd.read_csv(DATA_DIR / "test.csv")
writer_raw = test_cosine.argmax(axis=1)

print(f"Writers: {n_writers}, Train: {len(train_emb)}, Test: {len(test_emb)}")


def generate_sub(scores, pct, pred_indices, name, higher_is_known=True, out_dir=None):
    """Generate submission from OOD scores at given percentile."""
    if out_dir is None:
        out_dir = OUTPUT_DIR
    if higher_is_known:
        t = np.percentile(scores, pct)
        preds = [idx2writer[pred_indices[i]] if scores[i] >= t else "-1"
                 for i in range(len(df_test))]
    else:
        t = np.percentile(scores, 100 - pct)
        preds = [idx2writer[pred_indices[i]] if scores[i] <= t else "-1"
                 for i in range(len(df_test))]
    n_unk = sum(1 for p in preds if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds})
    sub.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"  {name}: {n_unk}/{len(sub)} unknown ({100*n_unk/len(sub):.1f}%)")


# ══════════════════════════════════════════════════════════════
# 1. BASELINE: Standard Mahalanobis (for comparison)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. Baseline Mahalanobis (single prototype per writer)")

class_means = np.zeros((n_writers, 512))
for c in range(n_writers):
    mask = train_labels == c
    class_means[c] = train_emb[mask].mean(axis=0)

residuals = train_emb - class_means[train_labels]
lw = LedoitWolf().fit(residuals)
cov_inv = np.linalg.inv(lw.covariance_)

def min_maha(emb, means, cov_inv):
    n = len(emb)
    min_dist = np.full(n, np.inf)
    for c in range(len(means)):
        diff = emb - means[c]
        dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        min_dist = np.minimum(min_dist, dist)
    return min_dist

baseline_maha = min_maha(test_emb, class_means, cov_inv)
print(f"  Test Maha: mean={baseline_maha.mean():.2f}, std={baseline_maha.std():.2f}")


# ══════════════════════════════════════════════════════════════
# 2. MULTI-PROTOTYPE: k-means per writer
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. Multi-prototype Mahalanobis (k-means per writer)")

for n_proto in [2, 3, 4]:
    print(f"\n  k={n_proto} prototypes per writer:")
    all_protos = []
    for c in range(n_writers):
        mask = train_labels == c
        c_emb = train_emb[mask]
        if len(c_emb) < n_proto:
            # Not enough samples, replicate mean
            all_protos.extend([c_emb.mean(axis=0)] * n_proto)
            continue
        km = KMeans(n_clusters=n_proto, n_init=5, random_state=42).fit(c_emb)
        all_protos.extend(km.cluster_centers_)

    all_protos = np.array(all_protos)  # (n_writers * n_proto, 512)

    # Mahalanobis with shared covariance (same as baseline)
    mp_maha = min_maha(test_emb, all_protos, cov_inv)
    print(f"    Test Maha: mean={mp_maha.mean():.2f}, std={mp_maha.std():.2f}")

    for pct in [50, 60, 70]:
        generate_sub(-mp_maha, pct, writer_raw,
                     f"submission_v4_mp{n_proto}_unk{pct}pct", higher_is_known=True)


# ══════════════════════════════════════════════════════════════
# 3. PCA + Mahalanobis
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. PCA + Mahalanobis (denoise embeddings)")

for n_dim in [64, 128, 256]:
    print(f"\n  PCA {n_dim}d:")
    pca = PCA(n_components=n_dim, random_state=42).fit(train_emb)
    train_pca = pca.transform(train_emb)
    test_pca = pca.transform(test_emb)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"    Variance explained: {var_explained:.4f}")

    # Per-class means in PCA space
    pca_means = np.zeros((n_writers, n_dim))
    for c in range(n_writers):
        mask = train_labels == c
        pca_means[c] = train_pca[mask].mean(axis=0)

    pca_residuals = train_pca - pca_means[train_labels]
    pca_lw = LedoitWolf().fit(pca_residuals)
    pca_cov_inv = np.linalg.inv(pca_lw.covariance_)

    pca_maha = min_maha(test_pca, pca_means, pca_cov_inv)
    print(f"    Test Maha: mean={pca_maha.mean():.2f}, std={pca_maha.std():.2f}")

    for pct in [50, 60, 70]:
        generate_sub(-pca_maha, pct, writer_raw,
                     f"submission_v4_pca{n_dim}_unk{pct}pct", higher_is_known=True)


# ══════════════════════════════════════════════════════════════
# 4. PCA + Multi-prototype (best of both)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. PCA + Multi-prototype combined")

for n_dim, n_proto in [(128, 3), (256, 2), (64, 4)]:
    print(f"\n  PCA-{n_dim}d + {n_proto}-proto:")
    pca = PCA(n_components=n_dim, random_state=42).fit(train_emb)
    train_pca = pca.transform(train_emb)
    test_pca = pca.transform(test_emb)

    all_protos = []
    for c in range(n_writers):
        mask = train_labels == c
        c_pca = train_pca[mask]
        if len(c_pca) < n_proto:
            all_protos.extend([c_pca.mean(axis=0)] * n_proto)
            continue
        km = KMeans(n_clusters=n_proto, n_init=5, random_state=42).fit(c_pca)
        all_protos.extend(km.cluster_centers_)

    all_protos = np.array(all_protos)

    pca_residuals = train_pca - np.array([
        all_protos[train_labels[i] * n_proto:train_labels[i] * n_proto + n_proto][
            np.argmin([np.linalg.norm(train_pca[i] - all_protos[train_labels[i]*n_proto+j])
                       for j in range(n_proto)])
        ] for i in range(len(train_pca))
    ])
    pca_lw = LedoitWolf().fit(pca_residuals)
    pca_cov_inv = np.linalg.inv(pca_lw.covariance_)

    combo_maha = min_maha(test_pca, all_protos, pca_cov_inv)
    print(f"    Test Maha: mean={combo_maha.mean():.2f}, std={combo_maha.std():.2f}")

    for pct in [50, 60, 70]:
        generate_sub(-combo_maha, pct, writer_raw,
                     f"submission_v4_pca{n_dim}_mp{n_proto}_unk{pct}pct", higher_is_known=True)


# ══════════════════════════════════════════════════════════════
# 5. Weighted blend: baseline Maha + PCA Maha
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. Weighted blend: baseline Maha + best PCA Maha")

# Normalize both to [0,1]
def norm01(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

base_norm = norm01(-baseline_maha)  # higher = more known

# Try PCA-128 as candidate
pca128 = PCA(n_components=128, random_state=42).fit(train_emb)
t_pca128 = pca128.transform(test_emb)
tr_pca128 = pca128.transform(train_emb)
m128 = np.zeros((n_writers, 128))
for c in range(n_writers):
    m128[c] = tr_pca128[train_labels == c].mean(axis=0)
r128 = tr_pca128 - m128[train_labels]
lw128 = LedoitWolf().fit(r128)
ci128 = np.linalg.inv(lw128.covariance_)
pca128_maha = min_maha(t_pca128, m128, ci128)
pca128_norm = norm01(-pca128_maha)

for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
    blended = alpha * base_norm + (1 - alpha) * pca128_norm
    for pct in [60]:
        generate_sub(blended, pct, writer_raw,
                     f"submission_v4_blend_a{int(alpha*10)}_unk{pct}pct", higher_is_known=True)


# ══════════════════════════════════════════════════════════════
# 6. Cluster-level OOD (whole-cluster known/unknown gating)
# ══════════════════════════════════════════════════════════════
# Cluster test pages with agglomerative cosine/average linkage, then score each
# cluster's CENTROID via PCA + Mahalanobis. Every page inherits its cluster's
# score, so 100+ pages from the same writer vote together instead of each page
# rejecting/accepting independently. CircleID test has ~115 pages per writer
# (5905 imgs / ~51 writers) — exactly the structure cluster-level should exploit.
print("\n" + "="*60)
print("6. Cluster-level OOD (PCA + agglomerative cosine-linkage)")

for n_dim in [64, 128]:
    print(f"\n  PCA-{n_dim}d:")
    pca_c = PCA(n_components=n_dim, random_state=42).fit(train_emb)
    train_pca_c = pca_c.transform(train_emb).astype(np.float32)
    test_pca_c = pca_c.transform(test_emb).astype(np.float32)

    proto_c = np.zeros((n_writers, n_dim), dtype=np.float32)
    for c in range(n_writers):
        proto_c[c] = train_pca_c[train_labels == c].mean(axis=0)
    residuals_c = train_pca_c - proto_c[train_labels]
    cov_inv_c = np.linalg.inv(LedoitWolf().fit(residuals_c).covariance_).astype(np.float32)

    for K in [44, 51, 75, 100, 150]:
        if K >= len(test_pca_c):
            continue
        clu = AgglomerativeClustering(
            n_clusters=K, metric="cosine", linkage="average"
        ).fit_predict(test_pca_c)
        centroids = np.stack([test_pca_c[clu == c].mean(axis=0) for c in range(K)])
        clu_maha = min_maha(centroids, proto_c, cov_inv_c)
        sizes = np.bincount(clu, minlength=K)
        page_score = -clu_maha[clu]  # higher = more known
        print(f"    K={K:>3}  sizes: min={sizes.min()} med={int(np.median(sizes))} "
              f"max={sizes.max()}  clu_maha: mean={clu_maha.mean():.2f} "
              f"std={clu_maha.std():.2f}")
        for pct in [60, 70, 80]:
            generate_sub(page_score, pct, writer_raw,
                         f"submission_v4_cluster_K{K}_pca{n_dim}_unk{pct}pct",
                         higher_is_known=True, out_dir=OUTPUT_DIR_CLUSTER)


print(f"\nDone!")
