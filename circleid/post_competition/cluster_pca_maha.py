#!/usr/bin/env python3
"""
PCA + Mahalanobis distance for cluster->writer matching.

Sweeps:
    PCA dim in {32, 64, 96, 128, 192, 256}
    shrinkage lambda in {0.0, 0.01, 0.1, 0.3, 0.5}
"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = _THIS_DIR / "submissions_pca_maha"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
K_CLUSTER = 60
N_KEEP = 7
GATING_T = 0.10

PCA_DIMS = (32, 64, 96, 128, 192, 256)
SHRINKAGES = (0.0, 0.01, 0.1, 0.3, 0.5)


def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def load_avg(seeds):
    bundles = [np.load(EMB_DIR / f"embeddings_seed_{s}.npz") for s in seeds]
    train_emb = np.mean([l2norm(b["train_emb"]) for b in bundles], axis=0)
    train_emb = l2norm(train_emb)
    train_labels = bundles[0]["train_labels"]
    test_emb = np.mean([l2norm(b["test_emb"]) for b in bundles], axis=0)
    test_emb = l2norm(test_emb)
    test_cosine = np.mean([b["test_cosine"] for b in bundles], axis=0)
    return train_emb, train_labels, test_emb, test_cosine


def emit(name, preds_str_list, df_test, n_test, **extra):
    n_unk = sum(1 for p in preds_str_list if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds_str_list})
    sub.to_csv(OUT_DIR / f"{name}.csv", index=False)
    row = {"name": name, "n_unknown": n_unk, "frac_unknown": n_unk / n_test}
    row.update(extra)
    return row


def compute_pca_maha(train_emb_pca, train_labels, centroids_pca, n_writers, shrinkage):
    """Returns (cluster_writer, cluster_neg_dist) for each of K cluster centroids.
    cluster_neg_dist[c] = -min_w maha_dist(centroid_c, proto_w)  (larger = more "known")
    """
    # Writer prototypes in PCA space
    proto_pca = np.zeros((n_writers, train_emb_pca.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        proto_pca[w] = train_emb_pca[train_labels == w].mean(axis=0)

    # Pooled covariance: each patch's residual from its writer prototype
    residuals = train_emb_pca - proto_pca[train_labels]
    Sigma = (residuals.T @ residuals) / max(1, residuals.shape[0] - n_writers)
    # Regularize with shrinkage toward scaled identity
    trace_avg = np.trace(Sigma) / Sigma.shape[0]
    Sigma_reg = (1.0 - shrinkage) * Sigma + shrinkage * trace_avg * np.eye(Sigma.shape[0])
    # Inverse (always solve via Cholesky for stability)
    try:
        L = np.linalg.cholesky(Sigma_reg)
        Sigma_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(Sigma.shape[0])))
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma_reg)

    # Mahalanobis distance: d(c, p)^2 = (c-p) Sigma_inv (c-p).T
    diff = centroids_pca[:, None, :] - proto_pca[None, :, :]  # (K, n_writers, dim)
    # Compute (diff @ Sigma_inv) @ diff.T diagonal -> use einsum
    tmp = diff @ Sigma_inv                                     # (K, n_writers, dim)
    sq = np.einsum("knd,knd->kn", tmp, diff)                   # (K, n_writers)
    sq = np.maximum(sq, 0.0)
    dist = np.sqrt(sq)
    cluster_writer = dist.argmin(axis=1)
    cluster_neg_dist = -dist.min(axis=1)
    return cluster_writer, cluster_neg_dist


def main():
    print(f"Loading + averaging seeds {SEEDS}...")
    train_emb, train_labels, test_emb, test_cosine = load_avg(SEEDS)
    print(f"  train_emb: {train_emb.shape}")
    print(f"  test_emb:  {test_emb.shape}")

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
    sorted_cos = np.sort(test_cosine, axis=1)
    margin_per_sample = sorted_cos[:, -1] - sorted_cos[:, -2]

    print(f"\nClustering K={K_CLUSTER} (cosine, average linkage)...")
    clu = AgglomerativeClustering(
        n_clusters=K_CLUSTER, metric="cosine", linkage="average"
    ).fit_predict(test_emb)
    centroids = l2norm(np.stack(
        [test_emb[clu == c].mean(axis=0) for c in range(K_CLUSTER)]))

    # Baseline (cosine matching, for comparison)
    proto_cos = np.zeros((n_writers, 512), dtype=np.float32)
    for w in range(n_writers):
        proto_cos[w] = train_emb[train_labels == w].mean(axis=0)
    proto_cos_l2 = l2norm(proto_cos)
    base_sim = centroids @ proto_cos_l2.T
    base_writer = base_sim.argmax(axis=1)
    base_known = base_sim.max(axis=1)
    base_sort = np.argsort(-base_known)
    base_keep = set(base_sort[:N_KEEP].tolist())
    print(f"  baseline (cosine) keep_set = {sorted(base_keep)}")

    def build_submission(cluster_writer, keep_set):
        preds = []
        for i in range(n_test):
            cid = int(clu[i])
            if cid not in keep_set:
                preds.append("-1")
                continue
            if margin_per_sample[i] >= GATING_T:
                preds.append(idx2writer[pred_per_sample[i]])
            else:
                preds.append(idx2writer[int(cluster_writer[cid])])
        return preds

    rows = []
    rows.append(emit("baseline_cosine_sanity",
                     build_submission(base_writer, base_keep),
                     df_test, n_test, pca_dim=0, shrinkage=0.0,
                     flip_writer=0, flip_keep=0))
    print("  baseline cosine sanity emitted (should reproduce 0.64996)")

    # ----- PCA-Maha sweep -----
    print(f"\nSweep: pca_dim in {PCA_DIMS}, shrinkage in {SHRINKAGES}")
    print(f"  {'tag':<28s} {'dim':>4s} {'shrk':>5s} {'flipW':>5s} {'flipK':>5s} {'n_unk':>6s}")
    for d in PCA_DIMS:
        if d > train_emb.shape[1]:
            continue
        # Fit PCA on per-patch train embeddings
        pca = PCA(n_components=d, random_state=0).fit(train_emb)
        train_pca = pca.transform(train_emb).astype(np.float32)
        centroids_pca = pca.transform(centroids).astype(np.float32)
        for shrk in SHRINKAGES:
            try:
                clu_writer_new, clu_known_new = compute_pca_maha(
                    train_pca, train_labels, centroids_pca, n_writers, shrk)
            except Exception as e:
                print(f"  dim={d} shrk={shrk} failed: {e}")
                continue
            sort_new = np.argsort(-clu_known_new)
            keep_new = set(sort_new[:N_KEEP].tolist())
            n_flip_writer = int((clu_writer_new != base_writer).sum())
            n_flip_keep = len(keep_new ^ base_keep) // 2
            tag = f"pcamaha_d{d:03d}_shrk{int(shrk*1000):03d}"
            preds = build_submission(clu_writer_new, keep_new)
            row = emit(tag, preds, df_test, n_test,
                       pca_dim=d, shrinkage=shrk,
                       flip_writer=n_flip_writer, flip_keep=n_flip_keep)
            rows.append(row)
            print(f"  {tag:<28s} {d:>4d} {shrk:>5.2f} {n_flip_writer:>5d} "
                  f"{n_flip_keep:>5d} {row['n_unknown']:>6d}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. {len(rows)} submissions in {OUT_DIR.resolve()}")
    print("\nDIAGNOSTIC GUIDE (same as cluster_subproto.py):")
    print("  flipW = how many of the 60 cluster -> writer assignments differ from baseline")
    print("  flipK = how many of the kept-7 clusters differ from baseline")
    print()
    print("  Recall: each flipK ~ -0.025 score loss. flipK == 0 = safe.")
    print()
    print("Submit priority:")
    print("  1. baseline_cosine_sanity.csv          -- must = 0.64996")
    print("  2. pcamaha_*.csv with flipK == 0 AND flipW in [1, 8]")
    print("     -- these refine writer matching using Maha geometry, no cluster disturbance")
    print("  3. pcamaha_d064_shrk010.csv            -- historical favorite (PCA-64 light shrinkage)")
    print("  4. AVOID submissions with flipK >= 2")


if __name__ == "__main__":
    main()
