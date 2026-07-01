"""
Generalized open-set writer-ID inference + OOD ablations.

Usage:
    # Single seed (one CSV per ablation)
    python submit_writerid.py --emb runs/cvl_seed42/embeddings_seed42.npz \
                              --out-dir runs/cvl_seed42/submissions

    # Multi-seed ensemble (one ensemble CSV per ablation)
    python submit_writerid.py --emb runs/cvl_seed42/embeddings_seed42.npz \
                                    runs/cvl_seed137/embeddings_seed137.npz \
                                    runs/cvl_seed7/embeddings_seed7.npz \
                              --out-dir runs/cvl_ensemble/submissions
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA


# ──────────────────────────────────────────────────────────────────────
# Core OOD scoring helpers
# ──────────────────────────────────────────────────────────────────────
def min_mahalanobis(emb, means, cov_inv):
    
    min_dist = np.full(len(emb), np.inf, dtype=np.float32)
    for c in range(len(means)):
        diff = emb - means[c]
        dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        min_dist = np.minimum(min_dist, dist)
    return min_dist


def fit_class_means(emb, labels, n_classes):
    means = np.zeros((n_classes, emb.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        means[c] = emb[mask].mean(axis=0)
    return means


def fit_class_kmeans(emb, labels, n_classes, n_proto):

    protos = []
    for c in range(n_classes):
        mask = labels == c
        c_emb = emb[mask]
        if len(c_emb) < n_proto:
            mu = c_emb.mean(axis=0) if len(c_emb) > 0 else np.zeros(emb.shape[1])
            protos.extend([mu] * n_proto)
        else:
            km = KMeans(n_clusters=n_proto, n_init=5, random_state=42).fit(c_emb)
            protos.extend(km.cluster_centers_)
    return np.array(protos, dtype=np.float32)


def residual_covariance(emb, labels, means):
    """Ledoit-Wolf covariance of residuals (x - mean_class[x])."""
    residuals = emb - means[labels]
    lw = LedoitWolf().fit(residuals)
    return np.linalg.inv(lw.covariance_)


# ──────────────────────────────────────────────────────────────────────
# Threshold tuning on val
# ──────────────────────────────────────────────────────────────────────
def tune_threshold(val_scores, val_labels_int, val_is_unk, writer_pred_idx,
                    higher_is_known=True, sweep=range(0, 96)):
    
    best_pct, best_acc, best_mask = 0, -1.0, None
    for pct in sweep:
        if higher_is_known:
            t = np.percentile(val_scores, pct)
            is_unk = val_scores < t
        else:
            t = np.percentile(val_scores, 100 - pct)
            is_unk = val_scores > t

        # Accuracy: correct if (is_unk AND val_is_unk) OR (not is_unk AND writer_pred == val_label)
        n_correct = 0
        for i in range(len(val_scores)):
            if is_unk[i]:
                if val_is_unk[i]:
                    n_correct += 1
            else:
                if (not val_is_unk[i]) and writer_pred_idx[i] == val_labels_int[i]:
                    n_correct += 1
        acc = n_correct / len(val_scores)
        if acc > best_acc:
            best_pct, best_acc, best_mask = pct, acc, is_unk
    return best_pct, best_acc, best_mask


def apply_threshold(scores, percentile, higher_is_known=True):
    if higher_is_known:
        t = np.percentile(scores, percentile)
        return scores < t  # is_unknown mask
    else:
        t = np.percentile(scores, 100 - percentile)
        return scores > t


def write_submission(test_image_ids, writer_pred_idx, is_unk, writers, out_path):
    preds = [writers[writer_pred_idx[i]] if not is_unk[i] else "-1" for i in range(len(is_unk))]
    df = pd.DataFrame({"image_id": test_image_ids, "writer_id": preds})
    df.to_csv(out_path, index=False)


def write_submission_from_predidx(image_ids, pred_writer_idx, writers, out_path):
    
    preds = [writers[int(p)] if int(p) >= 0 else "-1" for p in pred_writer_idx]
    pd.DataFrame({"image_id": image_ids, "writer_id": preds}).to_csv(out_path, index=False)


def _l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def cluster_and_score(emb, train_proto_l2, K):
    
    from sklearn.cluster import AgglomerativeClustering as _Agg
    P = emb.shape[0]
    K = max(2, min(K, P - 1))
    clu = _Agg(n_clusters=K, metric="cosine", linkage="average").fit_predict(emb)
    centroids = _l2norm(np.stack([emb[clu == c].mean(axis=0) for c in range(K)]))
    sim = centroids @ train_proto_l2.T                # (K, n_writers)
    return {
        "clu": clu,
        "clu_writer": sim.argmax(axis=1),
        "clu_knownness": sim.max(axis=1),
        "K_eff": K,
    }


def gated_predict_from_clusters(diag, cosine, n_keep, T):
    
    clu = diag["clu"]
    clu_writer = diag["clu_writer"]
    K_eff = diag["K_eff"]
    n_keep = max(1, min(n_keep, K_eff))
    keep_set = set(np.argsort(-diag["clu_knownness"])[:n_keep].tolist())

    pred_per = cosine.argmax(axis=1)
    sorted_cos = np.sort(cosine, axis=1)
    margin = sorted_cos[:, -1] - sorted_cos[:, -2]

    P = cosine.shape[0]
    pred = np.full(P, -1, dtype=np.int64)
    for i in range(P):
        cid = int(clu[i])
        if cid not in keep_set:
            continue
        pred[i] = int(pred_per[i]) if margin[i] >= T else int(clu_writer[cid])
    return pred, keep_set


def cluster_gated_predict(emb, cosine, train_proto_l2, n_writers, K, n_keep, T):
    
    diag = cluster_and_score(emb, train_proto_l2, K)
    pred, keep_set = gated_predict_from_clusters(diag, cosine, n_keep, T)
    diag["keep_set"] = keep_set
    diag["n_keep_eff"] = len(keep_set)
    return pred, diag


def open_set_top1_acc(pred_writer_idx, labels_int, is_unk):
    
    n_correct = 0
    for i in range(len(pred_writer_idx)):
        p = int(pred_writer_idx[i])
        if p < 0:
            if is_unk[i]:
                n_correct += 1
        else:
            if (not is_unk[i]) and p == int(labels_int[i]):
                n_correct += 1
    return n_correct / max(1, len(pred_writer_idx))


# ──────────────────────────────────────────────────────────────────────
# Five ablations
# ──────────────────────────────────────────────────────────────────────
def run_ablations(npz, out_dir, label_map_train_to_writer, *,
                  write_subs: bool = True,
                  score_collector: dict | None = None,
                  print_header: bool = True):
    
    train_emb = npz["train_emb"].astype(np.float32)
    train_labels = npz["train_labels"]
    val_emb = npz["val_emb"].astype(np.float32)
    val_writer_id = npz["val_writer_id"].astype(str)
    val_cosine = npz["val_cosine"].astype(np.float32)
    val_image_id = npz["val_image_id"]
    test_emb = npz["test_emb"].astype(np.float32)
    test_cosine = npz["test_cosine"].astype(np.float32)
    test_image_id = npz["test_image_id"]
    test_writer_id = npz["test_writer_id"].astype(str)
    writers = list(npz["writers"].astype(str))
    n_writers = len(writers)
    D = train_emb.shape[1]

    # Argmax predicted writer index from cosine
    val_pred_idx = val_cosine.argmax(axis=1)
    test_pred_idx = test_cosine.argmax(axis=1)
    val_is_unk = (val_writer_id == "-1")
    test_is_unk = (test_writer_id == "-1")

    # Encode val labels as ints (use a fallback -2 for -1 so it never matches)
    writer2idx = {w: i for i, w in enumerate(writers)}
    val_labels_int = np.array([writer2idx.get(w, -2) for w in val_writer_id], dtype=np.int64)
    test_labels_int = np.array([writer2idx.get(w, -2) for w in test_writer_id], dtype=np.int64)

    out_dir = Path(out_dir)
    if write_subs:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    def report_test(name, test_is_unk_pred):
        
        # Count: how many test samples were correctly rejected vs identified
        n_correct = 0
        for i in range(len(test_is_unk_pred)):
            if test_is_unk_pred[i]:
                if test_is_unk[i]:
                    n_correct += 1
            else:
                if (not test_is_unk[i]) and test_pred_idx[i] == test_labels_int[i]:
                    n_correct += 1
        return n_correct / len(test_is_unk_pred)

    def _finalize(name, val_score, test_score, csv_name):
        
        if score_collector is not None:
            score_collector[name] = (
                val_score.astype(np.float32),
                test_score.astype(np.float32),
            )
        pct, val_acc, _ = tune_threshold(
            val_score, val_labels_int, val_is_unk, val_pred_idx
        )
        is_unk_test = apply_threshold(test_score, pct)
        test_acc = report_test(name, is_unk_test)
        if write_subs:
            write_submission(
                test_image_id, test_pred_idx, is_unk_test, writers,
                out_dir / csv_name.format(pct=pct),
            )
        results.append((name, pct, val_acc, test_acc))
        return pct, val_acc, test_acc

    # ──────────────────────────────────────────────────────────────────
    # 1. Baseline Mahalanobis
    # ──────────────────────────────────────────────────────────────────
    if print_header:
        print("\n" + "=" * 60)
        print("1. Baseline Mahalanobis (single prototype)")
    means = fit_class_means(train_emb, train_labels, n_writers)
    cov_inv = residual_covariance(train_emb, train_labels, means)

    val_maha = min_mahalanobis(val_emb, means, cov_inv)
    test_maha = min_mahalanobis(test_emb, means, cov_inv)

    # Score: higher = more known. We use NEGATIVE Mahalanobis distance.
    val_score, test_score = -val_maha, -test_maha
    pct, val_acc, test_acc = _finalize(
        "baseline_maha", val_score, test_score, "baseline_maha_unk{pct}.csv"
    )
    print(f"  best pct={pct}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 2. Multi-prototype Mahalanobis
    # ──────────────────────────────────────────────────────────────────
    if print_header:
        print("\n2. Multi-prototype Mahalanobis")
    for k in [2, 3, 4]:
        protos = fit_class_kmeans(train_emb, train_labels, n_writers, k)
        val_maha = min_mahalanobis(val_emb, protos, cov_inv)
        test_maha = min_mahalanobis(test_emb, protos, cov_inv)
        val_score, test_score = -val_maha, -test_maha
        pct, val_acc, test_acc = _finalize(
            f"mp{k}", val_score, test_score, f"mp{k}_unk{{pct}}.csv"
        )
        print(f"  k={k}: pct={pct}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 3. PCA + Mahalanobis 
    # ──────────────────────────────────────────────────────────────────
    if print_header:
        print("\n3. PCA + Mahalanobis")
    pca_results = {}
    for dim in [64, 128, 256]:
        pca = PCA(n_components=min(dim, D - 1), random_state=42).fit(train_emb)
        tr = pca.transform(train_emb)
        vl = pca.transform(val_emb)
        ts = pca.transform(test_emb)
        m = fit_class_means(tr, train_labels, n_writers)
        ci = residual_covariance(tr, train_labels, m)
        val_maha = min_mahalanobis(vl, m, ci)
        test_maha = min_mahalanobis(ts, m, ci)
        val_score, test_score = -val_maha, -test_maha
        pct, val_acc, test_acc = _finalize(
            f"pca{dim}", val_score, test_score, f"pca{dim}_unk{{pct}}.csv"
        )
        pca_results[dim] = (tr, vl, ts, m, ci, val_maha, test_maha)
        print(f"  dim={dim}: pct={pct}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}  "
              f"(var_exp={pca.explained_variance_ratio_.sum():.3f})")

    # ──────────────────────────────────────────────────────────────────
    # 4. PCA + Multi-prototype combined
    # ──────────────────────────────────────────────────────────────────
    if print_header:
        print("\n4. PCA + Multi-prototype")
    for (dim, k) in [(128, 3), (256, 2), (64, 4)]:
        pca = PCA(n_components=min(dim, D - 1), random_state=42).fit(train_emb)
        tr = pca.transform(train_emb)
        vl = pca.transform(val_emb)
        ts = pca.transform(test_emb)
        protos = fit_class_kmeans(tr, train_labels, n_writers, k)
        # Covariance from train residuals to NEAREST prototype within class
        nearest_protos = np.zeros_like(tr)
        for i in range(len(tr)):
            class_protos = protos[train_labels[i] * k:train_labels[i] * k + k]
            d = np.linalg.norm(tr[i] - class_protos, axis=1)
            nearest_protos[i] = class_protos[np.argmin(d)]
        lw = LedoitWolf().fit(tr - nearest_protos)
        ci = np.linalg.inv(lw.covariance_)
        val_maha = min_mahalanobis(vl, protos, ci)
        test_maha = min_mahalanobis(ts, protos, ci)
        val_score, test_score = -val_maha, -test_maha
        pct, val_acc, test_acc = _finalize(
            f"pca{dim}_mp{k}", val_score, test_score,
            f"pca{dim}_mp{k}_unk{{pct}}.csv",
        )
        print(f"  dim={dim} k={k}: pct={pct}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 5. Weighted blend (baseline + PCA-128)
    # ──────────────────────────────────────────────────────────────────
    if print_header:
        print("\n5. Weighted blend (baseline + PCA-128)")
    base_val = -min_mahalanobis(val_emb, means, cov_inv)
    base_test = -min_mahalanobis(test_emb, means, cov_inv)
    if 128 in pca_results:
        _, _, _, _, _, p_val_m, p_test_m = pca_results[128]
        p_val, p_test = -p_val_m, -p_test_m

        def fit_norm01(x_fit):
            lo, hi = float(x_fit.min()), float(x_fit.max())
            return lambda x: (x - lo) / (hi - lo + 1e-10)

        norm_base, norm_pca = fit_norm01(base_val), fit_norm01(p_val)
        bv, bt = norm_base(base_val), norm_base(base_test)
        pv, pt = norm_pca(p_val), norm_pca(p_test)
        for alpha in [0.3, 0.5, 0.7, 0.8, 0.9]:
            val_score = alpha * bv + (1 - alpha) * pv
            test_score = alpha * bt + (1 - alpha) * pt
            pct, val_acc, test_acc = _finalize(
                f"blend_a{alpha:.1f}", val_score, test_score,
                f"blend_a{int(alpha*10)}_unk{{pct}}.csv",
            )
            print(f"  alpha={alpha}: pct={pct}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 6. Cluster-level OOD (whole-cluster known/unknown gating)
    # ──────────────────────────────────────────────────────────────────
    
    if print_header:
        print("\n6. Cluster-level OOD (PCA-128 + agglomerative cosine-linkage)")
    if 128 in pca_results:
        _, vl, ts, m, ci, _, _ = pca_results[128]
        n_val, n_test = len(vl), len(ts)

        cluster_configs = [
            ("Kw",    n_writers),
            ("K2w",   2 * n_writers),
            ("Kavg3", max(2, n_test // 3)),
        ]

        for tag, K in cluster_configs:
            K_val = max(2, min(K, n_val - 1))
            K_test = max(2, min(K, n_test - 1))

            val_clu = AgglomerativeClustering(
                n_clusters=K_val, metric="cosine", linkage="average"
            ).fit_predict(vl)
            test_clu = AgglomerativeClustering(
                n_clusters=K_test, metric="cosine", linkage="average"
            ).fit_predict(ts)

            val_centroids = np.stack([vl[val_clu == c].mean(axis=0) for c in range(K_val)])
            test_centroids = np.stack([ts[test_clu == c].mean(axis=0) for c in range(K_test)])

            val_clu_maha = min_mahalanobis(val_centroids, m, ci)
            test_clu_maha = min_mahalanobis(test_centroids, m, ci)

            # Each sample inherits its cluster's score (negate so higher = more known)
            val_score = -val_clu_maha[val_clu]
            test_score = -test_clu_maha[test_clu]

            pct, val_acc, test_acc = _finalize(
                f"cluster_{tag}", val_score, test_score,
                f"cluster_{tag}_unk{{pct}}.csv",
            )
            print(f"  {tag} (K_val={K_val}, K_test={K_test}): "
                  f"pct={pct}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ──────────────────────────────────────────────────────────────────
    # 7. K-sweep cluster-OOD with writer assignment + per-sample gating
    # ──────────────────────────────────────────────────────────────────
    if print_header:
        print("\n7. K-sweep cluster-OOD + cluster-level writer + gating "
              "(val-tuned K, n_keep, T)")

    
    train_emb_l2 = _l2norm(train_emb.astype(np.float32))
    val_emb_l2 = _l2norm(val_emb.astype(np.float32))
    test_emb_l2 = _l2norm(test_emb.astype(np.float32))
    proto = np.zeros((n_writers, D), dtype=np.float32)
    for w in range(n_writers):
        mask = train_labels == w
        if mask.sum() > 0:
            proto[w] = train_emb_l2[mask].mean(axis=0)
    proto_l2 = _l2norm(proto)

    
    K_VALUES = sorted(set([
        n_writers,
        n_writers + max(2, n_writers // 11),     
        n_writers + max(4, n_writers // 7),      
        int(round(n_writers * 1.36)),            
        int(round(n_writers * 1.5)),
        int(round(n_writers * 1.75)),
    ]))
    
    n_val_eff, n_test_eff = len(val_emb), len(test_emb)
    K_VALUES = [k for k in K_VALUES if 2 <= k < min(n_val_eff, n_test_eff)]
   
    N_KEEP_FRACS = (0.08, 0.12, 0.16, 0.20)
    T_VALUES = (0.0, 0.05, 0.10, 0.15)

    best = None  
    sweep_rows = []
    for K in K_VALUES:
        val_diag = cluster_and_score(val_emb_l2, proto_l2, K)
        for frac in N_KEEP_FRACS:
            n_keep = max(1, min(K, int(round(K * frac))))
            for T in T_VALUES:
                vp, _ = gated_predict_from_clusters(val_diag, val_cosine, n_keep, T)
                v_acc = open_set_top1_acc(vp, val_labels_int, val_is_unk)
                sweep_rows.append({"K": K, "n_keep": n_keep, "T": T, "val_acc": v_acc})
                if best is None or v_acc > best[0]:
                    best = (v_acc, K, n_keep, T)

    if best is not None:
        v_acc, K_best, n_keep_best, T_best = best
        
        test_diag = cluster_and_score(test_emb_l2, proto_l2, K_best)
        test_pred, test_keep_set = gated_predict_from_clusters(
            test_diag, test_cosine, n_keep_best, T_best
        )
        t_acc = open_set_top1_acc(test_pred, test_labels_int, test_is_unk)
        n_unk_test = int((test_pred < 0).sum())
        name = f"ksweep_K{K_best:03d}_keep{n_keep_best:02d}_T{int(T_best*1000):04d}"
        if write_subs:
            write_submission_from_predidx(
                test_image_id, test_pred, writers, out_dir / f"{name}.csv"
            )
        
        results.append((name, -1, v_acc, t_acc))
        print(f"  best: K={K_best}  n_keep={n_keep_best}  T={T_best:.3f}  "
              f"val_acc={v_acc:.4f}  test_acc={t_acc:.4f}  "
              f"n_unknown_test={n_unk_test}/{n_test_eff}")
        if write_subs:
            pd.DataFrame(sweep_rows).to_csv(
                out_dir / "_ksweep_grid.csv", index=False
            )

    # ──────────────────────────────────────────────────────────────────
    # 8. Positive-Negative prototype gate stacked on the K-sweep best.
    # ──────────────────────────────────────────────────────────────────
    if best is not None:
        if print_header:
            print("\n8. Positive-Negative prototype gate (val-tuned)")
        K_best_pn, n_keep_best_pn, T_best_pn = best[1], best[2], best[3]
        # Re-cluster val and test at the K-sweep best K
        val_diag = cluster_and_score(val_emb_l2, proto_l2, K_best_pn)
        test_diag = cluster_and_score(test_emb_l2, proto_l2, K_best_pn)
        # Per-writer top-1 confusor (nearest other writer's prototype)
        cross_sim = proto_l2 @ proto_l2.T
        np.fill_diagonal(cross_sim, -np.inf)
        confusor = cross_sim.argmax(axis=1)
        # Per-sample top-1 / margin
        val_pred_per = val_cosine.argmax(axis=1)
        val_margin = np.sort(val_cosine, axis=1)[:, -1] - np.sort(val_cosine, axis=1)[:, -2]
        test_pred_per = test_cosine.argmax(axis=1)
        test_margin = np.sort(test_cosine, axis=1)[:, -1] - np.sort(test_cosine, axis=1)[:, -2]

        def _pn_apply(diag, cosine_, pred_per_, margin_, n_keep_, T_, pn_m):
            keep = set(np.argsort(-diag["clu_knownness"])[:n_keep_].tolist())
            clu = diag["clu"]; clu_w = diag["clu_writer"]; emb_ = None
            preds = []
            for i in range(len(clu)):
                cid = int(clu[i])
                if cid not in keep:
                    preds.append(-1); continue
                w_pred = int(pred_per_[i]) if margin_[i] >= T_ else int(clu_w[cid])
                preds.append(w_pred)
            return np.array(preds, dtype=np.int64)

       
        best_pn = None
        for pn_m in (0.00, 0.02, 0.04, 0.06):
            vp = _pn_apply(val_diag, val_cosine, val_pred_per, val_margin,
                           n_keep_best_pn, T_best_pn, pn_m).copy()
            # PN-gate rejection
            for i in range(len(vp)):
                if vp[i] < 0:
                    continue
                w_pred = int(vp[i])
                pos_s = float(val_emb_l2[i] @ proto_l2[w_pred])
                neg_s = float(val_emb_l2[i] @ proto_l2[int(confusor[w_pred])])
                if pos_s - neg_s < pn_m:
                    vp[i] = -1
            v_acc = open_set_top1_acc(vp, val_labels_int, val_is_unk)
            if best_pn is None or v_acc > best_pn[0]:
                best_pn = (v_acc, pn_m)
        v_acc_pn, pn_m_best = best_pn
        # Apply best margin to test
        tp = _pn_apply(test_diag, test_cosine, test_pred_per, test_margin,
                       n_keep_best_pn, T_best_pn, pn_m_best).copy()
        for i in range(len(tp)):
            if tp[i] < 0:
                continue
            w_pred = int(tp[i])
            pos_s = float(test_emb_l2[i] @ proto_l2[w_pred])
            neg_s = float(test_emb_l2[i] @ proto_l2[int(confusor[w_pred])])
            if pos_s - neg_s < pn_m_best:
                tp[i] = -1
        t_acc_pn = open_set_top1_acc(tp, test_labels_int, test_is_unk)
        n_unk_pn = int((tp < 0).sum())
        name_pn = (f"ksweep_K{K_best_pn:03d}_keep{n_keep_best_pn:02d}"
                    f"_T{int(T_best_pn*1000):04d}_pngate_m{int(pn_m_best*1000):03d}")
        if write_subs:
            write_submission_from_predidx(test_image_id, tp, writers, out_dir / f"{name_pn}.csv")
        results.append((name_pn, -1, v_acc_pn, t_acc_pn))
        print(f"  best PN margin={pn_m_best:.3f}  val_acc={v_acc_pn:.4f}  "
              f"test_acc={t_acc_pn:.4f}  n_unk={n_unk_pn}/{n_test_eff}")
        # Save the predictions for the CV-scorer to stack on top
        _pn_test_pred = tp
        _pn_val_pred = None  # we'll recompute for val if needed
    else:
        _pn_test_pred = None

    # ──────────────────────────────────────────────────────────────────
    # 9. CV-calibrated 7-feature rejection scorer stacked on the PN-gate.
    # ──────────────────────────────────────────────────────────────────
    if best is not None and _pn_test_pred is not None:
        if print_header:
            print("\n9. CV-calibrated 7-feature rejection scorer (val-tuned p_threshold)")
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.preprocessing import StandardScaler

        def _feat7(emb, cos, proto_, conf, centroids_, assign_, sizes_):
            Q = emb.shape[0]
            F = np.zeros((Q, 7), dtype=np.float32)
            F[:, 0] = (centroids_ @ proto_.T).max(axis=1)[assign_]
            F[:, 1] = cos.max(axis=1)
            s = np.sort(cos, axis=1)
            F[:, 2] = s[:, -1] - s[:, -2]
            F[:, 3] = np.log1p(sizes_[assign_].astype(np.float32))
            F[:, 4] = (emb * centroids_[assign_]).sum(axis=1)
            pred = cos.argmax(axis=1)
            pos_ = cos[np.arange(Q), pred]
            neg_ = cos[np.arange(Q), conf[pred]]
            F[:, 5] = pos_ - neg_
            F[:, 6] = cos.var(axis=1)
            return F

        # Build CV-OOF training features on train_emb
        rng = np.random.default_rng(0)
        widx = np.arange(n_writers); rng.shuffle(widx)
        folds = np.array_split(widx, 5)
        cv_X_l, cv_y_l = [], []
        for held in folds:
            held_s = set(held.tolist())
            in_d = sorted(set(range(n_writers)) - held_s)
            p_loc = np.zeros((n_writers, D), dtype=np.float32)
            for w in in_d:
                m = train_labels == w
                if m.sum() > 0:
                    p_loc[w] = train_emb_l2[m].mean(axis=0)
            p_loc_l2 = _l2norm(p_loc + 1e-12)
            c_loc = p_loc_l2 @ p_loc_l2.T
            np.fill_diagonal(c_loc, -np.inf)
            for w in held_s:
                c_loc[w, :] = -np.inf; c_loc[:, w] = -np.inf
            conf_loc = c_loc.argmax(axis=1)
            samp = rng.choice(len(train_emb_l2), size=min(5000, len(train_emb_l2)), replace=False)
            eval_e = train_emb_l2[samp]
            eval_c = eval_e @ p_loc_l2.T
            from sklearn.cluster import AgglomerativeClustering as _AggCV
            clu_loc = _AggCV(n_clusters=K_best_pn, metric="cosine",
                              linkage="average").fit_predict(eval_e)
            cents = _l2norm(np.stack([eval_e[clu_loc == c].mean(axis=0)
                                       for c in range(K_best_pn)]))
            sz = np.bincount(clu_loc, minlength=K_best_pn)
            cv_X_l.append(_feat7(eval_e, eval_c, p_loc_l2, conf_loc, cents, clu_loc, sz))
            cv_y_l.append(np.array([1 if l_ not in held_s else 0 for l_ in train_labels[samp]]))
        X_all = np.concatenate(cv_X_l, axis=0); y_all = np.concatenate(cv_y_l, axis=0)
        scaler = StandardScaler().fit(X_all)
        lr = LogisticRegressionCV(cv=3, max_iter=2000, scoring="roc_auc",
                                   class_weight="balanced",
                                   random_state=0).fit(scaler.transform(X_all), y_all)

        # Compute test p_known
        test_X = _feat7(test_emb_l2, test_cosine, proto_l2, confusor,
                        np.stack([test_emb_l2[test_diag["clu"] == c].mean(axis=0)
                                   for c in range(K_best_pn)]) /
                        (np.linalg.norm(np.stack([test_emb_l2[test_diag["clu"] == c].mean(axis=0)
                                                    for c in range(K_best_pn)]), axis=1, keepdims=True) + 1e-12),
                        test_diag["clu"], np.bincount(test_diag["clu"], minlength=K_best_pn))
        p_known_test = lr.predict_proba(scaler.transform(test_X))[:, 1]
        # Compute val p_known similarly
        val_centroids = np.stack([val_emb_l2[val_diag["clu"] == c].mean(axis=0)
                                    for c in range(K_best_pn)])
        val_centroids = val_centroids / (np.linalg.norm(val_centroids, axis=1, keepdims=True) + 1e-12)
        val_X = _feat7(val_emb_l2, val_cosine, proto_l2, confusor,
                        val_centroids, val_diag["clu"],
                        np.bincount(val_diag["clu"], minlength=K_best_pn))
        p_known_val = lr.predict_proba(scaler.transform(val_X))[:, 1]

        # Compute base val PN-predictions to stack CV-scorer rejection on
        _pn_val_pred = _pn_apply(val_diag, val_cosine, val_pred_per, val_margin,
                                  n_keep_best_pn, T_best_pn, pn_m_best).copy()
        for i in range(len(_pn_val_pred)):
            if _pn_val_pred[i] < 0:
                continue
            w_pred = int(_pn_val_pred[i])
            pos_s = float(val_emb_l2[i] @ proto_l2[w_pred])
            neg_s = float(val_emb_l2[i] @ proto_l2[int(confusor[w_pred])])
            if pos_s - neg_s < pn_m_best:
                _pn_val_pred[i] = -1

        best_cv = None
        for p_th in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
            vp = _pn_val_pred.copy()
            for i in range(len(vp)):
                if vp[i] >= 0 and p_known_val[i] < p_th:
                    vp[i] = -1
            v_acc = open_set_top1_acc(vp, val_labels_int, val_is_unk)
            if best_cv is None or v_acc > best_cv[0]:
                best_cv = (v_acc, p_th)
        v_acc_cv, p_th_best = best_cv

        tp_cv = _pn_test_pred.copy()
        for i in range(len(tp_cv)):
            if tp_cv[i] >= 0 and p_known_test[i] < p_th_best:
                tp_cv[i] = -1
        t_acc_cv = open_set_top1_acc(tp_cv, test_labels_int, test_is_unk)
        n_unk_cv = int((tp_cv < 0).sum())
        name_cv = (f"ksweep_K{K_best_pn:03d}_keep{n_keep_best_pn:02d}"
                    f"_T{int(T_best_pn*1000):04d}_pn{int(pn_m_best*1000):03d}"
                    f"_cv{int(p_th_best*100):03d}")
        if write_subs:
            write_submission_from_predidx(test_image_id, tp_cv, writers, out_dir / f"{name_cv}.csv")
        results.append((name_cv, -1, v_acc_cv, t_acc_cv))
        print(f"  best CV p_threshold={p_th_best:.2f}  val_acc={v_acc_cv:.4f}  "
              f"test_acc={t_acc_cv:.4f}  n_unk={n_unk_cv}/{n_test_eff}")

    # ──────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────
    if write_subs:
        print("\n" + "=" * 60)
        print("SUMMARY (sorted by val_acc, then test_acc)")
        print("=" * 60)
        results.sort(key=lambda r: (-r[2], -r[3]))
        print(f"{'method':<32} {'pct':>4} {'val_acc':>10} {'test_acc':>10}")
        for name, pct, va, ta in results:
            print(f"{name:<32} {pct:>4} {va:>10.4f} {ta:>10.4f}")

        # Save summary CSV
        pd.DataFrame(results, columns=["method", "unk_pct", "val_acc", "test_acc"]).to_csv(
            out_dir / "_summary.csv", index=False
        )
        print(f"\nAll submissions + _summary.csv saved to {out_dir}/")
    return results


# ──────────────────────────────────────────────────────────────────────
# Multi-seed ensemble
# ──────────────────────────────────────────────────────────────────────
def run_ensemble(emb_paths, out_dir):
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = len(emb_paths)

    print(f"\nLoading {n_seeds} seeds for ensemble:")
    npzs = []
    for p in emb_paths:
        print(f"  {p}")
        npzs.append(np.load(p, allow_pickle=True))

    # Sanity: same writers, same test_image_id, same val_image_id across seeds.
    writers = list(npzs[0]["writers"].astype(str))
    ref_test_ids = npzs[0]["test_image_id"]
    ref_val_ids = npzs[0]["val_image_id"]
    for i, n in enumerate(npzs[1:], start=1):
        if list(n["writers"].astype(str)) != writers:
            raise SystemExit(f"Writer ordering differs between seed 0 and seed {i}")
        if not np.array_equal(n["test_image_id"], ref_test_ids):
            raise SystemExit(f"test_image_id differs between seed 0 and seed {i}")
        if not np.array_equal(n["val_image_id"], ref_val_ids):
            raise SystemExit(f"val_image_id differs between seed 0 and seed {i}")

    # ── Step 1: collect per-seed scores from each ablation ──
    per_seed_scores = []   # list[dict[name -> (val_score, test_score)]]
    for i, npz in enumerate(npzs):
        print(f"\n{'='*60}\nSeed {i+1}/{n_seeds}: running ablations\n{'='*60}")
        collector = {}
        run_ablations(npz, out_dir / f"_per_seed_{i}", None,
                      write_subs=False, score_collector=collector,
                      print_header=(i == 0))
        per_seed_scores.append(collector)

    # ── Step 2 & 3: average scores and cosines ──
    val_cosine_avg = np.mean(
        [npz["val_cosine"].astype(np.float32) for npz in npzs], axis=0
    )
    test_cosine_avg = np.mean(
        [npz["test_cosine"].astype(np.float32) for npz in npzs], axis=0
    )
    val_pred_idx = val_cosine_avg.argmax(axis=1)
    test_pred_idx = test_cosine_avg.argmax(axis=1)

    val_writer_id = npzs[0]["val_writer_id"].astype(str)
    test_writer_id = npzs[0]["test_writer_id"].astype(str)
    val_is_unk = (val_writer_id == "-1")
    test_is_unk = (test_writer_id == "-1")
    writer2idx = {w: i for i, w in enumerate(writers)}
    val_labels_int = np.array(
        [writer2idx.get(w, -2) for w in val_writer_id], dtype=np.int64
    )
    test_labels_int = np.array(
        [writer2idx.get(w, -2) for w in test_writer_id], dtype=np.int64
    )

    # ── Step 4 & 5: tune + write per method ──
    method_names = list(per_seed_scores[0].keys())
    results = []
    print(f"\n{'='*60}\nENSEMBLE: averaging scores across {n_seeds} seeds\n{'='*60}")
    for name in method_names:
        val_score = np.mean([s[name][0] for s in per_seed_scores], axis=0)
        test_score = np.mean([s[name][1] for s in per_seed_scores], axis=0)
        pct, val_acc, _ = tune_threshold(
            val_score, val_labels_int, val_is_unk, val_pred_idx
        )
        is_unk_test = apply_threshold(test_score, pct)

        # Test top-1 acc (only meaningful when test labels are present)
        n_correct = 0
        for i in range(len(is_unk_test)):
            if is_unk_test[i]:
                if test_is_unk[i]:
                    n_correct += 1
            else:
                if (not test_is_unk[i]) and test_pred_idx[i] == test_labels_int[i]:
                    n_correct += 1
        test_acc = n_correct / len(is_unk_test)

        write_submission(
            ref_test_ids, test_pred_idx, is_unk_test, writers,
            out_dir / f"{name}_ensemble_unk{pct}.csv",
        )
        results.append((name, pct, val_acc, test_acc))
        print(f"  {name:<22} pct={pct:>3}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    # ── Step 6: ensemble K-sweep cluster-OOD with writer assignment + gating ──
   
    print(f"\n{'='*60}\nENSEMBLE K-SWEEP CLUSTER-OOD ({n_seeds} seeds)\n{'='*60}")
    train_emb_avg = _l2norm(np.mean(
        [_l2norm(npz["train_emb"].astype(np.float32)) for npz in npzs], axis=0
    ))
    val_emb_avg = _l2norm(np.mean(
        [_l2norm(npz["val_emb"].astype(np.float32)) for npz in npzs], axis=0
    ))
    test_emb_avg = _l2norm(np.mean(
        [_l2norm(npz["test_emb"].astype(np.float32)) for npz in npzs], axis=0
    ))
    train_labels_ref = npzs[0]["train_labels"]
    n_writers = len(writers)
    D = train_emb_avg.shape[1]
    proto_avg = np.zeros((n_writers, D), dtype=np.float32)
    for w in range(n_writers):
        mask = train_labels_ref == w
        if mask.sum() > 0:
            proto_avg[w] = train_emb_avg[mask].mean(axis=0)
    proto_avg_l2 = _l2norm(proto_avg)

    n_val_eff, n_test_eff = len(val_emb_avg), len(test_emb_avg)
    K_VALUES = sorted(set([
        n_writers,
        n_writers + max(2, n_writers // 11),
        n_writers + max(4, n_writers // 7),
        int(round(n_writers * 1.36)),
        int(round(n_writers * 1.5)),
        int(round(n_writers * 1.75)),
    ]))
    K_VALUES = [k for k in K_VALUES if 2 <= k < min(n_val_eff, n_test_eff)]
    N_KEEP_FRACS = (0.08, 0.12, 0.16, 0.20)
    T_VALUES = (0.0, 0.05, 0.10, 0.15)

    best = None
    ens_sweep_rows = []
    for K in K_VALUES:
        val_diag = cluster_and_score(val_emb_avg, proto_avg_l2, K)
        for frac in N_KEEP_FRACS:
            n_keep = max(1, min(K, int(round(K * frac))))
            for T in T_VALUES:
                vp, _ = gated_predict_from_clusters(val_diag, val_cosine_avg, n_keep, T)
                v_acc = open_set_top1_acc(vp, val_labels_int, val_is_unk)
                ens_sweep_rows.append({"K": K, "n_keep": n_keep, "T": T, "val_acc": v_acc})
                if best is None or v_acc > best[0]:
                    best = (v_acc, K, n_keep, T)

    if best is not None:
        v_acc, K_best, n_keep_best, T_best = best
        test_diag = cluster_and_score(test_emb_avg, proto_avg_l2, K_best)
        test_pred, _ = gated_predict_from_clusters(
            test_diag, test_cosine_avg, n_keep_best, T_best
        )
        t_acc = open_set_top1_acc(test_pred, test_labels_int, test_is_unk)
        n_unk_test = int((test_pred < 0).sum())
        name = f"ksweep_K{K_best:03d}_keep{n_keep_best:02d}_T{int(T_best*1000):04d}_ensemble"
        write_submission_from_predidx(
            ref_test_ids, test_pred, writers, out_dir / f"{name}.csv"
        )
        pd.DataFrame(ens_sweep_rows).to_csv(out_dir / "_ksweep_grid_ensemble.csv", index=False)
        results.append((name, -1, v_acc, t_acc))
        print(f"  best: K={K_best}  n_keep={n_keep_best}  T={T_best:.3f}  "
              f"val_acc={v_acc:.4f}  test_acc={t_acc:.4f}  "
              f"n_unknown_test={n_unk_test}/{n_test_eff}")

    results.sort(key=lambda r: (-r[2], -r[3]))
    print(f"\n{'='*60}\nENSEMBLE SUMMARY ({n_seeds} seeds, sorted by val_acc)\n{'='*60}")
    print(f"{'method':<40} {'pct':>4} {'val_acc':>10} {'test_acc':>10}")
    for name, pct, va, ta in results:
        print(f"{name:<40} {pct:>4} {va:>10.4f} {ta:>10.4f}")

    pd.DataFrame(results, columns=["method", "unk_pct", "val_acc", "test_acc"]).to_csv(
        out_dir / "_summary_ensemble.csv", index=False,
    )

    # Save an ensembled npz so eval_metrics.py can score the ensemble directly.
    np.savez(
        out_dir / "embeddings_ensemble.npz",
        train_emb=npzs[0]["train_emb"],
        train_labels=npzs[0]["train_labels"],
        val_emb=npzs[0]["val_emb"],            # placeholder — not used by eval
        val_writer_id=npzs[0]["val_writer_id"],
        val_image_id=ref_val_ids,
        val_cosine=val_cosine_avg,
        test_emb=npzs[0]["test_emb"],          # placeholder — not used by eval
        test_writer_id=npzs[0]["test_writer_id"],
        test_image_id=ref_test_ids,
        test_cosine=test_cosine_avg,
        writers=npzs[0]["writers"],
    )
    print(f"\nEnsemble outputs saved to {out_dir}/")
    print(f"  Submissions:  {{method}}_ensemble_unk{{pct}}.csv")
    print(f"  Summary:      _summary_ensemble.csv")
    print(f"  Avg cosines:  embeddings_ensemble.npz  (feed this to eval_metrics.py)")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, nargs="+", required=True,
                    help="Path(s) to embeddings_seed{N}.npz from train_writerid.py. "
                         "Pass multiple to enable cross-seed score ensembling.")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Where to write submission CSVs + _summary.csv")
    return ap.parse_args()


def main():
    args = parse_args()

    if len(args.emb) == 1:
        print(f"Loading embeddings: {args.emb[0]}")
        npz = np.load(args.emb[0], allow_pickle=True)
        print(f"  train_emb: {npz['train_emb'].shape}")
        print(f"  val_emb:   {npz['val_emb'].shape}")
        print(f"  test_emb:  {npz['test_emb'].shape}")
        print(f"  writers:   {len(npz['writers'])}")
        run_ablations(npz, args.out_dir, None)
    else:
        run_ensemble(args.emb, args.out_dir)


if __name__ == "__main__":
    main()
