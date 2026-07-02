#!/usr/bin/env python3
"""
CV-calibrated multi-feature rejection scorer.

Plan:
  1. Build 5-fold CV on the train set: each fold holds out 8-9 writers as
     "fake unknowns" (their samples get -1 labels).
  2. For each fold, fit prototypes on the remaining 35-36 known writers, then
     compute 7 features per held-out sample.
  3. Train logistic regression on (features, is_truly_known) across all folds.
  4. Apply learned model to test embeddings to produce a per-sample
     known-probability. Use this on top of PN-gate(m=0) to add/remove
     rejections more principled than fixed thresholds.

Features per sample (7):
  f1 cluster_knownness    : max cosine of sample's cluster centroid to any writer proto
  f2 top1_cosine          : max cosine of sample to any writer prototype directly
  f3 margin (top1-top2)   : per-sample confidence margin
  f4 cluster_size         : size of the cluster this sample belongs to (relative)
  f5 sample_to_centroid   : how close is sample to its own cluster's centroid
  f6 pos_neg_diff         : pos_sim - neg_sim using top-1 confusor (PN gate signal)
  f7 train_cosine_var     : variance of cosine score across train prototypes

"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = _THIS_DIR / "submissions_cv_scorer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CSV = _THIS_DIR / "submissions_posthoc_3exp" / "03_pn_gate_m000.csv"  
SEEDS = [42, 137]
K_CLUSTER = 60
N_KEEP = 7
N_FOLDS = 5


def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def compute_features(query_emb, query_cosine,
                     proto_l2, confusor_idx,
                     cluster_centroids, cluster_assignment, cluster_sizes):
    
    Q = query_emb.shape[0]
    feats = np.zeros((Q, 7), dtype=np.float32)
    # f1 cluster knownness (sample inherits its cluster's max-cos to any proto)
    centroid_proto_sim = cluster_centroids @ proto_l2.T  # (K, n_writers)
    clu_knownness = centroid_proto_sim.max(axis=1)       # (K,)
    feats[:, 0] = clu_knownness[cluster_assignment]
    # f2 top1 cosine
    feats[:, 1] = query_cosine.max(axis=1)
    # f3 top1-top2 margin
    s = np.sort(query_cosine, axis=1)
    feats[:, 2] = s[:, -1] - s[:, -2]
    # f4 cluster size (log-scaled)
    feats[:, 3] = np.log1p(cluster_sizes[cluster_assignment].astype(np.float32))
    # f5 sample-to-centroid cosine
    feats[:, 4] = (query_emb * cluster_centroids[cluster_assignment]).sum(axis=1)
    # f6 pos_neg_diff using top-1 confusor of the per-sample-argmax writer
    pred = query_cosine.argmax(axis=1)
    pos_sim = query_cosine[np.arange(Q), pred]
    neg_idx = confusor_idx[pred]
    neg_sim = query_cosine[np.arange(Q), neg_idx]
    feats[:, 5] = pos_sim - neg_sim
    # f7 variance of train_cosine row (high variance = peaked dist, low = uniform)
    feats[:, 6] = query_cosine.var(axis=1)
    return feats


def main():
    rng = np.random.default_rng(0)
    print(f"Loading 2-seed average {SEEDS}...")
    bundles = [np.load(EMB_DIR / f"embeddings_seed_{s}.npz") for s in SEEDS]
    train_emb = l2norm(np.mean([l2norm(b["train_emb"]) for b in bundles], axis=0))
    test_emb = l2norm(np.mean([l2norm(b["test_emb"]) for b in bundles], axis=0))
    train_labels = bundles[0]["train_labels"]
    test_cosine = np.mean([b["test_cosine"] for b in bundles], axis=0)
    train_cosine_per_seed = [b["train_cosine"] for b in bundles]
    train_cosine = np.mean(train_cosine_per_seed, axis=0)

    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_add = pd.read_csv(DATA_DIR / "additional_train.csv")
    df_add_known = df_add[df_add["writer_id"] != "-1"]
    df_all = pd.concat([df_train, df_add_known], ignore_index=True)
    writers = sorted(df_all["writer_id"].unique())
    idx2writer = {i: w for i, w in enumerate(writers)}
    n_writers = len(writers)
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    n_test = len(df_test)
    print(f"  n_writers={n_writers}, n_test={n_test}, n_train_patches={len(train_emb)}")

    # ── BUILD CV-OOF features (target = is sample from a held-in writer?) ──
    print("\n[1] Build CV-OOF training features on train_emb...")
    writer_indices = np.arange(n_writers)
    rng.shuffle(writer_indices)
    folds = np.array_split(writer_indices, N_FOLDS)
    cv_X, cv_y = [], []
    for fi, held_writers in enumerate(folds):
        held_set = set(held_writers.tolist())
       
        train_mask = np.array([lab not in held_set for lab in train_labels])
        # Build proto from only in-distribution writers
        in_distrib = sorted(set(range(n_writers)) - held_set)
        proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
        for w in in_distrib:
            m = train_labels == w
            if m.sum() > 0:
                proto[w] = train_emb[m].mean(axis=0)
        
        proto_l2 = l2norm(proto + 1e-12)
        # Confusor per writer (within in-distribution only)
        cross = proto_l2 @ proto_l2.T
        np.fill_diagonal(cross, -np.inf)
        for w in held_set:
            cross[w, :] = -np.inf
            cross[:, w] = -np.inf
        confusor = cross.argmax(axis=1)
        
        idx = rng.choice(len(train_emb), size=min(5000, len(train_emb)), replace=False)
        eval_emb = train_emb[idx]
        eval_labels = train_labels[idx]
        # Compute eval_cosine using proto_l2
        eval_cos = eval_emb @ proto_l2.T
        # Cluster eval_emb at K=N_KEEP*estimate or just K=60-relative
        K_eff = min(K_CLUSTER, len(eval_emb) - 1)
        clu = AgglomerativeClustering(
            n_clusters=K_eff, metric="cosine", linkage="average"
        ).fit_predict(eval_emb)
        centroids = l2norm(np.stack([eval_emb[clu == c].mean(axis=0) for c in range(K_eff)]))
        sizes = np.bincount(clu, minlength=K_eff)
        # Features for these eval samples
        X = compute_features(eval_emb, eval_cos, proto_l2, confusor,
                              centroids, clu, sizes)
        # Target: y=1 if writer is in-distribution (KNOWN), y=0 if held-out (UNKNOWN)
        y = np.array([1 if lab not in held_set else 0 for lab in eval_labels])
        cv_X.append(X)
        cv_y.append(y)
        print(f"  fold {fi}: held={len(held_set)} writers, n_eval={len(idx)}, "
              f"known/unknown={(y==1).sum()}/{(y==0).sum()}")

    X_all = np.concatenate(cv_X, axis=0)
    y_all = np.concatenate(cv_y, axis=0)
    print(f"\n[2] Train logistic regression on {len(y_all)} CV samples...")
    scaler = StandardScaler().fit(X_all)
    X_all_s = scaler.transform(X_all)
    lr = LogisticRegressionCV(cv=3, max_iter=2000, scoring="roc_auc",
                              class_weight="balanced", random_state=0).fit(X_all_s, y_all)
    print(f"  feature coeffs: {dict(zip(['knownness','top1','margin','log_size','to_centroid','pos_neg','var'], lr.coef_[0].round(3)))}")
    print(f"  intercept: {float(lr.intercept_[0]):.3f}")
    cv_pred_prob = lr.predict_proba(X_all_s)[:, 1]
    from sklearn.metrics import roc_auc_score
    print(f"  CV AUROC: {roc_auc_score(y_all, cv_pred_prob):.4f}")

    # ── APPLY learned model to test ──
    print("\n[3] Apply to test...")
    # Build the FULL train-prototype set (all writers, no holdout)
    proto_full = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        m = train_labels == w
        if m.sum() > 0:
            proto_full[w] = train_emb[m].mean(axis=0)
    proto_full_l2 = l2norm(proto_full)
    cross = proto_full_l2 @ proto_full_l2.T
    np.fill_diagonal(cross, -np.inf)
    confusor_full = cross.argmax(axis=1)
    # Cluster test
    clu_test = AgglomerativeClustering(
        n_clusters=K_CLUSTER, metric="cosine", linkage="average"
    ).fit_predict(test_emb)
    centroids_test = l2norm(np.stack(
        [test_emb[clu_test == c].mean(axis=0) for c in range(K_CLUSTER)]))
    sizes_test = np.bincount(clu_test, minlength=K_CLUSTER)
    # Features for test
    X_test = compute_features(test_emb, test_cosine, proto_full_l2, confusor_full,
                               centroids_test, clu_test, sizes_test)
    X_test_s = scaler.transform(X_test)
    p_known = lr.predict_proba(X_test_s)[:, 1]
    print(f"  p_known stats: min={p_known.min():.3f} p25={np.percentile(p_known,25):.3f} "
          f"p50={np.percentile(p_known,50):.3f} p75={np.percentile(p_known,75):.3f} "
          f"max={p_known.max():.3f}")

    # ── Reproduce K=60 keep=7 PN-gate decisions, then OVERLAY learned score ──
    sim = centroids_test @ proto_full_l2.T
    clu_writer = sim.argmax(axis=1)
    clu_known = sim.max(axis=1)
    keep_set = set(np.argsort(-clu_known)[:N_KEEP].tolist())
    pred_per = test_cosine.argmax(axis=1)
    s = np.sort(test_cosine, axis=1)
    margin = s[:, -1] - s[:, -2]

    def base_pred_with_pn(i):
        cid = int(clu_test[i])
        if cid not in keep_set:
            return -1
        if margin[i] >= 0.10:
            w_pred = int(pred_per[i])
        else:
            w_pred = int(clu_writer[cid])
        pos_sim = float(test_emb[i] @ proto_full_l2[w_pred])
        neg_sim = float(test_emb[i] @ proto_full_l2[int(confusor_full[w_pred])])
        if pos_sim - neg_sim < 0.0:
            return -1
        return w_pred

    base_pred_idx = np.array([base_pred_with_pn(i) for i in range(n_test)])
    n_base_unk = int((base_pred_idx < 0).sum())
    print(f"  base PN-gate (m=0) n_unknown: {n_base_unk}")
   

    best_preds = pd.read_csv(BEST_CSV)["writer_id"].astype(str).tolist() \
        if BEST_CSV.exists() else None
    rows = []

    def emit(name, pred_idx):
        preds = ["-1" if p < 0 else idx2writer[int(p)] for p in pred_idx]
        n_unk = sum(1 for x in preds if x == "-1")
        pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds}).to_csv(
            OUT_DIR / f"{name}.csv", index=False)
        diff_best = sum(1 for a, b in zip(preds, best_preds) if a != b) if best_preds else "n/a"
        rows.append({"name": name, "n_unknown": n_unk, "diff_vs_best": diff_best})

    # Strategy A: REJECT additional low-p samples
    print("\n[4] Strategy A: REJECT samples where base_known but p_known < threshold")
    for th in (0.25, 0.30, 0.35, 0.40, 0.45):
        pred_idx = base_pred_idx.copy()
        flip_count = 0
        for i in range(n_test):
            if pred_idx[i] >= 0 and p_known[i] < th:
                pred_idx[i] = -1
                flip_count += 1
        emit(f"A_reject_pk_lt_{int(th*100):03d}", pred_idx)
        print(f"  th={th}: flipped {flip_count} to -1, n_unk={rows[-1]['n_unknown']}, "
              f"diff_vs_best={rows[-1]['diff_vs_best']}")

    # Strategy B: RECOVER high-p samples currently rejected
    print("\n[5] Strategy B: RECOVER samples where base=-1 but p_known > threshold")
    # For recovered samples, use per-sample argmax as the writer
    for th in (0.85, 0.90, 0.92, 0.95):
        pred_idx = base_pred_idx.copy()
        flip_count = 0
        for i in range(n_test):
            if pred_idx[i] < 0 and p_known[i] >= th:
                cid = int(clu_test[i])
                # Only recover if cluster is in keep_set (don't promote unkept-cluster samples)
                if cid in keep_set:
                    pred_idx[i] = int(pred_per[i])
                    flip_count += 1
        emit(f"B_recover_pk_gt_{int(th*100):03d}", pred_idx)
        print(f"  th={th}: recovered {flip_count} from -1, n_unk={rows[-1]['n_unknown']}, "
              f"diff_vs_best={rows[-1]['diff_vs_best']}")

    # Strategy C: do both (reject low, recover high)
    print("\n[6] Strategy C: combine A+B")
    for a_th, b_th in [(0.30, 0.90), (0.35, 0.90), (0.30, 0.95), (0.25, 0.92)]:
        pred_idx = base_pred_idx.copy()
        nA = nB = 0
        for i in range(n_test):
            if pred_idx[i] >= 0 and p_known[i] < a_th:
                pred_idx[i] = -1
                nA += 1
            elif pred_idx[i] < 0 and p_known[i] >= b_th:
                cid = int(clu_test[i])
                if cid in keep_set:
                    pred_idx[i] = int(pred_per[i])
                    nB += 1
        emit(f"C_a{int(a_th*100):03d}_b{int(b_th*100):03d}", pred_idx)
        print(f"  a_th={a_th} b_th={b_th}: A_rej={nA} B_rec={nB}, n_unk={rows[-1]['n_unknown']}, "
              f"diff={rows[-1]['diff_vs_best']}")

    df = pd.DataFrame(rows).sort_values("diff_vs_best")
    df.to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\n{'='*72}\n{len(rows)} variants in {OUT_DIR.resolve()}")
    print("\nSorted by diff vs current best 0.65045:")
    print(df.to_string(index=False))
    print("\nSubmit priority: variants with SMALL diff and reasonable Strategy direction:")
    print("  - A_reject  : risk = removing correct known predictions; gain = catching wrong known")
    print("  - B_recover : risk = adding wrong known predictions; gain = recovering correct ones")
    print("  - C combined: tries both directions")


if __name__ == "__main__":
    main()
