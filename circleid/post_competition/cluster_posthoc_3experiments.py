#!/usr/bin/env python3
"""
post-hoc experiments on the locked 2-seed (42, 137) trunk:

1. Power-norm sanity — apply signed-sqrt to writer_emb, then K=60 keep=7 T=0.10.
   Wrong layer for true power-norm (it should be inside NetVLAD before vlad_proj),
   but tests whether any signed-sqrt regularization helps.

2. Cluster-restricted DBA/AQE smoothing — for each test sample, replace its
   embedding with norm(e + alpha * mean(top-k same-cluster neighbors)).
   Constraint: only smooths within the same K=60 cluster (no cross-cluster
   bleed). Preserves kept-7 cluster identities by construction.

3. Positive-Negative prototype gate — for each writer, find its most-confusable
   other writer (the negative prototype = hardest cross-class). For samples
   predicted as writer w, if dist(sample, neg_proto[w]) < dist(sample, pos_proto[w]),
   reject as '-1' (the sample is closer to a confusor than the claimed writer).

"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = _THIS_DIR / "submissions_posthoc_3exp"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_CSV = _THIS_DIR / "submissions_k_sweep" / "K060_keep07_gated.csv"

SEEDS = [42, 137]
K = 60
N_KEEP = 7
GATING_T = 0.10


def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def load_avg(seeds):
    bundles = [np.load(EMB_DIR / f"embeddings_seed_{s}.npz") for s in seeds]
    train_emb = l2norm(np.mean([l2norm(b["train_emb"]) for b in bundles], axis=0))
    test_emb = l2norm(np.mean([l2norm(b["test_emb"]) for b in bundles], axis=0))
    train_labels = bundles[0]["train_labels"]
    test_cosine = np.mean([b["test_cosine"] for b in bundles], axis=0)
    return train_emb, train_labels, test_emb, test_cosine


def run_k60_keep7_gated(train_emb, train_labels, test_emb, test_cosine,
                        n_writers, idx2writer, n_test, df_test):
    """Standard winning rule. Returns predictions list + clu + keep_set."""
    clu = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    ).fit_predict(test_emb)
    centroids = l2norm(np.stack([test_emb[clu == c].mean(axis=0) for c in range(K)]))
    proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        mask = train_labels == w
        if mask.sum() > 0:
            proto[w] = train_emb[mask].mean(axis=0)
    proto_l2 = l2norm(proto)
    sim = centroids @ proto_l2.T
    clu_writer = sim.argmax(axis=1)
    clu_known = sim.max(axis=1)
    keep_set = set(np.argsort(-clu_known)[:N_KEEP].tolist())

    pred_per = test_cosine.argmax(axis=1)
    sorted_cos = np.sort(test_cosine, axis=1)
    margin = sorted_cos[:, -1] - sorted_cos[:, -2]

    preds = []
    for i in range(n_test):
        cid = int(clu[i])
        if cid not in keep_set:
            preds.append("-1")
        elif margin[i] >= GATING_T:
            preds.append(idx2writer[pred_per[i]])
        else:
            preds.append(idx2writer[int(clu_writer[cid])])
    return preds, clu, keep_set, clu_writer, pred_per, margin, proto_l2


def emit(name, preds, df_test, n_test, baseline_preds, **extra):
    n_unk = sum(1 for p in preds if p == "-1")
    pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds}).to_csv(
        OUT_DIR / f"{name}.csv", index=False)
    diff = sum(1 for a, b in zip(preds, baseline_preds) if a != b) if baseline_preds else "n/a"
    row = {"name": name, "n_unknown": n_unk, "diff_vs_baseline": diff}
    row.update(extra)
    return row


def signed_sqrt(x):
    """Power norm with alpha=0.5: y = sign(x) * sqrt(|x|)."""
    return np.sign(x) * np.sqrt(np.abs(x))


def main():
    print(f"Loading 2-seed average {SEEDS}...")
    train_emb, train_labels, test_emb, test_cosine = load_avg(SEEDS)

    df_train = pd.read_csv(DATA_DIR / "train.csv")
    df_add = pd.read_csv(DATA_DIR / "additional_train.csv")
    df_add_known = df_add[df_add["writer_id"] != "-1"]
    df_all = pd.concat([df_train, df_add_known], ignore_index=True)
    writers = sorted(df_all["writer_id"].unique())
    idx2writer = {i: w for i, w in enumerate(writers)}
    n_writers = len(writers)
    df_test = pd.read_csv(DATA_DIR / "test.csv")
    n_test = len(df_test)

    baseline_preds = pd.read_csv(BASELINE_CSV)["writer_id"].astype(str).tolist() \
        if BASELINE_CSV.exists() else None
    rows = []

    # ── Sanity: original K=60 keep=7 (should diff=0) ──
    print("\n[0] Sanity: standard K=60 keep=7 T=0.10")
    preds, clu, keep, cw, pp, margin, proto_l2 = run_k60_keep7_gated(
        train_emb, train_labels, test_emb, test_cosine, n_writers, idx2writer, n_test, df_test)
    rows.append(emit("00_sanity_baseline", preds, df_test, n_test, baseline_preds))
    print(f"  diff vs baseline: {rows[-1]['diff_vs_baseline']} (expect 0)")
    print(f"  kept clusters: {sorted(keep)}")

    # ── Experiment 1: post-hoc power-norm on writer_emb ──
    print("\n[1] Post-hoc power-norm (signed-sqrt) on writer_emb")
    test_emb_pn = l2norm(signed_sqrt(test_emb))
    train_emb_pn = l2norm(signed_sqrt(train_emb))
    preds_pn, _, kp1, _, _, _, _ = run_k60_keep7_gated(
        train_emb_pn, train_labels, test_emb_pn, test_cosine,
        n_writers, idx2writer, n_test, df_test)
    rows.append(emit("01_power_norm_alpha050", preds_pn, df_test, n_test, baseline_preds))
    print(f"  diff vs baseline: {rows[-1]['diff_vs_baseline']}, kept: {sorted(kp1)}")

    # Variant: only power-norm test side, keep train cosine
    print("  variant: power-norm test only")
    preds_pn_test, _, kp2, _, _, _, _ = run_k60_keep7_gated(
        train_emb, train_labels, test_emb_pn, test_cosine,
        n_writers, idx2writer, n_test, df_test)
    rows.append(emit("01b_power_norm_test_only", preds_pn_test, df_test, n_test, baseline_preds))
    print(f"  diff vs baseline: {rows[-1]['diff_vs_baseline']}, kept: {sorted(kp2)}")

    # ── Experiment 2: cluster-restricted DBA/AQE smoothing ──
    print("\n[2] Cluster-restricted DBA/AQE smoothing")
    for alpha, top_k in [(0.30, 3), (0.30, 5), (0.50, 3), (0.50, 5), (0.70, 5)]:
        new_emb = test_emb.copy()
        # Precompute cluster membership
        for c in range(K):
            members = np.where(clu == c)[0]
            if len(members) < 2:
                continue
            sub = test_emb[members]  # (n_c, D)
            sim_sub = sub @ sub.T   # (n_c, n_c)
            np.fill_diagonal(sim_sub, -np.inf)
            k_eff = min(top_k, len(members) - 1)
            top_idx = np.argpartition(-sim_sub, k_eff, axis=1)[:, :k_eff]
            for ii, mi in enumerate(members):
                nb = sub[top_idx[ii]].mean(axis=0)
                new_emb[mi] = test_emb[mi] + alpha * nb
        new_emb = l2norm(new_emb)
        preds_dba, _, kp_dba, _, _, _, _ = run_k60_keep7_gated(
            train_emb, train_labels, new_emb, test_cosine,
            n_writers, idx2writer, n_test, df_test)
        tag = f"02_dba_alpha{int(alpha*100):03d}_k{top_k}"
        rows.append(emit(tag, preds_dba, df_test, n_test, baseline_preds,
                         alpha=alpha, top_k=top_k))
        print(f"  alpha={alpha} top_k={top_k}: diff={rows[-1]['diff_vs_baseline']}, kept={sorted(kp_dba)}")

    # ── Experiment 3: Positive-Negative prototype gate ──
    print("\n[3] Positive-Negative prototype gate")
    proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        mask = train_labels == w
        if mask.sum() > 0:
            proto[w] = train_emb[mask].mean(axis=0)
    proto_l2 = l2norm(proto)
    # For each writer, find its confusor = argmax cos(proto[w], proto[other])
    cross_sim = proto_l2 @ proto_l2.T
    np.fill_diagonal(cross_sim, -np.inf)
    confusor = cross_sim.argmax(axis=1)  # confusor[w] = nearest other writer
    print(f"  3 sample confusors: {[(int(w), int(confusor[w])) for w in range(3)]}")

   
    for margin_extra in (0.00, 0.02, 0.04, 0.06, 0.10):
       
        preds_pn_gate = []
        for i in range(n_test):
            cid = int(clu[i])
            if cid not in keep:
                preds_pn_gate.append("-1")
                continue
           
            if margin[i] >= GATING_T:
                w_pred = int(pp[i])
            else:
                w_pred = int(cw[cid])
           
            pos_sim = float(test_emb[i] @ proto_l2[w_pred])
            neg_sim = float(test_emb[i] @ proto_l2[int(confusor[w_pred])])
            if pos_sim - neg_sim < margin_extra:
                preds_pn_gate.append("-1")  
            else:
                preds_pn_gate.append(idx2writer[w_pred])
        tag = f"03_pn_gate_m{int(margin_extra*1000):03d}"
        rows.append(emit(tag, preds_pn_gate, df_test, n_test, baseline_preds,
                         margin_extra=margin_extra))
        print(f"  margin_extra={margin_extra}: diff={rows[-1]['diff_vs_baseline']}, "
              f"n_unk={rows[-1]['n_unknown']}")

    df = pd.DataFrame(rows).sort_values("diff_vs_baseline")
    df.to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\n{'='*72}")
    print(f"Done — {len(rows)} variants in {OUT_DIR.resolve()}")
    print(f"\nSorted by diff from baseline 0.64996 (n_unknown=3858):")
    print(df.to_string(index=False, max_colwidth=80))
    print("\nSubmit priority:")
    print("  - variants with SMALL diff (1-10): they perturb very few predictions,")
    print("    each flip has ~50/50 chance of being correct -> low-risk lottery")
    print("  - variants with MEDIUM diff (10-50): structural change; could go either way")
    print("  - variants with LARGE diff (>100): different cluster decisions; high risk")


if __name__ == "__main__":
    main()
