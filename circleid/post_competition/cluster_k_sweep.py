#!/usr/bin/env python3
"""
ADVERSARIAL REVIEW INSIGHT: K=44 may be WRONG.
- Test set has ~16 unknown writers + 44 known
- With K=44, the 16 unknown writers' samples get FORCED into known-writer clusters
- This contaminates known clusters, hurting per-sample writer assignment AND rejection
- K=60-70 should give unknown writers their OWN clusters → cleaner separation

We test K in {44, 50, 55, 60, 65, 70, 80, 90} with appropriately-scaled n_keep
so that the kept clusters cover ~the same number of test samples as our
winning K=44 keep=6 config (~800 samples).

Plus gating sweep T in {0.075, 0.100, 0.125} on each.
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
OUT_DIR = _THIS_DIR / "submissions_k_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
# K sweep — far beyond 44 to find optimum
K_VALUES = (44, 50, 55, 60, 65, 70, 80, 90)
N_KEEP_VALUES = (5, 6, 7, 8, 9, 10, 12)
GATING_T = 0.100   # Use the optimal gating threshold


def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def load_avg(seeds):
    bundles = [np.load(EMB_DIR / f"embeddings_seed_{s}.npz") for s in seeds]
    train_emb = np.mean([l2norm(b["train_emb"]) for b in bundles], axis=0)
    test_emb = np.mean([l2norm(b["test_emb"]) for b in bundles], axis=0)
    train_emb = l2norm(train_emb)
    test_emb = l2norm(test_emb)
    train_labels = bundles[0]["train_labels"]
    test_cosine = np.mean([b["test_cosine"] for b in bundles], axis=0)
    return train_emb, train_labels, test_emb, test_cosine


def emit(name, preds_str_list, df_test, n_test, **extra):
    n_unk = sum(1 for p in preds_str_list if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds_str_list})
    fname = f"{name}.csv"
    sub.to_csv(OUT_DIR / fname, index=False)
    row = {"name": name, "n_unknown": n_unk, "frac_unknown": n_unk / n_test}
    row.update(extra)
    return row


def main():
    print(f"Loading + averaging seeds {SEEDS}...")
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

    print(f"  Writers: {n_writers} | test: {n_test}")
    pred_per_sample = test_cosine.argmax(axis=1)
    sorted_cos = np.sort(test_cosine, axis=1)
    margin_per_sample = sorted_cos[:, -1] - sorted_cos[:, -2]

    proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for w in range(n_writers):
        proto[w] = train_emb[train_labels == w].mean(axis=0)
    proto_l2 = l2norm(proto)

    rows = []

    for K in K_VALUES:
        print(f"\n[K={K}] Clustering...")
        clu = AgglomerativeClustering(
            n_clusters=K, metric="cosine", linkage="average"
        ).fit_predict(test_emb)
        centroids = l2norm(np.stack([test_emb[clu == c].mean(axis=0) for c in range(K)]))
        sim = centroids @ proto_l2.T
        clu_knownness = sim.max(axis=1)
        clu_writer = sim.argmax(axis=1)
        sort_order = np.argsort(-clu_knownness)

        # Diagnostic: cluster sizes
        sizes = np.bincount(clu, minlength=K)
        print(f"  sizes: min={sizes.min()} med={int(np.median(sizes))} max={sizes.max()}")

        for n_keep in N_KEEP_VALUES:
            if n_keep > K:
                continue
            keep_set = set(sort_order[:n_keep].tolist())

            # Variant A: pure cluster-level
            preds_clust = [idx2writer[int(clu_writer[clu[i]])] if int(clu[i]) in keep_set else "-1"
                           for i in range(n_test)]
            tag = f"K{K:03d}_keep{n_keep:02d}_cluster"
            rows.append(emit(tag, preds_clust, df_test, n_test, K=K, n_keep=n_keep))

            # Variant B: cluster + gating at T=0.10
            preds_gated = []
            for i in range(n_test):
                if int(clu[i]) not in keep_set:
                    preds_gated.append("-1")
                    continue
                if margin_per_sample[i] >= GATING_T:
                    preds_gated.append(idx2writer[pred_per_sample[i]])
                else:
                    preds_gated.append(idx2writer[int(clu_writer[clu[i]])])
            tag_g = f"K{K:03d}_keep{n_keep:02d}_gated"
            rows.append(emit(tag_g, preds_gated, df_test, n_test, K=K, n_keep=n_keep, T=GATING_T))

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. {len(rows)} submissions in {OUT_DIR.resolve()}")
    print("\nReference: K=44 keep=6 gated = 0.63220 baseline.")
    print("\nLikely best candidates to submit (test the K hypothesis):")
    print("  1. K044_keep06_gated.csv  (sanity, must be ~0.632)")
    print("  2. K060_keep08_gated.csv  (60 clusters, keep ~14%)")
    print("  3. K065_keep09_gated.csv  (65 clusters, keep ~14%)")
    print("  4. K070_keep10_gated.csv  (70 clusters, keep ~14%)")
    print("  5. K080_keep12_gated.csv  (80 clusters)")
    print("\nAlternatives:")
    print("  K050_keep07_gated.csv  (slightly more than 44)")
    print("  K055_keep07_gated.csv")


if __name__ == "__main__":
    main()
