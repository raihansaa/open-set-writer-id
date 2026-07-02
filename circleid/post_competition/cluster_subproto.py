#!/usr/bin/env python3
"""
Sub-prototype and confidence-filtered prototype refinement.

Approaches:
  A. Confidence-filtered mean
     For each writer w: keep top-P percentile of patches by train_cosine[i, w]
     then average. Sweep P in {0, 25, 50, 75}.

  B. Sub-center KMeans (k sub-prototypes per writer)
     For each writer w: KMeans on its patches into k sub-clusters.
     Resulting 44*k sub-prototypes; each cluster centroid argmaxes over ALL
     sub-prototypes. Sweep k in {2, 3, 4, 5}.

  C. Combined: confidence-filter then KMeans on filtered patches.
     Sweep (P, k) cross-product.

"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = _THIS_DIR / "submissions_subproto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
K_CLUSTER = 60
N_KEEP = 7
GATING_T = 0.10

CONF_PERCENTILES = (0, 25, 50, 75)        # P=0 means no filtering
SUB_K_VALUES = (1, 2, 3, 4, 5)            # k=1 means just mean (baseline)
KMEANS_INIT = 5
KMEANS_SEED = 0


def l2norm(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def load_avg(seeds):
    bundles = [np.load(EMB_DIR / f"embeddings_seed_{s}.npz") for s in seeds]
    # Per-patch train data is independent per seed; we keep both seeds'
    # patches concatenated for richer per-writer pools.
    train_emb_list = [b["train_emb"] for b in bundles]      # each (34650, 512)
    train_labels_list = [b["train_labels"] for b in bundles]
    train_cosine_list = [b["train_cosine"] for b in bundles]  # (34650, 44)
    # Sanity: labels should match across seeds
    assert np.array_equal(train_labels_list[0], train_labels_list[1]), \
        "train_labels differ across seeds!"
    train_labels = train_labels_list[0]
    # Average the per-patch embeddings across seeds (same patch indexing)
    train_emb = np.mean([l2norm(e) for e in train_emb_list], axis=0)
    train_emb = l2norm(train_emb)
    train_cosine = np.mean(train_cosine_list, axis=0)
    # Test side
    test_emb = np.mean([l2norm(b["test_emb"]) for b in bundles], axis=0)
    test_emb = l2norm(test_emb)
    test_cosine = np.mean([b["test_cosine"] for b in bundles], axis=0)
    return train_emb, train_labels, train_cosine, test_emb, test_cosine


def emit(name, preds_str_list, df_test, n_test, **extra):
    n_unk = sum(1 for p in preds_str_list if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds_str_list})
    sub.to_csv(OUT_DIR / f"{name}.csv", index=False)
    row = {"name": name, "n_unknown": n_unk, "frac_unknown": n_unk / n_test}
    row.update(extra)
    return row


def build_protos(train_emb, train_labels, train_cosine, n_writers,
                 conf_percentile: int, sub_k: int):
    
    proto_rows = []
    proto_owner = []
    for w in range(n_writers):
        mask = train_labels == w
        patches = train_emb[mask]                # ~800 x 512
        if conf_percentile > 0:
            conf = train_cosine[mask, w]         # ~800
            threshold = np.percentile(conf, conf_percentile)
            keep_mask = conf >= threshold
            if keep_mask.sum() < sub_k:
                # Fallback: keep all if filter too aggressive for k
                keep_mask = np.ones_like(conf, dtype=bool)
            patches = patches[keep_mask]
        patches = l2norm(patches)
        if sub_k == 1:
            sub_centers = patches.mean(axis=0, keepdims=True)
        else:
            km = KMeans(n_clusters=sub_k, n_init=KMEANS_INIT,
                        random_state=KMEANS_SEED).fit(patches)
            sub_centers = km.cluster_centers_
        for sc in sub_centers:
            proto_rows.append(sc)
            proto_owner.append(w)
    proto_mat = l2norm(np.stack(proto_rows).astype(np.float32))
    proto_owner = np.array(proto_owner, dtype=np.int64)
    return proto_mat, proto_owner


def main():
    print(f"Loading + averaging seeds {SEEDS}...")
    train_emb, train_labels, train_cosine, test_emb, test_cosine = load_avg(SEEDS)
    print(f"  train_emb: {train_emb.shape}  (per-patch)")
    print(f"  test_emb:  {test_emb.shape}  (per-image)")

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

    # ----- Compute BASELINE writer assignments (mean proto, k=1, P=0) -----
    print("Computing baseline (P=0, k=1) proto...")
    base_proto, base_owner = build_protos(train_emb, train_labels, train_cosine,
                                          n_writers, conf_percentile=0, sub_k=1)
    base_sim = centroids @ base_proto.T
    base_winning_proto = base_sim.argmax(axis=1)
    baseline_writer = base_owner[base_winning_proto]          # per-cluster writer
    baseline_known = base_sim.max(axis=1)                     # per-cluster knownness
    baseline_sort = np.argsort(-baseline_known)
    baseline_keep = set(baseline_sort[:N_KEEP].tolist())
    print(f"  baseline keep_set = {sorted(baseline_keep)}")

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
    rows.append(emit("baseline_singleT_010_subproto_sanity",
                     build_submission(baseline_writer, baseline_keep),
                     df_test, n_test, P=0, k=1, flip_writer=0, flip_keep=0))
    print("  baseline sanity submission emitted (should reproduce 0.64996)")

    # ----- Sweep combinations -----
    print(f"\nSweep: conf_percentile in {CONF_PERCENTILES}, sub_k in {SUB_K_VALUES}")
    print(f"  {'tag':<28s} {'P':>3s} {'k':>2s} {'flipW':>5s} {'flipK':>5s} {'n_unk':>6s}")
    for P in CONF_PERCENTILES:
        for k in SUB_K_VALUES:
            if P == 0 and k == 1:
                continue  # already covered as baseline
            try:
                proto_mat, proto_owner = build_protos(
                    train_emb, train_labels, train_cosine,
                    n_writers, conf_percentile=P, sub_k=k)
            except Exception as e:
                print(f"  P={P} k={k} failed: {e}")
                continue
            sim = centroids @ proto_mat.T
            winning_proto = sim.argmax(axis=1)
            clu_writer_new = proto_owner[winning_proto]
            clu_known_new = sim.max(axis=1)
            sort_new = np.argsort(-clu_known_new)
            keep_new = set(sort_new[:N_KEEP].tolist())

            n_flip_writer = int((clu_writer_new != baseline_writer).sum())
            n_flip_keep = len(keep_new ^ baseline_keep) // 2

            tag = f"subproto_P{P:02d}_k{k:02d}"
            preds = build_submission(clu_writer_new, keep_new)
            row = emit(tag, preds, df_test, n_test,
                       P=P, k=k, flip_writer=n_flip_writer, flip_keep=n_flip_keep)
            rows.append(row)
            print(f"  {tag:<28s} {P:>3d} {k:>2d} {n_flip_writer:>5d} "
                  f"{n_flip_keep:>5d} {row['n_unknown']:>6d}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. {len(rows)} submissions in {OUT_DIR.resolve()}")
    print("\nDIAGNOSTIC GUIDE:")
    print("  flipW = how many of the 60 cluster -> writer assignments differ from baseline")
    print("  flipK = how many of the kept-7 clusters differ from baseline")
    print()
    print("  Recall from k-reciprocal disaster: each flipK costs ~-0.025.")
    print("  So variants with flipK == 0 are the SAFE ones.")
    print("  Variants with flipK >= 2 risk catastrophic regression.")
    print()
    print("Submit priority (start with safest):")
    print("  1. baseline_singleT_010_subproto_sanity.csv  -- must = 0.64996")
    print("  2. Any subproto_*.csv with flipK==0 AND flipW in [1, 8]")
    print("     -- these refine writer assignments WITHOUT disturbing kept clusters")
    print("  3. subproto_*.csv with flipK==0 AND flipW==0 -- no-op confirmations")
    print("  4. AVOID submissions with flipK >= 2 unless prior submissions exhausted")


if __name__ == "__main__":
    main()
