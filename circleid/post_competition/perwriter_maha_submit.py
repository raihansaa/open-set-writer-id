#!/usr/bin/env python3
"""
Per-writer Mahalanobis with Ledoit-Wolf shrinkage on the EXISTING trunk.

"""

import os
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

DATA_DIR = Path(os.environ.get("CIRCLEID_DATA", Path(__file__).resolve().parent.parent / "icdar-2026-circleid-writer-identification"))
EMB_DIR = Path(os.environ.get("CIRCLEID_EMB", Path(__file__).resolve().parent / "outputs_skeleton"))
OUT_DIR = Path("./submissions_perwriter_maha")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 137]
PERCENTILES = (60, 65, 70, 75)


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


def emit(name, pred_idx, knownness, pct, df_test, idx2writer, n_test):
    thresh = np.percentile(knownness, pct)
    preds = [idx2writer[pred_idx[i]] if knownness[i] >= thresh else "-1"
             for i in range(n_test)]
    n_unk = sum(1 for p in preds if p == "-1")
    sub = pd.DataFrame({"image_id": df_test["image_id"], "writer_id": preds})
    fname = f"{name}_unk{pct}pct.csv"
    sub.to_csv(OUT_DIR / fname, index=False)
    print(f"  {fname}: {n_unk}/{n_test} unknown ({100*n_unk/n_test:.1f}%)")
    return {"name": name, "pct": pct, "n_unknown": n_unk, "threshold": float(thresh)}


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

    print(f"  Writers: {n_writers} | train_emb: {train_emb.shape} | test: {n_test}")

    rows = []

    # ── Baseline reproduction ──
    print("\n[0] max-cosine (baseline 0.521 sanity)")
    pred0 = test_cosine.argmax(axis=1)
    known0 = test_cosine.max(axis=1)
    for pct in PERCENTILES:
        rows.append(emit("baseline", pred0, known0, pct,
                         df_test, idx2writer, n_test))

    # ── Per-writer Maha in full 512-d ──
    print("\n[1] Per-writer Mahalanobis (Ledoit-Wolf, full 512-d)")
    D = train_emb.shape[1]
    proto_means = np.zeros((n_writers, D), dtype=np.float32)
    cov_invs = np.zeros((n_writers, D, D), dtype=np.float32)
    for w in range(n_writers):
        mask = train_labels == w
        emb_w = train_emb[mask]
        proto_means[w] = emb_w.mean(axis=0)
        if len(emb_w) < 2:
            cov_invs[w] = np.eye(D, dtype=np.float32)
            continue
        lw = LedoitWolf().fit(emb_w)
        try:
            cov_invs[w] = np.linalg.inv(lw.covariance_).astype(np.float32)
        except np.linalg.LinAlgError:
            cov_invs[w] = np.linalg.pinv(lw.covariance_).astype(np.float32)
    maha = np.full((n_test, n_writers), np.inf, dtype=np.float32)
    for w in range(n_writers):
        diff = test_emb - proto_means[w]
        maha[:, w] = np.sqrt(np.maximum(np.sum(diff @ cov_invs[w] * diff, axis=1), 0))
    pred1 = maha.argmin(axis=1)
    known1 = -maha.min(axis=1)
    print(f"  Maha range: [{maha.min():.2f}, {maha.max():.2f}]")
    for pct in PERCENTILES:
        rows.append(emit("perwriter_maha", pred1, known1, pct,
                         df_test, idx2writer, n_test))

    # ── Per-writer Maha after PCA-64 (combines both ideas) ──
    print("\n[2] PCA-64 + per-writer Maha")
    pca = PCA(n_components=64, random_state=42).fit(train_emb)
    train_pca = pca.transform(train_emb).astype(np.float32)
    test_pca = pca.transform(test_emb).astype(np.float32)
    proto_pca = np.zeros((n_writers, 64), dtype=np.float32)
    cov_pca = np.zeros((n_writers, 64, 64), dtype=np.float32)
    for w in range(n_writers):
        emb_w = train_pca[train_labels == w]
        proto_pca[w] = emb_w.mean(axis=0)
        lw = LedoitWolf().fit(emb_w)
        try:
            cov_pca[w] = np.linalg.inv(lw.covariance_).astype(np.float32)
        except np.linalg.LinAlgError:
            cov_pca[w] = np.linalg.pinv(lw.covariance_).astype(np.float32)
    maha2 = np.full((n_test, n_writers), np.inf, dtype=np.float32)
    for w in range(n_writers):
        diff = test_pca - proto_pca[w]
        maha2[:, w] = np.sqrt(np.maximum(np.sum(diff @ cov_pca[w] * diff, axis=1), 0))
    pred2 = maha2.argmin(axis=1)
    known2 = -maha2.min(axis=1)
    print(f"  PCA-64 + perwriter Maha range: [{maha2.min():.2f}, {maha2.max():.2f}]")
    for pct in PERCENTILES:
        rows.append(emit("pca64_perwriter_maha", pred2, known2, pct,
                         df_test, idx2writer, n_test))

    # ── Blend max-cos + per-writer Maha at 50/50 ──
    print("\n[3] Blend max-cos + per-writer Maha (z-norm 50/50)")
    def zscore(x):
        return (x - x.mean()) / (x.std() + 1e-12)
    blend = 0.5 * zscore(known0) + 0.5 * zscore(known1)
    pred3 = pred0   # keep max-cos argmax (which gave 0.521 alone)
    for pct in PERCENTILES:
        rows.append(emit("blend_maxcos_perwriter", pred3, blend, pct,
                         df_test, idx2writer, n_test))

    pd.DataFrame(rows).to_csv(OUT_DIR / "_summary.csv", index=False)
    print(f"\nDone. Submissions in {OUT_DIR.resolve()}")
    print("\nSubmit in priority order:")
    print("  1. baseline_unk70pct.csv         (sanity, should reproduce 0.521)")
    print("  2. perwriter_maha_unk70pct.csv   (the Codex fix, full 512-d)")
    print("  3. pca64_perwriter_maha_unk70pct.csv  (PCA + per-writer combined)")
    print("  4. blend_maxcos_perwriter_unk65pct.csv  (50/50 blend, only if 2 or 3 helps)")


if __name__ == "__main__":
    main()
