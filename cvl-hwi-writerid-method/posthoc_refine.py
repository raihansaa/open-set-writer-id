"""
Consumes the embeddings.npz produced by train_writerid.py (which now also saves
per-patch raw embeddings) and produces a refined npz with:
    * Page aggregation:   GMP (Generalized Max Pooling) instead of plain mean
    * Score refinement:   k-reciprocal re-ranking (Zhong CVPR 2017)


Usage:
    python posthoc_refine.py --emb runs/cvl/embeddings_seed42.npz
    python posthoc_refine.py --emb runs/cvl/embeddings_seed42.npz \
                              --aggregate gmp --rerank \
                              --out runs/cvl/embeddings_seed42_refined.npz

    # Then re-run inference/evaluation on the refined embeddings:
    python submit_writerid.py --emb runs/cvl/embeddings_seed42_refined.npz \
                              --out-dir runs/cvl/submissions_refined
    python eval_metrics.py    --emb runs/cvl/embeddings_seed42_refined.npz \
                              --sub-dir runs/cvl/submissions_refined
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize


# ──────────────────────────────────────────────────────────────────────
# Aggregation: GMP (Generalized Max Pooling)
# ──────────────────────────────────────────────────────────────────────
def gmp_page(patches: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Generalized Max Pooling for one page.

    patches : (N, D)
    lam     : ridge regularizer (Murray & Perronnin 2014 use λ in 1.0–100.0)
    returns : (D,) L2-normalized
    """
    N, D = patches.shape
    G = patches.T @ patches + lam * np.eye(D, dtype=patches.dtype)
    rhs = patches.T @ np.ones(N, dtype=patches.dtype)
    z = np.linalg.solve(G, rhs)
    return z / (np.linalg.norm(z) + 1e-9)


def quality_weighted_page(patches: np.ndarray, quality: np.ndarray,
                          eps: float = 0.05, power: float = 1.0) -> np.ndarray:
    """Quality-weighted mean: down-weight low-content patches.

    patches : (N, D)
    quality : (N,) per-patch quality in [0, 1] (e.g. ink density)
    eps     : floor on per-patch weight to avoid div-by-zero
    power   : sharpening exponent on quality (>1 = more aggressive down-weighting)
    returns : (D,) L2-normalized
    """
    w = np.maximum(quality, eps).astype(np.float32) ** power
    w = w / (w.sum() + 1e-9)
    out = (patches * w[:, None]).sum(axis=0)
    return out / (np.linalg.norm(out) + 1e-9)


def aggregate_all(raw_patches: np.ndarray, method: str, lam: float = 1.0,
                  quality: np.ndarray = None, q_power: float = 1.0) -> np.ndarray:
    """raw_patches: (P, N, D). Returns (P, D), L2-normalized.

    For method='quality_weighted', `quality` must be (P, N).
    """
    if method == "mean":
        out = raw_patches.mean(axis=1)
    elif method == "gmp":
        out = np.stack([gmp_page(raw_patches[p], lam) for p in range(len(raw_patches))])
    elif method == "quality_weighted":
        if quality is None:
            raise ValueError("quality_weighted needs the `quality` (P,N) array")
        out = np.stack([
            quality_weighted_page(raw_patches[p], quality[p], power=q_power)
            for p in range(len(raw_patches))
        ])
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    return normalize(out).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# k-reciprocal re-ranking (Zhong CVPR 2017)
# ──────────────────────────────────────────────────────────────────────
def k_reciprocal_rerank(q_emb: np.ndarray, g_emb: np.ndarray,
                        k1: int = 20, k2: int = 6,
                        lam: float = 0.3) -> np.ndarray:
    """Joint-set k-reciprocal re-ranking.

    q_emb : (Nq, D)    query  (test) embeddings
    g_emb : (Ng, D)    gallery (train, or train-writer-prototypes) embeddings
    Returns refined distance matrix (Nq, Ng), lower = closer.
    Defaults from Zhong CVPR 2017.
    """
    q = normalize(q_emb)
    g = normalize(g_emb)
    nq, ng = len(q), len(g)
    n_all = nq + ng
    all_e = np.concatenate([q, g], axis=0).astype(np.float32)

    # Original cosine distance ∈ [0, 2]
    orig = 1.0 - all_e @ all_e.T                          # (n_all, n_all)
    init_rank = np.argsort(orig, axis=1)                  # (n_all, n_all)

    # ── Build k-reciprocal expanded neighbor sets per row ──
    V = np.zeros((n_all, n_all), dtype=np.float32)
    half_k1 = max(1, int(round(k1 / 2)))
    for i in range(n_all):
        forward = init_rank[i, : k1 + 1]
        backward_ok = []
        for j in forward:
            if i in init_rank[j, : k1 + 1]:
                backward_ok.append(j)
        recip = set(backward_ok)
        # Expand with k1/2 reciprocals of each reciprocal
        for j in list(recip):
            fj = init_rank[j, : half_k1 + 1]
            for kk in fj:
                if j in init_rank[kk, : half_k1 + 1]:
                    recip.add(kk)
        recip = np.array(sorted(recip), dtype=np.int64)
        V[i, recip] = np.exp(-orig[i, recip])

    # Local query expansion: average V over top-k2 neighbors
    V_avg = np.zeros_like(V)
    for i in range(n_all):
        V_avg[i] = V[init_rank[i, :k2]].mean(axis=0)
    V = V_avg

    # Jaccard distance between query rows and gallery rows
    jaccard = np.zeros((nq, ng), dtype=np.float32)
    for i in range(nq):
        vi = V[i]
        # vectorized over gallery
        inter = np.minimum(vi[None, :], V[nq:]).sum(axis=1)
        union = np.maximum(vi[None, :], V[nq:]).sum(axis=1)
        jaccard[i] = 1.0 - inter / (union + 1e-9)

    # Final distance: blend with original cosine distance
    final = (1.0 - lam) * orig[:nq, nq:] + lam * jaccard
    return final.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Writer prototypes (mean of writer's train pages, L2-normalized)
# ──────────────────────────────────────────────────────────────────────
def build_writer_prototypes(train_emb: np.ndarray, train_labels: np.ndarray,
                            n_writers: int) -> np.ndarray:
    """One prototype per writer = L2(mean of their train page embeddings)."""
    proto = np.zeros((n_writers, train_emb.shape[1]), dtype=np.float32)
    for c in range(n_writers):
        mask = train_labels == c
        if mask.sum() > 0:
            proto[c] = train_emb[mask].mean(axis=0)
    return normalize(proto).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Main refinement pipeline
# ──────────────────────────────────────────────────────────────────────
def refine(emb_npz: dict, method: str, lam_gmp: float,
           do_rerank: bool, k1: int, k2: int, lam_rerank: float,
           q_power: float = 1.0, whiten: bool = False,
           whiten_dim: int = 256) -> dict:
    """Returns a dict ready to np.savez() into a refined.npz."""
    has_raw = all(k in emb_npz.files
                  for k in ("train_raw_patches", "val_raw_patches", "test_raw_patches"))
    has_quality = all(k in emb_npz.files
                      for k in ("train_quality", "val_quality", "test_quality"))

    if method == "gmp" and not has_raw:
        raise ValueError(
            "GMP aggregation requires raw_patches in the npz, which the older "
            "train_writerid.py did not save. Retrain with the updated script."
        )
    if method == "quality_weighted" and not (has_raw and has_quality):
        raise ValueError(
            "quality_weighted aggregation requires raw_patches AND quality "
            "arrays in the npz. Retrain with the updated script (saves "
            "train_quality / val_quality / test_quality)."
        )

    train_labels = emb_npz["train_labels"]
    writers = emb_npz["writers"]
    n_writers = len(writers)

    # ── 1. Re-aggregate patches into page embeddings ──────────────────
    if method == "mean" or not has_raw:
        train_emb = emb_npz["train_emb"].astype(np.float32)
        val_emb = emb_npz["val_emb"].astype(np.float32)
        test_emb = emb_npz["test_emb"].astype(np.float32)
        print(f"  aggregation: {method} (using pre-saved mean page emb)")
    elif method == "quality_weighted":
        print(f"  aggregation: quality_weighted (power={q_power})")
        train_emb = aggregate_all(emb_npz["train_raw_patches"], method,
                                  quality=emb_npz["train_quality"], q_power=q_power)
        val_emb = aggregate_all(emb_npz["val_raw_patches"], method,
                                quality=emb_npz["val_quality"], q_power=q_power)
        test_emb = aggregate_all(emb_npz["test_raw_patches"], method,
                                 quality=emb_npz["test_quality"], q_power=q_power)
    else:
        print(f"  aggregation: {method}  (λ={lam_gmp})")
        train_emb = aggregate_all(emb_npz["train_raw_patches"], method, lam_gmp)
        val_emb = aggregate_all(emb_npz["val_raw_patches"], method, lam_gmp)
        test_emb = aggregate_all(emb_npz["test_raw_patches"], method, lam_gmp)

    # ── 1b. (Optional) PCA-whitening — decorrelate page embeddings ────
    if whiten:
        from sklearn.decomposition import PCA
        n_comp = min(whiten_dim, train_emb.shape[0] - 1, train_emb.shape[1])
        pca = PCA(n_components=n_comp, whiten=True, random_state=0).fit(train_emb)
        train_emb = normalize(pca.transform(train_emb)).astype(np.float32)
        val_emb = normalize(pca.transform(val_emb)).astype(np.float32)
        test_emb = normalize(pca.transform(test_emb)).astype(np.float32)
        print(f"  PCA-whitening ON (dim={n_comp}, fit on train pages)")

    # ── 2. Rebuild writer prototypes + cosine matrices ────────────────
    protos = build_writer_prototypes(train_emb, train_labels, n_writers)
    val_cosine = normalize(val_emb) @ protos.T
    test_cosine = normalize(test_emb) @ protos.T
    print(f"  rebuilt writer prototypes (n_writers={n_writers}, D={train_emb.shape[1]})")

    # ── 3. (Optional) k-reciprocal re-ranking — replace test cosine ───
    if do_rerank:
        print(f"  k-reciprocal re-ranking (k1={k1}, k2={k2}, λ={lam_rerank})")
        dist_qg = k_reciprocal_rerank(test_emb, train_emb, k1, k2, lam_rerank)
        # Convert refined distances into per-class scores: nearest-train-page-per-class
        refined_cos = np.zeros((len(test_emb), n_writers), dtype=np.float32)
        for c in range(n_writers):
            mask = train_labels == c
            if mask.sum() == 0:
                refined_cos[:, c] = -np.inf
                continue
            # use the minimum refined distance to any train page of writer c
            refined_cos[:, c] = -dist_qg[:, mask].min(axis=1)  # higher = closer
        test_cosine = refined_cos

    out = {
        "train_emb": train_emb,
        "train_labels": train_labels,
        "val_emb": val_emb,
        "val_writer_id": emb_npz["val_writer_id"],
        "val_image_id": emb_npz["val_image_id"],
        "val_cosine": val_cosine.astype(np.float32),
        "test_emb": test_emb,
        "test_writer_id": emb_npz["test_writer_id"],
        "test_image_id": emb_npz["test_image_id"],
        "test_cosine": test_cosine.astype(np.float32),
        "writers": writers,
    }
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, required=True,
                    help="Path to embeddings_seed*.npz from train_writerid.py")
    ap.add_argument("--out", type=str, default=None,
                    help="Output path. Default: alongside input with _refined suffix")
    ap.add_argument("--aggregate", type=str, default="gmp",
                    choices=["mean", "gmp", "quality_weighted"],
                    help="Patch -> page aggregation (default: gmp). "
                         "quality_weighted weights each patch by ink density — "
                         "down-weights degraded/blank patches on historical pages.")
    ap.add_argument("--gmp-lambda", type=float, default=1.0,
                    help="GMP ridge regularizer (default: 1.0)")
    ap.add_argument("--quality-power", type=float, default=1.0,
                    help="Sharpening exponent on quality weights for "
                         "quality_weighted aggregation (default 1.0; try 1.5-2.0 "
                         "for more aggressive down-weighting of low-content patches).")
    ap.add_argument("--rerank", action="store_true",
                    help="Apply k-reciprocal re-ranking on test cosine")
    ap.add_argument("--rerank-k1", type=int, default=20)
    ap.add_argument("--rerank-k2", type=int, default=6)
    ap.add_argument("--rerank-lambda", type=float, default=0.3,
                    help="Blend factor: 0 = pure cosine, 1 = pure Jaccard")
    ap.add_argument("--whiten", action="store_true",
                    help="PCA-whiten page embeddings (fit on train) before "
                         "rebuilding prototypes/cosines — decorrelates VLAD "
                         "descriptors. Default OFF (backward-compatible).")
    ap.add_argument("--whiten-dim", type=int, default=256,
                    help="PCA-whitening target dimension (default 256)")
    return ap.parse_args()


def main():
    args = parse_args()
    emb = np.load(args.emb, allow_pickle=True)
    print(f"Loaded: {args.emb}")
    print(f"  keys: {sorted(emb.files)}")

    refined = refine(emb,
                     method=args.aggregate, lam_gmp=args.gmp_lambda,
                     do_rerank=args.rerank,
                     k1=args.rerank_k1, k2=args.rerank_k2,
                     lam_rerank=args.rerank_lambda,
                     q_power=args.quality_power,
                     whiten=args.whiten, whiten_dim=args.whiten_dim)

    out_path = Path(args.out) if args.out else Path(args.emb).with_name(
        Path(args.emb).stem + "_refined.npz"
    )
    np.savez(out_path, **refined)
    print(f"\nSaved refined embeddings -> {out_path}")
    print(f"  test_cosine shape: {refined['test_cosine'].shape}")


if __name__ == "__main__":
    main()
