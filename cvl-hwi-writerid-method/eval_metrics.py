"""
eval_metrics.py — full paper-grade metric suite for open-set writer ID.

Use ONLY on datasets where you have ground-truth test labels (CVL, HWI). 

Metrics computed:
    Classification (open-set)
        OS-Top-1            : correct (known OR -1)         | Kaggle's metric
        Known-only Top-1    : correct writer | given true is known
        Unknown rejection   : correct -1     | given true is unknown
        Balanced accuracy   : mean of per-class recall (incl. -1 class)
        Writer-macro Top-1  : Top-1 averaged equally across writers (not pages)

    OOD quality (threshold-free, uses continuous OOD score)
        AUROC               : separability of known vs unknown samples
        FPR@95TPR           : false-positive rate (unknown accepted as known)
                              when TPR (known accepted) is 95%

Usage:

    # CLI (single submission)
    python eval_metrics.py --emb runs/cvl/embeddings_seed42.npz \
                           --pred runs/cvl/submissions/pca64_unk70.csv

    # CLI (every submission in a folder)
    python eval_metrics.py --emb runs/cvl/embeddings_seed42.npz \
                           --sub-dir runs/cvl/submissions

    # CLI (multi-seed: auto mean ± std across seeds — parallel lists)
    python eval_metrics.py \
        --emb     runs/cvl_seed42/embeddings_seed42.npz \
                  runs/cvl_seed137/embeddings_seed137.npz \
                  runs/cvl_seed7/embeddings_seed7.npz \
        --sub-dir runs/cvl_seed42/submissions_baseline \
                  runs/cvl_seed137/submissions_baseline \
                  runs/cvl_seed7/submissions_baseline
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


_UNK_RE = re.compile(r"_unk\d+$")


def _method_family(stem: str) -> str:
    return _UNK_RE.sub("", stem)


def _unk_pct(stem: str):
    m = _UNK_RE.search(stem)
    return int(m.group(0)[4:]) if m else None


# ──────────────────────────────────────────────────────────────────────
# Metric implementations
# ──────────────────────────────────────────────────────────────────────
def os_top1(test_true: np.ndarray, test_pred: np.ndarray) -> float:
    return float((test_true == test_pred).mean())


def known_only_top1(test_true: np.ndarray, test_pred: np.ndarray) -> float:
    """Top-1 accuracy on rows where true label is NOT '-1'."""
    mask = test_true != "-1"
    if mask.sum() == 0:
        return float("nan")
    return float((test_true[mask] == test_pred[mask]).mean())


def unknown_rejection_rate(test_true: np.ndarray, test_pred: np.ndarray) -> float:
    """Fraction of true-unknown rows that got predicted '-1'."""
    mask = test_true == "-1"
    if mask.sum() == 0:
        return float("nan")
    return float((test_pred[mask] == "-1").mean())


def balanced_acc(test_true: np.ndarray, test_pred: np.ndarray) -> float:
    """Mean of per-class recall, including '-1' as a class."""
    return float(balanced_accuracy_score(test_true, test_pred))


def writer_macro_top1(test_true: np.ndarray, test_pred: np.ndarray) -> float:
    """Per-writer Top-1 accuracy, then unweighted mean across writers (incl. -1)."""
    accs = []
    for w in np.unique(test_true):
        mask = test_true == w
        if mask.sum() == 0:
            continue
        accs.append((test_pred[mask] == w).mean())
    return float(np.mean(accs)) if accs else float("nan")


def ood_auroc_fpr95(ood_score: np.ndarray, test_writer_id: np.ndarray):
    
    is_unknown = (test_writer_id == "-1").astype(np.int32)
    if is_unknown.sum() == 0 or is_unknown.sum() == len(is_unknown):
        return float("nan"), float("nan")
    auroc = float(roc_auc_score(is_unknown, ood_score))

    # FPR@95TPR — true positive = unknown correctly flagged
    order = np.argsort(-ood_score)
    sorted_unk = is_unknown[order]
    n_unk = is_unknown.sum()
    n_known = len(is_unknown) - n_unk
    # cumulative TPR (unknown) and FPR (known) at every threshold
    cum_unk = np.cumsum(sorted_unk) / n_unk
    cum_known = np.cumsum(1 - sorted_unk) / n_known
    idx = np.searchsorted(cum_unk, 0.95)
    fpr95 = float(cum_known[idx]) if idx < len(cum_known) else float("nan")
    return auroc, fpr95


def oscr_curve(score: np.ndarray, is_known_correct: np.ndarray,
               is_unknown: np.ndarray):
  
    n_known = int((~is_unknown).sum())
    n_unk = int(is_unknown.sum())
    if n_known == 0 or n_unk == 0:
        return None, None
    order = np.argsort(-score, kind="mergesort")          # accept highest scores first
    dir_ = np.concatenate([[0.0], np.cumsum(is_known_correct[order]) / n_known])
    fpr = np.concatenate([[0.0], np.cumsum(is_unknown[order]) / n_unk])
    return fpr, dir_


def _writer_clusters(test_true: np.ndarray):
    known: dict = {}
    for i, w in enumerate(test_true):
        if w != "-1":
            known.setdefault(w, []).append(i)
    known_groups = [np.asarray(v) for v in known.values()]
    unk_idx = np.where(test_true == "-1")[0]
    return known_groups, unk_idx


def cluster_bootstrap_ci(metric_fn, test_true: np.ndarray,
                         n_boot: int = 1000, seed: int = 0):
    """95% percentile CI. Resamples KNOWN writers (clustered) + UNKNOWN queries with
    replacement; metric_fn(idx) recomputes the metric on the resampled subset."""
    known_groups, unk_idx = _writer_clusters(test_true)
    if not known_groups or len(unk_idx) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_kw, n_u = len(known_groups), len(unk_idx)
    vals = []
    for _ in range(n_boot):
        k = np.concatenate([known_groups[j] for j in rng.integers(0, n_kw, n_kw)])
        u = unk_idx[rng.integers(0, n_u, n_u)]
        v = metric_fn(np.concatenate([k, u]))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _safe_auroc(is_unk: np.ndarray, ood: np.ndarray) -> float:
    s = int(is_unk.sum())
    if s == 0 or s == len(is_unk):
        return float("nan")
    return float(roc_auc_score(is_unk, ood))


def oscr_metrics(test_cosine: np.ndarray, writers: np.ndarray,
                 test_true: np.ndarray, n_boot: int = 0,
                 known_score: np.ndarray = None) -> dict:
    
    score = test_cosine.max(axis=1) if known_score is None else known_score
    pred = writers[test_cosine.argmax(axis=1)]            # argmax = identification half
    is_unknown = (test_true == "-1")
    is_known_correct = (~is_unknown) & (pred == test_true)
    fpr, dir_ = oscr_curve(score, is_known_correct, is_unknown)
    if fpr is None:
        return {"OSCR_AUC": float("nan"), "DIR_FAR1": float("nan"),
                "DIR_FAR5": float("nan")}
    out = {
        "OSCR_AUC": float(_trapz(dir_, fpr)),                 # integral of DIR over FPR
        "DIR_FAR1": float(np.interp(0.01, fpr, dir_)),    # OSCR y at FPR=1%
        "DIR_FAR5": float(np.interp(0.05, fpr, dir_)),    # OSCR y at FPR=5%
    }
    if n_boot > 0:
        ood = 1.0 - score

        def _oscr(idx):
            f, d = oscr_curve(score[idx], is_known_correct[idx], is_unknown[idx])
            return float(_trapz(d, f)) if f is not None else float("nan")

        out["OSCR_AUC_lo"], out["OSCR_AUC_hi"] = cluster_bootstrap_ci(
            _oscr, test_true, n_boot)
        out["AUROC_lo"], out["AUROC_hi"] = cluster_bootstrap_ci(
            lambda idx: _safe_auroc(is_unknown[idx], ood[idx]), test_true, n_boot)
    return out


# ──────────────────────────────────────────────────────────────────────
# Top-level compute
# ──────────────────────────────────────────────────────────────────────
def compute_all_metrics(test_true: np.ndarray, test_pred: np.ndarray,
                        test_emb: np.ndarray, ood_score: np.ndarray) -> dict:
    """One-call metric computation.

    test_true, test_pred : (N,) strings, '-1' for unknowns
    test_emb             : (N, D)
    ood_score            : (N,) — higher = more likely unknown
    """
    auroc, fpr95 = ood_auroc_fpr95(ood_score, test_true)
    return {
        "OS_Top1":              os_top1(test_true, test_pred),
        "Known_only_Top1":      known_only_top1(test_true, test_pred),
        "Unknown_rejection":    unknown_rejection_rate(test_true, test_pred),
        "Balanced_acc":         balanced_acc(test_true, test_pred),
        "Writer_macro_Top1":    writer_macro_top1(test_true, test_pred),
        "AUROC":                auroc,
        "FPR_at_95TPR":         fpr95,
    }


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def derive_ood_score_from_cosine(test_cosine: np.ndarray) -> np.ndarray:
    """Default OOD score = 1 - max_cosine. Higher = more likely unknown."""
    return 1.0 - test_cosine.max(axis=1)


def knn_known_score(test_emb: np.ndarray, gallery_emb: np.ndarray,
                    k: int = 1) -> np.ndarray:

    te = test_emb / (np.linalg.norm(test_emb, axis=1, keepdims=True) + 1e-9)
    ga = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-9)
    sims = te @ ga.T
    if k <= 1:
        return sims.max(axis=1)
    return np.sort(sims, axis=1)[:, -k]


def load_truth(npz: dict) -> tuple:
    """Extract test ground truth from the embeddings npz."""
    test_image_id = np.array([str(x) for x in npz["test_image_id"]])
    test_writer_id = np.array([str(x) for x in npz["test_writer_id"]])
    return test_image_id, test_writer_id


def evaluate_submission(npz: dict, sub_path: Path, n_boot: int = 0,
                        ood_mode: str = "msp") -> dict:
    """Evaluate ONE submission CSV against ground truth in the npz."""
    test_image_id, test_true = load_truth(npz)
    test_emb = npz["test_emb"].astype(np.float32)
    test_cosine = npz["test_cosine"].astype(np.float32)
    if ood_mode == "knn1":
        known_score = knn_known_score(
            test_emb, npz["train_emb"].astype(np.float32), k=1)
        ood_score = 1.0 - known_score
    else:
        known_score = None
        ood_score = derive_ood_score_from_cosine(test_cosine)

    sub = pd.read_csv(sub_path)
    sub["image_id"] = sub["image_id"].astype(str)
    sub["writer_id"] = sub["writer_id"].astype(str)
    # Align rows by image_id
    sub_indexed = sub.set_index("image_id").reindex(test_image_id)
    if sub_indexed["writer_id"].isna().any():
        n_missing = int(sub_indexed["writer_id"].isna().sum())
        raise ValueError(f"{sub_path.name}: {n_missing} image_ids missing")
    test_pred = sub_indexed["writer_id"].to_numpy()

    metrics = compute_all_metrics(test_true, test_pred, test_emb, ood_score)
    writers = np.array([str(w) for w in npz["writers"]])
    metrics.update(oscr_metrics(test_cosine, writers, test_true, n_boot,
                                known_score=known_score))
    return metrics


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=str, nargs="+", required=True,
                    help="Path(s) to embeddings_seed*.npz. Pass multiple for "
                         "mean ± std aggregation across seeds.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pred", type=str, nargs="+",
                   help="Submission CSV(s). One per --emb (or one shared).")
    g.add_argument("--sub-dir", type=str, nargs="+",
                   help="Folder(s) of submission CSVs. One per --emb (or one shared).")
    ap.add_argument("--out", type=str, default=None,
                    help="Where to write the aggregated CSV. "
                         "Default: next to first --sub-dir.")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="Writer-clustered bootstrap 95%% CIs (N resamples) for "
                         "OSCR-AUC + AUROC. 0 = off (default; keeps big sweeps fast).")
    ap.add_argument("--ood-score", choices=["msp", "knn1"], default="msp",
                    help="Rejection score for AUROC/FPR95/OSCR/DIR (threshold-free "
                         "half only; OS_Top1/Known come from the submission). "
                         "'msp' (default, unchanged) = 1 - max prototype cosine. "
                         "'knn1' = 1 - nearest gallery-page cosine (KNN-OOD k=1; "
                         "+0.011 CVL / +0.025 HWI AUROC, inductive).")
    return ap.parse_args()


def _broadcast(values, n, label):
    """Allow either one value (shared across seeds) or one-per-seed."""
    if values is None:
        return [None] * n
    if len(values) == 1:
        return values * n
    if len(values) != n:
        raise SystemExit(
            f"--{label}: expected 1 or {n} paths, got {len(values)}"
        )
    return values


def evaluate_per_seed(emb_paths, sub_dirs=None, preds=None, n_boot=0,
                      ood_mode="msp"):
    
    all_rows = []
    for seed_idx, emb_path in enumerate(emb_paths):
        emb_path = Path(emb_path)
        seed_tag = emb_path.stem.replace("embeddings_", "")  # e.g. "seed42"
        npz = np.load(emb_path, allow_pickle=True)

        # Discover submission files for this seed
        if preds is not None and preds[seed_idx] is not None:
            sub_files = [Path(preds[seed_idx])]
        else:
            sub_dir = Path(sub_dirs[seed_idx])
            sub_files = sorted(
                p for p in sub_dir.glob("*.csv") if not p.name.startswith("_")
            )

        for p in sub_files:
            try:
                m = evaluate_submission(npz, p, n_boot, ood_mode)
            except Exception as e:
                print(f"  [{seed_tag}] {p.name}: SKIPPED ({e})")
                continue
            all_rows.append({
                "seed": seed_idx,
                "seed_tag": seed_tag,
                "method": _method_family(p.stem),  # family name, groupable across seeds
                "unk_pct": _unk_pct(p.stem),       # per-seed chosen percentile (info)
                **m,
            })
    return pd.DataFrame(all_rows)


def aggregate_across_seeds(df_long: pd.DataFrame) -> pd.DataFrame:
    
    metric_cols = [c for c in df_long.columns
                   if c not in ("seed", "seed_tag", "method", "unk_pct")]
    g = df_long.groupby("method")
    df_mean = g[metric_cols].mean().add_suffix("_mean")
    df_std = g[metric_cols].std(ddof=1).add_suffix("_std")     # sample std (n-1)
    df_n = g.size().rename("n_seeds")
    df_unk = g["unk_pct"].apply(
        lambda s: ",".join(str(int(x)) for x in s.dropna())
    ).rename("unk_pcts")
    df_agg = pd.concat([df_n, df_unk, df_mean, df_std], axis=1).reset_index()

    # Interleave mean/std columns for readability
    ordered_cols = ["method", "n_seeds", "unk_pcts"]
    for m in metric_cols:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    return df_agg[ordered_cols].sort_values("OS_Top1_mean", ascending=False)


def _format_mean_std(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Pretty 'mean ± std' columns for printing."""
    metric_bases = sorted({c[:-5] for c in df_agg.columns if c.endswith("_mean")})
    keep_cols = [c for c in ["method", "n_seeds", "unk_pcts"] if c in df_agg.columns]
    out = df_agg[keep_cols].copy()
    for m in metric_bases:
        mean_c, std_c = f"{m}_mean", f"{m}_std"
        out[m] = df_agg.apply(
            lambda r: f"{r[mean_c]:.4f} ± {r[std_c]:.4f}", axis=1
        )
    return out


def main():
    args = parse_args()
    n_seeds = len(args.emb)
    sub_dirs = _broadcast(args.sub_dir, n_seeds, "sub-dir")
    preds = _broadcast(args.pred, n_seeds, "pred")

    # ── Single-seed, single --pred: print a small table and exit ──
    if n_seeds == 1 and args.pred:
        npz = np.load(args.emb[0], allow_pickle=True)
        m = evaluate_submission(npz, Path(args.pred[0]), args.bootstrap,
                                args.ood_score)
        print(f"\n{Path(args.pred[0]).name}")
        print("-" * 60)
        for k, v in m.items():
            print(f"  {k:<22} {v:.4f}")
        return

    # ── Long-form per-seed table ──
    df_long = evaluate_per_seed(args.emb, sub_dirs=sub_dirs, preds=preds,
                                n_boot=args.bootstrap, ood_mode=args.ood_score)
    if df_long.empty:
        raise SystemExit("No metrics computed — check --emb / --sub-dir paths.")

    # ── Single-seed path: keep original behavior ──
    if n_seeds == 1:
        df = df_long.drop(columns=["seed", "seed_tag"])
        df = df.sort_values("OS_Top1", ascending=False)
        out = Path(args.out) if args.out else Path(sub_dirs[0]) / "_full_metrics.csv"
        df.to_csv(out, index=False, float_format="%.4f")
        print("\nFull metrics (sorted by OS_Top1):")
        print("-" * 110)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print(f"\nSaved: {out}")
        return

    # ── Multi-seed path: aggregate to mean ± std ──
    df_agg = aggregate_across_seeds(df_long)
    out_long = (Path(args.out).with_name("_full_metrics_per_seed.csv")
                if args.out else Path(sub_dirs[0]) / "_full_metrics_per_seed.csv")
    out_agg = (Path(args.out) if args.out
               else Path(sub_dirs[0]) / "_full_metrics_aggregated.csv")
    df_long.to_csv(out_long, index=False, float_format="%.4f")
    df_agg.to_csv(out_agg, index=False, float_format="%.4f")

    print(f"\nSeeds evaluated: {n_seeds} "
          f"({', '.join(df_long['seed_tag'].unique())})")
    print("\nPer-seed metrics (sorted by OS_Top1):")
    print("-" * 110)
    print(df_long.sort_values(["method", "seed"])
                 .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nAggregated mean ± std across seeds (sorted by OS_Top1):")
    print("-" * 110)
    print(_format_mean_std(df_agg).to_string(index=False))
    print(f"\nSaved per-seed:    {out_long}")
    print(f"Saved aggregated:  {out_agg}")


if __name__ == "__main__":
    main()
