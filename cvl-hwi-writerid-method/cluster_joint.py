"""Open-set rule C: joint train+test agglomerative clustering.

Clusters the union of train (enrollment) + query embeddings; each query is assigned the
majority TRAIN-writer of its cluster, or rejected (-1) if the cluster has too few train
members or too weak a majority. K, min-train-count, and majority-share are tuned on the
validation Pseudo-unknown pool. Run on the posthoc-refined embeddings, leakage-clean.


Usage:  python cluster_joint.py runs/<dir>/embeddings_seed<N>.npz
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from submit_writerid import open_set_top1_acc


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def joint_cluster(tr, q, K):
    """Agglomerative (cosine, average linkage) on the train+query union."""
    U = np.vstack([tr, q])
    K = max(2, min(K, len(U) - 1))
    clu = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit_predict(U)
    return clu, len(tr)


def joint_assign(clu, ntr, tr_wid, nq, min_train, share):
    ctr, cq = clu[:ntr], clu[ntr:]
    pred = np.full(nq, -1, dtype=np.int64)
    for c in np.unique(clu):
        mem = tr_wid[ctr == c]
        if len(mem) < min_train:
            continue
        vals, cnts = np.unique(mem, return_counts=True)
        if cnts.max() / len(mem) < share:
            continue
        pred[cq == c] = vals[cnts.argmax()]
    return pred


def main(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    W = z["writers"].astype(str)
    w2i = {w: i for i, w in enumerate(W)}
    nW = len(W)
    lab = z["train_labels"].astype(int)

    tr = l2(z["train_emb"].astype(np.float64))
    vl = l2(z["val_emb"].astype(np.float64))
    ts = l2(z["test_emb"].astype(np.float64))

    vid = z["val_writer_id"].astype(str)
    tid = z["test_writer_id"].astype(str)
    v_unk, t_unk = (vid == "-1"), (tid == "-1")
    v_true = np.array([w2i.get(w, -1) for w in vid])
    t_true = np.array([w2i.get(w, -1) for w in tid])

    Kgrid = sorted({int(round(nW * f)) for f in (0.5, 1.0, 1.2, 1.5, 2.0)})
    val_clus = {K: joint_cluster(tr, vl, K) for K in Kgrid}
    test_clus = {K: joint_cluster(tr, ts, K) for K in Kgrid}

    def tune(min_trains, shares):
        best = None
        for K in Kgrid:
            clu, ntr = val_clus[K]
            for mt in min_trains:
                for sh in shares:
                    vp = joint_assign(clu, ntr, lab, len(vl), mt, sh)
                    vacc = open_set_top1_acc(vp, v_true, v_unk)
                    if best is None or vacc > best[0]:
                        best = (vacc, K, mt, sh)
        _, K, mt, sh = best
        clu, ntr = test_clus[K]
        tp = joint_assign(clu, ntr, lab, len(ts), mt, sh)
        os1 = open_set_top1_acc(tp, t_true, t_unk)
        known = float((tp[~t_unk] == t_true[~t_unk]).mean()) if (~t_unk).any() else float("nan")
        unk = float((tp[t_unk] == -1).mean()) if t_unk.any() else float("nan")
        return os1, known, unk, K, mt, sh

    print(f"method C: joint train+test clustering  ({npz_path})")
    for tag, mts, shs in (("ungated", (1,), (0.0,)),
                          ("gated  ", (1, 2, 3), (0.0, 0.5, 0.67, 1.0))):
        os1, kn, un, K, mt, sh = tune(mts, shs)
        print(f"  {tag}:  OS-Top1={os1:.4f}  Known={kn:.4f}  Unk_rej={un:.4f}  "
              f"(val-tuned K={K} min_train={mt} share={sh})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/repro_cvl_seed42/embeddings_seed42.npz")
