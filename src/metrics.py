"""Evaluation metrics (AUROC, AUPRC, FAR@80%R), scene-cluster bootstrap, and Holm-Bonferroni correction."""
import numpy as np
from scipy.stats import rankdata

def fast_auc(yv: np.ndarray, sv: np.ndarray) -> float:
    """Exact Mann-Whitney U / rank-based AUROC."""
    yv = np.asarray(yv, dtype=int)
    sv = np.asarray(sv, dtype=float)
    n1 = int(yv.sum())
    n0 = len(yv) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((rankdata(sv)[yv == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def fast_auprc(yv: np.ndarray, sv: np.ndarray) -> float:
    """Exact average precision (AUPRC) sorted by descending score."""
    yv = np.asarray(yv, dtype=int)
    sv = np.asarray(sv, dtype=float)
    n1 = int(yv.sum())
    if n1 == 0:
        return np.nan
    order = np.argsort(-sv, kind="mergesort")
    y_sorted = yv[order]
    tp = np.cumsum(y_sorted == 1)
    prec = tp / np.arange(1, len(yv) + 1)
    return float(prec[y_sorted == 1].sum() / n1)

def far_at_80_recall(yv: np.ndarray, sv: np.ndarray) -> float:
    """False alarm rate on negative clips at the 80% true positive recall threshold."""
    yv = np.asarray(yv, dtype=int)
    sv = np.asarray(sv, dtype=float)
    pos = sv[yv == 1]
    neg = sv[yv == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    thr = np.percentile(pos, 20.0)
    return float((neg >= thr).mean())

def holm_bonferroni(p_values: list) -> list:
    """Step-down Holm-Bonferroni multiple-comparison adjusted p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        val = min(1.0, p_values[orig_idx] * (m - rank_idx))
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj.tolist()

def scene_bootstrap(keys, score_dict, y: np.ndarray, dom: np.ndarray, scene: np.ndarray,
                    domains: list, n_rep: int = 500, seed: int = 0) -> dict:
    """Resample strict scenes with replacement within each domain across `n_rep` replicates."""
    rng = np.random.RandomState(seed)
    info = {}
    for d in domains:
        ii = np.where(dom == d)[0]
        sc = scene[ii]
        U = np.unique(sc)
        info[d] = (ii, U, {u: np.where(sc == u)[0] for u in U})
    R = {k: {d: [] for d in domains} for k in keys}
    for _ in range(n_rep):
        for d, (ii, U, idx_map) in info.items():
            j = np.concatenate([idx_map[u] for u in rng.choice(U, len(U), replace=True)])
            yj = y[ii][j]
            for k in keys:
                R[k][d].append(fast_auc(yj, score_dict[k][ii][j]))
    return {k: {d: np.array(v) for d, v in R[k].items()} for k in keys}
