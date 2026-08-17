"""Clustering in the common space and cluster-lineage continuity over time."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_congress(
    X: np.ndarray,
    k_range=range(2, 7),
    random_state: int = 0,
    min_cluster_frac: float = 0.05,
    parsimony_tol: float = 0.02,
) -> dict:
    """KMeans sweep over k; keeps the best-silhouette labels plus a fixed
    k=2 solution whose separation serves as a polarization proxy.

    Silhouette on a small chamber is noisy enough to spawn splinter
    clusters, so a k > 2 solution is only eligible if its smallest cluster
    holds >= min_cluster_frac of members, and the smallest eligible k within
    parsimony_tol of the best silhouette wins.
    """
    sils: dict[int, float] = {}
    labels_by_k: dict[int, np.ndarray] = {}
    eligible: list[int] = []
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        sils[k] = float(silhouette_score(X, km.labels_))
        labels_by_k[k] = km.labels_
        min_frac = np.bincount(km.labels_).min() / len(X)
        if k == 2 or min_frac >= min_cluster_frac:
            eligible.append(k)
    best_sil = max(sils[k] for k in eligible)
    best_k = min(k for k in eligible if sils[k] >= best_sil - parsimony_tol)
    k2_labels = labels_by_k.get(2)
    return {
        "k": best_k,
        "labels": labels_by_k[best_k],
        "silhouettes": sils,
        "silhouette_k2": sils.get(2, float("nan")),
        "labels_k2": k2_labels,
        "separation_k2": _separation(X, k2_labels) if k2_labels is not None else float("nan"),
    }


def _separation(X: np.ndarray, labels: np.ndarray) -> float:
    """Distance between the two cluster centroids in units of within-cluster
    spread — comparable across congresses because clouds are RMS-normalized."""
    c0, c1 = X[labels == 0].mean(axis=0), X[labels == 1].mean(axis=0)
    within = np.sqrt(
        np.mean(
            [
                np.mean(np.sum((X[labels == i] - c) ** 2, axis=1))
                for i, c in ((0, c0), (1, c1))
            ]
        )
    )
    return float(np.linalg.norm(c0 - c1) / within) if within > 0 else float("inf")


def lineage_relabel(
    ids_by_congress: dict[int, np.ndarray],
    labels_by_congress: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    """Turn per-congress cluster labels into persistent bloc ids.

    Clusters in consecutive congresses are matched by Hungarian assignment on
    how many shared members they have; a cluster with no inherited match
    starts a new bloc id. This recovers party-system lineages without any
    party labels.
    """
    blocs: dict[int, np.ndarray] = {}
    next_bloc = 0
    prev_map: dict[int, int] = {}  # icpsr -> bloc id in previous congress
    for t in sorted(labels_by_congress):
        ids, labels = ids_by_congress[t], labels_by_congress[t]
        clusters = np.unique(labels)
        mapping: dict[int, int] = {}
        if not prev_map:
            sizes = {c: int((labels == c).sum()) for c in clusters}
            for c in sorted(clusters, key=lambda c: (-sizes[c], c)):
                mapping[c] = next_bloc
                next_bloc += 1
        else:
            prev_blocs = sorted(set(prev_map.values()))
            overlap = np.zeros((len(prev_blocs), len(clusters)))
            bloc_row = {b: i for i, b in enumerate(prev_blocs)}
            cl_col = {c: j for j, c in enumerate(clusters)}
            for m, lab in zip(ids.tolist(), labels):
                if m in prev_map:
                    overlap[bloc_row[prev_map[m]], cl_col[lab]] += 1
            n = max(overlap.shape)
            padded = np.zeros((n, n))
            padded[: overlap.shape[0], : overlap.shape[1]] = overlap
            rows, cols = linear_sum_assignment(-padded)
            for r, c in zip(rows, cols):
                if c >= len(clusters):
                    continue
                if r < len(prev_blocs) and padded[r, c] > 0:
                    mapping[clusters[c]] = prev_blocs[r]
                else:
                    mapping[clusters[c]] = next_bloc
                    next_bloc += 1
        blocs[t] = np.array([mapping[lab] for lab in labels])
        prev_map = dict(zip(ids.tolist(), blocs[t].tolist()))
    return blocs
