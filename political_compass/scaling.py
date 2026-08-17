"""Per-congress scaling: agreement-score distances + classical MDS.

Agreement distances handle missing votes without imputation (zero-filling a
±1 matrix before PCA drags low-attendance members toward the origin), and
the member x member matrices are tiny (<= ~450 x 450), so a full
eigendecomposition is trivial.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def agreement_distance(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise disagreement distances from a member x rollcall matrix.

    V holds +1/-1 for cast votes and 0 for not-cast. Distance between two
    members = 1 - (share of agreements among rollcalls both cast). Pairs
    with no jointly cast rollcall (e.g. a member and their mid-term
    replacement) get the mean of their rows' defined distances.
    Returns (D, J) where J counts jointly cast rollcalls per pair.
    """
    Vf = V.astype(np.float64)
    M = (V != 0).astype(np.float64)
    C = Vf @ Vf.T  # agreements minus disagreements on joint rollcalls
    J = M @ M.T
    with np.errstate(divide="ignore", invalid="ignore"):
        D = 1.0 - 0.5 * (1.0 + C / J)
    np.fill_diagonal(D, 0.0)
    nan = np.isnan(D)
    if nan.any():
        row_mean = np.nanmean(D, axis=1)
        row_mean = np.where(np.isnan(row_mean), np.nanmean(D), row_mean)
        i, j = np.where(nan)
        D[i, j] = 0.5 * (row_mean[i] + row_mean[j])
    return D, J


def classical_mds(D: np.ndarray, k: int = 2, n_eigs: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Classical (Torgerson) MDS: coordinates for the top-k dimensions.

    Also returns the leading n_eigs eigenvalues (descending) — the spectrum
    is the per-congress dimensionality diagnostic. Agreement distances are
    not exactly Euclidean, so trailing eigenvalues can go negative; any
    negative eigenvalue within the top k collapses that coordinate to 0.
    """
    B = D.astype(np.float64) ** 2
    B -= B.mean(axis=0)
    B -= B.mean(axis=1)[:, None]
    B *= -0.5
    w, U = eigh(B)
    order = np.argsort(w)[::-1]
    top = w[order[:n_eigs]]
    wk = np.clip(w[order[:k]], 0.0, None)
    X = U[:, order[:k]] * np.sqrt(wk)
    return X, top
