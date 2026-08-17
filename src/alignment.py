"""Stitch per-congress embeddings into one common frame via shared members.

Each congress's MDS solution is only defined up to rotation/reflection (and
scale), so congresses are chained onto a reference frame with orthogonal
Procrustes fits on the members they share, then a generalized-Procrustes
pass removes the drift a long chain accumulates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes


def normalize_cloud(X: np.ndarray) -> np.ndarray:
    """Center and scale to RMS radius 1 — raw MDS scale is not comparable
    across eras; polarization is read from spectra/cluster separation instead."""
    Xc = X - X.mean(axis=0)
    rms = np.sqrt(np.mean(np.sum(Xc**2, axis=1)))
    return Xc / rms if rms > 0 else Xc


def _fit_rt(A: np.ndarray, B: np.ndarray):
    """Rotation(+reflection) and translation mapping A onto B (rows correspond)."""
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    R, _ = orthogonal_procrustes(A - muA, B - muB)
    return R, muA, muB


def _apply(X: np.ndarray, R: np.ndarray, muA: np.ndarray, muB: np.ndarray) -> np.ndarray:
    return (X - muA) @ R + muB


def chain_align(
    ids_by_congress: dict[int, np.ndarray],
    coords_by_congress: dict[int, np.ndarray],
    min_anchors: int = 10,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Walk congresses from the most recent backward, aligning each onto the
    already-aligned frame using shared members as anchors.

    A member's anchor target is their position in the nearest later congress
    they appear in, so the anchor pool for congress t is the union of all
    later appearances — not just t+1. Frame orientation is inherited from
    the most recent congress (the reference).
    """
    congresses = sorted(ids_by_congress, reverse=True)
    ref = congresses[0]
    aligned = {ref: normalize_cloud(coords_by_congress[ref])}
    latest = {m: p for m, p in zip(ids_by_congress[ref].tolist(), aligned[ref])}
    log = []
    for t in congresses[1:]:
        ids = ids_by_congress[t]
        X = normalize_cloud(coords_by_congress[t])
        mask = np.array([m in latest for m in ids.tolist()])
        n_anchors = int(mask.sum())
        if n_anchors < 3:
            raise RuntimeError(f"congress {t}: only {n_anchors} anchors, cannot align")
        A = X[mask]
        B = np.stack([latest[m] for m in ids[mask].tolist()])
        R, muA, muB = _fit_rt(A, B)
        Xa = _apply(X, R, muA, muB)
        resid = float(np.mean(np.linalg.norm(_apply(A, R, muA, muB) - B, axis=1)))
        log.append(
            {
                "congress": t,
                "n_anchors": n_anchors,
                "anchor_resid": resid,
                "low_anchors": n_anchors < min_anchors,
            }
        )
        aligned[t] = Xa
        for m, p in zip(ids.tolist(), Xa):
            latest[m] = p
    return aligned, pd.DataFrame(log).set_index("congress").sort_index()


def gpa_refine(
    ids_by_congress: dict[int, np.ndarray],
    aligned: dict[int, np.ndarray],
    ref: int,
    n_iter: int = 5000,
    tol: float = 1e-6,
) -> tuple[dict[int, np.ndarray], list[float]]:
    """Generalized-Procrustes refinement: repeatedly re-fit every congress
    against the consensus (per-member mean) positions until movement stops.

    Uses every multi-congress member, not just adjacent-congress overlap, so
    chain drift gets pulled out. Convergence is diffusion-like along the time
    chain (low-frequency "unbending" modes decay slowest), hence the high
    iteration cap; rotations/translations preserve each congress's shape, so
    there is no collapse to guard against. The global frame is re-pinned to
    the reference congress's incoming orientation at the end.
    """
    congresses = sorted(aligned)
    all_ids = sorted({m for ids in ids_by_congress.values() for m in ids.tolist()})
    row_of = {m: i for i, m in enumerate(all_ids)}
    rows = {t: np.array([row_of[m] for m in ids_by_congress[t].tolist()]) for t in congresses}
    big_rows = np.concatenate([rows[t] for t in congresses])
    n_members = len(all_ids)
    k = aligned[ref].shape[1]
    counts = np.bincount(big_rows, minlength=n_members).astype(np.float64)

    X = {t: aligned[t].copy() for t in congresses}
    original_ref = X[ref].copy()
    history: list[float] = []
    for _ in range(n_iter):
        P = np.vstack([X[t] for t in congresses])
        consensus = np.column_stack(
            [np.bincount(big_rows, weights=P[:, d], minlength=n_members) for d in range(k)]
        ) / counts[:, None]
        move = 0.0
        for t in congresses:
            B = consensus[rows[t]]
            R, muA, muB = _fit_rt(X[t], B)
            Xn = _apply(X[t], R, muA, muB)
            move = max(move, float(np.mean(np.linalg.norm(Xn - X[t], axis=1))))
            X[t] = Xn
        history.append(move)
        if move < tol:
            break
    R, muA, muB = _fit_rt(X[ref], original_ref)
    return {t: _apply(Xt, R, muA, muB) for t, Xt in X.items()}, history


def fix_signs(aligned: dict[int, np.ndarray], ref: int) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Deterministic global reflection: on each axis, the reference-congress
    member with the largest |coordinate| points positive. Semantic
    orientation (which side is 'left') waits for the names/parties phase."""
    X = aligned[ref]
    flips = np.sign(X[np.abs(X).argmax(axis=0), np.arange(X.shape[1])])
    flips[flips == 0] = 1.0
    return {t: Xt * flips for t, Xt in aligned.items()}, flips


def similarity_align(A: np.ndarray, B: np.ndarray, scale: bool = True) -> np.ndarray:
    """Best-fit similarity transform of A onto B (rotation/reflection,
    translation, optional isotropic scale). Utility for validation —
    comparing a recovered configuration against ground truth or a replicate."""
    muA, muB = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - muA, B - muB
    R, _ = orthogonal_procrustes(A0, B0)
    s = float(np.trace((A0 @ R).T @ B0) / np.trace(A0.T @ A0)) if scale else 1.0
    return s * (A0 @ R) + muB
