"""Split-half reliability on real data.

Scale the even-indexed and odd-indexed rollcalls of a congress separately,
map one half onto the other with a similarity Procrustes fit, and check the
two independent position estimates agree. Requires the parquet cache
(`python -m political_compass.pipeline` builds it). Runnable directly or via
pytest.
"""
from __future__ import annotations

import numpy as np

from political_compass import alignment, data_io, scaling

CONGRESSES = [119, 100, 80, 60, 40, 20]


def split_half_r(chamber: str, congress: int) -> tuple[float, int]:
    votes = data_io.load_votes(chamber, congress)
    ids, V, _ = data_io.build_vote_matrix(votes)
    halves = []
    for parity in (0, 1):
        Vh = V[:, parity::2]
        cast = (Vh != 0).sum(axis=1)
        keep = cast >= 10
        D, _ = scaling.agreement_distance(Vh[keep])
        X, _ = scaling.classical_mds(D, k=2)
        halves.append((ids[keep], alignment.normalize_cloud(X)))
    (ids_a, Xa), (ids_b, Xb) = halves
    common = np.intersect1d(ids_a, ids_b)
    A = Xa[np.searchsorted(ids_a, common)]
    B = Xb[np.searchsorted(ids_b, common)]
    B_fit = alignment.similarity_align(B, A)
    r = float(np.corrcoef(A.ravel(), B_fit.ravel())[0, 1])
    return r, len(common)


def test_split_half_reliability():
    if not data_io.MANIFEST.exists():
        # No cache means no real data (CI). Skip honestly rather than passing
        # vacuously or dying in build_cache with a FileNotFoundError.
        import pytest

        pytest.skip("parquet cache absent; run python -m political_compass.pipeline")
    rows = []
    for chamber in ("House", "Senate"):
        available = set(data_io.available_congresses(chamber))
        for t in CONGRESSES:
            if t not in available:
                continue
            r, n = split_half_r(chamber, t)
            rows.append((chamber, t, r, n))
            print(f"{chamber:6s} congress {t:3d} ({1787 + 2 * t}): r = {r:.3f}  ({n} members)")
    rs = np.array([r for _, _, r, _ in rows])
    print(f"mean r = {rs.mean():.3f}, min r = {rs.min():.3f}")
    assert rs.mean() > 0.85, f"mean split-half reliability too low: {rs.mean():.3f}"
    assert rs.min() > 0.6, f"worst split-half reliability too low: {rs.min():.3f}"


if __name__ == "__main__":
    test_split_half_reliability()
    print("split-half reliability ok")
