"""End-to-end recovery test on synthetic data with known ideal points.

Simulates two-blob 2D ideal points with member turnover and slow drift,
generates rollcall votes from random cutting planes, then checks that
scaling + chained alignment + GPA refinement recovers the true positions up
to one global similarity transform. Runnable directly (`python -m
tests.test_synthetic`) or via pytest.
"""
from __future__ import annotations

import numpy as np

from src import alignment, clustering, scaling


def simulate(
    n_congresses: int = 10,
    n_members: int = 80,
    n_rollcalls: int = 300,
    turnover: float = 0.25,
    drift: float = 0.05,
    missing: float = 0.08,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    next_id = 0

    def new_member():
        nonlocal next_id
        blob = int(rng.random() < 0.5)
        mu = np.array([-1.0, -0.3]) if blob == 0 else np.array([1.0, 0.3])
        member = {"id": next_id, "pos": mu + rng.normal(0, 0.35, 2), "blob": blob}
        next_id += 1
        return member

    roster = [new_member() for _ in range(n_members)]
    ids_by, votes_by, true_pos, true_blob = {}, {}, {}, {}
    for t in range(1, n_congresses + 1):
        if t > 1:
            for i in rng.choice(n_members, int(turnover * n_members), replace=False):
                roster[i] = new_member()
            for m in roster:
                m["pos"] = m["pos"] + rng.normal(0, drift, 2)
        P = np.stack([m["pos"] for m in roster])
        V = np.zeros((n_members, n_rollcalls), dtype=np.int8)
        for j in range(n_rollcalls):
            w = rng.normal(0, 1, 2)
            w /= np.linalg.norm(w)
            proj = P @ w
            cut = rng.normal(proj.mean(), 0.7)
            p_yea = 1.0 / (1.0 + np.exp(-4.0 * (proj - cut)))
            V[:, j] = np.where(rng.random(n_members) < p_yea, 1, -1)
        V[rng.random(V.shape) < missing] = 0
        ids_by[t] = np.array([m["id"] for m in roster])
        votes_by[t] = V
        true_pos[t] = P.copy()
        true_blob[t] = np.array([m["blob"] for m in roster])
    return ids_by, votes_by, true_pos, true_blob


def run_pipeline(ids_by, votes_by):
    coords = {}
    for t, V in votes_by.items():
        D, _ = scaling.agreement_distance(V)
        X, _ = scaling.classical_mds(D, k=2)
        coords[t] = X
    aligned, log = alignment.chain_align(ids_by, coords)
    refined, _ = alignment.gpa_refine(ids_by, aligned, ref=max(aligned))
    refined, _ = alignment.fix_signs(refined, ref=max(aligned))
    return refined, log


def test_position_recovery():
    ids_by, votes_by, true_pos, _ = simulate()
    est, log = run_pipeline(ids_by, votes_by)
    order = sorted(est)
    A = np.vstack([est[t] for t in order])
    B = np.vstack([true_pos[t] for t in order])
    A_fit = alignment.similarity_align(A, B)
    r = float(np.corrcoef(A_fit.ravel(), B.ravel())[0, 1])
    assert (log["n_anchors"] >= 10).all(), "anchor pool unexpectedly thin"
    assert r > 0.9, f"recovered-vs-true correlation too low: {r:.3f}"
    print(f"position recovery: r = {r:.3f} over {len(A)} member-congress points")


def test_cluster_recovery():
    from sklearn.metrics import adjusted_rand_score

    ids_by, votes_by, _, true_blob = simulate(seed=1)
    est, _ = run_pipeline(ids_by, votes_by)
    t = max(est)
    res = clustering.cluster_congress(est[t])
    ari = adjusted_rand_score(true_blob[t], res["labels_k2"])
    assert ari > 0.8, f"blob recovery ARI too low: {ari:.3f}"
    print(f"cluster recovery: ARI = {ari:.3f} at congress {t} (best k = {res['k']})")


def test_lineage_continuity():
    ids_by, votes_by, _, true_blob = simulate(seed=2)
    est, _ = run_pipeline(ids_by, votes_by)
    labels = {t: clustering.cluster_congress(est[t])["labels_k2"] for t in est}
    blocs = clustering.lineage_relabel(ids_by, labels)
    from sklearn.metrics import adjusted_rand_score

    aris = [adjusted_rand_score(true_blob[t], blocs[t]) for t in sorted(est)]
    n_blocs = len({b for arr in blocs.values() for b in arr.tolist()})
    assert min(aris) > 0.8, f"lineage broke somewhere: per-congress ARIs {aris}"
    assert n_blocs == 2, f"expected 2 persistent blocs, got {n_blocs}"
    print(f"lineage continuity: min ARI = {min(aris):.3f}, blocs = {n_blocs}")


if __name__ == "__main__":
    test_position_recovery()
    test_cluster_recovery()
    test_lineage_continuity()
    print("all synthetic tests passed")
