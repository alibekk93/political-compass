"""End-to-end pipeline: cache -> scale -> align -> cluster -> outputs.

Usage:
    python -m src.pipeline --chamber House
    python -m src.pipeline --chamber both --no-figures
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import alignment, clustering, data_io, scaling, viz


def run_chamber(chamber: str, outdir: Path, make_figures: bool = True) -> pd.DataFrame:
    t0 = time.time()
    congresses = data_io.available_congresses(chamber)
    print(f"[{chamber}] scaling {len(congresses)} congresses ...")

    ids_by, coords_by, nvotes_by = {}, {}, {}
    diag_rows = {}
    for t in congresses:
        votes = data_io.load_votes(chamber, t)
        try:
            ids, V, n_votes = data_io.build_vote_matrix(votes)
        except ValueError:
            print(f"[{chamber}] congress {t}: nothing survives filters, skipped")
            continue
        if len(ids) < 10:
            print(f"[{chamber}] congress {t}: only {len(ids)} members, skipped")
            continue
        D, _ = scaling.agreement_distance(V)
        X, eigs = scaling.classical_mds(D, k=2)
        ids_by[t], coords_by[t], nvotes_by[t] = ids, X, n_votes
        diag_rows[t] = {
            "n_members": len(ids),
            "n_rollcalls": V.shape[1],
            **{f"eig{i + 1}": float(e) for i, e in enumerate(eigs)},
        }

    print(f"[{chamber}] aligning ({time.time() - t0:.0f}s) ...")
    aligned, chain_log = alignment.chain_align(ids_by, coords_by)
    ref = max(aligned)
    refined, gpa_moves = alignment.gpa_refine(ids_by, aligned, ref)
    refined, _ = alignment.fix_signs(refined, ref)
    print(f"[{chamber}] GPA converged in {len(gpa_moves)} iterations "
          f"(final move {gpa_moves[-1]:.2e})")

    print(f"[{chamber}] clustering ...")
    results = {t: clustering.cluster_congress(refined[t]) for t in refined}
    blocs = clustering.lineage_relabel(ids_by, {t: r["labels"] for t, r in results.items()})

    rows = []
    for t in sorted(refined):
        X = refined[t]
        for i, m in enumerate(ids_by[t].tolist()):
            rows.append(
                {
                    "congress": t,
                    "chamber": chamber,
                    "icpsr": m,
                    "dim1": float(X[i, 0]),
                    "dim2": float(X[i, 1]),
                    "cluster": int(results[t]["labels"][i]),
                    "bloc": int(blocs[t][i]),
                    "n_votes": int(nvotes_by[t][i]),
                }
            )
    positions = pd.DataFrame(rows)

    diag = pd.DataFrame.from_dict(diag_rows, orient="index").sort_index()
    diag.index.name = "congress"
    for t, r in results.items():
        diag.loc[t, ["k", "silhouette_k2", "separation_k2"]] = (
            r["k"], r["silhouette_k2"], r["separation_k2"]
        )
    diag = diag.join(chain_log[["n_anchors", "anchor_resid"]])
    diag["n_blocs_to_date"] = [
        len({b for tt in sorted(blocs) if tt <= t for b in blocs[tt].tolist()})
        for t in diag.index
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    positions.to_csv(outdir / f"positions_{chamber}.csv", index=False)
    diag.to_csv(outdir / f"diagnostics_{chamber}.csv")
    print(f"[{chamber}] wrote {len(positions):,} member-congress positions "
          f"({positions['icpsr'].nunique():,} unique members, "
          f"{positions['bloc'].nunique()} bloc lineages)")

    if make_figures:
        for p in viz.make_all(positions, diag, outdir, chamber):
            print(f"[{chamber}] figure: {p.relative_to(outdir.parent)}")
    print(f"[{chamber}] done in {time.time() - t0:.0f}s")
    return positions


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-congress voting-space pipeline")
    ap.add_argument("--chamber", choices=["House", "Senate", "both"], default="House")
    ap.add_argument("--outdir", default=str(data_io.PROJECT_ROOT / "output"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--force-cache", action="store_true",
                    help="re-parse the raw CSV even if the cache exists")
    args = ap.parse_args()

    data_io.build_cache(force=args.force_cache)
    chambers = ["House", "Senate"] if args.chamber == "both" else [args.chamber]
    for chamber in chambers:
        run_chamber(chamber, Path(args.outdir), make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
