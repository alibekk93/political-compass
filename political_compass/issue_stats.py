"""Issue-area statistics for the notebook dashboard: how often each member
votes against their own party in each Clausen area, and what each congress
voted on.

Post hoc, like metadata.py: it reads the vote cache, the positions table,
member parties and roll-call topic weights (`issue_mappings.rollcall_weights`);
nothing fitted depends on it. Every recorded roll call counts, near-unanimous
ones included, so lone protest votes register. A member whose party is not one
of the congress's two largest is measured against whichever of the two they
vote with more. Roll calls whose area is unknown (`bill without area`) count
toward the all-votes rate only.

Writes output/issue_stats_{chamber}.csv (one row per member-congress) and
output/agenda_{chamber}.csv (one row per congress).

Usage:
    python -m political_compass.issue_stats
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_io, issue_mappings as im, metadata

OUTPUT_DIR = data_io.PROJECT_ROOT / "output"
SLUGS = {
    "Government Management": "gov_mgmt", "Foreign and Defense Policy": "foreign_defense",
    "Social Welfare": "social_welfare", "Civil Liberties": "civil_liberties",
    "Agriculture": "agriculture", "Miscellaneous Policy": "misc",
}


def congress_stats(votes: pd.DataFrame, party: pd.Series, weights: pd.DataFrame) -> pd.DataFrame:
    """Votes counted and defection rate per member, overall and per area, for one
    chamber-congress. votes: icpsr, rollnumber, vote (+1/-1); party: icpsr -> party;
    weights: rollnumber -> CLAUSEN_AREAS weights (unknown area = all zeros)."""
    V = votes.pivot_table(index="icpsr", columns="rollnumber", values="vote", fill_value=0)
    X = V.to_numpy(dtype=float)
    cast = X != 0
    p = party.reindex(V.index)
    top2 = p.value_counts().index[:2]
    majority = np.stack([np.sign(X[(p == q).to_numpy()].sum(axis=0)) for q in top2])
    # agreement with each major party's majorities decides independents and third parties
    has_maj = cast[:, None, :] & (majority[None] != 0)
    agree = ((X[:, None, :] == majority[None]) & has_maj).sum(axis=2) / np.maximum(has_maj.sum(axis=2), 1)
    ref = np.where(p.isin(top2), p.map({q: k for k, q in enumerate(top2)}).fillna(0), agree.argmax(axis=1))
    M = majority[ref.astype(int)]
    counted = cast & (M != 0)
    defect = counted & (X != M)
    W = weights.reindex(V.columns).fillna(0.0)[im.CLAUSEN_AREAS].to_numpy()
    rate = lambda d, n: np.divide(d, n, out=np.full(len(n), np.nan), where=n > 0)
    out = pd.DataFrame({"icpsr": V.index, "ref_party": np.asarray(top2)[ref.astype(int)],
                        "n_all": counted.sum(axis=1), "defect_all": rate(defect.sum(axis=1), counted.sum(axis=1))})
    n, d = counted.astype(float) @ W, defect.astype(float) @ W
    for k, area in enumerate(im.CLAUSEN_AREAS):
        out[f"n_{SLUGS[area]}"] = n[:, k].round(2)
        out[f"defect_{SLUGS[area]}"] = rate(d[:, k], n[:, k])
    return out


def build(chamber: str, rc: pd.DataFrame, weights: pd.DataFrame, members: pd.DataFrame):
    keep = pd.read_csv(OUTPUT_DIR / f"positions_{chamber}.csv", usecols=["congress", "icpsr"])
    mine = (rc["chamber"] == chamber).to_numpy()
    w = weights[mine].set_index(rc.loc[mine, "congress"].to_numpy())
    rolls = rc.loc[mine, "rollnumber"].to_numpy()
    known = w[im.CLAUSEN_AREAS].where(w["source"] != "bill without area", 0.0)
    stats = []
    for t, ids in keep.groupby("congress")["icpsr"]:
        wt = known.loc[[t]].set_axis(rolls[w.index == t])
        party = members.loc[members["congress"] == t].set_index("icpsr")["party"]
        s = congress_stats(data_io.load_votes(chamber, t), party, wt)
        stats.append(s[s["icpsr"].isin(ids)].assign(congress=t))
    stats = pd.concat(stats, ignore_index=True)
    stats = stats[["congress"] + [c for c in stats.columns if c != "congress"]]
    agenda = known.groupby(level=0).sum().rename(columns=SLUGS)
    agenda["area_unknown"] = (w["source"] == "bill without area").groupby(level=0).sum()
    agenda.index.name = "congress"
    return stats.round(4), agenda.round(2)


def main() -> None:
    rc = im.load_rollcalls()
    weights = im.rollcall_weights(rc)
    members = metadata.load_members()
    for chamber in ("House", "Senate"):
        stats, agenda = build(chamber, rc, weights, members[members["chamber"] == chamber])
        stats.to_csv(OUTPUT_DIR / f"issue_stats_{chamber}.csv", index=False)
        agenda.to_csv(OUTPUT_DIR / f"agenda_{chamber}.csv")
        print(f"[{chamber}] {len(stats):,} member-congresses, {len(agenda)} congresses -> "
              f"issue_stats_{chamber}.csv, agenda_{chamber}.csv")


if __name__ == "__main__":
    main()
