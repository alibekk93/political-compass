"""Post-hoc member metadata: names, parties, bloc labels, dim1 orientation.

Everything here runs on the finished positions table — after scaling,
alignment and clustering — so no label can reach anything that is fitted.
The only change metadata makes to the geometry is dim1's sign, a global
reflection that moves no distance, cluster or lineage; dim2 keeps the
deterministic convention from `alignment.fix_signs`.

Raw columns used: congress, chamber, icpsr, state_abbrev, party_code,
bioname (members) and party_code, party_name (parties). The members file
also carries Voteview's own NOMINATE scores (`nominate_*`,
`nokken_poole_*`); like `prob` in the votes file, they are never read.
"""
from __future__ import annotations

import pandas as pd

from .data_io import PROJECT_ROOT, first_year

MEMBERS_CSV = PROJECT_ROOT / "data" / "HSall_members.csv"
PARTIES_CSV = PROJECT_ROOT / "data" / "HSall_parties.csv"
DEMOCRAT, REPUBLICAN = 100, 200


def available() -> bool:
    return MEMBERS_CSV.exists() and PARTIES_CSV.exists()


def load_members() -> pd.DataFrame:
    """One row per (congress, chamber, icpsr) — unique in Voteview's file —
    with name, state, party code and party name.

    Presidents' announced positions sit in the votes file under both
    chambers' rollcalls (so they are scaled like members), while the member
    file lists them under chamber "President"; their rows are copied into
    both chambers so they join too.
    """
    members = pd.read_csv(
        MEMBERS_CSV,
        usecols=["congress", "chamber", "icpsr", "state_abbrev", "party_code", "bioname"],
    )
    pres = members[members["chamber"] == "President"]
    members = pd.concat(
        [members[members["chamber"] != "President"],
         pres.assign(chamber="House"), pres.assign(chamber="Senate")],
        ignore_index=True,
    )
    # party_code -> party_name is 1:1 across congresses in Voteview's file
    parties = pd.read_csv(PARTIES_CSV, usecols=["party_code", "party_name"])
    names = parties.drop_duplicates("party_code").set_index("party_code")["party_name"]
    # a few hundred rows store the code as "200.0", which makes pandas read floats
    code = members["party_code"].astype("Int64")
    return members.assign(party_code=code, party=code.map(names))


def attach(positions: pd.DataFrame) -> pd.DataFrame:
    """Add bioname, state_abbrev, party_code, party to each member-congress row."""
    return positions.merge(load_members(), on=["congress", "chamber", "icpsr"], how="left")


def orient_axes(positions: pd.DataFrame) -> tuple[pd.DataFrame, bool, float]:
    """Reflect dim1 so Republicans sit right of Democrats in the reference
    (latest) congress. Returns (positions, flipped, R-minus-D dim1 gap)."""
    ref = positions[positions["congress"] == positions["congress"].max()]
    means = ref.groupby("party_code")["dim1"].mean()
    gap = float(means[REPUBLICAN] - means[DEMOCRAT])
    if gap < 0:
        positions = positions.assign(dim1=-positions["dim1"])
    return positions, gap < 0, abs(gap)


def bloc_summary(positions: pd.DataFrame) -> pd.DataFrame:
    """One row per bloc lineage, named by the parties that dominate it.

    A bloc's dominant party in a congress is its plurality party there.
    `segments` run-length-encodes that over the bloc's life ("Jackson 19–24;
    Democrat 25–119"), `label` keeps just the party sequence, and `purity` is
    the share of the bloc's member-congress observations belonging to their
    congress's dominant party — how party-coherent the label is.
    """
    counts = positions.groupby(["bloc", "congress", "party"]).size().rename("n").reset_index()
    dominant = (
        counts.sort_values(["n", "party"], ascending=[False, True], kind="stable")
        .drop_duplicates(["bloc", "congress"])
        .sort_values(["bloc", "congress"])
    )
    rows = []
    for b, d in dominant.groupby("bloc"):
        g = positions[positions["bloc"] == b]
        runs: list[list] = []  # [party, first congress, last congress]
        for party, t in zip(d["party"], d["congress"]):
            if runs and runs[-1][0] == party:
                runs[-1][2] = t
            else:
                runs.append([party, t, t])
        rows.append(
            {
                "bloc": int(b),
                "first_congress": int(g["congress"].min()),
                "last_congress": int(g["congress"].max()),
                "first_year": first_year(int(g["congress"].min())),
                "last_year": first_year(int(g["congress"].max())) + 2,
                "n_obs": len(g),
                "n_members": g["icpsr"].nunique(),
                "label": " → ".join(p for p, _, _ in runs),
                "purity": round(float(d["n"].sum() / len(g)), 3),
                "segments": "; ".join(f"{p} {a}–{z}" if a != z else f"{p} {a}" for p, a, z in runs),
            }
        )
    return pd.DataFrame(rows)
