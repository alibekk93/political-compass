"""Crosswalk Voteview roll-call topic tags onto the six Clausen issue areas.

Clausen (1973) issue-area codes cover congresses 1-113; Congressional Research
Service (CRS) policy areas cover 93-119. Where both exist (93-113) each roll
call carries both, so the crosswalk is learned from the data: for every policy
area, the share of its roll calls that Clausen coding put in each area. Roll
calls with no policy area (nominations, treaties, votes without a bill) get
the Clausen mix that votes of the same kind had in the overlap.

Only roll calls the pipeline scales (`data_io.contested_rollcalls`) are used,
so the weights describe the votes that get interpreted. The source file's
NOMINATE columns are never read. Writes mappings/*.csv; see mappings/README.md.

Usage:
    python -m political_compass.issue_mappings
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data_io

ROLLCALLS_JSON = data_io.PROJECT_ROOT / "data" / "HSall_rollcalls.json"
MAPPINGS_DIR = data_io.PROJECT_ROOT / "mappings"
CLAUSEN_AREAS = [
    "Government Management", "Foreign and Defense Policy", "Social Welfare",
    "Civil Liberties", "Agriculture", "Miscellaneous Policy",
]
OVERLAP = (93, 113)  # congresses coded with both Clausen and CRS policy areas
LAST_CLAUSEN = OVERLAP[1]
# kinds of roll call that have no policy area, tested on bill_number in this order
VOTE_KINDS = [
    ("nomination", "regex:^PN"),
    ("treaty", "regex:^TREATYDOC"),
    ("no bill number", "missing"),
    ("bill without area", "otherwise"),
]


def load_rollcalls() -> pd.DataFrame:
    """House/Senate roll calls with their Clausen code and CRS policy area
    (None where absent). Both schemes carry at most one value per roll call."""
    with open(ROLLCALLS_JSON) as f:
        rc = pd.DataFrame(json.load(f))
    rc = rc[rc["chamber"].isin(["House", "Senate"])]
    first = lambda v: v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else None)
    return pd.DataFrame({
        "congress": rc["congress"], "chamber": rc["chamber"], "rollnumber": rc["rollnumber"],
        "bill_number": rc["bill_number"],
        "clausen": rc["clausen_codes"].map(first), "crs_policy_area": rc["crs_policy_area"].map(first),
    }).reset_index(drop=True)


def borrow_bill_area(rc: pd.DataFrame, sources: pd.DataFrame) -> pd.Series:
    """Policy area per roll call, filling gaps from the most common area among
    `sources` roll calls on the same bill in the same congress (an amendment
    vote often lacks the area its bill's passage vote carries)."""
    by_bill = (sources.dropna(subset=["crs_policy_area", "bill_number"])
               .groupby(["congress", "bill_number"])["crs_policy_area"].agg(lambda s: s.mode().iloc[0]))
    keys = pd.MultiIndex.from_arrays([rc["congress"], rc["bill_number"]])
    return rc["crs_policy_area"].fillna(pd.Series(by_bill.reindex(keys).to_numpy(), index=rc.index))


def vote_kind(bill_number: pd.Series) -> pd.Series:
    """Classify roll calls without a policy area by their bill number (VOTE_KINDS)."""
    b = bill_number.fillna("")
    kinds = np.select([b.str.match("^PN"), b.str.match("^TREATYDOC"), b.eq("")],
                      [k for k, _ in VOTE_KINDS[:3]], VOTE_KINDS[3][0])
    return pd.Series(kinds, index=bill_number.index)


def rollcall_weights(rc: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clausen-area weights per roll call of `rc` (default: every roll call), from
    the roll call's own Clausen code, else its policy area (borrowed from the same
    bill if missing), else its vote kind. Returns the six CLAUSEN_AREAS columns
    (summing to 1) plus `source`: "clausen", "policy area", or the vote kind."""
    rc = load_rollcalls() if rc is None else rc
    areas = pd.read_csv(MAPPINGS_DIR / "crs_policy_area_to_clausen.csv", index_col="crs_policy_area")
    kinds = pd.read_csv(MAPPINGS_DIR / "vote_kind_to_clausen.csv", index_col="vote_kind")
    w = pd.get_dummies(rc["clausen"]).reindex(columns=CLAUSEN_AREAS, fill_value=0).astype(float)
    source = pd.Series("clausen", index=rc.index)
    todo = w.sum(axis=1) == 0
    area = borrow_bill_area(rc, rc)
    by_area = todo & area.isin(areas.index)
    w.loc[by_area] = areas.loc[area[by_area], CLAUSEN_AREAS].to_numpy()
    source[by_area] = "policy area"
    rest = todo & ~by_area
    kind = vote_kind(rc.loc[rest, "bill_number"])
    w.loc[rest] = kinds.loc[kind, CLAUSEN_AREAS].to_numpy()
    source[rest] = kind
    return w.assign(source=source)


def clausen_weights(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per `key` value: roll-call count, majority Clausen area and its share,
    and the share in each Clausen area (the soft mapping)."""
    counts = pd.crosstab(df[key], df["clausen"]).reindex(columns=CLAUSEN_AREAS, fill_value=0)
    w = counts.div(counts.sum(axis=1), axis=0)
    return pd.DataFrame({
        "n_overlap": counts.sum(axis=1),
        "clausen_majority": w.idxmax(axis=1),
        "majority_share": w.max(axis=1).round(4),
    }).join(w.round(4))


def scaled_keys() -> set[tuple[str, int, int]]:
    """(chamber, congress, rollnumber) of every roll call the pipeline scales."""
    keys = set()
    for chamber in ("House", "Senate"):
        for t in data_io.available_congresses(chamber):
            keys |= {(chamber, t, int(r)) for r in data_io.contested_rollcalls(data_io.load_votes(chamber, t))}
    return keys


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    every = load_rollcalls()
    keys = scaled_keys()
    rc = every[[k in keys for k in zip(every["chamber"], every["congress"], every["rollnumber"])]].copy()
    overlap = rc[rc["congress"].between(*OVERLAP) & rc["clausen"].notna()]

    tagged = overlap[overlap["crs_policy_area"].notna()]
    areas = clausen_weights(tagged, "crs_policy_area")
    mid = (OVERLAP[0] + OVERLAP[1]) // 2
    halves = [tagged[tagged["congress"] <= mid], tagged[tagged["congress"] > mid]]
    first, second = (pd.crosstab(h["crs_policy_area"], h["clausen"]).idxmax(axis=1).reindex(areas.index)
                     for h in halves)
    areas.insert(3, "stable_across_halves", first == second)
    areas = areas.sort_values("n_overlap", ascending=False)
    areas.index.name = "crs_policy_area"

    rest = overlap[borrow_bill_area(overlap, every).isna()]
    kinds = clausen_weights(rest.assign(vote_kind=vote_kind(rest["bill_number"])), "vote_kind")
    kinds = kinds.reindex([k for k, _ in VOTE_KINDS])
    kinds.insert(0, "bill_number_match", [m for _, m in VOTE_KINDS])
    kinds.index.name = "vote_kind"

    _report(rc, every, tagged, areas, mid)
    return areas, kinds


def _report(rc, every, tagged, areas, mid) -> None:
    """Quality numbers quoted in mappings/README.md."""
    hard = tagged["crs_policy_area"].map(areas["clausen_majority"])
    print(f"overlap roll calls with both codes: {len(tagged):,}; areas mapped: {len(areas)}")
    print(f"majority-mapping agreement with Clausen: {(hard == tagged['clausen']).mean():.1%}")
    train, test = tagged[tagged["congress"] <= mid], tagged[tagged["congress"] > mid]
    fit = pd.crosstab(train["crs_policy_area"], train["clausen"]).idxmax(axis=1)
    print(f"fit on {OVERLAP[0]}-{mid}, tested on {mid + 1}-{OVERLAP[1]}: "
          f"{(test['crs_policy_area'].map(fit) == test['clausen']).mean():.1%}")
    soft = areas.loc[tagged["crs_policy_area"], CLAUSEN_AREAS].mean()
    mix = pd.DataFrame({"actual": tagged["clausen"].value_counts(normalize=True),
                        "majority": hard.value_counts(normalize=True), "weights": soft})
    print(mix.reindex(CLAUSEN_AREAS).fillna(0).round(3).to_string())
    modern = rc[rc["congress"] > LAST_CLAUSEN]
    unseen = set(modern["crs_policy_area"].dropna()) - set(areas.index)
    if unseen:
        print(f"WARNING: post-{LAST_CLAUSEN} policy areas with no mapping: {sorted(unseen)}")
    own = modern["crs_policy_area"].notna()
    borrowed = ~own & borrow_bill_area(modern, every).notna()
    kind = vote_kind(modern["bill_number"])
    print(f"congresses {LAST_CLAUSEN + 1}-{modern['congress'].max()}: {len(modern):,} scaled roll calls; "
          f"own policy area {own.mean():.0%}, borrowed from same bill {borrowed.mean():.0%}, "
          f"by vote kind: " + ", ".join(f"{k} {((~own & ~borrowed) & (kind == k)).mean():.0%}" for k, _ in VOTE_KINDS))


def main() -> None:
    areas, kinds = build()
    MAPPINGS_DIR.mkdir(exist_ok=True)
    areas.to_csv(MAPPINGS_DIR / "crs_policy_area_to_clausen.csv")
    kinds.to_csv(MAPPINGS_DIR / "vote_kind_to_clausen.csv")
    print(f"wrote {MAPPINGS_DIR / 'crs_policy_area_to_clausen.csv'} and {MAPPINGS_DIR / 'vote_kind_to_clausen.csv'}")


if __name__ == "__main__":
    main()
