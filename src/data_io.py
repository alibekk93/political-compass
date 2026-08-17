"""Parse the raw Voteview HSall_votes.csv into per-(chamber, congress) caches.

Raw columns: congress, chamber, rollnumber, icpsr, cast_code, prob.
Voteview cast codes: 1-3 = Yea variants, 4-6 = Nay variants, 7-9 =
present/not voting, 0 = not a member when the vote was taken. Only cast
votes are stored (vote = +1 Yea / -1 Nay); everything else stays implicit
as missingness. The `prob` column is an output of Voteview's own NOMINATE
model, so using it would be circular — it is never read.

A member's `icpsr` id is stable across congresses (Voteview issues a new id
on a party switch, so switchers simply drop out of the anchor pool).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "HSall_votes.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST = PROCESSED_DIR / "manifest.json"

CHUNK_ROWS = 2_000_000


def build_cache(force: bool = False) -> dict:
    """One-time parse of the raw CSV into a parquet file per (chamber, congress)."""
    if MANIFEST.exists() and not force:
        return json.loads(MANIFEST.read_text())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    parts: dict[tuple[str, int], list[pd.DataFrame]] = {}
    reader = pd.read_csv(
        RAW_CSV,
        usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"],
        dtype={
            "congress": np.int16,
            "chamber": "category",
            "rollnumber": np.int32,
            "icpsr": np.int32,
            "cast_code": np.int8,
        },
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        cast = chunk["cast_code"].to_numpy()
        vote = np.zeros(len(chunk), dtype=np.int8)
        vote[(cast >= 1) & (cast <= 3)] = 1
        vote[(cast >= 4) & (cast <= 6)] = -1
        chunk = chunk.assign(vote=vote)
        chunk = chunk[chunk["vote"] != 0]
        for (chamber, congress), g in chunk.groupby(["chamber", "congress"], observed=True):
            parts.setdefault((str(chamber), int(congress)), []).append(
                g[["icpsr", "rollnumber", "vote"]].reset_index(drop=True)
            )
    manifest: dict = {"keys": [], "rows": {}}
    for (chamber, congress), frames in sorted(parts.items()):
        df = pd.concat(frames, ignore_index=True)
        key = f"{chamber}_{congress:03d}"
        df.to_parquet(PROCESSED_DIR / f"{key}.parquet", index=False)
        manifest["keys"].append(key)
        manifest["rows"][key] = int(len(df))
    MANIFEST.write_text(json.dumps(manifest, indent=1))
    return manifest


def available_congresses(chamber: str) -> list[int]:
    manifest = build_cache()
    out = []
    for key in manifest["keys"]:
        cham, num = key.rsplit("_", 1)
        if cham == chamber:
            out.append(int(num))
    return sorted(out)


def load_votes(chamber: str, congress: int) -> pd.DataFrame:
    """Long-format cast votes for one (chamber, congress): icpsr, rollnumber, vote."""
    return pd.read_parquet(PROCESSED_DIR / f"{chamber}_{congress:03d}.parquet")


def build_vote_matrix(
    votes: pd.DataFrame, min_votes: int = 20, max_lopsided: float = 0.025
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long ±1 votes -> dense member x rollcall matrix (int8; 0 = not cast).

    Drops near-unanimous rollcalls (minority side < max_lopsided of casts —
    they carry no discriminative signal) and then members with fewer than
    min_votes casts on the surviving rollcalls (both are standard Voteview
    conventions). Returns (icpsr ids, matrix, n_votes per member).
    """
    tot = votes.groupby("rollnumber").size()
    yea = (
        votes.loc[votes["vote"] == 1]
        .groupby("rollnumber")
        .size()
        .reindex(tot.index, fill_value=0)
    )
    minority = np.minimum(yea, tot - yea) / tot
    votes = votes[votes["rollnumber"].isin(minority.index[minority >= max_lopsided])]
    n_votes = votes.groupby("icpsr").size()
    votes = votes[votes["icpsr"].isin(n_votes.index[n_votes >= min_votes])]
    if votes.empty:
        raise ValueError("no votes survive the rollcall/member filters")

    ids = np.sort(votes["icpsr"].unique()).astype(np.int64)
    rolls = np.sort(votes["rollnumber"].unique())
    V = np.zeros((len(ids), len(rolls)), dtype=np.int8)
    ri = np.searchsorted(ids, votes["icpsr"].to_numpy())
    ci = np.searchsorted(rolls, votes["rollnumber"].to_numpy())
    V[ri, ci] = votes["vote"].to_numpy()
    return ids, V, n_votes.loc[ids].to_numpy()
