# Political Compass — a common voting space for all 119 Congresses

Clusters members of the US Congress by voting behavior in a 2-D space that is
**comparable across congresses (1789–2027)**, built only from the Voteview vote
records — no names, parties, or issue labels enter any fitted step. Member
metadata is joined only afterward, to name blocs and orient dimension 1.

The obstacle: each congress votes on different rollcalls, so vote matrices from
different eras share no columns. The solution: scale each congress separately,
then stitch the spaces together using **members who serve in multiple congresses
as anchors** (Voteview's `icpsr` id is stable across a member's career). Adjacent
congresses share plenty of members (House 118→119: 375 of ~450; even 1869: 121),
so every link in the chain is well-determined.

📄 **[Read the manuscript](paper/manuscript.md)** — full write-up with figures and
results. This README covers the method and how to run it.

## Method

1. **Preprocess** (`political_compass/data_io.py`) — one-time parse of `data/HSall_votes.csv`
   (26.4M rows) into per-(chamber, congress) parquet caches. Yea variants → +1,
   Nay variants → −1, everything else missing; near-unanimous rollcalls
   (minority < 2.5%) and members with < 20 votes dropped (Voteview conventions).
2. **Per-congress scaling** (`political_compass/scaling.py`) — member×member *agreement
   distances* (1 − share of agreements on jointly cast rollcalls; robust to
   missing votes, no imputation) → classical MDS → 2-D coordinates, plus the
   top-10 eigenvalue spectrum as a dimensionality diagnostic.
3. **Alignment** (`political_compass/alignment.py`) — each congress's cloud is normalized
   (centered, RMS radius 1), then chained backward from congress 119 with
   orthogonal Procrustes fits on shared-member anchors (a member's target is
   their position in the nearest later congress). A generalized-Procrustes
   refinement pass then re-fits every congress against per-member consensus
   positions until convergence, removing accumulated chain drift.
4. **Clustering** (`political_compass/clustering.py`) — per-congress KMeans (k chosen by
   silhouette with a parsimony rule: smallest k within 0.02 of the best, and
   k > 2 solutions must have no cluster under 5% of members). Cluster labels are
   made continuous over time by Hungarian matching on shared members →
   persistent **bloc lineages** discovered without party labels.
5. **Labeling, post hoc** (`political_compass/metadata.py`) — only after all of
   the above, `HSall_members.csv` is joined on (congress, chamber, icpsr) to add
   names, states and parties. Dimension 1 is reflected so Republicans sit on
   the positive side in congress 119 — a global reflection, so no distance,
   cluster or lineage changes. Each bloc lineage is labeled by its dominant
   party per congress, run-length encoded over its life, with a purity score.
   Dimension 2's sign stays conventional. The file's NOMINATE columns are never
   read.

## Scaling and alignment in detail

### Scaling: turning one congress's votes into a map

Every congress starts as a member × rollcall matrix: +1 for Yea, −1 for Nay,
0 when the member didn't cast that vote. A toy congress with four members and
six rollcalls (`—` = not cast):

| member | r1 | r2 | r3 | r4 | r5 | r6 |
|--------|----|----|----|----|----|----|
| A      | +  | +  | −  | +  | −  | −  |
| B      | +  | +  | −  | +  | +  | −  |
| C      | −  | −  | +  | −  | +  | +  |
| D      | +  | −  | +  | —  | −  | +  |

Two filters run first, because some rows and columns carry no signal: a 98–2
rollcall makes nearly every pair of members "agree" and just shrinks all
distances uniformly, so near-unanimous rollcalls (minority side < 2.5%) are
dropped; members with < 20 cast votes are dropped because their distances to
everyone else would rest on a handful of observations.

**Agreement distance.** For each pair, look only at rollcalls *both* cast, and
define distance = 1 − (share of those where they voted the same way):

- A vs B: both cast all six, agree on five (they split only on r5) →
  d = 1/6 ≈ **0.17** — near-allies.
- A vs C: both cast all six, agree on none → d = **1.00** — perfect opposites.
- A vs D: D skipped r4, so only five rollcalls count; they agree on r1 and r5 →
  d = 3/5 = **0.60**. The denominator shrank instead of pretending the missing
  vote was information — this is why no imputation is needed.

Doing this for all pairs gives a symmetric distance matrix ([scaling.py](political_compass/scaling.py)
`agreement_distance`, computed with two matrix multiplications):

|   | A    | B    | C    | D    |
|---|------|------|------|------|
| A | 0    | 0.17 | 1.00 | 0.60 |
| B | 0.17 | 0    | 0.83 | 0.80 |
| C | 1.00 | 0.83 | 0    | 0.40 |
| D | 0.60 | 0.80 | 0.40 | 0    |

(The rare pair with *zero* jointly cast rollcalls — a member who died and their
replacement — gets the mean of its rows' defined distances.)

**Classical MDS.** Now find points on a plane whose pairwise straight-line
distances reproduce that matrix as closely as possible. Classical (Torgerson)
MDS does this non-iteratively: square the distances, double-center the matrix
(subtract row and column means — this converts distances into inner products),
and take the top eigenvectors, each scaled by the square root of its
eigenvalue ([scaling.py](political_compass/scaling.py) `classical_mds`). For the toy matrix:

| member | dim1  | dim2  |
|--------|-------|-------|
| A      | −0.43 | +0.15 |
| B      | −0.36 | −0.24 |
| C      | +0.52 | −0.17 |
| D      | +0.27 | +0.26 |

Dimension 1 alone separates the {A, B} camp from {C, D}, and the point
distances echo the table (A–C comes out at exactly 1.00, B–D at 0.80). The
eigenvalues here are (0.66, 0.18, 0.00, −0.11): the first dimension carries
most of the "distance mass", and the small negative one is normal — agreement
distances aren't exactly Euclidean, so trailing eigenvalues can dip below zero
and are simply not used. The eigenvalue *spectrum* is itself a finding: in the
real House, dimension 1 carries ~90% of the top-10 eigenvalue mass in the
2020s but only ~64% mid-century, when a genuine second dimension (the
conservative-coalition / civil-rights axis) was live.

### Alignment: putting 119 maps into one frame

MDS output has a built-in indeterminacy: distances don't change if the whole
map is rotated, reflected, or translated, so the solver returns an *arbitrary*
orientation. Scale each congress independently and one may come out with its
left bloc on the right, or the whole cloud turned 40°. Raw `dim1` in congress
60 and raw `dim1` in congress 61 are not the same axis, which is the whole
obstacle to cross-era comparison. Each cloud is first normalized — centered,
RMS radius 1 ([alignment.py](political_compass/alignment.py) `normalize_cloud`) — because raw MDS
scale isn't comparable across eras either; then the orientations are fixed
using the members two congresses share.

**Orthogonal Procrustes, by example.** Suppose three returning members have
these coordinates in a not-yet-aligned congress, and these already-aligned
positions from the congress after it:

| anchor | this congress (unaligned) | target (from later congress) |
|--------|---------------------------|------------------------------|
| M1     | (+0.9, +0.2)              | (−0.9, +0.2)                 |
| M2     | (−0.7, +0.4)              | (+0.7, +0.4)                 |
| M3     | (−0.2, −0.6)              | (+0.2, −0.6)                 |

Every target is the source with dim1 negated — this congress came out
mirror-imaged. Orthogonal Procrustes finds the rotation/reflection R
minimizing the summed squared anchor-to-target distance; here it recovers the
mirror exactly:

```
R = [[-1, 0],
     [ 0, 1]]
```

The transform is then applied to **every** member of the congress, so a
non-anchor freshman at (0.5, −0.5) rides along to (−0.5, −0.5). With real,
noisy anchors R is the least-squares compromise across the whole pool — one
member who genuinely moved is outvoted by the dozens to hundreds who didn't
(House anchor pools: minimum 46, median ~280).

**The chain.** Congress 119 is the reference frame. Congress 118 is aligned to
it, 117 to the result, and so on back to 1789 ([alignment.py](political_compass/alignment.py)
`chain_align`). A member's target is their position in the *nearest later*
congress they appear in, so the anchor pool for congress t is the union of
everyone serving in any later congress — members returning after a gap still
help. This leans on one assumption: returning members mostly keep voting like
themselves. Conveniently, Voteview issues a new `icpsr` to party switchers, so
the people most likely to violate the assumption are excluded automatically.
The mean leftover anchor-to-target distance after each fit is logged as
`anchor_resid` — it measures how much returning members actually moved, and it
spikes exactly where history says it should (peak 0.94 at congress 14, the
1815 Federalist collapse; ~0.1 in the stable modern era).

**Refinement.** A 118-link chain accumulates error like a game of telephone:
each fit is off by a little noise, and by congress 30 the frame can be
slightly twisted relative to congress 119. The generalized-Procrustes pass
([alignment.py](political_compass/alignment.py) `gpa_refine`) removes this: compute every
member's *consensus* position (their mean across all congresses served), re-fit
each congress against its members' consensus positions, recompute, repeat.
This uses every multi-congress overlap simultaneously — congress 40 is now
tied to congress 45 through a five-term member directly, not only through the
chain. Convergence behaves like diffusion along the chain (the slow mode is
the whole timeline gently "unbending"), needing ~2,900 iterations — cheap once
vectorized, and stopping early matters: at iteration 100 the worst-placed
congress is still 0.33 RMS-units from its converged position. Because the updates are
rotations and translations only, each congress's internal shape is preserved
exactly — refinement can't distort a congress, only re-orient it. Finally the
frame is re-pinned to congress 119's orientation and signs are fixed
deterministically (`fix_signs`); dimension 1's sign is then set by party in
the labeling step.

The synthetic test ([tests/test_synthetic.py](tests/test_synthetic.py)) exercises
exactly this machinery with known answers — simulated ideal points, simulated
votes, full pipeline — and recovers the truth at r = 0.99.

## Run it

Get the data from [voteview.com/data](https://voteview.com/data) (Congress:
"HSall" bulk files, roll call level) and place these three CSVs in `data/`
(not tracked in git — 700MB+; see [data/README.md](data/README.md) for details):

- `HSall_votes.csv` — the only file the fitting reads
- `HSall_members.csv`, `HSall_parties.csv` — joined after fitting to name
  members and blocs and orient dimension 1 (optional: without them the run
  completes unlabeled)

Dependencies are tracked in `pyproject.toml` — install with
`pip install -e .` (add `.[dev]` for pytest, jupyter, and plotly + ipywidgets
for the notebook dashboard).

```bash
python -m political_compass.pipeline --chamber both     # House and Senate (~1.5 min total)
python -m tests.test_synthetic            # end-to-end recovery on simulated data
python -m tests.test_split_half           # split-half reliability on real data
```

Tables land in `output/` (git-ignored, regenerate by re-running):

- `positions_{chamber}.csv` — one row per member-congress:
  `congress, chamber, icpsr, dim1, dim2, cluster, bloc, n_votes`, plus
  `state_abbrev, party_code, bioname, party` when the member file is present
- `blocs_{chamber}.csv` — one row per bloc lineage: span, size, `label`
  (dominant-party sequence, e.g. `Jackson → Democrat`), `purity` (share of
  its member-congresses in that congress's dominant party) and `segments`
  (the same sequence with congress ranges)
- `diagnostics_{chamber}.csv` — per congress: members/rollcalls kept, eigenvalue
  spectrum, chosen k, k=2 silhouette and cluster separation (polarization
  proxies), anchor count and post-alignment anchor residual

Figures land in [paper/figures/](paper/figures/) — tracked in git, so the
manuscript renders on GitHub: small-multiple maps of the space over time,
polarization timeline, anchor-residual timeline, eigenvalue heatmap, career
trajectories, for both chambers. Override either location with `--outdir` /
`--figdir`.

[mappings/](mappings/) (tracked) assigns every roll call, 1789–2027, to Clausen's
six issue areas, filling the post-2015 gap in Clausen coding from CRS policy
areas. Rebuild it with `python -m political_compass.issue_mappings` (needs
`data/HSall_rollcalls.json`); nothing in the pipeline reads it yet.

`python -m political_compass.issue_stats` (optional, needs the member file and
`data/HSall_rollcalls.json`) adds `issue_stats_{chamber}.csv` (per member-congress:
share of votes against own party, overall and per Clausen area) and
`agenda_{chamber}.csv` (per congress: roll calls per area) to `output/`.

`notebooks/explore.ipynb` is a thin viewer over these files, ending in an
interactive plotly dashboard: the map or either dimension over time, filtered
by chamber, years/congress, party and state, with member name search and a
table of the rows shown. With the issue tables present it also colors members
by how often they vote against their party (overall or per issue area), shows
what each congress voted on, compares up to six members (picked by search or by
clicking dots) on a whole-career issue radar and profile bars, and lists the
members most like a chosen one within the selected years.

## Validation

- **Synthetic end-to-end**: simulate known 2-D ideal points (two blobs, 25%
  turnover per congress, slow drift), generate votes from random cutting planes,
  run the full pipeline → recovered positions correlate with truth at
  **r = 0.99**; clusters and lineages recovered exactly.
- **Split-half reliability**: scaling odd- vs even-indexed rollcalls of a
  congress separately gives near-identical positions (mean r = 0.96 across six
  eras in both chambers, worst 0.90).
- **History checks (no labels used)**: the polarization U-curve emerges (high
  ~1900, low 1940s–70s, extreme today — House k=2 silhouette 0.9 in the 2020s);
  Senate lineage breaks land on Reconstruction (1873–75) and the New Deal
  (1935–37), and the House's on 1949; anchor residuals spike at 1815–1823
  (Federalist collapse) and decay to ~0.1 today; dimension 1 carries 90% of the
  top-10 eigenvalue mass in the modern House vs 64% mid-century.
- **Post-hoc labels** (`blocs_{chamber}.csv`): every major lineage is a party
  line (purity 0.79–0.98), but which party is not readable from dates alone.
  The House lineage unbroken since 1789 is Federalist → Whig → Republican (the
  Republican founding passes without a break), and the 1949 break is the
  Democratic line splitting North/South. Against DW-NOMINATE's dimension 1
  (comparison only, never read by the pipeline), the oriented dimension 1
  correlates at 0.98 / 0.97 (House / Senate) since 1981.

## Caveats

- **Chambers are separate spaces.** House and Senate never vote on the same
  rollcalls within a congress, so each chamber gets its own common space;
  unifying them via chamber-switchers is future work.
- **Only dimension 1 is oriented, and only exactly in the modern era.** Its
  positive side is the side Republicans occupy in congress 119, carried back
  through the common frame. Further back the direction of party conflict
  turns within the frame: ~20–35° off dimension 1 in the House after 1825,
  30–50° in the Senate after 1861, and past 90° in the 1821–1859 Senate, where
  party must be read as a direction in the plane. Dimension 2's sign is a
  deterministic convention with no meaning.
- **Presidents are in the data.** The vote file records presidents' announced
  positions under both chambers' rollcalls, so they are scaled as extra rows
  (49 House and 38 Senate president-congresses).
- **Party switchers** get a new `icpsr` from Voteview, so they drop out of the
  anchor pool across the switch (harmless — their position genuinely jumps).
- Short-lived small blocs are mostly sectional splits within a party (Northern
  vs Southern Democrats in 1945–47 and 1957–61; both parties in 1849–53) or
  clustering noise; check their `label` and `purity` in `blocs_{chamber}.csv`
  before interpreting.

## Future extensions

- Unified House+Senate space through members who switched chambers.
- Joint sparse-matrix factorization as a cross-check on the stitched space.
- Issue interpretation of dimensions via `HSall_rollcalls.csv` metadata.
