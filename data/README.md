# Data

Nothing in this directory is tracked in git (only this file) — the raw vote file
alone is ~670 MB.

## What to download

From [voteview.com/data](https://voteview.com/data), select **Congress**,
"HSall" bulk files, at the **roll call level**, and place the CSVs here:

| file | size | used for |
|------|------|----------|
| `HSall_votes.csv` | ~670 MB | **required** — 26.4M member-vote records, the only input the pipeline reads |
| `HSall_members.csv` | ~6 MB | not read yet; for the deferred name/party labelling of blocs |
| `HSall_parties.csv` | ~60 KB | not read yet; same |

Only `HSall_votes.csv` is needed to run the pipeline. The `prob` column in it is
an output of Voteview's own NOMINATE model and is deliberately never read — using
it would make the label-free claim circular.

## Generated

`data/processed/` — one parquet per (chamber, congress) plus `manifest.json`,
built on the first run by `python -m political_compass.pipeline`. Delete the
directory or pass `--force-cache` to re-parse.

## Citation

Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and
Luke Sonnet. *Voteview: Congressional Roll-Call Votes Database.*
https://voteview.com/
