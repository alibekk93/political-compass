# Data

Nothing in this directory is tracked in git (only this file) — the raw vote file
alone is ~670 MB.

## What to download

From [voteview.com/data](https://voteview.com/data), select **Congress**,
"HSall" bulk files, at the **roll call level**, and place the CSVs here:

| file | size | used for |
|------|------|----------|
| `HSall_votes.csv` | ~670 MB | **required** — 26.4M member-vote records, the only input to anything fitted |
| `HSall_members.csv` | ~6 MB | optional — joined after fitting for names, states and parties; orients dimension 1 and labels blocs |
| `HSall_parties.csv` | ~60 KB | optional — party code → party name for the same labels |

Only `HSall_votes.csv` is needed to run the pipeline; without the other two the
output is simply unlabeled. The `prob` column in the votes file and the
`nominate_*` / `nokken_poole_*` columns in the members file are outputs of
Voteview's own NOMINATE model and are deliberately never read — using them would
make the label-free claim circular.

Direct download links (the files Voteview's bulk-download page serves):

```bash
for f in votes members parties; do
  curl -fL -o data/HSall_$f.csv https://voteview.com/static/data/out/$f/HSall_$f.csv
done
```

## Generated

`data/processed/` — one parquet per (chamber, congress) plus `manifest.json`,
built on the first run by `python -m political_compass.pipeline`. Delete the
directory or pass `--force-cache` to re-parse.

## Citation

Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and
Luke Sonnet. *Voteview: Congressional Roll-Call Votes Database.*
https://voteview.com/
