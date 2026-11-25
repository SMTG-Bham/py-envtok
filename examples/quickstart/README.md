# Read a periodic selection report

Use this example to follow one selection from the eligible structures to the
exported subset. The [configuration](config.yaml) uses the bundled
[20-structure corpus](../../tests/fixtures/corpus.extxyz) and sets the
minimum support to two distinct records. Run from the repository root.

```bash
config=examples/quickstart/config.yaml
uv run envtok inspect --config "$config" --sample 20
uv run envtok run --config "$config"
```

`inspect` finds 20 training records with stored energies and forces and zero
duplicate IDs. `run` prints the selection directory as the JSON field
`selection`. Open `report.html` in that directory for the summary, or read
`coverage.json` for the counts below.

## What did the cover retain?

| Question | Field in `coverage.json` | Result on the fixture |
|---|---|---|
| Which records entered selection? | `processing.records_by_status`, `processing.eligible_by_policy` | All 20 have status `ok` and are eligible under the configured quarantine policy. |
| How much data was kept? | `structures`, `atoms` | 16/20 configurations (80%) and 56/76 atoms (73.7%). |
| Which guarantees were met? | `classes`, `presence` | 26/26 fitted classes. All signatures in each of the four presence families have a selected witness. |
| How much fine detail survived? | `full_tokens` | 32/33 distinct full tokens. Those 32 token types occur at 75/76 eligible sites. |

One structure needed CrystalNN after ChemEnv failed. Its two recovered sites
appear in `processing.sites`, and the structure still counts toward the
eligible pool. For your own data, check `processing.failures` for the stage,
fallback and fatal flag before interpreting retention.

The 56 retained atoms represent token types that occur at 75 sites in the
source. When the subset contains one example of a full token, **all**
eligible sites with that token contribute to `full_tokens.frequency_weighted`.

## Why was a particular structure kept?

`selected_manifest.parquet` has one row per selected configuration in export
order. Inspect `record_id`, `natoms` and `reason`. For example, from a Python
session opened in the selection directory, run

```python
import pyarrow.parquet as pq

manifest = pq.read_table(
    "selected_manifest.parquet", columns=["record_id", "natoms", "reason"]
)
print(manifest.to_pylist()[0])
```

On the fixture, nine records are sole suppliers of a class (`mandatory`),
four enter during greedy cover (`cover`), and three supply signatures still
missing after pruning (`repair:...`). To retain a specific eligible
configuration in your own run, put its complete `record_id` in
`selection.required_records`. The selector records `user` as its reason.

## Compare against random sampling

Set `run_dir` to the `selection` path printed by `run`. The baseline draws
whole records in seeded random order up to the same 56-atom ceiling.

```bash
uv run envtok baseline --run "$run_dir" --seeds 0 1 2
```

Read `baselines/summary.json` for the achieved cost and coverage of each draw.
Seed 1 keeps 55 atoms and covers 22 of the 26 classes. For the missing
signatures, open `baselines/random-seed1/coverage.json` and look under
`classes.missing_examples` and each `presence` family.

For a different threshold, `envtok sweep` writes `sweep_summary.json` in
the work directory. Each row's `directory` points to its selection report.
Compare atom retention and full-token coverage across rows. The [OMat24
example](../omat24/compression-demo/README.md) uses the same comparison on
1,024 structures with distinct parent IDs.
