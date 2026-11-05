# EnvTok

EnvTok selects a smaller set of complete atomistic structures from a large
database, with explicit, independently audited diversity guarantees.

It tokenises every atom's local chemical environment into an interpretable
**EnvTok** token. The token records the element, the magnetic moment, the
neighbour multiset, the ChemEnv polyhedron and its distortion, compression,
and asymmetry. Rare tokens are coarsened into coverage classes until each
class is supported by at least *T* distinct structures. EnvTok then picks a
low-atom-cost subset of original configurations covering every class and
every rare (element, geometry, coordination, compression, asymmetry)
combination.

An opt-in extension adds each structure's net charge to its tokens, so
distinct charge states are separate environments for as long as their support
allows (see [Charged structures](#charged-structures)).

The default profile `reference-723696d` reproduces the method of Moerman *et
al.*, *Atoms that Matter* (arXiv:2610.14392). It follows the reference
implementation
[`em819/omat24-chemenv-subset@723696d`](https://github.com/em819/omat24-chemenv-subset/tree/723696de2c26584b8a907ef900690c0caf3c7b68).
Every token of the golden fixture corpus matches that code running in its own
pinned environment. See [`docs/specification.md`](docs/specification.md) and
[`docs/discrepancies.md`](docs/discrepancies.md).

> A diversity cover does not preserve the sampling distribution, and it does
> not guarantee equivalent model accuracy. EnvTok reports compression by
> structures *and* by atoms, and validate the trade-off on your own data.
> The ~1% retention reported in the paper belongs to its OMat24 experiment.

## Install

EnvTok needs Python 3.14 and [uv](https://docs.astral.sh/uv/) 0.13 or newer.

```bash
uv sync
```

To read OMat24 ASE-LMDB shards, install the LMDB backend. fairchem and torch
are not needed.

```bash
uv pip install "envtok[omat24]"
```

## Quick start

Run the pipeline on the bundled 20-structure synthetic corpus:

```bash
uv run envtok run --config examples/quickstart/config.yaml
```

Or use the Python API:

```python
from envtok import Corpus, CoverageBuilder, Selector, Tokenizer

corpus = Corpus.from_source("omat24.yaml")  # or Corpus.from_extxyz / from_atoms
cache = Tokenizer(profile="reference-723696d").run(corpus, "runs/omat24", workers=16)
classes = CoverageBuilder(min_structure_support=100).fit(cache, split="train")
result = Selector(mode="cover", cost="atoms").select(classes)
result.validate()  # independent audit; raises CoverageError if anything is uncovered
result.write_manifest("runs/omat24/selection")
result.export("runs/omat24/selection/subset.extxyz", format="extxyz")
```

## Command-line interface

| Command | Purpose |
|---|---|
| `envtok inspect --config C [--sample N]` | validate a sample of records before a large run |
| `envtok tokenize --config C [--shard-index i --num-shards n]` | compute or resume the token cache; shards map to SLURM array tasks |
| `envtok index --config C` | merge committed partitions into the global token index |
| `envtok select --config C [--min-support T]` | fit classes, select, audit and write the manifest |
| `envtok run --config C` | run every stage and export the subset |
| `envtok sweep --config C --min-support 30 100 300` | select for several values of T, reusing one token cache |
| `envtok report --run DIR` | re-run the independent audit of a selection |
| `envtok export --run DIR [--format extxyz\|asedb] [--output PATH]` | export the selected original structures |
| `envtok baseline --run DIR --seeds 0 1 2` | equal-atom random baselines, audited against the same classes |

See [`examples/omat24/omat24.yaml`](examples/omat24/omat24.yaml) for a
complete configuration and
[`examples/slurm/tokenize_array.sbatch`](examples/slurm/tokenize_array.sbatch)
for array-job tokenisation.

## Molecules (experimental)

The separate `molecular_v1` profile selects intact finite molecules while
protecting total charge, spin multiplicity, electronic state and reference
method. It combines rooted graph tokens (or an explicit graph-unknown fallback)
with local SOAP geometry, conservative deduplication, and hard atom budgets.
Unknown state is never silently converted to neutral/singlet.

```bash
uv sync --extra molecular
uv run envtok run --config examples/molecular/omol/config.yaml
uv run envtok benchmark --config examples/molecular/omol/config.yaml --seeds 0 1 2
```

The bundled [OMol sample](examples/molecular/omol/README.md) contains 29 real
training configurations from a documented public mirror. It exercises the
pipeline and selector comparisons; downstream model accuracy is unvalidated.
See [molecular usage and data contracts](docs/molecular.md) for the Python API,
staged CLI, supplied-graph schema, safeguards and current limitations.

## Inputs

EnvTok reads:

- ASE iterables, including sequences, factories and one-shot iterators (which
  can be spooled to a managed extxyz store);
- extended XYZ, plain or compressed (`.gz`, `.bz2`, `.xz`);
- ASE databases;
- OMat24 ASE-LMDB shards, read natively.

Record IDs are namespaced by dataset, revision, shard and native key, so an
OMat24 `sid` is never assumed unique. Every discovered record is accounted
for as eligible, excluded or failed; nothing is skipped silently. The
`omat24-study` preset reproduces the study's La–Lu/Ac–Lr exclusion. The
default `general` preset excludes nothing.

## Charged structures

The reference token ignores charge. EnvTok offers two ways to keep charge
states apart:

- **Protected strata** (`selection.stratify_by: [charge]`) fit and cover each
  charge state separately, so every state gets a complete cover of its own.
- **The net-charge token field** adds the record's net charge, rounded with
  its sign kept, to every site token (`Fe_0_q-1_Cl-O-O-O_T:4_3_1_2`). Charge
  distinguishes environments at the two finest rungs. A charge state too rare
  to support its own class merges with the other states of its environment
  at the next rung, `R1q_drop_charge`.

```yaml
tokenizer:
  charge_policy: net_round      # or net_round_zero_if_missing; default disabled
  charge_key: charge            # the info / key-value key holding the net charge
```

```python
from envtok import Tokenizer, get_profile, with_net_charge

profile = with_net_charge(get_profile("reference-723696d"), "net_round", key="charge")
cache = Tokenizer(profile).run(corpus, "runs/charged")
```

Under `net_round`, a record without a numeric charge fails with reason
`missing_charge`; `net_round_zero_if_missing` treats it as neutral and counts
it as defaulted. The charge is read from the cached metadata when the index
is built, so switching the field on or off never repeats the geometry pass.
Run `envtok inspect` first to check that the key is present. See
[§1.7 of the specification](docs/specification.md#17-net-charge-extension-opt-in).

## Outputs

Each selection goes to its own configuration-addressed directory
`<workdir>/selection-T<T>-<fingerprint>/`, which contains:

- `selected_ids.txt` and `selected_manifest.parquet`: source locators, atom
  counts, metadata and selection reasons;
- `config.resolved.yaml` and `provenance.json`: versions and layered cache
  fingerprints;
- `coverage.json` and `report.html`: the independent audit and its readable
  report;
- the frozen class mapping;
- optionally `subset.extxyz`, exported from the original sources with labels
  preserved and no live calculator.

The report covers:

- structure and atom retention;
- class, full-token and frequency-weighted full-token coverage;
- presence coverage per rare-presence family;
- resolution loss per rung;
- processing quality;
- held-out out-of-vocabulary rates;
- input versus selected distributions.

## Development

```bash
uv run prek install
uv run prek run --all-files
uv run mypy
uv run pytest
```

The test suite enforces 100% branch coverage. Golden reference fixtures are
regenerated with the pinned reference environment described in
[`build_tools/reference/README.md`](build_tools/reference/README.md).

## Citing

If you use EnvTok, please cite the method paper and the reference
implementation:

- E. Moerman, I. Poltavsky, A. Charkin-Gorbulin, A. Tkatchenko, *Atoms that
  Matter: Largest Materials Datasets Are 99% Redundant, and Local Chemistry
  Can Guide the Required 1%*, arXiv:2610.14392 (2026).
- E. Moerman, *omat24-chemenv-subset* (software, v1.0.0, MIT licence),
  https://github.com/em819/omat24-chemenv-subset.

## License

EnvTok is released under the [MIT License](LICENSE). Code adapted from
omat24-chemenv-subset keeps its MIT notice in
[`LICENSES/omat24-chemenv-subset-MIT.txt`](LICENSES/omat24-chemenv-subset-MIT.txt).
