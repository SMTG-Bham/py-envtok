# EnvTok

EnvTok selects a smaller set of complete atomistic structures from a large
database, with explicit, independently audited diversity guarantees.

The 0.1 series is alpha software: APIs and saved-run formats may change between
releases. See the [release notes](https://github.com/SMTG-Bham/py-envtok/blob/main/CHANGELOG.md) for the initial release's scope
and limitations.

It tokenises every atom's local chemical environment into an interpretable
**EnvTok** token. The token records the element, the magnetic moment, the
neighbour multiset, the ChemEnv polyhedron and its distortion, compression,
and asymmetry. Rare tokens are coarsened into coverage classes targeting
support from at least *T* distinct structures, with a terminal per-element
class for environments that remain too rare. EnvTok then picks a
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
pinned environment. See [`docs/specification.md`](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/specification.md) and
[`docs/discrepancies.md`](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/discrepancies.md).

> A diversity cover does not preserve the sampling distribution, and it does
> not guarantee equivalent model accuracy. EnvTok reports compression by
> structures *and* by atoms. Validate the trade-off on your own data.
> The ~1% retention reported in the paper belongs to its OMat24 experiment.

Jump to [Quick start](#quick-start), [Configuration](#configure-your-dataset),
[CLI](#command-line-interface), [Molecules](#molecules-experimental),
[Inputs](#inputs), [Outputs](#outputs) or [Troubleshooting](#troubleshooting).

## How selection works

1. **Tokenise the source.** Validate each record, compute its local environments,
   and cache the measurements. Selection always keeps or removes a whole
   original configuration; it never extracts isolated atoms from a structure.
2. **Fit coverage classes.** Count support in distinct eligible training
   records, then progressively drop token detail according to the profile's
   coarsening ladder. Repeated equivalent sites within one structure count
   only once toward support.
3. **Select and repair.** Start with required records and sole suppliers of a
   class, greedily add coverage per atom, prune redundant structures, then
   repair missing rare-presence signatures. The result is deterministic for
   the same input and settings, but is not a proof of minimum atom cost.
4. **Audit independently.** Recompute coverage from the selected IDs and the
   frozen class mapping. The report distinguishes class coverage, original
   full-token coverage, processing quality, and the atoms and structures kept.

The coverage guarantee applies to the fitted classes and configured presence
constraints over the eligible pool. It does not mean every original full token
is retained. Held-out records do not influence the classes; unseen held-out
tokens are reported as out of vocabulary.

For periodic materials, start with the reference profile below. For finite
molecules with charge, spin and geometry protections, use the separate
[experimental molecular workflow](#molecules-experimental).

## Install

Use standard CPython 3.11–3.14; the CI matrix covers these interpreters on Linux.
Other Python versions, operating systems and free-threaded builds are outside
the qualification scope.
The supported development environment uses
[uv](https://docs.astral.sh/uv/) 0.13.6 or newer. From a source checkout:

```bash
git clone https://github.com/SMTG-Bham/py-envtok.git
cd py-envtok
uv sync --locked
uv run envtok --version
```

The commands below assume you are in the repository root. Use `uv sync --locked`
to require the committed dependency lockfile. The default sync also installs
the development tools; add `--no-dev` if you only need to run the package.
The checkout defaults to Python 3.14 via `.python-version`. To use another
supported interpreter, pass it explicitly to uv, for example
`uv sync --locked --python 3.11` and `uv run --python 3.11 envtok --version`.

To read OMat24 ASE-LMDB shards, install the LMDB backend. fairchem and torch
are not needed.

```bash
uv sync --extra omat24
```

The `molecular` extra adds RDKit and DScribe for the molecular workflow. Both
extras can be installed together with `uv sync --extra omat24 --extra molecular`.
An alternative [Conda setup](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/README.md#conda-environment) is also
provided.

To install a built release artifact into an existing Python 3.11–3.14 environment,
use `python -m pip install /path/to/envtok-0.1.0a1-py3-none-any.whl`. Optional
backends can be requested with
`python -m pip install '/path/to/envtok-0.1.0a1-py3-none-any.whl[molecular,omat24]'`.
The examples and fixture data below live in the source checkout; they are not
included in the installed Python package.

## Quick start

Inspect and run the pipeline on the bundled 20-structure synthetic corpus.
No dataset download or optional dependency is needed:

```bash
uv run envtok inspect --config examples/quickstart/config.yaml --sample 20
uv run envtok run --config examples/quickstart/config.yaml
```

The final command prints JSON containing the `selection` directory, under
`runs/quickstart/selection-T2-<fingerprint>/`. Open its `report.html` to review
coverage and retention; `subset.extxyz` contains the selected original
structures, and `selected_ids.txt` records their identities. Repeating the
command reuses committed tokenisation work.

The fixture is an engineering example, so it uses a small support threshold
of 2. Its retention rate is not a prediction for a real dataset.

### Python API

This example uses the same bundled corpus:

```python
from envtok import Corpus, CoverageBuilder, Selector, Tokenizer

corpus = Corpus.from_extxyz(
    "tests/fixtures/corpus.extxyz",
    dataset="fixtures",
    revision="v1",
    split="train",
)
cache = Tokenizer(profile="reference-723696d").run(corpus, "runs/quickstart-api")
classes = CoverageBuilder(min_structure_support=2).fit(cache, split="train")
result = Selector(mode="cover", cost="atoms").select(classes)
report = result.validate()  # raises CoverageError if required coverage is missing
output = result.write_manifest("runs/quickstart-api/selection")
result.export(output / "subset.extxyz", format="extxyz")
print(report["structures"], report["atoms"], report["classes"])
```

Use `Corpus.from_source("examples/quickstart/config.yaml")` to read the source
and corpus policy from YAML instead. The other factories are listed under
[Inputs](#inputs).

For a complete script that also checks exported geometry, atom order, metadata
and labels against the source:

```bash
uv run python examples/quickstart/quickstart.py
```

### Real OMat24 compression example

The [OMat24 compression example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/compression-demo/README.md)
uses 1,024 real structures from distinct parent materials. At T=100, it retains
755 structures and 5,077 of 6,705 atoms: **24.3% fewer atoms**, with every fitted
class and required presence signature covered. Detailed full-token coverage
is 76.7%; downstream model accuracy has not been tested.

Follow the example's sampling command to prepare the input from an existing
local OMat24 archive, then run:

```bash
uv run --extra omat24 envtok run --config examples/omat24/compression-demo/config.yaml
```

The example includes a threshold sweep, equal-atom random comparisons, saved
results and export checks. The [bundled 32-record sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/small-sample/README.md)
also runs without an archive, but shows no compression.

## Configure your dataset

Save a configuration such as `dataset.yaml` next to your `data/` directory:

```yaml
schema_version: 1
source:
  adapter: extxyz
  path: data/train.extxyz
  name: my-dataset
  revision: v1
  split: train

corpus:
  preset: general
  invalid_record_policy: quarantine

tokenizer:
  profile: reference-723696d

coverage:
  min_structure_support: 100
  split: train

selection:
  mode: cover
  cost: atoms

execution:
  workdir: runs/my-dataset
  workers: 4
  batch_size: 32
  resume: true

output:
  format: extxyz
```

Then run `uv run envtok inspect --config dataset.yaml`, followed by
`uv run envtok run --config dataset.yaml`. Input and work-directory paths are
resolved relative to the YAML file, not the shell's working directory.
Unknown configuration keys are rejected, and the resolved configuration with
all defaults is saved alongside the selection.

The main choices are:

| Setting | How to choose it |
|---|---|
| `source.revision` | Give the input a meaningful version label; it is part of each record's identity. |
| `source.split` / `source.split_key` | Use a constant split for a training-only source, or read per-record split metadata with `split_key`. |
| `coverage.split` | Fit and select on this split; the default is `train`. Match it to the source's split assignments. |
| `coverage.min_structure_support` | An integer *T* ≥ 2 controlling coarsening, not the number of structures to select or a retention percentage. |
| `corpus.invalid_record_policy` | `quarantine` excludes records with unusable local environments from selection; `reference` keeps them with sentinel tokens. |
| `selection.stratify_by` | Fit and cover metadata strata separately, for example `[charge, method]`, when these distinctions must survive selection. |
| `output.format` | `extxyz`, `asedb`, or `none` to write only the manifest and reports. |

Higher *T* asks for more source support before retaining a detailed class. Tokens
that never reach *T* enter a terminal per-element class even if its support is
lower. *T* does **not** require *T* selected representatives. The coarsening
ladder is not nested, so increasing *T* need not reduce selected atoms
monotonically; use a sweep and compare the reports.

The reference selector supports `mode: cover` with atom cost. Its
`mode: budget` is not implemented; hard atom budgets are available in the
separate molecular profile.

## Command-line interface

These are the reference-profile commands. Use `uv run envtok --help` or
`uv run envtok <command> --help` for arguments, and put `-v` before the command
to log progress, for example `uv run envtok -v run --config dataset.yaml`.

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

`--run` takes the **selection directory**, not the parent work directory.
Use the path printed by `run` or `select`. For example, replace `<fingerprint>`
with the suffix from your quick start:

```bash
RUN_DIR="runs/quickstart/selection-T2-<fingerprint>"
uv run envtok report --run "$RUN_DIR"
uv run envtok export --run "$RUN_DIR" --format asedb --output runs/quickstart/subset.db
uv run envtok baseline --run "$RUN_DIR" --seeds 0 1 2
```

`select` writes and audits a manifest; `run` also exports in the configured
format. `baseline` measures coverage of random subsets using the selection's
atom count as the budget; it does not measure downstream model accuracy.

### Restart, sweeps and parallel runs

The stages can be run separately on the same configuration:

```bash
uv run envtok tokenize --config dataset.yaml
uv run envtok index --config dataset.yaml
uv run envtok select --config dataset.yaml
uv run envtok sweep --config dataset.yaml --min-support 30 100 300
```

Tokenisation commits independent batches atomically. With
`execution.resume: true` (the default), rerun the same command after interruption
to reuse committed batches. `tokenize` with `resume: false` refuses committed
partitions. Selection and sweeps can still reuse a finished cache; use a new
work directory to force a fresh geometry run.

Changing *T* or selector settings reuses the token index and creates a separate
selection directory. Changing the net-charge token field reuses the geometry
measurements but builds a different token index. Changes to the source,
geometry settings, magnetic policy or batch size use a different geometry
cache. Keep original inputs available and unchanged for later export.

For local parallelism, set `execution.workers` and optionally
`max_inflight_batches` and `threads_per_worker`. `memory_budget` bounds the
estimated selection working state and configures DuckDB memory;
`temporary_disk_budget` limits DuckDB spill storage. These are not a global
memory cap for all geometry workers. `coverage.backend: duckdb` enables
external aggregation; `auto` selects it above 50 million incidence edges.

For a cluster, give each tokenisation task a distinct `--shard-index` from
0 to `--num-shards - 1`, all using the same configuration and shared work
directory. Wait for every task before running `index`, then `select` and
`export`.

See [`examples/omat24/omat24.yaml`](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/omat24.yaml) for a
complete configuration and
[`examples/slurm/tokenize_array.sbatch`](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/slurm/tokenize_array.sbatch)
for array-job tokenisation.

## Molecules (experimental)

The separate `molecular_v1` profile selects intact finite molecules while
protecting total charge, spin multiplicity, electronic state and reference
method. It combines rooted graph tokens (or an explicit graph-unknown fallback)
with local SOAP geometry, conservative deduplication, and hard atom budgets.
Unknown state is never silently converted to neutral/singlet.

This profile has its own configuration schema, selected by the top-level
`profile: molecular_v1`. Start from the bundled configuration:

```bash
uv sync --extra molecular
uv run --extra molecular envtok run --config examples/molecular/omol/config.yaml
uv run --extra molecular envtok benchmark --config examples/molecular/omol/config.yaml
```

To impose a hard atom cap, set exactly one of `atom_fraction` or `atom_budget`:

```yaml
selection:
  mode: budget
  atom_fraction: 0.25  # alternatively, replace this with atom_budget: 1000
```

The fraction is measured against atoms in the deduplicated training candidate
pool. For `mode: cover`, omit both budget settings. A budgeted run can finish
with `complete: false`: inspect `unmet_constraints` in `coverage.json` to see
what the selected subset does not cover. An incomplete heuristic cover is not
a proof that complete coverage is impossible within that budget.

Set `source.label_namespace` to identify the reference method, and map the
total-charge and spin-multiplicity fields with `state.charge_key` and
`state.multiplicity_key` (defaults: `total_charge` and `spin_multiplicity`).
Neither is inferred from partial charges or site magnetic moments. Unknown
states remain distinct, and the default strict training export requires known
charge and multiplicity.

Molecular runs write to `<workdir>/molecular-<fingerprint>/`. The CLI currently
reads one extended XYZ source and exports extended XYZ; the Python API also
accepts ASE iterables. This is a serial, in-memory pilot without worker-count
sharding. Its staged commands include `ingest`, `split`, `featurize`,
`build-targets`, `select`, `audit` and `export`; use `benchmark` for budget
comparisons rather than the reference profile's `sweep`.

The bundled [OMol sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/README.md) contains 29 real
training configurations from a documented public mirror. It exercises the
pipeline and selector comparisons; downstream model accuracy is unvalidated.
See [molecular usage and data contracts](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/molecular.md) for the Python API,
staged CLI, supplied-graph schema, safeguards and current limitations.

The [native OMol25 compression example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/compression-demo/README.md)
starts from 512 real configurations / 10,938 atoms. Unbudgeted cover mode
retains 146 configurations / 3,276 atoms (**70.0% fewer atoms**), covering
all 44 mandatory element/state targets and 517 modeled SOAP regions.
However, 91.3% of source environments fall outside the fitted prototype
radii; this is not complete geometric coverage. A finer 64-prototype check
retains 6,122 atoms (44.0% fewer). The example includes random and budget
comparisons, source/export checks, saved results and reproduction commands
for an existing local archive. It provides no downstream accuracy evidence.

Benchmark defaults are intentionally small: molecular comparisons use one 25%
atom budget and seed 0, plus a separate full-pool control. Larger sweeps are
opt-in through `--fractions` and `--seeds`. The [periodic plan](https://github.com/SMTG-Bham/py-envtok/blob/main/envtok_implementation_plan.md)
and [molecular plan](https://github.com/SMTG-Bham/py-envtok/blob/main/envtok_molecular_implementation_plan.md) describe the short
acceptance checks and remaining scientific work.

The [real OMat24 sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/small-sample/README.md) exercises native
storage, geometry, restart and export on 32 records. Small molecular pair-model
and from-scratch MACE comparisons are also runnable; see
[benchmark commands](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/README.md#small-acceptance-runs). Reports retain
failed accuracy screens and unsuitable evaluation data explicitly.

## Inputs

The reference workflow supports these sources:

| Input | YAML `source.adapter` | Python factory |
|---|---|---|
| ASE sequences, factories or iterators | Python API only | `Corpus.from_atoms(...)` |
| Extended XYZ, including `.gz`, `.bz2` and `.xz` | `extxyz` | `Corpus.from_extxyz(...)` |
| ASE databases | `asedb` | `Corpus.from_ase_db(...)` |
| OMat24 ASE-LMDB shards | `omat24` | `Corpus.from_omat24(...)` |

For a one-shot ASE iterator, pass `spool_to="runs/source.extxyz"` to
`Corpus.from_atoms(...)` to make the source retrievable for export and
resumable. Supply `name`, a meaningful `revision`, and `split="train"` when
building a training corpus. File-based factories use `dataset` instead of
`name`. By default, records are `unsplit`, whereas class fitting defaults to
`train`; set the split explicitly or intentionally fit `split="unsplit"`.

Record IDs are namespaced by dataset, revision, shard and native key, so an
OMat24 `sid` is never assumed unique. Every discovered record is accounted
for as eligible, excluded or failed; nothing is skipped silently. The
`omat24-study` preset reproduces the study's La–Lu/Ac–Lr exclusion. The
default `general` preset applies no element exclusions.

The reference profile uses source initial magnetic moments, rounds their
absolute values, and defaults missing moments to zero. Vector moments are
unsupported under that policy. Energy and force labels are not needed to
choose a diversity subset, but downstream training still needs suitable
labels. Stored energy, forces, stress and supported arrays are preserved on
export; no live calculator runs and the reference workflow does not convert
label units.

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
[§1.7 of the specification](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/specification.md#17-net-charge-extension-opt-in).

## Outputs

Each reference-profile selection goes to its own configuration-addressed
directory
`<workdir>/selection-T<T>-<fingerprint>/`, which contains:

- `selected_ids.txt` and `selected_manifest.parquet`: source locators, atom
  counts, metadata and selection reasons;
- `config.resolved.yaml` and `provenance.json`: versions and layered cache
  fingerprints;
- `coverage.json` and `report.html`: the independent audit and its readable
  report;
- the frozen class mapping;
- `run.json` for reloading a CLI selection;
- optionally `subset.extxyz` or `subset.db`, exported from the original sources
  with labels preserved and no live calculator.

The report covers:

- structure and atom retention;
- class, full-token and frequency-weighted full-token coverage;
- presence coverage per rare-presence family;
- resolution loss per rung;
- processing quality;
- held-out out-of-vocabulary rates;
- input versus selected distributions.

Start with `report.html` for a readable view and `coverage.json` for automated
checks. Read processing quality alongside coverage: a complete cover of the
eligible pool does not imply that every discovered input was usable. Compare
both atom and structure retention, since structures may differ greatly in size.

Molecular runs also write selected IDs, a manifest, resolved configuration,
provenance and reports. Their additional artifacts are `ingestion.json`,
`splits.json`, `targets.json` and `software-lock.json`. Reports include budget
compliance, unmet constraints and nearest-selected geometry distances. See the
[molecular output contract](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/molecular.md#commands-restart-and-outputs).

## Troubleshooting

| Symptom | What to check |
|---|---|
| Empty or unexpectedly small selection | Check source split assignments against `coverage.split`, exclusions, and failed or quarantined records in the report. |
| Missing charge failures | With `net_round`, the configured key must contain a numeric net charge. Use `inspect` to review charge sources. |
| Export cannot retrieve records | Keep the original sources available; spool one-shot iterators before tokenisation. |
| Indexing reports incomplete partitions | Finish or rerun every tokenisation shard with the same configuration before indexing. |
| Selection exceeds the configured memory policy | Review the estimate and available resources, then adjust `execution.memory_budget`; DuckDB aggregation does not remove the selector's own memory needs. |
| Molecular export rejects unknown state | Provide known total charge and multiplicity through the configured keys; see the molecular data contract for permissive export. |

## Documentation and examples

- [Specification](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/specification.md): token fields, coarsening ladder,
  selection algorithm, storage, audit and execution contracts.
- [Discrepancy log](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/discrepancies.md): differences between the paper,
  reference implementation and this package.
- [Molecular guide](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/molecular.md): data contracts, graphs, descriptors,
  safeguards and current limitations.
- [OMat24 compression example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/compression-demo/README.md):
  1,024 real structures, measured compression, coverage audits and random comparisons.
- [OMol25 compression example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/compression-demo/README.md):
  512 native molecular configurations, cover and budget comparisons, and geometry limits.
- [OMat24 sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/small-sample/README.md) and
  [OMol sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/README.md): real-data provenance and
  small qualification results.
- [Build and benchmark tools](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/README.md): acceptance runs,
  accuracy screens and packaging checks.
- [Release notes](https://github.com/SMTG-Bham/py-envtok/blob/main/CHANGELOG.md) and [release procedure](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/releasing.md):
  supported scope, artifact checks and publishing configuration.

## Development

Install the development environment and run the checks used by CI:

```bash
uv sync --locked
uv run prek install
uv run prek run --all-files
uv run mypy
uv run pytest
```

The test suite enforces 100% branch coverage. Golden reference fixtures are
regenerated with the pinned reference environment described in
[`build_tools/reference/README.md`](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/reference/README.md).
To exercise the real optional molecular backends, run
`uv run --extra molecular pytest`. The CI matrix covers Python 3.11, 3.12, 3.13
and 3.14 with the locked and lowest allowed direct dependency versions, plus a
locked molecular-extra job for each interpreter.

## Citing

If you use EnvTok, please cite the method paper and the reference
implementation:

- E. Moerman, I. Poltavsky, A. Charkin-Gorbulin, A. Tkatchenko, *Atoms that
  Matter: Largest Materials Datasets Are 99% Redundant, and Local Chemistry
  Can Guide the Required 1%*, arXiv:2610.14392 (2026).
- E. Moerman, *omat24-chemenv-subset* (software, v1.0.0, MIT licence),
  https://github.com/em819/omat24-chemenv-subset.

## License

EnvTok is released under the [MIT License](https://github.com/SMTG-Bham/py-envtok/blob/main/LICENSE). Code adapted from
omat24-chemenv-subset keeps its MIT notice in
[`LICENSES/omat24-chemenv-subset-MIT.txt`](https://github.com/SMTG-Bham/py-envtok/blob/main/LICENSES/omat24-chemenv-subset-MIT.txt).
