# py-envtok

Select whole structures from an atomistic dataset and see which local chemical
environments your subset covers. py-envtok counts the atoms you keep, audits
coverage, and exports the original geometries and labels for training.

[Install](#install) · [Quick start](#quick-start) · [Molecules](#molecules-experimental) · [Configure your dataset](#configure-your-dataset) · [Documentation](#documentation-and-examples)

The periodic workflow implements the EnvTok method from Moerman *et al.*'s
[*Atoms that Matter*](https://arxiv.org/abs/2610.14392). The experimental
molecular workflow adds state-aware selection and hard atom budgets.

The 0.1 series is alpha software. APIs and saved-run formats may change between
releases. See the [release notes](https://github.com/SMTG-Bham/py-envtok/blob/main/CHANGELOG.md).

## Install

Use CPython 3.11–3.14. Once the first PyPI release is available, run the
following commands. The Python import and CLI are both `envtok`.

```bash
python -m pip install --pre py-envtok
envtok --version
```

<details>
<summary>Optional extras and upgrades from older wheels</summary>

Install the `molecular` extra for RDKit and DScribe or `omat24` for the
OMat24 ASE-LMDB reader. The reader runs without fairchem or torch.

```bash
python -m pip install --pre 'py-envtok[molecular,omat24]'
```

If you installed an older GitHub wheel distributed as `envtok`, uninstall
that distribution before installing `py-envtok` in the same environment. Both
wheels provide the same import and CLI.

</details>

<details>
<summary>Install from a source checkout</summary>

### From a source checkout

The supported development environment uses [uv](https://docs.astral.sh/uv/)
0.13.6 or newer.

```bash
git clone https://github.com/SMTG-Bham/py-envtok.git
cd py-envtok
uv sync --locked
uv run envtok --version
```

The checkout defaults to Python 3.14. Choose another supported version with
`uv sync --locked --python 3.11` and pass the same version to `uv run`. Add
`--no-dev` to skip development dependencies or `--extra molecular --extra omat24`
to install both extras. The CI matrix qualifies standard CPython on Linux.

From an existing Python environment, use `python -m pip install .` or
`python -m pip install '.[molecular,omat24]'` in the checkout. To install a
built wheel, pass its path to `python -m pip install`. The
[Conda setup](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/README.md#conda-environment)
provides separate run and development environments.

</details>

## Quick start

Try the periodic workflow on the bundled 20-structure corpus. You only need
the checkout and uv.

```bash
git clone https://github.com/SMTG-Bham/py-envtok.git
cd py-envtok
uv run --locked --no-dev envtok inspect --config examples/quickstart/config.yaml --sample 20
uv run --locked --no-dev envtok run --config examples/quickstart/config.yaml
```

The run selects **16 of 20 structures and 56 of 76 atoms**, covering all
**26 fitted classes** and **32 of 33 full tokens** at a support threshold of 2.
Open `report.html` in the selection directory printed by the command to see
which environments the subset covers. The same directory contains the selected
structures in `subset.extxyz` and their IDs in `selected_ids.txt`.
Rerunning the command reuses committed tokenisation work.

The [worked example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/quickstart/README.md)
traces individual selections and compares a random subset at the same atom
ceiling.

<details>
<summary>Run the same workflow with the Python API</summary>

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

</details>

<details>
<summary>See the compression results on real OMat24 structures</summary>

### Real OMat24 compression example

The [OMat24 compression example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/compression-demo/README.md)
uses 1,024 real structures from distinct parent materials. At T=100, it retains
755 structures and 5,077 of 6,705 atoms: **24.3% fewer atoms**, with every fitted
class and required presence signature covered. Detailed full-token coverage
is 76.7%. A separate experiment held out 201 structures before fitting the
training covers. Across three seeds, every method's mean force error exceeded
the zero-force baseline. See the [periodic MACE training results](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/compression-demo/training.md).

Follow the example's sampling command to prepare the input from an existing
local OMat24 archive, then run:

```bash
uv run --extra omat24 envtok run --config examples/omat24/compression-demo/config.yaml
```

The example includes a threshold sweep, equal-atom random comparisons, saved
results and export checks. The [bundled 32-record sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/small-sample/README.md)
also runs without an archive, but shows no compression.

</details>

## Molecules (experimental)

The `molecular_v1` profile selects intact molecules while protecting total
charge, spin multiplicity, electronic state, and reference method. It combines
rooted graph tokens with local SOAP geometry and supports hard atom budgets.

From the repository root, run the bundled OMol configuration. Its top-level
`profile: molecular_v1` selects the molecular schema.

```bash
uv run --locked --no-dev --extra molecular envtok run --config examples/molecular/omol/config.yaml
uv run --locked --no-dev --extra molecular envtok benchmark --config examples/molecular/omol/config.yaml
```

<details>
<summary>Set an atom budget and map charge and spin fields</summary>

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

Molecular runs write to `<workdir>/molecular-<fingerprint>/`. The pipeline
reads one extended XYZ source. The Python adapters can also prepare records
from an ASE iterable. Featurisation supports local worker processes and
memory-mapped descriptors. Records and selection incidence stay in memory.

</details>

<details>
<summary>Explore the OMol samples and training results</summary>

The bundled [OMol sample](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/README.md) contains 29 real
training configurations with stored energy, force, charge and spin data. Its
selector benchmark compares stratified random, token-only and combined
selection at a 25% atom ceiling with seed 0, plus a full-pool control.
Use `--fractions` and `--seeds` to change the comparison.
The [molecular guide](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/molecular.md)
walks through the budget report, custom label units, Python API and saved-run
audit.

The [native OMol25 compression example](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/compression-demo/README.md)
starts from 512 real configurations / 10,938 atoms. Unbudgeted cover mode
retains 146 configurations / 3,276 atoms (**70.0% fewer atoms**), covering
all 44 mandatory element/state targets and 517 modeled SOAP regions.
However, 91.3% of source environments fall outside the fitted prototype
radii. A finer 64-prototype check retains 6,122 atoms (44.0% fewer).

A separate training experiment split the neutral singlets by composition before
selection. The 64-prototype cover used 52.4% fewer training atoms. Its mean
held-out force RMSE was 1.7% higher than the full-pool model's, and one of
three paired seeds exceeded the 10% regression threshold. See the
[molecular MACE training results](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/compression-demo/training.md).

</details>

## How selection works

The periodic workflow tokenises local environments, fits coverage classes to
the eligible training pool, selects a subset by atom cost, and audits the result.
The guarantee applies to those fitted classes and configured presence constraints.
The report also measures full-token coverage.

<details>
<summary>Read the selection and audit steps</summary>

1. **Tokenise the periodic source.** Record each site's element, rounded magnetic
   moment, neighbour composition, ChemEnv polyhedron and distortion, compression
   and asymmetry bins. Cache the measurements. An optional token field records
   the structure's [net charge](#charged-structures).
2. **Fit coverage classes.** Count support in distinct eligible training
   records, then progressively drop token detail according to the profile's
   coarsening ladder. Repeated equivalent sites within one structure count
   only once toward support.
3. **Select and repair.** Start with required records and sole suppliers of a
   class, greedily add coverage per atom, prune redundant structures, then
   repair missing signatures in four families. These pair element with
   polyhedron, coordination number, coordination and compression, or coordination
   and asymmetry. Each signature observed in the eligible pool needs one witness.
4. **Audit independently.** Recompute coverage from the selected IDs and the
   frozen class mapping. The report distinguishes class coverage, original
   full-token coverage, processing quality, and the atoms and structures kept.

Held-out records do not influence the classes. The report counts unseen held-out
tokens as out of vocabulary.

</details>

<details>
<summary>Compare with the paper and reference implementation</summary>

The default profile `reference-723696d` follows the paper's reference
implementation,
[`em819/omat24-chemenv-subset@723696d`](https://github.com/em819/omat24-chemenv-subset/tree/723696de2c26584b8a907ef900690c0caf3c7b68).
Every token of the golden fixture corpus matches that code running in its own
pinned environment. See the [specification](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/specification.md)
and [discrepancy log](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/discrepancies.md).

The paper reports roughly 1% retention for its OMat24 experiment.
Evaluate model accuracy on a held-out split when choosing a subset for training.

</details>

## Configure your dataset

Point a YAML file at your source and specify the training split, tokeniser,
coverage settings, and export format. Use `envtok inspect --config dataset.yaml`
to check the input before a full run.

<details>
<summary>Copy a periodic configuration and tune coverage</summary>

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

</details>

## Inputs

The periodic workflow reads extended XYZ, ASE databases, and OMat24 ASE-LMDB.
The Python API also accepts ASE atoms directly.

<details>
<summary>See adapters, record identities, and label handling</summary>

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

</details>

## Charged structures

Use protected strata to require coverage for each charge state, or add net
charge to the finer levels of the reference token.

<details>
<summary>Set the charge policy and inspect missing values</summary>

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

Under `net_round`, an absent charge fails with reason `missing_charge`.
`net_round_zero_if_missing` treats an absent charge as neutral and counts
it as defaulted. A string or boolean fails as `invalid_charge` under either
policy. The charge is read from the cached metadata when the index
is built, so switching the field on or off never repeats the geometry pass.
Run `envtok inspect` first to check that the key is present. See
[§1.7 of the specification](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/specification.md#17-net-charge-extension-opt-in).

</details>

## Command-line interface

Run `envtok --help` to list commands or `envtok <command> --help` for options.
Use `uv run envtok` from a source checkout.

<details>
<summary>Inspect, select, audit, and export</summary>

For progress logs, put `-v` before the command, as in
`uv run envtok -v run --config dataset.yaml`.

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
| `envtok baseline --run DIR --seeds 0 1 2` | random subsets up to the selected atom count, audited against the same classes |

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

</details>

<details>
<summary>Restart, sweep support thresholds, and run in parallel</summary>

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

</details>

## Outputs

Open `report.html` for a coverage overview and use `coverage.json` for the
underlying audit. The manifest records which original structures were selected.

<details>
<summary>Find manifests, provenance, and saved-run files</summary>

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

Start with `processing` in `coverage.json` to see which inputs were usable.
Then compare eligible and selected atom counts, coverage per presence family
and full-token coverage. The [periodic report walkthrough](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/quickstart/README.md)
works through a complete example. If you set `selection.stratify_by`, read
the audit for each stratum under `strata`. Each stratum has its own fitted
classes, and its own coverage requirement.

Molecular runs also write selected IDs, a manifest, resolved configuration,
provenance and reports. Their additional artifacts are `ingestion.json`,
`splits.json`, `targets.json` (indexing the tables under `targets/`) and
`software-lock.json`. Reports include budget
compliance, unmet constraints and nearest-selected geometry distances. See the
[molecular output contract](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/molecular.md#commands-restart-and-outputs).

</details>

## Documentation and examples

- [Periodic specification](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/specification.md) and [discrepancy log](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/discrepancies.md) describe the token fields, selection, and differences from the reference code.
- The [molecular guide](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/molecular.md) covers data contracts, budgets, and limitations.
- The [OMat24](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/compression-demo/README.md) and [OMol25](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/compression-demo/README.md) examples measure compression on real data. Smaller bundled [OMat24](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/omat24/small-sample/README.md) and [OMol](https://github.com/SMTG-Bham/py-envtok/blob/main/examples/molecular/omol/README.md) samples include qualification results and provenance.
- The [build tools](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/README.md) cover benchmarks and package checks. The [release guide](https://github.com/SMTG-Bham/py-envtok/blob/main/docs/releasing.md) covers publishing.

<details>
<summary>Troubleshoot a run</summary>

### Troubleshooting

| Symptom | What to check |
|---|---|
| Empty or unexpectedly small selection | Check source split assignments against `coverage.split`, exclusions, and failed or quarantined records in the report. |
| Missing charge failures | With `net_round`, the configured key must contain a numeric net charge. Use `inspect` to review charge sources. |
| Export cannot retrieve records | Keep the original sources available; spool one-shot iterators before tokenisation. |
| Indexing reports incomplete partitions | Finish or rerun every tokenisation shard with the same configuration before indexing. |
| Selection exceeds the configured memory policy | Review the estimate and available resources, then adjust `execution.memory_budget`; DuckDB aggregation does not remove the selector's own memory needs. |
| Molecular export rejects unknown state | Provide known total charge and multiplicity through the configured keys; see the molecular data contract for permissive export. |

</details>

<details>
<summary>Set up development and run CI checks</summary>

### Development

Install the development environment and run the checks used by CI:

```bash
uv sync --locked
uv run prek install
uv run prek run --all-files
uv run mypy
uv run pytest
```

To use Conda instead, create the `envtok-dev` environment described in the
[Conda setup](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/README.md#conda-environment) and run the same
commands without `uv run`.

The test suite enforces 100% branch coverage. Golden reference fixtures are
regenerated with the pinned reference environment described in
[`build_tools/reference/README.md`](https://github.com/SMTG-Bham/py-envtok/blob/main/build_tools/reference/README.md).
To exercise the real optional molecular backends, run
`uv run --extra molecular pytest`. The CI matrix covers Python 3.11, 3.12, 3.13
and 3.14 with the locked and lowest allowed direct dependency versions, plus a
locked molecular-extra job for each interpreter.

</details>

## Citing

If you use py-envtok, please cite both the method paper and its reference
implementation.

- E. Moerman, I. Poltavsky, A. Charkin-Gorbulin, A. Tkatchenko, *Atoms that
  Matter: Largest Materials Datasets Are 99% Redundant, and Local Chemistry
  Can Guide the Required 1%*, arXiv:2610.14392 (2026).
- E. Moerman, *omat24-chemenv-subset* (software, v1.0.0, MIT licence),
  https://github.com/em819/omat24-chemenv-subset.

## License

py-envtok is released under the [MIT License](https://github.com/SMTG-Bham/py-envtok/blob/main/LICENSE).
Adapted code retains the upstream MIT notices for
[omat24-chemenv-subset](https://github.com/SMTG-Bham/py-envtok/blob/main/LICENSES/omat24-chemenv-subset-MIT.txt)
and [pymatgen](https://github.com/SMTG-Bham/py-envtok/blob/main/LICENSES/pymatgen-MIT.txt).
