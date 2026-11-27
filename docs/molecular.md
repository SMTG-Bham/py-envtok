# Experimental finite-molecule profile

`molecular_v1` selects whole finite molecules. Its chemical targets distinguish
charge, spin multiplicity, electronic state and reference method. You can run
an unrestricted cover or set an atom ceiling. Both report the mandatory targets
achieved and the geometry regions represented by the selected molecules.

## Run the engineering example

From the repository root, run the [bundled configuration](../examples/molecular/config.yaml)
on 22 synthetic molecular configurations. The fixture includes known charge
and multiplicity, supplied graphs, distortions and a water-dimer separation
scan. Its energy and force labels come from a toy function.

```bash
uv run --extra molecular envtok run --config examples/molecular/config.yaml
```

Outputs are written beneath `runs/molecular/molecular-<fingerprint>/`. The YAML
source and work-directory paths are relative to the YAML file. To exercise the
pipeline without the molecular optional dependencies, set
`geometry.backend: radial` and retain
`chemistry.allow_validated_perception: false`. This descriptor is a continuous
element-resolved radial density, which is angle-blind; it is an engineering
alternative, not SOAP.

## Read a budgeted run

The [OMol25 sample](../examples/molecular/omol/README.md) contains 29 training
configurations and 650 atoms. Its configuration sets `selection.mode: budget`
and `atom_fraction: 0.25`. A fractional ceiling is rounded down after grouping,
deduplication and training-split assignment. For a fixed ceiling, replace
`atom_fraction` with an integer `atom_budget`. Omit both for `mode: cover`.

```bash
uv run --extra molecular envtok run --config examples/molecular/omol/config.yaml
```

Open `coverage.json` in the printed selection directory.

| Fields | What the sample shows |
|---|---|
| `achieved_cost`, `budget`, `selected_configurations` | 10 whole configurations cost 154 atoms, within the 162-atom ceiling. |
| `complete`, `covered_targets`, `mandatory_targets` | All 15 mandatory targets have the required number of distinct selected witnesses. An incomplete run lists each deficit in `unmet_constraints` and can still exit successfully. |
| `covered_refinement_targets`, `refinement_targets` | The subset represents 80 of 133 modeled geometry regions. These counts are separate from `complete`. |
| `geometry_overflow_environments` | 481 source environments lie outside the fitted prototype radii. This can include sites in selected molecules. |
| `nearest_selected_distances` | The distance audit compares source descriptors to selected environments in the same fitted class. Read `p95` alongside `sample_count` and `missing_class_count`. |

To study geometry resolution at a larger scale, the [512-configuration
example](../examples/molecular/omol/compression-demo/README.md) compares two
prototype caps on the same source descriptors.

## Data contract

### Configure an extended XYZ source

For a file with custom state and label names, save a YAML file beside the
source. In this example each frame stores its energy in `Atoms.info["E"]`,
its force vectors in `Atoms.arrays["F"]`, and its charge and multiplicity
in the scalar metadata fields `charge` and `mult`. Set the namespace to
the calculation method used for those labels.

```yaml
profile: molecular_v1
source:
  path: molecules.extxyz
  dataset: conformers
  revision: v1
  label_namespace: HF/sto-3g
  label_sources:
    energy: info:E
    forces: arrays:F
  label_units:
    energy: Hartree
    forces: Hartree/Bohr
state:
  charge_key: charge
  multiplicity_key: mult
```

The molecular adapter reads plain or compressed extended XYZ. For energy and
forces already stored in an ASE calculator in eV and eV/Å, the default source
and unit mappings apply. With custom mappings, `calc:`, `info:` and `arrays:`
select stored calculator results, frame metadata and per-atom arrays,
respectively. A frame's own `label_namespace` takes precedence over the
configured default.

### Use the Python API

To run the same file-backed pipeline as the CLI and reopen its saved selection,
use the directory returned by `run()`.

```python
from envtok.molecular import MolecularPipeline

pipeline = MolecularPipeline("examples/molecular/config.yaml")
directory = pipeline.run()
saved = MolecularPipeline.load_run(directory)
report = saved.audit()
print(report["selected_configurations"], report["achieved_cost"])
```

To inspect training candidates from an ASE iterable in memory, pass the frames
to `from_atoms` and `prepare_records`. Here the iterable comes from the bundled
fixture. The same calls accept frames produced by application code.

```python
from ase.io import iread
from envtok.molecular import from_atoms, prepare_records

frames = iread("examples/molecular/molecules.extxyz")
prepared = prepare_records(from_atoms(frames, label_namespace="synthetic-harmonic-v1"))
print(len(prepared.candidates), len(prepared.exclusions))
```

### State, labels and splits

The default charge and multiplicity keys are
`total_charge` and `spin_multiplicity`. Positive total charge means removed
electrons; multiplicity is the intended molecular value, conventionally 2S+1.
Neither value is inferred from partial charges or site magnetic moments.

The protected state key is `(label_namespace, total_charge, spin_multiplicity,
electronic_state_id)`. Unknown values remain `None`, occupy separate targets,
and are reported. `state.unknown_policy: reject` excludes records missing
charge or multiplicity. The electronic state ID is optional. With the
standard `strict_training_export: true`, export requires known charge and
multiplicity. Validation rejects malformed coordinates, periodic records,
invalid state metadata and invalid labels; graph perception failure alone uses
the geometry fallback.

Labels are normalized to eV and eV/angstrom with original units retained. The
adapter and exporter preserve original atom ordering, source positions, supported
per-atom arrays and JSON metadata, stored calculator results, source locators and
selected IDs. Custom labels need explicit source and unit mappings when they
are to participate in normalized-label validation. Export does not run a live
calculator. Arrays that cannot be represented faithfully in the extended XYZ
writer, including string columns containing whitespace and reserved `pos` or
`species` arrays, cause an error before replacing an existing output file.
Set `state.strict_training_export: false` to export selected records with
unknown charge or multiplicity. Permissive export preserves these unknowns
even if the source also contains an unrelated canonical-looking state key.

Official source `split` assignments are preserved. Development validation can
be made only within training records with `split.validation_fraction`. Related
families, trajectories, reaction families, scans and matched state pairs are
grouped before assignment; grouping retains aliases from records removed by
deduplication. Missing/conflicting grouping information is reported.
Only the deduplicated candidate training pool fits species vocabularies,
chemical supports and geometric prototypes.

Deduplication compares exact species/geometry/state/boundary conditions and
relevant external metadata within a split and label namespace. Label or graph
conflicts are retained and reported, not averaged or silently merged. The same
geometry at another charge, spin or reference level is a distinct calculation.
Cross-split duplicate warnings do
not silently modify official partitions.

## Supplied graphs and fallback

Prefer an atom-mapped dictionary in `Atoms.info["graph"]`, with
`graph_status: supplied` (the default for a supplied dictionary):

```python
atoms.info["graph"] = {
    "bonds": [[0, 1, "single"], [0, 2, "single"]],
    "formal_charges": [0, 0, 0],
    "radicals": [0, 0, 0],
    "stereochemistry": [None, None, None],
    "aromatic": [False, False, False],
    "rings": [False, False, False],
}
```

Endpoints are zero-based indices into the original atom order. Bond types are
`single`, `double`, `triple`, `aromatic` or their numeric equivalents
`1`, `2`, `3`, `1.5`. An optional fourth bond entry records stereochemistry.
All per-atom lists have the original atom count; omitted chemical fields are
unknown. Ring membership and aromaticity can be derived from the supplied
connectivity/bond types when their flags are omitted. Formal charges and radical
counts are never inferred by the supplied-graph parser.

Use atom-order-independent stereochemical categories such as R/S and E/Z.
Raw clockwise/counterclockwise tags depend on neighbour ordering and must be
mapped by the source adapter before use. When permuting atoms, remap bond
endpoints and every atom-indexed field together. Supply all explicit hydrogens
present in the calculation. Missing hydrogens are not synthesized.

The graph channel serializes canonical rooted induced subgraphs, initially at
radius two. A deterministic digest is checked against each complete serialized
signature. Coarsening reduces radius and retains the protected state and central
known formal-charge/radical fields. It does not use folded fingerprint bits as
exact chemical identifiers.

`graph_status: ambiguous`, missing graphs or malformed graphs use an independent
fallback channel containing central species, available central charge/radical
fields, neighbour composition and distance bins. Geometric proximity does not
become a chemical bond. A fallback token cannot cover a graph token.

RDKit perception is opt-in through `allow_validated_perception: true`. The first
implementation restricts perception to integral known charge, singlets, and a
small main-group vocabulary (H, B, C, N, O, F, Si, P, S, Cl, Br, I). It passes the
known charge and uses charged-fragment assignments. Unsupported states,
inferred radicals or failed perception use the fallback. This conservative
restriction is not a claim that RDKit cannot represent those states. Install
`envtok[molecular]` before enabling perception.

## Geometry and safeguards

SOAP uses DScribe with `periodic=False`, `average="off"` and one descriptor per
original atom. Pilot settings are a 5 angstrom cutoff, `n_max=4`, `l_max=4` and
`sigma=0.5` angstrom. Nonzero descriptors are normalized and cached in float32;
zero vectors remain explicitly reported. Set the species vocabulary for each
run or let the candidate training pool establish it. Compare cutoff and
resolution choices in development validation before scientific use.

Supplied metadata can add mandatory whole-configuration safeguards:

- `fragment_ids`: one string or integer per atom, stored in `Atoms.info` or a
  per-atom array. Fragment-pair minimum separations are binned independently of
  the local descriptor cutoff, with a default 1 angstrom width.
- `torsions`: a list such as
  `[{"id": "central-bond", "atoms": [0, 1, 2, 3], "symmetry": 1}]`.
  A stable `family_id` and corresponding torsion IDs are required. Angles are
  periodic, with default 30-degree bins; `symmetry` specifies equivalent repeats
  per full turn. Source adapters must establish equivalent atom permutations.
  Undefined torsions are errors, rather than silently dropped safeguards.

The hierarchy counts distinct configurations and reports distinct family support.
`min_source_support`, `min_selected_representatives`, and
`prototype_region_radius` are independent controls. Rare valid classes are
protected by default; `hierarchy.rare_policy: optional` explicitly reports them
as optional targets. Repeated equivalent atoms do not satisfy a requirement for
multiple selected configurations.

Geometry refinement uses deterministic family-balanced reservoirs, bounded
farthest-point prototypes and radius-limited assignments within compatible
chemical/state classes. An environment outside every prototype radius is
reported as overflow; selecting one overflow does not claim to cover every
other overflow. The independent nearest-selected distance audit describes
actual selected environments, rather than treating prototype membership as an
accuracy guarantee.

## Commands, restart and outputs

All stages accept `--config` and `profile: molecular_v1`:

| Command | Output or effect |
| --- | --- |
| `ingest`, `split` | Both validate, group and deduplicate the full source and write `ingestion.json` and `splits.json`. Molecular `inspect` also runs this stage over the full file, even with `--sample`. |
| `featurize` | Compute or reuse checked local descriptor arrays and token signatures. |
| `build-targets` | Fit protected hierarchy and geometric regions; write incidence. |
| `select` | Select, audit, and write selected IDs and the source manifest. |
| `audit` | Audit existing IDs against saved mandatory targets and the hard budget. |
| `export` | Audit and export the original selected configurations. |
| `run` | Select, audit and optionally export. |
| `benchmark` | Produce selector comparisons; these are diagnostic, not model accuracy. |

`benchmark` defaults to the three selectors at `--fractions 0.25 --seeds 0`,
with the full training pool included once as a separate control. To opt into a
larger comparison, pass e.g. `--fractions 0.25 0.5 --seeds 0 1`. The source
configuration defines the pool; the command does not silently subsample it.

For an individual selection, `selected_ids.txt` gives the export order and
`selected_manifest.parquet` records each molecule's state, source locator
and selection reason. Look in `provenance.json` for stage timings and cache
paths. `output.format: none` skips writing `subset.extxyz` during `run`.
The saved selection can still be exported later.

### Reopen a saved selection

Pass the directory printed by the synthetic example's `run` command to
`report` or `export`. Replace `<fingerprint>` with its actual suffix.

```bash
run_dir="runs/molecular/molecular-<fingerprint>"
uv run --extra molecular envtok report --run "$run_dir"
uv run --extra molecular envtok export --run "$run_dir" --output "$run_dir/subset.extxyz"
```

These commands use `MolecularPipeline.load_run`. The supplied run directory is
authoritative, including after moving it. Loading verifies the saved configuration
and software lock against its fingerprint, the original source against its
content digest, and the frozen target manifest against its saved digest. Audits
count the current selected IDs against those frozen targets without refitting
the hierarchy or prototypes. A changed runtime is reported through
`software_environment_matches_selection`; it does not select another run.

Every stage also checks that the source is unchanged. `select()` and `export()`
hash it in full. Between them, the source is hashed again whenever its size,
modification time, change time or inode differs from the last verified hash;
otherwise the unchanged stat signature stands in for the hash. Because file
timestamps are coarse, a signature is trusted only when the source had been
unchanged for two seconds before that verification. An edit that keeps all
four fields, which an ordinary write to a settled file cannot do, is caught at
the next full hash. On network file systems that cache attributes, a change may
likewise only be seen at the next full hash.

The resolved source and feature-cache paths are absolute. Moving only the run
directory does not move its source or rewrite those paths. The original source
must remain available and unchanged. If the saved feature cache is unavailable,
coverage and budget auditing continue, while `geometry_audit_status` is
`unavailable_feature_cache` and nearest-selected distances are unavailable.
When features are present, their metadata and descriptor-array digests are
checked; corrupt or changed cached content is rejected rather than recomputed
as part of the audit.

### Feature cache and target tables

The feature cache stores every candidate descriptor as one uncompressed float32
array, `descriptors.npy`, with records in the order of `records.parquet`. That
table holds each record's metadata, and its token hierarchy as integer token
IDs; `signatures.parquet` holds each token's signature once, and its row order
numbers the tokens. `features.json` records the SHA-256 digest of each file,
and `commit.json`, written last, records the digest of `features.json`.
Featurisation writes and hashes each record's descriptor rows as they are
produced, so a new cache is not read back, and later stages read
memory-mapped views of the descriptors rather than holding them in memory.
Loaded features share one string per token and one tuple per distinct token
path, and a record's signatures are read from the table only when one is first
looked up. Geometry regions and the nearest-selected audit take the descriptor
array directly and convert it to float64 in small chunks, so no class is held in
float64 at once.

A run's `targets.json` likewise records the digests of the frozen tables under
`targets/`: the sorted token vocabulary, incidence, regions, hierarchy,
classes, requirements, geometry classes and overflow as Parquet, the float64
prototypes as `prototypes.npy`, and each descriptor row's fitted class as
`atom_classes.npy`. Every table is checked against its digest when the targets
are opened. The audit reads only the incidence, requirements, atom classes and
overflow count, and from the feature cache only record IDs, row counts, graph
statuses and the descriptors. Token signatures are kept in the feature cache
only.

Runs and feature caches written by `molecular_v1.5`, which kept targets and
record metadata in single JSON documents, and earlier caches with
`descriptors.npz` are still read when auditing or exporting a saved run.
Because descriptors are memory-mapped, do not edit cache files while a pipeline
is using them: their digests are checked when they are loaded, not on every
access.

For new runs, configuration/source/software fingerprints address output
directories. Descriptor arrays can be reused across selection settings; source,
state mappings, representation settings/version and recorded software changes
use another feature cache. Repeating deterministic stages preserves selected
IDs. Set `execution.resume: false` in a new run configuration to recompute
features. The pilot runs on a single machine. A fresh run holds full molecular
records through featurisation.
Target fitting, selection and audit then use compact candidate metadata and
in-memory incidence. Calling `ingest()` explicitly can materialise all records
again. Export reads the selected frames in selection order from the verified
source. Indexed plain files yield one frame at a time. Sequential compressed
files buffer selected frames during a filtered pass so export can restore their
selection order.

On a saved run, audit can rebuild the candidate metadata in a pass over the
source using the recorded split assignments. If the split manifest is missing
or fails verification, the pipeline repeats preparation from the source.

### Workers and memory

Featurisation can run in local worker processes. Set `execution.workers` to
the number of processes and `execution.threads_per_worker` to cap the native
threads in each (both default to 1). Each worker returns one table of distinct
tokens and signature payloads per batch. Atom hierarchies and safeguards refer
to rows of that table. The parent reconstructs each record's signatures and
checks for conflicts using a 16-byte BLAKE2b digest per SHA-256 token. Serial
featurisation uses the same check across records. An explicitly supplied
`SignatureRegistry` still retains the full payloads.

Worker settings are left out of run and feature fingerprints, so
changing them reuses an existing run. Starting the workers takes several
seconds, because each imports the descriptor and perception libraries, so
pools that featurise in a few seconds gain little. Use at most one worker per
physical core. Workers take batches of at most 64 records, and at most
`2 * workers` batches wait to be written, so finished batches do not pile up.

Set `execution.memory_budget` (for example `"64GB"`, or a byte count) to refuse
a run whose estimated peak memory exceeds it. Before featurisation, the
pipeline adds this process's resident memory, which includes the loaded
records, to `estimate_molecular_bytes`: 512 MiB for the descriptor and
perception libraries, 1.5 KiB per candidate atom for features, fitted targets,
selection and the audit, and 4 KiB per atom of each worker batch waiting to be
written. These constants were calibrated, with headroom, on perturbed copies of
OMol records, whose fine tokens are almost all distinct; pools that repeat
chemical environments need less. Memory-mapped descriptors are read through
the page cache and are not counted. Like the worker settings, the budget does
not change run or feature fingerprints.

### Scale measurements

On the 512-record native OMol25 compression sample, the collision registry
retained 4.04 MiB of Python allocations for 21,733 tokens, down from 9.63 MiB
with payload strings. Across eight 64-record SOAP batches, serialized worker
results (including descriptors) fell from 37.27 to 36.34 MiB. Feature-cache
files, selection and export matched byte for byte with one and two workers.
Whole-run parent peak RSS on this sample was similar before and after.

Reusing exact identities during preparation reduced the median time to validate,
group and deduplicate those 512 records from 0.163 to 0.097 seconds over three
warm single-process runs. Peak traced Python allocations during preparation
fell from 472 to 325 KiB. Both measurements exclude reading the source and the
memory occupied by the input records.

`build_tools/benchmark_molecular.py` measured perturbed copies of the 29 bundled
OMol records, with `omol_metadata` removed, through `select()` in budget mode
(25% of atoms, 10% reserve) with SOAP descriptors on one 20-core machine with
7.6 GiB of memory. The noise makes almost every fine token distinct while the
coarse classes saturate, so these pools are not a substitute for real data.
Peaks are anonymous memory, resident or swapped; the loaded records alone
account for about 25 KiB each.

| Version | Records | Workers | Peak memory | Targets | Feature metadata | Descriptors |
|---|---:|---:|---:|---:|---:|---:|
| `molecular_v1.5` | 7,424 | 1 | 1,244 MiB | 349 MiB | 129 MiB | 432 MiB |
| `molecular_v1.5` | 29,696 | 4 | 4,569 MiB | 1,272 MiB | 441 MiB | 1,727 MiB |
| `molecular_v1.5` | 100,050 | — | about 15 GiB (extrapolated) | about 4.1 GiB (extrapolated) | at most 1.4 GiB (extrapolated) | 5,817 MiB |
| `molecular_v1.6` | 7,424 | 1 | 750 MiB | 58 MiB | 12 MiB | 432 MiB |
| `molecular_v1.6` | 29,696 | 4 | 1,991 MiB | 222 MiB | 37 MiB | 1,727 MiB |
| `molecular_v1.6` | 100,050 | 6 | 5,376 MiB | 649 MiB | 99 MiB | 5,817 MiB |

Both versions select the same records and write byte-identical coverage
reports. Most of the remaining target size is the float64 prototypes
(569 MiB at 100,050 records). At 100,050 records the 5.8 GiB descriptor file did
not fit in that machine's page cache. The select stage took 480 s but used only
91 s of CPU, waiting on disk reads for the rest; target fitting took 410 s, and
how much of that was paging was not measured.

On the 512-record native OMol25 compression sample, a single-worker check of
the record-lifetime change counted 512 original ASE atom objects still reachable
after featurisation before the change and zero afterward. The selected IDs,
selected manifest, coverage report and exported structures matched byte for
byte. Peak process RSS through export was about 490 MiB before and 485 MiB
after in this small run. Peak memory at 100k records with this change remains
unmeasured.
