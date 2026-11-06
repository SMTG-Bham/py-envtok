# Experimental finite-molecule profile

`molecular_v1` implements the data, representation and selection stages of
[the molecular plan](molecular-plan.md). It selects
whole original configurations, with explicit charge/spin protections, chemical
coverage and optional geometry refinement. It uses a separate pipeline from
`reference-723696d`; the periodic reference profile keeps its existing behavior.

The implementation is intended for small measured pools. The simplified plan
starts with the bundled 29-record sample, one 25% atom budget and one seed.
Selection coverage and descriptor distances do not establish preserved model
accuracy. Application validation, full-scale sharding and broader chemistry
profiles remain follow-up work.

The [native OMol25 compression example](../examples/molecular/omol/compression-demo/README.md)
extends the runnable examples to 512 real configurations, with unbudgeted
cover, random and budget comparisons, and a finer-prototype sensitivity
check. Its reports separate mandatory targets, modeled geometry regions and
source environments outside the prototype radii.

## Run the engineering example

The bundled generator creates 22 configurations with known charge/multiplicity,
supplied graphs, distortions, and water-dimer separation scans. Its labels are a
synthetic toy function and must not be used to claim quantum-model accuracy.

```bash
uv sync --extra molecular
uv run --extra molecular python examples/molecular/generate_fixture.py
uv run --extra molecular envtok run --config examples/molecular/config.yaml
```

Outputs are written beneath `runs/molecular/molecular-<fingerprint>/`. The YAML
source and work-directory paths are relative to the YAML file. To exercise the
pipeline without the molecular optional dependencies, set
`geometry.backend: radial` and retain
`chemistry.allow_validated_perception: false`. This descriptor is a continuous
element-resolved radial density, which is angle-blind; it is an engineering
alternative, not SOAP.

A [real OMol25 smoke sample](../examples/molecular/omol/README.md) also contains
29 finite CHNO configurations and 650 atoms from the public ColabFit training
mirror. It is an ordered snapshot filtered from the first 100 source rows,
with the exact response saved and hashed. It is not a random or representative
scientific pilot. Its README documents source fields, units, attribution, and
the reproduction command. A separate tiny CPU training/evaluation screen is
available in `build_tools/benchmark_molecular_accuracy.py`; its linear pair model
is an engineering baseline, not a qualified molecular potential.

Use `selection.mode: budget` with either `atom_fraction: 0.25` or an integer
`atom_budget` to impose a hard cap. Remove both fields for `mode: cover`.
A tiny budget can produce a valid, incomplete selection. Check `complete` and
`unmet_constraints` in `coverage.json`; the pipeline never drops a protected
state distinction to make the result appear complete. A failed heuristic cover
is not an infeasibility proof.

## Data contract

The initial adapter reads ASE extended XYZ, including compressed files. Python
callers can use `from_atoms` with an iterable of ASE configurations.

```python
from envtok.molecular import MolecularPipeline, from_atoms, prepare_records

prepared = prepare_records(from_atoms(frames, label_namespace="method-basis-v1"))
training_pool = prepared.candidates

pipeline = MolecularPipeline("examples/molecular/config.yaml")
result = pipeline.select()
report = pipeline.audit()
output = pipeline.export()
```

Here, `frames` is an application-provided ASE iterable. To inspect or export an
existing selection, bind its actual saved directory:

```python
saved = MolecularPipeline.load_run("runs/molecular/molecular-<fingerprint>")
report = saved.audit()
output = saved.export("selected-molecules.extxyz")
```

Map actual source metadata explicitly through `state.charge_key`,
`state.multiplicity_key`, `source.family_key`, `source.label_sources` and
`source.label_units`. The default charge and multiplicity keys are
`total_charge` and `spin_multiplicity`. Positive total charge means removed
electrons; multiplicity is the intended molecular value, conventionally 2S+1.
Neither value is inferred from partial charges or site magnetic moments.

The protected state key is `(label_namespace, total_charge, spin_multiplicity,
electronic_state_id)`. Unknown values remain `None`, occupy separate targets,
and are reported. `unknown_policy: reject` excludes them instead. With the
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
Permissive export keeps mapped unknown state fields unknown, even when the
source also contains an unrelated canonical-looking charge or spin key.

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
| `ingest`, `split` | Validate, group and deduplicate; write ingestion and split reports. |
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

For an existing run, use its saved directory rather than recomputing a directory
from the current software environment. Replace `<fingerprint>` with the run's
actual directory suffix:

```bash
envtok report --run "runs/molecular/molecular-<fingerprint>"
envtok export --run "runs/molecular/molecular-<fingerprint>" --output subset.extxyz
```

These commands use `MolecularPipeline.load_run`. The supplied run directory is
authoritative, including after moving it. Loading verifies the saved configuration
and software lock against its fingerprint, the original source against its
content digest, and the frozen target manifest against its saved digest. Audits
count the current selected IDs against those frozen targets without refitting
the hierarchy or prototypes. A changed runtime is reported through
`software_environment_matches_selection`; it does not select another run.

The resolved source and feature-cache paths are absolute. Moving only the run
directory does not move its source or rewrite those paths. The original source
must remain available and unchanged. If the saved feature cache is unavailable,
coverage and budget auditing continue, while `geometry_audit_status` is
`unavailable_feature_cache` and nearest-selected distances are unavailable.
When features are present, their metadata and descriptor-array digests are
checked; corrupt or changed cached content is rejected rather than recomputed
as part of the audit.

For new runs, configuration/source/software fingerprints address output
directories. Descriptor arrays can be reused across selection settings; source,
state mappings, representation settings/version and recorded software changes
use another feature cache. Repeating deterministic stages preserves selected
IDs. Set `execution.resume: false` in a new run configuration to recompute
features. This pilot implementation holds records and incidence in memory and
does not yet provide worker-count sharding.

Runs retain `selected_ids.txt`, `selected_manifest.parquet`,
`config.resolved.yaml`, `software-lock.json`, `provenance.json`, `ingestion.json`,
`splits.json`, `targets.json`, `coverage.json`, `report.html`, and optionally
`subset.extxyz`. Reports include achieved cost, unmet constraints, coarsening
lineage, graph/fallback status, distributions, prototype overflow, selection
reasons and preprocessing/selection timings.

Small from-scratch pair-model and MACE screens are now available in
`build_tools/benchmark_molecular_accuracy.py` and `build_tools/benchmark_mace.py`.
Both use a grouped development holdout. The pair runner supports a declared
budget and supplied source capped at 128 records; MACE uses the bundled neutral
singlets, a fixed 50% budget and a separate installed training interpreter.
The [sample report](../examples/molecular/omol/README.md) records the results and
limitations. Mixed-state models must consume the relevant state inputs. Wider
datasets, model convergence, repeated seeds, application panels and declared
tolerances remain necessary before general compression or accuracy claims.
