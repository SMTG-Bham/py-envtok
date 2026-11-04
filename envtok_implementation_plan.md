# EnvTok Python package implementation plan

**Purpose:** Build a reusable Python package that accepts a database of atomistic structures, computes interpretable local-environment tokens, and returns a smaller set of complete structures with explicit diversity guarantees.

**Initial target:** Periodic materials, with OMat24 as the first large integration example. Input-format support and scientific descriptor support are separate capabilities.

**Working package name:** `envtok` throughout this plan; check distribution-name availability before release.

**Specification date:** 4 November 2026.

## 1. Outcome and scope

Implement a pipeline with independently reusable stages:

`dataset adapter → validated records → token cache → coverage classes → subset selection → manifest and report → optional structure export`.

The output is a subset of original configurations. Selection does not crop individual atoms, change geometries, average labels, or create synthetic structures. Energies, forces and stresses are optional inputs to the selection pipeline; preserve them when exporting labelled data.

The minimum useful release must:

1. Accept an ASE iterable, extended XYZ, ASE databases, and OMat24 ASE-LMDB shards.
2. Implement a pinned EnvTok reference profile.
3. Count support across distinct input configurations and construct the reference coverage classes.
4. Select whole structures using atom-weighted greedy cover, remove redundant selections, and repair the required rare-environment coverage.
5. Cache expensive calculations and support bounded parallel processing with reliable restart.
6. Return stable source IDs, a provenance manifest, and an independently verified coverage report.

Measure compression separately by structure count and atom count. A diversity cover does not preserve the original sampling distribution and is not a guarantee of equivalent MLIP accuracy. Treat the retained fraction as an outcome to validate on each database, rather than promising a universal 1% target. The paper's very strong compression result belongs to its particular OMat24 experiment.[1]

Training models, running DFT, molecular tokenisation, active learning and a graphical interface are outside the first release. The architecture must allow these uses without requiring them.

## 2. Freeze the scientific target before refactoring

Use the inspected reference commit:

`723696de2c26584b8a907ef900690c0caf3c7b68`

from `em819/omat24-chemenv-subset`. The repository is MIT licensed; retain the applicable notices when adapting code and cite the method.[2]

### 2.1 Explicit compatibility profile

Name the first algorithm profile `reference-723696d`. This identifies scientific behaviour, not a promise that a modern implementation reproduces every historical selected ID.

The paper and released code differ in consequential details:

| Topic | Paper v1 | Inspected reference code |
|---|---|---|
| Coarsening sequence | Table S1 enumerates 13 rungs | Generates 33 rungs from 11 identity levels and 3 modifier states |
| Magnetic token | Described as an integer-rounded site moment | `round(abs(initial_magmoms[i]))`; Python ties-to-even |
| Geometry presence constraint | Written using element and polyhedron | Uses a reliability-adjusted polyhedron, replacing poorly fitted or undefined geometries by `UNCL` |
| Rare-presence repair | Candidate-selection details not specified | Processes four constraint families sequentially; maximises new signatures, then favours fewer atoms and lower internal index |

These are observable differences, so do not combine the paper's ladder with the code's other settings and call the result an exact reproduction.[1,3,4]

Implement the code profile first. Add `paper-v1` only as an explicitly separate, documented interpretation with its own fixtures. A 13-rung version is not necessary for the MVP.

Freeze the full dependency environment used for reference fixtures. The repository records, among other versions, ASE 3.25.0, pymatgen 2025.6.14 and fairchem-core 2.3.0. Resolve and record indirect library defaults rather than silently using the latest defaults.[2,5]

### 2.2 Scope of compatibility

Test equivalence at three levels:

- Site fields and full tokens.
- Full-token-to-class assignments for a fixed corpus and threshold.
- Selected coverage, costs and IDs on small cases with controlled candidate ordering.

The production package should use stable record IDs for deterministic tie-breaking. A dedicated legacy test can reproduce the reference row order; without that order, two equally good covers can legitimately choose different IDs.

Keep corpus policies separate from the descriptor profile. The reference code excludes structures containing La–Lu or Ac–Lr; the general package must not silently apply that restriction to every input. Provide an opt-in `omat24-study` corpus preset with that exact filter. A paper-literal policy excluding all Z >= 89 is a separate choice for inputs containing heavier elements.

## 3. Public API and package boundaries

Keep the core scientific algorithm independent of database engines, HPC schedulers and MLIP libraries.

| Module | Responsibility |
|---|---|
| `records.py` | Typed structure records, source references and label metadata |
| `adapters/` | ASE iterable, extxyz, ASE DB and OMat24 readers; source retrieval |
| `tokenize/` | Neighbours, geometry, raw measurements and token encoding |
| `profiles/` | Versioned reference settings, ladders and corpus presets |
| `store/` | Partitioned tables, stable dictionaries, checkpoints and cache validation |
| `coarsen.py` | Exact unresolved-pool support and class assignment |
| `select/` | Transparent greedy baseline, optimised selector, pruning and constraints |
| `report.py` | Independent coverage audit, compression and distribution diagnostics |
| `export.py` | Manifest export and source-preserving structure/label export |
| `cli.py` | Configuration and stage orchestration |

Suggested core dependencies are ASE, NumPy, pymatgen, PyArrow, DuckDB and a small CLI/configuration layer. Add SciPy or Numba when profiling justifies them. Keep fairchem/database backends and MPI support in optional extras. No MACE, PyTorch model checkpoint or GPU should be required by the core selection API.

Proposed interface, to be implemented:

```python
from envtok import Corpus, Tokenizer, CoverageBuilder, Selector

corpus = Corpus.from_source("omat24.yaml")
cache = Tokenizer(profile="reference-723696d").run(
    corpus, workdir="runs/omat24/tokens", workers=16
)
classes = CoverageBuilder(
    profile="reference-723696d", min_structure_support=100
).fit(cache, split="train")
result = Selector(mode="cover", cost="atoms").select(classes)
result.validate()
result.write_manifest("runs/omat24/selection")
result.export("runs/omat24/selection/subset.extxyz", format="extxyz")
```

These are proposed names and contracts, not commands or APIs currently provided by the upstream repository.

## 4. Dataset adapter contract

### 4.1 Structure record

Each adapter yields a `StructureRecord` containing:

| Field | Requirement |
|---|---|
| `record_id` | Stable, globally unique configuration ID |
| `source_ref` | Dataset name/revision, shard and native row key or frame locator |
| `atoms` | Species, positions, cell and the three periodicity flags |
| `natoms` | Original atom count; never infer selection cost from filtered token counts |
| `split` | Training, validation, test or an explicit unsplit state |
| `group_id` | Parent material, trajectory or calculation family when available |
| `metadata` | Charge, multiplicity, method, temperature, pressure, defect labels and other provenance when available |
| `labels` | Optional energy, force, stress, magnetic and other arrays with units and source keys |

Use source identity and revision to namespace native IDs. Do not assume that `sid` is unique across datasets, shards or trajectories. A frame ID and a parent-material ID serve different purposes.

Adapters must support streaming iteration, an input manifest and later retrieval of selected records. A one-shot iterable must be spooled to a managed, versioned source store if later export is requested. Otherwise expose manifest-only operation and explain its retrieval limitations.

### 4.2 First adapters

- **ASE iterable:** smallest integration point for user code.
- **Extxyz:** stream frames and build a reusable locator index where possible. Handle compressed files by an explicit sequential-read or staging policy.
- **ASE database:** retain database row keys and provenance.
- **OMat24:** read its ASE-compatible LMDB files through a tested backend. The official dataset card documents `AseDBDataset` and `get_atoms`; isolate those version-sensitive calls inside this adapter.[6]

Do not convert the entire OMat24 corpus to text before tokenisation. Read native shards and export only selected records. A generic Parquet or custom database adapter is appropriate once an explicit geometry schema exists; a table of formulas alone cannot supply EnvTok environments.

### 4.3 Preflight and eligibility

Validate representative records before scheduling a large run, then validate every record during processing:

- Positive atom count, finite coordinates, valid species, a usable cell for periodic profiles and consistent array shapes.
- Correct unit conventions; convert only with explicit mappings.
- Duplicate source IDs, missing source references and source-file changes.
- Supported periodicity and geometry domain.
- Optional metadata and label availability.

Keep an accounting identity: every discovered record is eligible, explicitly excluded, or failed/quarantined. No silent skipping.

For the reference corpus comparison, retain the reference fallback/sentinel behaviour. For normal production use, quarantine unreadable structures and structures with irrecoverably invalid sites by default; recovered neighbour sets may remain eligible with status flags. Report both policies and their effect on the eligible denominator.

Do not silently remove close contacts or large-force structures: unusual but valid environments may be the configurations the selection is intended to retain.

## 5. Implement and test tokenisation

Represent a full site token as typed fields:

`(central_element, moment_bin, neighbour_multiset, geometry, csm_bin, compression_bin, asymmetry_bin)`.

Store canonical neighbour composition as element/count pairs, together with an explicit coordination number. Generate the historical string representation only for interoperability. Some ChemEnv geometry names contain underscores, making naive underscore splitting unsafe.[3]

### 5.1 Neighbours and geometry

Port the reference's geometry functions into small, independently testable functions:

1. Preserve original atom order and cell. Disable structure refinement.
2. Use the reference centroid convention and `csm_wcs_ctwcc` measure.
3. Construct ChemEnv structure environments once per structure.
4. Evaluate normalized distance cutoffs 1.4 and 1.7 with angle cutoff 0.3.
5. Adopt the wider neighbour set for a site when an added contact has normalized Voronoi weight at least 0.45.
6. Assign the geometry and obtain CSM.
7. Apply CrystalNN recovery when the reference does; retain explicit recovery status.

The maximum distance factor is 2.0 in the reference dependency setup. Explicitly freeze symmetry tolerances and all inherited ChemEnv strategy settings. Modern documentation lists defaults that must not be assumed to define the historical profile.[3,7]

Preserve each periodic neighbour's image translation. Distinct images of the same atom are distinct contacts. The multiset, geometry, compression and asymmetry must use the same chosen neighbour set.

For compatibility, also retain the reference's particular added-contact comparison: a multiset of (element, distance rounded to 0.01 angstrom), with the maximum Voronoi weight associated with a matching key. Replacing that comparison with (site index, periodic image) may be a useful improvement, but changes the profile. Keep image-resolved vectors for the actual distance/asymmetry measurements in both cases.

Symmetry reuse is an optimisation, with a validated tolerance and an all-sites fallback. Reuse geometric results only: attach each original atom's own moment after expansion. The reference uses absolute scalar initial moments; a geometric symmetry orbit does not establish magnetic equivalence.

If magnetic labels are absent, the compatibility token uses zero. Preserve whether the moment was supplied or defaulted in metadata. A geometry-only option must consistently disable the moment field for the entire run. Reject unsupported vector/noncollinear moment arrays with an actionable message rather than silently coercing them. Do not infer quantum spin multiplicity or net charge from this field.

### 5.2 Raw measurements and bins

For neighbour displacement vector `r_ij`, distance `d_ij`, and Cordero covalent radii `R_i, R_j`, compute:

```text
q_ij = d_ij / (R_i + R_j)
rho_i = min_j q_ij
A_i = norm(sum_j exp(-(q_ij - 1) / 0.1) * r_ij / d_ij)
```

There is no normalization of A by coordination number or total weight. The minimum is over normalized distances, not necessarily the shortest distance in angstroms.[3]

| Field | Bin definitions |
|---|---|
| CSM | 0: <=1; 1: (1,5]; 2: (5,10); 3: >=10 |
| Compression | 0: <0.60; 1: [0.60,0.70); 2: [0.70,0.85); 3: [0.85,1.15); 4: >=1.15 |
| Asymmetry | 0: <1; 1: [1,3); 2: >=3 |
| Undefined | 999 in the reference interchange representation |

When `cache_raw_measurements=true`, retain raw CSM, normalized minimum/maximum distances, A, neighbour provenance and status with a lossless mapping to every source site, as well as the binned token. One representative raw value per binned token is insufficient because different raw measurements can share a bin. If this cache is disabled, changing bins requires feature recomputation. Use an explicit typed missing value internally, with a lossless mapping to the reference sentinel.

Use `UNKNOWN:CN` for a recovered or unclassified geometry with known neighbours, and an invalid status when no usable neighbour set exists. An unknown coordination number must not become 0 or 1 merely because a placeholder string was parsed.

Version the radius table and record missing-radius handling. The reference accepts tabulated Cordero radii only for Z <= 96; checking array length or positive values alone admits ASE's higher-Z placeholders. It skips unavailable neighbour radii and returns undefined if no usable contacts remain. A stricter all-contacts-required policy is a separate profile change; do not introduce it silently into compatibility tests.

Check effective behaviour for large CSM values with fixtures. Undefined geometry and severe distortion are different cases; avoid accidentally introducing a CSM rejection threshold while refactoring library calls.

### 5.3 Determinism and cache invalidation

The reference notes that some high-coordination ChemEnv fits can have nondeterministic raw CSM. It observed stable bins in its data, which is not a universal guarantee.[5]

Pin library versions, control random state per stable record where supported, and test token stability across processes and repeated runs. Record backend nondeterminism if it remains. Do not promise bitwise reproducibility across arbitrary library versions or platforms.

Use layered cache fingerprints:

- **Source:** revision and immutable shard checksums/identifiers.
- **Geometry:** neighbour and symmetry settings, backend versions and radius table.
- **Token:** magnetic policy and bin definitions.
- **Selection:** eligible corpus, ladder, T, constraints, costs and tie policy.

Changing T never requires repeating geometry. Changing bins can reuse raw measurements. Changing neighbour rules, radii or asymmetry lambda requires appropriate recomputation unless complete image-resolved neighbour vectors were retained.

## 6. Persistent intermediate data

Use partitioned Parquet for transport and inspection, DuckDB for external grouping/joining, and compact numeric arrays for selection.

| Table | Main columns |
|---|---|
| `structures` | record ID, locator, atom count, split, group, metadata, status |
| `tokens` | full-token ID and canonical typed fields |
| `structure_tokens` | record ID, full-token ID, atomic multiplicity |
| `site_tokens` | original atom index, full-token ID and raw diagnostics; required or losslessly represented elsewhere when raw caching is enabled |
| `token_classes` | full-token ID, class ID, rung and unresolved-pool support |
| `classes` | class signature, rung, support and terminal-residue flag |
| `structure_classes` | unique record/class incidence; multiplicity stored separately if needed |
| `selection` | record ID, selection reason, order and cost |
| `failures` | record/site, stage, exception category, fallback and retry outcome |

Full-token counts measure atoms. Class incidence for the baseline measures presence. Keep these concepts in separate columns and tests.

Avoid repeating long token strings on every atom. Intern canonical keys globally and remap partition-local IDs. Either sort canonical keys deterministically or use stable digests with collision verification; Python's process-randomized `hash()` must not define persistent identity.

Use 64-bit counts and offsets where corpus scale requires them. Select ID widths only after checking the actual cardinality. Persist each stage's schema and provenance before downstream work.

## 7. Build the coverage classes exactly

### 7.1 Reference ladder

Let E denote the central element, M moment, L the full ligand multiset, G geometry, D CSM bin, C coordination number, B compression and A asymmetry.

Let G* equal G when D <= 1 and `UNCL` otherwise, including undefined CSM. Let U(L) be unique ligand elements and H(L) ligand multiplicities binned into 1, 2, 3–4, 5–6 and 7+.

The reference's 11 identity projections are:[4]

| Level | Identity fields |
|---|---|
| R0 | E, M, L, G, D |
| R1 | E, M, L, G |
| R2 | E, L, G |
| R2a | E, L, G* |
| R2b | E, H(L), G*, C |
| R3 | E, U(L), G*, C |
| R4 | E, U(L), C |
| R5 | E, U(L) |
| R6 | E, G*, C |
| R7 | E, C |
| R8 | E |

Try all 11 identities with (B,A), then all 11 with A, then all 11 without either modifier. This modifier-outer ordering generates 33 rungs. The first rung is the full token.

Construct every projection from the original full token. This is not a nested tree: reliability-adjusted geometry can reintroduce a distinction, and R6 restores geometry/CN after R5.

### 7.2 Exact unresolved-pool support

Initialize the unresolved vocabulary only from tokens present in the frozen eligible training pool, even if the cache also contains failed, excluded or held-out records. For each rung, consider only full tokens not assigned at an earlier rung. For each projected signature:

`support = number of distinct eligible record IDs containing at least one unresolved full token with that signature`.

Assign every unresolved token belonging to a signature whose support reaches T. Remove those tokens from the unresolved pool. After all rungs, assign remaining tokens to a per-element terminal class even if its support is below T.

Pseudocode:

```python
unresolved = full_token_ids_in_eligible_training_pool
mapping = {}

for rung in ordered_rungs:
    projected = project_original_tokens(unresolved, rung)
    # Join to structure_tokens; deduplicate (record_id, signature).
    support = count_distinct_records(projected, structure_tokens)
    accepted = signatures_with_support_at_least(support, T)
    newly_assigned = tokens_in_signatures(projected, accepted)
    assign(mapping, newly_assigned, rung, accepted)
    unresolved -= newly_assigned

assign_per_element_terminal_classes(mapping, unresolved)
assert every_full_token_has_exactly_one_class(mapping)
```

Class identity includes the rung and canonical signature. Persist the observed support and whether the class is a terminal residue.

Preserve the reference validation `T >= n_coverage + 1`, hence integer `T >= 2` for the single-coverage baseline. Allowing T=1 is a documented extension, not part of this compatibility profile.

Do not count atoms, sum overlapping token-support counts, or reuse already-resolved tokens to increase support at a coarser rung. For example, two equivalent sites in one frame contribute one supporting frame.

For external aggregation, project within disjoint record partitions, deduplicate (record, signature), then merge signature counts across partitions. Partition-local coarsening is not equivalent to global coarsening.

Increasing T usually trades resolution for compression, but this ordered, nonnested procedure does not guarantee nested class sets or a monotonic selected atom count. Sweep candidate values empirically, rebuilding classes from full tokens for each value.

## 8. Select complete configurations

### 8.1 Baseline cover objective

For eligible structure s with atom count n_s and represented class set C_s, seek a low-cost subset S satisfying:

```text
minimise sum(n_s for s in S)
subject to union(C_s for s in S) == all required classes
```

Implement the transparent baseline first:

1. Identify any genuinely mandatory sole suppliers.
2. Select the candidate maximising newly covered classes / original atom count.
3. Break ties by smaller atom count, then stable record ID.
4. Update uncovered classes until all are covered.
5. Remove redundant selected structures in descending atom count, preserving all class requirements.
6. Repair the four rare-presence families below.
7. Recompute coverage independently from the selected manifest.

This is a greedy heuristic, not a certificate of minimum size. Keep `natoms` as explicit cost; token/class counts may change under alternate profiles and must not redefine cost.

Optimise only after matching the transparent implementation. A lazy heap works because marginal coverage gain cannot increase as classes become covered. Store sparse structure-to-class and class-to-structure incidence with compact arrays; avoid a Python object for every atom or edge.

### 8.2 Rare-presence constraints

The reference repairs, in order:

1. (central element, G*).
2. (central element, coordination number).
3. (central element, coordination number, compression bin).
4. (central element, coordination number, asymmetry bin).

Targets include every signature present in the eligible corpus, without a support threshold. During each family, the reference chooses the structure adding the most missing signatures, breaking ties by smaller atom count then lower internal material index. The package substitutes its documented stable record order unless running a legacy-order fixture. This is distinct from the atom-normalized main cover score.[4]

Repair occurs after pruning. Any later optional prune must preserve both ordinary coverage and every presence constraint. Record whether a structure was selected by the cover, repair, a user constraint or an optional distribution supplement.

### 8.3 Coverage and budget modes

The MVP exposes `cover`. Reserve a later `budget` mode with a different contract:

| Mode | Contract |
|---|---|
| `cover` | Cover all classes and required presence signatures; report resulting cost |
| `budget` | Stay within a requested atom budget and maximise coverage heuristically; report every unmet requirement |

A practical route to a small full cover is a T sweep followed by choosing the best satisfactory run. Do not implement a binary search that assumes monotonicity.

In budget mode, reserve or integrate hard user constraints, account for repair cost, and never overshoot silently. If the heuristic cannot find a compliant solution, say that; do not claim mathematical infeasibility without a proof.

Do not expose the upstream `n_coverage > 1` option as “independent examples”: that implementation counts site multiplicities. A future `min_distinct_records` or `min_distinct_groups` feature must explicitly count different records or parent groups.[4]

## 9. Scale and recover reliably

The reference documents a costly geometry pass and a selection stage requesting about 400 GB RAM; its Python allocations are not controlled by DuckDB's memory setting. Treat those as motivation to measure and redesign memory use, not as timings promised for the new package.[2,8]

### 9.1 Execution

- Stream input; keep a bounded queue of tasks.
- Start with local process workers and modest batches. Open database readers inside workers when required.
- Give every worker independent output partitions; avoid concurrent writes to one DuckDB file.
- Use immutable shard names, temporary writes and atomic completion markers.
- Resume only shards whose source and configuration fingerprints match.
- Merge committed shard manifests deterministically; a retry must not duplicate rows.
- Coordinate process count and native-library thread count.

For HPC, run tokenisation as a SLURM array over explicit source shards, then execute global aggregation and selection as later jobs. MPI is optional; avoid constructing a future for every configuration in the entire corpus.

### 9.2 Memory

Separate the stages with different memory needs:

- Geometry: bounded per-worker structure state.
- Aggregation: partitioned/external sorting and distinct grouping.
- Selection: compact/memory-mapped incidence, counters, candidate state and a compiled or numeric lazy-priority structure when needed.

DuckDB supports larger-than-memory processing, but its limit does not cap every allocation in the process. Set a temporary-disk budget, leave memory for Python and arrays, and measure total resident memory.[9]

Estimate sparse edge count, vocabulary size, disk use and peak RSS on a representative pilot before building the full index. Refuse a run whose estimated working state exceeds its configured resource policy, or require a supported larger-capacity backend. Memory mapping is not by itself a runtime guarantee: measure page faults and I/O during selection.

Do not select independently inside each shard and concatenate those subsets while claiming a global cover. Optional hierarchical or candidate-reduction methods need separate names, coverage audits and quality comparisons.

### 9.3 Pilot sizes and performance records

Suggested progression: 100–1,000 curated fixtures, 10,000 heterogeneous records, 100,000 records, an available OMat24 1M subset, then a full corpus.

At each stage record structures and atoms per second, failure/fallback rates by chemistry, vocabulary/incidence size, peak RSS, disk bytes and completion/restart behaviour. Include varied cell sizes and distorted structures; a pilot of only tiny ideal cells will misestimate costs.

Profile before adding C++/Rust/CUDA. Geometry fitting and data handling must be measured separately. No accelerator implementation is required for the initial release.

## 10. Preserve the information needed for training

### 10.1 Data leakage

Preserve official train/validation/test boundaries. Fit the token vocabulary, unresolved-pool supports, coverage targets, T choices and selection on training data only.

For a database without established splits, create parent/trajectory-aware splits before selection. Distinct frame support is not a measure of statistical independence.

Evaluate validation/test raw tokens against the frozen training mapping. Unseen full tokens remain explicitly OOV under exact lookup. An optional coarser projection-based novelty diagnostic must be named separately; never let evaluation records modify the fitted classes.

### 10.2 Electronic states and label fidelity

The original token has no net-charge field, and the reference moment bin discards sign. Geometrically identical records may still carry distinct electronic states or reference methods.[3]

Preserve charge, multiplicity, magnetic initialization, DFT method/Hubbard settings and label provenance. For charged-defect work, allow protected keys such as (host, defect type, charge), subject to a meaningful caller-supplied grouping.

The simplest first extension is protected metadata strata or separate selection within compatible state/fidelity pools, followed by a global audit. State-conditioned environment classes are a later explicitly versioned extension. Geometry-only deduplication must not delete distinct charge states or merge different methods' labels.

### 10.3 Distribution and long-range context

Report differences between input and output element, composition, cell-size, source, geometry, compression and asymmetry distributions. When labels exist, include energy/force/stress summaries as diagnostics without making them a requirement of EnvTok.

A cover may retain one representative of a class and underrepresent important variation within it. Optional remedies include multiple independent witnesses, a stratified random supplement, or additional context descriptors. Evaluate each as an extension.

A molecule/slab profile needs its own validated neighbour and geometry treatment. The generic selection engine can be reused; accepting an ASE object does not establish scientific validity for nonperiodic chemistry. Do not impose artificial periodicity and advertise that as molecular EnvTok.

## 11. Output contract and reporting

Every successful selection creates:

- `selected_ids.txt`: stable record IDs.
- `selected_manifest.parquet`: source locators, atom counts, groups, state/method metadata and selection reasons.
- `config.resolved.yaml`: explicit scientific and execution settings.
- `provenance.json`: source revisions, input fingerprints, package/dependency versions and cache fingerprints.
- `coverage.json`: independently recomputed numerical results.
- `report.html` or equivalent readable report.
- Optional `subset.extxyz`, ASE database or another source-compatible export.

The report must distinguish:

| Metric | Definition |
|---|---|
| Structure retention | selected eligible records / all eligible records |
| Atom retention | selected original atoms / eligible original atoms |
| Class coverage | selected coarsened classes / all required classes |
| Full-token coverage | distinct original tokens in selection / distinct original tokens in eligible input |
| Frequency-weighted full-token coverage | input atom frequency belonging to selected full tokens / eligible atoms |
| Rare-presence coverage | satisfied / required signatures, for each family |
| Resolution loss | tokens and atoms assigned at each rung, including terminal residue |
| Processing quality | excluded/failed records, invalid sites, fallback rates and missing metadata |

Complete class coverage does not imply complete full-token coverage. Showing both prevents a heavily coarsened result from appearing artificially comprehensive. Audit against the frozen class universe fitted on the full eligible training pool; never rebuild or coarsen the targets on the selected subset to calculate its coverage.

Export from original source records. Preserve atom order, cell, periodicity, energy/force/stress conventions, charge and method metadata. Avoid attaching a live calculator that can recompute missing labels. Compare representative exports with source data using numerical tolerances appropriate to the file format. ASE provides streaming I/O and format-specific handling, so the adapter/export tests must cover where results are actually stored in the pinned ASE version.[10]

## 12. Proposed configuration and CLI

Example configuration for a pilot; resource values are illustrative:

```yaml
schema_version: 1
source:
  adapter: omat24
  path: /data/omat24/train
  revision: REQUIRED_DATASET_REVISION
  split: train
  id_policy: namespaced_source_record

corpus:
  preset: general
  exclude_elements: []
  deduplicate: false
  invalid_record_policy: quarantine

tokenizer:
  profile: reference-723696d
  magnetic_policy: source_abs_round_zero_if_missing
  cache_raw_measurements: true

coverage:
  min_structure_support: 100
  support_unit: record

selection:
  mode: cover
  cost: atoms
  prune: true
  presence_constraints: reference
  tie_break: natoms_then_record_id

execution:
  workers: 16
  max_inflight_batches: 32
  memory_budget: 32GB
  temporary_disk_budget: 200GB
  workdir: runs/omat24
  resume: true

output:
  format: extxyz
  preserve_labels: true
```

Here T=100 is a starting comparison value, not a recommended constant for every database. For a study-style OMat24 reproduction use the explicit `omat24-study` corpus preset and the reference failure/candidate-order policies.

Proposed CLI:

```bash
envtok inspect --config omat24.yaml
envtok tokenize --config omat24.yaml
envtok index --config omat24.yaml
envtok select --config omat24.yaml --min-support 100
envtok report --run runs/omat24/selection-T100
envtok export --run runs/omat24/selection-T100 --format extxyz
```

Also provide `envtok run --config ...` for the complete pipeline and a T-sweep command that reuses the token cache. Each selection gets a separate configuration-addressed directory. Never resume from an unrelated fixed-name checkpoint in the working directory.

## 13. Verification and acceptance criteria

Use tests that check scientific contracts and real failure modes.

### 13.1 Tokenisation

- Ideal and perturbed NaCl, tetrahedral and octahedral sites, mixed-ligand environments and asymmetric compression.
- Exact bin boundaries and values immediately either side.
- Same element with site-dependent positive/negative moments; verify the chosen profile's absolute-value behaviour.
- Geometry names containing underscores.
- Periodic self-images and multiple images of one atom.
- Translation, rotation, atom permutation and periodic wrapping, with correctly transformed cells and tolerances.
- Symmetry-enabled versus all-sites calculation on a representative sample.
- CrystalNN recovery, unknown geometry, irrecoverable failure and missing radius cases.
- Repeated runs and worker counts for deterministic tokens, with explicit handling of boundary-sensitive cases.

### 13.2 Coarsening and selection

- Multiple sites in one structure count once for support.
- Different full tokens sharing one signature in one structure still count once.
- Already-resolved tokens cannot support a later rung.
- Every full token has exactly one final class; terminal residues are retained.
- A fixture that distinguishes 13-rung and 33-rung behaviour.
- Invalid neighbour placeholders cannot masquerade as CN=1.
- A small cover checked by enumeration to validate coverage/cost accounting; do not require greedy optimality in general.
- Lazy and transparent selectors agree under identical tie-breaking.
- Pruning preserves all class coverage; repairs preserve every presence requirement.
- When budget mode is implemented, it never exceeds the budget silently.
- Optional repeated coverage counts records/groups, never atomic multiplicity.

### 13.3 Pipeline and scientific usefulness

- Serial, parallel and resumed processing produce the same completed record set without duplicates.
- Stale source/configuration caches are rejected.
- Export preserves source labels and their alignment.
- Validation/test examples cannot alter training selection.
- Distinct charge/method records survive when protected by configuration.
- Observed peak memory stays within the declared execution policy on the qualification dataset.

For scientific validation, compare EnvTok subsets with several random subsets at matched atom budgets, using the same training settings and fixed held-out evaluation. Include target properties and MD stability if those are the intended uses. This evaluation can remain in an optional examples/benchmark workflow.

Release claims should state the corpus, profile, T, cost, retained atoms/structures, coverage, failure rates and downstream results. Software coverage invariants and MLIP accuracy are separate acceptance gates.

## 14. Implementation milestones

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| 1. Specification and fixtures | Pinned reference checkout, profile document, small licensed fixtures, discrepancy log | Exact expected token/class behaviour recorded |
| 2. Working local pipeline | ASE/extxyz adapters, pure tokenisation, transparent coarsening/cover, manifest export | A small dataset compresses end to end with independently checked coverage |
| 3. OMat24 integration | Tested LMDB adapter, source IDs, metadata/labels, pilot report | Native shards work without whole-corpus text conversion |
| 4. Scale and restart | Bounded workers, immutable shards, external aggregation, compact selector | Same semantics as baseline; measured memory and reliable resume |
| 5. Scientific qualification | T sweep, equal-atom random baselines, held-out task results | Compression/accuracy tradeoff documented for a real use case |
| 6. Package release | Versioned API/CLI/config schema, tests, examples, documentation, citation/license | Fresh environment reproduces the documented small example |

Defer budget optimisation, multiple distinct witnesses, metadata-conditioned classes, molecular descriptors and new accelerators until the baseline is correct and measured.

## 15. First tasks for an implementation agent

1. Create the package skeleton and a short specification containing the decisions above.
2. Pin the upstream commit and capture a reference environment.
3. Extract pure geometry/binning functions without importing the entire MPI workflow.
4. Create a small, explicit fixture corpus and record original IDs/atom order.
5. Implement the typed record/token schemas and ASE/extxyz adapters.
6. Implement the 33-rung unresolved-support builder with a simple exact backend.
7. Implement transparent cover, prune, repair and independent report generation.
8. Only after this works, add OMat24 and optimise storage/execution.

The first demonstration should show an input dataset, selected original configurations, full provenance, and verified class/presence coverage. A lower row count alone is not a successful demonstration.

## Sources

1. Moerman et al., *Atoms that Matter: Largest Materials Datasets Are 99% Redundant, and Local Chemistry Can Guide the Required 1%*, arXiv:2610.14392v1. Methods and supplementary S1–S2: https://arxiv.org/pdf/2610.14392v1
2. Reference repository README and MIT licence at inspected commit: https://github.com/em819/omat24-chemenv-subset/tree/723696de2c26584b8a907ef900690c0caf3c7b68
3. Reference tokeniser: https://github.com/em819/omat24-chemenv-subset/blob/723696de2c26584b8a907ef900690c0caf3c7b68/01_token_construction/tokenize_omat24.py
4. Reference selector: https://github.com/em819/omat24-chemenv-subset/blob/723696de2c26584b8a907ef900690c0caf3c7b68/04_subset_selection/select_subset.py
5. Pipeline behaviour and environment: https://github.com/em819/omat24-chemenv-subset/blob/723696de2c26584b8a907ef900690c0caf3c7b68/PIPELINE.md and https://github.com/em819/omat24-chemenv-subset/blob/723696de2c26584b8a907ef900690c0caf3c7b68/envs/envA-chemenv/requirements.txt
6. Official OMat24 dataset card: https://huggingface.co/datasets/facebook/OMAT24
7. ChemEnv API: https://pymatgen.org/pymatgen.analysis.chemenv.coordination_environments.html
8. Reference selection memory and settings: https://github.com/em819/omat24-chemenv-subset/blob/723696de2c26584b8a907ef900690c0caf3c7b68/04_subset_selection/README.md
9. DuckDB performance and memory documentation: https://duckdb.org/docs/stable/guides/performance/how_to_tune_workloads.html and https://duckdb.org/docs/stable/operations_manual/limits.html
10. ASE I/O and format documentation: https://docs.ase-lib.org/ase/io/io.html and https://docs.ase-lib.org/ase/io/formatoptions.html

All architecture, API names, execution policies and release milestones in this plan are proposed package design. The algorithm profile identifies which reference behaviours they are intended to preserve.
