# EnvTok for molecules implementation plan

Date: 5 November 2026

## 1 Purpose and scientific scope

Build a Python workflow that takes a molecular configuration database and returns a smaller set of intact configurations, with traceable coverage of chemistry, electronic state, and geometry. The intended outputs are training subsets or candidates for more expensive quantum calculations.

Use a separate, versioned molecular representation profile with a reusable selection engine. The recommended first implementation combines state-conditioned chemical tokens with continuous geometric refinement. Its success criterion is retained predictive accuracy and useful reductions in total cost.

Published EnvTok uses discrete local environments and approximate weighted set cover, with total selected atoms as the cost. Its descriptor has no explicit molecular charge field; its molecular results are described as ongoing work. Molecular compression performance therefore needs a separate benchmark. [R1](https://arxiv.org/pdf/2610.14392v1)

The design below is a proposed molecular extension. Proposed interfaces and configuration examples are not commands that already exist.

## 2 Main design decisions

| Component | Decision |
| --- | --- |
| Unit of selection | A complete configuration with its original labels and atom ordering. |
| Initial domain | Finite main-group molecules, including known charge and spin states and multiple conformers. |
| Chemical representation | Interpretable rooted graph environments where connectivity is reliable, with a geometric fallback where it is not. |
| Geometry representation | Per-atom continuous descriptors plus targeted torsion or intermolecular separation safeguards. |
| Electronic state | Protect total charge, multiplicity, and known electronic-state distinctions throughout selection. |
| Optimisation | Chemical coverage followed by geometric refinement under an explicit cost budget. |
| Reference labels | Optional for the default selector; required for downstream accuracy assessment. |
| First benchmark | A manageable subset of one molecular dataset and one electronic-structure reference. |
| Scale | Measure the pilot before processing millions of configurations. |
| Extensions | Broader elements, reactions, clusters, metals, polymers, and periodic molecular phases after the relevant validation gates. |

A useful selection profile has three independent controls: how finely chemistry is distinguished, how many configurations must represent a class, and how much geometric diversity is retained within it. Do not use one threshold to stand in for all three.

## 3 Input records and provenance

Define a typed internal record with the following fields. These are package fields; each source adapter must map its actual release schema explicitly.

| Field | Requirement and interpretation |
| --- | --- |
| `record_id` and source identifiers | Stable ID, dataset name, release/version, source record ID, and shard location. |
| `atomic_numbers` and `positions` | All atoms present in the calculation, including explicit hydrogens; positions in angstrom. |
| `pbc` and `cell` | Explicit boundary conditions; finite molecules and clusters use nonperiodic geometry. |
| `total_charge` | Net charge in units of the positive elementary charge; positive means electrons removed. Unknown is an explicit missing value. |
| `spin_multiplicity` | Intended molecular multiplicity, conventionally 2S+1; retain the source convention and mapping. |
| `electronic_state_id` | Optional physically meaningful state category when several labelled states share charge and multiplicity. |
| `label_namespace` | Electronic-structure method, basis, reference convention, and relevant calculation settings. |
| `family_id` and grouping metadata | Parent molecular identity, trajectory, reaction family, distance scan, and matched state-pair IDs when available. |
| `graph` and `graph_status` | Optional atom-mapped bonds, bond types, formal charges, radicals, stereochemistry, and supplied/inferred/ambiguous status. |
| `energy` and `forces` | Optional labels, normalised to eV and eV/angstrom with the original units recorded. |
| Additional labels and conditions | Dipoles, partial charges and their scheme, local spins, fragment IDs, external fields, and other source metadata. |

Do not infer total charge from partial charges unless a documented source convention explicitly warrants it. Do not infer multiplicity by summing rounded site magnetic moments. An intended molecular multiplicity is also different from a calculated spin-contaminated expectation value of S squared.

For a strict charge-aware training export, require known charge and multiplicity or an explicit source-level statement establishing them. A permissive ingestion mode may retain unknown values in separate categories, but it must report them and must not silently label them neutral singlets.

Preserve all labels even when selection does not use them. Any coordinate or atom-order transformation used for matching must leave the exported coordinates, force correspondence, and tensor labels consistent. The easiest first implementation keeps source ordering and coordinates unchanged.

Start with ASE-readable extended XYZ and an iterator adapter. Add the chosen OMol release adapter after checking its accessible manifest and field semantics. Do not assume that every OMol record has a reliable molecular graph. The official dataset location is listed in [R10](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md); release-specific download and schema details are an implementation dependency.

## 4 Data preparation and split protection

### Validate records

Check array shapes, finite numbers, units, atom mapping, declared state metadata, and available calculation-status flags. Record every exclusion and its reason.

Distinguish invalid labels from unusual but valid chemistry. A high energy, large force, stretched bond, or RDKit sanitisation failure is not by itself a reason to discard a configuration. Keep valid difficult structures in an explicit category.

Use one label namespace for the first pilot. When later mixing methods or datasets, select within namespaces or preserve their distinctions explicitly; identical geometry at different electronic-structure levels is not a duplicate labelled calculation.

### Establish splits

Preserve authoritative source train, validation, and test assignments. For a development split made inside the source training partition, group conformers, trajectories, reaction paths, distance scans, and matched electronic-state families before assigning folds.

Where metadata is incomplete, use a documented conservative grouping rule and report the uncertainty. A graph identifier helps with stable molecules but is insufficient for every reaction or metal complex.

Fit class frequencies, geometric normalisation, prototype selection, and any dimensionality reduction on the candidate training pool only. Use development validation to choose settings. Keep the final test set blind.

### Deduplicate conservatively

Begin with exact-record duplicates within a split and label namespace. Include species, coordinates, charge, multiplicity, electronic state, boundary conditions, and relevant external conditions in the identity. Compare labels before merging and keep duplicate lineage.

The same geometry at different charge or multiplicity is a different labelled state. The same SMILES, graph, token set, or rounded descriptor is not proof that two configurations are duplicates.

Flag conflicting labels for the same declared state and reference instead of averaging them automatically. Add symmetry-aware near-duplicate removal only as a later, separately validated option. Check for cross-split duplicates without quietly changing official benchmark partitions.

## 5 Molecular representation

### Separate chemical bonds from geometric neighbours

Use two channels:

- A chemical graph encodes known connectivity, bond order, aromaticity, rings, formal charges, and local graph context.
- A geometric neighbourhood includes atoms within a physical cutoff under the declared boundary conditions, irrespective of whether they share a covalent bond.

For finite molecules, use nonperiodic neighbour construction. Avoid depending on an arbitrary vacuum cell to make a periodic Voronoi frontend behave molecularly. ASE provides neighbour searches and covalent-radius estimates; proximity must still be distinguished from a chemical bond assignment. [R4](https://docs.ase-lib.org/ase/neighborlist.html)

Prefer supplied, atom-mapped graphs. If bonds must be inferred for a stable main-group molecule, pass its known charge and record the inference settings. RDKit exposes choices that affect formal-charge versus radical assignments, so perception requires validation. [R5](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html)

Retain a geometry fallback for ambiguous graphs, reactive structures, radicals, and metal coordination. Give fallback targets their own namespace: a graph-unknown configuration cannot claim to cover a specific validated graph environment.

### Construct chemical tokens

For each atom with a validated graph, create a rooted signature initially extending two graph bonds from the centre. Include:

- Central element and known formal charge.
- Known radical and stereochemical information.
- Bond types and neighbouring elements.
- Aromaticity and ring context.
- Second-shell connectivity.

Use a stable canonical serialization, a deterministic digest, and collision checking against the serialized signature. Do not equate a folded fingerprint bit with an exact environment class.

RDKit's circular fingerprint machinery provides useful atom-centred graph infrastructure, but define the package's own interpretable token serialization and version it. [R6](https://www.rdkit.org/docs/GettingStartedInPython.html)

For the geometric fallback, use the central element, a documented local composition/coordination summary, and geometric bins. Preserve a status identifying which chemical fields are unavailable. Shape fitting can provide additional interpretable fields, especially for coordination centres, but it is optional in the first release.

ChemEnv exposes local geometry objects accepting centre and neighbour coordinates, allowing its shape routines to be reused behind an appropriate molecular neighbour frontend. [R7](https://pymatgen.org/pymatgen.analysis.chemenv.coordination_environments.html)

### Add continuous geometry

Use local SOAP as the initial continuous descriptor backend, with separate descriptors for each atom. DScribe supports finite molecular systems and element-resolved local output. [R3](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html)

Proposed pilot settings are `r_cut = 5.0 angstrom`, `n_max = 4`, `l_max = 4`, `sigma = 0.5 angstrom`, `periodic = false`, and no averaging over centres. These are starting parameters to calibrate, not established molecular EnvTok defaults. Compare at least one larger cutoff and one higher resolution on a small development subset.

Fix the species vocabulary for each run. Normalise nonzero vectors before computing distances; explicitly handle zero or degenerate vectors. Cache descriptors in float32 when its numerical behaviour is adequate, while retaining reference labels at source precision.

Use distances only between compatible descriptor configurations and meaningful chemical/state categories. Avoid a sole whole-molecule mean descriptor because it can dilute a rare local environment.

Dense, fully element-resolved SOAP becomes expensive as the vocabulary grows. Before expanding beyond a small element set, measure feature count, bytes per atom, and throughput. Evaluate compressed descriptors only with an explicit chemical-resolution benchmark.

### Preserve state in coverage

A coverage target should have the following conceptual form:

```math
u_{s,i} = (\nu_s, Q_s, M_s, e_s, h_{s,i}, t_{s,i})
```

Here, `nu` is the label namespace, `Q` the total charge, `M` the multiplicity, `e` a meaningful electronic-state category if available, `h` the representation channel, and `t` the local token.

For a single reference and electronic ground-state convention, constant fields need not be repeated in storage. Do not use a unique record ID as an electronic-state category: that would prevent useful compression.

Conditioning a local class on total molecular charge is a conservative selection rule. It does not assign that charge to every atom or establish a local charge distribution.

Never coarsen away total charge, multiplicity, central element, or a supplied distinction between electronic states that the intended model must resolve. Preserve known central formal-charge and radical categories in the first molecular profile. Unknown values remain distinct.

## 6 Conformation and interaction safeguards

Local chemical coverage is necessary but may select only one member of a large conformational ensemble. Add geometry refinement from the first scientifically evaluated version.

For stable molecular families, preserve torsional diversity when relevant dihedrals are available. Represent angles periodically, for example with sine and cosine, and account for equivalent atom permutations. Compare corresponding torsions within compatible graphs rather than concatenating unrelated molecular torsion lists.

For molecular clusters, preserve intermolecular contacts and separation ranges. Two charged fragments can remain outside every local cutoff while their interaction energy changes with separation. Add fragment-separation targets when fragment identity is reliable; a larger local cutoff alone is not a general solution.

For reactive data, retain reaction-family and path-region metadata when supplied and use continuous geometry rather than insisting on a single fixed bond graph.

For labelled data with charge-localisation information, optional local electronic features may distinguish states sharing total charge. Keep partial-charge schemes separate and test whether they improve selection. Intended total charge and multiplicity remain available before labelling; DFT-derived partial charges do not.

Later polymer support should protect chain length, end groups, branching, stereosequence, and relevant intra- and interchain contacts. Molecular liquids and crystals require periodic neighbours and a separate validated profile.

## 7 Hierarchy and support

Build a hierarchy that progressively reduces geometric bin resolution or graph radius while retaining protected state and chemical fields. Keep the original fine token and every mapping to a coarser class.

Count support in distinct deduplicated training configurations. Also count parent families and trajectory groups so that repeated frames do not masquerade as broad chemical evidence.

Initial tuning may compare support thresholds of 5, 20, and 100 configurations. Treat them as experimental settings. Optionally require support from multiple parent groups once group metadata is reliable.

Rare valid classes that remain below threshold at the coarsest permitted level must be explicitly protected or listed as unresolved/optional targets under a declared policy. Do not silently delete them.

Keep these settings separate:

- `min_source_support` chooses the coarsening level.
- `min_selected_representatives` sets selected configuration multiplicity.
- `geometry_resolution` controls diversity within a class.

The first minimum cover may require one selected configuration per class. Additional geometrically distinct representatives are then added by refinement. Repeated atoms in one configuration do not satisfy a requirement for multiple configurations.

## 8 Selection algorithm

### Define cost and modes

Set the initial configuration cost to its number of atoms, including hydrogen. Report both selected atoms and selected configurations.

Provide a pluggable cost function for later use with measured training or labelling cost. Atom count is only a proxy, especially across very different sizes or electronic-structure methods.

Implement two explicit modes:

- `cover`: seek a low-cost subset covering every declared mandatory target, and return the achieved cost.
- `budget`: respect a hard total cost and maximise the declared coverage/diversity objective, while reporting whether mandatory constraints were achieved.

Do not silently change the representation or state protections to fit a budget. If a heuristic fails to find a feasible cover, say so; that is not a mathematical proof of infeasibility. A certified lower bound or exact solve on the stated candidate universe is needed for that stronger claim.

### Stage A Select a chemical core

Let `U_s` be the distinct targets contributed by configuration `s`. Greedily choose the configuration with the largest uncovered target weight per unit cost:

```math
g(s \mid S) =
\frac{\sum_{u \in U_s \setminus U(S)} w_u}{c_s}.
```

Use this score for the one-representative case and restrict Stage A to mandatory targets. Start with equal target weights. Explicitly protected categories remain constraints rather than merely receiving very large numerical weights. Use deterministic tie-breaking and a manifest-recorded ordering.

In budget mode, admit a candidate only if its marginal atom cost fits the remaining budget. If no eligible candidate can make progress while mandatory targets remain, stop with an incomplete result and list those targets. An unconstrained cover may be computed separately as a diagnostic, but it must not become an over-budget output.

When multiplicity constraints apply, let `n_u(S)` count distinct selected configurations contributing target `u`, with requirement `r_u`. Reward each newly added configuration only up to the remaining deficit `max(0, r_u - n_u(S))`. For a bundle, cap its target contribution at that deficit and charge only the cost of configurations not already in `S`.

Add available known state-pair bundles as optional protected groups for applications that depend on charge or spin differences. Apply the same marginal-cost and remaining-budget rules to those bundles.

Prune a selected configuration only when every mandatory coverage and multiplicity constraint remains satisfied. This is an approximate weighted-cover algorithm, not a guarantee of the global minimum. [R11](https://doi.org/10.1287/moor.4.3.233)

### Stage B Refine geometry

Implement one refinement approach first:

1. Build bounded, deterministic reservoirs of local continuous descriptors within protected chemical/state classes, with sampling balanced across parent groups.
2. Choose geometric prototypes by farthest-point sampling within those reservoirs.
3. Assign candidate environments to prototype regions in batches, using a documented maximum descriptor distance. Give each region a separate refinement target.
4. Spend the remaining budget on configurations that add the most uncovered refinement weight per unit cost.
5. Update all targets contributed by a selected whole configuration, not just the atom that motivated its selection.

Start with a small prototype cap and increase it only where within-class distances or downstream errors justify it. Cap extreme reservoir growth and keep explicit rare-state witnesses.

Prototype regions define an approximation to geometric coverage. Use Euclidean distance between unit-normalised descriptors and a stored region radius, initially 0.2 as a proposed pilot setting to calibrate. An environment farther than that radius from every prototype remains unresolved; either create another region or report it as overflow. One selected overflow example must never count as covering all other distant examples.

Report nearest-selected distances on a separate training-pool audit sample, including large-distance tails. Region assignment alone does not certify that the ultimately selected configurations represent every assigned environment within the radius.

For the first budgeted experiments, compare a chemical core plus geometric refinement with a modest stratified background reserve. A proposed starting reserve is 10% of the atom budget for common conformers and size strata, selected from the remainder after accounting for overlap. If the mandatory core exhausts the budget, report that the reserve/refinement target was not achieved.

### Control size bias

The cheapest cover can favour small molecules. Add explicit size strata or representative larger systems where large-system behaviour is part of the target application. Record these constraints and give comparison baselines the same compulsory protections.

Do not require every unique molecule to survive by default: that can defeat cross-molecule compression. Protect meaningful classes and application categories, and evaluate transfer to held-out molecular families.

## 9 Python architecture and execution

Keep source adapters, representations, and selection independent.

| Module | Responsibility |
| --- | --- |
| `records` and `adapters` | Schema, unit mapping, provenance, and streaming source readers. |
| `validation`, `splits`, and `dedup` | Data checks, grouping, duplicate lineage, and split manifests. |
| `representations` | Molecular graph, geometric fallback, SOAP, optional ChemEnv, and future periodic profiles. |
| `targets` | Hierarchies, support counts, protected categories, and integer target IDs. |
| `selection` | Chemical cover, budget handling, geometric refinement, and deterministic pruning. |
| `audit` | Coverage, distributions, nearest-selected distances, and failure reports. |
| `export` | Intact records, selected IDs, metadata mapping, and provenance. |
| `benchmarks` | Optional model training and application tests. |

Use a small CPU-based core with NumPy, SciPy, ASE, and PyArrow. Add RDKit and DScribe for the molecular profile, scikit-learn where prototype/PCA utilities are needed, and pymatgen only for the optional ChemEnv path.

Keep PyTorch/MACE, model uncertainty tools, quantum calculators, GPU libraries, and approximate-neighbour indices in optional extras. Use a tested Python 3.11 or newer environment and lock exact compatible versions for each release. Persist the environment lock and descriptor configuration with every selection.

A proposed CLI is:

```text
envtok ingest --config molecular.yaml
envtok split --config molecular.yaml
envtok featurize --config molecular.yaml
envtok build-targets --config molecular.yaml
envtok select --config molecular.yaml
envtok audit --config molecular.yaml
envtok export --config molecular.yaml
```

Each stage should support deterministic restart from its manifest. Cache keys must include source content, representation version, state mapping, bin boundaries, descriptor settings, and software versions.

## 10 Scale and storage

Use sharded records and compact integer structure-to-target incidence arrays. Stream the final export from selected original IDs. Avoid loading all labels into memory during descriptor construction.

Parallelise validation and featurisation by shard using worker processes or HPC job arrays. Merge counts deterministically. Keep global selection separate so that different workers do not produce mutually incompatible local covers.

Do not build a dense all-pairs distance matrix. Use bounded descriptor reservoirs, batched assignments, and streaming nearest-selected distance updates. Measure memory and throughput per configuration and per atom.

Benchmark preprocessing first at roughly 100,000 records, then 1 million if the memory model supports it. These are scaling experiments, separate from the smaller model-training pilot.

At full scale, a restricted candidate pool may be necessary. Retain witnesses for every mandatory target, enough distinct witnesses for multiplicity constraints, diverse alternatives, and a stratified background. Preserving witnesses protects basic coverability but does not preserve the globally cheapest cover or all geometric diversity. Record the candidate-pool restriction in the result.

Avoid promising full OMol throughput before measuring the actual descriptor and incidence bottlenecks.

## 11 Pilot experiment

Use one version of OMol25 and one reference level initially. Its state diversity and evaluation tasks make it suitable for the intended molecular questions. [R2](https://arxiv.org/html/2505.08762v2)

A proposed first training pool is 20,000 to 50,000 finite CHNO configurations with at most 60 atoms, sampled reproducibly from the official training partition. Include multiple conformers and valid charge/multiplicity combinations actually present in the data. Do not generate an arbitrary Cartesian product of charge and spin.

Build small diagnostic panels for matched electronic states and noncovalent separation scans. Expand scientific claims to clusters, reactions, metals, and broader elements only after those categories have their own tests.

Preserve official validation/test assignments and construct any smaller development split only inside the training partition.

### Selectors

Compare:

1. Stratified random, with matching state/domain/size protections.
2. Molecular token coverage alone.
3. Molecular token coverage plus geometric refinement.
4. Continuous geometry selection alone, once the first three work.

Use selected-atom budgets of 5%, 10%, and 25% of the candidate pool, plus the full-pool reference. Attempt 1% only if coverage and accuracy results justify it. These fractions are experimental checkpoints, not promised compression ratios.

For affordable screening, begin with the first three selectors at 25% and one seed. Advance the useful settings to the learning curve and repeat decisive comparisons with at least three model seeds. Record selection randomness separately.

A budget at which mandatory coverage was not achieved must remain clearly marked; do not hide violations inside an average coverage score.

### Model choice

First establish a small from-scratch neutral-singlet MACE baseline. For mixed charge and multiplicity, use a pinned model configuration that actually consumes both inputs. If the data include additional electronic branches sharing charge and multiplicity, provide a suitable state input or benchmark a clearly specified branch separately; state labels in the selection manifest alone cannot make the model distinguish them.

The authors' MACE-OMOL training script explicitly configures charge and spin keys and graph-level embeddings. Use that as an interface reference while choosing a much smaller model for the pilot; verify the mappings with a controlled state-sensitivity test. [R8](https://github.com/ACEsuit/mace-foundations/blob/main/mace_omol/mace-omol.sh)

Use identical architectures, reference conventions, losses, and convergence policies across selectors. Report both converged accuracy and a matched-compute comparison. Match processed atoms or measured training work, rather than assuming equal epochs provide equal optimisation effort.

A MACE-POLAR fine-tuning arm can assess practical adaptation, but it is a separate experiment: existing OMol pretraining may supply information absent from the compressed fine-tuning set. [R9](https://github.com/ACEsuit/mace/releases)

MACE-SCF is an optional later benchmark for the intended charge physics. Check the exact model family's label requirements and spin inputs. The fixed-point direct-training path requires reference multipoles and Fermi-level information; ordinary partial charges are not interchangeable with those density coefficients. [R12](https://mace-scf.readthedocs.io/en/stable/fixed_point_scf/model.html)

## 12 Validation and acceptance

### Scientific correctness tests

| Test | Expected behaviour |
| --- | --- |
| Rotation and translation | Token identity and invariant descriptor distances are unchanged within tolerance. |
| Atom permutation | Equivalent atoms/configurations produce equivalent environments while label correspondence survives export. |
| Same geometry at different charge | Distinct protected state targets survive coarsening. |
| Same geometry at different multiplicity | Distinct protected state targets survive coarsening. |
| Unknown metadata | Remains unknown and is reported; never becomes neutral/singlet silently. |
| Same graph with different torsions | Geometric refinement detects relevant conformational diversity. |
| Bond stretching or proton transfer | Geometry remains usable when graph assignment becomes ambiguous. |
| Fragment separation beyond local cutoff | Explicit interaction safeguards retain the tested separation distinctions. |
| Rare valid class | Retained under its policy or listed as uncovered with a reason. |
| Budget too small for the found cover | Clear unmet-constraint result, without an unsupported infeasibility certificate. |
| Export round trip | Atom order, units, states, labels, and provenance are retained. |
| Worker count and restart | Stable selected IDs for deterministic settings. |

### Accuracy and cost measurements

Report energy and force MAE/RMSE, force-error tails, and results by charge, multiplicity, chemistry, molecular size, and representation channel.

Include relative conformer energies and ranking, geometry optimisation behaviour, torsional profiles, same-geometry charge/spin energy and force differences, and intermolecular separation curves where labels permit them. Assess short MD runs with fixed starting structures and explicitly fixed simulation conditions.

Use complete, consistently referenced state pairs and interaction-energy components for difference-based evaluations. Do not compare arbitrary independently shifted state energies.

Report token coverage and geometric distances as diagnostics, not substitutes for model accuracy. Use confidence intervals over molecular families or trajectories, and report training-seed variability separately.

Measure preprocessing time, selection time, peak memory, cache size, training time, and retained atoms/configurations. Total project savings depend on whether selection costs are repaid by fewer labels or repeated training runs.

### Decision rule

Before the final learning curve, define application tolerances and the maximum acceptable regression relative to the full pilot-pool model. A provisional engineering screen can use no more than a 10% relative increase in overall force RMSE, alongside acceptable conformer/state-difference errors and no serious failure in protected categories. Replace that screen with application-specific absolute tolerances before claiming suitability for production simulations.

Select the smallest tested feasible budget meeting the declared criteria. If geometric refinement adds no useful accuracy at matched cost, retain the simpler method. If neither deliberate selector improves on stratified random, report that result before scaling the more expensive pipeline.

## 13 Milestones

| Milestone | Deliverable | Acceptance gate |
| --- | --- | --- |
| 1 Data contract | Reader, schema, validation and export | State and label round-trip tests pass; exclusions are traceable. |
| 2 Molecular representation | Graph tokens, fallback channel, SOAP and protected state keys | Invariance, charge/spin distinction, and conformer diagnostics pass. |
| 3 Selection | Hierarchy, chemical cover, refinement, budget reporting and manifests | Deterministic selected IDs and honest achieved-constraint reports. |
| 4 Scientific pilot | Random, token-only and combined comparisons | Accuracy and cost curves show whether compression is useful. |
| 5 Scale | Sharded processing, compact incidence, measured candidate reduction | Throughput and memory are demonstrated on the intended size range. |
| 6 Broader chemistry | Separate profiles/panels for clusters, reactions, metals, polymers and periodic systems | Each new domain passes representation and application tests. |

The first working release should complete milestones 1 through 4. Full-dataset processing and model-informed active learning should follow measured evidence from that pilot.

## 14 Outputs from every run

Produce the selected-ID manifest, exact configuration and software lock, source/split hashes, achieved cost, target coverage, geometric coverage diagnostics, and distributions by state, chemistry and size.

Also retain coarsening lineage, missing-data and graph-failure reports, exclusions, duplicate lineage, and the provenance of every selected configuration. Include a concise explanation of unmet constraints and any restricted candidate universe.

Export original whole configurations in a documented model-training format plus a machine-readable manifest. Keep selected source IDs available even if materialised training files are moved or regenerated.

## 15 Initial configuration example

This is a proposed schema to implement. The numerical values are pilot settings to test.

```yaml
profile: molecular_v1
source:
  adapter: ase_extxyz
  path: data/molecules.extxyz
  label_namespace: one_pinned_reference
state:
  charge_key: total_charge
  multiplicity_key: spin_multiplicity
  unknown_policy: separate_and_report
  strict_training_export: true
split:
  preserve_source_splits: true
  group_related_states: true
  group_trajectories_and_scans: true
chemistry:
  graph_radius: 2
  prefer_supplied_graph: true
  allow_validated_perception: true
  ambiguous_graph: geometry_fallback
geometry:
  backend: soap
  r_cut_angstrom: 5.0
  n_max: 4
  l_max: 4
  sigma_angstrom: 0.5
  average: "off"
  normalize: true
  cache_dtype: float32
hierarchy:
  min_source_support: 20
  min_selected_representatives: 1
  protect_charge_and_multiplicity: true
  protect_known_central_formal_charge: true
selection:
  mode: budget
  cost: atoms
  atom_fraction: 0.25
  chemical_core: greedy_cover
  geometric_refinement: reservoir_fps_regions
  prototype_region_radius: 0.20
  background_reserve_fraction: 0.10
  unresolved_constraints: report_without_relaxing
reproducibility:
  seed: 20261105
  stable_tie_break: record_id
```

Resolve source-specific charge/spin keys in the adapter rather than assuming this example matches every dataset or model. When exporting to MACE, map the package fields to the pinned training configuration explicitly.

## 16 References

The references establish the original method and available components. The molecular token design, safeguards, pilot parameters, and acceptance workflow above are proposed implementation choices.

- **R1** [Moerman et al., *Atoms that Matter: Largest Materials Datasets Are 99% Redundant, and Local Chemistry Can Guide the Required 1%*, arXiv:2610.14392v1.](https://arxiv.org/pdf/2610.14392v1)
- **R2** [Levine et al., *The Open Molecules 2025 Dataset, Evaluations, and Models*, v2.](https://arxiv.org/html/2505.08762v2)
- **R3** [DScribe SOAP documentation.](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html)
- **R4** [ASE neighbour-list documentation.](https://docs.ase-lib.org/ase/neighborlist.html)
- **R5** [RDKit bond perception API.](https://www.rdkit.org/docs/source/rdkit.Chem.rdDetermineBonds.html)
- **R6** [RDKit circular fingerprints and atom-environment mappings.](https://www.rdkit.org/docs/GettingStartedInPython.html)
- **R7** [pymatgen ChemEnv geometry API.](https://pymatgen.org/pymatgen.analysis.chemenv.coordination_environments.html)
- **R8** [Authors' MACE-OMOL training configuration.](https://github.com/ACEsuit/mace-foundations/blob/main/mace_omol/mace-omol.sh)
- **R9** [Official MACE release notes, including charge/spin embeddings and MACE-POLAR.](https://github.com/ACEsuit/mace/releases)
- **R10** [Official OMol25 dataset manifest location.](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md)
- **R11** [Chvatal, *A Greedy Heuristic for the Set-Covering Problem*, Mathematics of Operations Research 4, 233–235 (1979).](https://doi.org/10.1287/moor.4.3.233)
- **R12** [MACE-SCF fixed-point model and data requirements.](https://mace-scf.readthedocs.io/en/stable/fixed_point_scf/model.html)
- **Original implementation reference** [EnvTok source repository](https://github.com/em819/omat24-chemenv-subset)

Inspect the original repository's current code and licence before reusing implementation. The plan does not assume that the molecular features described here already exist in that repository.

