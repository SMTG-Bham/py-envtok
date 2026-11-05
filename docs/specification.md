# EnvTok specification

This document defines the scientific profile `reference-723696d` and the
package contracts EnvTok implements. Where the paper and the reference code
differ, or where EnvTok deliberately differs from the reference, the
difference is recorded in [`discrepancies.md`](discrepancies.md).

The pipeline has independently reusable stages:

`adapter → validated records → token cache → coverage classes → selection → audit, manifest and report → optional structure export`

Selection returns a subset of the original configurations. It never crops
atoms, changes geometries, averages labels or creates structures. A diversity
cover does not preserve the sampling distribution, and it does not guarantee
equivalent model accuracy. Compression is always reported by structure count
and by atom count separately.

## 1. Profile `reference-723696d`

The profile reproduces the scientific behaviour of
[`em819/omat24-chemenv-subset`](https://github.com/em819/omat24-chemenv-subset)
at commit `723696de2c26584b8a907ef900690c0caf3c7b68`. That repository is
distributed under the MIT licence, and its notice is in
`LICENSES/omat24-chemenv-subset-MIT.txt`. The profile identifies behaviour
only. EnvTok breaks ties by stable record IDs, so it does not promise to
reproduce every historical selected ID.

All settings live in `envtok.profiles.reference`. Settings the reference left
to library defaults are written out explicitly. Their values were introspected
in the reference environment and are stored in
`tests/fixtures/reference/defaults-723696d.json`. A test checks that the
frozen values match.

### 1.1 Geometry

| Setting | Value |
|---|---|
| Structure refinement | `none`: the original cell and atom order are analysed |
| Centring | `centroid`, central site included in the centroid |
| ChemEnv sites | every site (`only_cations=False`) |
| Voronoi | `maximum_distance_factor=2.0`, `minimum_angle_factor=0.05`, normalized distance and angle tolerances 0.05 and 0.03, `optimization=2` |
| Strategy | `SimplestChemenvStrategy`: angle cutoff 0.3, `additional_condition=1`, CSM cutoff 10, measure `csm_wcs_ctwcc` |
| Neighbour sets | normalized distance cutoffs 1.4 (default) and 1.7 (extended) |
| Extended set adopted | if a contact it adds has normalized `VoronoiNN(tol=0)` solid-angle weight ≥ 0.45 |
| Added-contact comparison | multiset of (element, distance rounded to 0.01 Å); weight is the largest Voronoi weight of a matching key |
| Rescue | `CrystalNN` with its default settings when ChemEnv assigns no environment |
| Symmetry reuse | `SpacegroupAnalyzer(symprec=0.01, angle_tolerance=5)` orbits, with all-site fallback |

ChemEnv structure environments are computed once per structure and read at
both cutoffs. Each periodic contact keeps its image translation, so distinct
images of the same atom are distinct contacts. The neighbour multiset, the
geometry, the compression and the asymmetry of a site always come from the
same chosen neighbour set.

Symmetry reuse is purely an optimisation. Geometric results are reused across
an orbit, while each atom keeps its own initial moment. If spglib fails or
returns an inconsistent dataset, every site is analysed individually and the
fallback is recorded.

### 1.2 Raw measurements

For each contact *j* of site *i*, with displacement `r_ij`, distance `d_ij`
and Cordero covalent radii `R_i` and `R_j`:

```text
q_ij  = d_ij / (R_i + R_j)
rho_i = min_j q_ij
A_i   = | sum_j exp(-(q_ij - 1) / 0.1) * r_ij / d_ij |
```

`A` is not normalised by coordination number. The minimum is taken over
normalized distances. The radius table is frozen from ASE 3.25.0
(`cordero2008/ase-3.25.0`) and covers Z ≤ 96 only. ASE's placeholders for
heavier elements are never used. Contacts without a radius, and zero-length
contacts, are skipped. If no contact remains, ρ and A are undefined.

### 1.3 Token

A full site token is the typed tuple

`(central_element, moment, neighbours, geometry, csm_bin, dmin_bin, asym_bin)`

| Field | Definition |
|---|---|
| `moment` | `round(abs(initial_magmom))`, rounding ties to even; 0 when the record has no initial moments; `NA` for every site when `magnetic_policy: disabled` |
| `neighbours` | sorted (element, count) pairs of the chosen set; `None` (`INVALID`) when no usable set exists |
| `geometry` | ChemEnv symbol (for example `O:6` or `BO_1:8`), `UNKNOWN:<cn>` for a known set without a fitted polyhedron, or `INVALID` |
| `csm_bin` | 0 if CSM ≤ 1; 1 if ≤ 5; 2 if < 10; otherwise 3 (a NaN CSM lands in 3, as in the reference). Undefined when ChemEnv gives no CSM |
| `dmin_bin` | ρ: 0 below 0.60; 1 for [0.60, 0.70); 2 for [0.70, 0.85); 3 for [0.85, 1.15); 4 at or above 1.15 |
| `asym_bin` | A: 0 below 1; 1 for [1, 3); 2 at or above 3 |

Undefined values are typed `None` internally. In the interchange string they
map losslessly to `999`. The interchange string is
`E_spin_nb_poly_f5_dmin_abin`, for example
`Fe_0_Cl-O-O-O_T:4_3_1_2`. Some ChemEnv symbols contain underscores, so the
string is parsed with three fields anchored at the start and three at the
end. Keys whose neighbour list is not sorted are rejected.

Moments must be collinear scalars. Vector moments fail validation with an
actionable message and are never silently coerced. The moment field does not
encode net charge or spin multiplicity. Net charge is an opt-in token field
(§1.7).

### 1.4 Signature fields

Rungs and presence families join token fields with `|`. The derived fields
reproduce the reference SQL byte for byte. A test runs the reference SQL in
DuckDB to check this.

| Field | Definition |
|---|---|
| `poly_reliable` (G*) | the geometry when `csm_bin ∈ {0, 1}`, otherwise `UNCL` (including an undefined CSM) |
| `nb_binned` (H(L)) | sorted `element:bin` labels, with multiplicity bins `1`, `2`, `34`, `56`, `7p` |
| `nbset` (U(L)) | sorted distinct neighbour elements |
| `cn_sig` | the coordination number, or `NA` for an invalid or empty neighbour set (so a placeholder can never pass as CN = 1) |

### 1.5 Ladder

There are 11 identity levels and 3 modifier states, `(dmin, a_bin)`, then
`(a_bin)`, then none. The modifier loop is the outer one, which gives 33
rungs, finest first. Every projection is built from the original full token.
This is not a nested tree: `R2a` re-splits by CSM reliability, and `R6`/`R7`
restore geometry and coordination number after `R5`.

| # | Rung | Signature fields |
|---|---|---|
| 0 | `R0_full` | elem \| spin \| nb \| poly \| f5 \| dmin \| a_bin |
| 1 | `R1dA_drop_f5` | elem \| spin \| nb \| poly \| dmin \| a_bin |
| 2 | `R2dA_drop_spin` | elem \| nb \| poly \| dmin \| a_bin |
| 3 | `R2adA_unrel_poly` | elem \| nb \| poly_reliable \| dmin \| a_bin |
| 4 | `R2bdA_binned_mult` | elem \| nb_binned \| poly_reliable \| cn_sig \| dmin \| a_bin |
| 5 | `R3dA_nbset_poly` | elem \| nbset \| poly_reliable \| cn_sig \| dmin \| a_bin |
| 6 | `R4dA_nbset_cn` | elem \| nbset \| cn_sig \| dmin \| a_bin |
| 7 | `R5dA_nbset` | elem \| nbset \| dmin \| a_bin |
| 8 | `R6dA_elem_poly` | elem \| poly_reliable \| cn_sig \| dmin \| a_bin |
| 9 | `R7dA_elem_cn` | elem \| cn_sig \| dmin \| a_bin |
| 10 | `R8dA_elem` | elem \| dmin \| a_bin |
| 11 | `R0A` | elem \| spin \| nb \| poly \| f5 \| a_bin |
| 12 | `R1A_drop_f5` | elem \| spin \| nb \| poly \| a_bin |
| 13 | `R2A_drop_spin` | elem \| nb \| poly \| a_bin |
| 14 | `R2aA_unrel_poly` | elem \| nb \| poly_reliable \| a_bin |
| 15 | `R2bA_binned_mult` | elem \| nb_binned \| poly_reliable \| cn_sig \| a_bin |
| 16 | `R3A_nbset_poly` | elem \| nbset \| poly_reliable \| cn_sig \| a_bin |
| 17 | `R4A_nbset_cn` | elem \| nbset \| cn_sig \| a_bin |
| 18 | `R5A_nbset` | elem \| nbset \| a_bin |
| 19 | `R6A_elem_poly` | elem \| poly_reliable \| cn_sig \| a_bin |
| 20 | `R7A_elem_cn` | elem \| cn_sig \| a_bin |
| 21 | `R8A_elem` | elem \| a_bin |
| 22 | `R0` | elem \| spin \| nb \| poly \| f5 |
| 23 | `R1_drop_f5` | elem \| spin \| nb \| poly |
| 24 | `R2_drop_spin` | elem \| nb \| poly |
| 25 | `R2a_unrel_poly` | elem \| nb \| poly_reliable |
| 26 | `R2b_binned_mult` | elem \| nb_binned \| poly_reliable \| cn_sig |
| 27 | `R3_nbset_poly` | elem \| nbset \| poly_reliable \| cn_sig |
| 28 | `R4_nbset_cn` | elem \| nbset \| cn_sig |
| 29 | `R5_nbset` | elem \| nbset |
| 30 | `R6_elem_poly` | elem \| poly_reliable \| cn_sig |
| 31 | `R7_elem_cn` | elem \| cn_sig |
| 32 | `R8_elem` | elem |

Terminal residues use the signature `__CATCHALL__|<element>` at rung −1.

### 1.6 Presence families

These signatures are repaired after pruning, in this order, with no support
threshold.

| Family | Signature fields |
|---|---|
| `F_elem_poly` | elem \| poly_reliable |
| `F_elem_cn` | elem \| cn_sig |
| `F_elem_cn_dmin` | elem \| cn_sig \| dmin |
| `F_elem_cn_abin` | elem \| cn_sig \| a_bin |

### 1.7 Net-charge extension (opt-in)

The reference token has no charge field. `tokenizer.charge_policy` (or
`envtok.with_net_charge(profile, policy, key=...)`) adds one. The profile is
then named `<profile>+net-charge`, and every other setting of the profile is
unchanged. The extension is off by default, so `reference-723696d` tokens,
classes and fingerprints are exactly as before.

| Policy | Charge field |
|---|---|
| `disabled` (default) | none: the reference token |
| `net_round` | `round(q)` of the record's net charge q, ties to even, sign kept (−1.5 → −2, 2.5 → 2). A tokenised record without a charge fails with reason `missing_charge` |
| `net_round_zero_if_missing` | as `net_round`, but a record without a charge is treated as neutral and counted as `defaulted` |

**Source.** q is read from the record's scalar metadata under
`tokenizer.charge_key` (default `charge`): `Atoms.info` for ASE iterables and
extended XYZ, and key-value pairs or data for ASE databases and OMat24. It must
be an integer or a float. A string or a boolean fails the record with reason
`invalid_charge` under both enabled policies, because a present but unusable
value is a data error, not a missing one. NaN and infinite floats never reach
the cache (non-finite floats are dropped from metadata), so they count as
missing. Every site of a record carries the same charge field: it describes
the cell, not the atom.

**Token.** The full token gains an eighth field after the moment, written
`q<charge>` in the interchange string:
`E_spin_q<charge>_nb_poly_f5_dmin_abin`, for example
`Fe_0_q-1_Cl-O-O-O_T:4_3_1_2`. The `q` prefix cannot begin a neighbour field,
because element symbols are capitalised, so the parser recognises the field
unambiguously. Non-canonical spellings such as `q+1`, `q01` and `q-0` are
rejected. The signature field `charge` has the same text (`q-1`).

**Ladder.** The net-charge ladder has 12 identity levels × 3 modifier states
= 36 rungs. Charge joins `R0` and `R1`, and a new identity level `R1q`
(`drop_charge`, fields elem | spin | nb | poly) sits between `R1` and `R2`:

| # | Rung | Signature fields |
|---|---|---|
| 0 | `R0_full` | elem \| spin \| charge \| nb \| poly \| f5 \| dmin \| a_bin |
| 1 | `R1dA_drop_f5` | elem \| spin \| charge \| nb \| poly \| dmin \| a_bin |
| 2 | `R1qdA_drop_charge` | elem \| spin \| nb \| poly \| dmin \| a_bin |
| 3 … 11 | `R2dA_drop_spin` … `R8dA_elem` | as reference rungs 2 … 10 |
| 12 … 23 | `R0A`, `R1A_drop_f5`, `R1qA_drop_charge`, `R2A_drop_spin` … | the same with `a_bin` only |
| 24 … 35 | `R0`, `R1_drop_f5`, `R1q_drop_charge`, `R2_drop_spin` … | the same with no modifier |

Charge is the first electronic field to be dropped. It describes the whole
cell, so most sites of a charged supercell are otherwise bulk-like. A charge
state supported by at least T records keeps its own classes at `R0` and
`R1`. A rarer one merges with the other charge states of its environment
from `R1q` onwards, instead of forcing its records into the cover. The
presence families are unchanged. A profile whose ladder and charge policy
disagree (a charge field that no rung uses, or a rung using `charge` without a
charge policy) is rejected.

**Charge field or protected strata.** The two answer different questions.
`stratify_by: [charge]` (§3) fits and covers every charge state separately,
so each state is complete against its own classes. The charge field fits
one set of classes in which charge distinguishes environments only while
its support allows.

**Resolution and accounting.** The charge is resolved when the token index
is built, from the metadata cached at tokenisation. Records that fail for a
missing or invalid charge become `failed`, with a fatal `failures` row
(stage `charge`), and their sites leave the vocabulary and the incidence.
Records already failed or excluded keep their own status. The indexed
`structures` table records `net_charge` (the value found), `charge` (the
token field) and `charge_source` (`supplied`, `defaulted`, `missing`,
`invalid`, or `disabled` for profiles without the field) for every record,
and `envtok inspect` counts a sample by charge source before a large run.

## 2. Coverage classes

**Eligible pool.** The pool contains the records of the fit split (training
by default) that have status `ok` and pass the invalid-record policy.
Failed, excluded, quarantined and held-out records never influence the
classes.

**Coarsening.** The unresolved vocabulary starts as the full tokens present
in the pool. For each rung in order, each unresolved token is projected to
the rung's signature. The support of a signature is the number of distinct
eligible record IDs that contain at least one unresolved token with that
signature. Every unresolved token whose signature has support ≥ T is assigned
to the class `(rung, signature)` and leaves the pool. Tokens still unresolved
after the last rung join the per-element terminal class, whatever its
support. Every pool token ends in exactly one class. Class identity is
`(rung, signature)`. Dense class IDs follow the sort order `(rung, signature)`.

**Counting rules.** Atoms are never counted. Two equivalent sites in one
record contribute one supporting record, and so do two different tokens of
one record that share a signature. Tokens resolved at an earlier rung never
support a later rung.

**Threshold.** T is an integer with T ≥ 2, from `n_coverage + 1` under single
coverage. Classes are rebuilt from the full tokens for every T. The procedure
is not nested, so retained atoms need not change monotonically with T; sweep
T empirically.

**Backends.** Two backends implement the same contract and are tested to be
identical. `memory` uses NumPy over the compact incidence. `duckdb` runs
external aggregation over the Parquet incidence and can spill to disk. `auto`
chooses `duckdb` above 50 million edges.

**Held-out records.** Validation and test tokens are looked up in the frozen
mapping by exact match. Unseen full tokens are out of vocabulary and never
modify the classes.

## 3. Selection (`mode: cover`)

Each record *s* has an atom count `n_s` (its original atom count) and a class
set `C_s`. Selection seeks a low-cost subset S that covers every class, then
enforces the presence families:

1. **User-required records** (`reason=user`).
2. **Mandatory sole suppliers**: the only record containing some class
   (`reason=mandatory`).
3. **Greedy cover**: repeatedly take the record that maximises (newly covered
   classes) / `n_s`. Ties go to fewer atoms, then to the lower stable rank
   (`reason=cover`).
4. **Prune**: visit records largest first (atoms descending, then rank) and
   remove each record whose classes all remain covered without it. Repeat
   until nothing changes. User-required records are never removed.
5. **Repair**, for each presence family in order: while signatures are
   missing, take the record adding the most missing signatures. Ties go to
   fewer atoms, then to the lower rank (`reason=repair:<family>`).
6. Optionally, `prune_after_repair` runs a final prune that preserves both
   the classes and every presence signature. This step is not part of the
   reference.

The stable rank is the position in sorted record-ID order. With
`tie_break: natoms_then_native_id` it is the sorted native-ID order, which
reproduces the reference's `ORDER BY mat_id`.

**Implementations.** The transparent implementation rescans every candidate
with exact fractions. The lazy implementation keeps a max-heap of stale upper
bounds and accepts a re-scored entry only when its fresh key beats every
stored bound. It uses float64 keys when `max_gain · max_atoms² < 2⁵²`, where
they are provably exact, and fractions otherwise. Both implementations
produce identical picks. This is a greedy heuristic, not a certificate of
minimum cost.

**Budget mode.** `mode: budget` is reserved and raises
`NotImplementedError`. The option `n_coverage > 1` of the reference is not
exposed, because it counts site multiplicities rather than distinct records.

**Protected strata.** `stratify_by: [keys]` partitions the eligible pool by
metadata values, then fits and selects within each stratum. The union is
audited globally, and every stratum must be complete against its own
classes. Geometry-only selection therefore cannot merge distinct charge
states or methods. The net-charge token field (§1.7) is the alternative
when charge should refine the classes rather than partition the pool.

## 4. Records, adapters and eligibility

| Field | Contract |
|---|---|
| `record_id` | `dataset@revision:shard:key` by default. `namespaced_native_id` gives `dataset@revision:native_id`, and duplicate native IDs are rejected |
| `source_ref` | dataset, revision, shard, native key or frame, and optional native ID (for example OMat24 `sid`) |
| `atoms` | species, positions, cell and periodicity, unmodified |
| `natoms` | original atom count, which is the selection cost |
| `split` | constant per source, or read from an `info` key |
| `group_id` | from an `info` key when configured |
| `metadata` | JSON-scalar `info` entries (charge, method, temperature and so on) |
| `labels` | energy, forces and stress from stored calculator results, `info` or `arrays`, with declared units; EnvTok never converts units |

The adapters are an in-memory ASE iterable (spoolable to extxyz), extxyz
(byte-offset index, with compressed files read sequentially or staged), ASE
databases, and OMat24 ASE-LMDB shards. The OMat24 shards are read natively
with `ase-db-backends` and opened read-only and lock-free. Rows go through
`_get_row`, then `toatoms`, then `info.update(row.data)`, as in fairchem's
`AseDBDataset`.

**Accounting.** Every discovered record ends as `ok`, `excluded` or `failed`.
Records are never skipped silently.

- `failed`: unreadable records, invalid species, non-finite positions,
  degenerate cells and noncollinear moments. Under a net-charge profile, a
  record without a usable charge also fails, at index time (§1.7).
- `excluded`: unsupported periodicity, and corpus element exclusions applied
  before geometry. The presets are `general`, with no exclusions, and
  `omat24-study`, which excludes exactly La–Lu and Ac–Lr. `exclude_z_at_least`
  provides a Z ≥ 89 policy.
- `invalid_record_policy`: `quarantine`, the default, makes records with
  INVALID sites ineligible. `reference` keeps them with sentinel tokens. Both
  eligible denominators are reported.

Close contacts and large forces are never filtered.

## 5. Storage and fingerprints

```text
<workdir>/cache/<cache-key>/cache.json, plan.json
<workdir>/cache/<cache-key>/partitions/part-NNNNNN/{structures,sites,neighbours,failures}.parquet + _COMMITTED.json
<workdir>/cache/<cache-key>/tokens/<token-key>/{structures,tokens,structure_tokens,site_tokens,failures}.parquet + index.json
<workdir>/selection-T<T>-<selection-fingerprint>/
```

| Fingerprint | Covers |
|---|---|
| source | adapter kind; file identities (size, mtime, optional SHA-256); record options |
| geometry | geometry settings, radius table, asymmetry λ, moment policy, backend versions, implementation version |
| corpus | element exclusions applied before geometry |
| token | geometry fingerprint, bins, moment policy; the charge policy and key only when the charge field is enabled |
| classes | index, split, invalid-record policy, stratum, ladder, presence families, T |
| selection | classes plus every selector setting |

**Cache keys.** The cache key hashes the source, geometry and corpus
fingerprints, the batch size and neighbour retention. A changed source or
setting therefore never resumes from an unrelated checkpoint. Each partition
records the cache fingerprint, and a mismatching partition is rejected.

**What triggers recomputation.** Changing T reuses everything up to the
index. Changing bins reuses the raw per-site measurements, which are always
stored. Enabling, disabling or re-keying the charge field reuses them too:
the charge comes from the cached metadata. Changing the moment policy repeats
the geometry pass. With moments disabled, the stored moments are zero,
noncollinear moments no longer fail a record, and the symmetry analysis sees
no magnetic moments, so its orbits differ. Changing radii or λ needs the
image-resolved neighbour vectors, retained when
`cache_raw_measurements: true`, or else recomputation.

**Token IDs.** Token IDs are ranks of the sorted canonical keys over the
whole cache, so they never depend on partitioning or scheduling.
Persistent identity never uses Python's `hash()`.

## 6. Output contract

| File | Content |
|---|---|
| `selected_ids.txt` | stable record IDs in selection order |
| `selected_manifest.parquet` | source locators, atom counts, split, group, metadata, reason, order |
| `config.resolved.yaml` | the complete configuration with defaults filled in |
| `provenance.json` | versions, platform, reference commit, every fingerprint, source description, peak RSS |
| `coverage.json` | the independent audit and distribution diagnostics |
| `report.html` | a readable report |
| `classes/` | frozen token-to-class mapping, classes, presence targets |
| `subset.extxyz` or `.db` | optional export from the original sources |

The audit recomputes coverage from the selected IDs, the cached incidence and
the frozen mapping with DuckDB SQL. It shares no code with the selector, and
it audits against the classes fitted on the full eligible pool, never on the
subset. It reports:

- structure and atom retention;
- class coverage;
- full-token and frequency-weighted full-token coverage;
- presence coverage per family;
- resolution loss per rung, in tokens, atoms and classes;
- processing quality: statuses, failures and fallbacks, site sources,
  missing labels and moments, records by charge source (charge profiles),
  and eligibility under both policies;
- held-out out-of-vocabulary rates;
- input and selected distributions: elements, composition, cell size,
  source, geometry, the CSM, compression and asymmetry bins, net charge
  (charge profiles), and label summaries.

Exports preserve atom order, cell, periodicity, `info`, arrays and stored
energy, forces and stress. Every calculator is frozen to a
`SinglePointCalculator` holding its stored results, so nothing can
recompute a missing label.

## 7. Execution

Tokenisation streams a deterministic batch plan. Workers write independent
partitions under temporary names and commit them atomically.
`--shard-index/--num-shards` splits the plan across SLURM array tasks, and
`index` merges the shards afterwards. A local pool keeps at most
`max_inflight_batches` batches queued and caps native-library threads per
worker. If a worker dies, the pool is recreated and uncommitted batches are
resubmitted.

DuckDB receives `memory_budget × duckdb_memory_fraction`, leaving the rest
for Python and NumPy, and spills within `temporary_disk_budget`. Before
building its incidence, selection estimates its working state and refuses to
run above `memory_budget`.

### 7.1 Small acceptance check and historical scale measurements

The default selection benchmark now uses 1,000 synthetic records, 200 possible
tokens and T=5. Large runs are opt-in. The separate
[`qualify_periodic.py`](../build_tools/qualify_periodic.py) check exercises the
full geometry path on 20 generated fixtures through native ASE-LMDB storage.
Its [recorded result](../examples/quickstart/qualification-results.json) verifies
serial/two-worker/interrupted-restart agreement, complete coverage and intact
exports. It is not a real OMat24 measurement. Reported peak RSS covers the parent
process only and does not certify a total parallel memory bound.

The earlier synthetic measurements below remain historical context; repeating
them is not a prerequisite for the small acceptance checks.

`build_tools/benchmark_selection.py` builds a synthetic eligible pool with a
Zipf-like token frequency and 2–30 sites per record. It then times
coarsening (memory backend) and the full selection (mandatory, greedy,
prune and four repairs). Measured on a 20-core WSL2 workstation:

| Records | Edges | Tokens | T | Classes | Coarsen | Select | Peak RSS |
|---|---|---|---|---|---|---|---|
| 200,000 | 2.1 M | 19,458 | 50 | 3,345 | 0.8 s | 3.0 s | 0.5 GB |
| 1,000,000 | 10.7 M | 49,656 | 100 | 6,658 | 4.0 s | 17.6 s | 1.3 GB |

These figures exclude the ChemEnv geometry pass, which dominates the cost of
real corpora and must be measured on a representative pilot (varied cell
sizes and distorted structures, not only small ideal cells). For larger
incidences, use the DuckDB coarsening backend and set `memory_budget` so
that selection refuses to run before it exceeds the available memory.
