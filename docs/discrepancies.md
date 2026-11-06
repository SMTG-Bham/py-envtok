# Discrepancy log

This log records three kinds of difference:

- between the paper and the reference code;
- between the reference environment and the modern dependency stack;
- deliberate changes in this package relative to the reference implementation.

Reference: `em819/omat24-chemenv-subset` at commit
`723696de2c26584b8a907ef900690c0caf3c7b68`.

## 1. Paper v1 versus the reference code

EnvTok implements the code's behaviour as profile `reference-723696d`.
`paper-v1` is not implemented, and a hybrid of the two would not reproduce
either.

| Topic | Paper v1 (arXiv:2610.14392v1) | Reference code |
|---|---|---|
| Coarsening ladder | Table S1 lists 13 rungs: `R0_full`, `R1dA` … `R8dA`, `R8A`, `R8` | 33 rungs: 11 identity levels × 3 modifier states, with the modifier loop outermost (`R0A` … `R8A`, `R0` … `R8` included) |
| Magnetic field | "integer-rounded magnetic moment"; the tie rule is unstated | `round(abs(initial_magmom))` with Python ties-to-even (2.5 → 2, 3.5 → 4) |
| Geometry presence floor | (element, polyhedron) | (element, reliability-adjusted polyhedron): poorly fitted or undefined geometries become `UNCL` |
| Presence repair | families listed; candidate choice unstated | four families in order; most new signatures, then fewer atoms, then lower internal index |
| Corpus filter | lanthanides (Z = 57–71) and actinides (Z ≥ 89) | exactly La–Lu and Ac–Lr; elements above Lr would pass |

`envtok.coarsen.paper_table_s1_ladder()` builds the 13-rung ladder for
comparison only. `tests/test_coarsen.py` contains a fixture on which the two
ladders assign different classes: `R0A` versus `R8A_elem`.

## 2. Modern stack versus the reference environment

The golden fixtures (`tests/fixtures/reference/tokens-723696d.json`) come
from the vendored reference functions running in the reference environment:
Python 3.10.13, ASE 3.25.0, pymatgen 2025.6.14, numpy 2.2.6, scipy 1.15.3 and
spglib 2.6.0. The development stack is Python 3.14, ASE 3.29, pymatgen
2026.10 (pymatgen-core 2026.10), numpy 2.4, scipy 1.18 and spglib 2.8.

| Check | Result |
|---|---|
| Effective defaults of every relied-on library call (`introspect_defaults.py`) | identical |
| Token keys on the 20-structure golden corpus (76 sites) | identical; no allowlisted discrepancies |
| Representatives and 1.4/1.7 neighbour-set choices | identical |
| Raw CSM, ρ_min, ρ_max and A | equal within 1e-9; differences are at the 1e-16 floating-point level (for example CSM 0.274747200646913 versus 0.27474720064691294) |
| EnvTok port versus vendored reference code, both on the modern stack | bitwise identical raw values |
| Reference SQL on DuckDB 2.0 | needs `SET lambda_syntax = 'ENABLE_SINGLE_ARROW'` (the reference ran DuckDB 1.4.2); results are identical |

The corpus covers the cases the specification lists:

- octahedral, tetrahedral, cubic and 12-coordinate sites;
- rattled, compressed, stretched and one-sided environments;
- mixed ligands;
- underscore geometries (`BO_3:8`, `BS_2:10`);
- ChemEnv's own `UNKNOWN:14`;
- a ChemEnv failure rescued by CrystalNN (graphene with vacuum);
- a missing covalent radius (Rf);
- positive and negative moments, including ties;
- self-image contacts;
- the extended 1.7 neighbour set (cubic SrTiO₃ O sites).

Bin-boundary sensitivity is not universally excluded. The reference reports
nondeterministic raw CSM for some high-coordination fits. EnvTok seeds
NumPy's global generator per record and tests stability across repeats,
worker counts, symmetry reuse, rigid motions, permutations and wrapping. It
does not promise bitwise reproducibility across arbitrary library versions or
platforms.

## 3. Deliberate differences from the reference implementation

| Area | Reference | EnvTok | Effect |
|---|---|---|---|
| Tie-break order | internal `mat_index`, i.e. `ORDER BY mat_id` (OMat24 `sid`) | sorted stable record ID; `tie_break: natoms_then_native_id` reproduces the reference order | equally good covers may pick different IDs |
| Greedy score | float32 `help / site_count`, key `(-score, site_count, row)` | exact comparisons: float64 under a proven bound, fractions otherwise | identical picks on tested instances; avoids float32 near-ties at scale |
| Selection cost | matrix row sums (token counts) | explicit original `natoms` | identical when every atom is tokenised; robust under other profiles |
| Material identity | grouped by `sid`; a repeated `sid` merges frames | namespaced `dataset@revision:shard:key`; native-ID policy rejects duplicates | no silent merging across shards or trajectories |
| Corpus filter | always excludes La–Lu and Ac–Lr | `general` preset by default; `omat24-study` reproduces the reference | the general package never filters silently |
| Records with INVALID sites | kept with sentinel tokens | `quarantine` by default; `reference` policy keeps them; the report shows both denominators | production default avoids unusable environments |
| Failures | an exception aborts the MPI task | the record is quarantined with stage, category and message; partitions commit atomically | runs complete and can resume |
| `setup_structure` failure | outside the `try`, so it crashes | treated like a ChemEnv failure (CrystalNN rescue) | only affects inputs that crashed the reference |
| Structure passed to ChemEnv | the dataset `Atoms`, calculator included | a clean copy: numbers, positions, cell, pbc and the `magmom` site property | labels cannot influence geometry; identical tokens on fixtures |
| Third-party warnings | printed | captured and counted per record (`warnings_json`) | caller warning filters cannot change results |
| Raw caching | raw ρ_min, ρ_max, A and CSM stored with bins | raw per-site scalars always stored; `cache_raw_measurements` controls image-resolved neighbour vectors | re-binning never repeats geometry |
| OMat24 reader | fairchem `AseDBDataset` (torch stack) | `ase-db-backends` directly, mirroring `AseDBDataset.get_atoms` | no torch dependency; the same row-to-atoms logic |
| Token strings | the parser accepts any field order | keys with unsorted neighbour lists are rejected | two strings can never denote one token |
| Coarsening support | DuckDB `COUNT(DISTINCT mat_index)` | the same contract in NumPy (`memory`) or DuckDB (`duckdb`), tested equal to the reference SQL | external aggregation remains available |
| Final prune | none after the top-up | optional `prune_after_repair` (off) preserving classes and presence | extension only |
| `n_coverage > 1` | counts site multiplicities | not exposed; T ≥ 2 enforced | avoids presenting multiplicity as independent examples |
| Moment policy and caches | not applicable (moments always used) | the moment policy is part of the geometry fingerprint, so `+geometry-only` runs never share partitions with moment-aware runs | switching the policy repeats the geometry pass instead of reusing zeroed moments |
| Net charge | the token has no charge field, so charge states share tokens | optional `tokenizer.charge_policy` adds the record's rounded net charge to every site token (`+net-charge` profiles, 36-rung ladder with `R1q_drop_charge`); off by default | extension only; reference tokens and fingerprints unchanged |

### Packaging notes

- The supported range is standard CPython 3.11–3.14, with a Linux CI matrix
  covering each interpreter. Python 3.15 and free-threaded builds are outside
  the qualification scope.
- `duckdb>=2.0` is required because the Arrow result API changed between
  1.4 and 2.0.
- `lmdb>=1.7.5` is required with `ase-db-backends`. Older releases have no
  Python 3.14 wheel and fall back to a non-functional CFFI build.
