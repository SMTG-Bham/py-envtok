# Changelog

## Unreleased

### Added

- Molecular `execution.memory_budget`. Before featurisation, the pipeline adds
  its resident memory, which includes the loaded records, to
  `estimate_molecular_bytes` for the remaining stages, and refuses to continue
  above the budget. The budget does not change run fingerprints.
- `geometry.DescriptorPool` lets `build_geometry_regions` and
  `nearest_selected_distances` take descriptors as arrays, such as the
  memory-mapped cache, with results identical to the per-atom mapping.

### Changed

- Molecular feature caches and run targets are stored as Parquet tables and
  NumPy arrays, indexed by small `features.json` and `targets.json` manifests
  that record each file's digest. Token strings are stored once and shared when
  loaded, signatures are read only when looked up, and audits read only the
  tables they need. `targets.json` no longer repeats the token signatures,
  which stay in the feature cache. The molecular implementation version is now
  `molecular_v1.6`, so new runs recompute `molecular_v1.5` feature caches;
  saved `molecular_v1.5` and earlier runs still load, audit and export.
- Periodic tokenisation scores ChemEnv's candidate neighbour orderings in
  batches rather than one pymatgen call each, with bit-identical symmetry
  measures. On the bundled OMat24 sample it is about 1.5 times faster on one
  worker and 1.3–1.4 times with four. The batched finder is used only with the
  pymatgen code it was checked against, and only if a start-up comparison with
  pymatgen on the running machine's BLAS matches exactly; otherwise pymatgen's
  own finder runs.
- Molecular hierarchy fitting counts token support without a set per token,
  and parallel featurisation keeps at most `2 * workers` batches of at most 64
  records waiting to be written, rather than every finished batch. Selections
  and features are unchanged.
- On 29,696 perturbed OMol records with 4 workers, peak anonymous memory fell
  from 4,569 to 1,991 MiB, the frozen targets from 1,272 to 222 MiB and the
  feature metadata from 441 to 37 MiB, with byte-identical selections,
  coverage reports and selected manifests. A 100,050-record run peaked at
  5,376 MiB, of which 2,650 MiB was the loaded records. See the
  [molecular guide](docs/molecular.md#scale-measurements).

## 0.1.0a2 — 2026-11-24

### Fixed

- Molecular graph perception no longer falls back for molecules with a nonzero
  coordinate smaller than 1e-4 in magnitude. Such coordinates were written to
  RDKit in exponent notation, which its XYZ reader rejects, so perception
  silently reported `fallback_perception_unsupported`. Coordinates are now
  written positionally and parse to the same values; other coordinates keep
  their previous text. The molecular implementation version is now
  `molecular_v1.4`, so runs and feature caches from `molecular_v1.3` are not
  reused.

### Changed

- Molecular selection, geometry regions, the nearest-selected audit and token
  construction are faster and use less memory, with identical selections and
  tokens. Selection uses lazy-heap greedy stages that score only each
  component's own targets; the previous rescanning loops remain available as
  `select_candidates(..., algorithm="transparent")`. `build_tools/benchmark_molecular.py`
  measures how the stages scale.
- Molecular featurisation can run in local worker processes with
  `execution.workers` and `execution.threads_per_worker`; worker settings do not
  change run fingerprints.
- The molecular feature cache now stores descriptors in one memory-mapped
  `descriptors.npy` instead of a compressed `descriptors.npz` with one member
  per record, and `features.json` stores each signature once. The molecular
  implementation version is now `molecular_v1.5`, so new runs recompute
  `molecular_v1.4` feature caches; saved older runs still load their
  `descriptors.npz` for auditing and export.
- Molecular stages no longer re-hash the whole source on every call. `select()`
  and `export()` hash it in full; other stages re-hash it only when its stat
  signature changes, as described in the [molecular guide](docs/molecular.md).
  `run()` audits the selection once instead of again before exporting.

## 0.1.0a1 — 2026-11-24

Initial alpha release. Public APIs and saved-run formats may change in later
versions; keep the source data and resolved configuration needed to rerun a job.

### Included

- Reference-compatible local-environment tokens, support-based coarsening,
  whole-structure selection and independent class/presence coverage audits.
- ASE iterable, extended XYZ, ASE database and optional OMat24 ASE-LMDB inputs;
  original structure and label preservation on export.
- Resumable token caches, bounded local workers, external token aggregation,
  sharded tokenisation, selection reports and a Python API and CLI.
- Optional net-charge tokens and protected metadata strata.
- Experimental `molecular_v1` workflow with explicit charge/spin state, graph
  and SOAP representations, conservative deduplication, hard atom budgets and
  reports of unmet constraints.
- Reference golden fixtures, bundled periodic and molecular engineering
  samples, and isolated wheel/sdist end-to-end qualification.

### Supported scope and known limits

- Standard CPython 3.11–3.14 on Linux are the CI qualification targets. Python 3.15,
  free-threaded builds and other operating systems are unqualified.
- Coverage is defined over fitted classes and presence constraints in the
  eligible training pool. It does not imply complete full-token coverage,
  preserved sampling distributions or equivalent model accuracy.
- Periodic selection supports coverage mode with atom cost; periodic hard
  budgets are not implemented.
- Molecular selection is an experimental serial, in-memory workflow. A hard
  budget can leave constraints unmet; inspect the audit before using the subset.
- Small bundled experiments establish workflow correctness. Molecular model
  accuracy remains unvalidated, and the bundled periodic development data is
  unsuitable for its small accuracy model. No general compression ratio,
  million-record throughput or scientific accuracy claim is made.

See the [usage guide](README.md), [molecular contracts](docs/molecular.md) and
[release procedure](docs/releasing.md).
