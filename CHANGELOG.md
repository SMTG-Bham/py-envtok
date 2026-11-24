# Changelog

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
