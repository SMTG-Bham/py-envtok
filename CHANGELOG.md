# Changelog

## Unreleased

### Fixed

- Molecular graph perception no longer falls back for molecules with a nonzero
  coordinate smaller than 1e-4 in magnitude. Such coordinates were written to
  RDKit in exponent notation, which its XYZ reader rejects, so perception
  silently reported `fallback_perception_unsupported`. Coordinates are now
  written positionally and parse to the same values; other coordinates keep
  their previous text. The molecular implementation version is now
  `molecular_v1.4`, so runs and feature caches from `molecular_v1.3` are not
  reused.

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
