# EnvTok periodic implementation plan

The November 2026 acceptance plan focuses on small, reproducible checks.
Algorithm details live in the
[specification](specification.md), [discrepancy log](discrepancies.md)
and [reference-fixture guide](../build_tools/reference/README.md).

## Goal and current state

Select original whole periodic structures with independently verified chemical
coverage. The reference tokenizer, adapters (including native ASE-LMDB),
coarsening, selection, export, cache/restart and bounded workers are implemented.
API/CLI, tests, examples and package builds are also present.

Keep the `reference-723696d` behavior, stable source IDs, original atom/label
alignment, explicit exclusions and training/held-out separation. Reduced
benchmarks do not change these correctness requirements.

## Small acceptance checks

| Check | Size and settings | Pass condition |
| --- | --- | --- |
| Local pipeline | Existing 20-structure fixture; T=2 | Independent class/presence audit passes; exported originals retain labels. |
| Native storage and restart | Same fixture in an ASE-LMDB shard; 1 and 2 workers, then resume | Completed records and selected IDs agree; cached partitions are reused. |
| Selection resources | 1,000 synthetic records, 200 possible tokens, T=5 | Record time, peak RSS, selected atoms and estimated selection memory. |
| Installed package | Run the whole fixture quickstart from built wheel and sdist | Tokenize, select, audit and export work outside the source installation. |
| Real-data follow-up | Bundled 32-record native OMat24 training sample; 8 fixed development records | Record source revision, geometry time, failures, disk/RSS and coverage. |

The native fixture checks the actual storage and geometry path, but is generated
locally: it does not count as a real OMat24 experiment. Historical synthetic
200k/1M results remain in the specification; repeating those runs is unnecessary.

## Commands

```bash
uv run python examples/quickstart/quickstart.py
uv run --extra omat24 python build_tools/qualify_periodic.py
uv run python build_tools/benchmark_selection.py
uv run python build_tools/benchmark_periodic_accuracy.py
uv build --no-sources
uv run python build_tools/qualify_artifacts.py
```

The periodic runner records a small end-to-end report; the selection runner
prints JSON. Use explicit flags for a larger synthetic run only after the small
checks expose a reason to do so. CI runs the installed-package quickstart.

## Implementation status

- The [native fixture check](../examples/quickstart/qualification-results.json) passed:
  20 structures, 16 selected, 56/76 atoms, 26/26 classes, identical worker/restart
  results and preserved exports.
- The [installed-artifact check](../examples/quickstart/artifact-qualification-results.json)
  installs wheel and sdist with their dependencies in separate fresh environments.
  Both full quickstarts passed. CI runs the same acceptance command.
- A [real OMat24 sample](../examples/omat24/small-sample/README.md) now contains
  32 training and 8 parent-disjoint development records from a local official 1M
  archive. Native row IDs, original labels, hashes, exclusions and licence survive.
  Its real native-storage/restart/export check passed with 75/75 classes and zero
  failed records. All 32 structures were required, so no compression is claimed.
  The run exposed and fixed empty/numeric-string metadata corruption in XYZ
  export. CI now repeats the bounded real-data check.
- The periodic accuracy runner implements one T, one matched-atom random seed
  and full-pool/EnvTok/random comparisons with a small periodic energy/force model.
  It refuses unseen development elements, overlapping parent groups or an overly
  broad vocabulary. The bundled heterogeneous sample has unseen Ba/Cr and cannot
  qualify this small model; retain that explicit data gate instead of dropping
  held-out records.
- The larger archive example measures [coverage on 1,024 structures](../examples/omat24/compression-demo/README.md).
  Its [MACE experiment](../examples/omat24/compression-demo/training.md) fits
  covers on 823 training structures and evaluates 201 parent-disjoint
  holdouts. Both covers reduce optimiser time. Mean force RMSE exceeds the
  zero-force baseline for all four training selections.
- [Publication readiness](../examples/quickstart/publication-readiness.json) records
  the pre-release name check and expected workflow settings. For candidate
  qualification and publishing, see the [release procedure](releasing.md).

## Deferred

Million-record qualification, full OMat24 reproduction, threshold/seed learning
curves, MD/application validation, periodic budget optimization, repeated
witnesses, `paper-v1` and new accelerators are optional follow-ups. Do not claim
preserved MLIP accuracy or full-corpus throughput from the small checks.
