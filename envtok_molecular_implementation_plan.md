# EnvTok molecular implementation plan

Updated: 5 November 2026. This replaces the 20k–50k training pilot and 100k/1M
scaling targets with a small development screen. Detailed contracts and supported
settings live in [molecular usage](docs/molecular.md).

## Goal and current state

Select intact finite-molecule configurations while protecting charge, spin,
chemical identity and useful geometry. Data validation/export, graph and fallback
tokens, SOAP, hierarchy, hard budgets, refinement, independent audits and restart
manifests are implemented. The bundled real OMol sample contains 29 configurations
and 650 atoms; it is an ordered engineering sample, not a representative dataset.
See its [provenance](examples/molecular/omol/README.md).

Keep unknown states explicit, preserve source splits and labels, fit all selection
representations only on training data, and report unmet constraints. A smaller
benchmark must not relax state protection to manufacture a complete cover.

## Small acceptance checks

| Check | Size and settings | Output |
| --- | --- | --- |
| State/selection workflow | Existing 29-record sample; random, token-only and combined; 25% atoms; seed 0 | Selected IDs, coverage, unmet targets, actual atom cost and nearest-selected distances. |
| Actual training/evaluation | Neutral-singlet part of that sample; grouped development split; one 25% budget plus full training pool | CPU energy/force baseline, identical held-out records and settings, errors and timings. |
| Restart and export | Same sample and saved manifests | Stable IDs, intact original records, verified cached content. |
| Optional next sample | At most 128 configurations total, with about 20% grouped development holdout | Repeat the same single-budget screen; source and grouping provenance required. |

The full training pool is always a separate control; do not repeat it for every
selector. Default comparison work is three selections at one budget and one seed.
More budgets and seeds require explicit arguments. No new dataset download or
GPU/model checkpoint is needed for the bundled checks.

## Commands

```bash
uv run --extra molecular envtok run --config examples/molecular/omol/config.yaml
uv run --extra molecular envtok benchmark --config examples/molecular/omol/config.yaml
uv run --extra molecular python build_tools/qualify_molecular.py
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py --fraction 0.5
uv run --extra molecular python build_tools/benchmark_mace.py --mace-python /path/to/mace/python
```

An optional selector sweep can use `--fractions 0.25 0.5 --seeds 0 1`.
The cheap accuracy runner fits a linear radial pair model with analytic forces;
it checks that selection, model fitting and held-out evaluation connect correctly.
It is not MACE and cannot establish production MLIP accuracy. The neutral-singlet
scope deliberately leaves mixed-state model qualification outstanding. The MACE
runner uses a separate existing MACE interpreter; it downloads no checkpoint.
It uses one fixed small architecture and 200 CPU training steps for every method.

## Acceptance and next actions

The [recorded CPU screen](examples/molecular/omol/small-accuracy-results.json)
used 17 training and 4 composition-disjoint held-out neutral singlets. All three
subsets covered 7/7 targets, but force RMSE rose by 53% (random), 54% (token-only)
and 30% (combined) relative to the full pool. All failed the 10% regression screen.
Keep the full tiny training pool as the accuracy reference; this result supports
no 25% compression claim.

The [50% pair-model follow-up](examples/molecular/omol/half-budget-accuracy-results.json)
keeps the same split and settings. Combined selection improves to about 12% higher
force RMSE than the full-pool reference, still above the 10% screen threshold.
The larger budget is explicit; the default remains one 25% run.

## Implemented checks and remaining interpretation

- [Workflow qualification](examples/molecular/omol/qualification-results.json)
  checks the actual 29-record sample, saved-run auditing, reused features, stable
  IDs and original state/label/atom alignment on export. CI repeats this check.
- The configurable pair-model runner accepts a supplied source of at most 128
  records, refuses an oversized pool, and keeps its development split independent
  of budget changes. Empty selections and uncovered targets remain visible.
- [Actual MACE training](examples/molecular/omol/mace-small-results.json) uses the
  same 17 training and 4 held-out neutral singlets, one 50% budget and seed 0.
  Four fresh CPU models use the same architecture, optimizer and step count;
  only training records fit elemental references. The report retains initial and
  final errors, a zero-force reference, training losses, processed atoms, source/
  split/config hashes and runtime versions. No held-out checkpoint selection.
- The 200-step result has token-only/combined relative passes, but every trained
  model is worse than predicting zero forces on the tiny holdout. Model accuracy
  and compression usefulness remain unproven despite successful execution.
- A relative force-RMSE screen alone is insufficient: tiny models can all be
  inaccurate. Fixed steps are not convergence or matched processed-atom cost.
  Application tolerances, mixed-state inputs and more representative evaluation
  remain prerequisites for any scientific accuracy or compression claim.

## Deferred

Parallel molecular featurization, sharded/compact storage, 100k/1M throughput,
geometry-only comparisons, broad learning curves, conformer/interaction/MD
qualification and separate reaction/cluster/metal/polymer/periodic profiles stay
outside this small acceptance pass. The current pipeline holds records and
incidence in memory. These limits remain documented, rather than being treated
as prerequisites for a useful local workflow.
