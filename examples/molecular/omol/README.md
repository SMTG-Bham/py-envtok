# OMol25 training smoke sample

For a larger compression case, see the
[512-configuration native OMol25 example](compression-demo/README.md), with
unbudgeted cover, random/budget comparisons and explicit geometry limits.
It uses a separate native archive; the mirror sample and accuracy screens
below remain available.

This is a real, deliberately small OMol25 sample: **29 finite CHNO
configurations and 650 atoms**, filtered from the first 100 rows of the public
[ColabFit OMol25 train 4M mirror](https://huggingface.co/datasets/colabfit/OMol25_train_4M).
It preserves the source **training** partition, coordinates, atom ordering,
energy/forces, charge, multiplicity, and supplied electronic metadata.
It is an ordered engineering sample, not a random or representative scientific
pilot. Some records describe reactive structures or solvated clusters, so this
example explicitly uses the graph-unknown geometric channel.

The sample has 21 neutral singlets, 6 charge +1 singlets, 1 neutral doublet,
and 1 charge -1 doublet. No artificial charge/spin combinations were generated.

Run from the repository root after installing the molecular dependencies:

```bash
uv run --extra molecular envtok run --config examples/molecular/omol/config.yaml
```

The 25% atom budget is a diagnostic. The report may contain unmet mandatory
targets; it does not establish feasibility or predictive accuracy. The original
smoke run below did not train a model; a separate tiny CPU screen now does.

The recorded `molecular_v1.3` smoke run selected **10 configurations / 154 atoms**
within the 162-atom budget, covering all 15 mandatory chemical/state targets.
It also reported **481 local environments outside every prototype radius**.
That overflow is explicit; chemical coverage does not establish geometric
fidelity or predictive accuracy. [The results snapshot](smoke-results.json)
includes random/token/combined comparisons at 5%, 10%, 25% and 100% budgets
with three selection seeds, plus larger-cutoff and higher-resolution SOAP
checks. This is the historical larger sweep; the current benchmark command
defaults to one 25% budget and seed 0, plus a separate full-pool control.
The comparison harness adds shared state/size constraints, so it has
23 mandatory targets. Descriptor distances from different settings are not
directly comparable.

## Tiny training/evaluation screen

```bash
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py
```

The [recorded screen](small-accuracy-results.json) uses 21 neutral singlets:
17 training configurations and 4 composition-disjoint development configurations.
It fits a small regularized radial pair model with analytic forces, using the
same loss and settings for each selection. The model has no charge/spin or
many-body terms and is not a MACE benchmark.

| Training selection | Configurations / atoms | Holdout force RMSE (eV/angstrom) | Relative to full |
| --- | --- | --- | --- |
| Full pool | 17 / 446 | 1.035 | 1.00 |
| Stratified random | 5 / 102 | 1.580 | 1.53 |
| Token-only | 6 / 111 | 1.596 | 1.54 |
| Combined | 6 / 99 | 1.349 | 1.30 |

All three subsets cover all seven mandatory training targets, but **all fail**
the screen requiring at most 10% higher force RMSE than the full-pool model.
The result shows that coverage alone does not preserve this model's accuracy.
The report includes energy errors, force tails, selected IDs, grouping warnings,
source hash, software versions and coefficients. One tiny ordered sample and one
development split cannot establish a general compression ratio.

The [50% budget follow-up](half-budget-accuracy-results.json) uses the same
training pool, split and model settings. Combined selection reaches force RMSE
1.162 eV/angstrom versus 1.035 for the full pool (about 12% higher), still failing
the 10% screen. Reproduce it with `--fraction 0.5`; the default remains 25%.
Use `--output` to retain a comparison separately; default reports go under `runs/`.

The [workflow qualification](qualification-results.json) checks all 29 records
through selection, saved-run audit, feature reuse, deterministic restart and
state/label/atom-order-preserving export.

## Small from-scratch MACE screen

```bash
uv run --extra molecular python build_tools/benchmark_mace.py \
  --mace-python /path/to/mace/python
```

The [MACE result](mace-small-results.json) uses an existing MACE environment and
downloads no checkpoint. It trains one fresh model for the full pool and each
of the three selectors at a 50% atom ceiling, using 17 training and the same
4 held-out neutral singlets. Each model has one interaction, eight channels,
1,512 parameters and 200 fixed full-batch Adam steps on one CPU thread.

The report includes initial and final held-out errors, a zero-force reference,
the full training-loss trace, processed atoms, checkpoint/config/source hashes,
runtime versions and exact selected IDs. The final fixed step is used regardless
of held-out error. This is a bounded execution screen: the models are not
established as converged, and equal steps process different numbers of atoms.
A relative error screen does not establish useful absolute model accuracy.

| Training selection | Holdout force RMSE (eV/angstrom) | Relative screen |
| --- | --- | --- |
| Full pool | 0.894 | Reference |
| Stratified random | 1.009 | Fail |
| Token-only | 0.855 | Pass |
| Combined | 0.928 | Pass |

**All four trained models are worse than predicting zero forces** (RMSE 0.758)
and the initial model (0.756) on this tiny holdout. Decreasing training loss has
not translated to useful held-out force accuracy. The two relative passes
therefore demonstrate neither a good molecular potential nor useful compression.

## Source reproduction

`preview.json.gz` stores the complete 1,021,912-byte, 100-row source response
compressed with gzip (decompress using `gzip -dc preview.json.gz`), including
rows excluded by the declared CHNO/60-atom filter. `provenance.json` records its
SHA-256, the observed mirror revision, selection row indices, source IDs,
licence, units, and output SHA-256. The Hugging Face viewer API is not pinned to
a revision; the saved response hash is the exact reproducibility reference.

Reproduce the extxyz and provenance files offline:

```bash
uv run python build_tools/download_omol_sample.py
```

Add `--download` to request the same bounded preview again. The downloader
checks the mirror revision and exact payload digest before writing. It never
downloads a full dataset shard or accesses the gated original repository.

The release-specific field mapping is explicit:

| Package field | ColabFit/OMol source |
| --- | --- |
| Total charge | `property_metadata.charge` |
| Molecular spin multiplicity | `property_metadata.spin` |
| Energy, eV | `energy` |
| Forces, eV/angstrom | `atomic_forces` |
| Method and basis | `method` and `property_metadata.basis_set` |
| Original calculation locator | `property_metadata.source` |

ColabFit's top-level `multiplicity` counts duplicate occurrences; **it is not
molecular spin multiplicity**. Its schema and units are documented in the
[ColabFit configuration schema](https://materials.colabfit.org/docs/configuration_schema).
The spin mapping agrees with the
[official FAIRChem OMol loader](https://github.com/facebookresearch/fairchem/blob/3b0fa63389d1955e78a806078100afb25b5922c9/benchmarks/omol/force_mae.py).

No reliable parent-family mapping or supplied bond graph is established by
this converter. Grouping uncertainty is reported by the package, which uses
its conservative composition fallback. All original metadata remains in the
saved response and the exported `omol_metadata`/`colabfit_metadata` fields.

The data is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original work: Levine et al., *The Open Molecules 2025 Dataset, Evaluations,
and Models*, [arXiv:2505.08762](https://arxiv.org/abs/2505.08762) (2025).
ColabFit redistribution: Levine et al., *OMol25 train 4M*,
[DOI:10.60732/b6f9382a](https://doi.org/10.60732/b6f9382a) (2025).
Changes here: bounded row extraction, CHNO/size filtering, explicit field
mapping, and precision-preserving extxyz conversion.
