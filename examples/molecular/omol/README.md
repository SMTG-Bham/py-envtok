# OMol25 training smoke sample

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
targets; it does not establish feasibility or predictive accuracy. No model
has been trained on this sample.

The recorded `molecular_v1.3` smoke run selected **10 configurations / 154 atoms**
within the 162-atom budget, covering all 15 mandatory chemical/state targets.
It also reported **481 local environments outside every prototype radius**.
That overflow is explicit; chemical coverage does not establish geometric
fidelity or predictive accuracy. [The results snapshot](smoke-results.json)
includes random/token/combined comparisons at 5%, 10%, 25% and 100% budgets
with three selection seeds, plus larger-cutoff and higher-resolution SOAP
checks. The comparison harness adds shared state/size constraints, so it has
23 mandatory targets. Descriptor distances from different settings are not
directly comparable.

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
