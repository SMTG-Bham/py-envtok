# Conda environment

uv is the recommended way to work on this project (see the root `README.md`).
If you prefer Conda, create and activate the optional environment from the
repository root, then install the package and its development tools with pip:

```bash
conda env create -f build_tools/environment.yml
conda activate envtok
python -m pip install --group dev -e .
```

This path resolves the latest versions allowed by `pyproject.toml` rather than
the exact versions pinned in `uv.lock`.

## Small acceptance runs

Run from the repository root:

```bash
uv run python build_tools/benchmark_selection.py
uv run --extra omat24 python build_tools/qualify_periodic.py
uv run --extra molecular python build_tools/qualify_molecular.py
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py --fraction 0.5
uv run --extra molecular python build_tools/benchmark_mace.py --mace-python /path/to/mace/python
uv run python build_tools/benchmark_periodic_accuracy.py
uv build --no-sources
uv run python build_tools/qualify_artifacts.py
```

The selection benchmark defaults to 1,000 synthetic records, 200 possible tokens
and T=5. The periodic check uses 20 generated fixtures in native LMDB storage,
including real geometry, two worker counts and interrupted restart. Supply
`--source` and `--revision` to the periodic runner for a small real native shard.
The molecular screen trains a cheap CPU energy/force model on the bundled OMol
neutral singlets and evaluates a grouped development holdout. All are preliminary
checks; generated fixtures and tiny fitted models do not establish scientific
accuracy or large-corpus throughput. See each script's `--help` for output paths.

The [real periodic sample](../examples/omat24/small-sample/README.md) includes a
native source, fixed development split and exact acquisition provenance. Its
accuracy check reports missing development elements and excessive vocabulary
before training, rather than quietly reducing the holdout. A supplied suitable
small-chemistry source can run the periodic model comparison.

For measured compression, the [larger OMat24 example](../examples/omat24/compression-demo/README.md)
uses `sample_omat24.py` to retain 1,024 distinct-parent structures from a local
archive. At T=100 it keeps 755 structures and 5,077 of 6,705 atoms, a 24.3%
atom reduction with complete class and required-presence coverage. The example
documents sampling, the threshold sweep, random comparisons and verified
exports; it does not establish downstream model accuracy.

The [OMat24 MACE training comparison](../examples/omat24/compression-demo/training.md)
uses `benchmark_omat_compression_training.py` with a separate installed MACE
interpreter. It freezes a parent-disjoint holdout before tokenisation and
selection, then trains full-pool, T=2, T=100 and matched-budget random models
with three model seeds. The periodic worker preserves cells and image
neighbors and reports force/energy errors, training time and processed atoms.

The [native OMol25 example](../examples/molecular/omol/compression-demo/README.md)
uses `sample_omol.py` to retain 512 finite CHNO configurations from a local
4M archive. Its unbudgeted cover keeps 3,276 of 10,938 atoms (70.0% fewer),
covering all mandatory element/state targets and modeled geometry regions.
Prototype overflow remains substantial and is reported alongside a finer
geometry check, random baselines and budget comparisons. Native-source and
export checks preserve metadata, charge/spin, atom ordering and labels; this
is a selection demonstration, not evidence of model accuracy retention.

The [native OMol training comparison](../examples/molecular/omol/compression-demo/training.md)
uses `benchmark_omol_compression_training.py` with a separate installed MACE
interpreter. It freezes a composition-grouped neutral-singlet holdout before
selection, then trains 12 small CPU models: full pool, two EnvTok covers and
matched-budget random selection, each with three model seeds and 1,000 fixed
full-batch epochs. It records prediction errors alongside training time and
processed atoms; coverage alone is not the accuracy criterion.

The older bundled-sample `benchmark_mace.py` runner uses a separate installed
MACE interpreter, keeping PyTorch out of the selection dependencies. It trains
four fresh CPU models for
200 fixed steps at a single 50% atom ceiling; it records initial/final errors,
zero-force error and training losses. The run proves the training workflow, not
convergence, a model advantage, or simulation accuracy. The pair-model runner
also accepts `--source`, `--label-namespace` and a maximum of 128 records.

`qualify_artifacts.py` requires `uv` on PATH (or `--uv /path/to/uv`) and installs
the wheel and sdist in separate environments without system-site-packages.
Use `--offline` only when the artifact and all dependency wheels are cached.
