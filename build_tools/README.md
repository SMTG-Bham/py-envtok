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

The optional MACE runner uses a separate installed MACE interpreter, keeping
PyTorch out of the selection dependencies. It trains four fresh CPU models for
200 fixed steps at a single 50% atom ceiling; it records initial/final errors,
zero-force error and training losses. The run proves the training workflow, not
convergence, a model advantage, or simulation accuracy. The pair-model runner
also accepts `--source`, `--label-namespace` and a maximum of 128 records.

`qualify_artifacts.py` requires `uv` on PATH (or `--uv /path/to/uv`) and installs
the wheel and sdist in separate environments without system-site-packages.
Use `--offline` only when the artifact and all dependency wheels are cached.
