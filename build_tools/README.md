# Build and benchmark tools

Standalone scripts for acceptance runs, accuracy screens, dataset sampling and
packaging checks. They are not part of the installed package.

## Conda environment

uv is the recommended way to work on this project (see the root `README.md`).
If you prefer Conda, two environment files are provided. Conda supplies Python
from conda-forge and pip installs this checkout in editable mode.

For development, create `envtok-dev` from the repository root. It includes the
`omat24` and `molecular` extras, so the full test suite runs, and the `dev`
dependency group (pytest, mypy, ruff and prek):

```bash
conda env create -f build_tools/environment_dev.yml
conda activate envtok-dev
python -m pip install --group dev
```

The last step is separate because pip cannot install a dependency group from
an environment file; run it from the repository root, where it reads
`pyproject.toml`. Then run the same checks as CI without the `uv run` prefix:

```bash
prek install
prek run --all-files
mypy
pytest
```

To only run EnvTok from the checkout, without extras or development tools:

```bash
conda env create -f build_tools/environment.yml
conda activate envtok
```

Add an extra later with, for example, `python -m pip install -e '.[molecular]'`.
Both environments resolve the latest versions allowed by `pyproject.toml`
rather than the exact versions pinned in `uv.lock`; the development file also
keeps NumPy below 2.5, whose deprecation warnings from ASE fail the test suite.
In an activated environment,
drop the `uv run` prefix and its `--extra` options from the commands below, for
example `python build_tools/benchmark_selection.py`. `uv build` and
`qualify_artifacts.py` still need uv on PATH.

## Small acceptance runs

From the repository root, check selection, restart and export on the bundled
data. The periodic runner copies 20 synthetic configurations into native LMDB
storage and compares serial, parallel and resumed runs. The molecular runner
uses 29 real OMol configurations and checks selected IDs, charge, multiplicity
and labels on export.

```bash
uv run --extra omat24 python build_tools/qualify_periodic.py
uv run --extra molecular python build_tools/qualify_molecular.py
```

Read `runs/periodic-check.json` and `runs/molecular-qualification.json`, or
choose another destination with `--output`. To check the native reader against
a real shard, use the [32-record OMat24 instructions](../examples/omat24/small-sample/README.md).

## Measure runtime and memory

```bash
uv run python build_tools/benchmark_selection.py
uv run python build_tools/benchmark_molecular.py selection --records 1000 4000
uv run --extra molecular python build_tools/benchmark_molecular.py pipeline \
  --config examples/molecular/omol/config.yaml --workdir runs/omol-timing
```

The first command measures periodic coarsening and selection on a synthetic
incidence table of 1,000 records, 200 possible tokens and T=5. The second
times molecular cover and budget selection on synthetic candidates. For
end-to-end molecular timings through selection, use the third command and
record whether the feature cache was reused. Pass `--fresh` to recompute
features. The [scale measurements](../docs/molecular.md#scale-measurements)
give the descriptor settings, pool sizes and memory measurements from larger
runs.

## Compare model errors on held-out data

```bash
uv run python build_tools/benchmark_periodic_accuracy.py
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py
uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py \
  --fraction 0.5 --workdir runs/half-budget --output runs/half-budget/results.json
```

The [real periodic sample](../examples/omat24/small-sample/README.md) includes a
native source, fixed development split and exact acquisition provenance. Its
accuracy check reports missing development elements and excessive vocabulary
before training, rather than quietly reducing the holdout. A supplied suitable
small-chemistry source can run the periodic model comparison. Read
`runs/periodic-accuracy/results.json` for the preflight outcome. The molecular
pair-model runs use the same grouped development split at two atom ceilings
and save their comparisons in `runs/small-accuracy/results.json` and
`runs/half-budget/results.json`.

## Native archive examples

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

For a MACE screen that runs on the bundled data, see the [29-record sample
instructions](../examples/molecular/omol/README.md#small-from-scratch-mace-screen).

## Qualify built packages

```bash
uv build --no-sources
uv run python build_tools/qualify_artifacts.py
```

`qualify_artifacts.py` requires `uv` on PATH (or `--uv /path/to/uv`) and installs
the wheel and sdist in separate environments without system-site-packages.
Use `--offline` only when the artifact and all dependency wheels are cached.
See the [release procedure](../docs/releasing.md) for the expected artifacts
and the checks on each candidate.
