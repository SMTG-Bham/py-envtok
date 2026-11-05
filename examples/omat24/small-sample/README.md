# Small real OMat24 sample

This directory contains **32 training and eight development configurations**
from the `train/rattled-relax/train.aselmdb` member of an existing local OMat24
1M archive. No dataset was downloaded for this check. The 202 MB source shard
is staged temporarily; only these 40 original configurations are retained.

Each record has a different supplied `parent_id`. The development set comes
from the official **training** partition and has no parent overlap with the
small training set. It is not the official validation set. Sampling takes the
first eligible native rows with at most 20 atoms, complete finite energy,
force and stress labels, and no La–Lu or Ac–Lr. The first 59 rows supplied the
40 retained records; 19 were excluded by that element policy.

The sample is deliberately small and ordered. Its 32 training records contain
52 elements, while development includes two additional elements (Ba and Cr).
It is useful for exercising native ingestion, geometry, coverage, restart and
export. It does not establish a representative compression/accuracy result.

The recorded engineering run processed all 32 training records (237 atoms)
without failures or exclusions, covering all 75 classes and every required
presence signature. All 32 records were needed: this small heterogeneous pool
showed **no compression**. Serial, two-worker and interrupted/resumed runs
produced identical records, tokens and selected IDs, and every exported frame
passed geometry, label and metadata round-trip checks. Total times were 45.60 s
serial, 25.58 s with two workers, and 26.27 s after resuming 5.58 s of interrupted
work. Outputs occupied approximately 403 KB per run; the parent-process peak
was 387 MiB, excluding worker memory. See `qualification-results.json` for the
full measurements and dependency versions.

The `.aselmdb` files contain intact native-format records; `.extxyz` files
provide the same structures and labels for small training experiments.
Additional metadata records each original native row, `sid`, parent and source
partition. String metadata use ASE's JSON interchange form so empty values
cannot consume the following key and numeric-looking IDs remain strings.

`provenance.json` records the original archive and member SHA-256 hashes,
sample-file hashes, retained row IDs, exclusions, units and grouping. The
upstream archive checksum has not been independently verified. The dataset
identity follows the existing local archive and its native record metadata.

## Run a small example

From the repository root, run the complete pipeline on the bundled native
training shard; no dataset download is needed:

```bash
uv run --extra omat24 envtok run --config examples/omat24/small-sample/config.yaml
```

With EnvTok and its OMat24 dependencies already installed, the equivalent
command is `python -m envtok run --config examples/omat24/small-sample/config.yaml`.
The configuration uses support threshold **T = 2** to suit this tiny pool:
support is counted in distinct source structures and controls coarsening,
not the number of structures selected.

Expect **32 structures / 237 atoms selected**, covering **75 of 75 classes**
and every required presence signature. This heterogeneous sample shows
**no compression**. Open `report.html` in the printed selection directory,
`runs/omat24-small-demo/selection-T2-<fingerprint>/`, to inspect the audit.
The same directory contains `subset.extxyz` with the selected original
structures and labels. Repeating the command reuses completed tokenisation.

For a larger run that demonstrates compression, see the
[1,024-structure example](../compression-demo/README.md).

Recreate the sample from an existing archive:

```bash
uv run python build_tools/sample_omat24.py --archive /data/omat24_1M.tar.gz
```

Run the bounded engineering qualification (32 records, one and two workers,
plus interruption and resume):

```bash
uv run python build_tools/qualify_periodic.py \
  --source examples/omat24/small-sample/train.aselmdb \
  --revision local-1M-shard-d7be82771964f041 \
  --source-provenance examples/omat24/small-sample/provenance.json \
  --output examples/omat24/small-sample/qualification-results.json
```

The report separates parent-process peak RSS from unmeasured worker memory.
All caches and intermediate exports are temporary. The runner checks the
source hash and revision against the acquisition manifest before processing.

The source data are licensed **CC BY 4.0**, independently of the package's MIT
license. Cite Barroso-Luque et al., [*Open Materials 2024 (OMat24) Inorganic
Materials Dataset and Models*](https://arxiv.org/abs/2410.12771). The
[official dataset card](https://huggingface.co/datasets/facebook/OMAT24)
documents the native format, label units, license and 1M subsplit.
