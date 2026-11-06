# OMat24 examples

Start with the [compression demonstration](compression-demo/README.md) to
see EnvTok reduce a real OMat24 pool from **1,024 to 755 structures** and
**6,705 to 5,077 atoms**. At T=100 this removes 24.3% of the atoms while
covering every fitted class and required presence signature. The example
includes a threshold sweep, random comparisons and checked exports.
The [MACE training comparison](compression-demo/training.md) measures training
time and held-out prediction errors after making a parent-disjoint split of
that sample.

From the repository root, prepare the sample from an existing local archive
and run the supplied configuration:

```bash
uv run --extra omat24 python build_tools/sample_omat24.py \
  --archive /path/to/omat24_1M.tar.gz \
  --train 1024 --development 8 --max-atoms 20 \
  --output-directory runs/omat24-compression-demo/data

uv run --extra omat24 envtok run \
  --config examples/omat24/compression-demo/config.yaml
```

Sample files and generated reports stay under `runs/omat24-compression-demo/`.
Open `report.html` in the printed selection directory; `subset.extxyz`
contains the selected original structures and labels.

| Example | Input | Purpose |
|---|---|---|
| [Compression demonstration](compression-demo/README.md) | 1,024 structures sampled from a local OMat24 archive | Measured compression, coverage and random comparisons |
| [MACE training comparison](compression-demo/training.md) | 823 training structures and 201 parent-disjoint holdouts | Full-pool, compressed and random-subset training with three model seeds |
| [Small sample](small-sample/README.md) | 32 bundled structures; no archive needed | Fast ingestion, tokenisation and export example; no compression |
| [Full-dataset configuration](omat24.yaml) | Your native OMat24 shards | Template for larger runs; edit source paths, revision and resources |

The compression result comes from an ordered sample of small cells in one
training shard. It does not establish dataset-wide compression or model
accuracy. The [detailed example](compression-demo/README.md) reports the
retained full-token coverage and the sample's provenance and limitations.
