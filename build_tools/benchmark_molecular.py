"""Measure molecular selection, geometry and pipeline stage times at growing sizes.

Synthetic pools isolate how each stage scales with the number of records. The
pool shape is an assumption, roughly the 512-record OMol compression demo:
5-60 atoms per record, Zipf-distributed classes, and mandatory targets and
geometry regions growing as the square root of the pool. Confirm any
extrapolation on real data with ``--config``.

Usage::

    uv run python build_tools/benchmark_molecular.py selection --records 1000 4000
    uv run python build_tools/benchmark_molecular.py geometry --records 1000 4000
    uv run --extra molecular python build_tools/benchmark_molecular.py \
        pipeline --config examples/molecular/omol/config.yaml

``pipeline`` reports each stage's time and peak resident memory, and the size
of every artifact. Peak anonymous memory excludes file pages, such as the
memory-mapped descriptor cache, which the kernel can drop under pressure. To
measure a larger pool, ``replicate`` writes perturbed copies of a source, which
``pipeline --source`` then runs with the configuration's other settings::

    uv run --extra molecular python build_tools/benchmark_molecular.py \
        replicate --source examples/molecular/omol/sample.extxyz --copies 128 \
        --drop-info omol_metadata --output runs/bench/omol-x128.extxyz
    uv run --extra molecular python build_tools/benchmark_molecular.py \
        pipeline --config examples/molecular/omol/config.yaml \
        --source runs/bench/omol-x128.extxyz --workdir runs/bench --fresh

Copies of a few molecules are not a real pool: perturbed coordinates make
nearly every fine token unique, while the coarse classes saturate.
"""

import argparse
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from envtok.execution import peak_rss_bytes
from envtok.molecular.geometry import build_geometry_regions, nearest_selected_distances
from envtok.molecular.selection import Candidate, Target, select_candidates


def selection_pool(n_records: int, seed: int = 0) -> list[Candidate]:
    """Synthetic candidates with atom costs, targets, regions and families."""
    rng = np.random.default_rng(seed)
    n_targets = max(int(4 * n_records**0.5), 2)
    n_regions = max(int(25 * n_records**0.5), 2)
    candidates = []
    for index in range(n_records):
        natoms = int(rng.integers(5, 61))
        targets = np.minimum(rng.zipf(2.0, size=natoms) - 1, n_targets - 1)
        regions = np.minimum(rng.zipf(1.5, size=natoms) - 1, n_regions - 1)
        candidates.append(
            Candidate(
                f"r{index:09d}",
                float(natoms),
                frozenset(f"t{t}" for t in targets.tolist()),
                frozenset(f"g{g}" for g in regions.tolist()),
                f"family{index // 4}",
            )
        )
    return candidates


def descriptor_pool(
    n_records: int, seed: int = 0, dimension: int = 680, n_classes: int = 60
) -> dict[str, list[tuple[str, Any]]]:
    """Synthetic unit descriptors (SOAP-sized by default) with Zipf classes."""
    rng = np.random.default_rng(seed)
    rows = {}
    for index in range(n_records):
        natoms = int(rng.integers(5, 61))
        classes = np.minimum(rng.zipf(1.6, size=natoms) - 1, n_classes - 1)
        vectors = rng.random((natoms, dimension)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        rows[f"r{index:09d}"] = [
            (f"c{c}", vector)
            for c, vector in zip(classes.tolist(), vectors, strict=True)
        ]
    return rows


def _timed(function: Callable[[], object]) -> float:
    start = time.perf_counter()
    function()
    return round(time.perf_counter() - start, 3)


def bench_selection(n_records: int, *, cover_only: bool = False) -> dict[str, Any]:
    """Time cover mode and budget mode with a 10% background reserve."""
    candidates = selection_pool(n_records)
    mandatory = {token: Target() for row in candidates for token in row.targets}
    budget = 0.5 * sum(row.cost for row in candidates)
    result = {
        "records": n_records,
        "mandatory_targets": len(mandatory),
        "cover_seconds": _timed(lambda: select_candidates(candidates, mandatory)),
    }
    if not cover_only:
        result["budget_reserve_seconds"] = _timed(
            lambda: select_candidates(
                candidates,
                mandatory,
                mode="budget",
                budget=budget,
                background_reserve_fraction=0.1,
            )
        )
    return result


def bench_geometry(n_records: int) -> dict[str, Any]:
    """Time region fitting and the nearest-selected audit with 25% selected."""
    rows = descriptor_pool(n_records)
    groups = {record_id: f"family{i // 4}" for i, record_id in enumerate(rows)}
    selected = sorted(rows)[: max(n_records // 4, 1)]
    return {
        "records": n_records,
        "atoms": sum(len(atoms) for atoms in rows.values()),
        "regions_seconds": _timed(
            lambda: build_geometry_regions(rows, group_ids=groups, seed=0)
        ),
        "audit_seconds": _timed(
            lambda: nearest_selected_distances(rows, selected, seed=0)
        ),
    }


class _MemorySampler(threading.Thread):
    """Sample this process's resident memory from ``/proc`` (Linux only)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peaks = {"VmRSS": 0, "RssAnon": 0}
        self.running = Path("/proc/self/status").exists()

    def read(self) -> dict[str, int]:
        values = {}
        with Path("/proc/self/status").open() as stream:
            for line in stream:
                key, _, value = line.partition(":")
                if key in self.peaks:
                    values[key] = int(value.split()[0]) * 1024
        return values

    def reset(self) -> None:
        if self.running:
            self.peaks = self.read()

    def stage_peaks(self) -> dict[str, float | None]:
        """Peak total and anonymous resident MiB since the last reset."""
        if not self.running:
            return {"peak_rss_mib": None, "peak_anon_mib": None}
        current = self.read()
        return {
            "peak_rss_mib": round(
                max(self.peaks["VmRSS"], current["VmRSS"]) / 2**20, 1
            ),
            "peak_anon_mib": round(
                max(self.peaks["RssAnon"], current["RssAnon"]) / 2**20, 1
            ),
        }

    def run(self) -> None:
        while self.running:
            for key, value in self.read().items():
                self.peaks[key] = max(self.peaks[key], value)
            time.sleep(0.02)


def _sizes(directory: Path) -> dict[str, float]:
    return {
        path.name: round(path.stat().st_size / 2**20, 2)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def bench_pipeline(
    config: str,
    *,
    source: str | None = None,
    workdir: str | None = None,
    workers: int | None = None,
    fresh: bool = False,
) -> dict[str, Any]:
    """Run each stage of a real molecular configuration without exporting."""
    import yaml  # noqa: PLC0415

    from envtok.molecular.config import load_molecular_config  # noqa: PLC0415
    from envtok.molecular.pipeline import MolecularPipeline  # noqa: PLC0415

    settings = load_molecular_config(config).resolved()
    if source is not None:
        settings["source"]["path"] = str(Path(source).resolve())
    if workdir is not None:
        settings["execution"]["workdir"] = str(Path(workdir).resolve())
    if workers is not None:
        settings["execution"]["workers"] = workers
    if fresh:
        settings["execution"]["resume"] = False
    sampler = _MemorySampler()
    sampler.start()
    pipeline = MolecularPipeline(yaml.safe_load(yaml.safe_dump(settings)))
    stages = {}
    for stage in ("ingest", "featurize", "build_targets", "select"):
        sampler.reset()
        stages[stage] = {"seconds": _timed(getattr(pipeline, stage))}
        stages[stage] |= sampler.stage_peaks()
    sampler.running = False
    candidates, mandatory, hierarchy, _ = pipeline.build_targets()
    return {
        "records": len(candidates),
        "atoms": int(sum(row.cost for row in candidates)),
        "fine_tokens": len(hierarchy.mapping),
        "mandatory_targets": len(mandatory),
        "stages": stages,
        "recorded": pipeline.timings,
        "feature_cache_mib": _sizes(pipeline.feature_directory),
        "run_mib": _sizes(pipeline.directory),
    }


def replicate(
    source: str, copies: int, output: str, *, sigma: float, drop_info: list[str]
) -> dict[str, Any]:
    """Write ``copies`` perturbed copies of every record with distinct IDs."""
    from ase.io import iread, write  # noqa: PLC0415

    rng = np.random.default_rng(0)
    frames = list(iread(source, ":"))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as stream:
        for copy_index in range(copies):
            batch = []
            for atoms in frames:
                copy = atoms.copy()
                copy.calc = atoms.calc
                copy.positions = atoms.positions + rng.normal(
                    0, sigma, atoms.positions.shape
                )
                for key in drop_info:
                    copy.info.pop(key, None)
                if "sid" in copy.info:
                    copy.info["sid"] = f"{copy.info['sid']}_{copy_index}"
                batch.append(copy)
            write(stream, batch, format="extxyz")
    partial.replace(path)
    return {"output": str(path), "records": copies * len(frames)}


def main() -> None:
    """Run the chosen benchmark and print one JSON line per size."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "kind", choices=["selection", "geometry", "pipeline", "replicate"]
    )
    parser.add_argument("--records", type=int, nargs="+", default=[1000])
    parser.add_argument("--config", help="molecular configuration for 'pipeline'")
    parser.add_argument(
        "--cover-only", action="store_true", help="skip budget mode in 'selection'"
    )
    parser.add_argument("--source", help="source to replicate, or to run instead")
    parser.add_argument("--workdir", help="'pipeline' work directory override")
    parser.add_argument("--workers", type=int, help="'pipeline' featurisation workers")
    parser.add_argument(
        "--fresh", action="store_true", help="'pipeline' recomputes cached features"
    )
    parser.add_argument("--copies", type=int, default=16, help="'replicate' copies")
    parser.add_argument("--output", help="'replicate' output extxyz path")
    parser.add_argument(
        "--sigma", type=float, default=0.05, help="'replicate' noise in Angstrom"
    )
    parser.add_argument(
        "--drop-info",
        action="append",
        default=[],
        help="'replicate' drops this per-record info key (repeatable)",
    )
    args = parser.parse_args()
    if args.kind == "pipeline":
        if args.config is None:
            parser.error("pipeline requires --config")
        results = [
            bench_pipeline(
                args.config,
                source=args.source,
                workdir=args.workdir,
                workers=args.workers,
                fresh=args.fresh,
            )
        ]
    elif args.kind == "replicate":
        if args.source is None or args.output is None:
            parser.error("replicate requires --source and --output")
        results = [
            replicate(
                args.source,
                args.copies,
                args.output,
                sigma=args.sigma,
                drop_info=args.drop_info,
            )
        ]
    elif args.kind == "selection":
        results = [bench_selection(n, cover_only=args.cover_only) for n in args.records]
    else:
        results = [bench_geometry(n) for n in args.records]
    for result in results:
        peak = peak_rss_bytes()
        result["peak_rss_mib"] = None if peak is None else round(peak / 2**20, 1)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
