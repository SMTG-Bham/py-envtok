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
"""

import argparse
import json
import time
from collections.abc import Callable
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


def bench_pipeline(config: str) -> dict[str, Any]:
    """Run each stage of a real molecular configuration without exporting."""
    from envtok.molecular.pipeline import MolecularPipeline  # noqa: PLC0415

    pipeline = MolecularPipeline(config)
    stages = {}
    for stage in ("ingest", "featurize", "build_targets", "select"):
        stages[f"{stage}_seconds"] = _timed(getattr(pipeline, stage))
    return {
        "records": len(pipeline.ingest().candidates),
        **stages,
        "recorded": pipeline.timings,
    }


def main() -> None:
    """Run the chosen benchmark and print one JSON line per size."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["selection", "geometry", "pipeline"])
    parser.add_argument("--records", type=int, nargs="+", default=[1000])
    parser.add_argument("--config", help="molecular configuration for 'pipeline'")
    parser.add_argument(
        "--cover-only", action="store_true", help="skip budget mode in 'selection'"
    )
    args = parser.parse_args()
    if args.kind == "pipeline":
        if args.config is None:
            parser.error("pipeline requires --config")
        results = [bench_pipeline(args.config)]
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
