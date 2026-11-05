"""Run a small periodic pipeline check through a native ASE-LMDB source.

By default, the 20 generated reference fixtures are written to a temporary
LMDB shard. This checks real ChemEnv execution, serial/parallel agreement,
interrupted-run recovery, independent coverage and exported labels. It is an
engineering check, not real OMat24 or downstream model qualification.

Usage::

    uv run python build_tools/qualify_periodic.py
    uv run python build_tools/qualify_periodic.py --source small.aselmdb --revision v1

User-supplied sources are read in place and limited to 100 records by default.
Only the JSON report survives; temporary caches and exports are removed.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import ase.db
import ase.io
import numpy as np

from envtok.adapters.asedb import close_shared_databases
from envtok.errors import EnvTokError
from envtok.execution import peak_rss_bytes
from envtok.export import selected_structures
from envtok.fingerprint import package_versions
from envtok.pipeline import Pipeline
from envtok.provenance import DEPENDENCIES
from envtok.selection import SelectionResult
from envtok.store.cache import TokenCache
from envtok.store.executor import FaultPlan

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "corpus.extxyz"


def _fixture_shard(directory: Path) -> Path:
    """Write generated structures, metadata and labels to one native shard."""
    path = directory / "generated.aselmdb"
    database = ase.db.connect(str(path))
    try:
        for atoms in ase.io.iread(FIXTURE, format="extxyz"):
            database.write(atoms, data=dict(atoms.info))
    finally:
        database.close()
    return path


def _pipeline(source: Path, revision: str, workdir: Path, workers: int) -> Pipeline:
    """Build a bounded, explicitly train-only engineering configuration."""
    return Pipeline(
        {
            "schema_version": 1,
            "source": {
                "adapter": "omat24",
                "path": str(source),
                "name": "periodic-check",
                "revision": revision,
                "split": "train",
                "checksum": "full",
            },
            "coverage": {"min_structure_support": 2},
            "execution": {
                "workdir": str(workdir),
                "workers": workers,
                "batch_size": 5,
                "max_inflight_batches": 2,
                "memory_budget": "2GB",
                "temporary_disk_budget": "2GB",
            },
        }
    )


def _check_export(pipeline: Pipeline, result: SelectionResult, path: Path) -> int:
    """Compare the written frames with selected originals and stored labels."""
    count = 0
    originals = selected_structures(result, pipeline.corpus.adapter)
    for original, exported in zip(
        originals, ase.io.iread(path, format="extxyz"), strict=True
    ):
        np.testing.assert_array_equal(original.numbers, exported.numbers)
        np.testing.assert_array_equal(original.pbc, exported.pbc)
        for left, right in (
            (original.positions, exported.positions),
            (original.cell.array, exported.cell.array),
        ):
            np.testing.assert_allclose(left, right, rtol=0, atol=1e-7)
        for key, value in original.arrays.items():
            if np.issubdtype(value.dtype, np.number):
                np.testing.assert_allclose(
                    value, exported.arrays[key], rtol=0, atol=1e-7
                )
            else:
                np.testing.assert_array_equal(value, exported.arrays[key])
        for key, value in original.info.items():
            np.testing.assert_equal(value, exported.info[key])
        stored = original.calc.results if original.calc is not None else {}
        written = exported.calc.results if exported.calc is not None else {}
        for key in ("energy", "forces", "stress"):
            if (key in stored) != (key in written):
                raise RuntimeError(f"export changed availability of {key}")
            if key in stored:
                np.testing.assert_allclose(stored[key], written[key], rtol=0, atol=1e-7)
        if original.info["envtok_record_id"] != exported.info["envtok_record_id"]:
            raise RuntimeError("export changed source identity or frame order")
        count += 1
    return count


def _run(pipeline: Pipeline) -> tuple[dict[str, Any], TokenCache]:
    """Time complete processing, independently audit and check export fidelity."""
    started = time.perf_counter()
    cache = pipeline.cache()
    tokenized = time.perf_counter()
    result = pipeline.fit_and_select(cache)
    selected = time.perf_counter()
    directory = pipeline.selection_dir(result)
    result.write_manifest(directory, config=pipeline.config.resolved())
    result.validate()
    path = pipeline.export(result, directory)
    if path is None:
        raise RuntimeError("qualification requires an exported subset")
    exported = _check_export(pipeline, result, path)
    finished = time.perf_counter()
    report = json.loads((directory / "coverage.json").read_text())
    ids = cache.read("structures", ["record_id"])["record_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise RuntimeError("completed cache contains duplicate record IDs")
    return {
        "workers": pipeline.execution.workers,
        "cache": cache.summary,
        "structures": report["structures"],
        "atoms": report["atoms"],
        "classes": report["classes"],
        "presence": report["presence"],
        "processing": report["processing"],
        "selected_ids": result.record_ids,
        "exported_frames_checked": exported,
        "seconds": {
            "tokenize_and_index": round(tokenized - started, 3),
            "coarsen_and_select": round(selected - tokenized, 3),
            "audit_manifest_export_check": round(finished - selected, 3),
            "total": round(finished - started, 3),
        },
        "disk_bytes": sum(
            p.stat().st_size for p in pipeline.workdir.rglob("*") if p.is_file()
        ),
    }, cache


def _equal_caches(left: TokenCache, right: TokenCache) -> None:
    """Require equal completed records, tokens and structure/token incidence."""
    for name, columns, order in (
        ("structures", ["record_id", "status", "status_reason"], ["record_id"]),
        ("tokens", ["token_id", "token_key"], ["token_id"]),
        (
            "structure_tokens",
            ["record_id", "token_id", "multiplicity"],
            ["record_id", "token_id"],
        ),
    ):
        sort = [(key, "ascending") for key in order]
        if (
            not left.read(name, columns)
            .sort_by(sort)
            .equals(right.read(name, columns).sort_by(sort))
        ):
            raise RuntimeError(f"serial/parallel/restart mismatch in {name}")


def _interrupt(pipeline: Pipeline) -> tuple[dict[Path, int], float]:
    """Stop after the first committed batch, retaining it for a later run."""
    started = time.perf_counter()
    try:
        pipeline.tokenizer().tokenize(
            pipeline.corpus, pipeline.workdir, faults=FaultPlan(raise_after=1)
        )
    except EnvTokError as error:
        if "deliberate" not in str(error):
            raise
    else:
        raise RuntimeError("deliberate interruption did not occur")
    partitions = pipeline.cache_dir() / "partitions"
    existing = {p: p.stat().st_mtime_ns for p in partitions.rglob("*") if p.is_file()}
    if not existing:
        raise RuntimeError("interrupted run did not retain a committed batch")
    return existing, time.perf_counter() - started


def qualify(
    source: Path, revision: str, workdir: Path, max_records: int
) -> dict[str, Any]:
    """Run serial, parallel and interrupted/resumed cases on the same source."""
    serial = _pipeline(source, revision, workdir / "serial", 1)
    for count, _ in enumerate(serial.corpus.adapter.locators(), start=1):
        if count > max_records:
            raise ValueError(
                f"source exceeds --max-records {max_records}; use a small pilot"
            )
    serial_report, serial_cache = _run(serial)
    parallel_report, parallel_cache = _run(
        _pipeline(source, revision, workdir / "parallel", 2)
    )
    restart_dir = workdir / "restart"
    existing, interruption_seconds = _interrupt(
        _pipeline(source, revision, restart_dir, 1)
    )
    restart_report, restart_cache = _run(_pipeline(source, revision, restart_dir, 2))
    for other_report, other_cache in (
        (parallel_report, parallel_cache),
        (restart_report, restart_cache),
    ):
        _equal_caches(serial_cache, other_cache)
        if serial_report["selected_ids"] != other_report["selected_ids"]:
            raise RuntimeError("serial/parallel/restart selected IDs differ")
    if any(p.stat().st_mtime_ns != modified for p, modified in existing.items()):
        raise RuntimeError("resume rewrote a previously committed partition")
    return {
        "checks_passed": {
            "serial_parallel_restart_records_and_tokens_equal": True,
            "serial_parallel_restart_selected_ids_equal": True,
            "no_duplicate_completed_records": True,
            "committed_partition_reused_after_interruption": True,
            "independent_coverage_and_export_round_trip": True,
        },
        "interruption_seconds": round(interruption_seconds, 3),
        "runs": {
            "serial": serial_report,
            "parallel": parallel_report,
            "resumed": restart_report,
        },
        "parent_process_peak_rss_bytes": peak_rss_bytes(),
        "memory_measurement_scope": (
            "Parent process lifetime high-water mark; excludes worker RSS."
        ),
    }


def main() -> None:
    """Write a compact engineering report without downloading any dataset."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", type=Path, help="small native ASE-LMDB file or directory"
    )
    parser.add_argument("--revision", help="required provenance revision for --source")
    parser.add_argument(
        "--source-provenance",
        type=Path,
        help="acquisition manifest binding source hash",
    )
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs" / "periodic-check.json"
    )
    args = parser.parse_args()
    if args.source is not None and not args.revision:
        parser.error("--source requires --revision")
    if args.source_provenance is not None and args.source is None:
        parser.error("--source-provenance requires --source")
    if args.max_records < 1:
        parser.error("--max-records must be positive")
    acquisition = {}
    if args.source_provenance is not None:
        acquisition = json.loads(args.source_provenance.read_text())
        with args.source.open("rb") as handle:
            source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if acquisition.get("files_sha256", {}).get(args.source.name) != source_hash:
            parser.error("source content does not match its acquisition manifest")
        if acquisition.get("revision") != args.revision:
            parser.error("revision does not match the acquisition manifest")
    with TemporaryDirectory(prefix="envtok-periodic-check-") as temporary:
        workdir = Path(temporary)
        source = args.source.resolve() if args.source else _fixture_shard(workdir)
        revision = args.revision or hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        try:
            result = qualify(source, revision, workdir, args.max_records)
        finally:
            close_shared_databases()
    report = {
        "schema_version": 1,
        "purpose": (
            "small periodic engineering qualification; downstream accuracy unvalidated"
        ),
        "source": {
            "kind": "user_supplied_lmdb"
            if args.source
            else "generated_reference_fixture",
            "path": str(args.source.resolve())
            if args.source
            else "tests/fixtures/corpus.extxyz",
            "revision": revision,
            "native_adapter": "omat24",
            "real_omat24_claim": acquisition.get("dataset") == "OMat24",
            "acquisition": {
                key: acquisition[key]
                for key in (
                    "dataset",
                    "member",
                    "member_sha256",
                    "archive_sha256",
                    "official_download_url",
                    "upstream_archive_checksum_verified",
                    "sampling",
                )
                if key in acquisition
            },
        },
        "profile": "reference-723696d",
        "min_structure_support": 2,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": package_versions((*DEPENDENCIES, "ase-db-backends", "lmdb")),
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Checks passed; report: {args.output}")


if __name__ == "__main__":
    main()
