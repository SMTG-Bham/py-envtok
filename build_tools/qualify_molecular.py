"""Check the small molecular selection, restart and export workflow.

Run with the molecular extra. The default is the bundled 29-record OMol sample;
this checks engineering contracts and reports incomplete coverage honestly.
"""

import argparse
import hashlib
import json
import time
from itertools import islice
from pathlib import Path
from typing import Any

import ase.io
import numpy as np

from envtok.molecular.benchmarks import compare_selectors
from envtok.molecular.data import iter_extxyz
from envtok.molecular.pipeline import MolecularPipeline

ROOT = Path(__file__).resolve().parents[1]


def _check_export(pipeline: MolecularPipeline, output: Path) -> int:
    """Verify source IDs, atom alignment, states and normalized labels."""
    ids = (pipeline.directory / "selected_ids.txt").read_text().splitlines()
    originals = {record.record_id: record for record in pipeline.ingest().candidates}
    namespace = pipeline.settings["source"]["label_namespace"]
    exported = iter_extxyz(output, label_namespace=namespace)
    for record_id, record in zip(ids, exported, strict=True):
        original = originals[record_id]
        if record.atoms.info["envtok_record_id"] != record_id:
            raise RuntimeError("export changed source identity or frame order")
        if record.state_key != original.state_key:
            raise RuntimeError("export changed protected molecular state")
        np.testing.assert_array_equal(record.atoms.numbers, original.atoms.numbers)
        np.testing.assert_array_equal(record.atoms.pbc, original.atoms.pbc)
        np.testing.assert_allclose(record.atoms.cell, original.atoms.cell, atol=1e-12)
        for key, value in original.atoms.arrays.items():
            actual = record.atoms.arrays[key]
            if np.issubdtype(value.dtype, np.number):
                np.testing.assert_allclose(actual, value, rtol=1e-12, atol=1e-12)
            else:
                np.testing.assert_array_equal(actual, value)
        for key, label in original.labels.items():
            if record.labels[key].unit != label.unit:
                raise RuntimeError(f"export changed label unit: {key}")
            np.testing.assert_allclose(
                record.labels[key].value, label.value, rtol=1e-12, atol=1e-12
            )
    return len(ids)


def qualify(config: Path) -> dict[str, Any]:
    """Run and resume one <=128-record pool with one comparison seed/budget."""
    started = time.perf_counter()
    pipeline = MolecularPipeline(config)
    count = sum(1 for _ in islice(ase.io.iread(pipeline.source), 129))
    if count > 128:  # noqa: PLR2004 - explicit small-plan bound
        raise ValueError("molecular qualification requires at most 128 source records")
    directory = pipeline.run()
    ids_path = directory / "selected_ids.txt"
    first_ids = ids_path.read_text()
    cached = {
        path.name: (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in pipeline.feature_directory.iterdir()
        if path.is_file()
    }
    first_report = pipeline.audit()
    repeated = MolecularPipeline(config)
    if repeated.run() != directory or ids_path.read_text() != first_ids:
        raise RuntimeError("restart changed the selection")
    for name, (modified, digest) in cached.items():
        path = pipeline.feature_directory / name
        if (
            path.stat().st_mtime_ns != modified
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError("restart rewrote committed molecular features")
    loaded = MolecularPipeline.load_run(directory)
    saved_report = loaded.audit()
    if saved_report != first_report:
        raise RuntimeError("saved-run audit disagrees with the original audit")
    checked = _check_export(loaded, loaded.export())
    comparison = compare_selectors(loaded, fractions=(0.25,), seeds=(0,))
    return {
        "status": "engineering_checks_passed",
        "checks": {
            "restart_selected_ids_equal": True,
            "feature_cache_reused": True,
            "saved_run_audit_equal": True,
            "export_states_labels_and_atom_alignment": True,
        },
        "source_sha256": pipeline.source_hash,
        "source_records": count,
        "exported_records_checked": checked,
        "selected_ids": first_ids.splitlines(),
        "coverage": saved_report,
        "comparison": comparison,
        "software": pipeline.software,
        "seconds": time.perf_counter() - started,
        "interpretation": (
            "Passing means restart/export/audit consistency, not full target coverage "
            "or model accuracy. Coverage and unmet constraints are reported separately."
        ),
    }


def main() -> None:
    """Save a bounded workflow report without downloading data."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=ROOT / "examples/molecular/omol/config.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/molecular-qualification.json"
    )
    args = parser.parse_args()
    report = qualify(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Molecular workflow checks passed: {args.output}")


if __name__ == "__main__":
    main()
