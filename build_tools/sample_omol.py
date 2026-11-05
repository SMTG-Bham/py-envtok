"""Extract a bounded native OMol25 sample from an existing local 4M archive.

The ordered training sample keeps finite CHNO configurations with at most 60
atoms by default. No data are downloaded and no artificial records are added.
This is separate from the bundled ColabFit mirror smoke sample; native fields
and the exact archive/member hashes define its provenance. It is not a random
or representative sample and supplies no downstream accuracy evidence.
"""

import argparse
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from ase import Atoms

from envtok.adapters.asedb import (
    close_shared_databases,
    open_database,
    row_atoms,
    row_ids,
)
from envtok.molecular.data import export_records, from_atoms
from envtok.molecular.records import MolecularRecord, validate_molecular_record
from envtok.records import IssueSeverity

ROOT = Path(__file__).resolve().parents[1]
MEMBER = "train_4M/data0009.aselmdb"
DATASET = "OMol25-native-4M"
NAMESPACE = "OMol25:native-4M"
MAX_MEMBER_BYTES = 512 * 2**20
MAX_RETAINED = 1024
MAX_ROWS_SCANNED = 10_000
ELEMENTS = frozenset({1, 6, 7, 8})


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_member(archive_path: Path, target: Path) -> dict[str, Any]:
    """Copy one known regular training member into a bounded native source."""
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name.removeprefix("./") != MEMBER:
                continue
            if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                raise ValueError("expected a regular training shard at most 512 MiB")
            reader = archive.extractfile(member)
            if reader is None:
                raise ValueError("training shard has no readable contents")
            with reader, target.open("wb") as output:
                shutil.copyfileobj(reader, output)
            return {
                "member": member.name,
                "member_bytes": member.size,
                "member_sha256": sha256(target),
            }
    raise ValueError(f"archive does not contain {MEMBER}")


def _prepare_frame(atoms: Atoms, row_key: int, revision: str) -> None:
    """Add explicit molecular state and lineage without discarding native info."""
    native = deepcopy(atoms.info)
    # ASE reads a top-level None as a bare true flag. Nested JSON preserves it.
    atoms.info = {key: value for key, value in atoms.info.items() if value is not None}
    atoms.info.update(
        omol_native_metadata=native,
        sid=f"{MEMBER}:{row_key}",
        total_charge=atoms.info.get("charge"),
        spin_multiplicity=atoms.info.get("spin"),
        split="train",
        source_split="train",
        source_member=MEMBER,
        source_row_key=str(row_key),
        source_record_id=f"{DATASET}@{revision}:{MEMBER}:{row_key}",
        source_dataset=DATASET,
        source_revision=revision,
        label_namespace=NAMESPACE,
        graph_status="missing",
    )
    for scheme in ("lowdin", "mulliken", "nbo"):
        for label in ("charges", "spins"):
            key = f"{scheme}_{label}"
            raw = atoms.info.get(key)
            if isinstance(raw, list | tuple | np.ndarray):
                values = np.asarray(raw)
                if (
                    values.shape == (len(atoms),)
                    and np.issubdtype(values.dtype, np.number)
                    and np.isfinite(values).all()
                    and key not in atoms.arrays
                ):
                    atoms.new_array(key, values.astype(float))


def _records(frames: list[Atoms], revision: str) -> list[MolecularRecord]:
    return list(
        from_atoms(
            frames,
            label_namespace=NAMESPACE,
            dataset=DATASET,
            revision=revision,
            unknown_policy="reject",
        )
    )


def _skip_reason(atoms: Atoms, max_atoms: int) -> str | None:
    if not set(atoms.numbers.tolist()) <= ELEMENTS:
        return "non_CHNO"
    if len(atoms) > max_atoms:
        return "over_atom_limit"
    if atoms.pbc.any():
        return "periodic"
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        return "empty_or_nonfinite_geometry"
    results = atoms.calc.results if atoms.calc is not None else {}
    if any(
        key not in results or not np.isfinite(results[key]).all()
        for key in ("energy", "forces")
    ):
        return "missing_or_nonfinite_labels"
    return None


def sample(
    shard: Path, revision: str, count: int = 512, max_atoms: int = 60
) -> tuple[list[MolecularRecord], list[dict[str, Any]], list[dict[str, str]]]:
    """Keep ordered valid native records and account for every scanned row."""
    if not 1 <= count <= MAX_RETAINED or max_atoms < 1:
        raise ValueError(f"positive limits required; retain at most {MAX_RETAINED}")
    frames: list[Atoms] = []
    lineage: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    database = open_database(shard)
    try:
        for row_key in row_ids(database)[:MAX_ROWS_SCANNED]:
            atoms = row_atoms(database, row_key)
            reason = _skip_reason(atoms, max_atoms)
            if reason is None:
                _prepare_frame(atoms, row_key, revision)
                issues = validate_molecular_record(_records([atoms], revision)[0])
                failures = sorted(
                    {
                        issue.code
                        for issue in issues
                        if issue.severity != IssueSeverity.WARN
                    }
                )
                if failures:
                    reason = "record_validation:" + ",".join(failures)
            if reason is not None:
                skipped.append({"source_row_key": str(row_key), "reason": reason})
                continue
            frames.append(atoms)
            lineage.append(
                {
                    "source_row_key": str(row_key),
                    "source_record_id": atoms.info["source_record_id"],
                    "sid": atoms.info["sid"],
                    "source": atoms.info.get("source"),
                    "data_id": atoms.info.get("data_id"),
                    "natoms": len(atoms),
                    "total_charge": atoms.info["total_charge"],
                    "spin_multiplicity": atoms.info["spin_multiplicity"],
                }
            )
            if len(frames) == count:
                break
    finally:
        close_shared_databases()
    if len(frames) != count:
        raise ValueError("bounded scan did not find enough eligible native records")
    return _records(frames, revision), lineage, skipped


def write_sample(records: list[MolecularRecord], directory: Path) -> dict[str, str]:
    """Write original geometry, native metadata, state and labels as extended XYZ."""
    directory.mkdir(parents=True, exist_ok=True)
    output = export_records(records, directory / "sample.extxyz")
    return {output.name: sha256(output)}


def manifest(
    archive: Path,
    member_metadata: dict[str, Any],
    records: list[MolecularRecord],
    lineage: list[dict[str, Any]],
    skipped: list[dict[str, str]],
    files: dict[str, str],
    count: int,
    max_atoms: int,
) -> dict[str, Any]:
    """Describe native acquisition, filtering, state mapping and exact hashes."""
    revision = f"local-4M-shard-{member_metadata['member_sha256'][:16]}"
    return {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": revision,
        "origin": "existing local OMol25 4M archive; native training shard",
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256(archive),
        "upstream_archive_checksum_verified": False,
        **member_metadata,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_card": "https://huggingface.co/datasets/facebook/OMol25",
        "citation": (
            "Levine et al., The Open Molecules 2025 Dataset, Evaluations, "
            "and Models, arXiv:2505.08762 (2025)"
        ),
        "sampling": {
            "order": "ascending native row key; first eligible records",
            "count": count,
            "elements": ["H", "C", "N", "O"],
            "max_atoms": max_atoms,
            "max_rows_scanned": MAX_ROWS_SCANNED,
            "rows_read": len(lineage) + len(skipped),
            "skipped": dict(Counter(row["reason"] for row in skipped)),
            "source_split": "train",
            "limitation": (
                "ordered sample from one native shard, separate from the ColabFit "
                "mirror smoke sample; not representative and no accuracy claim"
            ),
        },
        "configurations": len(records),
        "atoms": sum(record.natoms for record in records),
        "state_counts": dict(
            Counter(
                f"charge={record.total_charge:g},multiplicity={record.spin_multiplicity}"
                for record in records
            )
        ),
        "source_categories": dict(Counter(str(row["data_id"]) for row in lineage)),
        "label_namespace": NAMESPACE,
        "field_mapping": {
            "total_charge": "native charge",
            "spin_multiplicity": "native spin (total molecular multiplicity)",
            "energy_eV": "native stored calculator energy",
            "forces_eV_per_angstrom": "native stored calculator forces",
        },
        "method_metadata": (
            "Native records do not establish per-record method/basis metadata; "
            "label namespace identifies the native OMol25 4M source only."
        ),
        "metadata_preservation": (
            "Complete native metadata is stored in omol_native_metadata; "
            "top-level nulls are omitted to avoid ASE converting them to true. "
            "Well-formed per-atom charge/spin lists are also copied to arrays."
        ),
        "graph": "no supplied validated graph; use geometry fallback",
        "grouping": (
            "No verified family mapping is inferred from native source locators; "
            "the pipeline reports its conservative composition fallback."
        ),
        "files_sha256": files,
        "records": lineage,
        "skipped_records": skipped,
    }


def main() -> None:
    """Create a bounded native sample and its complete acquisition manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs" / "omol-compression-demo" / "data",
    )
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--max-atoms", type=int, default=60)
    args = parser.parse_args()
    if not 1 <= args.count <= MAX_RETAINED or args.max_atoms < 1:
        parser.error(f"positive limits required; retain at most {MAX_RETAINED}")
    with TemporaryDirectory(prefix="envtok-omol-sample-") as temporary:
        directory = Path(temporary)
        shard = directory / "source.aselmdb"
        metadata = stage_member(args.archive, shard)
        revision = f"local-4M-shard-{metadata['member_sha256'][:16]}"
        records, lineage, skipped = sample(shard, revision, args.count, args.max_atoms)
        files = write_sample(records, directory)
        provenance = manifest(
            args.archive,
            metadata,
            records,
            lineage,
            skipped,
            files,
            args.count,
            args.max_atoms,
        )
        (directory / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        for name in (*files, "provenance.json"):
            shutil.copyfile(directory / name, args.output_directory / name)
    print(
        f"Saved {len(records)} configurations / {provenance['atoms']} atoms "
        f"to {args.output_directory}"
    )


if __name__ == "__main__":
    main()
