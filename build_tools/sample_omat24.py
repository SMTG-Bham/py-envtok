"""Extract a tiny, traceable OMat24 sample from an existing local archive.

No data are downloaded. Only one bounded native shard is staged temporarily;
32 training records and eight disjoint-parent development records are retained
by default. The development records come from the source training partition,
not the official validation partition. Ordered sampling is an engineering
convenience and does not establish a representative accuracy benchmark.

Usage::

    uv run python build_tools/sample_omat24.py --archive /data/omat24_1M.tar.gz
"""

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import ase.db
import numpy as np
from ase import Atoms

from envtok.adapters import OMat24Adapter, RecordOptions, spool_atoms
from envtok.adapters.asedb import close_shared_databases
from envtok.profiles import corpus_policy
from envtok.records import Split, validate_record

ROOT = Path(__file__).resolve().parents[1]
MEMBER = "./train/rattled-relax/train.aselmdb"
SOURCE_URL = (
    "https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/251210/"
    "omat24_1M_251210.tar.gz"
)
MAX_MEMBER_BYTES = 512 * 2**20
MAX_RETAINED = 96


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_member(archive_path: Path, target: Path) -> dict[str, Any]:
    """Copy one known training member into a temporary native source file."""
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name != MEMBER:
                continue
            if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                raise ValueError(
                    "expected regular training shard no larger than 512 MiB"
                )
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


def sample(
    shard: Path, revision: str, train_count: int, dev_count: int, max_atoms: int
) -> tuple[dict[str, list[Atoms]], list[dict[str, Any]], dict[str, int]]:
    """Keep ordered valid small records with distinct, supplied parent IDs."""
    adapter = OMat24Adapter(
        shard,
        RecordOptions(dataset="omat24", revision=revision, split=Split.TRAIN),
    )
    frames: dict[str, list[Atoms]] = {"train": [], "development": []}
    policy = corpus_policy("omat24-study")
    lineage: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    parents: set[str] = set()
    try:
        for scanned, record in enumerate(adapter.iter_records(), start=1):
            if scanned > 1000:  # noqa: PLR2004 - deliberately bounded acquisition
                break
            parent = record.metadata.get("parent_id")
            reason = None
            if record.natoms > max_atoms:
                reason = "over_atom_limit"
            elif policy.excluded_symbols(record.atoms.get_chemical_symbols()):
                reason = "study_f_block_exclusion"
            elif validate_record(record):
                reason = "record_validation"
            elif not isinstance(parent, str) or not parent:
                reason = "missing_parent_id"
            elif parent in parents:
                reason = "duplicate_parent_id"
            elif any(
                key not in record.labels
                or not np.isfinite(record.labels[key].value).all()
                for key in ("energy", "forces", "stress")
            ):
                reason = "missing_or_nonfinite_labels"
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            partition = "train" if len(frames["train"]) < train_count else "development"
            source_id = (
                f"omat24@{revision}:{MEMBER.removeprefix('./')}:{record.source_ref.key}"
            )
            atoms = record.atoms
            atoms.info.update(
                split="train" if partition == "train" else "validation",
                source_split="train",
                source_member=MEMBER,
                source_row_key=record.source_ref.key,
                source_record_id=source_id,
            )
            frames[partition].append(atoms)
            parents.add(str(parent))
            lineage.append(
                {
                    "partition": partition,
                    "source_record_id": source_id,
                    "source_row_key": record.source_ref.key,
                    "sid": record.source_ref.native_id,
                    "parent_id": parent,
                    "natoms": record.natoms,
                }
            )
            if (
                len(frames["train"]) == train_count
                and len(frames["development"]) == dev_count
            ):
                break
    finally:
        adapter.close()
        close_shared_databases()
    if len(frames["train"]) != train_count or len(frames["development"]) != dev_count:
        raise ValueError(
            "bounded scan did not find enough eligible distinct-parent records"
        )
    return frames, lineage, skipped


def write_sample(frames: dict[str, list[Atoms]], directory: Path) -> dict[str, str]:
    """Write small native and extxyz sources with original geometry and labels."""
    hashes = {}
    for name, atoms in frames.items():
        xyz = directory / f"{name}.extxyz"
        spool_atoms(atoms, xyz)
        hashes[xyz.name] = sha256(xyz)
        native = directory / f"{name}.aselmdb"
        database = ase.db.connect(str(native))
        try:
            for frame in atoms:
                database.write(frame, data=dict(frame.info))
        finally:
            database.close()
        hashes[native.name] = sha256(native)
    return hashes


def main() -> None:
    """Create reproducible acquisition provenance and a tiny fixed sample."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "examples" / "omat24" / "small-sample",
    )
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--development", type=int, default=8)
    parser.add_argument("--max-atoms", type=int, default=20)
    args = parser.parse_args()
    if (
        min(args.train, args.development, args.max_atoms) < 1
        or args.train + args.development > MAX_RETAINED
    ):
        parser.error("counts and max-atoms must be positive; retain at most 96 records")
    with TemporaryDirectory(prefix="envtok-omat24-sample-") as temporary:
        directory = Path(temporary)
        shard = directory / "source.aselmdb"
        provenance = stage_member(args.archive, shard)
        revision = f"local-1M-shard-{provenance['member_sha256'][:16]}"
        frames, lineage, skipped = sample(
            shard, revision, args.train, args.development, args.max_atoms
        )
        files = write_sample(frames, directory)
        manifest = {
            "schema_version": 1,
            "dataset": "OMat24",
            "origin": (
                "existing local OMat24 1M archive; native rattled-relax training shard"
            ),
            "archive_name": args.archive.name,
            "archive_bytes": args.archive.stat().st_size,
            "archive_sha256": sha256(args.archive),
            "official_download_url": SOURCE_URL,
            "upstream_archive_checksum_verified": False,
            "license": "CC-BY-4.0",
            "citation": "https://arxiv.org/abs/2410.12771",
            "dataset_card": "https://huggingface.co/datasets/facebook/OMAT24",
            **provenance,
            "revision": revision,
            "sampling": {
                "order": "ascending native row key; first eligible distinct parents",
                "train": args.train,
                "development": args.development,
                "max_atoms": args.max_atoms,
                "max_rows_scanned": 1000,
                "rows_read": len(lineage) + sum(skipped.values()),
                "excluded_elements": "La-Lu and Ac-Lr (OMat24 study preset)",
                "skipped": skipped,
                "group_key": "parent_id",
                "parent_overlap": 0,
                "source_split": "train for both outputs",
                "development_split": (
                    "held out inside official training; not official validation"
                ),
                "limitation": (
                    "small ordered sample; "
                    "no representative or production accuracy claim"
                ),
            },
            "labels": {"energy": "eV", "forces": "eV/Ang", "stress": "eV/Ang^3"},
            "files_sha256": files,
            "records": lineage,
        }
        (directory / "provenance.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        args.output_directory.mkdir(parents=True, exist_ok=True)
        for name in (*files, "provenance.json"):
            shutil.copyfile(directory / name, args.output_directory / name)
    print(
        f"Saved {args.train} train + {args.development} development records "
        f"to {args.output_directory}"
    )


if __name__ == "__main__":
    main()
