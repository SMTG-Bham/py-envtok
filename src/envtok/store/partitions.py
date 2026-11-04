"""Immutable, atomically committed output partitions.

Each batch of records is written by exactly one worker into a temporary
directory, which is renamed to ``part-NNNNNN`` only after every table and the
``_COMMITTED.json`` marker are on disk. A crash therefore leaves either a
complete partition or an ignorable temporary directory, never a half-written
partition, and a retried batch can never duplicate rows.
"""

import json
import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from envtok.errors import StaleCacheError
from envtok.store.schema import FAILURES, NEIGHBOURS, SITES, STRUCTURES, rows_to_table
from envtok.tokenizer.pipeline import TokenizedRecord

COMMIT_MARKER = "_COMMITTED.json"
TABLES = ("structures", "sites", "neighbours", "failures")


def partition_name(index: int) -> str:
    """Return the directory name of batch ``index``."""
    return f"part-{index:06d}"


def write_partition(
    root: Path,
    index: int,
    results: Sequence[TokenizedRecord],
    *,
    fingerprint: str,
    keep_neighbours: bool,
) -> Path:
    """Write one batch atomically.

    Args:
        root: Partitions directory.
        index: Batch index.
        results: Tokenised records of the batch, in batch order.
        fingerprint: Cache fingerprint recorded in the commit marker.
        keep_neighbours: Whether to write the neighbours table.

    Returns:
        The committed partition directory. If the batch was already committed
        (a retried duplicate), the existing partition is kept.
    """
    final = root / partition_name(index)
    if (final / COMMIT_MARKER).exists():
        return final
    temporary = root / f".{partition_name(index)}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    tables = {
        "structures": rows_to_table([r.structure for r in results], STRUCTURES),
        "sites": rows_to_table([s for r in results for s in r.sites], SITES),
        "failures": rows_to_table([f for r in results for f in r.failures], FAILURES),
    }
    if keep_neighbours:
        tables["neighbours"] = rows_to_table(
            [n for r in results for n in r.neighbours], NEIGHBOURS
        )
    for name, table in tables.items():
        pq.write_table(table, temporary / f"{name}.parquet", compression="zstd")
    marker = {
        "index": index,
        "fingerprint": fingerprint,
        "record_ids": [r.structure.record_id for r in results],
        "tables": sorted(tables),
        "committed_at": datetime.now(UTC).isoformat(),
    }
    (temporary / COMMIT_MARKER).write_text(json.dumps(marker, indent=1))
    try:
        temporary.rename(final)
    except OSError:
        if not (final / COMMIT_MARKER).exists():
            raise
        shutil.rmtree(temporary)
    return final


def read_marker(partition: Path) -> dict[str, Any] | None:
    """Return a partition's commit marker, or ``None`` if it is not committed."""
    marker = partition / COMMIT_MARKER
    if not marker.exists():
        return None
    data: dict[str, Any] = json.loads(marker.read_text())
    return data


def committed_partitions(root: Path, fingerprint: str) -> dict[int, Path]:
    """Return committed partitions by batch index, validating their fingerprints.

    Args:
        root: Partitions directory.
        fingerprint: Expected cache fingerprint.

    Returns:
        Mapping from batch index to partition directory.

    Raises:
        StaleCacheError: If a committed partition carries another fingerprint.
    """
    found: dict[int, Path] = {}
    if not root.exists():
        return found
    for path in sorted(root.glob("part-*")):
        marker = read_marker(path)
        if marker is None:
            continue
        if marker["fingerprint"] != fingerprint:
            raise StaleCacheError(
                f"partition {path} was written for fingerprint "
                f"{marker['fingerprint'][:12]}, not {fingerprint[:12]}; it is stale "
                "or was copied from another run. Remove it or use a new workdir"
            )
        found[int(marker["index"])] = path
    return found


def remove_temporary(root: Path) -> int:
    """Delete leftover temporary directories of interrupted writes.

    Returns:
        The number of directories removed.
    """
    leftovers = list(root.glob(".part-*.tmp-*")) if root.exists() else []
    for path in leftovers:
        shutil.rmtree(path, ignore_errors=True)
    return len(leftovers)
