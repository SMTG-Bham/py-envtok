"""ASE database adapters (SQLite/JSON databases and OMat24 ASE-LMDB shards).

Row IDs are the native keys. A row's ``key_value_pairs`` and ``data`` are
merged into ``atoms.info``: data last, as fairchem's ``AseDBDataset`` does, so
OMat24's ``sid`` is available as ``info["sid"]``. Stored calculator results
come back as a ``SinglePointCalculator``; nothing is ever recomputed.

OMat24 shards are read through ``ase.db.connect`` with the ``aselmdb``
backend from ``ase-db-backends`` (install ``envtok[omat24]``), read-only and
without a lock file. The version-sensitive calls are confined to
:func:`open_database` and :func:`row_atoms`.
"""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, cast, override

import ase.db
import numpy as np
from ase import Atoms

from envtok.adapters.base import (
    Adapter,
    Locator,
    RecordOptions,
    file_fingerprint,
    make_record,
    shard_name,
    sorted_files,
)
from envtok.errors import ConfigError
from envtok.records import SourceRef, StructureRecord

_SHARED_LMDB: dict[str, tuple[tuple[int, int, int], Any]] = {}
"""Read-only LMDB handles of this process, keyed by path, with the file identity
they were opened for; an environment may be opened once per process only."""


def _identity(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def open_database(path: Path) -> Any:
    """Open an ASE database read-only.

    LMDB environments can be open only once per process, so ``.aselmdb``
    handles are shared by every adapter of the process (see
    :func:`close_shared_databases`). A shard replaced on disk is reopened.

    Args:
        path: Database file; ``.aselmdb`` files use the LMDB backend.

    Returns:
        The ASE database object.

    Raises:
        ConfigError: If the LMDB backend is not installed.
    """
    if path.suffix != ".aselmdb":
        return ase.db.connect(str(path), use_lock_file=False)
    key = str(path.resolve())
    identity = _identity(path)
    shared = _SHARED_LMDB.get(key)
    if shared is not None and shared[0] == identity:
        return shared[1]
    if shared is not None:
        _close(_SHARED_LMDB.pop(key)[1])
    try:
        database = ase.db.connect(str(path), readonly=True, use_lock_file=False)
    except ImportError as exc:
        raise ConfigError(
            f"reading {path} needs the ASE-LMDB backend; install envtok[omat24] ({exc})"
        ) from exc
    _SHARED_LMDB[key] = (identity, database)
    return database


def _close(database: Any) -> None:
    closer = getattr(database, "close", None)
    if closer is not None:
        closer()


def close_shared_databases() -> None:
    """Close every shared read-only LMDB handle of this process."""
    for _, database in _SHARED_LMDB.values():
        _close(database)
    _SHARED_LMDB.clear()


def row_ids(database: Any) -> list[int]:
    """Return every row ID in ascending order."""
    ids = getattr(database, "ids", None)
    if ids is not None:
        return sorted(int(i) for i in ids)
    return sorted(int(row.id) for row in database.select())


def row_atoms(database: Any, row_id: int) -> Atoms:
    """Return the atoms of one row with its key-value pairs and data in ``info``.

    Args:
        database: An open ASE database.
        row_id: Row ID.

    Returns:
        Atoms with stored results as a single-point calculator.
    """
    getter = getattr(database, "_get_row", None)
    row = getter(row_id) if getter is not None else database.get(id=row_id)
    atoms = cast("Atoms", row.toatoms())
    atoms.info.update(dict(row.key_value_pairs))
    if isinstance(row.data, dict):
        for key, value in row.data.items():
            atoms.info[key] = value.item() if isinstance(value, np.generic) else value
    return atoms


class AseDbAdapter(Adapter):
    """Read one ASE database file, or every matching database below a directory.

    Args:
        path: Database file or directory.
        options: Record options.
        patterns: Glob patterns used when ``path`` is a directory.
        checksum: Source fingerprint mode, ``"stat"`` or ``"full"``.
    """

    kind = "asedb"
    default_patterns: ClassVar[tuple[str, ...]] = ("**/*.db", "**/*.json")

    def __init__(
        self,
        path: str | Path,
        options: RecordOptions,
        *,
        patterns: Sequence[str] | None = None,
        checksum: str = "stat",
    ) -> None:
        """Resolve the database files."""
        self.root = Path(path)
        self.options = options
        self.patterns = (
            tuple(patterns) if patterns is not None else self.default_patterns
        )
        self.checksum = checksum
        self.files = sorted_files(self.root, self.patterns)
        self._open: dict[str, Any] = {}

    @override
    def spec(self) -> dict[str, Any]:
        """Return the adapter description."""
        return {
            "kind": self.kind,
            "path": str(self.root),
            "options": self.options.as_dict(),
            "patterns": list(self.patterns),
            "checksum": self.checksum,
        }

    @override
    def source_fingerprint(self) -> dict[str, Any]:
        """Fingerprint every database file and the record options."""
        return {
            "kind": self.kind,
            "files": [
                {
                    "shard": shard_name(p, self.root),
                    **file_fingerprint(p, checksum=self.checksum),
                }
                for p in self.files
            ],
            "options": self.options.as_dict(),
        }

    def _database(self, shard: str) -> Any:
        if shard not in self._open:
            path = next(p for p in self.files if shard_name(p, self.root) == shard)
            self._open[shard] = open_database(path)
        return self._open[shard]

    @override
    def locators(self) -> Iterator[Locator]:
        """Yield one locator per row, databases in sorted order."""
        ordinal = 0
        for path in self.files:
            shard = shard_name(path, self.root)
            for row_id in row_ids(self._database(shard)):
                yield Locator(ordinal, shard, str(row_id))
                ordinal += 1

    @override
    def read(self, locator: Locator) -> StructureRecord:
        """Read one row."""
        atoms = row_atoms(self._database(locator.shard), int(locator.key))
        return make_record(atoms, locator.shard, locator.key, self.options)

    @override
    def retrieve(self, refs: Sequence[SourceRef]) -> Iterator[tuple[SourceRef, Atoms]]:
        """Yield the original atoms of the referenced rows."""
        for ref in refs:
            yield ref, row_atoms(self._database(ref.shard), int(ref.key))

    @override
    def close(self) -> None:
        """Close databases owned by this adapter (shared LMDB handles stay open)."""
        shared = {id(database) for _, database in _SHARED_LMDB.values()}
        for database in self._open.values():
            if id(database) not in shared:
                _close(database)
        self._open.clear()


class OMat24Adapter(AseDbAdapter):
    """Read OMat24 ASE-LMDB shards natively, without text conversion.

    ``path`` is the directory holding the sub-datasets (``rattled-300/``,
    ``aimd-from-PBE-1000-npt/``, ...) or a single sub-dataset or shard. The
    sub-dataset name (first path component of the shard) is added to the
    record metadata as ``omat_subset``.

    Args:
        path: OMat24 directory, sub-dataset directory or ``.aselmdb`` file.
        options: Record options (``native_id_key`` defaults to ``sid``).
        patterns: Glob patterns used when ``path`` is a directory.
        checksum: Source fingerprint mode, ``"stat"`` or ``"full"``.
    """

    kind = "omat24"
    default_patterns: ClassVar[tuple[str, ...]] = ("**/*.aselmdb",)

    @override
    def read(self, locator: Locator) -> StructureRecord:
        """Read one row and label its sub-dataset."""
        atoms = row_atoms(self._database(locator.shard), int(locator.key))
        subset = (
            locator.shard.split("/", 1)[0] if "/" in locator.shard else self.root.name
        )
        atoms.info.setdefault("omat_subset", subset)
        return make_record(atoms, locator.shard, locator.key, self.options)
