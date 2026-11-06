"""Extended XYZ adapter.

Plain files are indexed once by the byte offset of every frame's atom-count
line, so workers can seek directly to a frame (the index travels inside the
locators). Compressed files (``.gz``, ``.bz2``, ``.xz``) cannot be seeked
cheaply and follow an explicit policy:

* ``sequential`` (default): records are streamed in one pass; retrieval for
  export is a second sequential pass.
* ``stage``: each file is decompressed once into a staging directory and then
  treated as a plain file.

Frames are parsed with ``ase.io`` (``parallel=False``, so an MPI launcher can
never broadcast one rank's frame to every rank).
"""

import bz2
import gzip
import hashlib
import io
import lzma
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import IO, Any, cast

import ase.io
from ase import Atoms
from typing_extensions import override

from envtok.adapters.base import (
    Adapter,
    Locator,
    RecordOptions,
    file_fingerprint,
    make_record,
    shard_name,
    sorted_files,
)
from envtok.errors import ConfigError, UnsupportedInputError
from envtok.records import SourceRef, StructureRecord

_COMPRESSED_SUFFIXES = (".gz", ".bz2", ".xz")
DEFAULT_PATTERNS = tuple(
    f"**/*{stem}{suffix}"
    for stem in (".xyz", ".extxyz")
    for suffix in ("", ".gz", ".bz2", ".xz")
)
COMPRESSION_POLICIES = ("sequential", "stage")


def _compression(path: Path) -> str | None:
    return path.suffix if path.suffix in _COMPRESSED_SUFFIXES else None


def _open(path: Path) -> IO[bytes]:
    suffix = _compression(path)
    if suffix == ".gz":
        return cast("IO[bytes]", gzip.open(path, "rb"))
    if suffix == ".bz2":
        return cast("IO[bytes]", bz2.open(path, "rb"))
    if suffix == ".xz":
        return cast("IO[bytes]", lzma.open(path, "rb"))
    return path.open("rb")


def scan_frames(handle: IO[bytes], name: str) -> list[int]:
    """Return the byte offset of every frame's atom-count line.

    One sequential pass; lines are counted, not parsed. Blank lines between
    frames are tolerated.

    Args:
        handle: Binary stream positioned at the start of the file.
        name: File name used in error messages.

    Returns:
        Offsets in stream coordinates.

    Raises:
        UnsupportedInputError: If a frame does not start with an integer line.
    """
    offsets = []
    while True:
        position = handle.tell()
        line = handle.readline()
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        try:
            natoms = int(text)
        except ValueError:
            raise UnsupportedInputError(
                f"{name}: expected an integer atom-count line at byte {position}, "
                f"got {text[:60]!r}"
            ) from None
        offsets.append(position)
        handle.readline()
        for _ in range(natoms):
            handle.readline()
    return offsets


def read_frame_at(path: Path, offset: int) -> Atoms:
    """Seek to ``offset`` and parse exactly one extxyz frame.

    Args:
        path: Uncompressed extxyz file.
        offset: Byte offset of the frame's atom-count line.

    Returns:
        The parsed atoms.
    """
    with path.open("rb") as handle:
        handle.seek(offset)
        natoms = int(handle.readline())
        rest = [handle.readline() for _ in range(natoms + 1)]
    text = f"{natoms}\n" + b"".join(rest).decode().replace("\r\n", "\n")
    atoms = ase.io.read(io.StringIO(text), format="extxyz", index=0, parallel=False)
    return cast("Atoms", atoms)


class ExtxyzAdapter(Adapter):
    """Read one extxyz file or every extxyz file below a directory.

    Args:
        path: A file or a directory searched recursively.
        options: Record options.
        patterns: Glob patterns used when ``path`` is a directory.
        compressed: ``"sequential"`` or ``"stage"``.
        staging_dir: Directory for decompressed copies (``stage`` policy).
        checksum: Source fingerprint mode, ``"stat"`` or ``"full"``.
    """

    kind = "extxyz"

    def __init__(
        self,
        path: str | Path,
        options: RecordOptions,
        *,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
        compressed: str = "sequential",
        staging_dir: str | Path | None = None,
        checksum: str = "stat",
    ) -> None:
        """Resolve the source files."""
        if compressed not in COMPRESSION_POLICIES:
            raise ConfigError(
                f"compressed must be one of {COMPRESSION_POLICIES}, got {compressed!r}"
            )
        if compressed == "stage" and staging_dir is None:
            raise ConfigError("the 'stage' compression policy needs staging_dir")
        self.root = Path(path)
        self.options = options
        self.patterns = tuple(patterns)
        self.compressed = compressed
        self.staging_dir = None if staging_dir is None else Path(staging_dir)
        self.checksum = checksum
        self.files = sorted_files(self.root, self.patterns)
        self._offsets: dict[Path, list[int]] = {}

    @property
    def _any_sequential(self) -> bool:
        return self.compressed == "sequential" and any(
            _compression(path) for path in self.files
        )

    @property
    @override
    def supports_random_access(self) -> bool:
        """False when any file is compressed and streamed sequentially."""
        return not self._any_sequential

    @override
    def spec(self) -> dict[str, Any]:
        """Return the adapter description."""
        return {
            "kind": self.kind,
            "path": str(self.root),
            "options": self.options.as_dict(),
            "patterns": list(self.patterns),
            "compressed": self.compressed,
            "staging_dir": None if self.staging_dir is None else str(self.staging_dir),
            "checksum": self.checksum,
        }

    @override
    def source_fingerprint(self) -> dict[str, Any]:
        """Fingerprint every source file and the record options."""
        return {
            "kind": self.kind,
            "files": [
                {
                    "shard": shard_name(path, self.root),
                    **file_fingerprint(path, checksum=self.checksum),
                }
                for path in self.files
            ],
            "compressed": self.compressed,
            "options": self.options.as_dict(),
        }

    def _readable(self, path: Path) -> Path:
        """Return the path to read (a staged copy for staged compressed files)."""
        if not _compression(path) or self.compressed == "sequential":
            return path
        assert self.staging_dir is not None  # noqa: S101 - checked in __init__
        digest = hashlib.sha256(
            repr(sorted(file_fingerprint(path).items())).encode() + str(path).encode()
        ).hexdigest()[:16]
        staged = self.staging_dir / f"{path.stem}.{digest}"
        if not staged.exists():
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            partial = staged.with_name(staged.name + ".partial")
            with _open(path) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            partial.replace(staged)
        return staged

    def _frame_offsets(self, path: Path) -> list[int]:
        if path not in self._offsets:
            with _open(self._readable(path)) as handle:
                self._offsets[path] = scan_frames(handle, path.name)
        return self._offsets[path]

    @override
    def locators(self) -> Iterator[Locator]:
        """Yield one locator per frame, files in sorted order."""
        ordinal = 0
        for path in self.files:
            shard = shard_name(path, self.root)
            sequential = self.compressed == "sequential" and _compression(path)
            for frame, offset in enumerate(self._frame_offsets(path)):
                yield Locator(
                    ordinal, shard, str(frame), None if sequential else offset
                )
                ordinal += 1

    def _path_of(self, shard: str) -> Path:
        for path in self.files:
            if shard_name(path, self.root) == shard:
                return path
        raise KeyError(shard)

    @override
    def read(self, locator: Locator) -> StructureRecord:
        """Read one frame.

        Raises:
            UnsupportedInputError: For sequentially streamed compressed files,
                which have no random access.
        """
        if locator.offset is None:
            raise UnsupportedInputError(
                f"{locator.shard} is compressed and read sequentially; use "
                "iter_records() or the 'stage' compression policy"
            )
        path = self._readable(self._path_of(locator.shard))
        atoms = read_frame_at(path, locator.offset)
        return make_record(atoms, locator.shard, locator.key, self.options)

    def _iter_file(self, path: Path) -> Iterator[Atoms]:
        with _open(self._readable(path)) as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8")
            yield from ase.io.iread(text, index=":", format="extxyz", parallel=False)

    @override
    def iter_records(self) -> Iterator[StructureRecord]:
        """Stream every frame sequentially (one pass per file)."""
        for path in self.files:
            shard = shard_name(path, self.root)
            for frame, atoms in enumerate(self._iter_file(path)):
                yield make_record(atoms, shard, str(frame), self.options)

    @override
    def retrieve(self, refs: Sequence[SourceRef]) -> Iterator[tuple[SourceRef, Atoms]]:
        """Yield original atoms; compressed sequential files are rescanned once."""
        wanted: dict[str, set[str]] = {}
        for ref in refs:
            wanted.setdefault(ref.shard, set()).add(ref.key)
        found: dict[tuple[str, str], Atoms] = {}
        for shard, keys in wanted.items():
            path = self._path_of(shard)
            if self.compressed == "sequential" and _compression(path):
                for frame, atoms in enumerate(self._iter_file(path)):
                    if str(frame) in keys:
                        found[shard, str(frame)] = atoms
            else:
                offsets = self._frame_offsets(path)
                readable = self._readable(path)
                for key in keys:
                    found[shard, key] = read_frame_at(readable, offsets[int(key)])
        for ref in refs:
            yield ref, found[ref.shard, ref.key]
