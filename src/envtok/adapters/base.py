"""Dataset adapter contract and shared record construction.

An adapter streams :class:`~envtok.records.StructureRecord` objects from one
source and can later retrieve the original configurations for export. Record
IDs are namespaced by dataset name and revision, so native identifiers such
as OMat24 ``sid`` are never assumed unique across datasets, shards or
trajectories.
"""

import abc
import enum
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ase import Atoms

from envtok.adapters._labels import DEFAULT_LABEL_SOURCES, DEFAULT_UNITS, extract_labels
from envtok.errors import ConfigError
from envtok.records import Label, SourceRef, Split, StructureRecord, scalar_metadata


class IdPolicy(enum.StrEnum):
    """How record IDs are formed.

    Attributes:
        NAMESPACED_SOURCE_RECORD: ``dataset@revision:shard:key`` (default).
            Unique by construction.
        NAMESPACED_NATIVE_ID: ``dataset@revision:native_id``. Requires the
            native ID to be unique; duplicates are rejected at index time.
    """

    NAMESPACED_SOURCE_RECORD = "namespaced_source_record"
    NAMESPACED_NATIVE_ID = "namespaced_native_id"


@dataclass(frozen=True, slots=True)
class RecordOptions:
    """Options shared by every adapter for turning atoms into records.

    Attributes:
        dataset: Dataset name used to namespace record IDs.
        revision: Dataset revision; part of every record ID.
        id_policy: Record ID policy.
        native_id_key: ``info`` key holding a native identifier (``sid``).
        split: Constant split for every record, unless ``split_key`` is set.
        split_key: ``info`` key holding a per-record split name.
        group_key: ``info`` key holding a parent material/trajectory ID.
        label_sources: Label name to ``calc:<key>``, ``info:<key>`` or
            ``arrays:<key>``.
        label_units: Declared unit of each label. EnvTok never converts units.
    """

    dataset: str
    revision: str
    id_policy: IdPolicy = IdPolicy.NAMESPACED_SOURCE_RECORD
    native_id_key: str | None = "sid"
    split: Split = Split.UNSPLIT
    split_key: str | None = None
    group_key: str | None = None
    label_sources: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LABEL_SOURCES)
    )
    label_units: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))

    def __post_init__(self) -> None:
        """Validate the options and normalise enum values given as text."""
        object.__setattr__(self, "id_policy", IdPolicy(self.id_policy))
        object.__setattr__(self, "split", Split.parse(self.split))
        if not self.dataset or not self.revision:
            raise ConfigError("dataset name and revision must be non-empty")
        if self.id_policy is IdPolicy.NAMESPACED_NATIVE_ID and not self.native_id_key:
            raise ConfigError("id_policy namespaced_native_id requires native_id_key")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for fingerprints."""
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "id_policy": self.id_policy.value,
            "native_id_key": self.native_id_key,
            "split": self.split.value,
            "split_key": self.split_key,
            "group_key": self.group_key,
            "label_sources": dict(sorted(self.label_sources.items())),
            "label_units": dict(sorted(self.label_units.items())),
        }


@dataclass(frozen=True, slots=True)
class Locator:
    """Deterministic position of one record in a source.

    Attributes:
        ordinal: Global position in the adapter's iteration order.
        shard: Shard identifier.
        key: Native key within the shard.
        offset: Optional byte offset (extxyz).
    """

    ordinal: int
    shard: str
    key: str
    offset: int | None = None


def make_record(
    atoms: Atoms, shard: str, key: str, options: RecordOptions
) -> StructureRecord:
    """Build a record from atoms read at ``shard``/``key``.

    Args:
        atoms: Atoms as read from the source (labels in ``calc``/``info``).
        shard: Shard identifier relative to the source root.
        key: Native row key or frame index.
        options: Record options.

    Returns:
        The record.

    Raises:
        ConfigError: If the native ID policy needs an ID the frame lacks.
    """
    info = atoms.info
    native = info.get(options.native_id_key) if options.native_id_key else None
    native_id = None if native is None else str(native)
    namespace = f"{options.dataset}@{options.revision}"
    if options.id_policy is IdPolicy.NAMESPACED_NATIVE_ID:
        if native_id is None:
            raise ConfigError(
                f"record {shard}:{key} has no {options.native_id_key!r} for "
                "id_policy namespaced_native_id"
            )
        record_id = f"{namespace}:{native_id}"
    else:
        record_id = f"{namespace}:{shard}:{key}"
    split = (
        Split.parse(info[options.split_key])
        if options.split_key and options.split_key in info
        else options.split
    )
    group = info.get(options.group_key) if options.group_key else None
    labels: dict[str, Label] = extract_labels(
        atoms, options.label_sources, options.label_units
    )
    consumed = {
        spec.split(":", 1)[1]
        for spec in options.label_sources.values()
        if spec.startswith("info:")
    }
    return StructureRecord(
        record_id=record_id,
        source_ref=SourceRef(
            dataset=options.dataset,
            revision=options.revision,
            shard=shard,
            key=key,
            native_id=native_id,
        ),
        atoms=atoms,
        split=split,
        group_id=None if group is None else str(group),
        metadata=scalar_metadata(info, skip=frozenset(consumed)),
        labels=labels,
    )


def file_fingerprint(path: Path, *, checksum: str = "stat") -> dict[str, Any]:
    """Identify a source file for cache validation.

    Args:
        path: The file.
        checksum: ``"stat"`` (size and modification time, cheap) or ``"full"``
            (adds a SHA-256 of the contents).

    Returns:
        A JSON-compatible fingerprint.

    Raises:
        ConfigError: If ``checksum`` is not ``stat`` or ``full``.
    """
    if checksum not in {"stat", "full"}:
        raise ConfigError(f"checksum must be 'stat' or 'full', got {checksum!r}")
    stat = path.stat()
    result: dict[str, Any] = {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if checksum == "full":
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        result["sha256"] = digest.hexdigest()
    return result


class Adapter(abc.ABC):
    """Base class of dataset adapters.

    Subclasses define how locators are enumerated and records are read. An
    adapter must be reconstructible in a worker process from :meth:`spec`.
    """

    kind: str = "abstract"
    options: RecordOptions

    @property
    def supports_random_access(self) -> bool:
        """Whether :meth:`read` can fetch a record by locator in any process."""
        return True

    @property
    def supports_retrieval(self) -> bool:
        """Whether original records can be retrieved later for export."""
        return True

    @property
    def retrieval_limitation(self) -> str | None:
        """Explanation shown when retrieval is unavailable."""
        return None

    @abc.abstractmethod
    def spec(self) -> dict[str, Any]:
        """Return a picklable, JSON-compatible description of the adapter."""

    @abc.abstractmethod
    def source_fingerprint(self) -> dict[str, Any]:
        """Return a fingerprint of the source content and adapter options."""

    @abc.abstractmethod
    def locators(self) -> Iterator[Locator]:
        """Yield every record locator in a deterministic order."""

    @abc.abstractmethod
    def read(self, locator: Locator) -> StructureRecord:
        """Read one record.

        Args:
            locator: A locator produced by :meth:`locators`.

        Returns:
            The record.
        """

    def iter_records(self) -> Iterator[StructureRecord]:
        """Yield every record in locator order."""
        for locator in self.locators():
            yield self.read(locator)

    @abc.abstractmethod
    def retrieve(self, refs: Sequence[SourceRef]) -> Iterator[tuple[SourceRef, Atoms]]:
        """Yield the original atoms of the referenced records.

        Args:
            refs: Source references of the records to retrieve.

        Returns:
            ``(ref, atoms)`` pairs in the order of ``refs``.
        """

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release open resources."""


def sorted_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    """Return files under ``root`` matching any pattern, sorted by relative path.

    Args:
        root: A file or directory.
        patterns: Glob patterns applied recursively when ``root`` is a directory.

    Returns:
        Sorted matching files.

    Raises:
        ConfigError: If ``root`` does not exist or nothing matches.
    """
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ConfigError(f"source path {root} does not exist")
    found = {
        path for pattern in patterns for path in root.glob(pattern) if path.is_file()
    }
    if not found:
        raise ConfigError(f"no files matching {list(patterns)} under {root}")
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def shard_name(path: Path, root: Path) -> str:
    """Return the shard identifier of ``path`` relative to the source root."""
    if root.is_file():
        return path.name
    return path.relative_to(root).as_posix()
