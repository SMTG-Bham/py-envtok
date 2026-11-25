"""Columnar molecular target manifests with digest-checked tables.

A run's ``targets.json`` is a small manifest. It records the SHA-256 digest of
each table under ``targets/``, which hold the frozen hierarchy, the incidence,
the geometry regions and the overflow environments as Parquet tables and a
NumPy array of prototypes. Tokens are stored once, in a sorted vocabulary, and
referenced by position, so each token string is created once when loaded.
Consumers can read only the tables they need. Runs written before
``molecular_v1.6`` hold everything in ``targets.json`` and are still read.
"""

from __future__ import annotations

import array
import hashlib
import itertools
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from envtok.errors import ConfigError, StaleCacheError
from envtok.molecular.geometry import GeometryRegions
from envtok.molecular.selection import Candidate, HierarchyResult, Target

Built: TypeAlias = tuple[
    tuple[Candidate, ...], dict[str, Target], HierarchyResult, GeometryRegions
]
AtomClasses: TypeAlias = tuple[Sequence[str], npt.NDArray[np.integer[Any]]]
"""Class names, and each descriptor row's position in them."""

TARGETS = "targets.json"
"""The run's target manifest; ``provenance.json`` records its digest."""
TARGET_TABLES = "targets"
"""Directory beside ``targets.json`` holding the tables it lists."""
_LAYOUT = "columnar-v1"
_TOKENS = "tokens.parquet"
_INCIDENCE = "incidence.parquet"
_REGIONS = "regions.parquet"
_HIERARCHY = "hierarchy.parquet"
_CLASSES = "classes.parquet"
_REQUIREMENTS = "requirements.parquet"
_GEOMETRY = "geometry.parquet"
_PROTOTYPES = "prototypes.npy"
_ATOM_CLASSES = "atom_classes.npy"
_OVERFLOW = "overflow.parquet"
_CORRUPT = "frozen molecular targets are missing or corrupt"
_LEGACY_FIELDS = {
    "incidence",
    "requirements",
    "mapping",
    "lineage",
    "support",
    "group_support",
    "rare_targets",
    "optional_targets",
    "prototypes",
    "region_targets",
    "overflow",
    "sampled_counts",
    "region_radius",
}


def file_hash(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write indented, key-sorted JSON atomically, streaming it to disk."""
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def read_json(path: Path, required: set[str]) -> dict[str, Any]:
    """Read a JSON object with at least the ``required`` keys."""
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not required <= value.keys():
            raise ValueError("missing manifest fields")
    except (OSError, ValueError) as exc:
        raise StaleCacheError(f"cannot read molecular manifest {path}: {exc}") from exc
    return value


class HashingWriter:
    """Pass bytes to a binary stream while computing their SHA-256 digest."""

    def __init__(self, stream: BinaryIO) -> None:
        """Wrap ``stream``."""
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data: Any) -> int:
        """Hash and write ``data``, which supports the buffer protocol."""
        self.digest.update(data)
        return self.stream.write(data)


def as_bytes(values: npt.ArrayLike, dtype: Any) -> memoryview:
    """A flat byte view of ``values`` as a C-ordered array of ``dtype``."""
    return np.ascontiguousarray(values, dtype=dtype).data.cast("B")


def write_array_header(
    stream: HashingWriter, dtype: Any, shape: tuple[int, ...]
) -> None:
    """Start a C-ordered ``.npy`` file whose rows are then written in order."""
    np.lib.format.write_array_header_1_0(
        stream,
        {
            "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
            "fortran_order": False,
            "shape": shape,
        },
    )


def write_table(path: Path, table: pa.Table) -> str:
    """Write a Parquet table atomically and return its digest."""
    partial = path.with_name(path.name + ".partial")
    pq.write_table(table, partial, compression="zstd")
    partial.replace(path)
    return file_hash(path)


def list_offsets(lengths: Sequence[int]) -> pa.Array:
    """Large-list offsets for rows of the given lengths."""
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return pa.array(offsets, pa.int64())


def token_lists(lengths: Sequence[int], values: array.array[int]) -> pa.Array:
    """A list-of-token-ID column from per-row lengths and concatenated IDs."""
    return pa.LargeListArray.from_arrays(
        list_offsets(lengths),
        pa.array(np.frombuffer(values, dtype=np.int32), pa.int32()),
    )


def list_columns(
    table: pa.Table, name: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int32]]:
    """Offsets into, and the concatenated values of, a list column."""
    column = table.column(name).combine_chunks()
    return column.offsets.to_numpy(), column.values.to_numpy(zero_copy_only=False)


def _ids(index: Mapping[str, int], tokens: Iterable[str]) -> list[int]:
    return sorted(index[token] for token in tokens)


def _target_tokens(built: Built) -> set[str]:
    candidates, mandatory, hierarchy, regions = built
    tokens: set[str] = set(mandatory)
    for row in candidates:
        tokens.update(row.targets, row.refinement_targets)
    for path in hierarchy.lineage.values():
        tokens.update(path)
    tokens.update(hierarchy.mapping.values(), hierarchy.support)
    for targets in regions.targets.values():
        tokens.update(targets)
    tokens.update(regions.prototypes, (row[2] for row in regions.overflow))
    return tokens


def write_targets(directory: Path, built: Built, atom_classes: AtomClasses) -> None:
    """Write the frozen target tables, then ``targets.json`` listing their digests.

    ``atom_classes`` gives the fitted class of every descriptor row, in
    feature-cache order, so audits need not map each atom's tokens again.

    Raises:
        ConfigError: If costs are not whole atom counts, or prototype
            dimensions differ between classes.
    """
    candidates, mandatory, hierarchy, regions = built
    if any(type(row.cost) is not int for row in candidates):
        raise ConfigError("frozen molecular targets require whole-number atom costs")
    if len({value.shape[1:] for value in regions.prototypes.values()}) > 1:
        raise ConfigError("geometry prototypes must share one descriptor dimension")
    folder = directory / TARGET_TABLES
    folder.mkdir(exist_ok=True)
    vocabulary = sorted(_target_tokens(built))
    index = {token: position for position, token in enumerate(vocabulary)}
    classes = sorted(regions.prototypes)
    required = sorted(mandatory)
    tables: list[tuple[str, Callable[[], pa.Table]]] = [
        (_TOKENS, lambda: pa.table({"token": pa.array(vocabulary, pa.string())})),
        (_INCIDENCE, lambda: _incidence_table(candidates, index)),
        (_REGIONS, lambda: _region_table(regions, index)),
        (_HIERARCHY, lambda: _hierarchy_table(hierarchy, index)),
        (_CLASSES, lambda: _class_table(hierarchy, index)),
        (
            _REQUIREMENTS,
            lambda: pa.table(
                {
                    "token": pa.array([index[key] for key in required], pa.int32()),
                    "requirement": pa.array(
                        [mandatory[key].requirement for key in required], pa.int64()
                    ),
                    "weight": pa.array(
                        [mandatory[key].weight for key in required], pa.float64()
                    ),
                }
            ),
        ),
        (
            _GEOMETRY,
            lambda: pa.table(
                {
                    "class": pa.array([index[key] for key in classes], pa.int32()),
                    "sampled_count": pa.array(
                        [regions.sampled_counts[key] for key in classes], pa.int64()
                    ),
                    "prototype_rows": pa.array(
                        [len(regions.prototypes[key]) for key in classes], pa.int64()
                    ),
                }
            ),
        ),
        (_OVERFLOW, lambda: _overflow_table(regions, index)),
    ]
    digests = {name: write_table(folder / name, build()) for name, build in tables}
    digests[_PROTOTYPES] = _write_prototypes(folder / _PROTOTYPES, regions)
    names, codes = atom_classes
    positions = np.asarray([index[name] for name in names], dtype=np.int32)
    digests[_ATOM_CLASSES] = _write_array(
        folder / _ATOM_CLASSES, positions[np.asarray(codes, dtype=np.int64)]
    )
    write_json(
        directory / TARGETS,
        {"layout": _LAYOUT, "region_radius": regions.radius, "files": digests},
    )


def _incidence_table(
    candidates: Sequence[Candidate], index: Mapping[str, int]
) -> pa.Table:
    targets: array.array[int] = array.array("i")
    refinements: array.array[int] = array.array("i")
    target_lengths, refinement_lengths = [], []
    for row in candidates:
        ids = _ids(index, row.targets)
        targets.extend(ids)
        target_lengths.append(len(ids))
        ids = _ids(index, row.refinement_targets)
        refinements.extend(ids)
        refinement_lengths.append(len(ids))
    return pa.table(
        {
            "record_id": pa.array([row.record_id for row in candidates], pa.string()),
            "cost": pa.array([row.cost for row in candidates], pa.int64()),
            "group_id": pa.array([row.group_id for row in candidates], pa.string()),
            "targets": token_lists(target_lengths, targets),
            "refinement_targets": token_lists(refinement_lengths, refinements),
        }
    )


def _overflow_table(regions: GeometryRegions, index: Mapping[str, int]) -> pa.Table:
    rows = regions.overflow
    return pa.table(
        {
            "record_id": pa.array([row[0] for row in rows], pa.string()),
            "atom_index": pa.array([row[1] for row in rows], pa.int64()),
            "class": pa.array([index[row[2]] for row in rows], pa.int32()),
            "distance": pa.array([row[3] for row in rows], pa.float64()),
        }
    )


def _region_table(regions: GeometryRegions, index: Mapping[str, int]) -> pa.Table:
    values: array.array[int] = array.array("i")
    lengths = []
    for record_id in sorted(regions.targets):
        ids = _ids(index, regions.targets[record_id])
        values.extend(ids)
        lengths.append(len(ids))
    return pa.table(
        {
            "record_id": pa.array(sorted(regions.targets), pa.string()),
            "targets": token_lists(lengths, values),
        }
    )


def _hierarchy_table(hierarchy: HierarchyResult, index: Mapping[str, int]) -> pa.Table:
    fine = sorted(hierarchy.mapping)
    values: array.array[int] = array.array("i")
    lengths = []
    for token in fine:
        path = hierarchy.lineage[token]
        values.extend(index[step] for step in path)
        lengths.append(len(path))
    return pa.table(
        {
            "fine": pa.array([index[token] for token in fine], pa.int32()),
            "class": pa.array(
                [index[hierarchy.mapping[token]] for token in fine], pa.int32()
            ),
            "lineage": token_lists(lengths, values),
        }
    )


def _class_table(hierarchy: HierarchyResult, index: Mapping[str, int]) -> pa.Table:
    classes = sorted(hierarchy.support)
    columns: dict[str, pa.Array] = {
        "token": pa.array([index[token] for token in classes], pa.int32())
    }
    for name in ("support", "group_support", "source_support", "source_group_support"):
        values = getattr(hierarchy, name)
        columns[name] = pa.array([values[token] for token in classes], pa.int64())
    for name, flagged in (
        ("rare", hierarchy.rare_targets),
        ("optional", hierarchy.optional_targets),
    ):
        columns[name] = pa.array([token in flagged for token in classes], pa.bool_())
    return pa.table(columns)


def _write_array(path: Path, values: npt.NDArray[Any]) -> str:
    """Write a 1-D or 2-D array atomically and return its digest."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as raw:
        stream = HashingWriter(raw)
        write_array_header(stream, values.dtype, values.shape)
        stream.write(as_bytes(values, values.dtype))
    partial.replace(path)
    return stream.digest.hexdigest()


def _write_prototypes(path: Path, regions: GeometryRegions) -> str:
    """Stream every class's prototypes, in class order, into one float64 array."""
    classes = sorted(regions.prototypes)
    rows = sum(len(regions.prototypes[key]) for key in classes)
    width = regions.prototypes[classes[0]].shape[1] if classes else 0
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as raw:
        stream = HashingWriter(raw)
        write_array_header(stream, np.float64, (rows, width))
        for key in classes:
            stream.write(as_bytes(regions.prototypes[key], np.float64))
    partial.replace(path)
    return stream.digest.hexdigest()


class FrozenTargets:
    """Digest-checked access to a run's frozen target tables.

    Every table listed in ``targets.json`` is checked against its digest when
    the targets are opened. Each method reads only the tables it needs.
    """

    def __init__(self, directory: Path, *, verify: bool = True) -> None:
        """Open ``directory/targets.json`` and verify the tables it lists.

        Raises:
            StaleCacheError: If the manifest or a table is missing or changed.
        """
        self.directory = directory
        self.folder = directory / TARGET_TABLES
        manifest = read_json(directory / TARGETS, set())
        self.legacy: dict[str, Any] | None = None
        if "layout" not in manifest:
            if not manifest.keys() >= _LEGACY_FIELDS:
                raise StaleCacheError(_CORRUPT)
            self.legacy = manifest
            return
        files = manifest.get("files")
        if manifest["layout"] != _LAYOUT or not isinstance(files, dict):
            raise StaleCacheError(_CORRUPT)
        for name in (
            _TOKENS,
            _INCIDENCE,
            _REGIONS,
            _HIERARCHY,
            _CLASSES,
            _REQUIREMENTS,
            _GEOMETRY,
            _PROTOTYPES,
            _OVERFLOW,
            _ATOM_CLASSES,
        ):
            path = self.folder / name
            if not path.is_file() or (verify and file_hash(path) != files.get(name)):
                raise StaleCacheError(_CORRUPT)
        self.radius = manifest["region_radius"]

    @classmethod
    def unverified(cls, directory: Path) -> FrozenTargets:
        """Open targets this process has just written, without re-reading them."""
        return cls(directory, verify=False)

    def _table(self, name: str, columns: list[str] | None = None) -> pa.Table:
        return pq.read_table(self.folder / name, columns=columns)

    @cached_property
    def tokens(self) -> list[str]:
        """The sorted token vocabulary that target tables refer to by position."""
        tokens: list[str] = self._table(_TOKENS).column("token").to_pylist()
        return tokens

    def _token_sets(self, table: pa.Table, name: str) -> list[frozenset[str]]:
        offsets, values = list_columns(table, name)
        tokens = self.tokens
        return [
            frozenset(map(tokens.__getitem__, values[start:stop].tolist()))
            for start, stop in itertools.pairwise(offsets.tolist())
        ]

    def candidates(self) -> tuple[Candidate, ...]:
        """Whole-record costs, mandatory targets and refinement targets."""
        if self.legacy is not None:
            return self._legacy[0]
        return self._candidates({})

    def _candidates(
        self, regions: Mapping[str, frozenset[str]]
    ) -> tuple[Candidate, ...]:
        """Incidence rows; refinements equal to a record's regions share its set."""
        table = self._table(_INCIDENCE)
        rows = []
        for record_id, cost, group_id, targets, refinement in zip(
            table.column("record_id").to_pylist(),
            table.column("cost").to_pylist(),
            table.column("group_id").to_pylist(),
            self._token_sets(table, "targets"),
            self._token_sets(table, "refinement_targets"),
            strict=True,
        ):
            region = regions.get(record_id)
            shared = region if region == refinement else refinement
            rows.append(Candidate(record_id, cost, targets, shared, group_id))
        return tuple(rows)

    def requirements(self) -> dict[str, Target]:
        """Mandatory targets in token order."""
        if self.legacy is not None:
            return self._legacy[1]
        table = self._table(_REQUIREMENTS)
        return {
            self.tokens[token]: Target(requirement, weight)
            for token, requirement, weight in zip(
                table.column("token").to_pylist(),
                table.column("requirement").to_pylist(),
                table.column("weight").to_pylist(),
                strict=True,
            )
        }

    def mapping(self) -> dict[str, str]:
        """Each fine token's fitted class, in token order."""
        if self.legacy is not None:
            return self._legacy[2].mapping
        table = self._table(_HIERARCHY, ["fine", "class"])
        tokens = self.tokens
        return {
            tokens[fine]: tokens[cls]
            for fine, cls in zip(
                table.column("fine").to_pylist(),
                table.column("class").to_pylist(),
                strict=True,
            )
        }

    def overflow_count(self) -> int:
        """Environments outside every prototype radius."""
        if self.legacy is not None:
            return len(self.legacy["overflow"])
        return int(pq.ParquetFile(self.folder / _OVERFLOW).metadata.num_rows)

    def atom_classes(self) -> AtomClasses | None:
        """Every descriptor row's class, as positions in the token vocabulary.

        Runs saved before ``molecular_v1.6`` did not record them.
        """
        if self.legacy is not None:
            return None
        codes = np.load(self.folder / _ATOM_CLASSES, mmap_mode="r", allow_pickle=False)
        return self.tokens, codes

    def prototypes(self) -> dict[str, npt.NDArray[np.float64]]:
        """Each class's prototypes, as views of the memory-mapped array."""
        if self.legacy is not None:
            return self._legacy[3].prototypes
        table = self._table(_GEOMETRY)
        matrix = np.load(self.folder / _PROTOTYPES, mmap_mode="r", allow_pickle=False)
        bounds = np.cumsum([0, *table.column("prototype_rows").to_pylist()]).tolist()
        return {
            self.tokens[cls]: matrix[start:stop]
            for cls, (start, stop) in zip(
                table.column("class").to_pylist(),
                itertools.pairwise(bounds),
                strict=True,
            )
        }

    @cached_property
    def _legacy(self) -> Built:
        assert self.legacy is not None  # noqa: S101 - only reached for legacy runs
        return _legacy_targets(self.legacy)

    def load(self) -> Built:
        """Candidates, mandatory targets, fitted hierarchy and geometry regions."""
        if self.legacy is not None:
            return self._legacy
        tokens = self.tokens
        regions_table = self._table(_REGIONS)
        region_targets = dict(
            zip(
                regions_table.column("record_id").to_pylist(),
                self._token_sets(regions_table, "targets"),
                strict=True,
            )
        )
        candidates = self._candidates(region_targets)
        names = {row.record_id: row.record_id for row in candidates}
        table = self._table(_HIERARCHY)
        offsets, values = list_columns(table, "lineage")
        fine = [tokens[token] for token in table.column("fine").to_pylist()]
        lineage = {
            token: tuple(map(tokens.__getitem__, values[start:stop].tolist()))
            for token, (start, stop) in zip(
                fine, itertools.pairwise(offsets.tolist()), strict=True
            )
        }
        classes = self._table(_CLASSES)
        names_of = [tokens[token] for token in classes.column("token").to_pylist()]

        def counts(column: str) -> dict[str, int]:
            return dict(zip(names_of, classes.column(column).to_pylist(), strict=True))

        def flagged(column: str) -> frozenset[str]:
            return frozenset(
                token
                for token, flag in zip(
                    names_of, classes.column(column).to_pylist(), strict=True
                )
                if flag
            )

        hierarchy = HierarchyResult(
            targets={row.record_id: row.targets for row in candidates},
            mapping=self.mapping(),
            support=counts("support"),
            group_support=counts("group_support"),
            rare_targets=flagged("rare"),
            optional_targets=flagged("optional"),
            lineage=lineage,
            source_support=counts("source_support"),
            source_group_support=counts("source_group_support"),
        )
        geometry = self._table(_GEOMETRY)
        overflow = self._table(_OVERFLOW)
        regions = GeometryRegions(
            targets=region_targets,
            prototypes=self.prototypes(),
            overflow=tuple(
                (names.get(record_id, record_id), atom, tokens[cls], distance)
                for record_id, atom, cls, distance in zip(
                    overflow.column("record_id").to_pylist(),
                    overflow.column("atom_index").to_pylist(),
                    overflow.column("class").to_pylist(),
                    overflow.column("distance").to_pylist(),
                    strict=True,
                )
            ),
            sampled_counts={
                tokens[cls]: count
                for cls, count in zip(
                    geometry.column("class").to_pylist(),
                    geometry.column("sampled_count").to_pylist(),
                    strict=True,
                )
            },
            radius=self.radius,
        )
        return candidates, self.requirements(), hierarchy, regions


def _legacy_targets(data: Mapping[str, Any]) -> Built:
    """Targets saved before ``molecular_v1.6``, all in one JSON document."""
    candidates = tuple(
        Candidate(
            row["record_id"],
            row["cost"],
            frozenset(row["targets"]),
            frozenset(row["refinement_targets"]),
            row.get("group_id"),
        )
        for row in data["incidence"]
    )
    mandatory = {key: Target(**value) for key, value in data["requirements"].items()}
    hierarchy = HierarchyResult(
        targets={row.record_id: row.targets for row in candidates},
        mapping=data["mapping"],
        support=data["support"],
        group_support=data["group_support"],
        rare_targets=frozenset(data["rare_targets"]),
        optional_targets=frozenset(data["optional_targets"]),
        lineage={key: tuple(value) for key, value in data["lineage"].items()},
        source_support=data["source_support"],
        source_group_support=data["source_group_support"],
    )
    regions = GeometryRegions(
        targets={
            key: frozenset(value) for key, value in data["region_targets"].items()
        },
        prototypes={
            key: np.asarray(value, dtype=np.float64)
            for key, value in data["prototypes"].items()
        },
        overflow=tuple(tuple(row) for row in data["overflow"]),
        sampled_counts=data["sampled_counts"],
        radius=data["region_radius"],
    )
    return candidates, mandatory, hierarchy, regions
