"""Bounded geometric prototypes and explicit unresolved environments.

These pilot utilities never form a configuration-by-configuration distance
matrix. Prototype selection retains at most the declared reservoir per class;
assignment and nearest-selected audits convert environments to float64 in
bounded chunks, with the same arithmetic as one environment at a time.

Environments are given either as a mapping from each record to its
``(class, vector)`` pairs, or as a :class:`DescriptorPool` of arrays, such as a
memory-mapped descriptor cache, which needs no Python object per environment.
Both give identical results.
"""

import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

from envtok.errors import ConfigError
from envtok.molecular.selection import _positive_integer

Descriptor: TypeAlias = tuple[str, npt.NDArray[np.float64]]
DescriptorRows: TypeAlias = Mapping[str, Sequence[Descriptor]]
EnvironmentRef: TypeAlias = tuple[str, int, npt.NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class GeometryRegions:
    """Approximate regions, bounded prototypes and individually reported overflow."""

    targets: dict[str, frozenset[str]]
    prototypes: dict[str, npt.NDArray[np.float64]]
    overflow: tuple[tuple[str, int, str, float], ...]
    sampled_counts: dict[str, int]
    radius: float


_NONFINITE = "local descriptors must be nonempty finite vectors"
_MATRIX = 2
_CHUNK_BYTES = 1 << 21
"""Float64 working-block size: bounds memory and keeps blocks cache-resident."""


@dataclass(frozen=True, slots=True)
class DescriptorPool:
    """Environments as arrays, one descriptor row per environment.

    Record ``i`` owns rows ``offsets[i]:offsets[i + 1]`` of ``vectors``, in atom
    order, and ``classes[row]`` is the position of that row's class in
    ``class_ids``. ``vectors`` may be any 2-D array, including a memory map.
    """

    record_ids: Sequence[str]
    offsets: npt.NDArray[np.integer[Any]]
    class_ids: Sequence[str]
    classes: npt.NDArray[np.integer[Any]]
    vectors: npt.NDArray[Any]


Descriptors: TypeAlias = DescriptorRows | DescriptorPool


@dataclass(slots=True)
class _Gathered:
    """One class's environments while a descriptor mapping is grouped."""

    records: list[int] = field(default_factory=list)
    atoms: list[int] = field(default_factory=list)
    vectors: list[Any] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _ClassRows:
    """One class's environments in record then atom order, unconverted.

    ``records`` index ``names``. Vectors are either listed one per environment
    or are the given ``rows`` of a shared ``matrix``.
    """

    names: Sequence[str]
    records: npt.NDArray[np.int64]
    atoms: npt.NDArray[np.int64]
    vectors: list[Any] | None = None
    matrix: npt.NDArray[Any] | None = None
    rows: npt.NDArray[np.int64] | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def dimension(self) -> int:
        if self.matrix is not None:
            return int(self.matrix.shape[1])
        assert self.vectors is not None  # noqa: S101 - one source is always set
        return len(self.vectors[0])

    def block(self, key: slice | list[int]) -> npt.NDArray[np.float64]:
        """Float64 unit rows for a range or list of positions."""
        if self.matrix is not None:
            assert self.rows is not None  # noqa: S101 - set with the matrix
            return _unit_rows(self.matrix[self.rows[key]])
        assert self.vectors is not None  # noqa: S101 - one source is always set
        if isinstance(key, slice):
            return _unit_rows(self.vectors[key])
        return _unit_rows([self.vectors[position] for position in key])


def _unit_rows(vectors: Sequence[Any] | npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Return float64 unit rows, bit-identical to normalising each vector alone.

    ``np.linalg.norm`` of a 1-D vector is ``sqrt(x.dot(x))``; ``np.vecdot``
    computes each row's dot with the same kernel, unlike ``norm(axis=1)``.
    Zero-norm rows are kept unchanged.
    """
    matrix = np.array(vectors, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ConfigError(_NONFINITE)
    norms = np.sqrt(np.vecdot(matrix, matrix))[:, None]
    np.divide(matrix, norms, out=matrix, where=norms != 0)
    return matrix


def _chunks(count: int, row_bytes: int) -> list[tuple[int, int]]:
    """Split ``count`` rows into ranges whose float64 working set stays bounded."""
    step = max(1, _CHUNK_BYTES // max(row_bytes, 1))
    return [(start, min(start + step, count)) for start in range(0, count, step)]


def _order(seed: int, *parts: object) -> str:
    return hashlib.sha256(repr((seed, *parts)).encode()).hexdigest()


def _record_ids(descriptors: Descriptors) -> Sequence[str]:
    if isinstance(descriptors, DescriptorPool):
        return descriptors.record_ids
    return list(descriptors)


def _classes(descriptors: Descriptors) -> dict[str, _ClassRows]:
    """Group environments by class, in sorted record then atom order.

    Classes appear in order of first environment. For a mapping, shapes are
    checked in that order and values for finiteness when each class is
    converted; an earlier non-finite vector still takes precedence over a
    dimension mismatch.
    """
    if isinstance(descriptors, DescriptorPool):
        return _pool_classes(descriptors)
    names = sorted(descriptors)
    gathered: dict[str, _Gathered] = {}
    dimensions: dict[str, int] = {}
    for record_index, record_id in enumerate(names):
        for atom_index, (class_id, vector) in enumerate(descriptors[record_id]):
            shape = np.shape(vector)
            if len(shape) != 1 or not shape[0]:
                raise ConfigError(_NONFINITE)
            if dimensions.setdefault(class_id, shape[0]) != shape[0]:
                earlier = [v for rows in gathered.values() for v in rows.vectors]
                if not all(np.isfinite(v).all() for v in [*earlier, vector]):
                    raise ConfigError(_NONFINITE)
                raise ConfigError(f"incompatible descriptor dimensions for {class_id}")
            rows = gathered.setdefault(class_id, _Gathered())
            rows.records.append(record_index)
            rows.atoms.append(atom_index)
            rows.vectors.append(vector)
    return {
        class_id: _ClassRows(
            names,
            np.asarray(rows.records, dtype=np.int64),
            np.asarray(rows.atoms, dtype=np.int64),
            vectors=rows.vectors,
        )
        for class_id, rows in gathered.items()
    }


def _pool_classes(pool: DescriptorPool) -> dict[str, _ClassRows]:
    """Group a pool's rows by class without a Python object per environment."""
    count = len(pool.record_ids)
    offsets = np.asarray(pool.offsets, dtype=np.int64)
    codes = np.asarray(pool.classes, dtype=np.int64)
    environments = len(codes)
    if (
        offsets.shape != (count + 1,)
        or offsets[0] != 0
        or offsets[-1] != environments
        or (np.diff(offsets) < 0).any()
        or pool.vectors.ndim != _MATRIX
        or len(pool.vectors) != environments
        or (environments and not pool.vectors.shape[1])
        or ((codes < 0) | (codes >= len(pool.class_ids))).any()
        or len(set(pool.record_ids)) != count
    ):
        raise ConfigError("descriptor pool rows, offsets and classes are inconsistent")
    order = np.asarray(
        sorted(range(count), key=pool.record_ids.__getitem__), dtype=np.int64
    )
    lengths = np.diff(offsets)[order]
    starts = np.cumsum(lengths) - lengths
    positions = np.arange(environments, dtype=np.int64)
    rows = np.repeat(offsets[:-1][order] - starts, lengths) + positions
    records = np.repeat(order, lengths)
    atoms = positions - np.repeat(starts, lengths)
    codes = codes[rows]
    grouped = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[grouped], np.arange(len(pool.class_ids) + 1))
    present, first = np.unique(codes, return_index=True)
    classes = {}
    for code in present[np.argsort(first)].tolist():
        members = grouped[bounds[code] : bounds[code + 1]]
        classes[pool.class_ids[code]] = _ClassRows(
            pool.record_ids,
            records[members],
            atoms[members],
            matrix=pool.vectors,
            rows=rows[members],
        )
    return classes


def _reservoir(
    rows: _ClassRows,
    groups: Mapping[str, str | None],
    limit: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Round-robin over seeded group order, each group in seeded atom order.

    Only the first ``depth`` environments of each group can be drawn, so each
    group is ranked with a partial sort rather than a full one.
    """
    record_ids = [rows.names[record] for record in rows.records.tolist()]
    atoms = rows.atoms.tolist()
    buckets: dict[str, list[int]] = defaultdict(list)
    for position, record_id in enumerate(record_ids):
        buckets[groups.get(record_id) or record_id].append(position)
    target = min(limit, len(record_ids))
    sizes = sorted((len(bucket) for bucket in buckets.values()), reverse=True)
    depth = drawn = 0
    while drawn < target:
        depth += 1
        drawn += sum(1 for size in sizes if size >= depth)

    def rank(position: int) -> str:
        return _order(seed, record_ids[position], atoms[position])

    ranked = {
        group: heapq.nsmallest(depth, bucket, key=rank)
        for group, bucket in buckets.items()
    }
    ordered_groups = sorted(ranked, key=lambda group: _order(seed, group))
    chosen: list[int] = []
    level = 0
    while len(chosen) < target:
        for group in ordered_groups:
            if level < len(ranked[group]):
                chosen.append(ranked[group][level])
                if len(chosen) == target:
                    break
        level += 1
    return rows.block(chosen)


def _prototypes(
    reservoir: npt.NDArray[np.float64], radius: float, cap: int
) -> npt.NDArray[np.float64]:
    chosen = [0]
    distances = np.linalg.norm(reservoir - reservoir[0], axis=1)
    while len(chosen) < min(cap, len(reservoir)):
        index = int(np.argmax(distances))
        if distances[index] <= radius:
            break
        chosen.append(index)
        distances = np.minimum(
            distances, np.linalg.norm(reservoir - reservoir[index], axis=1)
        )
    return reservoir[chosen].copy()


def build_geometry_regions(
    descriptors: Descriptors,
    *,
    group_ids: Mapping[str, str | None] | None = None,
    radius: float = 0.2,
    prototype_cap: int = 16,
    reservoir_size: int = 256,
    seed: int = 0,
) -> GeometryRegions:
    """Fit group-balanced reservoirs, FPS prototypes and radius-limited regions.

    Zero vectors are retained as an explicit degenerate descriptor value. An
    environment outside every prototype radius gets no region target and is
    reported separately; selecting one overflow never covers another overflow.
    """
    if not math.isfinite(radius) or radius < 0:
        raise ConfigError("prototype region radius must be finite and nonnegative")
    for value in (prototype_cap, reservoir_size):
        if not _positive_integer(value):
            raise ConfigError(
                "prototype_cap and reservoir_size must be positive integers"
            )
    classes = _classes(descriptors)
    targets: dict[str, set[str]] = {
        rid: set() for rid in sorted(_record_ids(descriptors))
    }
    prototypes: dict[str, npt.NDArray[np.float64]] = {}
    overflow: list[tuple[str, int, str, float]] = []
    sampled_counts: dict[str, int] = {}
    for class_id, rows in sorted(classes.items()):
        reservoir = _reservoir(rows, group_ids or {}, reservoir_size, seed)
        centres = _prototypes(reservoir, radius, prototype_cap)
        prototypes[class_id] = centres
        sampled_counts[class_id] = len(reservoir)
        namespace = hashlib.sha256(class_id.encode()).hexdigest()
        labels = [f"geometry:{namespace}:{index}" for index in range(len(centres))]
        record_ids = [rows.names[record] for record in rows.records.tolist()]
        atoms = rows.atoms.tolist()
        for start, stop in _chunks(len(rows), centres.nbytes):
            block = rows.block(slice(start, stop))
            # Same difference-then-norm arithmetic as one environment at a time.
            distances = np.linalg.norm(centres[None, :, :] - block[:, None, :], axis=2)
            nearest = np.argmin(distances, axis=1)
            best = distances[np.arange(len(block)), nearest]
            for offset, (index, distance) in enumerate(
                zip(nearest.tolist(), best.tolist(), strict=True)
            ):
                record_id = record_ids[start + offset]
                if distance <= radius:
                    targets[record_id].add(labels[index])
                else:
                    overflow.append(
                        (record_id, atoms[start + offset], class_id, distance)
                    )
    return GeometryRegions(
        targets={rid: frozenset(values) for rid, values in targets.items()},
        prototypes=prototypes,
        overflow=tuple(overflow),
        sampled_counts=sampled_counts,
        radius=radius,
    )


def nearest_selected_distances(
    descriptors: Descriptors,
    selected_ids: Sequence[str],
    *,
    sample_size: int = 4096,
    seed: int = 0,
) -> dict[str, object]:
    """Audit a deterministic sample against compatible selected local descriptors.

    This measures actual selected environments independently of prototype region
    assignment. Missing selected classes are counted explicitly; finite distances
    are summarized separately without hiding missing witnesses in an average.
    """
    if not _positive_integer(sample_size):
        raise ConfigError("audit sample_size must be a positive integer")
    selected = set(selected_ids)
    if not selected <= set(_record_ids(descriptors)):
        raise ConfigError("selected audit IDs must belong to the descriptor pool")
    classes = _classes(descriptors)

    def environments() -> Iterator[tuple[str, str, int]]:
        for class_id, rows in classes.items():
            atoms = rows.atoms.tolist()
            for position, record in enumerate(rows.records.tolist()):
                key = _order(seed, class_id, rows.names[record], atoms[position])
                yield key, class_id, position

    # Keys are distinct digests, so this is the order of the keys alone.
    samples = [
        (class_id, position)
        for _, class_id, position in heapq.nsmallest(sample_size, environments())
    ]
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, (class_id, _) in enumerate(samples):
        by_class[class_id].append(index)
    nearest: list[float | None] = [None] * len(samples)
    # Every class indexes the same record names, so mark the selection once.
    names = next(iter(classes.values())).names if classes else ()
    chosen = np.fromiter((name in selected for name in names), dtype=bool)
    for class_id, rows in classes.items():
        indices = by_class.get(class_id, [])
        queries = (
            rows.block([samples[i][1] for i in indices])
            if indices
            else np.empty((0, 0))
        )
        witness = chosen[rows.records]
        chunks = _chunks(len(rows), 8 * rows.dimension)
        pending: list[npt.NDArray[np.float64]] = []
        buffered = 0
        for start, stop in chunks:
            # Every chunk is converted, which also checks every vector is finite.
            block = rows.block(slice(start, stop))[witness[start:stop]]
            if indices and len(block):
                pending.append(block)
                buffered += len(block)
            if pending and (buffered >= chunks[0][1] or stop == len(rows)):
                witnesses = np.concatenate(pending)
                pending, buffered = [], 0
                for index, query in zip(indices, queries, strict=True):
                    # Same difference-then-norm arithmetic as one witness at a time.
                    difference = query - witnesses
                    value = float(np.sqrt(np.vecdot(difference, difference)).min())
                    previous = nearest[index]
                    nearest[index] = value if previous is None else min(previous, value)
    distances = [value for value in nearest if value is not None]
    missing = len(samples) - len(distances)
    return {
        "sample_count": len(samples),
        "missing_class_count": missing,
        "mean": float(np.mean(distances)) if distances else None,
        "p95": float(np.quantile(distances, 0.95)) if distances else None,
        "maximum": max(distances) if distances else None,
    }
