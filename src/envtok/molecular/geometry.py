"""Bounded geometric prototypes and explicit unresolved environments.

These pilot utilities never form a configuration-by-configuration distance
matrix. Prototype selection retains at most the declared reservoir per class;
assignment and nearest-selected audits convert environments to float64 in
bounded chunks, with the same arithmetic as one environment at a time.
"""

import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
_CHUNK_BYTES = 1 << 21
"""Float64 working-block size: bounds memory and keeps blocks cache-resident."""


@dataclass(slots=True)
class _ClassRows:
    """One class's environments in record then atom order, unconverted."""

    record_ids: list[str] = field(default_factory=list)
    atom_indices: list[int] = field(default_factory=list)
    vectors: list[Any] = field(default_factory=list)


def _unit_rows(vectors: Sequence[Any]) -> npt.NDArray[np.float64]:
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


def _classes(descriptors: DescriptorRows) -> dict[str, _ClassRows]:
    """Group environments by class, checking shapes in record and atom order.

    Values are checked for finiteness when each class is converted; an earlier
    non-finite vector still takes precedence over a dimension mismatch.
    """
    classes: dict[str, _ClassRows] = {}
    dimensions: dict[str, int] = {}
    for record_id, atoms in sorted(descriptors.items()):
        for atom_index, (class_id, vector) in enumerate(atoms):
            shape = np.shape(vector)
            if len(shape) != 1 or not shape[0]:
                raise ConfigError(_NONFINITE)
            if dimensions.setdefault(class_id, shape[0]) != shape[0]:
                earlier = [v for rows in classes.values() for v in rows.vectors]
                if not all(np.isfinite(v).all() for v in [*earlier, vector]):
                    raise ConfigError(_NONFINITE)
                raise ConfigError(f"incompatible descriptor dimensions for {class_id}")
            rows = classes.setdefault(class_id, _ClassRows())
            rows.record_ids.append(record_id)
            rows.atom_indices.append(atom_index)
            rows.vectors.append(vector)
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
    buckets: dict[str, list[int]] = defaultdict(list)
    for position, record_id in enumerate(rows.record_ids):
        buckets[groups.get(record_id) or record_id].append(position)
    target = min(limit, len(rows.record_ids))
    sizes = sorted((len(bucket) for bucket in buckets.values()), reverse=True)
    depth = drawn = 0
    while drawn < target:
        depth += 1
        drawn += sum(1 for size in sizes if size >= depth)

    def rank(position: int) -> str:
        return _order(seed, rows.record_ids[position], rows.atom_indices[position])

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
    return _unit_rows([rows.vectors[position] for position in chosen])


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
    descriptors: DescriptorRows,
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
    targets: dict[str, set[str]] = {rid: set() for rid in sorted(descriptors)}
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
        for start, stop in _chunks(len(rows.vectors), centres.nbytes):
            block = _unit_rows(rows.vectors[start:stop])
            # Same difference-then-norm arithmetic as one environment at a time.
            distances = np.linalg.norm(centres[None, :, :] - block[:, None, :], axis=2)
            nearest = np.argmin(distances, axis=1)
            best = distances[np.arange(len(block)), nearest]
            for offset, (index, distance) in enumerate(
                zip(nearest.tolist(), best.tolist(), strict=True)
            ):
                record_id = rows.record_ids[start + offset]
                if distance <= radius:
                    targets[record_id].add(labels[index])
                else:
                    overflow.append(
                        (
                            record_id,
                            rows.atom_indices[start + offset],
                            class_id,
                            distance,
                        )
                    )
    return GeometryRegions(
        targets={rid: frozenset(values) for rid, values in targets.items()},
        prototypes=prototypes,
        overflow=tuple(overflow),
        sampled_counts=sampled_counts,
        radius=radius,
    )


def nearest_selected_distances(
    descriptors: DescriptorRows,
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
    if not selected <= descriptors.keys():
        raise ConfigError("selected audit IDs must belong to the descriptor pool")
    classes = _classes(descriptors)
    samples = heapq.nsmallest(
        sample_size,
        (
            (class_id, position)
            for class_id, rows in classes.items()
            for position in range(len(rows.record_ids))
        ),
        key=lambda item: _order(
            seed,
            item[0],
            classes[item[0]].record_ids[item[1]],
            classes[item[0]].atom_indices[item[1]],
        ),
    )
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, (class_id, _) in enumerate(samples):
        by_class[class_id].append(index)
    nearest: list[float | None] = [None] * len(samples)
    for class_id, rows in classes.items():
        indices = by_class.get(class_id, [])
        queries = (
            _unit_rows([rows.vectors[samples[i][1]] for i in indices])
            if indices
            else np.empty((0, 0))
        )
        witness = np.fromiter(
            (record_id in selected for record_id in rows.record_ids), dtype=bool
        )
        chunks = _chunks(len(rows.vectors), 8 * len(rows.vectors[0]))
        pending: list[npt.NDArray[np.float64]] = []
        buffered = 0
        for start, stop in chunks:
            # Every chunk is converted, which also checks every vector is finite.
            block = _unit_rows(rows.vectors[start:stop])[witness[start:stop]]
            if indices and len(block):
                pending.append(block)
                buffered += len(block)
            if pending and (buffered >= chunks[0][1] or stop == len(rows.vectors)):
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
