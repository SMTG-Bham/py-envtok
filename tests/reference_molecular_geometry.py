# Frozen copy of src/envtok/molecular/geometry.py at 6064532: a test oracle
# for the optimised code. Do not edit, except to repoint internal imports.
"""Bounded geometric prototypes and explicit unresolved environments.

These pilot utilities never form a configuration-by-configuration distance
matrix. Prototype selection retains at most the declared reservoir per class;
assignment and nearest-selected audits process one environment at a time.
"""

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from reference_molecular_selection import _positive_integer

from envtok.errors import ConfigError

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


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ConfigError("local descriptors must be nonempty finite vectors")
    norm = float(np.linalg.norm(array))
    return array / norm if norm else array.copy()


def _order(seed: int, *parts: object) -> str:
    return hashlib.sha256(repr((seed, *parts)).encode()).hexdigest()


def _rows(descriptors: DescriptorRows) -> dict[str, list[EnvironmentRef]]:
    rows: dict[str, list[EnvironmentRef]] = defaultdict(list)
    dimensions: dict[str, int] = {}
    for record_id, atoms in sorted(descriptors.items()):
        for atom_index, (class_id, vector) in enumerate(atoms):
            normalized = _unit(vector)
            if class_id in dimensions and dimensions[class_id] != normalized.size:
                raise ConfigError(f"incompatible descriptor dimensions for {class_id}")
            dimensions[class_id] = normalized.size
            rows[class_id].append((record_id, atom_index, normalized))
    return dict(rows)


def _reservoir(
    rows: Sequence[EnvironmentRef],
    groups: Mapping[str, str | None],
    limit: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    buckets: dict[str, list[EnvironmentRef]] = defaultdict(list)
    for row in rows:
        buckets[groups.get(row[0]) or row[0]].append(row)
    for group, bucket in buckets.items():
        buckets[group] = sorted(bucket, key=lambda row: _order(seed, row[0], row[1]))
    ordered_groups = sorted(buckets, key=lambda group: _order(seed, group))
    chosen: list[npt.NDArray[np.float64]] = []
    depth = 0
    while len(chosen) < min(limit, len(rows)):
        for group in ordered_groups:
            if depth < len(buckets[group]):
                chosen.append(buckets[group][depth][2])
                if len(chosen) == limit:
                    return np.stack(chosen)
        depth += 1
    return np.stack(chosen)


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
    rows = _rows(descriptors)
    targets: dict[str, set[str]] = {rid: set() for rid in sorted(descriptors)}
    prototypes: dict[str, npt.NDArray[np.float64]] = {}
    overflow: list[tuple[str, int, str, float]] = []
    sampled_counts: dict[str, int] = {}
    for class_id, environments in sorted(rows.items()):
        reservoir = _reservoir(environments, group_ids or {}, reservoir_size, seed)
        centres = _prototypes(reservoir, radius, prototype_cap)
        prototypes[class_id] = centres
        sampled_counts[class_id] = len(reservoir)
        namespace = hashlib.sha256(class_id.encode()).hexdigest()
        for record_id, atom_index, vector in environments:
            distances = np.linalg.norm(centres - vector, axis=1)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            if distance <= radius:
                targets[record_id].add(f"geometry:{namespace}:{nearest}")
            else:
                overflow.append((record_id, atom_index, class_id, distance))
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
    rows = _rows(descriptors)
    samples = sorted(
        (
            (class_id, *row)
            for class_id, environments in rows.items()
            for row in environments
        ),
        key=lambda row: _order(seed, row[0], row[1], row[2]),
    )[:sample_size]
    distances: list[float] = []
    missing = 0
    witnesses = {
        class_id: [
            vector for record_id, _, vector in environments if record_id in selected
        ]
        for class_id, environments in rows.items()
    }
    for class_id, _, _, vector in samples:
        if not witnesses[class_id]:
            missing += 1
            continue
        distances.append(
            min(float(np.linalg.norm(vector - other)) for other in witnesses[class_id])
        )
    return {
        "sample_count": len(samples),
        "missing_class_count": missing,
        "mean": float(np.mean(distances)) if distances else None,
        "p95": float(np.quantile(distances, 0.95)) if distances else None,
        "maximum": max(distances) if distances else None,
    }
