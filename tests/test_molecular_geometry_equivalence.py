"""Chunked geometry must reproduce the frozen per-environment reference exactly."""

from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import reference_molecular_geometry as reference

from envtok.errors import ConfigError
from envtok.molecular import geometry


def random_descriptors(
    seed: int,
) -> tuple[dict[str, list[tuple[str, Any]]], dict[str, str | None]]:
    rng = np.random.default_rng(seed)
    dimensions = {
        f"c{i}": int(rng.choice([1, 2, 3, 7, 17, 64, 680]))
        for i in range(int(rng.integers(1, 5)))
    }
    classes = list(dimensions)
    rows: dict[str, list[tuple[str, Any]]] = {}
    groups: dict[str, str | None] = {}
    for index in range(int(rng.integers(1, 30))):
        record_id = f"r{int(rng.integers(0, 10**6)):06d}-{index}"
        atoms: list[tuple[str, Any]] = []
        for _ in range(int(rng.integers(0, 8))):
            class_id = classes[int(rng.integers(0, len(classes)))]
            size = dimensions[class_id]
            draw = rng.random()
            if draw < 0.05:
                vector = np.zeros(size, dtype=np.float32)
            elif draw < 0.1 and atoms and atoms[-1][0] == class_id:
                vector = atoms[-1][1].copy()
            else:
                vector = rng.normal(size=size).astype(
                    np.float32 if rng.random() < 0.7 else np.float64
                )
            atoms.append((class_id, vector))
        rows[record_id] = atoms
        groups[record_id] = None if rng.random() < 0.3 else f"g{rng.integers(0, 4)}"
    return rows, groups


def assert_regions_equal(current: Any, frozen: Any) -> None:
    assert current.targets == frozen.targets
    assert current.overflow == frozen.overflow
    assert current.sampled_counts == frozen.sampled_counts
    assert current.radius == frozen.radius
    assert current.prototypes.keys() == frozen.prototypes.keys()
    for key, value in frozen.prototypes.items():
        assert current.prototypes[key].dtype == value.dtype
        assert np.array_equal(current.prototypes[key], value)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 8, 9, 16, 17, 31, 64, 127, 680, 681])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unit_rows_match_per_vector_normalisation(
    size: int, dtype: type[np.floating[Any]]
) -> None:
    rng = np.random.default_rng(size)
    vectors: list[npt.NDArray[Any]] = [
        rng.normal(scale=scale, size=size).astype(dtype)
        for scale in (1e-3, 1.0, 1e3, 1e-160)
        for _ in range(50)
    ]
    vectors += [np.zeros(size, dtype=dtype), np.full(size, 1e-200)]
    expected = np.stack([reference._unit(vector) for vector in vectors])
    assert np.array_equal(geometry._unit_rows(vectors), expected)


@pytest.mark.parametrize("chunk_bytes", [None, 64])
@pytest.mark.parametrize("block", range(6))
def test_regions_and_audit_match_frozen_reference(
    block: int, chunk_bytes: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if chunk_bytes is not None:
        monkeypatch.setattr(geometry, "_CHUNK_BYTES", chunk_bytes)
    for seed in range(block * 25, (block + 1) * 25):
        rows, groups = random_descriptors(seed)
        rng = np.random.default_rng(seed)
        options: dict[str, Any] = {
            "group_ids": groups if rng.random() < 0.7 else None,
            "radius": float(rng.choice([0.0, 0.05, 0.3, 1.0, 2.0])),
            "prototype_cap": int(rng.integers(1, 7)),
            "reservoir_size": int(rng.integers(1, 20)),
            "seed": int(rng.integers(0, 3)),
        }
        assert_regions_equal(
            geometry.build_geometry_regions(rows, **options),
            reference.build_geometry_regions(rows, **options),
        )
        chosen = [rid for rid in rows if rng.random() < 0.3]
        sample_size = int(rng.choice([1, 5, 4096]))
        assert geometry.nearest_selected_distances(
            rows, chosen, sample_size=sample_size, seed=options["seed"]
        ) == reference.nearest_selected_distances(
            rows, chosen, sample_size=sample_size, seed=options["seed"]
        )


def vector(*values: float) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({"a": [("x", vector(float("nan"), 0)), ("x", vector(1))]}, "finite"),
        ({"a": [("x", vector(1, 0)), ("x", vector(float("inf")))]}, "finite"),
        ({"a": [("x", vector(1, 0)), ("x", vector(1))]}, "dimensions"),
        (
            {"a": [("x", vector(1, 0)), ("x", vector(1)), ("y", vector(np.nan))]},
            "dimensions",
        ),
        ({"a": [("x", vector(1, 0))], "b": [("y", vector(np.nan, 1))]}, "finite"),
    ],
)
def test_errors_match_frozen_reference(
    rows: dict[str, list[tuple[str, Any]]], message: str
) -> None:
    for module in (geometry, reference):
        with pytest.raises(ConfigError, match=message):
            module.build_geometry_regions(rows)
        with pytest.raises(ConfigError, match=message):
            module.nearest_selected_distances(rows, [], sample_size=1)


def as_pool(
    rows: dict[str, list[tuple[str, Any]]], dimension: int, seed: int
) -> geometry.DescriptorPool:
    """The same environments as arrays, with records in a shuffled order."""
    order = list(rows)
    np.random.default_rng(seed).shuffle(order)
    class_ids = sorted({class_id for atoms in rows.values() for class_id, _ in atoms})
    index = {class_id: position for position, class_id in enumerate(class_ids)}
    atoms = [atom for record_id in order for atom in rows[record_id]]
    return geometry.DescriptorPool(
        record_ids=order,
        offsets=np.cumsum([0, *(len(rows[record_id]) for record_id in order)]),
        class_ids=class_ids,
        classes=np.asarray([index[class_id] for class_id, _ in atoms], dtype=np.int32),
        vectors=np.stack([vector for _, vector in atoms])
        if atoms
        else np.zeros((0, dimension), dtype=np.float32),
    )


@pytest.mark.parametrize("chunk_bytes", [None, 64])
def test_descriptor_pool_matches_the_mapping(
    chunk_bytes: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if chunk_bytes is not None:
        monkeypatch.setattr(geometry, "_CHUNK_BYTES", chunk_bytes)
    for seed in range(60):
        rng = np.random.default_rng(seed)
        dimension = int(rng.choice([1, 3, 17, 680]))
        rows: dict[str, list[tuple[str, Any]]] = {}
        groups: dict[str, str | None] = {}
        for index in range(int(rng.integers(0, 25))):
            record_id = f"r{int(rng.integers(0, 10**6)):06d}-{index}"
            rows[record_id] = [
                (
                    f"c{int(rng.integers(0, 4))}",
                    np.zeros(dimension, np.float32)
                    if rng.random() < 0.05
                    else rng.normal(size=dimension).astype(np.float32),
                )
                for _ in range(int(rng.integers(0, 8)))
            ]
            groups[record_id] = None if rng.random() < 0.3 else f"g{index % 3}"
        pool = as_pool(rows, dimension, seed)
        options: dict[str, Any] = {
            "group_ids": groups,
            "radius": float(rng.choice([0.0, 0.3, 1.0])),
            "prototype_cap": int(rng.integers(1, 5)),
            "reservoir_size": int(rng.integers(1, 12)),
            "seed": seed % 3,
        }
        assert_regions_equal(
            geometry.build_geometry_regions(pool, **options),
            geometry.build_geometry_regions(rows, **options),
        )
        chosen = [rid for rid in rows if rng.random() < 0.3]
        assert geometry.nearest_selected_distances(
            pool, chosen, sample_size=7, seed=seed
        ) == geometry.nearest_selected_distances(rows, chosen, sample_size=7, seed=seed)


def test_pool_regions_report_python_atom_indices() -> None:
    rows = {"a": [("x", vector(1, 0)), ("x", vector(0, 1))]}
    regions = geometry.build_geometry_regions(
        as_pool(rows, 2, 0), radius=0.1, prototype_cap=1
    )
    assert [type(row[1]) for row in regions.overflow] == [int]


def pool_with(**changes: Any) -> geometry.DescriptorPool:
    base = geometry.DescriptorPool(
        record_ids=["a", "b"],
        offsets=np.asarray([0, 1, 2]),
        class_ids=["x"],
        classes=np.asarray([0, 0]),
        vectors=np.ones((2, 3), dtype=np.float32),
    )
    return replace(base, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"offsets": np.asarray([0, 2])},
        {"offsets": np.asarray([1, 1, 2])},
        {"offsets": np.asarray([0, 1, 3])},
        {"offsets": np.asarray([0, 2, 1])},
        {"vectors": np.ones(2, dtype=np.float32)},
        {"vectors": np.ones((3, 3), dtype=np.float32)},
        {"vectors": np.ones((2, 0), dtype=np.float32)},
        {"classes": np.asarray([0, 1])},
        {"classes": np.asarray([-1, 0])},
        {"record_ids": ["a", "a"]},
    ],
)
def test_inconsistent_pools_are_rejected(changes: dict[str, Any]) -> None:
    pool = pool_with(**changes)
    with pytest.raises(ConfigError, match="inconsistent"):
        geometry.build_geometry_regions(pool)
    with pytest.raises(ConfigError, match="inconsistent"):
        geometry.nearest_selected_distances(pool, [])


def test_nonfinite_pool_vectors_are_rejected() -> None:
    pool = pool_with(vectors=np.asarray([[1, 0, 0], [np.nan, 0, 0]], np.float32))
    with pytest.raises(ConfigError, match="finite"):
        geometry.build_geometry_regions(pool)
    with pytest.raises(ConfigError, match="finite"):
        geometry.nearest_selected_distances(pool, ["a"])
