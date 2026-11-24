"""Chunked geometry must reproduce the frozen per-environment reference exactly."""

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
