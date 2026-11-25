"""Batched ChemEnv symmetry measures must be bit-for-bit pymatgen's per-call code."""

import inspect
import math
import random
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from pymatgen.analysis.chemenv.coordination_environments import (
    coordination_geometries,
)
from pymatgen.analysis.chemenv.coordination_environments import (
    coordination_geometry_finder as finder_module,
)
from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import Plane

from envtok.profiles import REFERENCE_723696D
from envtok.tokenizer import chemenv
from envtok.tokenizer.chemenv import BatchedLocalGeometryFinder
from envtok.tokenizer.geometry import GeometryEngine
from envtok.tokenizer.pipeline import geometry_structure

SEPARATION_PLANE = "SEPARATION_PLANE"
SAFE_LIMIT = 800


def same(left: Any, right: Any) -> bool:
    """Equal in type, structure and every bit, with NaN equal to NaN."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and list(left) == list(right)
            and all(same(left[key], right[key]) for key in left)
        )
    if isinstance(left, list | tuple):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(same(a, b) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return bool(left == right)


@pytest.fixture(autouse=True)
def fresh_support_check() -> Iterator[None]:
    chemenv.batched_supported.cache_clear()
    yield
    chemenv.batched_supported.cache_clear()


@pytest.mark.parametrize("seed", range(4))
def test_symmetry_measures_match_pymatgen_bitwise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(150):
        rows = int(rng.integers(2, 22))
        count = int(rng.integers(1, 40))
        perfect = rng.normal(size=(rows, 3))
        noise = float(rng.choice([1e-4, 0.05, 0.5]))
        base = perfect * rng.uniform(0.5, 2.0) + rng.normal(scale=noise, size=(rows, 3))
        stacked = np.stack([base[rng.permutation(rows)] for _ in range(count)])
        # Offset the stack in memory so orderings start at varied alignments.
        offset = int(rng.integers(0, 7))
        buffer = np.empty(offset + stacked.size)
        distorted = buffer[offset:].reshape(stacked.shape)
        distorted[...] = stacked
        measures, scales, rotations = chemenv.symmetry_measures(distorted, perfect)
        for index in range(count):
            want = finder_module.symmetry_measure(
                np.ascontiguousarray(distorted[index]), perfect
            )
            assert same(measures[index], want["symmetry_measure"])
            assert same(scales[index], want["scaling_factor"])
            assert same(rotations[index], want["rotation_matrix"])


def test_symmetry_measure_of_a_point_like_polyhedron_is_zero() -> None:
    distorted = np.random.default_rng(0).normal(size=(3, 4, 3))
    perfect = np.zeros((4, 3))
    measures, _, _ = chemenv.symmetry_measures(distorted, perfect)
    want = [finder_module.symmetry_measure(d, perfect) for d in distorted]
    assert measures == [w["symmetry_measure"] for w in want] == [0.0, 0.0, 0.0]


def plane_cases() -> Iterator[tuple[str, Any]]:
    for geometry in (
        coordination_geometries.AllCoordinationGeometries().get_implemented_geometries()
    ):
        for algorithm in geometry.algorithms:
            if algorithm.algorithm_type == SEPARATION_PLANE:
                yield geometry.mp_symbol, algorithm


def finders(safe: bool) -> tuple[Any, Any]:
    options = {
        "permutations_safe_override": False,
        "plane_ordering_override": True,
        "plane_safe_permutations": safe,
    }
    return finder_module.LocalGeometryFinder(**options), BatchedLocalGeometryFinder(
        **options
    )


def separation(
    rng: np.random.Generator, algorithm: Any, coordination: int
) -> list[np.ndarray]:
    order = rng.permutation(coordination)
    sizes = np.cumsum(algorithm.separation)[:-1]
    return [part.astype(np.int64) for part in np.split(order, sizes)]


@pytest.mark.parametrize("safe", [False, True])
def test_every_separation_plane_algorithm_matches_pymatgen(safe: bool) -> None:
    rng = np.random.default_rng(7)
    random.seed(7)
    np.random.seed(7)  # noqa: NPY002 - pymatgen's random environments use it
    stock, batched = finders(safe)
    checked = 0
    for symbol, algorithm in plane_cases():
        if (
            safe
            and len(
                algorithm.safe_separation_permutations(
                    ordered_plane=algorithm.ordered_plane,
                    ordered_point_groups=algorithm.ordered_point_groups,
                )
            )
            > SAFE_LIMIT
        ):
            continue
        stock.setup_test_perfect_environment(
            symbol,
            randomness=True,
            random_translation="RANDOM",
            random_rotation="RANDOM",
            random_scale="RANDOM",
        )
        batched.local_geometry = stock.local_geometry
        geometry = stock.allcg.get_geometry_from_mp_symbol(symbol)
        perfect = finder_module.AbstractGeometry.from_cg(
            geometry, centering_type="centroid", include_central_site_in_centroid=True
        ).points_wcs_ctwcc()
        coords = stock.local_geometry.coords
        for _ in range(2):
            indices = separation(rng, algorithm, len(coords))
            first, second = rng.choice(len(coords), size=2, replace=False)
            plane = Plane.from_3points(
                coords[first], coords[second], stock.local_geometry.central_site
            )
            for _ in range(2):  # The second call reads the cached orderings.
                arguments = (geometry, algorithm, plane)
                want = (
                    finder_module.LocalGeometryFinder._cg_csm_separation_plane_optim2(
                        stock,
                        *arguments,
                        points_perfect=perfect,
                        separation_indices=[part.copy() for part in indices],
                    )
                )
                got = batched._cg_csm_separation_plane_optim2(
                    *arguments,
                    points_perfect=perfect,
                    separation_indices=[part.copy() for part in indices],
                )
                assert same(list(want), list(got)), (symbol, algorithm.separation)
                checked += 1
    assert checked > (100 if safe else 300)


def test_a_plane_without_orderings_scores_nothing() -> None:
    _, batched = finders(safe=False)
    batched.setup_test_perfect_environment("O:6")
    plane = SimpleNamespace(
        argsorted_ref_separation_perm=[0, 1, 2, 3, 4, 5],
        ordered_plane=False,
        permutations=[],
        algorithm_type=SEPARATION_PLANE,
    )
    indices = [np.arange(1), np.arange(1, 5), np.arange(5, 6)]
    assert batched._cg_csm_separation_plane_optim2(
        None, plane, None, points_perfect=None, separation_indices=indices
    ) == ([], [], [], False)


def test_structure_environments_match_pymatgen(
    corpus_frames: list[Atoms], monkeypatch: pytest.MonkeyPatch
) -> None:
    # pymatgen's random fallback draws from an unseeded default_rng(); seed it
    # the same way for both finders so only the batched search can differ.
    default_rng = np.random.default_rng
    monkeypatch.setattr(np.random, "default_rng", lambda *_: default_rng(0))
    settings = REFERENCE_723696D.geometry
    options = {
        "permutations_safe_override": settings.permutations_safe_override,
        "plane_ordering_override": settings.plane_ordering_override,
        "plane_safe_permutations": settings.plane_safe_permutations,
    }
    for atoms in corpus_frames[:6]:
        structure = geometry_structure(atoms, np.zeros(len(atoms)))
        results = []
        for finder in (
            finder_module.LocalGeometryFinder(**options),
            BatchedLocalGeometryFinder(**options),
        ):
            finder.setup_parameters(
                centering_type=settings.centering_type,
                include_central_site_in_centroid=settings.include_central_site_in_centroid,
                structure_refinement=settings.structure_refinement,
            )
            finder.setup_structure(structure)
            environments = finder.compute_structure_environments(
                only_cations=False,
                maximum_distance_factor=settings.maximum_distance_factor,
                minimum_angle_factor=settings.minimum_angle_factor,
                optimization=settings.optimization,
            )
            results.append(
                [
                    {
                        cn: [env and env.coord_geoms for env in envs]
                        for cn, envs in site.items()
                    }
                    if site
                    else site
                    for site in environments.ce_list
                ]
            )
        assert same(results[0], results[1])


def test_engine_uses_the_batched_finder() -> None:
    engine = GeometryEngine(REFERENCE_723696D.geometry, REFERENCE_723696D.measures)
    assert chemenv.batched_supported()
    assert type(engine.finder) is BatchedLocalGeometryFinder


def test_changed_pymatgen_code_falls_back_to_the_stock_finder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Any = finder_module.symmetry_measure

    def changed(*args: Any) -> Any:
        return original(*args)

    monkeypatch.setattr(finder_module, "symmetry_measure", changed)
    assert not chemenv.batched_supported()
    assert type(chemenv.make_finder()) is finder_module.LocalGeometryFinder


def test_a_different_svd_falls_back_to_the_stock_finder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def other_svd(matrix: Any) -> Any:
        return np.linalg.svd(matrix)

    monkeypatch.setattr(finder_module, "svd", other_svd)
    assert not chemenv.batched_supported()


def test_unreadable_source_falls_back_to_the_stock_finder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_target: Any) -> str:
        raise OSError("source not available")

    monkeypatch.setattr(inspect, "getsource", unavailable)
    assert not chemenv.batched_supported()


def test_a_blas_that_sums_differently_falls_back_to_the_stock_finder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    measures = chemenv.symmetry_measures

    def drifting(distorted: Any, perfect: Any) -> Any:
        values, scales, rotations = measures(distorted, perfect)
        return values, np.nextafter(scales, np.inf), rotations

    monkeypatch.setattr(chemenv, "symmetry_measures", drifting)
    assert not chemenv.batched_supported()
