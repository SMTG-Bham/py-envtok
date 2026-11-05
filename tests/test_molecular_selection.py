from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from envtok.errors import ConfigError
from envtok.molecular.geometry import (
    DescriptorRows,
    build_geometry_regions,
    nearest_selected_distances,
)
from envtok.molecular.selection import (
    Candidate,
    Target,
    build_hierarchy,
    select_candidates,
)


def candidate(
    name: str,
    cost: float,
    targets: str = "",
    refinement: str = "",
    group: str | None = None,
) -> Candidate:
    return Candidate(name, cost, frozenset(targets), frozenset(refinement), group)


def vector(*values: float) -> npt.NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64)


def test_hierarchy_counts_records_and_families_without_erasing_state() -> None:
    hierarchy = build_hierarchy(
        {
            "a": [("neutral:fine1", "neutral:coarse")] * 10,
            "b": [("neutral:fine2", "neutral:coarse")],
            "c": [("charged:fine1", "charged:coarse")],
        },
        2,
        family_ids={"a": "same_family", "b": "same_family"},
    )
    assert hierarchy.mapping == {
        "charged:fine1": "charged:coarse",
        "neutral:fine1": "neutral:coarse",
        "neutral:fine2": "neutral:coarse",
    }
    assert hierarchy.support == {"neutral:coarse": 2, "charged:coarse": 1}
    assert hierarchy.group_support == {"neutral:coarse": 1, "charged:coarse": 1}
    assert hierarchy.rare_targets == {"charged:coarse"}
    assert not hierarchy.optional_targets
    assert hierarchy.targets["a"] == {"neutral:coarse"}
    assert hierarchy.lineage["neutral:fine1"] == ("neutral:fine1", "neutral:coarse")


def test_hierarchy_keeps_supported_fine_and_reports_optional_rare() -> None:
    hierarchy = build_hierarchy(
        {"a": [("fine", "coarse")], "b": [("fine", "coarse")], "c": [("rare",)]},
        2,
        rare_policy="optional",
    )
    assert hierarchy.mapping["fine"] == "fine"
    assert hierarchy.optional_targets == {"rare"}
    assert build_hierarchy({}, 1).mapping == {}


@pytest.mark.parametrize("support", [0, True, 1.5])
def test_hierarchy_rejects_invalid_support(support: object) -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        build_hierarchy({}, cast(int, support))


@pytest.mark.parametrize("paths", [[()], [("",)], [("a", "b"), ("a", "c")]])
def test_hierarchy_rejects_invalid_lineages(paths: Sequence[Sequence[str]]) -> None:
    with pytest.raises(ConfigError):
        build_hierarchy({"a": paths}, 1)


def test_hierarchy_rejects_unknown_rare_policy() -> None:
    with pytest.raises(ConfigError, match="rare_policy"):
        build_hierarchy({}, 1, rare_policy="discard")


def test_cover_greedy_pruning_and_deterministic_ties() -> None:
    candidates = [
        candidate("a", 1, "ab"),
        candidate("b", 2, "abc"),
        candidate("c", 5, "ab"),
    ]
    result = select_candidates(candidates)
    assert result.selected_ids == ("b",)
    assert result.achieved_cost == 2
    assert result.complete
    assert result.target_counts == {"a": 1, "b": 1, "c": 1}
    assert result.reasons == {"b": "chemical_core"}
    assert result == select_candidates(list(reversed(candidates)))
    assert select_candidates(candidates, prune=False).selected_ids == ("a", "b")
    assert select_candidates(
        [candidate("b", 1, "a"), candidate("a", 1, "a")]
    ).selected_ids == ("a",)


def test_budget_never_overspends_or_claims_infeasibility() -> None:
    result = select_candidates(
        [candidate("large", 5, "ab"), candidate("small", 2, "a", "x")],
        mode="budget",
        budget=3,
    )
    assert result.selected_ids == ("small",)
    assert result.achieved_cost == 2
    assert result.unmet == {"b": 1}
    assert not result.complete
    assert result.refinement_covered == {"x"}
    empty = select_candidates([candidate("a", 1, "a")], mode="budget", budget=0)
    assert not empty.selected_ids
    assert empty.unmet == {"a": 1}


def test_multiplicity_counts_configurations_and_missing_targets() -> None:
    result = select_candidates(
        [candidate("a", 1, "aaa"), candidate("b", 2, "a")],
        {"a": Target(2), "missing": Target()},
    )
    assert result.selected_ids == ("a", "b")
    assert result.target_counts == {"a": 2, "missing": 0}
    assert result.unmet == {"missing": 1}
    complete = select_candidates(
        [candidate("a", 1, "a"), candidate("b", 2, "a")], {"a": Target(2)}
    )
    assert complete.complete
    assert len(complete.selected_ids) == 2


def test_weighted_targets_and_fractional_costs() -> None:
    result = select_candidates(
        [candidate("a", 0.1, "a"), candidate("b", 0.2, "b")],
        {"a": Target(weight=1), "b": Target(weight=10)},
        mode="budget",
        budget=0.3,
    )
    assert result.achieved_cost == 0.3
    assert result.complete
    weighted = select_candidates(
        [candidate("a", 1, "a"), candidate("b", 1, "b")],
        {"a": Target(weight=1), "b": Target(weight=10)},
        mode="budget",
        budget=1,
    )
    assert weighted.selected_ids == ("b",)


def test_bundles_are_atomic_transitive_and_respect_remaining_budget() -> None:
    candidates = [
        candidate("a", 1, "a"),
        candidate("b", 2, "b"),
        candidate("c", 1, "c"),
    ]
    bundles = [frozenset({"a", "b"}), frozenset({"b", "c"})]
    result = select_candidates(candidates, bundles=bundles)
    assert result.selected_ids == ("a", "b", "c")
    assert result.achieved_cost == 4
    too_small = select_candidates(candidates, bundles=bundles, mode="budget", budget=3)
    assert too_small.selected_ids == ()
    assert too_small.unmet == {"a": 1, "b": 1, "c": 1}
    capped = select_candidates(candidates, {"a": Target()}, bundles=bundles)
    assert capped.achieved_cost == 4


@pytest.mark.parametrize("bundle", [frozenset(), frozenset({"missing"})])
def test_invalid_bundle(bundle: frozenset[str]) -> None:
    with pytest.raises(ConfigError, match="bundle"):
        select_candidates([candidate("a", 1)], bundles=[bundle])


def test_geometry_updates_every_target_and_survives_pruning() -> None:
    candidates = [
        candidate("core", 1, "a", "x"),
        candidate("conformer", 2, "a", "yz"),
        candidate("single", 2, "a", "y"),
    ]
    result = select_candidates(candidates, mode="budget", budget=3)
    assert result.selected_ids == ("conformer", "core")
    assert result.refinement_covered == set("xyz")
    assert result.reasons["conformer"] == "geometry"
    assert select_candidates(candidates).selected_ids == result.selected_ids
    only_geometry = select_candidates(candidates, {}, refinement_weights={"y": 10.0})
    assert only_geometry.refinement_covered == set("xyz")


def test_background_reserve_balances_groups_and_stays_inside_budget() -> None:
    candidates = [
        candidate("core", 1, "a", "x", "g1"),
        candidate("conformer", 1, "a", "y", "g1"),
        candidate("background", 1, "a", "y", "g2"),
        candidate("large", 100, "a", "z", "g3"),
    ]
    result = select_candidates(
        candidates, mode="budget", budget=3, background_reserve_fraction=1 / 3
    )
    assert result.complete
    assert result.achieved_cost == 3
    assert result.background_requested_cost == pytest.approx(1)
    assert result.background_achieved_cost == 1
    assert "background" in result.reasons.values()
    assert result == select_candidates(
        list(reversed(candidates)),
        mode="budget",
        budget=3,
        background_reserve_fraction=1 / 3,
    )
    exhausted = select_candidates(
        [candidate("core", 3, "a")],
        mode="budget",
        budget=3,
        background_reserve_fraction=0.5,
    )
    assert exhausted.background_requested_cost == 1.5
    assert exhausted.background_achieved_cost == 0


@pytest.mark.parametrize(
    ("mode", "budget", "reserve"),
    [
        ("unknown", None, 0),
        ("budget", None, 0),
        ("budget", float("inf"), 0),
        ("budget", -1, 0),
        ("cover", 1, 0),
        ("cover", None, -0.1),
        ("cover", None, float("nan")),
        ("cover", None, 1.1),
        ("cover", None, 0.1),
    ],
)
def test_invalid_modes_and_budgets(
    mode: str, budget: float | None, reserve: float
) -> None:
    with pytest.raises(ConfigError):
        select_candidates(
            [], mode=mode, budget=budget, background_reserve_fraction=reserve
        )


@pytest.mark.parametrize(
    "candidates",
    [
        [candidate("same", 1), candidate("same", 1)],
        [candidate("", 1)],
        [candidate("a", float("inf"))],
        [candidate("a", 0)],
    ],
)
def test_invalid_candidates(candidates: list[Candidate]) -> None:
    with pytest.raises(ConfigError):
        select_candidates(candidates)


@pytest.mark.parametrize(
    "target",
    [
        Target(0),
        Target(True),
        Target(weight=0),
        Target(weight=float("nan")),
        Target(cast(int, 1.5)),
    ],
)
def test_invalid_targets(target: Target) -> None:
    with pytest.raises(ConfigError):
        select_candidates([], {"a": target})


@pytest.mark.parametrize("weight", [0, float("nan")])
def test_invalid_refinement_weights(weight: float) -> None:
    with pytest.raises(ConfigError, match="refinement"):
        select_candidates(
            [candidate("a", 1, refinement="x")], refinement_weights={"x": weight}
        )


def test_empty_selection() -> None:
    result = select_candidates([])
    assert result.complete
    assert result.achieved_cost == 0
    assert result.selected_ids == ()


def test_geometry_radius_overflow_and_normalization() -> None:
    descriptors: DescriptorRows = {
        "a": [("neutral", vector(1, 0)), ("neutral", vector(5, 0))],
        "b": [("neutral", vector(0, 1))],
        "c": [("charged", vector(1, 0))],
        "zero": [("neutral", vector(0, 0))],
        "empty": [],
    }
    regions = build_geometry_regions(descriptors, radius=0.01, prototype_cap=1)
    assert len(regions.overflow) >= 2
    assert regions.targets["empty"] == frozenset()
    assert regions.targets["c"]
    covered = [rid for rid in ("a", "b", "zero") if regions.targets[rid]]
    assert len(covered) == 1
    complete = build_geometry_regions(descriptors, radius=0.01, prototype_cap=10)
    assert not complete.overflow
    assert len(complete.prototypes["neutral"]) == 3
    assert complete.targets["a"] != complete.targets["c"]
    reverse = build_geometry_regions(
        dict(reversed(list(descriptors.items()))), radius=0.01, prototype_cap=10
    )
    assert reverse.targets == complete.targets
    assert reverse.sampled_counts == complete.sampled_counts
    repeated = build_geometry_regions({"a": [("x", vector(1, 0))] * 4}, radius=0.2)
    assert len(repeated.prototypes["x"]) == 1
    assert not build_geometry_regions({}).prototypes


def test_geometry_reservoir_is_bounded_and_balances_groups() -> None:
    descriptors: DescriptorRows = {
        "many": [("x", vector(1, 0))] * 20,
        "rare": [("x", vector(0, 1))],
    }
    result = build_geometry_regions(
        descriptors,
        group_ids={"many": "trajectory", "rare": "rare"},
        reservoir_size=2,
        prototype_cap=2,
    )
    assert result.sampled_counts == {"x": 2}
    assert not result.overflow
    assert len(result.prototypes["x"]) == 2
    deeper = build_geometry_regions(descriptors, reservoir_size=10, prototype_cap=2)
    assert deeper.sampled_counts == {"x": 10}


@pytest.mark.parametrize("radius", [-1, float("nan")])
def test_invalid_geometry_radius(radius: float) -> None:
    with pytest.raises(ConfigError, match="radius"):
        build_geometry_regions({}, radius=radius)


@pytest.mark.parametrize("value", [0, True, 1.5])
def test_invalid_geometry_caps(value: object) -> None:
    with pytest.raises(ConfigError, match="positive integer"):
        build_geometry_regions({}, prototype_cap=cast(int, value))
    with pytest.raises(ConfigError, match="positive integer"):
        build_geometry_regions({}, reservoir_size=cast(int, value))


@pytest.mark.parametrize("array", [vector(), vector(float("nan")), np.zeros((2, 2))])
def test_invalid_descriptor_arrays(array: npt.NDArray[np.float64]) -> None:
    with pytest.raises(ConfigError, match="finite vectors"):
        build_geometry_regions({"a": [("x", array)]})


def test_incompatible_dimensions() -> None:
    with pytest.raises(ConfigError, match="dimensions"):
        build_geometry_regions({"a": [("x", vector(1)), ("x", vector(1, 2))]})


def test_nearest_selected_audit_is_separate_from_regions() -> None:
    descriptors: DescriptorRows = {
        "a": [("x", vector(1, 0))],
        "b": [("x", vector(0, 1))],
        "c": [("other", vector(1, 0))],
    }
    result = nearest_selected_distances(descriptors, ["a"])
    assert result["sample_count"] == 3
    assert result["missing_class_count"] == 1
    assert result["maximum"] == pytest.approx(np.sqrt(2))
    assert result["mean"] == pytest.approx(np.sqrt(2) / 2)
    assert (
        nearest_selected_distances(descriptors, [], sample_size=1)["sample_count"] == 1
    )
    assert nearest_selected_distances(descriptors, [])["mean"] is None
    assert nearest_selected_distances({}, [])["sample_count"] == 0
    with pytest.raises(ConfigError, match="IDs"):
        nearest_selected_distances(descriptors, ["missing"])
    with pytest.raises(ConfigError, match="sample_size"):
        nearest_selected_distances(descriptors, ["a"], sample_size=0)


def test_hierarchy_distinguishes_assigned_support_and_source_support() -> None:
    result = build_hierarchy(
        {
            "a": [("common", "parent")],
            "b": [("common", "parent")],
            "c": [("rare", "parent")],
        },
        2,
    )
    assert result.source_support["parent"] == 3
    assert result.support["parent"] == 1
    assert result.group_support["parent"] == 1
    assert result.rare_targets == {"parent"}
