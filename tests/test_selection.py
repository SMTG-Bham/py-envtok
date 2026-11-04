"""Cover selection: reference agreement, exactness, invariants and settings."""

import itertools
from typing import Any

import numpy as np
import pytest
from helpers import random_pool
from reference_oracles import reference_selection

from envtok.coarsen import CoverageBuilder, CoverageClasses, Pool
from envtok.errors import ConfigError, ResourcePolicyError
from envtok.execution import ExecutionSettings
from envtok.selection import (
    Selected,
    SelectionResult,
    Selector,
    class_incidence,
    combine_strata,
    presence_incidence,
    random_subset,
    strata_values,
    tie_ranks,
)
from envtok.selection import greedy as greedy_module
from envtok.selection.greedy import float_keys_exact, greedy_lazy, greedy_transparent
from envtok.selection.incidence import Incidence, check_budget, estimate_selection_bytes
from envtok.selection.steps import prune, repair, sole_suppliers


def classes_for(seed: int, threshold: int = 3, **pool: Any) -> CoverageClasses:
    return CoverageBuilder(min_structure_support=threshold).fit_pool(
        random_pool(seed, **pool)
    )


def dense_matrix(classes: CoverageClasses) -> np.ndarray:
    pool = classes.pool
    matrix = np.zeros((len(pool.record_ids), classes.n_classes), dtype=np.float32)
    position = np.searchsorted(pool.token_ids, pool.edge_token)
    np.add.at(
        matrix,
        (pool.edge_record, classes.token_class[position]),
        pool.edge_multiplicity.astype(np.float32),
    )
    return matrix


def material_signatures(classes: CoverageClasses) -> list[dict[str, set[str]]]:
    pool = classes.pool
    position = np.searchsorted(pool.token_ids, pool.edge_token)
    result: list[dict[str, set[str]]] = [
        {f: set() for f in classes.presence_signatures} for _ in pool.record_ids
    ]
    for record, token in zip(pool.edge_record.tolist(), position.tolist(), strict=True):
        for family, index in classes.token_presence.items():
            result[record][family].add(
                classes.presence_signatures[family][index[token]]
            )
    return result


@pytest.mark.parametrize(
    ("seed", "threshold"), [(0, 2), (1, 3), (2, 4), (3, 3), (6, 2), (8, 5)]
)
def test_selection_matches_the_reference_procedure(seed: int, threshold: int) -> None:
    classes = classes_for(seed, threshold, n_records=60, vocabulary=80)
    reference = reference_selection(
        dense_matrix(classes),
        material_signatures(classes),
        [f.name for f in classes.presence_families],
    )
    unpruned = Selector(prune=False, presence_constraints="none").select(classes)
    assert [s.record_index for s in unpruned.selected] == reference[
        "forced"
    ] + reference["greedy"]
    assert {s.reason for s in unpruned.selected} <= {"mandatory", "cover"}
    full = Selector().select(classes)
    assert [s.record_index for s in full.selected] == reference["kept"] + reference[
        "topup"
    ]
    pruned = set(reference["forced"] + reference["greedy"]) - set(reference["kept"])
    index = {rid: i for i, rid in enumerate(classes.pool.record_ids.tolist())}
    assert {index[r] for r in full.pruned} == pruned
    repairs = [s.record_index for s in full.selected if s.reason.startswith("repair:")]
    assert repairs == reference["topup"]


def tie_heavy(seed: int) -> tuple[Incidence, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, items = 40, 25
    edges = [
        (r, int(i))
        for r in range(rows)
        for i in rng.choice(items, size=rng.integers(1, 5))
    ]
    incidence = Incidence.from_edges(
        np.array([e[0] for e in edges]), np.array([e[1] for e in edges]), rows, items
    )
    natoms = rng.choice([1, 2, 2, 3, 4], size=rows).astype(np.int64)
    rank = rng.permutation(rows).astype(np.int64)
    return incidence, natoms, rank


@pytest.mark.parametrize("seed", range(12))
def test_lazy_and_transparent_greedy_agree(
    seed: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    incidence, natoms, rank = tie_heavy(seed)

    def run(function: Any) -> list[int]:
        covered = np.zeros(incidence.n_items, dtype=bool)
        covered[:2] = True
        candidates = np.ones(incidence.n_rows, dtype=bool)
        return list(function(incidence, natoms, rank, covered, candidates))

    transparent = run(greedy_transparent)
    assert run(greedy_lazy) == transparent
    monkeypatch.setattr(greedy_module, "float_keys_exact", lambda *_: False)
    assert run(greedy_lazy) == transparent


def test_float_key_exactness_bound() -> None:
    assert float_keys_exact(1_000, 10_000)
    assert not float_keys_exact(10**6, 10**6)


def test_greedy_with_nothing_to_cover() -> None:
    incidence = Incidence.from_edges(np.array([0]), np.array([0]), 2, 2)
    covered = np.array([True, False])
    candidates = np.array([True, True])
    natoms = np.array([1, 1])
    assert (
        greedy_lazy(incidence, natoms, natoms, covered.copy(), candidates.copy()) == []
    )
    assert greedy_transparent(incidence, natoms, natoms, covered, candidates) == []


def optimal_cost(incidence: Incidence, natoms: np.ndarray) -> int:
    best = int(natoms.sum())
    rows = range(incidence.n_rows)
    for size in range(1, incidence.n_rows + 1):
        for subset in itertools.combinations(rows, size):
            covered = np.zeros(incidence.n_items, dtype=bool)
            for row in subset:
                covered[incidence.row(row)] = True
            if covered.all():
                best = min(best, int(natoms[list(subset)].sum()))
    return best


@pytest.mark.parametrize("seed", range(6))
def test_small_covers_are_valid_and_costs_are_accounted(seed: int) -> None:
    classes = classes_for(seed + 20, 2, n_records=9, vocabulary=14)
    incidence = class_incidence(classes)
    result = Selector(presence_constraints="none").select(classes)
    covered = np.zeros(incidence.n_items, dtype=bool)
    for item in result.selected:
        covered[incidence.row(item.record_index)] = True
    assert covered.all()
    assert result.natoms == int(
        classes.pool.natoms[[s.record_index for s in result.selected]].sum()
    )
    assert result.natoms >= optimal_cost(incidence, classes.pool.natoms)


class TestIncidence:
    def test_construction_and_queries(self) -> None:
        inc = Incidence.from_edges(np.array([1, 0, 1, 1]), np.array([2, 0, 2, 0]), 3, 4)
        assert inc.indices.dtype == np.int32
        assert inc.n_edges == 3
        assert inc.row(1).tolist() == [0, 2]
        assert inc.row(2).tolist() == []
        assert inc.item_degrees().tolist() == [2, 0, 1, 0]
        assert inc.row_counts(np.array([True, False, True, False])).tolist() == [
            1,
            2,
            0,
        ]
        transposed = inc.transpose()
        assert transposed.row(0).tolist() == [0, 1]
        assert transposed.n_rows == 4

    def test_wide_item_ranges_use_64_bit_indices(self) -> None:
        inc = Incidence.from_edges(np.array([0]), np.array([2**31 + 5]), 1, 2**31 + 10)
        assert inc.indices.dtype == np.int64
        assert inc.row(0).tolist() == [2**31 + 5]

    def test_resource_budget(self) -> None:
        estimate = estimate_selection_bytes(1000, 10_000, 500)
        assert estimate > 10_000 * 4
        check_budget(estimate, None, "x")
        check_budget(estimate, estimate, "x")
        with pytest.raises(ResourcePolicyError, match="memory_budget"):
            check_budget(estimate, estimate - 1, "selection")
        assert estimate_selection_bytes(1, 1, 2**31) > estimate_selection_bytes(
            1, 1, 10
        )


class TestSteps:
    def incidence(self) -> Incidence:
        # items: 0 only in row 0; 1 in rows 0,1; 2 in rows 1,2; 3 in row 2
        return Incidence.from_edges(
            np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 1, 2, 2, 3]), 3, 4
        )

    def test_sole_suppliers(self) -> None:
        assert sole_suppliers(self.incidence()).tolist() == [0, 2]

    def test_prune_removes_largest_redundant_first_and_respects_protection(
        self,
    ) -> None:
        inc = self.incidence()
        natoms = np.array([1, 5, 1])
        rank = np.arange(3)
        assert prune([0, 1, 2], [inc], natoms, rank) == ([0, 2], [1])
        assert prune([0, 1, 2], [inc], natoms, rank, frozenset({1})) == ([0, 1, 2], [])
        assert prune([], [inc], natoms, rank) == ([], [])
        extra = Incidence.from_edges(np.array([1]), np.array([0]), 3, 1)
        assert prune([0, 1, 2], [inc, extra], natoms, rank) == ([0, 1, 2], [])

    def test_repair_prefers_gain_then_atoms_then_rank(self) -> None:
        inc = Incidence.from_edges(
            np.array([0, 1, 1, 2, 2, 3]), np.array([0, 1, 2, 1, 2, 3]), 4, 5
        )
        natoms = np.array([1, 3, 2, 1])
        rank = np.arange(4)
        added, unreachable = repair([0], inc, natoms, rank)
        assert added == [2, 3]  # rows 1 and 2 tie on gain; row 2 is smaller
        assert unreachable == 1  # item 4 has no supplier
        assert repair([0, 1, 3], inc, natoms, rank) == ([], 1)


class TestSelector:
    def test_settings_validation(self) -> None:
        with pytest.raises(NotImplementedError, match="budget mode is reserved"):
            Selector(mode="budget")
        with pytest.raises(ConfigError, match="mode"):
            Selector(mode="greedy")
        with pytest.raises(ConfigError, match="cost"):
            Selector(cost="structures")
        for kwargs in (
            {"presence_constraints": "paper"},
            {"tie_break": "random"},
            {"algorithm": "ilp"},
        ):
            with pytest.raises(ConfigError, match=next(iter(kwargs))):
                Selector(**kwargs)  # type: ignore[arg-type]

    def test_required_records_are_kept_and_unknown_ones_rejected(self) -> None:
        classes = classes_for(4)
        target = str(classes.pool.record_ids[-1])
        result = Selector(required_records=[target]).select(classes)
        assert result.selected[0].record_id == target
        assert result.selected[0].reason == "user"
        assert target not in result.pruned
        with pytest.raises(ConfigError, match="not eligible"):
            Selector(required_records=["nope"]).select(classes)
        forced = classes_for(0, 2, n_records=60, vocabulary=80)
        mandatory = sole_suppliers(class_incidence(forced))
        both = str(forced.pool.record_ids[int(mandatory[0])])
        reasons = {
            s.record_id: s.reason
            for s in Selector(required_records=[both]).select(forced).selected
        }
        assert reasons[both] == "user"

    def test_presence_repair_closes_every_family(self) -> None:
        classes = classes_for(9, 4)
        result = Selector().select(classes)
        chosen = {s.record_index for s in result.selected}
        for family in classes.presence_signatures:
            inc = presence_incidence(classes, family)
            present = set()
            for row in chosen:
                present |= set(inc.row(row).tolist())
            assert present == set(range(inc.n_items))
        assert result.unreachable == dict.fromkeys(classes.presence_signatures, 0)

    def test_prune_after_repair_preserves_classes_and_presence(self) -> None:
        classes = classes_for(10, 2)
        result = Selector(prune_after_repair=True).select(classes)
        plain = Selector().select(classes)
        assert result.natoms <= plain.natoms
        incidences = [class_incidence(classes)] + [
            presence_incidence(classes, f) for f in classes.presence_signatures
        ]
        for inc in incidences:
            covered = np.zeros(inc.n_items, dtype=bool)
            for item in result.selected:
                covered[inc.row(item.record_index)] = True
            assert covered.all()

    def test_tie_break_by_native_id(self) -> None:
        classes = classes_for(12)
        ranks = tie_ranks(classes, "natoms_then_native_id")
        assert ranks.tolist() == list(range(len(ranks)))[::-1]
        assert tie_ranks(classes, "natoms_then_record_id").tolist() == list(
            range(len(ranks))
        )
        native = Selector(tie_break="natoms_then_native_id").select(classes)
        assert native.settings["tie_break"] == "natoms_then_native_id"

    def test_transparent_algorithm_and_memory_policy(self) -> None:
        classes = classes_for(13)
        lazy = Selector().select(classes)
        transparent = Selector(algorithm="transparent").select(classes)
        assert lazy.record_ids == transparent.record_ids
        with pytest.raises(ResourcePolicyError):
            Selector(execution=ExecutionSettings(memory_budget=10)).select(classes)

    def test_result_properties(self) -> None:
        classes = classes_for(14)
        result = Selector().select(classes)
        assert result.natoms == sum(s.natoms for s in result.selected)
        assert result.record_ids == [s.record_id for s in result.selected]
        assert [s.order for s in result.selected] == list(range(len(result.selected)))
        assert result.fingerprint == Selector().select(classes).fingerprint
        assert result.fingerprint != Selector(prune=False).select(classes).fingerprint
        with pytest.raises(ConfigError, match="in-memory pool"):
            result.validate()


class TestStrataAndBaselines:
    def test_strata_values_are_sorted_and_distinct(self) -> None:
        metadata = [
            {"q": 0, "m": "PBE"},
            {"q": -1, "m": "PBE"},
            {"q": 0, "m": "PBE"},
            {},
        ]
        assert strata_values(metadata, ["q"]) == [{"q": -1}, {"q": 0}, {"q": None}]

    def test_combine_strata_deduplicates_and_labels(self) -> None:
        classes = classes_for(15)
        part = Selector().select(classes)
        combined = combine_strata(
            classes, [({"q": 0}, part), ({"q": 1}, part)], {"x": 1}
        )
        assert combined.record_ids == part.record_ids
        assert all(s.reason.endswith("@q=0") for s in combined.selected)
        assert combined.strata[1][0] == {"q": 1}
        assert len(combined.pruned) == 2 * len(part.pruned)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_subsets_never_exceed_the_budget(self, seed: int) -> None:
        classes = classes_for(16)
        budget = int(classes.pool.natoms.sum()) // 3
        rows = random_subset(classes, budget, seed)
        assert int(classes.pool.natoms[rows].sum()) <= budget
        assert len(set(rows)) == len(rows)
        assert random_subset(classes, budget, seed) == rows
        assert random_subset(classes, 0, seed) == []


def test_selected_dataclass_round_trip() -> None:
    item = Selected(1, "r", "cover", 0, 4)
    result = SelectionResult(classes=classes_for(17), settings={}, selected=[item])
    assert result.natoms == 4
    assert result.unreachable == {}


def test_pool_type_is_exported() -> None:
    assert Pool.__name__ == "Pool"
