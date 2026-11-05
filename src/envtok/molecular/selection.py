"""Deterministic molecular target hierarchies and whole-record selection.

The greedy selector is an approximation. An incomplete result reports missing
constraints and never asserts mathematical infeasibility.
"""

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from envtok.errors import ConfigError


@dataclass(frozen=True, slots=True)
class Candidate:
    """An intact configuration and the distinct targets it contributes."""

    record_id: str
    cost: float
    targets: frozenset[str]
    refinement_targets: frozenset[str] = frozenset()
    group_id: str | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """Required number of distinct configurations and marginal target weight."""

    requirement: int = 1
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class HierarchyResult:
    """Frozen fine-to-coarse mapping fitted on candidate training records."""

    targets: dict[str, frozenset[str]]
    mapping: dict[str, str]
    support: dict[str, int]
    group_support: dict[str, int]
    rare_targets: frozenset[str]
    optional_targets: frozenset[str]
    lineage: dict[str, tuple[str, ...]]
    source_support: dict[str, int]
    source_group_support: dict[str, int]


def build_hierarchy(
    lineages: Mapping[str, Sequence[Sequence[str]]],
    min_source_support: int,
    *,
    family_ids: Mapping[str, str | None] | None = None,
    rare_policy: str = "protect",
) -> HierarchyResult:
    """Coarsen each fine token to its first sufficiently supported ancestor.

    Each input atom has a fine-to-coarse token sequence. Tokens must already
    include protected state and central chemistry fields at every rung. Support
    counts distinct configurations, and unknown families count as record groups.
    Rare classes remain mandatory unless explicitly marked optional.
    """
    if not _positive_integer(min_source_support):
        raise ConfigError("min_source_support must be a positive integer")
    if rare_policy not in {"protect", "optional"}:
        raise ConfigError("rare_policy must be protect or optional")
    lineage: dict[str, tuple[str, ...]] = {}
    witnesses: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    for record_id, atoms in sorted(lineages.items()):
        family = (family_ids or {}).get(record_id) or record_id
        for atom in atoms:
            path = tuple(atom)
            if not path or any(not token for token in path):
                raise ConfigError("token lineages must contain nonempty token IDs")
            if path[0] in lineage and lineage[path[0]] != path:
                raise ConfigError(f"conflicting lineage for token {path[0]}")
            lineage[path[0]] = path
            for token in path:
                witnesses[token].add(record_id)
                families[token].add(family)
    mapping: dict[str, str] = {}
    rare: set[str] = set()
    for fine, path in sorted(lineage.items()):
        chosen = next(
            (token for token in path if len(witnesses[token]) >= min_source_support),
            path[-1],
        )
        mapping[fine] = chosen
        if len(witnesses[chosen]) < min_source_support:
            rare.add(chosen)
    classes = set(mapping.values())
    targets = {
        record_id: frozenset(mapping[atom[0]] for atom in atoms)
        for record_id, atoms in sorted(lineages.items())
    }
    assigned: dict[str, set[str]] = defaultdict(set)
    assigned_groups: dict[str, set[str]] = defaultdict(set)
    for record_id, tokens in targets.items():
        for token in tokens:
            assigned[token].add(record_id)
            assigned_groups[token].add((family_ids or {}).get(record_id) or record_id)
    rare.update(token for token in classes if len(assigned[token]) < min_source_support)
    return HierarchyResult(
        targets=targets,
        mapping=mapping,
        support={token: len(assigned[token]) for token in sorted(classes)},
        group_support={token: len(assigned_groups[token]) for token in sorted(classes)},
        rare_targets=frozenset(rare),
        optional_targets=frozenset(rare if rare_policy == "optional" else ()),
        lineage=lineage,
        source_support={token: len(witnesses[token]) for token in sorted(classes)},
        source_group_support={token: len(families[token]) for token in sorted(classes)},
    )


@dataclass(frozen=True, slots=True)
class MolecularSelectionResult:
    """Achieved selection, independently recomputed counts, and unmet targets."""

    selected_ids: tuple[str, ...]
    achieved_cost: float
    target_counts: dict[str, int]
    unmet: dict[str, int]
    refinement_covered: frozenset[str]
    reasons: dict[str, str]
    budget: float | None
    mode: str
    candidate_count: int
    background_requested_cost: float = 0.0
    background_achieved_cost: float = 0.0

    @property
    def complete(self) -> bool:
        """Whether every declared mandatory multiplicity was achieved."""
        return not self.unmet


@dataclass(slots=True)
class _Selection:
    candidates: dict[str, Candidate]
    mandatory: dict[str, Target]
    components: tuple[frozenset[str], ...]
    limit: Fraction | None
    selected: set[str] = field(default_factory=set)
    counts: Counter[str] = field(default_factory=Counter)
    refined: set[str] = field(default_factory=set)
    reasons: dict[str, str] = field(default_factory=dict)
    cost: Fraction = Fraction(0)

    def marginal_cost(self, ids: frozenset[str] | set[str]) -> Fraction:
        return sum(
            (_fraction(self.candidates[rid].cost) for rid in sorted(ids)),
            start=Fraction(0),
        )

    def add(self, ids: frozenset[str] | set[str], reason: str) -> None:
        for record_id in sorted(ids):
            candidate = self.candidates[record_id]
            self.selected.add(record_id)
            self.counts.update(candidate.targets)
            self.refined.update(candidate.refinement_targets)
            self.reasons[record_id] = reason
            self.cost += _fraction(candidate.cost)

    def chemical_gain(self, ids: frozenset[str] | set[str]) -> Fraction:
        contribution: Counter[str] = Counter()
        for record_id in ids:
            contribution.update(self.candidates[record_id].targets)
        return sum(
            (
                _fraction(target.weight)
                * min(
                    contribution[token], max(0, target.requirement - self.counts[token])
                )
                for token, target in self.mandatory.items()
            ),
            start=Fraction(0),
        )

    def refine_gain(
        self, ids: frozenset[str] | set[str], weights: Mapping[str, float]
    ) -> Fraction:
        newly_covered = (
            set().union(*(self.candidates[rid].refinement_targets for rid in ids))
            - self.refined
        )
        return sum(
            (_fraction(weights[token]) for token in sorted(newly_covered)),
            start=Fraction(0),
        )

    def greedy(self, *, weights: Mapping[str, float] | None = None) -> None:
        while True:
            options: list[tuple[Fraction, Fraction, tuple[str, ...]]] = []
            for component in self.components:
                remaining = component - self.selected
                if not remaining:
                    continue
                cost = self.marginal_cost(remaining)
                if self.limit is not None and self.cost + cost > self.limit:
                    continue
                gain = (
                    self.chemical_gain(remaining)
                    if weights is None
                    else self.refine_gain(remaining, weights)
                )
                if gain > 0:
                    options.append((-gain / cost, cost, tuple(sorted(remaining))))
            if not options:
                return
            self.add(
                set(min(options)[2]), "chemical_core" if weights is None else "geometry"
            )

    def prune(self) -> None:
        ordered = sorted(
            self.components,
            key=lambda ids: (
                -self.marginal_cost(ids & self.selected),
                tuple(sorted(ids)),
            ),
        )
        for component in ordered:
            removing = component & self.selected
            if not removing:
                continue
            decrement: Counter[str] = Counter()
            for record_id in removing:
                decrement.update(self.candidates[record_id].targets)
            if any(
                self.counts[token] - decrement[token] < target.requirement
                for token, target in self.mandatory.items()
            ):
                continue
            self.selected.difference_update(removing)
            self.counts.subtract(decrement)
            self.cost -= self.marginal_cost(removing)
            for record_id in removing:
                del self.reasons[record_id]
        self.refined = set().union(
            *(self.candidates[rid].refinement_targets for rid in self.selected)
        )

    def background(self, seed: int, limit: Fraction) -> None:
        while True:
            groups: Counter[str] = Counter(
                self.candidates[rid].group_id or rid for rid in self.selected
            )
            options: list[tuple[int, str, tuple[str, ...]]] = []
            for component in self.components:
                remaining = component - self.selected
                if not remaining or self.cost + self.marginal_cost(remaining) > limit:
                    continue
                ids = tuple(sorted(remaining))
                group_count = sum(
                    groups[self.candidates[rid].group_id or rid] for rid in ids
                )
                order = hashlib.sha256(f"{seed}:{ids!r}".encode()).hexdigest()
                options.append((group_count, order, ids))
            if not options:
                return
            self.add(set(min(options)[2]), "background")


def _positive_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fraction(value: float) -> Fraction:
    return Fraction(str(value))


def _components(
    candidates: Mapping[str, Candidate], bundles: Sequence[frozenset[str]]
) -> tuple[frozenset[str], ...]:
    components = {record_id: {record_id} for record_id in candidates}
    for bundle in bundles:
        if not bundle or not bundle <= candidates.keys():
            raise ConfigError("each protected bundle must contain known candidate IDs")
        merged = set().union(*(components[rid] for rid in bundle))
        for record_id in merged:
            components[record_id] = merged
    return tuple(sorted({frozenset(ids) for ids in components.values()}, key=sorted))


def _validate_selection(
    candidates: Sequence[Candidate],
    mandatory: Mapping[str, Target],
    mode: str,
    budget: float | None,
    reserve: float,
) -> None:
    if mode not in {"cover", "budget"}:
        raise ConfigError("selection mode must be cover or budget")
    if mode == "budget" and (budget is None or not math.isfinite(budget) or budget < 0):
        raise ConfigError("budget mode requires a finite nonnegative budget")
    if mode == "cover" and budget is not None:
        raise ConfigError("cover mode has no hard budget; use budget mode")
    if not math.isfinite(reserve) or not 0 <= reserve <= 1:
        raise ConfigError("background_reserve_fraction must lie in [0, 1]")
    if reserve and mode != "budget":
        raise ConfigError("a background reserve requires budget mode")
    if len({candidate.record_id for candidate in candidates}) != len(candidates):
        raise ConfigError("candidate record IDs must be unique")
    for candidate in candidates:
        if (
            not candidate.record_id
            or not math.isfinite(candidate.cost)
            or candidate.cost <= 0
        ):
            raise ConfigError("candidates need nonempty IDs and finite positive costs")
    for target in mandatory.values():
        if (
            not _positive_integer(target.requirement)
            or not math.isfinite(target.weight)
            or target.weight <= 0
        ):
            raise ConfigError(
                "target requirements must be positive integers and weights positive"
            )


def select_candidates(
    candidates: Sequence[Candidate],
    mandatory: Mapping[str, Target] | None = None,
    *,
    mode: str = "cover",
    budget: float | None = None,
    refinement_weights: Mapping[str, float] | None = None,
    bundles: Sequence[frozenset[str]] = (),
    background_reserve_fraction: float = 0.0,
    seed: int = 0,
    prune: bool = True,
) -> MolecularSelectionResult:
    """Select a chemical core, refine geometry, and fill a background reserve.

    Protected bundles are indivisible: selecting any member selects their union,
    including transitive overlapping bundles. All stages charge marginal whole
    configuration costs. Geometry and background run only after mandatory cover
    succeeds. Pruning runs before refinement so geometric witnesses survive.
    """
    required = (
        {token: Target() for candidate in candidates for token in candidate.targets}
        if mandatory is None
        else dict(mandatory)
    )
    _validate_selection(candidates, required, mode, budget, background_reserve_fraction)
    weights = {
        token: (refinement_weights or {}).get(token, 1.0)
        for candidate in candidates
        for token in candidate.refinement_targets
    }
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ConfigError("refinement weights must be finite and positive")
    by_id = {candidate.record_id: candidate for candidate in candidates}
    hard_limit = None if budget is None else _fraction(budget)
    state = _Selection(by_id, required, _components(by_id, bundles), hard_limit)
    state.greedy()
    if prune:
        state.prune()
    reserve = (
        Fraction(0)
        if budget is None
        else _fraction(background_reserve_fraction) * _fraction(budget)
    )
    complete = all(
        state.counts[token] >= target.requirement for token, target in required.items()
    )
    background_cost = Fraction(0)
    if complete:
        if hard_limit is not None:
            state.limit = max(state.cost, hard_limit - reserve)
        state.greedy(weights=weights)
        if hard_limit is not None and reserve:
            state.limit = hard_limit
            before = state.cost
            state.background(seed, hard_limit)
            background_cost = state.cost - before
    counts: Counter[str] = Counter()
    for record_id in sorted(state.selected):
        counts.update(by_id[record_id].targets)
    return MolecularSelectionResult(
        selected_ids=tuple(sorted(state.selected)),
        achieved_cost=float(state.cost),
        target_counts={token: counts[token] for token in sorted(required)},
        unmet={
            token: target.requirement - counts[token]
            for token, target in sorted(required.items())
            if counts[token] < target.requirement
        },
        refinement_covered=frozenset(state.refined),
        reasons=dict(sorted(state.reasons.items())),
        budget=budget,
        mode=mode,
        candidate_count=len(candidates),
        background_requested_cost=float(reserve),
        background_achieved_cost=float(background_cost),
    )
