"""Deterministic molecular target hierarchies and whole-record selection.

The greedy selector is an approximation. An incomplete result reports missing
constraints and never asserts mathematical infeasibility.
"""

import hashlib
import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from envtok.errors import ConfigError

ALGORITHMS = ("lazy", "transparent")


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
    lineage, witnesses, families = _witnesses(lineages, family_ids)
    mapping: dict[str, str] = {}
    rare: set[str] = set()
    for fine, path in sorted(lineage.items()):
        chosen = next(
            (token for token in path if witnesses[token] >= min_source_support),
            path[-1],
        )
        mapping[fine] = chosen
        if witnesses[chosen] < min_source_support:
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
        source_support={token: witnesses[token] for token in sorted(classes)},
        source_group_support={
            token: 1 if isinstance(families[token], str) else len(families[token])
            for token in sorted(classes)
        },
    )


def _witnesses(
    lineages: Mapping[str, Sequence[Sequence[str]]],
    family_ids: Mapping[str, str | None] | None,
) -> tuple[dict[str, tuple[str, ...]], Counter[str], dict[str, str | set[str]]]:
    """Each fine token's lineage, and the distinct records and families per token.

    A token's families are kept as one family ID until a second appears, as
    most fine tokens occur in one family.
    """
    lineage: dict[str, tuple[str, ...]] = {}
    witnesses: Counter[str] = Counter()
    families: dict[str, str | set[str]] = {}
    for record_id, atoms in sorted(lineages.items()):
        family = (family_ids or {}).get(record_id) or record_id
        seen: set[str] = set()
        for atom in atoms:
            path = tuple(atom)
            if not path or any(not token for token in path):
                raise ConfigError("token lineages must contain nonempty token IDs")
            if path[0] in lineage and lineage[path[0]] != path:
                raise ConfigError(f"conflicting lineage for token {path[0]}")
            lineage[path[0]] = path
            seen.update(path)
        witnesses.update(seen)
        for token in seen:
            known = families.setdefault(token, family)
            if isinstance(known, set):
                known.add(family)
            elif known != family:
                families[token] = {known, family}
    return lineage, witnesses, families


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
    costs: dict[str, Fraction] = field(init=False)
    weights: dict[str, Fraction] = field(init=False)

    def __post_init__(self) -> None:
        # Exact values are parsed once; Fraction(str(x)) is the hot path otherwise.
        self.costs = {rid: _fraction(row.cost) for rid, row in self.candidates.items()}
        self.weights = {
            token: _fraction(target.weight) for token, target in self.mandatory.items()
        }

    def marginal_cost(self, ids: frozenset[str] | set[str]) -> Fraction:
        return sum((self.costs[rid] for rid in ids), start=Fraction(0))

    def add(self, ids: frozenset[str] | set[str], reason: str) -> None:
        for record_id in sorted(ids):
            candidate = self.candidates[record_id]
            self.selected.add(record_id)
            self.counts.update(candidate.targets)
            self.refined.update(candidate.refinement_targets)
            self.reasons[record_id] = reason
            self.cost += self.costs[record_id]

    def chemical_gain(self, ids: frozenset[str] | set[str]) -> Fraction:
        """Weighted unmet multiplicity supplied, over the records' own targets.

        Targets outside the contribution add ``min(0, ...) = 0``, so this equals
        the sum over every mandatory target.
        """
        contribution: Counter[str] = Counter()
        for record_id in ids:
            contribution.update(self.candidates[record_id].targets)
        gain = Fraction(0)
        for token, supplied in contribution.items():
            target = self.mandatory.get(token)
            if target is not None:
                deficit = target.requirement - self.counts[token]
                if deficit > 0:
                    gain += self.weights[token] * min(supplied, deficit)
        return gain

    def refine_gain(
        self, ids: frozenset[str] | set[str], weights: Mapping[str, Fraction]
    ) -> Fraction:
        newly_covered = (
            set().union(*(self.candidates[rid].refinement_targets for rid in ids))
            - self.refined
        )
        return sum((weights[token] for token in newly_covered), start=Fraction(0))

    def greedy(
        self, *, weights: Mapping[str, float] | None = None, lazy: bool = True
    ) -> None:
        """Add the best gain-per-cost component until none gains or fits.

        Ties go to the lower cost, then to the lexicographically first sorted
        record IDs. The lazy and transparent versions return identical picks.
        """
        exact = (
            None if weights is None else {t: _fraction(w) for t, w in weights.items()}
        )
        reason = "chemical_core" if weights is None else "geometry"

        def gain(ids: frozenset[str] | set[str]) -> Fraction:
            return (
                self.chemical_gain(ids)
                if exact is None
                else self.refine_gain(ids, exact)
            )

        (self._greedy_lazy if lazy else self._greedy_transparent)(gain, reason)

    def _options(
        self, gain: Callable[[frozenset[str]], Fraction]
    ) -> list[tuple[Fraction, Fraction, tuple[str, ...]]]:
        options = []
        for component in self.components:
            remaining = component - self.selected
            if not remaining:
                continue
            cost = self.marginal_cost(remaining)
            if self.limit is not None and self.cost + cost > self.limit:
                continue
            value = gain(remaining)
            if value > 0:
                options.append((-value / cost, cost, tuple(sorted(remaining))))
        return options

    def _greedy_transparent(
        self, gain: Callable[[frozenset[str]], Fraction], reason: str
    ) -> None:
        """Readable specification: rescan every component in every round."""
        while options := self._options(gain):
            self.add(set(min(options)[2]), reason)

    def _greedy_lazy(
        self, gain: Callable[[frozenset[str]], Fraction], reason: str
    ) -> None:
        """Lazy-heap (CELF) greedy with the transparent version's picks.

        Components are disjoint and always added or removed whole, so each
        one's cost is fixed. Gains are weighted truncated coverage, which never
        rises as records are added, so a stored key is a lower bound on the
        current one: the popped component wins once its fresh key still beats
        every stored key. The total cost only rises within a stage, so a
        component over the limit never fits again. The heap is built for each
        call because pruning between stages can raise gains.
        """
        heap = self._options(gain)
        heapq.heapify(heap)
        while heap:
            _, cost, ids = heapq.heappop(heap)
            if self.limit is not None and self.cost + cost > self.limit:
                continue
            value = gain(frozenset(ids))
            if value <= 0:
                continue
            key = (-value / cost, cost, ids)
            if heap and key > heap[0]:
                heapq.heappush(heap, key)
                continue
            self.add(set(ids), reason)

    def prune(self) -> None:
        # One unmet target blocks every removal, as each removal must leave all
        # mandatory targets at their requirement; removals never repair a target.
        if all(
            self.counts[token] >= target.requirement
            for token, target in self.mandatory.items()
        ):
            ordered = sorted(
                (
                    component
                    for component in self.components
                    if component & self.selected
                ),
                key=lambda ids: (
                    -self.marginal_cost(ids & self.selected),
                    tuple(sorted(ids)),
                ),
            )
            for component in ordered:
                removing = component & self.selected
                decrement: Counter[str] = Counter()
                for record_id in removing:
                    decrement.update(self.candidates[record_id].targets)
                if any(
                    self.counts[token] - removed < self.mandatory[token].requirement
                    for token, removed in decrement.items()
                    if token in self.mandatory
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

    def background(self, seed: int, limit: Fraction, *, lazy: bool = True) -> None:
        """Fill the reserve, preferring groups with fewer selected records.

        Ties go to a seeded hash of the component's sorted IDs, then the IDs.
        The lazy and transparent versions return identical picks.
        """
        (self._background_lazy if lazy else self._background_transparent)(seed, limit)

    def _group(self, record_id: str) -> str:
        return self.candidates[record_id].group_id or record_id

    def _background_transparent(self, seed: int, limit: Fraction) -> None:
        """Readable specification: recount and rescan every component each round."""
        while True:
            groups: Counter[str] = Counter(self._group(rid) for rid in self.selected)
            options: list[tuple[int, str, tuple[str, ...]]] = []
            for component in self.components:
                remaining = component - self.selected
                if not remaining or self.cost + self.marginal_cost(remaining) > limit:
                    continue
                ids = tuple(sorted(remaining))
                group_count = sum(groups[self._group(rid)] for rid in ids)
                options.append((group_count, _background_order(seed, ids), ids))
            if not options:
                return
            self.add(set(min(options)[2]), "background")

    def _background_lazy(self, seed: int, limit: Fraction) -> None:
        """Lazy heap with incrementally maintained group counts.

        Group counts only rise as records are added, so a stored key is a lower
        bound on the current one, and the running cost only rises, so a
        component over the limit never fits again. The order hash is fixed per
        component and computed once.
        """
        groups: Counter[str] = Counter(self._group(rid) for rid in self.selected)

        def count(ids: tuple[str, ...]) -> int:
            return sum(groups[self._group(rid)] for rid in ids)

        heap: list[tuple[int, str, tuple[str, ...], Fraction]] = []
        for component in self.components:
            remaining = component - self.selected
            if not remaining:
                continue
            cost = self.marginal_cost(remaining)
            if self.cost + cost > limit:
                continue
            ids = tuple(sorted(remaining))
            heap.append((count(ids), _background_order(seed, ids), ids, cost))
        heapq.heapify(heap)
        while heap:
            _, order, ids, cost = heapq.heappop(heap)
            if self.cost + cost > limit:
                continue
            key = (count(ids), order, ids, cost)
            if heap and key > heap[0]:
                heapq.heappush(heap, key)
                continue
            self.add(set(ids), "background")
            groups.update(self._group(rid) for rid in ids)


def _background_order(seed: int, ids: tuple[str, ...]) -> str:
    return hashlib.sha256(f"{seed}:{ids!r}".encode()).hexdigest()


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
    algorithm: str = "lazy",
) -> MolecularSelectionResult:
    """Select a chemical core, refine geometry, and fill a background reserve.

    Protected bundles are indivisible: selecting any member selects their union,
    including transitive overlapping bundles. All stages charge marginal whole
    configuration costs. Geometry and background run only after mandatory cover
    succeeds. Pruning runs before refinement so geometric witnesses survive.
    ``algorithm`` chooses the ``"lazy"`` heap greedy or the ``"transparent"``
    rescanning specification; both return identical selections.
    """
    if algorithm not in ALGORITHMS:
        raise ConfigError(f"selection algorithm must be one of {ALGORITHMS}")
    lazy = algorithm == "lazy"
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
    state.greedy(lazy=lazy)
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
        state.greedy(weights=weights, lazy=lazy)
        if hard_limit is not None and reserve:
            state.limit = hard_limit
            before = state.cost
            state.background(seed, hard_limit, lazy=lazy)
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
