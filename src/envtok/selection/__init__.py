"""Select complete configurations that cover every coverage class.

Pipeline (``mode="cover"``): user-required records, mandatory sole
suppliers, atom-normalised greedy cover, largest-first redundancy pruning,
then repair of the four rare-presence families. Every selected record keeps
the reason it was selected. Selection never edits structures: the output is a
subset of original configurations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from envtok.coarsen import CoverageClasses
from envtok.errors import ConfigError
from envtok.execution import ExecutionSettings
from envtok.fingerprint import digest
from envtok.selection.greedy import greedy_lazy, greedy_transparent
from envtok.selection.incidence import (
    Incidence,
    check_budget,
    estimate_selection_bytes,
)
from envtok.selection.steps import prune, repair, sole_suppliers

MODES = ("cover", "budget")
TIE_BREAKS = ("natoms_then_record_id", "natoms_then_native_id")
ALGORITHMS = ("lazy", "transparent")
PRESENCE_CONSTRAINTS = ("reference", "none")


def class_incidence(classes: CoverageClasses) -> Incidence:
    """Record-by-class presence incidence of the eligible pool."""
    pool = classes.pool
    position = np.searchsorted(pool.token_ids, pool.edge_token)
    return Incidence.from_edges(
        pool.edge_record,
        classes.token_class[position],
        len(pool.record_ids),
        classes.n_classes,
    )


def presence_incidence(classes: CoverageClasses, family: str) -> Incidence:
    """Record-by-signature incidence of one presence family."""
    pool = classes.pool
    position = np.searchsorted(pool.token_ids, pool.edge_token)
    return Incidence.from_edges(
        pool.edge_record,
        classes.token_presence[family][position],
        len(pool.record_ids),
        len(classes.presence_signatures[family]),
    )


def tie_ranks(classes: CoverageClasses, tie_break: str) -> npt.NDArray[np.int64]:
    """Stable rank of every pool record for tie-breaking (lower wins).

    Args:
        classes: Fitted classes (records are in sorted record-ID order).
        tie_break: ``natoms_then_record_id`` or ``natoms_then_native_id``
            (the reference's ORDER BY mat_id, for legacy reproduction).

    Returns:
        Rank per record.
    """
    n = len(classes.pool.record_ids)
    if tie_break == "natoms_then_record_id":
        return np.arange(n, dtype=np.int64)
    keys = [(native or "", i) for i, native in enumerate(classes.pool.native_ids)]
    ranks = np.empty(n, dtype=np.int64)
    ranks[[i for _, i in sorted(keys)]] = np.arange(n)
    return ranks


@dataclass(frozen=True, slots=True)
class Selected:
    """One selected record."""

    record_index: int
    record_id: str
    reason: str
    order: int
    natoms: int


@dataclass(slots=True)
class SelectionResult:
    """A selection and everything needed to audit, report and export it.

    Attributes:
        classes: The coverage classes the selection was made for.
        settings: Resolved selector settings.
        selected: Selected records in selection order.
        pruned: Record IDs removed by pruning, in removal order.
        unreachable: Unreachable presence signatures per family.
        strata: Per-stratum results of a stratified selection.
    """

    classes: CoverageClasses
    settings: dict[str, Any]
    selected: list[Selected]
    pruned: list[str] = field(default_factory=list)
    unreachable: dict[str, int] = field(default_factory=dict)
    strata: list[tuple[dict[str, Any], SelectionResult]] = field(default_factory=list)

    @property
    def record_ids(self) -> list[str]:
        """Selected record IDs in selection order."""
        return [s.record_id for s in self.selected]

    @property
    def natoms(self) -> int:
        """Total selected atoms."""
        return sum(s.natoms for s in self.selected)

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the selection configuration (not of its outcome)."""
        return digest({"classes": self.classes.fingerprint, "settings": self.settings})

    def validate(self) -> dict[str, Any]:
        """Independently audit the selection and enforce its contract.

        Returns:
            The coverage report.

        Raises:
            CoverageError: If any class or presence signature is uncovered.
        """
        from envtok.report import audit, enforce  # noqa: PLC0415 - avoid cycle

        report = audit(self)
        enforce(report)
        return report

    def write_manifest(self, directory: str | Path, **kwargs: Any) -> Path:
        """Write the manifest, provenance, coverage and report files.

        See :func:`envtok.export.write_manifest` for keyword arguments.
        """
        from envtok.export import write_manifest  # noqa: PLC0415 - avoid cycle

        return write_manifest(self, Path(directory), **kwargs)

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        """Export the selected original structures with their labels.

        See :func:`envtok.export.export_structures` for keyword arguments.
        """
        from envtok.export import export_structures  # noqa: PLC0415 - avoid cycle

        return export_structures(self, Path(path), **kwargs)


class Selector:
    """Choose whole configurations covering the fitted classes.

    Args:
        mode: ``"cover"``. ``"budget"`` is reserved for a later release.
        cost: ``"atoms"`` (original atom count; the only supported cost).
        prune: Remove redundant selections after the cover.
        presence_constraints: ``"reference"`` repairs the four rare-presence
            families after pruning; ``"none"`` skips them.
        tie_break: ``natoms_then_record_id`` or ``natoms_then_native_id``.
        required_records: Record IDs that must be selected (user constraint).
        prune_after_repair: Optional final prune preserving classes and every
            presence signature (not part of the reference profile).
        algorithm: ``"lazy"`` or ``"transparent"`` (identical results).
        execution: Resource policy (memory budget check).
    """

    def __init__(
        self,
        mode: str = "cover",
        cost: str = "atoms",
        *,
        prune: bool = True,
        presence_constraints: str = "reference",
        tie_break: str = "natoms_then_record_id",
        required_records: Iterable[str] = (),
        prune_after_repair: bool = False,
        algorithm: str = "lazy",
        execution: ExecutionSettings | None = None,
    ) -> None:
        """Validate the settings."""
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "budget":
            raise NotImplementedError(
                "budget mode is reserved: it will stay within an atom budget and "
                "report every unmet requirement. Use mode='cover' with a T sweep"
            )
        if cost != "atoms":
            raise ConfigError("cost must be 'atoms' (original atom count)")
        for name, value, allowed in (
            ("presence_constraints", presence_constraints, PRESENCE_CONSTRAINTS),
            ("tie_break", tie_break, TIE_BREAKS),
            ("algorithm", algorithm, ALGORITHMS),
        ):
            if value not in allowed:
                raise ConfigError(f"{name} must be one of {allowed}, got {value!r}")
        self.settings: dict[str, Any] = {
            "mode": mode,
            "cost": cost,
            "prune": prune,
            "presence_constraints": presence_constraints,
            "tie_break": tie_break,
            "required_records": sorted(set(required_records)),
            "prune_after_repair": prune_after_repair,
        }
        self.algorithm = algorithm
        self.execution = execution or ExecutionSettings()

    def select(self, classes: CoverageClasses) -> SelectionResult:
        """Select a cover of ``classes``.

        Args:
            classes: Fitted coverage classes.

        Returns:
            The selection.

        Raises:
            ConfigError: If a required record is not in the eligible pool.
        """
        settings = self.settings
        pool = classes.pool
        incidence = class_incidence(classes)
        check_budget(
            estimate_selection_bytes(
                incidence.n_rows, incidence.n_edges, incidence.n_items
            ),
            self.execution.memory_budget,
            "selection",
        )
        natoms = pool.natoms
        rank = tie_ranks(classes, settings["tie_break"])
        index = {rid: i for i, rid in enumerate(pool.record_ids.tolist())}
        missing = [rid for rid in settings["required_records"] if rid not in index]
        if missing:
            raise ConfigError(f"required records are not eligible: {missing[:5]}")
        reasons: dict[int, str] = {}
        picks: list[int] = []

        def take(rows: Iterable[int], reason: str) -> None:
            for row in rows:
                if row not in reasons:
                    reasons[row] = reason
                    picks.append(row)

        take(sorted(index[r] for r in settings["required_records"]), "user")
        take(sole_suppliers(incidence).tolist(), "mandatory")
        covered = np.zeros(incidence.n_items, dtype=bool)
        for row in picks:
            covered[incidence.row(row)] = True
        candidates = np.ones(incidence.n_rows, dtype=bool)
        candidates[picks] = False
        greedy = greedy_lazy if self.algorithm == "lazy" else greedy_transparent
        take(greedy(incidence, natoms, rank, covered, candidates), "cover")
        protected = frozenset(index[r] for r in settings["required_records"])
        pruned: list[int] = []
        if settings["prune"]:
            picks, pruned = prune(picks, [incidence], natoms, rank, protected)
            for row in pruned:
                del reasons[row]
        unreachable: dict[str, int] = {}
        families: list[Incidence] = []
        if settings["presence_constraints"] == "reference":
            for family in classes.presence_families:
                family_incidence = presence_incidence(classes, family.name)
                families.append(family_incidence)
                added, unreachable[family.name] = repair(
                    picks, family_incidence, natoms, rank
                )
                take(added, f"repair:{family.name}")
        if settings["prune_after_repair"]:
            picks, late = prune(picks, [incidence, *families], natoms, rank, protected)
            pruned.extend(late)
            for row in late:
                del reasons[row]
        selected = [
            Selected(
                record_index=row,
                record_id=str(pool.record_ids[row]),
                reason=reasons[row],
                order=order,
                natoms=int(natoms[row]),
            )
            for order, row in enumerate(picks)
        ]
        return SelectionResult(
            classes=classes,
            settings={**settings, "algorithm_equivalence": "lazy==transparent"},
            selected=selected,
            pruned=[str(pool.record_ids[r]) for r in pruned],
            unreachable=unreachable,
        )


def strata_values(
    classes_pool_metadata: Sequence[dict[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Return the distinct values of the protected keys, in sorted order.

    Args:
        classes_pool_metadata: Metadata dictionaries of the eligible records.
        keys: Protected metadata keys.

    Returns:
        One mapping per stratum.
    """
    seen = {tuple(meta.get(k) for k in keys) for meta in classes_pool_metadata}
    ordered = sorted(seen, key=lambda values: [repr(v) for v in values])
    return [dict(zip(keys, values, strict=True)) for values in ordered]


def combine_strata(
    global_classes: CoverageClasses,
    parts: Sequence[tuple[dict[str, Any], SelectionResult]],
    settings: dict[str, Any],
) -> SelectionResult:
    """Merge per-stratum selections into one result audited against global classes.

    Args:
        global_classes: Classes fitted on the whole eligible pool.
        parts: ``(stratum, selection)`` pairs.
        settings: Selector settings plus ``stratify_by``.

    Returns:
        The combined selection; :meth:`SelectionResult.validate` checks every
        stratum against its own classes and reports global coverage.
    """
    index = {rid: i for i, rid in enumerate(global_classes.pool.record_ids.tolist())}
    selected: list[Selected] = []
    seen: set[str] = set()
    for stratum, part in parts:
        label = ",".join(f"{k}={v}" for k, v in stratum.items())
        for item in part.selected:
            if item.record_id in seen:
                continue
            seen.add(item.record_id)
            selected.append(
                Selected(
                    record_index=index[item.record_id],
                    record_id=item.record_id,
                    reason=f"{item.reason}@{label}",
                    order=len(selected),
                    natoms=item.natoms,
                )
            )
    return SelectionResult(
        classes=global_classes,
        settings=settings,
        selected=selected,
        pruned=[rid for _, part in parts for rid in part.pruned],
        unreachable={},
        strata=list(parts),
    )


def random_subset(classes: CoverageClasses, atom_budget: int, seed: int) -> list[int]:
    """Draw a uniform random subset of eligible records within an atom budget.

    Records are visited in a seeded random order and added while the total
    stays within the budget, so the budget is never exceeded.

    Args:
        classes: Fitted classes (defines the eligible pool).
        atom_budget: Maximum total atoms.
        seed: Random seed.

    Returns:
        Selected record indices in draw order.
    """
    rng = np.random.default_rng(seed)
    total = 0
    chosen: list[int] = []
    for row in rng.permutation(len(classes.pool.record_ids)).tolist():
        atoms = int(classes.pool.natoms[row])
        if total + atoms <= atom_budget:
            chosen.append(row)
            total += atoms
    return chosen


__all__ = [
    "Selected",
    "SelectionResult",
    "Selector",
    "class_incidence",
    "combine_strata",
    "presence_incidence",
    "random_subset",
    "strata_values",
    "tie_ranks",
]
