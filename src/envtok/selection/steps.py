"""Mandatory suppliers, redundancy pruning and rare-presence repair."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from envtok.selection.incidence import Incidence


def sole_suppliers(incidence: Incidence) -> npt.NDArray[np.int64]:
    """Return records that are the only supplier of at least one item.

    Such records belong to every feasible cover, so selecting them first
    never enlarges the optimum.

    Args:
        incidence: Record-by-item incidence.

    Returns:
        Sorted record indices.
    """
    by_item = incidence.transpose()
    single = np.flatnonzero(np.diff(by_item.indptr) == 1)
    return np.unique(by_item.indices[by_item.indptr[single]].astype(np.int64))


def prune(
    selected: Sequence[int],
    requirements: Sequence[Incidence],
    natoms: npt.NDArray[np.int64],
    rank: npt.NDArray[np.int64],
    protected: frozenset[int] = frozenset(),
) -> tuple[list[int], list[int]]:
    """Remove selected records whose every requirement stays covered without them.

    Records are visited largest first (atoms descending, then rank ascending),
    as in the reference, until a sweep removes nothing. A sole supplier of any
    requirement can never be removed.

    Args:
        selected: Selected records in pick order.
        requirements: Incidences whose items must all stay covered.
        natoms: Atom count per record.
        rank: Stable tie-break rank per record.
        protected: Records that must stay (user constraints).

    Returns:
        ``(kept, dropped)``; ``kept`` preserves pick order, ``dropped`` is in
        removal order.
    """
    chosen = list(selected)
    if not chosen:
        return [], []
    keep = dict.fromkeys(chosen, True)
    counts = [
        np.bincount(
            np.concatenate([inc.row(r) for r in chosen]).astype(np.int64),
            minlength=inc.n_items,
        )
        for inc in requirements
    ]
    order = sorted(chosen, key=lambda r: (-int(natoms[r]), int(rank[r])))
    dropped: list[int] = []
    changed = True
    while changed:
        changed = False
        for record in order:
            if not keep[record] or record in protected:
                continue
            rows = [inc.row(record) for inc in requirements]
            # Droppable when another selected record also covers every item.
            if all(np.all(c[items] > 1) for c, items in zip(counts, rows, strict=True)):
                keep[record] = False
                dropped.append(record)
                changed = True
                for c, items in zip(counts, rows, strict=True):
                    c[items] -= 1
    return [r for r in chosen if keep[r]], dropped


def repair(
    selected: Sequence[int],
    incidence: Incidence,
    natoms: npt.NDArray[np.int64],
    rank: npt.NDArray[np.int64],
) -> tuple[list[int], int]:
    """Add records until every signature of one presence family is present.

    Among records supplying missing signatures, the one adding the most
    missing signatures wins; ties go to fewer atoms, then to the lower rank.
    This differs from the atom-normalised cover score, as in the reference.

    Args:
        selected: Currently selected records.
        incidence: Record-by-signature incidence of the family.
        natoms: Atom count per record.
        rank: Stable tie-break rank per record.

    Returns:
        ``(added, unreachable)``: records added in order and the number of
        signatures no record supplies (zero for targets built from the pool).
    """
    present = np.zeros(incidence.n_items, dtype=bool)
    for record in selected:
        present[incidence.row(record)] = True
    missing = ~present
    by_item = incidence.transpose()
    supply: dict[int, set[int]] = {}
    for item in np.flatnonzero(missing).tolist():
        for record in by_item.row(item).tolist():
            supply.setdefault(int(record), set()).add(item)
    need = set(np.flatnonzero(missing).tolist())
    added: list[int] = []
    while need and supply:
        record, gain = max(
            ((r, len(items & need)) for r, items in supply.items()),
            key=lambda kv: (kv[1], -int(natoms[kv[0]]), -int(rank[kv[0]])),
        )
        if gain == 0:
            break
        added.append(record)
        need -= supply.pop(record)
    return added, len(need)
