"""Greedy weighted set cover with atom cost: a transparent and a lazy version.

At each step the candidate with the highest ``newly covered items / atoms`` is
selected; ties are broken by fewer atoms, then by the stable record rank. Both
implementations use exact comparisons and return identical picks:

* :func:`greedy_transparent` rescans every candidate each step using exact
  fractions. It is the readable specification.
* :func:`greedy_lazy` keeps a max-heap of stale upper bounds (marginal gain
  can only fall as items get covered) and re-scores the top lazily, accepting
  it only when its fresh key beats every stored bound (CELF/Minoux).

This is a heuristic, not a certificate of minimum cost.
"""

import heapq
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from envtok.selection.incidence import Incidence

FLOAT_EXACT_LIMIT = 2**52


def float_keys_exact(max_gain: int, max_atoms: int) -> bool:
    """Whether float64 gain/atom ratios order candidates exactly.

    Distinct ratios ``a/b`` and ``c/d`` differ by at least ``1/(b d)``, while
    correctly rounded division errs by at most ``2**-53`` times the ratio, so
    float64 ordering is exact (and equal ratios compare equal) whenever
    ``max_gain * max_atoms**2 < 2**52``.

    Args:
        max_gain: Largest possible gain (items in one record).
        max_atoms: Largest atom count.

    Returns:
        True when float64 keys are exact.
    """
    return max_gain * max_atoms * max_atoms < FLOAT_EXACT_LIMIT


def greedy_transparent(
    incidence: Incidence,
    natoms: npt.NDArray[np.int64],
    rank: npt.NDArray[np.int64],
    covered: npt.NDArray[np.bool_],
    candidates: npt.NDArray[np.bool_],
) -> list[int]:
    """Transparent greedy cover by exhaustive rescans.

    Args:
        incidence: Record-by-item incidence.
        natoms: Atom cost of every record.
        rank: Stable tie-break rank of every record (lower wins).
        covered: Items already covered; updated in place.
        candidates: Records that may be selected; updated in place.

    Returns:
        Selected records in pick order.
    """
    picks: list[int] = []
    while not covered.all():
        best: tuple[Fraction, int, int] | None = None
        best_row = -1
        for row in np.flatnonzero(candidates).tolist():
            gain = int((~covered[incidence.row(row)]).sum())
            if gain == 0:
                continue
            key = (Fraction(gain, int(natoms[row])), -int(natoms[row]), -int(rank[row]))
            if best is None or key > best:
                best, best_row = key, row
        if best is None:
            break
        covered[incidence.row(best_row)] = True
        candidates[best_row] = False
        picks.append(best_row)
    return picks


def greedy_lazy(
    incidence: Incidence,
    natoms: npt.NDArray[np.int64],
    rank: npt.NDArray[np.int64],
    covered: npt.NDArray[np.bool_],
    candidates: npt.NDArray[np.bool_],
) -> list[int]:
    """Lazy-heap greedy cover, equivalent to :func:`greedy_transparent`.

    Args:
        incidence: Record-by-item incidence.
        natoms: Atom cost of every record.
        rank: Stable tie-break rank of every record (lower wins).
        covered: Items already covered; updated in place.
        candidates: Records that may be selected; updated in place.

    Returns:
        Selected records in pick order.
    """
    gains = incidence.row_counts(~covered)
    eligible = np.flatnonzero(candidates & (gains > 0))
    if eligible.size == 0:
        return []
    exact_floats = float_keys_exact(int(gains.max()), int(natoms[eligible].max()))

    def score(gain: int, row: int) -> float | Fraction:
        if exact_floats:
            return -(gain / int(natoms[row]))
        return -Fraction(gain, int(natoms[row]))

    heap: list[tuple[float | Fraction, int, int, int]] = [
        (score(int(gains[row]), row), int(natoms[row]), int(rank[row]), row)
        for row in eligible.tolist()
    ]
    heapq.heapify(heap)
    remaining = int((~covered).sum())
    picks: list[int] = []
    while remaining and heap:
        _, _, _, row = heapq.heappop(heap)
        items = incidence.row(row)
        gain = int((~covered[items]).sum())
        if gain == 0:
            continue
        key = (score(gain, row), int(natoms[row]), int(rank[row]), row)
        if heap and key > heap[0]:
            heapq.heappush(heap, key)
            continue
        covered[items] = True
        candidates[row] = False
        remaining -= gain
        picks.append(row)
    return picks
