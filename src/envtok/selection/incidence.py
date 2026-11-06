"""Compact record-by-item presence incidence (CSR) and its resource estimate.

Rows are eligible records in canonical order; items are coverage classes or
presence signatures. Each (record, item) pair is stored once (presence, not
atomic multiplicity), with 64-bit row offsets and an index width chosen from
the number of items. No Python object is created per atom or edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from envtok._arrays import sorted_unique
from envtok.errors import ResourcePolicyError

HEAP_ENTRY_BYTES = 120
"""Approximate size of one lazy-greedy heap entry (tuple of four numbers)."""


@dataclass(frozen=True, slots=True)
class Incidence:
    """Presence incidence in CSR form.

    Attributes:
        n_rows: Number of rows (records).
        n_items: Number of items (classes or signatures).
        indptr: Row offsets, length ``n_rows + 1``.
        indices: Sorted, unique item indices of every row.
    """

    n_rows: int
    n_items: int
    indptr: npt.NDArray[np.int64]
    indices: npt.NDArray[Any]

    @classmethod
    def from_edges(
        cls,
        rows: npt.NDArray[np.int64],
        items: npt.NDArray[np.int64],
        n_rows: int,
        n_items: int,
    ) -> Incidence:
        """Build the incidence from (row, item) edges, removing duplicates.

        Args:
            rows: Row index of every edge.
            items: Item index of every edge.
            n_rows: Number of rows.
            n_items: Number of items.

        Returns:
            The incidence.
        """
        width = np.int32 if n_items < 2**31 else np.int64
        pairs = sorted_unique(
            rows.astype(np.int64) * max(n_items, 1) + items.astype(np.int64)
        )
        unique_rows = pairs // max(n_items, 1)
        indices = (pairs % max(n_items, 1)).astype(width)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(np.bincount(unique_rows, minlength=n_rows), out=indptr[1:])
        return cls(n_rows=n_rows, n_items=n_items, indptr=indptr, indices=indices)

    @property
    def n_edges(self) -> int:
        """Number of stored (row, item) pairs."""
        return int(self.indices.size)

    def row(self, row: int) -> npt.NDArray[Any]:
        """Return the items of one row."""
        return self.indices[self.indptr[row] : self.indptr[row + 1]]

    def transpose(self) -> Incidence:
        """Return the item-by-row incidence (rows stay sorted within each item)."""
        rows = np.repeat(np.arange(self.n_rows, dtype=np.int64), np.diff(self.indptr))
        order = np.argsort(self.indices, kind="stable")
        indptr = np.zeros(self.n_items + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.indices, minlength=self.n_items), out=indptr[1:])
        width = np.int32 if self.n_rows < 2**31 else np.int64
        return Incidence(
            n_rows=self.n_items,
            n_items=self.n_rows,
            indptr=indptr,
            indices=rows[order].astype(width),
        )

    def item_degrees(self) -> npt.NDArray[np.int64]:
        """Number of distinct rows containing each item."""
        return np.bincount(self.indices, minlength=self.n_items).astype(np.int64)

    def row_counts(self, mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int64]:
        """Number of items with ``mask`` true in every row.

        Args:
            mask: Boolean per item.

        Returns:
            Count per row.
        """
        flags = mask[self.indices].astype(np.int64)
        totals = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(flags)])
        counts: npt.NDArray[np.int64] = (
            totals[self.indptr[1:]] - totals[self.indptr[:-1]]
        )
        return counts


def estimate_selection_bytes(n_rows: int, n_edges: int, n_items: int) -> int:
    """Estimate the selection working state in bytes.

    Covers the CSR arrays (forward and transposed), per-record cost, rank and
    flag arrays, per-item coverage state and the lazy-greedy heap.

    Args:
        n_rows: Records.
        n_edges: Incidence edges.
        n_items: Classes or signatures.

    Returns:
        Estimated bytes.
    """
    width = 4 if n_items < 2**31 else 8
    csr = 2 * (n_edges * width + (max(n_rows, n_items) + 1) * 8)
    per_row = n_rows * (8 + 8 + 1 + HEAP_ENTRY_BYTES)
    per_item = n_items * (1 + 8)
    return int(csr + per_row + per_item)


def check_budget(estimate: int, budget: int | None, what: str) -> None:
    """Refuse a stage whose estimated working state exceeds the policy.

    Args:
        estimate: Estimated bytes.
        budget: Configured budget in bytes (``None`` disables the check).
        what: Stage description for the message.

    Raises:
        ResourcePolicyError: If the estimate exceeds the budget.
    """
    if budget is not None and estimate > budget:
        raise ResourcePolicyError(
            f"{what} needs an estimated {estimate / 2**30:.2f} GiB, above the "
            f"configured memory_budget of {budget / 2**30:.2f} GiB; raise the "
            "budget or reduce the corpus"
        )
