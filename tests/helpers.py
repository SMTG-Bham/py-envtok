"""Helpers shared by the test modules."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms

from envtok.adapters import RecordOptions, make_record
from envtok.records import Split, StructureRecord

if TYPE_CHECKING:
    from envtok.coarsen import Pool
    from envtok.tokenizer.tokens import SiteToken

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus.extxyz"
GOLDEN = FIXTURES / "reference" / "tokens-723696d.json"


def record_of(atoms: Atoms, key: str = "0", **options: Any) -> StructureRecord:
    """Build a record from atoms with test record options."""
    base: dict[str, Any] = {"dataset": "fx", "revision": "1", "split": Split.TRAIN}
    base.update(options)
    return make_record(atoms, "corpus.extxyz", key, RecordOptions(**base))


def random_token(rng: np.random.Generator) -> SiteToken:
    """Draw a token from a small alphabet that exercises every rung."""
    from envtok.tokenizer.tokens import SiteToken, composition  # noqa: PLC0415

    element = str(rng.choice(["Na", "Cl", "O", "Fe"]))
    if rng.random() < 0.06:
        return SiteToken(
            element, int(rng.integers(0, 2)), None, "INVALID", None, None, None
        )
    count = int(rng.integers(0, 9)) if rng.random() < 0.97 else 0
    neighbours = composition(
        [str(x) for x in rng.choice(["O", "Cl", "Fe"], size=count)]
    )
    geometry = str(rng.choice(["O:6", "T:4", "BO_1:8", "UNKNOWN:4", "S:1"]))
    csm = None if geometry.startswith("UNKNOWN") else int(rng.integers(0, 4))
    defined = rng.random() < 0.95
    return SiteToken(
        element,
        int(rng.integers(0, 3)),
        neighbours,
        geometry,
        csm,
        int(rng.integers(0, 5)) if defined else None,
        int(rng.integers(0, 3)) if defined else None,
    )


def random_pool(seed: int, n_records: int = 40, vocabulary: int = 60) -> Pool:
    """Build a synthetic eligible pool with skewed token frequencies."""
    from envtok.coarsen import Pool  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    keys = sorted({random_token(rng).key() for _ in range(vocabulary)})
    weights = 1.0 / np.arange(1, len(keys) + 1)
    weights /= weights.sum()
    edges: dict[tuple[int, int], int] = {}
    for record in range(n_records):
        for _ in range(int(rng.integers(1, 7))):
            token = int(rng.choice(len(keys), p=weights))
            edges[record, token] = edges.get((record, token), 0) + 1
    used = sorted({t for _, t in edges})
    remap = {t: i for i, t in enumerate(used)}
    pairs = sorted(edges.items())
    natoms = np.zeros(n_records, dtype=np.int64)
    for (record, _), count in pairs:
        natoms[record] += count
    return Pool(
        record_ids=np.array([f"syn:{r:04d}" for r in range(n_records)], dtype=np.str_),
        natoms=natoms,
        native_ids=[f"n{(n_records - r):04d}" for r in range(n_records)],
        edge_record=np.array([r for (r, _), _ in pairs], dtype=np.int64),
        edge_token=np.array([remap[t] for (_, t), _ in pairs], dtype=np.int64),
        edge_multiplicity=np.array([c for _, c in pairs], dtype=np.int64),
        token_ids=np.arange(len(used), dtype=np.int64),
        token_keys=[keys[t] for t in used],
    )
