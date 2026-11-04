"""Helpers shared by the test modules."""

from pathlib import Path
from typing import Any

from ase import Atoms

from envtok.adapters import RecordOptions, make_record
from envtok.records import Split, StructureRecord

FIXTURES = Path(__file__).parent / "fixtures"
CORPUS = FIXTURES / "corpus.extxyz"
GOLDEN = FIXTURES / "reference" / "tokens-723696d.json"


def record_of(atoms: Atoms, key: str = "0", **options: Any) -> StructureRecord:
    """Build a record from atoms with test record options."""
    base: dict[str, Any] = {"dataset": "fx", "revision": "1", "split": Split.TRAIN}
    base.update(options)
    return make_record(atoms, "corpus.extxyz", key, RecordOptions(**base))
