"""Independent regressions for state-safe and lossless molecular export."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from envtok.errors import ConfigError
from envtok.molecular.data import (
    export_records,
    from_atoms,
    iter_extxyz,
    prepare_records,
)


def _atoms(**metadata: Any) -> Atoms:
    return Atoms(
        "H2",
        positions=[[0, 0, 0], [0, 0, 0.71]],
        info={"total_charge": 0, "spin_multiplicity": 1, **metadata},
    )


def test_permissive_export_does_not_reintroduce_unmapped_canonical_state(
    tmp_path: Path,
) -> None:
    atoms = _atoms(Q=None, M=None)
    record = next(
        from_atoms(
            [atoms],
            label_namespace="ref",
            charge_key="Q",
            multiplicity_key="M",
        )
    )
    assert record.total_charge is None
    assert record.spin_multiplicity is None
    path = export_records(
        [record], tmp_path / "unknown.extxyz", strict_training_export=False
    )
    exported = read(path)
    assert isinstance(exported, Atoms)
    assert "total_charge" not in exported.info
    assert "spin_multiplicity" not in exported.info
    reread = next(iter_extxyz(path, label_namespace="ref"))
    assert reread.total_charge is None
    assert reread.spin_multiplicity is None
    assert atoms.info["total_charge"] == 0
    assert atoms.info["spin_multiplicity"] == 1


@pytest.mark.parametrize(
    "values", [["first atom", "second atom"], ["a\tb", "c"], ["a\nb", "c"]]
)
def test_export_rejects_string_columns_that_extxyz_would_silently_truncate(
    tmp_path: Path, values: list[str]
) -> None:
    atoms = _atoms()
    atoms.new_array("atom_names", np.asarray(values))
    record = next(from_atoms([atoms], label_namespace="ref"))
    target = tmp_path / "strings.extxyz"
    target.write_text("original output remains intact\n")
    with pytest.raises(ConfigError):
        export_records([record], target)
    assert target.read_text() == "original output remains intact\n"
    assert not target.with_name(f".{target.name}.partial").exists()


@pytest.mark.parametrize("name", ["pos", "species"])
def test_export_cannot_overwrite_original_geometry_with_reserved_array_names(
    tmp_path: Path, name: str
) -> None:
    atoms = _atoms()
    values = np.zeros((2, 3)) if name == "pos" else np.array(["C", "C"])
    atoms.new_array(name, values)
    record = next(from_atoms([atoms], label_namespace="ref"))
    with pytest.raises(ConfigError):
        export_records([record], tmp_path / "reserved.extxyz")
    np.testing.assert_array_equal(atoms.numbers, [1, 1])
    np.testing.assert_array_equal(atoms.positions[1], [0, 0, 0.71])


def test_programmatic_external_metadata_prevents_deduplication() -> None:
    first, second = from_atoms([_atoms(), _atoms()], label_namespace="ref")
    first = replace(
        first, record=replace(first.record, metadata={"external_field": 0.0})
    )
    second = replace(
        second, record=replace(second.record, metadata={"external_field": 1.0})
    )
    prepared = prepare_records([first, second])
    assert len(prepared.records) == 2
    assert not prepared.duplicate_lineage
