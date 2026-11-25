"""Molecular data correctness, state safety, split protection and export tests."""

import gzip
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixBondLengths, FixCartesian
from helpers import record_of

from envtok.errors import ConfigError, DuplicateRecordError
from envtok.molecular.data import (
    _jsonable,
    _write_frame,
    exact_identity,
    export_records,
    from_atoms,
    iter_extxyz,
    prepare_records,
)
from envtok.molecular.records import (
    MolecularRecord,
    adapt_record,
    validate_molecular_record,
)
from envtok.records import IssueSeverity, Label, Split


def water(**info: Any) -> Atoms:
    return Atoms(
        "OH2",
        positions=[[0, 0, 0], [0.12345678912345678, 0.8, 0], [-0.7, 0.3, 0]],
        info={"total_charge": 0, "spin_multiplicity": 1, **info},
    )


def adapt(atoms: Atoms, key: str = "0", **kwargs: Any) -> MolecularRecord:
    return adapt_record(
        record_of(atoms, key), label_namespace="PBE/reference", **kwargs
    )


def test_state_round_trip_preserves_precision_arrays_and_nested_metadata(
    tmp_path: Path,
) -> None:
    atoms = water(family_id="water", electronic_state_id="ground", nested={"k": [1, 2]})
    forces = np.arange(9).reshape(3, 3) / 13
    atoms.calc = SinglePointCalculator(
        atoms, energy=-3.1234567891234567, forces=forces, dipole=[0.1, 0.2, 0.3]
    )
    atoms.new_array("tensor", np.arange(27, dtype=float).reshape(3, 3, 3))
    atoms.new_array("active", np.array([True, False, True]))
    record = next(from_atoms([atoms], label_namespace="PBE/reference"))
    assert record.state_key == ("PBE/reference", 0.0, 1, "ground")
    assert record.natoms == 3
    target = export_records([record], tmp_path / "subset.extxyz")
    restored_records = list(iter_extxyz(target, label_namespace="fallback"))
    assert len(restored_records) == 1
    restored = restored_records[0]
    assert restored.state_key == record.state_key
    assert np.array_equal(restored.atoms.positions, atoms.positions)
    assert np.array_equal(restored.atoms.numbers, atoms.numbers)
    assert np.array_equal(restored.labels["forces"].value, forces)
    assert restored.labels["energy"].value == record.labels["energy"].value
    assert np.array_equal(restored.atoms.arrays["tensor"], atoms.arrays["tensor"])
    assert np.array_equal(restored.atoms.arrays["active"], atoms.arrays["active"])
    assert restored.atoms.info["nested"] == {"k": [1, 2]}
    assert restored.atoms.info["envtok_record_id"] == record.record_id
    assert restored.atoms.info["envtok_source_ref"]["shard"] == "memory"
    assert "envtok_record_id" not in atoms.info
    assert restored.atoms.calc.results["dipole"].tolist() == [0.1, 0.2, 0.3]


def test_constraint_identity_and_extxyz_round_trip(tmp_path: Path) -> None:
    fixed = water()
    fixed.set_constraint(FixAtoms(indices=[1]))
    same = water()
    same.set_constraint(FixCartesian(1, mask=[True, True, True]))
    partial = water()
    partial.set_constraint(
        [FixCartesian(0, [True, False, False]), FixAtoms(indices=[1])]
    )
    empty = water()
    empty.set_constraint(FixAtoms(indices=[]))
    records = list(
        from_atoms([water(), fixed, same, partial, empty], label_namespace="test")
    )
    prepared = prepare_records(records)
    assert len(prepared.candidates) == 3
    assert len(prepare_records(records, include_constraints=False).candidates) == 1
    target = export_records(prepared.candidates, tmp_path / "constrained.extxyz")
    again = list(iter_extxyz(target, label_namespace="test"))
    assert isinstance(again[1].atoms.constraints[0], FixAtoms)
    assert again[1].atoms.constraints[0].get_indices().tolist() == [1]
    assert [c.mask.tolist() for c in again[2].atoms.constraints] == [
        [True, False, False],
        [True, True, True],
        [False, False, False],
    ]
    fixed.set_constraint(FixBondLengths([(0, 1)]))
    with pytest.raises(
        ConfigError, match="unsupported molecular constraint: FixBondLengths"
    ):
        export_records([adapt(fixed)], target)


def test_missing_state_is_never_inferred_or_defaulted(tmp_path: Path) -> None:
    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.set_initial_charges([1])
    atoms.set_initial_magnetic_moments([2])
    record = adapt(atoms)
    assert record.total_charge is record.spin_multiplicity is None
    assert record.state_key == ("PBE/reference", None, None, None)
    assert {issue.code for issue in record.issues} == {
        "unknown_total_charge",
        "unknown_spin_multiplicity",
    }
    assert all(issue.severity is IssueSeverity.WARN for issue in record.issues)
    with pytest.raises(ConfigError, match="requires charge"):
        export_records([record], tmp_path / "missing.extxyz")
    assert not (tmp_path / ".missing.extxyz.partial").exists()
    restored = next(
        iter_extxyz(
            export_records(
                [record], tmp_path / "ok.extxyz", strict_training_export=False
            ),
            label_namespace="PBE/reference",
        )
    )
    assert restored.total_charge is restored.spin_multiplicity is None
    assert len(prepare_records([adapt(atoms, unknown_policy="reject")]).exclusions) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_charge", True),
        ("total_charge", "0"),
        ("total_charge", float("nan")),
        ("spin_multiplicity", float("inf")),
        ("spin_multiplicity", 0),
        ("spin_multiplicity", 1.5),
    ],
)
def test_invalid_state_is_excluded_with_reason(field: str, value: object) -> None:
    prepared = prepare_records([adapt(water(**{field: value}))])
    assert not prepared.candidates
    assert f"invalid_{field}" in {
        reason["code"] for reason in prepared.exclusions[0]["reasons"]
    }


def test_source_mapping_and_numpy_state() -> None:
    atoms = Atoms(
        "H",
        info={
            "q": np.int64(-1),
            "mult": np.int64(1),
            "family": "a",
            "branch": "excited",
        },
    )
    record = adapt(
        atoms,
        charge_key="q",
        multiplicity_key="mult",
        family_key="family",
        state_key="branch",
    )
    assert record.state_key == ("PBE/reference", -1.0, 1, "excited")
    assert record.family_id == "a"
    assert _jsonable({"x": (np.float64(2),)}) == {"x": [2.0]}


def test_export_normalises_nested_numpy_source_metadata(tmp_path: Path) -> None:
    atoms = water(
        total_charge=np.int64(0),
        spin_multiplicity=np.int64(1),
        nested={"array": np.asarray([1, 2]), "value": np.int64(5)},
    )
    target = export_records([adapt(atoms)], tmp_path / "numpy.extxyz")
    restored = next(iter_extxyz(target, label_namespace="method"))
    assert restored.atoms.info["nested"] == {"array": [1, 2], "value": 5}
    assert restored.atoms.info["envtok_original_state"]["total_charge"] == 0


def test_charge_zero_has_one_canonical_state_identity() -> None:
    integer = MolecularRecord(record_of(water()), total_charge=0)
    positive = replace(integer, total_charge=0.0)
    negative = replace(integer, total_charge=-0.0)
    assert json.dumps(integer.state_key) == json.dumps(positive.state_key)
    assert json.dumps(integer.state_key) == json.dumps(negative.state_key)
    floating_multiplicity: Any = 1.0
    integer_spin = replace(integer, spin_multiplicity=1)
    floating_spin = replace(integer, spin_multiplicity=floating_multiplicity)
    assert json.dumps(integer_spin.state_key) == json.dumps(floating_spin.state_key)


def test_real_omol_smoke_sample_retains_source_state_labels_and_partition() -> None:
    directory = Path(__file__).parents[1] / "examples" / "molecular" / "omol"
    payload = gzip.decompress((directory / "preview.json.gz").read_bytes())
    provenance = json.loads((directory / "provenance.json").read_text())
    assert hashlib.sha256(payload).hexdigest() == provenance["snapshot_sha256"]
    assert (
        hashlib.sha256((directory / "sample.extxyz").read_bytes()).hexdigest()
        == provenance["output_sha256"]
    )
    source = {entry["row_idx"]: entry["row"] for entry in json.loads(payload)["rows"]}
    records = list(iter_extxyz(directory / "sample.extxyz", label_namespace="unused"))
    assert len(records) == 29
    assert sum(record.natoms for record in records) == 650
    for record in records:
        row = source[record.atoms.info["source_preview_row"]]
        meta = json.loads(row["property_metadata"])
        assert record.split is Split.TRAIN
        assert record.total_charge == meta["charge"]
        assert record.spin_multiplicity == meta["spin"]
        assert np.array_equal(record.atoms.positions, row["positions"])
        assert np.array_equal(record.labels["forces"].value, row["atomic_forces"])
        assert record.labels["energy"].value == row["energy"]
        assert set(record.atoms.numbers.tolist()) <= {1, 6, 7, 8}
        assert not validate_molecular_record(record)
    assert len(prepare_records(records).candidates) == 29


@pytest.mark.parametrize(
    "kwargs", [{"unknown_policy": "neutral"}, {"label_namespace": ""}]
)
def test_invalid_adapter_settings(kwargs: dict[str, str]) -> None:
    options = {"label_namespace": "PBE", **kwargs}
    with pytest.raises(ConfigError):
        adapt_record(record_of(water()), **options)


@pytest.mark.parametrize("namespace", [None, 3, ""])
def test_invalid_source_namespace(namespace: object) -> None:
    with pytest.raises(ConfigError, match="non-empty string"):
        adapt(water(label_namespace=namespace))


def test_graph_preservation_and_ambiguous_graph() -> None:
    graph = {"bonds": [[0, 1, "single"], [0, 2, "single"]], "formal_charges": [0, 0, 0]}
    record = adapt(water(graph=graph))
    assert record.graph == graph
    assert record.graph_status == "supplied"
    ambiguous = adapt(water(graph="unmapped graph"))
    assert ambiguous.graph is None
    assert ambiguous.graph_status == "ambiguous"
    assert (
        adapt(water(graph=graph, graph_status="ambiguous")).graph_status == "ambiguous"
    )


def test_units_normalise_and_record_original_units(tmp_path: Path) -> None:
    atoms = water(E=-1.0, F=np.ones((3, 3)))
    records = list(
        from_atoms(
            [atoms],
            label_namespace="HF",
            label_sources={"energy": "info:E", "forces": "info:F"},
            label_units={"energy": "Hartree", "forces": "Hartree/Bohr"},
        )
    )
    assert records[0].labels["energy"].value == pytest.approx(-units.Hartree)
    assert records[0].labels["forces"].value == pytest.approx(
        np.ones((3, 3)) * units.Hartree / units.Bohr
    )
    assert records[0].original_label_units == {
        "energy": "Hartree",
        "forces": "Hartree/Bohr",
    }
    output = export_records(records, tmp_path / "units.extxyz")
    restored = next(iter_extxyz(output, label_namespace="HF"))
    assert restored.labels["energy"].value == records[0].labels["energy"].value
    assert restored.atoms.info["E"] == -1.0
    assert (
        restored.atoms.info["envtok_original_label_units"]["forces"] == "Hartree/Bohr"
    )
    custom = adapt_record(
        replace(
            record_of(water()),
            labels={"other": Label(np.asarray(3.0), "custom", "info:other")},
        ),
        label_namespace="method",
    )
    assert custom.labels["other"].unit == "custom"
    invalid = next(
        from_atoms(
            [atoms],
            label_namespace="HF",
            label_sources={"energy": "info:E"},
            label_units={"energy": "unknown"},
        )
    )
    assert (
        prepare_records([invalid]).exclusions[0]["reasons"][0]["code"]
        == "unsupported_label_unit"
    )
    # Exercise the same unit/schema options through the on-disk adapter.
    ase.io.write(tmp_path / "input.extxyz", atoms)
    file_record = next(
        iter_extxyz(
            tmp_path / "input.extxyz",
            label_namespace="HF",
            label_sources={"energy": "info:E"},
            label_units={"energy": "Ha"},
        )
    )
    assert file_record.labels["energy"].value == pytest.approx(-units.Hartree)


def test_validation_keeps_unusual_geometries_but_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    unusual = water()
    unusual.positions[1] *= 100
    unusual.calc = SinglePointCalculator(
        unusual, energy=1e10, forces=np.ones((3, 3)) * 1e9
    )
    assert len(prepare_records([adapt(unusual)]).candidates) == 1
    periodic = water()
    periodic.pbc = True
    assert any(
        issue.code == "unsupported_periodicity"
        for issue in validate_molecular_record(adapt(periodic))
    )
    failed = adapt(water(calculation_status="unconverged"))
    assert (
        prepare_records([failed]).exclusions[0]["reasons"][0]["code"]
        == "failed_calculation"
    )
    with pytest.raises(ConfigError, match="invalid record"):
        export_records([failed], tmp_path / "invalid.extxyz")
    bad_label = replace(
        record_of(water()),
        labels={"forces": Label(np.zeros((2, 3)), "eV/Ang", "arrays:forces")},
    )
    assert prepare_records(
        [adapt_record(bad_label, label_namespace="method")]
    ).exclusions
    nonfinite = replace(
        record_of(water()),
        labels={"energy": Label(np.asarray(np.nan), "eV", "calc:energy")},
    )
    assert prepare_records(
        [adapt_record(nonfinite, label_namespace="method")]
    ).exclusions
    vector_moments = water()
    vector_moments.set_initial_magnetic_moments(np.ones((3, 3)))
    assert not validate_molecular_record(adapt(vector_moments))


def test_exact_dedup_keeps_states_conditions_and_labels_distinct() -> None:
    originals = [
        water(sid="first"),
        water(sid="duplicate"),
        water(total_charge=1),
        water(spin_multiplicity=3),
        water(electronic_state_id="excited"),
        water(external_field=[1, 0, 0]),
        water(label_namespace="other"),
    ]
    records = list(from_atoms(originals, label_namespace="method"))
    prepared = prepare_records(records)
    assert len(prepared.records) == 6
    assert prepared.duplicate_lineage == {records[0].record_id: [records[1].record_id]}
    assert len({exact_identity(record) for record in records}) == 6
    shifted = water()
    shifted.positions += 1
    assert exact_identity(adapt(shifted)) != exact_identity(adapt(water()))


def test_conflicting_labels_and_cross_split_duplicates_remain_reported() -> None:
    atoms = [water(), water(), water(split="test")]
    for energy, frame in zip([1, 2, 1], atoms, strict=True):
        frame.calc = SinglePointCalculator(frame, energy=energy)
    records = list(from_atoms(atoms, label_namespace="method"))
    prepared = prepare_records(records)
    assert len(prepared.records) == 3
    assert len(prepared.candidates) == 2
    assert prepared.conflicts[0]["record_ids"] == [
        records[0].record_id,
        records[1].record_id,
    ]
    assert len(prepared.cross_split_duplicates) == 2
    assert {item["conflicting_labels"] for item in prepared.cross_split_duplicates} == {
        False,
        True,
    }
    assert any(
        "source assignments preserved" in warning for warning in prepared.group_warnings
    )


def test_development_split_groups_transitive_relations_and_preserves_official() -> None:
    atoms = [
        water(family_id="A"),
        water(family_id="A", trajectory_id="t"),
        water(family_id="B", trajectory_id="t"),
        water(family_id="B", state_pair_id="p"),
        water(family_id="C", state_pair_id="p"),
        water(family_id="A", split="test"),
        water(family_id="D", split="validation"),
    ]
    for index, frame in enumerate(atoms):
        frame.positions *= index + 1
    records = list(from_atoms(atoms, label_namespace="method"))
    prepared = prepare_records(records, validation_fraction=0.999999, seed=5)
    assert all(item.split is Split.VALIDATION for item in prepared.records[:5])
    assert prepared.records[5].split is Split.TEST
    assert prepared.records[6].split is Split.VALIDATION
    assert not prepared.candidates
    reverse = prepare_records(reversed(records), validation_fraction=0.999999, seed=5)
    assert reverse.split_manifest == prepared.split_manifest
    assert prepare_records([]).records == ()


def test_duplicates_ids_and_invalid_fraction_are_rejected() -> None:
    record = adapt(water())
    with pytest.raises(DuplicateRecordError):
        prepare_records([record, record])
    with pytest.raises(ConfigError, match="validation_fraction"):
        prepare_records([record], validation_fraction=1)


def test_deduplication_preserves_all_source_family_links() -> None:
    first, duplicate, related = (
        water(family_id="A"),
        water(family_id="B"),
        water(family_id="B"),
    )
    related.positions[1, 0] += 0.1
    records = list(from_atoms([first, duplicate, related], label_namespace="method"))
    for seed in range(10):
        prepared = prepare_records(records, validation_fraction=0.5, seed=seed)
        assert len(prepared.records) == 2
        assert prepared.records[0].split is prepared.records[1].split
        assert prepared.duplicate_lineage == {
            records[0].record_id: [records[1].record_id]
        }


def test_unmapped_info_labels_and_supplied_graph_distinctions_are_not_merged() -> None:
    frames = [water(energy=1.0), water(energy=2.0)]
    prepared = prepare_records(from_atoms(frames, label_namespace="method"))
    assert len(prepared.records) == 2
    assert len(prepared.conflicts) == 1
    assert not prepared.duplicate_lineage
    neutral = {
        "bonds": [[0, 1, "single"], [0, 2, "single"]],
        "formal_charges": [0, 0, 0],
    }
    charged = {**neutral, "formal_charges": [-1, 1, 0]}
    prepared = prepare_records(
        from_atoms(
            [water(graph=neutral), water(graph=charged)], label_namespace="method"
        )
    )
    assert len(prepared.records) == 2
    assert len(prepared.conflicts) == 1


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("total_charge", float("nan"), "invalid_total_charge"),
        ("spin_multiplicity", 0, "invalid_spin_multiplicity"),
        ("spin_multiplicity", 1.5, "invalid_spin_multiplicity"),
        ("label_namespace", "", "missing_label_namespace"),
    ],
)
def test_direct_records_are_independently_validated(
    field: str, value: Any, code: str
) -> None:
    record = replace(adapt(water()), **{field: value})
    assert code in {issue.code for issue in validate_molecular_record(record)}
    assert len(prepare_records([record]).exclusions) == 1


def test_binary_and_empty_string_arrays_are_rejected(tmp_path: Path) -> None:
    for array in (np.array([b"one", b"two", b"three"]), np.array(["", "two", "three"])):
        atoms = water()
        atoms.new_array("names", array)
        with pytest.raises(ConfigError):
            export_records([adapt(atoms)], tmp_path / "invalid.extxyz")


def test_export_bad_dtype_is_atomic(tmp_path: Path) -> None:
    atoms = water()
    atoms.new_array("complex", np.ones(3, dtype=np.complex128))
    target = tmp_path / "protected.extxyz"
    target.write_text("existing")
    with pytest.raises(ConfigError, match="unsupported extxyz array dtype"):
        export_records([adapt(atoms)], target)
    assert target.read_text() == "existing"
    assert not (tmp_path / ".protected.extxyz.partial").exists()
    handle = io.StringIO()
    _write_frame(handle, water())
    assert "Properties=species:S:1:pos:R:3" in handle.getvalue()
