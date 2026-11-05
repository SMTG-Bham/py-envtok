"""Check the physical derivatives and split boundaries of the small benchmark."""

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from envtok.molecular.data import iter_extxyz
from envtok.records import Label, Split

SCREEN = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "build_tools/benchmark_molecular_accuracy.py"
    )
)


def test_pair_forces_are_negative_energy_gradient() -> None:
    basis = SCREEN["PairBasis"]((1, 8))
    atoms = Atoms("H2O", positions=[[0.1, 0.2, 0.3], [1.2, 0.1, 0.0], [0.2, 1.1, 0.3]])
    coefficients = np.random.default_rng(7).normal(size=basis.size)
    _, force_design = basis.design(atoms)
    forces = force_design @ coefficients
    finite_difference = np.zeros_like(forces)
    step = 1e-6
    for index in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[index, axis] += step
            minus.positions[index, axis] -= step
            finite_difference[index, axis] = -(
                basis.design(plus)[0] @ coefficients
                - basis.design(minus)[0] @ coefficients
            ) / (2 * step)
    np.testing.assert_allclose(forces, finite_difference, atol=1e-8)
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(atoms.positions, forces).sum(axis=0), 0, atol=1e-12
    )


def test_cutoff_is_smooth_and_periodic_inputs_are_rejected() -> None:
    basis = SCREEN["PairBasis"]((1,))
    atoms = Atoms("H2", positions=[[0, 0, 0], [basis.cutoff, 0, 0]])
    energy, forces = basis.design(atoms)
    np.testing.assert_array_equal(energy[1:], 0)
    np.testing.assert_array_equal(forces, 0)
    atoms.positions[1, 0] -= 1e-8
    np.testing.assert_allclose(basis.design(atoms)[1], 0, atol=1e-7)
    atoms.pbc = True
    with pytest.raises(ValueError, match="finite"):
        basis.design(atoms)


def test_grouped_holdout_is_disjoint_and_does_not_affect_training() -> None:
    records = list(iter_extxyz(SCREEN["SOURCE"], label_namespace=SCREEN["NAMESPACE"]))
    prepared = SCREEN["split_neutral_pool"](records)
    held_out = [
        record for record in prepared.records if record.split is Split.VALIDATION
    ]
    train = prepared.candidates
    assert len(prepared.records) == 21
    assert held_out
    assert train
    assert {record.atoms.get_chemical_formula() for record in train}.isdisjoint(
        record.atoms.get_chemical_formula() for record in held_out
    )
    assert all(
        record.total_charge == 0 and record.spin_multiplicity == 1 for record in train
    )
    basis = SCREEN["PairBasis"]((1, 6, 7, 8))
    coefficients, _ = SCREEN["fit_model"](train, basis)
    changed = [
        replace(
            record,
            record=replace(
                record.record,
                labels={
                    "energy": Label(np.asarray(1e12), "eV", "test"),
                    "forces": Label(
                        np.full((record.natoms, 3), 1e12), "eV/Ang", "test"
                    ),
                },
            ),
        )
        for record in held_out
    ]
    before: dict[str, Any] = SCREEN["predict"](held_out, basis, coefficients)
    after: dict[str, Any] = SCREEN["predict"](changed, basis, coefficients)
    for key, prediction in before.items():
        assert prediction.energy == after[key].energy
        np.testing.assert_array_equal(prediction.forces, after[key].forces)
    with pytest.raises(ValueError, match="empty"):
        SCREEN["fit_model"]([], basis)


@pytest.mark.parametrize("fraction", [0, -1, 1.1, float("nan"), float("inf")])
def test_screen_rejects_invalid_budget(tmp_path: Path, fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        SCREEN["run_screen"](tmp_path, fraction=fraction)


@pytest.mark.parametrize("limit", [0, 129])
def test_screen_keeps_small_pool_limit(tmp_path: Path, limit: int) -> None:
    with pytest.raises(ValueError, match="max_records"):
        SCREEN["run_screen"](tmp_path, max_records=limit)


def test_screen_refuses_oversized_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source exceeds"):
        SCREEN["run_screen"](tmp_path, max_records=2)


def test_larger_budget_keeps_the_same_holdout_and_full_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SCREEN["MolecularPipeline"]

    def radial_pipeline(settings: dict[str, Any]) -> Any:
        settings["geometry"]["backend"] = "radial"
        return original(settings)

    monkeypatch.setitem(
        SCREEN["run_screen"].__globals__, "MolecularPipeline", radial_pipeline
    )
    small = SCREEN["run_screen"](tmp_path / "quarter", fraction=0.25)
    larger = SCREEN["run_screen"](tmp_path / "half", fraction=0.5)
    assert small["split_manifest"] == larger["split_manifest"]
    assert small["source_sha256"] == larger["source_sha256"]
    assert larger["atom_fraction"] == 0.5
    np.testing.assert_allclose(
        small["runs"][0]["coefficients"], larger["runs"][0]["coefficients"]
    )
    for run in larger["runs"][1:]:
        assert run["selection"]["fraction"] == 0.5
        assert run["selection"]["achieved_cost"] <= run["selection"]["budget"]
