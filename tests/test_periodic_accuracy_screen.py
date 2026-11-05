"""Check periodic pair physics and the small scientific screen's admission gates."""

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.records import Split

SCREEN = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "build_tools/benchmark_periodic_accuracy.py"
    )
)


def _sources(directory: Path) -> tuple[Path, Path]:
    paths = []
    for split, scales in (
        ("train", (0.94, 0.98, 1.02, 1.06)),
        ("validation", (0.96, 1.04)),
    ):
        frames = []
        for index, scale in enumerate(scales):
            atoms = bulk("Cu", "fcc", a=3.6 * scale, cubic=True)
            atoms.positions[0] += [0.03, 0.01 * index, -0.02]
            atoms.calc = EMT()
            energy, forces = atoms.get_potential_energy(), atoms.get_forces()
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            atoms.info.update(split=split, parent_id=f"{split}-parent-{index}")
            frames.append(atoms)
        path = directory / f"{split}.extxyz"
        ase.io.write(path, frames, format="extxyz")
        paths.append(path)
    return paths[0], paths[1]


def test_periodic_pair_forces_images_and_wrapping() -> None:
    basis = SCREEN["MODEL"]["PairBasis"]((29,), periodic=True)
    atoms = Atoms(
        "Cu2",
        positions=[[0.2, 0.3, 0.1], [1.4, 1.2, 1.6]],
        cell=[3.1, 3.2, 3.3],
        pbc=True,
    )
    coefficients = np.random.default_rng(9).normal(size=basis.size)
    energy, derivatives = basis.design(atoms)
    forces = derivatives @ coefficients
    numerical = np.zeros_like(forces)
    step = 1e-6
    for index in range(len(atoms)):
        for axis in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[index, axis] += step
            minus.positions[index, axis] -= step
            numerical[index, axis] = -(
                basis.design(plus)[0] @ coefficients
                - basis.design(minus)[0] @ coefficients
            ) / (2 * step)
    np.testing.assert_allclose(forces, numerical, atol=1e-8)
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=1e-12)
    wrapped = atoms.copy()
    wrapped.positions[1] += atoms.cell[0]
    np.testing.assert_allclose(basis.design(wrapped)[0], energy, atol=1e-12)
    np.testing.assert_allclose(basis.design(wrapped)[1], derivatives, atol=1e-12)
    single = Atoms("Cu", cell=[3, 3, 3], pbc=True)
    self_energy, self_forces = basis.design(single)
    assert np.any(self_energy[1:] > 0)
    np.testing.assert_allclose(self_forces, 0, atol=1e-12)


def test_periodic_data_gate_rejects_parent_leakage_and_unseen_elements(
    tmp_path: Path,
) -> None:
    train_path, development_path = _sources(tmp_path)
    _, train = SCREEN["_read"](train_path, Split.TRAIN, 128)
    _, development = SCREEN["_read"](development_path, Split.VALIDATION, 32)
    changed_atoms = development[0].atoms.copy()
    changed_atoms.numbers[0] = 56
    changed = [replace(development[0], atoms=changed_atoms, group_id=train[0].group_id)]
    gate = SCREEN["preflight"](train, changed)
    assert not gate["passed"]
    assert gate["missing_development_species"] == ["Ba"]
    assert gate["overlapping_parent_ids"] == [train[0].group_id]
    broad_chemistry = []
    for atomic_number in range(1, 8):
        atoms = train[0].atoms.copy()
        atoms.numbers[:] = atomic_number
        broad_chemistry.append(
            replace(train[0], record_id=f"species-{atomic_number}", atoms=atoms)
        )
    broad_gate = SCREEN["preflight"](broad_chemistry, development)
    assert "model_size_exceeds_small_screen_limit" in broad_gate["issues"]
    assert broad_gate["required_model_features"] == 203
    nonfinite = development[0].atoms.copy()
    nonfinite.positions[0, 0] = float("nan")
    invalid = SCREEN["preflight"](train, [replace(development[0], atoms=nonfinite)])
    assert not invalid["passed"]
    assert any(issue.startswith("invalid_record:") for issue in invalid["issues"])
    with pytest.raises(ValueError, match="record limit"):
        SCREEN["_read"](train_path, Split.TRAIN, 1)
    with pytest.raises(ValueError, match="different declared split"):
        SCREEN["_read"](train_path, Split.VALIDATION, 32)


def test_periodic_screen_fits_only_training_labels(tmp_path: Path) -> None:
    train_path, development_path = _sources(tmp_path)
    first: dict[str, Any] = SCREEN["run_screen"](
        tmp_path / "first", train_path=train_path, development_path=development_path
    )
    assert first["status"] == "engineering_accuracy_screen_only"
    assert len(first["runs"]) == 3
    frames = ase.io.read(development_path, index=":")
    for atoms in frames:
        atoms.calc = SinglePointCalculator(
            atoms, energy=1e9, forces=np.full((len(atoms), 3), 1e8)
        )
    ase.io.write(development_path, frames, format="extxyz")
    second = SCREEN["run_screen"](
        tmp_path / "second", train_path=train_path, development_path=development_path
    )
    for before, after in zip(first["runs"], second["runs"], strict=True):
        assert before["selected_ids"] == after["selected_ids"]
        np.testing.assert_array_equal(before["coefficients"], after["coefficients"])
        assert before["development"] != after["development"]
    assert (
        first["runs"][2]["training"]["training_atoms"] <= first["random_atom_ceiling"]
    )
