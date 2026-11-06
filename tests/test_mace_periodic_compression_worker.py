"""Validate periodic jobs, parent splits and held-out label isolation."""

import runpy
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

WORKER = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "build_tools/mace_periodic_compression_worker.py"
    )
)


def job() -> dict[str, Any]:
    return {
        "training": deepcopy(WORKER["TRAINING_SETTINGS"]),
        "train_ids": ["a", "b"],
        "holdout_ids": ["c"],
        "train_parent_ids": ["parent-a", "parent-b"],
        "holdout_parent_ids": ["parent-c"],
        "species": [1, 6, 7, 8],
        "train_file": "/data/train.extxyz",
        "holdout_file": "/data/holdout.extxyz",
    }


def frame() -> Atoms:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[4, 4, 4], pbc=True)
    atoms.info.update(envtok_record_id="a", parent_id="parent-a")
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((2, 3)))
    return atoms


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["c"], list(map(str, range(2048)))])
def test_job_rejects_split_leakage_duplicates_and_unbounded_inputs(
    ids: list[str],
) -> None:
    config = job()
    config["train_ids"] = ids
    with pytest.raises(ValueError, match="disjoint unique"):
        WORKER["validate_job"](config)


@pytest.mark.parametrize(
    ("key", "value"), [("steps", 0), ("steps", 10001), ("seed", -1), ("seed", True)]
)
def test_job_rejects_invalid_training_workloads(key: str, value: int) -> None:
    config = job()
    config["training"][key] = value
    with pytest.raises(ValueError, match="bounded integers"):
        WORKER["validate_job"](config)


def test_job_preserves_comparable_architecture_and_allows_explicit_seeds() -> None:
    config = job()
    config["training"].update(seed=2, steps=500, threads=4)
    WORKER["validate_job"](config)
    config["training"]["hidden_irreps"] = "16x0e"
    with pytest.raises(ValueError, match="fixed training settings"):
        WORKER["validate_job"](config)


@pytest.mark.parametrize("species", [[6, 1], [1, 1], [], [0], [119], [True]])
def test_job_requires_valid_unique_species(species: list[int]) -> None:
    config = job()
    config["species"] = species
    with pytest.raises(ValueError, match="species"):
        WORKER["validate_job"](config)


def test_holdout_validation_does_not_access_reference_labels() -> None:
    atoms = frame()
    atoms.calc = None
    WORKER["validate_frames"]([atoms], ["a"], [1], labels=False)
    atoms.info.update(total_charge=2, spin_multiplicity=3)
    WORKER["validate_frames"]([atoms], ["a"], [1], labels=False)


@pytest.mark.parametrize(
    "bad", ["forces", "energy", "periodic", "cell", "species", "ids", "parent"]
)
def test_frame_validation_rejects_invalid_training_or_identity(bad: str) -> None:
    atoms = frame()
    species, ids = [1], ["a"]
    if bad == "forces":
        atoms.calc.results["forces"][0, 0] = np.nan
    elif bad == "energy":
        atoms.calc.results["energy"] = np.inf
    elif bad == "periodic":
        atoms.pbc = [True, True, False]
    elif bad == "cell":
        atoms.cell[2] = atoms.cell[1]
    elif bad == "parent":
        del atoms.info["parent_id"]
    elif bad == "species":
        species = [6]
    else:
        ids = ["b"]
    with pytest.raises(ValueError, match=r"finite|IDs|parent"):
        WORKER["validate_frames"]([atoms], ids, species, labels=True)


def test_metrics_report_per_atom_energy_and_component_forces() -> None:
    measured = WORKER["error_metrics"]([2.0, -4.0], [[1.0, -2.0, 3.0]], [2, 4])
    assert measured["energy"]["mae"] == 3
    assert measured["energy_per_atom"]["rmse"] == 1
    assert measured["forces"]["rmse"] == pytest.approx(np.sqrt(14 / 3))


@pytest.mark.parametrize("threads", [0, 2, True, 8])
def test_job_rejects_undeclared_thread_counts(threads: int) -> None:
    config = job()
    config["training"]["threads"] = threads
    with pytest.raises(ValueError, match="threads"):
        WORKER["validate_job"](config)


@pytest.mark.parametrize("parents", [["parent-a"], [], ["other", "extra"], [None]])
def test_parent_split_must_be_disjoint_and_aligned(parents: list[Any]) -> None:
    config = job()
    config["holdout_parent_ids"] = parents
    with pytest.raises(ValueError, match="parent identities"):
        WORKER["validate_job"](config)
