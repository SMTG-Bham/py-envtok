"""Faster token construction must emit the frozen reference's tokens exactly."""

from typing import Any

import numpy as np
import pytest
import reference_molecular_representations as reference
from ase import Atoms
from ase.build import molecule

from envtok.molecular import representations as rep
from envtok.molecular.records import MolecularRecord
from envtok.records import SourceRef, Split, StructureRecord

BOND_TYPES = ("single", "double", "triple", "aromatic", 1, 1.5, "2")
STEREO = (None, None, "STEREOE", "STEREOZ", "STEREONONE")


def record(atoms: Atoms, **kwargs: Any) -> MolecularRecord:
    base = StructureRecord("r", SourceRef("test", "v1", "s", "0"), atoms, Split.TRAIN)
    return MolecularRecord(
        base, total_charge=0, spin_multiplicity=1, label_namespace="test", **kwargs
    )


def assert_same_features(row: MolecularRecord, **settings: Any) -> None:
    current = rep.featurize(
        row, rep.RepresentationSettings(backend="radial", **settings)
    )
    frozen = reference.featurize(
        row, reference.RepresentationSettings(backend="radial", **settings)
    )
    assert current.graph_status == frozen.graph_status
    assert current.token_hierarchy == frozen.token_hierarchy
    assert current.signatures == frozen.signatures
    assert current.safeguard_targets == frozen.safeguard_targets
    assert np.array_equal(current.descriptors, frozen.descriptors)


def edge_positions(rng: np.random.Generator, cutoff: float, width: float) -> Any:
    """Atoms on a line at exact and nearly exact cutoff and bin-edge distances."""
    offsets = [
        0.0,
        cutoff,
        cutoff + 1e-11,
        cutoff - 1e-11,
        width,
        2 * width + 4e-11,
        3 * width - 4e-11,
        float(rng.uniform(0, cutoff)),
    ]
    return [[x, 0.0, 0.0] for x in offsets]


@pytest.mark.parametrize("seed", range(40))
def test_fallback_tokens_match_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    cutoff = float(rng.choice([1.5, 2.0, 5.0]))
    width = float(rng.choice([0.1, 0.25, 0.3]))
    radius = int(rng.integers(0, 4))
    if seed % 2:
        positions = edge_positions(rng, cutoff, width)
    else:
        positions = rng.normal(scale=cutoff / 2, size=(int(rng.integers(1, 25)), 3))
    numbers = rng.choice([1, 6, 7, 8], size=len(positions))
    assert_same_features(
        record(Atoms(numbers=numbers, positions=positions)),
        species=(1, 6, 7, 8),
        graph_radius=radius,
        r_cut_angstrom=cutoff,
        distance_bin_angstrom=width,
    )


def random_graph(rng: np.random.Generator, natoms: int) -> dict[str, Any]:
    bonds: dict[tuple[int, int], list[Any]] = {}
    for atom in range(1, natoms):
        other = int(rng.integers(0, atom))
        bonds[other, atom] = [other, atom, BOND_TYPES[int(rng.integers(0, 3))]]
    for _ in range(int(rng.integers(0, natoms))):
        i, j = sorted(int(v) for v in rng.choice(natoms, size=2, replace=False))
        bond = [i, j, BOND_TYPES[int(rng.integers(0, len(BOND_TYPES)))]]
        stereo = STEREO[int(rng.integers(0, len(STEREO)))]
        bonds[i, j] = bond if stereo is None else [*bond, stereo]
    graph: dict[str, Any] = {"bonds": list(bonds.values())}
    if rng.random() < 0.5:
        graph["formal_charges"] = [0] * natoms
    return graph


@pytest.mark.parametrize("seed", range(60))
def test_graph_tokens_match_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    natoms = int(rng.integers(1, 14))
    # Few distinct elements make symmetric cells, twins and branching common.
    numbers = rng.choice([6, 6, 6, 1], size=natoms)
    atoms = Atoms(numbers=numbers, positions=rng.normal(size=(natoms, 3)))
    assert_same_features(
        record(atoms, graph=random_graph(rng, natoms), graph_status="supplied"),
        species=(1, 6),
        graph_radius=int(rng.integers(0, 4)),
    )


@pytest.mark.parametrize(
    "name", ["C6H6", "CH4", "isobutane", "cyclobutane", "trans-butane", "CH3CH2OH"]
)
def test_perceived_graph_tokens_match_reference(name: str) -> None:
    pytest.importorskip("rdkit")
    assert_same_features(
        record(molecule(name)),
        species=(1, 6, 8),
        allow_validated_perception=True,
        graph_radius=3,
    )
