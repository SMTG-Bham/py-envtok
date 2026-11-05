"""Generate a deterministic molecular smoke fixture with explicitly synthetic labels.

The energies/forces are a coordinate-displacement toy function. They are not
quantum calculations, model reference data or evidence of molecular accuracy.
"""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


def _graph(symbol: str, charge: int, count: int) -> dict[str, Any]:
    charges = [charge, *([0] * (count - 1))]
    radicals = [abs(charge), *([0] * (count - 1))]
    return {
        "bonds": [[0, atom, "single"] for atom in range(1, count)],
        "formal_charges": charges,
        "radicals": radicals,
        "stereochemistry": [None] * count,
        "aromatic": [False] * count,
        "rings": [False] * count,
        "source": f"explicit {symbol} fixture connectivity",
    }


def make_fixture() -> list[Atoms]:
    """Return neutral/radical states, distorted molecules and separation scans."""
    frames: list[Atoms] = []
    for name, charge, multiplicity in (("H2O", 0, 1), ("H2O", 1, 2), ("CH4", 0, 1)):
        equilibrium = molecule(name)
        graph = _graph(name, charge, len(equilibrium))
        for index, scale in enumerate((0.90, 0.96, 1.0, 1.05, 1.13, 1.22)):
            atoms = equilibrium.copy()
            atoms.positions[1:] = equilibrium.positions[0] + scale * (
                equilibrium.positions[1:] - equilibrium.positions[0]
            )
            delta = atoms.positions - equilibrium.positions
            atoms.info.update(
                sid=f"{name}-q{charge}-{index}",
                family_id=name,
                total_charge=charge,
                spin_multiplicity=multiplicity,
                label_namespace="synthetic-harmonic-v1",
                split="train",
                graph=deepcopy(graph),
                graph_status="supplied",
                synthetic_labels=True,
            )
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=float(0.5 * np.sum(delta**2) + charge),
                forces=-delta,
            )
            frames.append(atoms)
    water = molecule("H2O")
    for index, distance in enumerate((7.2, 8.2, 10.2, 13.2)):
        other = water.copy()
        other.translate((distance, 0, 0))
        atoms = water + other
        graph = _graph("H2O", 0, 3)
        graph["bonds"].extend([[3, 4, "single"], [3, 5, "single"]])
        for key in (
            "formal_charges",
            "radicals",
            "stereochemistry",
            "aromatic",
            "rings",
        ):
            graph[key] *= 2
        atoms.info.update(
            sid=f"water-dimer-{index}",
            family_id="water-dimer-scan",
            scan_id="synthetic-separation",
            total_charge=0,
            spin_multiplicity=1,
            label_namespace="synthetic-harmonic-v1",
            split="train",
            graph=graph,
            graph_status="supplied",
            fragment_ids=["left"] * 3 + ["right"] * 3,
            synthetic_labels=True,
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=0.0, forces=np.zeros((len(atoms), 3))
        )
        frames.append(atoms)
    return frames


def main() -> None:
    """Write the fixture beside the example configuration or at --output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("molecules.extxyz")
    )
    target = parser.parse_args().output
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = make_fixture()
    write(target, frames, format="extxyz")
    print(f"Wrote {len(frames)} synthetic configurations to {target}")


if __name__ == "__main__":
    main()
