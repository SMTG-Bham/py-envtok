"""Run a tiny CPU energy/force learning screen on the bundled OMol sample.

This is a transparent radial pair-potential regression, not a MACE benchmark or
evidence of molecular simulation accuracy. It uses neutral singlets, one grouped
development holdout, one atom budget and one seed. No models are downloaded.

Usage::

    uv run --extra molecular python build_tools/benchmark_molecular_accuracy.py
"""

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement, islice
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.neighborlist import neighbor_list

from envtok.molecular.benchmarks import (
    Prediction,
    compare_selectors,
    evaluate_predictions,
)
from envtok.molecular.data import (
    PreparedRecords,
    export_records,
    iter_extxyz,
    prepare_records,
)
from envtok.molecular.pipeline import MolecularPipeline
from envtok.molecular.records import MolecularRecord
from envtok.records import Split, StructureRecord

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/molecular/omol/sample.extxyz"
NAMESPACE = "OMol25:ORCA6:wB97M-V:def2-TZVPD:RI-J:COSX:DEFGRID3"
MAX_FORCE_RMSE_RATIO = 1.10


@dataclass(frozen=True)
class PairBasis:
    """Element intercepts and Gaussian radial pairs with a smooth cosine cutoff."""

    species: tuple[int, ...]
    cutoff: float = 5.0
    width: float = 0.5
    centres: tuple[float, ...] = (0.7, 1.1, 1.5, 2.0, 2.6, 3.3, 4.1)
    periodic: bool = False

    @property
    def size(self) -> int:
        """Number of linear energy coefficients."""
        return len(self.species) + (
            len(self.species) * (len(self.species) + 1) // 2
        ) * len(self.centres)

    def design(
        self, atoms: Atoms
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return energy features and their negative position derivatives."""
        if (atoms.pbc.any() and not self.periodic) or not set(atoms.numbers) <= set(
            self.species
        ):
            raise ValueError("pair basis requires finite structures and fitted species")
        energy = np.zeros(self.size)
        forces = np.zeros((len(atoms), 3, self.size))
        offsets = {
            pair: len(self.species) + index * len(self.centres)
            for index, pair in enumerate(combinations_with_replacement(self.species, 2))
        }
        for index, number in enumerate(self.species):
            energy[index] = np.count_nonzero(atoms.numbers == number)
        centres = np.asarray(self.centres)
        for left, right, vector, weight in self.contacts(atoms):
            distance = float(np.linalg.norm(vector))
            if distance <= 0:
                raise ValueError("pair basis cannot fit coincident atoms")
            if distance >= self.cutoff:
                continue
            pair = tuple(sorted((int(atoms.numbers[left]), int(atoms.numbers[right]))))
            start = offsets[pair]
            columns = slice(start, start + len(centres))
            gaussian = np.exp(-0.5 * ((distance - centres) / self.width) ** 2)
            cutoff = 0.5 * (1 + np.cos(np.pi * distance / self.cutoff))
            cutoff_derivative = (
                -0.5 * np.pi / self.cutoff * np.sin(np.pi * distance / self.cutoff)
            )
            derivative = gaussian * (
                cutoff_derivative - cutoff * (distance - centres) / self.width**2
            )
            energy[columns] += weight * gaussian * cutoff
            contribution = weight * np.outer(vector / distance, derivative)
            forces[left, :, columns] += contribution
            forces[right, :, columns] -= contribution
        return energy, forces

    def contacts(
        self, atoms: Atoms
    ) -> Iterator[tuple[int, int, npt.NDArray[np.float64], float]]:
        """Yield image-resolved contacts, counting directed periodic pairs once."""
        if self.periodic:
            first, second, vectors = neighbor_list("ijD", atoms, self.cutoff)
            for left, right, vector in zip(first, second, vectors, strict=True):
                yield int(left), int(right), vector, 0.5
        else:
            for left in range(len(atoms)):
                for right in range(left + 1, len(atoms)):
                    yield (
                        left,
                        right,
                        atoms.positions[right] - atoms.positions[left],
                        1.0,
                    )


def fit_model(
    records: Sequence[MolecularRecord | StructureRecord],
    basis: PairBasis,
    ridge: float = 1e-4,
) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    """Fit selected training labels with identical per-record loss and ridge.

    The loss averages squared per-atom energy error plus mean squared force
    component error across training configurations. Element intercepts are
    unpenalized; pair coefficients receive the same ridge in every comparison.
    """
    if not records:
        raise ValueError("cannot train an empty selection")
    matrices, targets = [], []
    for record in records:
        energy, forces = basis.design(record.atoms)
        matrices.extend(
            [
                energy[None, :] / record.natoms,
                forces.reshape(-1, basis.size) / math.sqrt(3 * record.natoms),
            ]
        )
        targets.extend(
            [
                np.asarray([float(record.labels["energy"].value) / record.natoms]),
                record.labels["forces"].value.reshape(-1)
                / math.sqrt(3 * record.natoms),
            ]
        )
    design = np.vstack(matrices) / math.sqrt(len(records))
    target = np.concatenate(targets) / math.sqrt(len(records))
    penalty = np.eye(basis.size)[len(basis.species) :] * math.sqrt(ridge)
    coefficients, _, rank, _ = np.linalg.lstsq(
        np.vstack([design, penalty]), np.r_[target, np.zeros(len(penalty))], rcond=None
    )
    return coefficients, {
        "training_configurations": len(records),
        "training_atoms": sum(record.natoms for record in records),
        "element_count_rank": int(
            np.linalg.matrix_rank(design[:, : len(basis.species)])
        ),
        "regularized_design_rank": int(rank),
        "parameters": basis.size,
        "ridge": ridge,
    }


def predict(
    records: Sequence[MolecularRecord | StructureRecord],
    basis: PairBasis,
    coefficients: npt.NDArray[np.float64],
) -> dict[str, Prediction]:
    """Predict energy and conservative forces without reading reference labels."""
    predictions = {}
    for record in records:
        energy, forces = basis.design(record.atoms)
        predictions[record.record_id] = Prediction(
            record.record_id, float(energy @ coefficients), forces @ coefficients
        )
    return predictions


def split_neutral_pool(
    records: Sequence[MolecularRecord], *, seed: int = 20261105
) -> PreparedRecords:
    """Split only source-training neutral singlets, preserving related groups."""
    neutral = [
        record
        for record in records
        if record.split is Split.TRAIN
        and record.total_charge == 0
        and record.spin_multiplicity == 1
    ]
    prepared = prepare_records(neutral, validation_fraction=0.25, seed=seed)
    train = prepared.candidates
    held_out = [
        record for record in prepared.records if record.split is Split.VALIDATION
    ]
    train_compositions = {record.atoms.get_chemical_formula() for record in train}
    held_out_compositions = {record.atoms.get_chemical_formula() for record in held_out}
    if not train or not held_out or train_compositions & held_out_compositions:
        raise ValueError(
            "screen needs nonempty, composition-disjoint train/holdout pools"
        )
    return prepared


def run_screen(
    workdir: Path,
    *,
    seed: int = 0,
    fraction: float = 0.25,
    source_path: Path = SOURCE,
    label_namespace: str = NAMESPACE,
    max_records: int = 128,
) -> dict[str, Any]:
    """Compare one declared atom budget against a fixed full training pool.

    The development split is independent of the budget and selection seed.
    Refuse oversized sources instead of silently sampling a different pool.
    """
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be finite and in (0, 1]")
    if not 1 <= max_records <= 128:  # noqa: PLR2004 - bounded pilot contract
        raise ValueError("max_records must be between 1 and 128")
    started = time.perf_counter()
    source = list(
        islice(
            iter_extxyz(source_path, label_namespace=label_namespace), max_records + 1
        )
    )
    if len(source) > max_records:
        raise ValueError(f"source exceeds the {max_records}-record screen limit")
    split_seed = 20261105
    prepared = split_neutral_pool(source, seed=split_seed)
    workdir.mkdir(parents=True, exist_ok=True)
    pool_path = workdir / "neutral-pool.extxyz"
    export_records(prepared.records, pool_path)
    pipeline = MolecularPipeline(
        {
            "profile": "molecular_v1",
            "source": {
                "path": str(pool_path.resolve()),
                "label_namespace": label_namespace,
            },
            "chemistry": {"allow_validated_perception": False},
            "geometry": {"backend": "soap"},
            "hierarchy": {"min_source_support": 5},
            "selection": {"mode": "budget", "atom_fraction": fraction},
            "execution": {"workdir": str((workdir / "selection").resolve())},
            "reproducibility": {"seed": seed},
        }
    )
    comparisons = compare_selectors(pipeline, fractions=(fraction,), seeds=(seed,))
    records = pipeline.ingest().records
    train = {record.record_id: record for record in pipeline.ingest().candidates}
    held_out = tuple(record for record in records if record.split is Split.VALIDATION)
    basis = PairBasis(
        tuple(
            sorted({int(z) for record in train.values() for z in record.atoms.numbers})
        )
    )
    results = []
    for selection in [comparisons["full_pool"], *comparisons["runs"]]:
        chosen = [train[record_id] for record_id in selection["selected_ids"]]
        result = {"selector": selection["selector"], "selection": selection}
        if chosen:
            fit_started = time.perf_counter()
            coefficients, diagnostics = fit_model(chosen, basis)
            result.update(
                {
                    "status": "evaluated"
                    if selection["complete"]
                    else "evaluated_incomplete_coverage",
                    "training": diagnostics,
                    "fit_seconds": time.perf_counter() - fit_started,
                    "holdout": evaluate_predictions(
                        held_out, predict(held_out, basis, coefficients)
                    ),
                    "coefficients": coefficients.tolist(),
                }
            )
        else:
            result.update({"status": "empty_selection_not_trained", "holdout": None})
        results.append(result)
    full_rmse = results[0]["holdout"]["aggregate"]["forces"]["rmse"]
    for result in results[1:]:
        result["force_rmse_ratio_to_full"] = (
            result["holdout"]["aggregate"]["forces"]["rmse"] / full_rmse
            if result["holdout"] and full_rmse
            else None
        )
        ratio = result["force_rmse_ratio_to_full"]
        result["engineering_screen_pass"] = bool(
            result["selection"]["complete"]
            and ratio is not None
            and ratio <= MAX_FORCE_RMSE_RATIO
        )
    return {
        "status": "engineering_accuracy_screen_only",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "label_namespace": label_namespace,
        "atom_fraction": fraction,
        "selection_seed": seed,
        "split_seed": split_seed,
        "source_records": len(source),
        "neutral_singlet_records": len(prepared.records),
        "training_configurations": len(train),
        "holdout_configurations": len(held_out),
        "split_policy": (
            "source train only; supplied family/trajectory links, exact-composition "
            "fallback; fixed 25% group-hash holdout"
        ),
        "engineering_screen_rule": (
            "all mandatory targets covered and holdout force RMSE <= 1.10 times "
            "the full training-pool reference; not a scientific acceptance claim"
        ),
        "split_manifest": {
            record.atoms.info["envtok_record_id"]: record.split.value
            for record in records
        },
        "source_id_mapping": {
            record.record_id: record.atoms.info["envtok_record_id"]
            for record in records
        },
        "group_warnings": list(prepared.group_warnings),
        "model": {
            "name": "linear_gaussian_radial_pairs_v1",
            "species": basis.species,
            "cutoff_angstrom": basis.cutoff,
            "width_angstrom": basis.width,
            "centres_angstrom": basis.centres,
            "loss": (
                "mean over records of squared energy error per atom "
                "plus mean squared force-component error"
            ),
            "ridge": 1e-4,
            "intercepts": (
                "unpenalized element counts fitted independently "
                "on each selected training set"
            ),
        },
        "runs": results,
        "elapsed_seconds": time.perf_counter() - started,
        "software": pipeline.software,
        "limitations": [
            "Bounded small source sample, not a representative scientific pilot.",
            "Development holdout only; no official test set or hyperparameter search.",
            "Pair model lacks angular, many-body and state physics; this is not MACE.",
            "Incomplete mandatory coverage remains a failed subset despite model fit.",
            "One split and seed cannot establish uncertainty or simulation accuracy.",
        ],
    }


def main() -> None:
    """Write machine-readable results from the bounded reproducible screen."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, default=ROOT / "runs/small-accuracy")
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--label-namespace", default=NAMESPACE)
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/small-accuracy/results.json",
    )
    args = parser.parse_args()
    report = run_screen(
        args.workdir,
        seed=args.seed,
        fraction=args.fraction,
        source_path=args.source,
        label_namespace=args.label_namespace,
        max_records=args.max_records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    for result in report["runs"]:
        metrics = result["holdout"]["aggregate"] if result["holdout"] else None
        print(result["selector"], result["status"], metrics)
    print(args.output)


if __name__ == "__main__":
    main()
