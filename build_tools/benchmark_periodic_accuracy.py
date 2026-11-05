"""Run a bounded periodic energy/force screen, with explicit data admission gates.

The default 32/8 OMat24 sample is deliberately checked before geometry or model
allocation. Unseen holdout elements, excessive chemistry or overlapping parents
produce an insufficient-data report rather than invented accuracy measurements.
Suitable sources run T=2, one atom-matched random seed and a full-pool reference.
"""

import argparse
import hashlib
import json
import runpy
import time
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from ase.data import chemical_symbols

from envtok import Corpus, CoverageBuilder, Selector, Tokenizer
from envtok.errors import CoverageError
from envtok.fingerprint import package_versions
from envtok.provenance import DEPENDENCIES
from envtok.records import IssueSeverity, Split, StructureRecord, validate_record
from envtok.report import audit_classes, enforce
from envtok.selection import random_subset

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/omat24/small-sample"
MODEL = runpy.run_path(str(ROOT / "build_tools/benchmark_molecular_accuracy.py"))
MAX_SPECIES = 6
MAX_FEATURES = 512
MIN_CELL_VOLUME = 1e-10


def _read(path: Path, split: Split, limit: int) -> tuple[Corpus, list[StructureRecord]]:
    """Read a declared bounded partition without sampling or changing its split."""
    corpus = Corpus.from_extxyz(
        path,
        dataset="periodic-accuracy",
        revision=hashlib.sha256(path.read_bytes()).hexdigest(),
        split=split,
        split_key="split",
        group_key="parent_id",
    )
    records = list(islice(corpus.adapter.iter_records(), limit + 1))
    if len(records) > limit:
        raise ValueError(f"{split.value} source exceeds its {limit}-record limit")
    if any(record.split is not split for record in records):
        raise ValueError(f"{split.value} source contains a different declared split")
    return corpus, records


def preflight(
    train: Sequence[StructureRecord], development: Sequence[StructureRecord]
) -> dict[str, Any]:
    """Gate model size, state of labels and train/holdout chemistry and parents."""
    species = sorted({int(z) for record in train for z in record.atoms.numbers})
    unseen = sorted(
        {int(z) for record in development for z in record.atoms.numbers} - set(species)
    )
    parents = {record.group_id for record in train}
    development_parents = {record.group_id for record in development}
    overlap = sorted((parents & development_parents) - {None})
    features = len(species) + len(species) * (len(species) + 1) // 2 * len(
        MODEL["PairBasis"].centres
    )
    issues = []
    if not train or not development:
        issues.append("empty_training_or_development_partition")
    if None in parents or None in development_parents:
        issues.append("missing_parent_ids")
    if overlap:
        issues.append("overlapping_train_development_parents")
    if unseen:
        issues.append("development_contains_elements_absent_from_training")
    if len(species) > MAX_SPECIES or features > MAX_FEATURES:
        issues.append("model_size_exceeds_small_screen_limit")
    for record in [*train, *development]:
        issues.extend(
            f"invalid_record:{issue.code}"
            for issue in validate_record(record)
            if issue.severity is not IssueSeverity.WARN
        )
        if (
            not record.atoms.pbc.all()
            or abs(np.linalg.det(record.atoms.cell)) < MIN_CELL_VOLUME
        ):
            issues.append("unsupported_periodicity_or_cell")
        for key, shape, unit in (
            ("energy", (), "eV"),
            ("forces", (record.natoms, 3), "eV/Ang"),
        ):
            label = record.labels.get(key)
            if (
                label is None
                or label.unit != unit
                or label.value.shape != shape
                or not np.isfinite(label.value).all()
            ):
                issues.append("missing_or_invalid_energy_force_labels")
    return {
        "passed": not issues,
        "issues": sorted(set(issues)),
        "training_species": [chemical_symbols[z] for z in species],
        "training_atomic_numbers": species,
        "missing_development_species": [chemical_symbols[z] for z in unseen],
        "overlapping_parent_ids": overlap,
        "required_model_features": features,
        "limits": {"training_species": MAX_SPECIES, "model_features": MAX_FEATURES},
    }


def _metrics(
    records: Sequence[StructureRecord], predictions: dict[str, Any]
) -> dict[str, Any]:
    """Measure fixed-holdout errors using normalized reference units."""
    energy = np.asarray(
        [
            predictions[r.record_id].energy - float(r.labels["energy"].value)
            for r in records
        ]
    )
    forces = np.concatenate(
        [predictions[r.record_id].forces - r.labels["forces"].value for r in records]
    )
    return {
        "energy_mae_eV": float(np.abs(energy).mean()),
        "energy_rmse_eV": float(np.sqrt(np.square(energy).mean())),
        "force_mae_eV_per_Ang": float(np.abs(forces).mean()),
        "force_rmse_eV_per_Ang": float(np.sqrt(np.square(forces).mean())),
    }


def run_screen(
    workdir: Path,
    *,
    train_path: Path = SAMPLE / "train.extxyz",
    development_path: Path = SAMPLE / "development.extxyz",
    seed: int = 0,
) -> dict[str, Any]:
    """Fit on training records only, or explain why the small data are inadequate."""
    started = time.perf_counter()
    corpus, train = _read(train_path, Split.TRAIN, 128)
    _, development = _read(development_path, Split.VALIDATION, 32)
    gate = preflight(train, development)
    report: dict[str, Any] = {
        "status": "insufficient_data",
        "min_structure_support": 2,
        "selection_seed": seed,
        "training_configurations": len(train),
        "development_configurations": len(development),
        "source_sha256": {
            "train": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            "development": hashlib.sha256(development_path.read_bytes()).hexdigest(),
        },
        "split_manifest": {
            r.record_id: {"split": r.split.value, "parent_id": r.group_id}
            for r in [*train, *development]
        },
        "data_gate": gate,
        "runs": [],
        "software": package_versions(DEPENDENCIES),
        "limitations": [
            "Small development screen; no test-set or production accuracy claim.",
            "Radial pairs omit angular, many-body and magnetic physics; not MACE.",
            "Random structures share the selected-atom ceiling and may underfill it.",
        ],
    }
    if gate["passed"]:
        report.update(_evaluate(corpus, train, development, workdir, seed, gate))
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return report


def _evaluate(
    corpus: Corpus,
    train: list[StructureRecord],
    development: list[StructureRecord],
    workdir: Path,
    seed: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Select and fit identical small periodic models against one fixed holdout."""
    cache = Tokenizer(profile="reference-723696d").run(corpus, workdir / "tokens")
    classes = CoverageBuilder(min_structure_support=2).fit(cache, split="train")
    training = {record.record_id: record for record in train}
    if set(classes.pool.record_ids.tolist()) != set(training):
        return {
            "status": "insufficient_data",
            "data_gate": {
                **gate,
                "passed": False,
                "issues": ["some_training_records_failed_tokenizer_eligibility"],
                "ineligible_training_ids": sorted(
                    set(training) - set(classes.pool.record_ids.tolist())
                ),
            },
        }
    selected = Selector().select(classes)
    selected.validate()
    random_rows = random_subset(classes, selected.natoms, seed)
    basis = MODEL["PairBasis"](tuple(gate["training_atomic_numbers"]), periodic=True)
    runs = []
    for name, ids in (
        ("full_pool", sorted(training)),
        ("envtok", selected.record_ids),
        ("random", [str(classes.pool.record_ids[row]) for row in random_rows]),
    ):
        coverage = audit_classes(classes, ids)
        try:
            enforce({**coverage, "selector": {"presence_constraints": "reference"}})
            complete = True
        except CoverageError:
            complete = False
        fitting = time.perf_counter()
        coefficients, diagnostics = MODEL["fit_model"](
            [training[rid] for rid in ids], basis
        )
        runs.append(
            {
                "selector": name,
                "selected_ids": ids,
                "mandatory_coverage_complete": complete,
                "coverage": coverage,
                "training": diagnostics,
                "coefficients": coefficients.tolist(),
                "fit_seconds": round(time.perf_counter() - fitting, 3),
                "development": _metrics(
                    development, MODEL["predict"](development, basis, coefficients)
                ),
            }
        )
    full_rmse = runs[0]["development"]["force_rmse_eV_per_Ang"]
    for run in runs[1:]:
        rmse = run["development"]["force_rmse_eV_per_Ang"]
        run["force_rmse_ratio_to_full"] = rmse / full_rmse if full_rmse else None
        run["engineering_screen_pass"] = (
            run["mandatory_coverage_complete"] and rmse <= 1.1 * full_rmse
        )
    return {
        "status": "engineering_accuracy_screen_only",
        "random_atom_ceiling": selected.natoms,
        "model": {
            "name": "periodic_linear_radial_pairs_v1",
            "species": basis.species,
            "cutoff_angstrom": basis.cutoff,
            "width_angstrom": basis.width,
            "centres_angstrom": basis.centres,
            "ridge": 1e-4,
            "loss": (
                "per-record squared per-atom energy error "
                "plus mean squared force-component error"
            ),
            "intercepts": "unpenalized element counts fitted on each training subset",
        },
        "runs": runs,
    }


def main() -> None:
    """Write measured results or an explicit insufficient-data report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=SAMPLE / "train.extxyz")
    parser.add_argument(
        "--development", type=Path, default=SAMPLE / "development.extxyz"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workdir", type=Path, default=ROOT / "runs/periodic-accuracy")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/periodic-accuracy/results.json"
    )
    args = parser.parse_args()
    report = run_screen(
        args.workdir,
        train_path=args.train,
        development_path=args.development,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"{report['status']}: {args.output}")


if __name__ == "__main__":
    main()
