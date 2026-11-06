"""Train matched fresh periodic MACE models on parent-disjoint OMat24 data.

Freeze the parent split before tokenisation, compare two EnvTok covers with
full training and atom-matched random selection, and evaluate one held-out set.
Only an existing local sample and an existing MACE interpreter are used.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
from copy import deepcopy
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols

from envtok import Corpus, CoverageBuilder, ExecutionSettings, Selector, Tokenizer
from envtok.adapters import spool_atoms
from envtok.fingerprint import package_versions
from envtok.profiles import corpus_policy
from envtok.provenance import DEPENDENCIES
from envtok.records import IssueSeverity, Split, StructureRecord, validate_record
from envtok.report import audit_classes
from envtok.selection import random_subset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/omat24-compression-demo/data/train.extxyz"
REVISION = "local-1M-shard-d7be82771964f041"
WORKER = ROOT / "build_tools/mace_periodic_compression_worker.py"
SUMMARY_HELPER = ROOT / "build_tools/benchmark_omol_compression_training.py"
SPLIT_SEED = 20261105
HOLDOUT_FRACTION = 0.20
SEEDS = (0, 1, 2)
MAX_RECORDS = 2048
MIN_CELL_VOLUME = 1e-10


def digest(path: Path) -> str:
    """Hash an input, source, or output file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write finite, readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def relative(path: Path) -> str:
    """Use repository-relative paths in portable snapshots when possible."""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def read_source(source: Path) -> list[StructureRecord]:
    """Validate a bounded native-derived sample without silently dropping rows."""
    corpus = Corpus.from_extxyz(
        source,
        dataset="omat24-training",
        revision=REVISION,
        split=Split.TRAIN,
        split_key="split",
        group_key="parent_id",
        id_policy="namespaced_native_id",
        native_id_key="sid",
        policy=corpus_policy("omat24-study"),
    )
    records = list(islice(corpus.adapter.iter_records(), MAX_RECORDS + 1))
    if not records or len(records) > MAX_RECORDS:
        raise ValueError("source must contain 1-2048 records")
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("source record IDs must be unique")
    for record in records:
        if record.split is not Split.TRAIN:
            raise ValueError("source must contain only declared training records")
        if not isinstance(record.group_id, str) or not record.group_id.strip():
            raise ValueError("every source record requires a supplied parent ID")
        if (
            not record.atoms.pbc.all()
            or abs(np.linalg.det(record.atoms.cell.array)) < MIN_CELL_VOLUME
        ):
            raise ValueError("every source must have full periodicity and nonzero cell")
        if any(
            issue.severity is not IssueSeverity.WARN
            for issue in validate_record(record)
        ):
            raise ValueError("source contains an invalid record")
        if corpus.policy.excluded_symbols(record.atoms.get_chemical_symbols()):
            raise ValueError("source contains elements excluded by the study policy")
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
                raise ValueError(
                    "source needs finite energy and force labels in declared units"
                )
    return records


def split_records(
    records: list[StructureRecord],
) -> tuple[list[StructureRecord], list[StructureRecord]]:
    """Assign complete parent groups using only a declared seed and parent ID."""
    train, holdout = [], []
    for record in records:
        fraction = (
            int(
                hashlib.sha256(f"{SPLIT_SEED}:{record.group_id}".encode()).hexdigest(),
                16,
            )
            / 2**256
        )
        split = Split.VALIDATION if fraction < HOLDOUT_FRACTION else Split.TRAIN
        (holdout if split is Split.VALIDATION else train).append(
            replace(record, split=split)
        )
    if not train or not holdout:
        raise ValueError("parent-hash split needs nonempty training and holdout")
    if {r.group_id for r in train} & {r.group_id for r in holdout}:
        raise ValueError("training and holdout parents must be disjoint")
    return train, holdout


def export_records(records: list[StructureRecord], path: Path) -> str:
    """Preserve source geometry, metadata and stored labels with split annotations."""
    frames = []
    for record in records:
        atoms = record.atoms.copy()
        atoms.info = deepcopy(record.atoms.info)
        atoms.info.update(envtok_record_id=record.record_id, split=record.split.value)
        if record.atoms.calc is not None:
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = deepcopy(record.atoms.calc.results)
        frames.append(atoms)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path.with_name(path.name + ".candidate")
    checksum = spool_atoms(frames, candidate)
    if path.exists() and digest(path) == checksum:
        candidate.unlink()
    else:
        candidate.replace(path)
    return checksum


def force_statistics(records: list[StructureRecord]) -> dict[str, Any]:
    """Describe label scale without filtering any large-force configurations."""
    values = np.concatenate([r.labels["forces"].value.reshape(-1) for r in records])
    peaks = sorted(
        (
            {
                "record_id": r.record_id,
                "source_record_id": r.atoms.info.get("source_record_id"),
                "max_abs_component_eV_per_Ang": float(
                    np.max(np.abs(r.labels["forces"].value))
                ),
            }
            for r in records
        ),
        key=lambda row: row["max_abs_component_eV_per_Ang"],
        reverse=True,
    )
    return {
        "component_rms_eV_per_Ang": float(np.sqrt(np.mean(values**2))),
        "max_abs_component_eV_per_Ang": float(np.max(np.abs(values))),
        "abs_component_p99_eV_per_Ang": float(np.quantile(np.abs(values), 0.99)),
        "largest_force_records": peaks[:5],
        "outlier_filter_applied": False,
    }


def count_rank(records: list[StructureRecord], species: list[int]) -> int:
    """Check whether selected compositions determine all element reference energies."""
    counts = np.asarray(
        [[np.count_nonzero(r.atoms.numbers == z) for z in species] for r in records]
    )
    return int(np.linalg.matrix_rank(counts))


def prepare(
    source: Path, workdir: Path, *, workers: int = 8
) -> tuple[dict[str, Any], list[StructureRecord]]:
    """Freeze the parent split, then fit coverage targets on training records only."""
    original = read_source(source)
    train, holdout = split_records(original)
    species = sorted({int(z) for r in train for z in r.atoms.numbers})
    if {int(z) for r in holdout for z in r.atoms.numbers} - set(species):
        raise ValueError("heldout elements absent from training")
    if count_rank(train, species) != len(species):
        raise ValueError("training element-count matrix is rank deficient")
    workdir.mkdir(parents=True, exist_ok=True)
    train_source = workdir / "training-pool.extxyz"
    export_records(train, train_source)
    holdout_file = workdir / "holdout.extxyz"
    holdout_hash = export_records(holdout, holdout_file)
    corpus = Corpus.from_extxyz(
        train_source,
        dataset="omat24-training",
        revision=REVISION,
        split=Split.TRAIN,
        split_key="split",
        group_key="parent_id",
        id_policy="namespaced_native_id",
        native_id_key="sid",
        policy=corpus_policy("omat24-study"),
    )
    training = {r.record_id: r for r in corpus.adapter.iter_records()}
    if set(training) != {r.record_id for r in train}:
        raise ValueError("training export changed stable source identities")
    started = time.perf_counter()
    cache = Tokenizer(
        profile="reference-723696d",
        execution=ExecutionSettings(
            workers=workers, batch_size=8, threads_per_worker=1
        ),
    ).run(corpus, workdir / "tokens")
    token_seconds = time.perf_counter() - started
    selections = []
    for threshold in (2, 100):
        started = time.perf_counter()
        classes = CoverageBuilder(min_structure_support=threshold).fit(
            cache, split="train"
        )
        if set(classes.pool.record_ids.tolist()) != set(training):
            raise ValueError(
                "some declared training records failed tokenizer eligibility"
            )
        selected = Selector(presence_constraints="reference").select(classes)
        coverage = selected.validate()
        directory = workdir / f"envtok_T{threshold}"
        selected.write_manifest(directory / "selection")
        selections.append(
            {
                "selector": f"envtok_T{threshold}",
                "min_structure_support": threshold,
                "selected_ids": selected.record_ids,
                "coverage": coverage,
                "selection_seconds": time.perf_counter() - started,
                "selection_directory": relative(directory / "selection"),
            }
        )
    # The last class mapping is T=100 and is frozen for full/random diagnostics.
    full_ids = sorted(training)
    full = {
        "selector": "full_pool",
        "selected_ids": full_ids,
        "coverage": audit_classes(classes, full_ids),
        "coverage_support_threshold": 100,
    }
    budget = selections[-1]["coverage"]["atoms"]["selected"]
    random_rows = random_subset(classes, budget, seed=0)
    random_ids = [str(classes.pool.record_ids[row]) for row in random_rows]
    selections = [
        full,
        *selections,
        {
            "selector": "random",
            "selection_seed": 0,
            "atom_ceiling": budget,
            "selected_ids": random_ids,
            "coverage": audit_classes(classes, random_ids),
            "coverage_support_threshold": 100,
        },
    ]
    heldout_ids = {r.record_id for r in holdout}
    for selection in selections:
        ids = selection["selected_ids"]
        if not ids or set(ids) - training.keys() or set(ids) & heldout_ids:
            raise ValueError("selection must contain training records only")
        chosen = [training[rid] for rid in ids]
        rank = count_rank(chosen, species)
        if rank != len(species):
            raise ValueError("selected element-count matrix is rank deficient")
        output = workdir / selection["selector"] / "train.extxyz"
        selection.update(
            train_file=relative(output),
            train_sha256=export_records(chosen, output),
            training_configurations=len(chosen),
            training_atoms=sum(r.natoms for r in chosen),
            element_count_rank=rank,
            force_statistics=force_statistics(chosen),
        )
    experiment = {
        "source": relative(source),
        "source_sha256": digest(source),
        "source_records": len(original),
        "source_atoms": sum(r.natoms for r in original),
        "dataset": "OMat24",
        "source_revision": REVISION,
        "split_seed": SPLIT_SEED,
        "holdout_fraction_rule": HOLDOUT_FRACTION,
        "split_rule": "sha256(seed:parent_id) / 2**256 < 0.20",
        "split_manifest": {
            r.record_id: {
                "split": r.split.value,
                "parent_id": r.group_id,
                "source_record_id": r.atoms.info.get("source_record_id"),
            }
            for r in [*train, *holdout]
        },
        "training_configurations": len(train),
        "training_atoms": sum(r.natoms for r in train),
        "holdout_configurations": len(holdout),
        "holdout_atoms": sum(r.natoms for r in holdout),
        "training_parents": len({r.group_id for r in train}),
        "holdout_parents": len({r.group_id for r in holdout}),
        "holdout_file": relative(holdout_file),
        "holdout_sha256": holdout_hash,
        "holdout_ids": [r.record_id for r in holdout],
        "species": species,
        "chemical_symbols": [chemical_symbols[z] for z in species],
        "training_force_statistics": force_statistics(train),
        "holdout_force_statistics": force_statistics(holdout),
        "holdout_zero_force_rmse_eV_per_Ang": force_statistics(holdout)[
            "component_rms_eV_per_Ang"
        ],
        "tokenization_seconds": token_seconds,
        "token_cache": relative(cache.token_dir),
        "selection_software": package_versions(DEPENDENCIES),
        "selections": selections,
        "checks": {
            "parent_disjoint": True,
            "selection_train_only": True,
            "all_training_records_tokenized": True,
            "all_selections_element_count_full_rank": True,
        },
    }
    return experiment, holdout


def metrics(
    records: list[StructureRecord], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Independently measure paired periodic energy and Cartesian force errors."""
    predictions = {row["record_id"]: row for row in rows}
    if len(predictions) != len(rows) or set(predictions) != {
        r.record_id for r in records
    }:
        raise ValueError("prediction IDs must exactly match unique heldout IDs")
    energy_errors, forces = [], []
    for record in records:
        row = predictions[record.record_id]
        energy, force = np.asarray(row["energy"]), np.asarray(row["forces"])
        if energy.shape != () or force.shape != (record.natoms, 3):
            raise ValueError("prediction shapes differ from source atom order")
        if not np.isfinite(energy).all() or not np.isfinite(force).all():
            raise ValueError("predictions must be finite")
        energy_errors.append(float(energy) - float(record.labels["energy"].value))
        forces.append(force - record.labels["forces"].value)
    errors = np.asarray(energy_errors)
    aggregates = {}
    for key, values, unit in (
        ("energy", errors, "eV"),
        (
            "energy_per_atom",
            errors / np.asarray([r.natoms for r in records]),
            "eV/atom",
        ),
        ("forces", np.concatenate(forces), "eV/Ang"),
    ):
        aggregates[key] = {
            "mae": float(np.abs(values).mean()),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "unit": unit,
        }
    return {
        "aggregate": {
            "configurations": len(records),
            "atoms": sum(r.natoms for r in records),
            **aggregates,
        }
    }


def train_job(
    job: dict[str, Any], directory: Path, mace_python: Path
) -> tuple[dict[str, Any], bool]:
    """Run sequentially or reuse hash-verified inputs, predictions and checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    job_path, output, stamp = (
        directory / name
        for name in ("job.json", "worker-results.json", "completed.json")
    )
    expected = {
        "job": job,
        "worker_sha256": digest(WORKER),
        "train_sha256": digest(Path(job["train_file"])),
        "holdout_sha256": digest(Path(job["holdout_file"])),
        "python": str(mace_python.resolve()),
    }
    checkpoint = directory / "model-state.pt"
    if stamp.exists() and output.exists() and checkpoint.exists():
        saved = json.loads(stamp.read_text())
        if (
            saved.get("inputs") == expected
            and saved.get("result_sha256") == digest(output)
            and saved.get("checkpoint_sha256") == digest(checkpoint)
        ):
            return json.loads(output.read_text()), True
    write_json(job_path, job)
    environment = os.environ | {
        "MPLCONFIGDIR": str((directory / "matplotlib").resolve()),
        "OMP_NUM_THREADS": str(job["training"]["threads"]),
        "OPENBLAS_NUM_THREADS": "1",
    }
    with (directory / "worker.log").open("w") as log:
        subprocess.run(  # noqa: S603 - explicit local interpreter and script, no shell
            [str(mace_python), str(WORKER), "--worker", str(job_path.resolve())],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=7200,
        )
    measured = json.loads(output.read_text())
    write_json(
        stamp,
        {
            "inputs": expected,
            "result_sha256": digest(output),
            "checkpoint_sha256": digest(checkpoint),
        },
    )
    return measured, False


def main() -> None:
    """Prepare frozen selections and run the bounded twelve-model comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / "runs/omat24-compression-training"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples/omat24/compression-demo/training-results.json",
    )
    parser.add_argument("--mace-python", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, choices=(1, 4), default=4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not args.prepare_only and args.mace_python is None:
        parser.error("--mace-python must identify an existing MACE interpreter")
    started = time.perf_counter()
    workdir = args.workdir.resolve()
    experiment, holdout = prepare(args.source.resolve(), workdir, workers=args.workers)
    write_json(workdir / "experiment.json", experiment)
    print(
        json.dumps(
            {
                "train": experiment["training_configurations"],
                "holdout": experiment["holdout_configurations"],
                "selections": [
                    {
                        k: s[k]
                        for k in (
                            "selector",
                            "training_configurations",
                            "training_atoms",
                        )
                    }
                    for s in experiment["selections"]
                ],
            }
        ),
        flush=True,
    )
    if args.prepare_only:
        return
    worker = runpy.run_path(str(WORKER))
    settings = worker["TRAINING_SETTINGS"] | {
        "steps": args.steps,
        "threads": args.threads,
    }
    runs = []
    for seed in SEEDS:
        for selection in experiment["selections"]:
            name = selection["selector"]
            directory = workdir / name / f"seed{seed}"
            job = {
                "training": settings | {"seed": seed},
                "species": experiment["species"],
                "train_file": str((ROOT / selection["train_file"]).resolve()),
                "holdout_file": str((ROOT / experiment["holdout_file"]).resolve()),
                "train_ids": selection["selected_ids"],
                "holdout_ids": experiment["holdout_ids"],
                "train_parent_ids": [
                    experiment["split_manifest"][rid]["parent_id"]
                    for rid in selection["selected_ids"]
                ],
                "holdout_parent_ids": [
                    experiment["split_manifest"][rid]["parent_id"]
                    for rid in experiment["holdout_ids"]
                ],
            }
            worker["validate_job"](job)
            print(
                f"Training {name}, seed {seed}, {selection['training_atoms']} atoms",
                flush=True,
            )
            measured, reused = train_job(job, directory, args.mace_python.resolve())
            final = metrics(holdout, measured.pop("predictions"))
            initial = metrics(holdout, measured.pop("initial_predictions"))
            force = final["aggregate"]["forces"]["rmse"]
            if measured["processed_atoms"] != args.steps * selection["training_atoms"]:
                raise ValueError("processed atom count differs from frozen job")
            runs.append(
                {
                    "selector": name,
                    "seed": seed,
                    "training_configurations": selection["training_configurations"],
                    "training_atoms": selection["training_atoms"],
                    "run_directory": relative(directory),
                    "reused_checkpoint": reused,
                    "worker": measured,
                    "holdout": final,
                    "initial_holdout": initial,
                    "beats_zero_force": force
                    < experiment["holdout_zero_force_rmse_eV_per_Ang"],
                    "beats_initial_force_rmse": force
                    < initial["aggregate"]["forces"]["rmse"],
                }
            )
            write_json(
                workdir / "progress.json", {"experiment": experiment, "runs": runs}
            )
            print(
                json.dumps(
                    {
                        "selector": name,
                        "seed": seed,
                        "force_rmse": force,
                        "fit_seconds": measured["fit_seconds"],
                        "reused": reused,
                    }
                ),
                flush=True,
            )
    summarize = runpy.run_path(str(SUMMARY_HELPER))["summarize"]
    result = {
        "status": "actual_from_scratch_periodic_mace_training_comparison",
        "experiment": experiment,
        "training": settings,
        "model_seeds": list(SEEDS),
        "runs": runs,
        "summary": summarize(runs),
        "script_sha256": digest(Path(__file__)),
        "worker_sha256": digest(WORKER),
        "summary_helper_sha256": digest(SUMMARY_HELPER),
        "invocation_elapsed_seconds": time.perf_counter() - started,
        "models_reused_from_verified_checkpoints": sum(
            r["reused_checkpoint"] for r in runs
        ),
        "optimizer_fit_seconds_total": sum(r["worker"]["fit_seconds"] for r in runs),
        "limitations": [
            (
                "Small ordered native OMat24 sample; internal "
                "parent-disjoint holdout, not official test data."
            ),
            (
                "Periodic geometry/species model has no explicit magnetic-state "
                "conditioning; stress is not trained or evaluated."
            ),
            (
                "Large source force outliers are retained; "
                "the train/holdout force scales differ substantially."
            ),
            (
                "Fixed full-batch epochs do not establish "
                "convergence; no holdout checkpoint or setting selection."
            ),
            (
                "Three model seeds on fixed selections "
                "do not measure random-selection or dataset-split variance."
            ),
            (
                "Complete coarsened-class/presence coverage does not preserve "
                "every detailed token or guarantee accuracy."
            ),
        ],
    }
    write_json(args.output, result)
    write_json(workdir / "results.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
