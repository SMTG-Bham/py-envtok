"""Train matched fresh CPU MACE models on a split of the native OMol sample.

Selection is fitted after a frozen composition-grouped development split.
The experiment compares full training, two cover resolutions, and random
selection at the smaller cover's achieved atom ceiling. No model is downloaded.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from envtok.molecular.benchmarks import (
    Prediction,
    _random_subset,
    _stratum,
    evaluate_predictions,
)
from envtok.molecular.data import export_records, iter_extxyz
from envtok.molecular.pipeline import MolecularPipeline, audit_selection
from envtok.records import Split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/omol-compression-demo/data/sample.extxyz"
NAMESPACE = "OMol25:native-4M"
SPLIT_SEED = 20261105
SEEDS = (0, 1, 2)
MAX_RECORDS = 1024
FORCE_RATIO_LIMIT = 1.10
WORKER = ROOT / "build_tools/mace_compression_worker.py"
SPLIT_HELPER = ROOT / "build_tools/benchmark_molecular_accuracy.py"


def digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON, creating the requested output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def relative(path: Path) -> str:
    """Use repository-relative paths in portable result snapshots when possible."""
    return (
        str(path.resolve().relative_to(ROOT))
        if path.is_relative_to(ROOT)
        else str(path)
    )


def prepare(  # noqa: PLR0915 - explicit split and selection stages
    source: Path, workdir: Path
) -> tuple[dict[str, Any], list[Any]]:
    """Freeze the split before fitting any selection targets or geometry regions."""
    source_records = list(
        islice(iter_extxyz(source, label_namespace=NAMESPACE), MAX_RECORDS + 1)
    )
    if len(source_records) > MAX_RECORDS:
        raise ValueError("training demonstration is bounded to 1024 source records")
    helper = runpy.run_path(str(SPLIT_HELPER))
    prepared = helper["split_neutral_pool"](source_records, seed=SPLIT_SEED)
    workdir.mkdir(parents=True, exist_ok=True)
    pool = workdir / "neutral-pool.extxyz"
    export_records(prepared.records, pool, strict_training_export=True)
    selections = []
    pipelines = []
    for cap in (16, 64):
        pipeline = MolecularPipeline(
            {
                "profile": "molecular_v1",
                "source": {"path": str(pool.resolve()), "label_namespace": NAMESPACE},
                "state": {"unknown_policy": "reject", "strict_training_export": True},
                "chemistry": {"allow_validated_perception": False},
                "geometry": {"backend": "soap", "species": [1, 6, 7, 8]},
                "hierarchy": {"min_source_support": 5},
                "selection": {
                    "mode": "cover",
                    "prototype_cap": cap,
                    "prototype_region_radius": 0.20,
                    "reservoir_size": 256,
                },
                "execution": {"workdir": str((workdir / "selection").resolve())},
            }
        )
        cache_hit = (pipeline.feature_directory / "commit.json").exists()
        started = time.perf_counter()
        result = pipeline.select()
        elapsed = time.perf_counter() - started
        coverage = json.loads((pipeline.directory / "coverage.json").read_text())
        selections.append(
            {
                "selector": f"envtok_cap{cap}",
                "selected_ids": list(result.selected_ids),
                "coverage": coverage,
                "selection_seconds": elapsed,
                "feature_cache_hit": cache_hit,
                "selection_directory": relative(pipeline.directory),
            }
        )
        pipelines.append(pipeline)
    pipeline = pipelines[0]
    records = pipeline.ingest().records
    train = {record.record_id: record for record in pipeline.ingest().candidates}
    holdout = [record for record in records if record.split is Split.VALIDATION]
    holdout_ids = {record.record_id for record in holdout}
    train_formulas = {record.atoms.get_chemical_formula() for record in train.values()}
    holdout_formulas = {record.atoms.get_chemical_formula() for record in holdout}
    if train_formulas & holdout_formulas or set(train) & holdout_ids:
        raise ValueError("training and held-out compositions must be disjoint")
    candidates, mandatory, _, _ = pipeline.build_targets()
    stratified = tuple(
        replace(row, group_id=_stratum(train[row.record_id])) for row in candidates
    )
    budget = int(selections[0]["coverage"]["achieved_cost"])
    # Fix selection seed separately from the model seeds, isolating model variation.
    random_ids = _random_subset(stratified, budget, seed=0)
    selections.append(
        {
            "selector": "stratified_random",
            "selection_seed": 0,
            "selected_ids": list(random_ids),
            "coverage": audit_selection(
                candidates, mandatory, random_ids, budget=budget
            ),
        }
    )
    full_ids = sorted(train)
    selections.insert(
        0,
        {
            "selector": "full_pool",
            "selected_ids": full_ids,
            "coverage": audit_selection(candidates, mandatory, full_ids),
        },
    )
    species = sorted({int(z) for r in train.values() for z in r.atoms.numbers})
    holdout_file = workdir / "holdout.extxyz"
    export_records(holdout, holdout_file, strict_training_export=True)
    for selection in selections:
        ids = selection["selected_ids"]
        if not ids or set(ids) - set(train) or set(ids) & holdout_ids:
            raise ValueError("selection must be a nonempty subset of training only")
        chosen = [train[rid] for rid in ids]
        counts = np.asarray(
            [[np.count_nonzero(r.atoms.numbers == z) for z in species] for r in chosen]
        )
        if np.linalg.matrix_rank(counts) != len(species):
            raise ValueError(
                "selected elemental reference-energy fit is rank deficient"
            )
        train_file = workdir / selection["selector"] / "train.extxyz"
        train_file.parent.mkdir(parents=True, exist_ok=True)
        export_records(chosen, train_file, strict_training_export=True)
        selection.update(
            train_file=relative(train_file),
            train_sha256=digest(train_file),
            training_atoms=sum(r.natoms for r in chosen),
            training_configurations=len(chosen),
        )
    original_ids = {
        record.record_id: record.atoms.info["envtok_record_id"] for record in records
    }
    experiment = {
        "source": relative(source),
        "source_sha256": digest(source),
        "source_records": len(source_records),
        "label_namespace": NAMESPACE,
        "neutral_singlet_records": len(records),
        "training_configurations": len(train),
        "training_atoms": sum(r.natoms for r in train.values()),
        "holdout_configurations": len(holdout),
        "holdout_atoms": sum(r.natoms for r in holdout),
        "training_compositions": len(train_formulas),
        "holdout_compositions": len(holdout_formulas),
        "split_seed": SPLIT_SEED,
        "split_manifest": {r.record_id: r.split.value for r in records},
        "source_id_mapping": original_ids,
        "species": species,
        "holdout_file": relative(holdout_file),
        "holdout_sha256": digest(holdout_file),
        "holdout_ids": [r.record_id for r in holdout],
        "holdout_zero_force_rmse_eV_per_Ang": float(
            np.sqrt(
                np.mean(
                    np.concatenate(
                        [r.labels["forces"].value.reshape(-1) for r in holdout]
                    )
                    ** 2
                )
            )
        ),
        "checks": {
            "composition_disjoint": True,
            "selection_train_only": True,
            "element_count_rank_full_in_all_selections": True,
        },
        "software": pipeline.software,
        "selections": selections,
        "split_helper_sha256": digest(SPLIT_HELPER),
    }
    return experiment, holdout


def metrics(holdout: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score fixed predictions, including energy errors normalized per atom."""
    predictions = {
        row["record_id"]: Prediction(
            row["record_id"], row["energy"], np.asarray(row["forces"])
        )
        for row in rows
    }
    scored = evaluate_predictions(holdout, predictions)
    errors = np.asarray(
        [
            (predictions[r.record_id].energy - float(r.labels["energy"].value))
            / r.natoms
            for r in holdout
        ]
    )
    scored["aggregate"]["energy_per_atom"] = {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "unit": "eV/atom",
    }
    return scored


def train_job(
    job: dict[str, Any], directory: Path, mace_python: Path
) -> dict[str, Any]:
    """Run sequentially; reuse a result only when job, code and inputs match."""
    directory.mkdir(parents=True, exist_ok=True)
    job_path = directory / "job.json"
    result_path = directory / "worker-results.json"
    stamp = directory / "completed.json"
    expected = {
        "job": job,
        "worker_sha256": digest(WORKER),
        "train_sha256": digest(Path(job["train_file"])),
        "holdout_sha256": digest(Path(job["holdout_file"])),
        "python": str(mace_python.resolve()),
    }
    if stamp.exists() and result_path.exists():
        saved = json.loads(stamp.read_text())
        checkpoint = directory / "model-state.pt"
        if (
            saved.get("inputs") == expected
            and checkpoint.exists()
            and saved.get("result_sha256") == digest(result_path)
            and saved.get("checkpoint_sha256") == digest(checkpoint)
        ):
            return json.loads(result_path.read_text())
    write_json(job_path, job)
    environment = os.environ | {
        "MPLCONFIGDIR": str((directory / "matplotlib").resolve()),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    with (directory / "worker.log").open("w") as log:
        subprocess.run(  # noqa: S603 - explicit local interpreter, no shell
            [str(mace_python), str(WORKER), "--worker", str(job_path.resolve())],
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=3600,
        )
    measured = json.loads(result_path.read_text())
    write_json(
        stamp,
        {
            "inputs": expected,
            "result_sha256": digest(result_path),
            "checkpoint_sha256": digest(directory / "model-state.pt"),
        },
    )
    return measured


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize three paired initializations without choosing a winning seed."""
    full = {r["seed"]: r for r in runs if r["selector"] == "full_pool"}
    summary = []
    for name in dict.fromkeys(r["selector"] for r in runs):
        rows = [r for r in runs if r["selector"] == name]
        force = [r["holdout"]["aggregate"]["forces"]["rmse"] for r in rows]
        energy = [r["holdout"]["aggregate"]["energy_per_atom"]["rmse"] for r in rows]
        ratios = [
            r["holdout"]["aggregate"]["forces"]["rmse"]
            / full[r["seed"]]["holdout"]["aggregate"]["forces"]["rmse"]
            for r in rows
        ]
        seconds = [r["worker"]["fit_seconds"] for r in rows]
        summary.append(
            {
                "selector": name,
                "seeds": [r["seed"] for r in rows],
                "training_atoms": rows[0]["training_atoms"],
                "training_configurations": rows[0]["training_configurations"],
                "atoms_removed_fraction": 1
                - rows[0]["training_atoms"] / full[rows[0]["seed"]]["training_atoms"],
                "force_rmse_mean": float(np.mean(force)),
                "force_rmse_std": float(np.std(force, ddof=1)),
                "energy_per_atom_rmse_mean": float(np.mean(energy)),
                "energy_per_atom_rmse_std": float(np.std(energy, ddof=1)),
                "paired_force_rmse_ratios_to_full": ratios,
                "all_seeds_within_10_percent_of_full_force_rmse": all(
                    r <= FORCE_RATIO_LIMIT for r in ratios
                ),
                "all_seeds_beat_zero_force": all(r["beats_zero_force"] for r in rows),
                "fit_seconds_median": float(np.median(seconds)),
                "fit_speedup_median": float(
                    np.median(
                        [
                            full[r["seed"]]["worker"]["fit_seconds"]
                            / r["worker"]["fit_seconds"]
                            for r in rows
                        ]
                    )
                ),
            }
        )
    return summary


def main() -> None:
    """Run the frozen split, coverage selections, and matched training jobs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument(
        "--workdir", type=Path, default=ROOT / "runs/omol-compression-training"
    )
    parser.add_argument("--mace-python", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples/molecular/omol/compression-demo/training-results.json",
    )
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    started = time.perf_counter()
    workdir = args.workdir.resolve()
    worker = runpy.run_path(str(WORKER))
    settings = worker["TRAINING_SETTINGS"] | {"steps": args.steps}
    experiment, holdout = prepare(args.source.resolve(), workdir)
    write_json(workdir / "experiment.json", experiment)
    print(
        json.dumps(
            {
                k: experiment[k]
                for k in (
                    "training_configurations",
                    "holdout_configurations",
                    "training_atoms",
                )
            }
        ),
        flush=True,
    )
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
            }
            worker["validate_job"](job)
            print(
                f"Training {name}, seed {seed}, {selection['training_atoms']} atoms",
                flush=True,
            )
            measured = train_job(job, directory, args.mace_python.resolve())
            final = metrics(holdout, measured.pop("predictions"))
            initial = metrics(holdout, measured.pop("initial_predictions"))
            force = final["aggregate"]["forces"]["rmse"]
            result = {
                "selector": name,
                "seed": seed,
                "training_configurations": selection["training_configurations"],
                "training_atoms": selection["training_atoms"],
                "run_directory": relative(directory),
                "worker": measured,
                "holdout": final,
                "initial_holdout": initial,
                "beats_zero_force": force
                < experiment["holdout_zero_force_rmse_eV_per_Ang"],
                "beats_initial_force_rmse": force
                < initial["aggregate"]["forces"]["rmse"],
            }
            if measured["processed_atoms"] != args.steps * selection["training_atoms"]:
                raise ValueError("worker processed atom count differs from frozen job")
            runs.append(result)
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
                    }
                ),
                flush=True,
            )
    result = {
        "status": "actual_from_scratch_mace_training_comparison",
        "experiment": experiment,
        "training": settings,
        "model_seeds": list(SEEDS),
        "runs": runs,
        "summary": summarize(runs),
        "script_sha256": digest(Path(__file__)),
        "worker_sha256": digest(WORKER),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            (
                "Small ordered native CHNO sample, neutral "
                "singlets only; one composition-grouped development split."
            ),
            (
                "No verified reaction/trajectory-family mapping "
                "and no official test-set evaluation."
            ),
            (
                "Tiny 1512-parameter model, fixed epochs; convergence "
                "and simulation accuracy are not established."
            ),
            (
                "Equal full-batch epochs; selected data change "
                "processed atoms and compute, not optimizer updates."
            ),
            (
                "Three model seeds on fixed "
                "selections; random-selection variance is not measured."
            ),
            (
                "Coverage is of modeled targets, not "
                "every source geometry; inspect prototype overflow."
            ),
        ],
    }
    write_json(args.output, result)
    write_json(workdir / "results.json", result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
