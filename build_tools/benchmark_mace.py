"""Train four tiny, fresh CPU MACE models on the bundled neutral OMol screen.

Run in the envtok molecular environment, passing an existing MACE interpreter:
    python build_tools/benchmark_mace.py --mace-python /path/to/python

The optional worker imports MACE only in that interpreter. It never downloads a
model. Two hundred full-batch optimizer steps are a small screen, not convergence
or a matched-processed-atoms scientific comparison.
"""

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAX_RECORDS = 128
FORCE_RATIO_LIMIT = 1.10
TRAINING_SETTINGS = {
    "seed": 0,
    "steps": 200,
    "learning_rate": 0.01,
    "device": "cpu",
    "dtype": "float64",
    "threads": 1,
    "r_max": 4.5,
    "num_interactions": 1,
    "hidden_irreps": "8x0e",
    "MLP_irreps": "16x0e",
    "interaction": "RealAgnosticResidualInteractionBlock",
    "first_interaction": "RealAgnosticInteractionBlock",
    "gate": "silu",
    "max_ell": 1,
    "correlation": 2,
    "radial_MLP": [16, 16],
    "num_bessel": 6,
    "num_polynomial_cutoff": 5,
    "avg_num_neighbors": 10.0,
    "optimizer": "Adam",
    "loss": "energy_per_atom_mse + force_component_mse",
    "atomic_energies": "least_squares_selected_training_only",
    "checkpoint": "final_fixed_step_no_holdout_selection",
}


def digest(path: Path) -> str:
    """Hash an input, source file or model checkpoint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write finite, reproducible JSON syntax."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_job(job: dict[str, Any]) -> None:
    """Reject split leakage or settings that invalidate matched model comparisons."""
    train_ids, holdout_ids = job["train_ids"], job["holdout_ids"]
    if (
        not train_ids
        or not holdout_ids
        or len(train_ids) != len(set(train_ids))
        or len(holdout_ids) != len(set(holdout_ids))
        or set(train_ids) & set(holdout_ids)
        or len(train_ids) + len(holdout_ids) > MAX_RECORDS
    ):
        raise ValueError("job needs disjoint unique train/holdout IDs within the cap")
    if job["training"] != TRAINING_SETTINGS:
        raise ValueError("all compared jobs must use the fixed training settings")


def worker(job_path: Path) -> None:  # noqa: PLR0915 - small standalone training worker
    """Train exclusively on a job's training file, then predict the held-out file."""
    import torch  # noqa: PLC0415 - optional interpreter dependency

    # e3nn 0.4.4's installed tensor constants include ordinary slice objects.
    torch.serialization.add_safe_globals([slice])
    from ase.io import read  # noqa: PLC0415
    from e3nn import o3  # noqa: PLC0415
    from mace import data, modules, tools  # noqa: PLC0415
    from mace.data.atomic_data import get_data_loader  # noqa: PLC0415

    started = time.perf_counter()
    job = json.loads(job_path.read_text())
    validate_job(job)
    settings = job["training"]
    torch.set_num_threads(settings["threads"])
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(settings["seed"])
    np.random.seed(settings["seed"])  # noqa: NPY002 - MACE may use legacy RNG
    torch.use_deterministic_algorithms(True)
    species = job["species"]
    z_table = tools.AtomicNumberTable(species)
    train_atoms = read(job["train_file"], index=":")
    if not train_atoms or len(train_atoms) > MAX_RECORDS:
        raise ValueError("worker needs 1-128 training configurations")
    if [atoms.info["envtok_record_id"] for atoms in train_atoms] != job["train_ids"]:
        raise ValueError("training file IDs differ from the frozen job")

    def graph(atoms: Any, *, labels: bool) -> Any:
        properties = (
            {"energy": atoms.get_potential_energy(), "forces": atoms.get_forces()}
            if labels
            else {}
        )
        config = data.Configuration(
            atomic_numbers=atoms.numbers,
            positions=atoms.positions,
            properties=properties,
            property_weights={"energy": 1.0, "forces": 1.0},
            cell=atoms.cell.array,
            pbc=tuple(atoms.pbc),
        )
        return data.AtomicData.from_config(
            config, z_table=z_table, cutoff=settings["r_max"]
        )

    counts = np.asarray(
        [
            [np.count_nonzero(atoms.numbers == number) for number in species]
            for atoms in train_atoms
        ]
    )
    labels = np.asarray([atoms.get_potential_energy() for atoms in train_atoms])
    atomic_energies, _, rank, _ = np.linalg.lstsq(counts, labels, rcond=None)
    dataset = [graph(atoms, labels=True) for atoms in train_atoms]
    model = modules.MACE(
        r_max=settings["r_max"],
        num_bessel=settings["num_bessel"],
        num_polynomial_cutoff=settings["num_polynomial_cutoff"],
        max_ell=settings["max_ell"],
        interaction_cls=modules.interaction_classes[settings["interaction"]],
        interaction_cls_first=modules.interaction_classes[
            settings["first_interaction"]
        ],
        num_interactions=settings["num_interactions"],
        num_elements=len(species),
        hidden_irreps=o3.Irreps(settings["hidden_irreps"]),
        MLP_irreps=o3.Irreps(settings["MLP_irreps"]),
        atomic_energies=atomic_energies,
        avg_num_neighbors=settings["avg_num_neighbors"],
        atomic_numbers=species,
        correlation=settings["correlation"],
        gate=torch.nn.functional.silu,
        radial_MLP=settings["radial_MLP"],
    )
    loader = get_data_loader(dataset, batch_size=len(dataset), shuffle=False)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    losses = []
    fit_started = time.perf_counter()
    model.train()
    for _ in range(settings["steps"]):
        batch = next(iter(loader))
        optimizer.zero_grad()
        output = model(batch.to_dict(), training=True, compute_force=True)
        natoms = batch.ptr[1:] - batch.ptr[:-1]
        energy_loss = ((output["energy"] - batch.energy) / natoms).square().mean()
        force_loss = (output["forces"] - batch.forces).square().mean()
        loss = energy_loss + force_loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    fit_seconds = time.perf_counter() - fit_started
    checkpoint = job_path.parent / "model-state.pt"
    torch.save(model.state_dict(), checkpoint)
    # Load held-out geometry only after fitting and fixing the final checkpoint.
    holdout_atoms = read(job["holdout_file"], index=":")
    if [atoms.info["envtok_record_id"] for atoms in holdout_atoms] != job[
        "holdout_ids"
    ]:
        raise ValueError("holdout file IDs differ from the frozen job")
    model.eval()

    def predict_holdout() -> list[dict[str, Any]]:
        predictions = []
        for atoms in holdout_atoms:
            batch = next(
                iter(get_data_loader([graph(atoms, labels=False)], batch_size=1))
            )
            output = model(batch.to_dict(), training=False, compute_force=True)
            predictions.append(
                {
                    "record_id": atoms.info["envtok_record_id"],
                    "energy": float(output["energy"].detach().reshape(-1)[0]),
                    "forces": output["forces"].detach().numpy().tolist(),
                }
            )
        return predictions

    predictions = predict_holdout()
    model.load_state_dict(initial_state)
    initial_predictions = predict_holdout()
    write_json(
        job_path.parent / "worker-results.json",
        {
            "predictions": predictions,
            "initial_predictions": initial_predictions,
            "training": settings,
            "train_sha256": digest(Path(job["train_file"])),
            "holdout_sha256": digest(Path(job["holdout_file"])),
            "checkpoint_sha256": digest(checkpoint),
            "element_count_rank": int(rank),
            "atomic_energies": atomic_energies.tolist(),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "training_loss": losses,
            "processed_atoms": settings["steps"]
            * sum(len(atoms) for atoms in train_atoms),
            "fit_seconds": fit_seconds,
            "worker_seconds": time.perf_counter() - started,
            "software": {
                name: version(name)
                for name in ("mace-torch", "torch", "e3nn", "ase", "numpy")
            },
            "python": sys.version,
            "mace_source_sha256": digest(
                Path(sys.modules[modules.MACE.__module__].__file__)
            ),
        },
    )


def run_screen(workdir: Path, mace_python: Path) -> dict[str, Any]:  # noqa: PLR0915 - explicit experiment stages
    """Prepare a fixed grouped split and run matching optional MACE workers."""
    from envtok.molecular.benchmarks import (  # noqa: PLC0415 - core-side only
        Prediction,
        compare_selectors,
        evaluate_predictions,
    )
    from envtok.molecular.data import export_records, iter_extxyz  # noqa: PLC0415
    from envtok.molecular.pipeline import MolecularPipeline  # noqa: PLC0415
    from envtok.records import Split  # noqa: PLC0415

    started = time.perf_counter()
    helper = runpy.run_path(str(ROOT / "build_tools/benchmark_molecular_accuracy.py"))
    original = list(iter_extxyz(helper["SOURCE"], label_namespace=helper["NAMESPACE"]))
    if len(original) > MAX_RECORDS:
        raise ValueError("this screen is capped at 128 source configurations")
    prepared = helper["split_neutral_pool"](original)
    workdir.mkdir(parents=True, exist_ok=True)
    pool_path = workdir / "neutral-pool.extxyz"
    export_records(prepared.records, pool_path)
    pipeline = MolecularPipeline(
        {
            "profile": "molecular_v1",
            "source": {
                "path": str(pool_path.resolve()),
                "label_namespace": helper["NAMESPACE"],
            },
            "chemistry": {"allow_validated_perception": False},
            "geometry": {"backend": "soap"},
            "hierarchy": {"min_source_support": 5},
            "selection": {"mode": "budget", "atom_fraction": 0.5},
            "execution": {"workdir": str((workdir / "selection").resolve())},
            "reproducibility": {"seed": 0},
        }
    )
    comparisons = compare_selectors(pipeline, fractions=(0.5,), seeds=(0,))
    records = pipeline.ingest().records
    train = {record.record_id: record for record in pipeline.ingest().candidates}
    holdout = tuple(record for record in records if record.split is Split.VALIDATION)
    species = sorted(
        {int(z) for record in train.values() for z in record.atoms.numbers}
    )
    holdout_file = workdir / "holdout.extxyz"
    export_records(holdout, holdout_file)
    results = []
    for selection in [comparisons["full_pool"], *comparisons["runs"]]:
        name = selection["selector"]
        directory = workdir / name
        directory.mkdir(exist_ok=True)
        train_file = directory / "train.extxyz"
        chosen = [train[record_id] for record_id in selection["selected_ids"]]
        if not chosen:
            results.append(
                {
                    "selector": name,
                    "status": "empty_selection",
                    "selection": selection,
                    "holdout": None,
                }
            )
            continue
        export_records(chosen, train_file)
        job = {
            "training": TRAINING_SETTINGS,
            "species": species,
            "train_file": str(train_file.resolve()),
            "holdout_file": str(holdout_file.resolve()),
            "train_ids": [record.record_id for record in chosen],
            "holdout_ids": [record.record_id for record in holdout],
        }
        validate_job(job)
        job_path = directory / "job.json"
        write_json(job_path, job)
        environment = os.environ | {
            "MPLCONFIGDIR": str((workdir / "matplotlib").resolve()),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        with (directory / "worker.log").open("w") as log:
            subprocess.run(  # noqa: S603 - explicit local Python and script, no shell
                [
                    str(mace_python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    str(job_path.resolve()),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                timeout=300,
            )
        measured = json.loads((directory / "worker-results.json").read_text())
        metrics = {}
        for key in ("predictions", "initial_predictions"):
            predictions = {
                row["record_id"]: Prediction(
                    row["record_id"], row["energy"], np.asarray(row["forces"])
                )
                for row in measured.pop(key)
            }
            metrics[key] = evaluate_predictions(holdout, predictions)
        result = {
            "selector": name,
            "status": "evaluated_fixed_step_unconverged",
            "selection": selection,
            "worker": measured,
            "holdout": metrics["predictions"],
            "initial_holdout": metrics["initial_predictions"],
        }
        results.append(result)
        print(name, result["holdout"]["aggregate"], flush=True)
    reference = results[0]["holdout"]["aggregate"]["forces"]["rmse"]
    zero_force_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.concatenate(
                        [
                            record.labels["forces"].value.reshape(-1)
                            for record in holdout
                        ]
                    )
                )
            )
        )
    )
    for result in results:
        final_rmse = (
            result["holdout"]["aggregate"]["forces"]["rmse"]
            if result["holdout"]
            else None
        )
        initial_rmse = (
            result["initial_holdout"]["aggregate"]["forces"]["rmse"]
            if result["holdout"]
            else None
        )
        result["force_rmse_ratio_to_zero"] = (
            final_rmse / zero_force_rmse
            if final_rmse is not None and zero_force_rmse
            else None
        )
        result["beats_zero_force"] = bool(
            final_rmse is not None and final_rmse < zero_force_rmse
        )
        result["beats_initial_model"] = bool(
            final_rmse is not None
            and initial_rmse is not None
            and final_rmse < initial_rmse
        )
    for result in results[1:]:
        ratio = (
            result["holdout"]["aggregate"]["forces"]["rmse"] / reference
            if result["holdout"] and reference
            else None
        )
        result["force_rmse_ratio_to_full"] = ratio
        result["engineering_screen_pass"] = bool(
            result["selection"]["complete"]
            and ratio is not None
            and ratio <= FORCE_RATIO_LIMIT
        )
    split_manifest = {
        record.atoms.info["envtok_record_id"]: record.split.value for record in records
    }
    return {
        "status": "actual_from_scratch_mace_engineering_screen",
        "accuracy_claim_supported": False,
        "source_sha256": digest(helper["SOURCE"]),
        "script_sha256": digest(Path(__file__)),
        "split_helper_sha256": digest(
            ROOT / "build_tools/benchmark_molecular_accuracy.py"
        ),
        "split_sha256": hashlib.sha256(
            json.dumps(split_manifest, sort_keys=True).encode()
        ).hexdigest(),
        "model_config_sha256": hashlib.sha256(
            json.dumps(TRAINING_SETTINGS, sort_keys=True).encode()
        ).hexdigest(),
        "selection_software": pipeline.software,
        "training": TRAINING_SETTINGS,
        "source_records": len(original),
        "neutral_singlet_records": len(prepared.records),
        "train_configurations": len(train),
        "holdout_configurations": len(holdout),
        "split_seed": 20261105,
        "selection_seed": 0,
        "split_manifest": split_manifest,
        "source_id_mapping": {
            record.record_id: record.atoms.info["envtok_record_id"]
            for record in records
        },
        "holdout_ids": [record.record_id for record in holdout],
        "holdout_zero_force_rmse_eV_per_Ang": zero_force_rmse,
        "runs": results,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Only 21 neutral singlets and four development holdouts.",
            "Two hundred fixed steps do not establish convergence or accuracy.",
            "Equal optimizer updates; processed atoms and compute differ by subset.",
            "No mixed-state, torsion, interaction-curve or dynamics qualification.",
        ],
    }


def main() -> None:
    """Dispatch to the optional worker or run the complete bounded experiment."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--mace-python", type=Path)
    parser.add_argument("--workdir", type=Path, default=ROOT / "runs/mace-small")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples/molecular/omol/mace-small-results.json",
    )
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.mace_python is None:
        parser.error("--mace-python must identify an existing MACE Python environment")
    result = run_screen(args.workdir, args.mace_python)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
