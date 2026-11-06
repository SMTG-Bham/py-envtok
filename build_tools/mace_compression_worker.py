"""Train one fresh CPU MACE model for the bounded OMol compression experiment.

Run in an existing MACE environment with ``--worker /absolute/path/job.json``.
The model uses only positions and elements; input structures must therefore be
neutral singlets. Holdout geometry is loaded only after the final checkpoint is
fixed. The script never downloads pretrained weights or tunes on holdout errors.
"""

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

MAX_ATOMIC_NUMBER = 118
MAX_RECORDS = 1024
MAX_STEPS = 10000
MAX_SEED = 2**32 - 1
TRAINING_SETTINGS = {
    "seed": 0,
    "steps": 1000,
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
    """Hash an input, source file or checkpoint."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write portable finite JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_job(job: dict[str, Any]) -> None:
    """Reject split leakage, nonmatching architectures and unbounded workloads."""
    train_ids, holdout_ids = job["train_ids"], job["holdout_ids"]
    ids = [*train_ids, *holdout_ids]
    if (
        not train_ids
        or not holdout_ids
        or any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
        or len(ids) > MAX_RECORDS
    ):
        raise ValueError("job needs disjoint unique train/holdout IDs within the cap")
    settings = job["training"]
    expected = TRAINING_SETTINGS | {
        "seed": settings.get("seed"),
        "steps": settings.get("steps"),
    }
    if settings != expected:
        raise ValueError("all compared jobs must use the fixed training settings")
    if (
        type(settings["seed"]) is not int
        or not 0 <= settings["seed"] <= MAX_SEED
        or type(settings["steps"]) is not int
        or not 1 <= settings["steps"] <= MAX_STEPS
    ):
        raise ValueError("seed and steps must be bounded integers")
    species = job["species"]
    if (
        not isinstance(species, list)
        or not species
        or any(
            type(number) is not int or not 1 <= number <= MAX_ATOMIC_NUMBER
            for number in species
        )
        or species != sorted(set(species))
    ):
        raise ValueError("species must be sorted unique atomic numbers")
    if any(not Path(job[key]).is_absolute() for key in ("train_file", "holdout_file")):
        raise ValueError("training and holdout files must use absolute paths")


def validate_frames(
    frames: list[Any], ids: list[str], species: list[int], *, labels: bool
) -> None:
    """Validate state and coordinates, inspecting labels only for training."""
    if [atoms.info.get("envtok_record_id") for atoms in frames] != ids:
        raise ValueError("file IDs differ from the frozen job")
    for atoms in frames:
        if (
            not len(atoms)
            or atoms.pbc.any()
            or not set(atoms.numbers) <= set(species)
            or not np.isfinite(atoms.positions).all()
            or not np.isfinite(atoms.cell.array).all()
        ):
            raise ValueError("requires finite nonperiodic structures in fitted species")
        if (
            atoms.info.get("total_charge") != 0
            or atoms.info.get("spin_multiplicity") != 1
        ):
            raise ValueError("geometry-only MACE comparison requires neutral singlets")
        if labels:
            energy, forces = (
                np.asarray(atoms.get_potential_energy()),
                atoms.get_forces(),
            )
            if (
                energy.shape != ()
                or forces.shape != (len(atoms), 3)
                or not np.isfinite(energy).all()
                or not np.isfinite(forces).all()
            ):
                raise ValueError("training energy and force labels must be finite")


def error_metrics(energy_error: Any, force_error: Any, natoms: Any) -> dict[str, Any]:
    """Report energy errors per configuration and atom, and force components."""
    values = {
        "energy": (np.asarray(energy_error), "eV"),
        "energy_per_atom": (np.asarray(energy_error) / np.asarray(natoms), "eV/atom"),
        "forces": (np.asarray(force_error), "eV/Ang"),
    }
    return {
        key: {
            "mae": float(np.abs(value).mean()),
            "rmse": float(np.sqrt(np.square(value).mean())),
            "unit": unit,
        }
        for key, (value, unit) in values.items()
    }


def worker(job_path: Path) -> None:  # noqa: PLR0915 - explicit isolated training stages
    """Fit only training labels, freeze the final checkpoint, then predict holdout."""
    import torch  # noqa: PLC0415 - optional interpreter dependency

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
    validate_frames(train_atoms, job["train_ids"], species, labels=True)

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
    energies = np.asarray([atoms.get_potential_energy() for atoms in train_atoms])
    atomic_energies, _, rank, _ = np.linalg.lstsq(counts, energies, rcond=None)
    if rank != len(species):
        raise ValueError(
            "selected training compositions cannot determine atomic energies"
        )
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

    def training_metrics() -> dict[str, Any]:
        model.eval()
        batch = next(iter(loader))
        output = model(batch.to_dict(), training=False, compute_force=True)
        return error_metrics(
            (output["energy"] - batch.energy).detach().numpy(),
            (output["forces"] - batch.forces).detach().numpy(),
            (batch.ptr[1:] - batch.ptr[:-1]).numpy(),
        )

    initial_training_metrics = training_metrics()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    losses, energy_losses, force_losses = [], [], []
    fit_started = time.perf_counter()
    model.train()
    for step in range(settings["steps"]):
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
        energy_losses.append(float(energy_loss.detach()))
        force_losses.append(float(force_loss.detach()))
        if (step + 1) % 100 == 0 or step + 1 == settings["steps"]:
            print(
                f"step {step + 1}/{settings['steps']} loss={losses[-1]:.6g} "
                f"seconds={time.perf_counter() - fit_started:.1f}",
                flush=True,
            )
    fit_seconds = time.perf_counter() - fit_started
    checkpoint = job_path.parent / "model-state.pt"
    torch.save(model.state_dict(), checkpoint)
    final_training_metrics = training_metrics()
    # Held-out files are read only after the checkpoint has been fixed.
    holdout_atoms = read(job["holdout_file"], index=":")
    validate_frames(holdout_atoms, job["holdout_ids"], species, labels=False)
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
            "training_metrics": {
                "initial": initial_training_metrics,
                "final": final_training_metrics,
            },
            "training": settings,
            "train_sha256": digest(Path(job["train_file"])),
            "holdout_sha256": digest(Path(job["holdout_file"])),
            "job_sha256": digest(job_path),
            "worker_source_sha256": digest(Path(__file__)),
            "checkpoint_sha256": digest(checkpoint),
            "element_count_rank": int(rank),
            "atomic_energies": atomic_energies.tolist(),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "training_configurations": len(train_atoms),
            "training_atoms": sum(len(atoms) for atoms in train_atoms),
            "training_loss": losses,
            "energy_loss": energy_losses,
            "force_loss": force_losses,
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


def main() -> None:
    """Run one explicit bounded local training job."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worker", type=Path, required=True)
    args = parser.parse_args()
    worker(args.worker)


if __name__ == "__main__":
    main()
