"""Reproducible selector comparisons and explicitly paired model error metrics.

Coverage comparisons are engineering diagnostics. They do not constitute the
scientific model-training pilot and report accuracy as pending.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from envtok.errors import ConfigError
from envtok.molecular.data import export_records
from envtok.molecular.geometry import nearest_selected_distances
from envtok.molecular.records import MolecularRecord
from envtok.molecular.selection import (
    Candidate,
    Target,
    select_candidates,
)

if TYPE_CHECKING:
    from envtok.molecular.pipeline import MolecularPipeline


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _size_bucket(natoms: int) -> str:
    for upper in (10, 30, 60):
        if natoms < upper:
            return f"<{upper}"
    return ">=60"


def _stratum(record: MolecularRecord) -> str:
    return json.dumps(
        (record.state_key, _size_bucket(record.natoms)), separators=(",", ":")
    )


def _random_subset(
    candidates: Sequence[Candidate], budget: int, seed: int
) -> tuple[str, ...]:
    buckets: dict[str, list[Candidate]] = defaultdict(list)
    for row in candidates:
        buckets[row.group_id or row.record_id].append(row)

    def order(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()

    for group, rows in buckets.items():
        buckets[group] = sorted(rows, key=lambda row: order(row.record_id))
    groups = sorted(buckets, key=order)
    selected: list[str] = []
    cost = 0.0
    depth = 0
    while any(depth < len(rows) for rows in buckets.values()):
        for group in groups:
            if depth >= len(buckets[group]):
                continue
            row = buckets[group][depth]
            if cost + row.cost <= budget:
                selected.append(row.record_id)
                cost += row.cost
        depth += 1
    return tuple(sorted(selected))


def _write_run(
    directory: Path,
    selected_ids: Sequence[str],
    records: Mapping[str, MolecularRecord],
    report: dict[str, Any],
    *,
    export_subsets: bool,
    strict_export: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "selected_ids.txt").write_text(
        "".join(f"{rid}\n" for rid in selected_ids)
    )
    (directory / "manifest.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    if export_subsets:
        export_records(
            [records[rid] for rid in selected_ids],
            directory / "subset.extxyz",
            strict_training_export=strict_export,
        )


def compare_selectors(
    pipeline: MolecularPipeline,
    *,
    fractions: Sequence[float] = (0.25,),
    seeds: Sequence[int] = (0,),
    export_subsets: bool = False,
) -> dict[str, Any]:
    """Compare stratified random, token-only and combined at matched atom ceilings.

    All methods have the same state/namespace/size strata. The token-only method
    uses a chemical core followed by random background to fill its atom budget.
    Every run reports chemical violations, achieved atoms and selected IDs; the
    full training pool is included separately as the model reference. Defaults
    run one 25% budget and one seed; larger sweeps are opt-in. This writes
    reproducible manifests, optionally materializing intact training subsets.
    """
    from envtok.molecular.pipeline import audit_selection  # noqa: PLC0415 - avoid pipeline cycle

    if not fractions or any(
        not math.isfinite(value) or not 0 < value <= 1 for value in fractions
    ):
        raise ConfigError("benchmark fractions must lie in (0, 1]")
    if not seeds or any(not _nonnegative_integer(seed) for seed in seeds):
        raise ConfigError("benchmark seeds must be nonnegative integers")
    original, mandatory, hierarchy, regions = pipeline.build_targets()
    records = {record.record_id: record for record in pipeline.ingest().candidates}
    candidates = tuple(
        replace(
            row,
            group_id=_stratum(records[row.record_id]),
            refinement_targets=regions.targets[row.record_id],
        )
        for row in original
    )
    strata = {str(row.group_id) for row in candidates}
    protected = {f"benchmark_stratum:{group}": Target() for group in strata}
    candidates = tuple(
        replace(row, targets=row.targets | {f"benchmark_stratum:{row.group_id}"})
        for row in candidates
    )
    required = mandatory | protected
    total_atoms = sum(records[row.record_id].natoms for row in candidates)
    descriptors = pipeline.descriptor_rows(hierarchy)
    root = pipeline.directory / "benchmarks"
    strict_export = bool(pipeline.settings["state"]["strict_training_export"])
    full_ids = tuple(sorted(records))
    full = audit_selection(candidates, required, full_ids, budget=total_atoms)
    full.update({"selector": "full_pool", "fraction": 1.0, "selected_ids": full_ids})
    _write_run(
        root / "full_pool",
        full_ids,
        records,
        full,
        export_subsets=export_subsets,
        strict_export=strict_export,
    )
    runs: list[dict[str, Any]] = []
    for fraction in fractions:
        budget = math.floor(total_atoms * fraction)
        for seed in seeds:
            for selector in ("stratified_random", "token_only", "combined"):
                start = time.perf_counter()
                if selector == "stratified_random":
                    ids = _random_subset(candidates, budget, seed)
                else:
                    pool = (
                        candidates
                        if selector == "combined"
                        else tuple(
                            replace(row, refinement_targets=frozenset())
                            for row in candidates
                        )
                    )
                    reserve = (
                        pipeline.settings["selection"]["background_reserve_fraction"]
                        if selector == "combined"
                        else 1.0
                    )
                    ids = select_candidates(
                        pool,
                        required,
                        mode="budget",
                        budget=budget,
                        background_reserve_fraction=reserve,
                        seed=seed,
                    ).selected_ids
                elapsed = time.perf_counter() - start
                report = audit_selection(candidates, required, ids, budget=budget)
                report.update(
                    {
                        "selector": selector,
                        "fraction": fraction,
                        "selection_seed": seed,
                        "selected_ids": ids,
                        "selection_seconds": elapsed,
                        "protected_strata_met": not (
                            set(report["unmet_constraints"]) & protected.keys()
                        ),
                        "nearest_selected": nearest_selected_distances(
                            descriptors, ids, seed=seed
                        ),
                    }
                )
                key = f"{selector}-fraction{fraction:g}-seed{seed}"
                _write_run(
                    root / key,
                    ids,
                    records,
                    report,
                    export_subsets=export_subsets,
                    strict_export=strict_export,
                )
                runs.append(report)
    result = {
        "status": "model_accuracy_pending",
        "explanation": (
            "Selector diagnostics only. Train matched models and evaluate held-out "
            "predictions before claiming accuracy retention or scientific compression."
        ),
        "source_sha256": pipeline.source_hash,
        "pipeline_fingerprint": pipeline.fingerprint,
        "budget_unit": "atoms_including_hydrogen",
        "token_only_budget_fill": "chemical_core_then_stratified_background",
        "state_and_size_strata": sorted(strata),
        "model_seeds": None,
        "full_pool": full,
        "runs": runs,
    }
    (root / "comparison.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


@dataclass(frozen=True, slots=True)
class Prediction:
    """Model outputs in eV and eV/Ang, keyed to original source atom order."""

    record_id: str
    energy: float
    forces: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _Error:
    record: MolecularRecord
    energy: float
    forces: npt.NDArray[np.float64]


def _errors(
    records: Sequence[MolecularRecord], predictions: Mapping[str, Prediction]
) -> list[_Error]:
    ids = [record.record_id for record in records]
    if len(set(ids)) != len(ids) or set(ids) != predictions.keys():
        raise ConfigError(
            "prediction IDs must exactly match unique reference record IDs"
        )
    errors: list[_Error] = []
    for record in records:
        prediction = predictions[record.record_id]
        if prediction.record_id != record.record_id:
            raise ConfigError("prediction record_id does not match its mapping key")
        if "energy" not in record.labels or "forces" not in record.labels:
            raise ConfigError(
                "prediction evaluation requires reference energy and forces"
            )
        energy = record.labels["energy"]
        forces = record.labels["forces"]
        if energy.unit != "eV" or forces.unit != "eV/Ang":
            raise ConfigError("evaluation references must use eV and eV/Ang")
        predicted_energy = np.asarray(prediction.energy, dtype=float)
        predicted_forces = np.asarray(prediction.forces, dtype=float)
        if (
            energy.value.shape != ()
            or predicted_energy.shape != ()
            or forces.value.shape != (record.natoms, 3)
            or predicted_forces.shape != (record.natoms, 3)
        ):
            raise ConfigError("energy/force shapes must match original atom ordering")
        if not all(
            np.isfinite(array).all()
            for array in (
                energy.value,
                forces.value,
                predicted_energy,
                predicted_forces,
            )
        ):
            raise ConfigError("reference and prediction labels must be finite")
        errors.append(
            _Error(
                record,
                float(predicted_energy) - float(energy.value),
                predicted_forces - forces.value,
            )
        )
    return errors


def _metrics(errors: Sequence[_Error]) -> dict[str, Any]:
    energy = np.asarray([error.energy for error in errors])
    forces = np.concatenate([error.forces for error in errors])
    vector_errors = np.linalg.norm(forces, axis=1)
    return {
        "configurations": len(errors),
        "atoms": len(forces),
        "energy": {
            "mae": float(np.abs(energy).mean()),
            "rmse": float(np.sqrt(np.square(energy).mean())),
            "unit": "eV",
        },
        "forces": {
            "mae": float(np.abs(forces).mean()),
            "rmse": float(np.sqrt(np.square(forces).mean())),
            "p95_vector_error": float(np.quantile(vector_errors, 0.95)),
            "p99_vector_error": float(np.quantile(vector_errors, 0.99)),
            "unit": "eV/Ang",
        },
    }


def _stratified_metrics(errors: Sequence[_Error], attribute: str) -> dict[str, Any]:
    groups: dict[str, list[_Error]] = defaultdict(list)
    for error in errors:
        value = (
            _size_bucket(error.record.natoms)
            if attribute == "size"
            else getattr(error.record, attribute)
        )
        groups["unknown" if value is None else str(value)].append(error)
    return {key: _metrics(group) for key, group in sorted(groups.items())}


def evaluate_predictions(
    records: Sequence[MolecularRecord],
    predictions: Mapping[str, Prediction],
    *,
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Score matched energy/force predictions and optional family bootstrap CIs.

    References must have normalized units and predictions must retain source atom
    ordering. Energy errors are per configuration; force MAE/RMSE are component
    errors, while force tails use per-atom error vector norms. Family bootstrap
    resamples parent families, preserving all their frames together.
    """
    if not records:
        raise ConfigError("prediction evaluation needs at least one reference record")
    if not _nonnegative_integer(bootstrap_samples):
        raise ConfigError("bootstrap_samples must be a nonnegative integer")
    errors = _errors(records, predictions)
    result = {
        "status": "predictions_evaluated",
        "aggregate": _metrics(errors),
        "by_charge": _stratified_metrics(errors, "total_charge"),
        "by_multiplicity": _stratified_metrics(errors, "spin_multiplicity"),
        "by_size": _stratified_metrics(errors, "size"),
        "by_label_namespace": _stratified_metrics(errors, "label_namespace"),
    }
    if bootstrap_samples:
        groups: dict[str, list[_Error]] = defaultdict(list)
        for error in errors:
            groups[error.record.family_id or error.record.record_id].append(error)
        keys = sorted(groups)
        rng = np.random.default_rng(seed)
        sampled: list[tuple[float, float]] = []
        for _ in range(bootstrap_samples):
            sample = [
                error
                for index in rng.integers(len(keys), size=len(keys))
                for error in groups[keys[index]]
            ]
            metrics = _metrics(sample)
            sampled.append((metrics["energy"]["rmse"], metrics["forces"]["rmse"]))
        intervals = np.quantile(np.asarray(sampled), [0.025, 0.975], axis=0)
        result["family_bootstrap"] = {
            "samples": bootstrap_samples,
            "seed": seed,
            "family_count": len(groups),
            "families_complete": all(
                error.record.family_id is not None for error in errors
            ),
            "energy_rmse_95ci": intervals[:, 0].tolist(),
            "force_rmse_95ci": intervals[:, 1].tolist(),
        }
    return result
