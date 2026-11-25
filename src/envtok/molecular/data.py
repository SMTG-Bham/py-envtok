"""Finite molecular ingestion, conservative preparation and lossless export."""

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.io.extxyz import key_val_dict_to_str

from envtok._extxyz import encode_string_metadata
from envtok.adapters._labels import extract_labels
from envtok.adapters.base import RecordOptions, make_record
from envtok.adapters.extxyz import ExtxyzAdapter
from envtok.errors import ConfigError, DuplicateRecordError
from envtok.molecular.records import (
    MolecularRecord,
    adapt_record,
    validate_molecular_record,
)
from envtok.records import IssueSeverity, Split

_GROUP_KEYS = (
    "family_id",
    "trajectory_id",
    "reaction_family_id",
    "scan_id",
    "state_pair_id",
)
_PROVENANCE_KEYS = {
    "sid",
    "record_id",
    "source_id",
    "split",
    "label_namespace",
    "total_charge",
    "spin_multiplicity",
    "electronic_state_id",
    "graph",
    "graph_status",
    "envtok_record_id",
    "envtok_source_ref",
    "envtok_original_labels",
    "envtok_original_state",
    "envtok_original_label_units",
    "envtok_array_shapes",
    *_GROUP_KEYS,
}


@dataclass(frozen=True, slots=True)
class PreparedRecords:
    """Eligible deduplicated records and an explicit preparation audit."""

    records: tuple[MolecularRecord, ...]
    candidates: tuple[MolecularRecord, ...]
    exclusions: tuple[dict[str, Any], ...]
    duplicate_lineage: dict[str, list[str]]
    conflicts: tuple[dict[str, Any], ...]
    cross_split_duplicates: tuple[dict[str, Any], ...]
    group_warnings: tuple[str, ...]
    split_manifest: dict[str, str]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _options(
    dataset: str,
    revision: str,
    split: Split | str,
    family_key: str,
    label_sources: Mapping[str, str] | None,
    label_units: Mapping[str, str] | None,
) -> RecordOptions:
    kwargs: dict[str, Any] = {}
    if label_sources is not None:
        kwargs["label_sources"] = label_sources
    if label_units is not None:
        kwargs["label_units"] = label_units
    return RecordOptions(
        dataset=dataset,
        revision=revision,
        split=Split.parse(split),
        split_key="split",
        group_key=family_key,
        **kwargs,
    )


def _restore_shapes(atoms: Atoms) -> None:
    shapes = atoms.info.get("envtok_array_shapes", {})
    for name, shape in shapes.items():
        values = (
            atoms.arrays if name in atoms.arrays else getattr(atoms.calc, "results", {})
        )
        values[name] = values[name].reshape(shape)


def from_atoms(  # noqa: PLR0913 - explicit adapter options
    frames: Iterable[Atoms],
    *,
    label_namespace: str,
    charge_key: str = "total_charge",
    multiplicity_key: str = "spin_multiplicity",
    unknown_policy: str = "separate_and_report",
    dataset: str = "molecular",
    revision: str = "unspecified",
    split: Split | str = Split.TRAIN,
    family_key: str = "family_id",
    label_sources: Mapping[str, str] | None = None,
    label_units: Mapping[str, str] | None = None,
) -> Iterator[MolecularRecord]:
    """Stream an ASE iterable with original calculators' stored results intact."""
    options = _options(dataset, revision, split, family_key, label_sources, label_units)
    for index, original in enumerate(frames):
        atoms = original.copy()
        atoms.info = deepcopy(original.info)
        # Freeze stored results without calling the source calculator.
        if original.calc is not None:
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = deepcopy(original.calc.results)
        _restore_shapes(atoms)
        yield adapt_record(
            make_record(atoms, "memory", str(index), options),
            label_namespace=label_namespace,
            charge_key=charge_key,
            multiplicity_key=multiplicity_key,
            unknown_policy=unknown_policy,
            family_key=family_key,
        )


def iter_extxyz(  # noqa: PLR0913 - explicit adapter options
    path: str | Path,
    *,
    label_namespace: str,
    charge_key: str = "total_charge",
    multiplicity_key: str = "spin_multiplicity",
    unknown_policy: str = "separate_and_report",
    dataset: str = "molecular",
    revision: str = "unspecified",
    split: Split | str = Split.TRAIN,
    family_key: str = "family_id",
    label_sources: Mapping[str, str] | None = None,
    label_units: Mapping[str, str] | None = None,
) -> Iterator[MolecularRecord]:
    """Read plain/compressed extended XYZ with explicit state and unit mapping."""
    adapter = ExtxyzAdapter(
        path, _options(dataset, revision, split, family_key, label_sources, label_units)
    )
    for source_record in adapter.iter_records():
        record = source_record
        if record.atoms.info.get("envtok_array_shapes"):
            _restore_shapes(record.atoms)
            record = replace(
                record,
                labels=extract_labels(
                    record.atoms,
                    adapter.options.label_sources,
                    adapter.options.label_units,
                ),
            )
        yield adapt_record(
            record,
            label_namespace=label_namespace,
            charge_key=charge_key,
            multiplicity_key=multiplicity_key,
            unknown_policy=unknown_policy,
            family_key=family_key,
        )


def _move_mask(atoms: Atoms) -> npt.NDArray[np.bool_] | None:
    """Canonical extxyz movement mask, combining constraints on the same atom."""
    if not atoms.constraints:
        return None
    movable = np.ones((len(atoms), 3), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            movable[constraint.index] = False
        elif isinstance(constraint, FixCartesian):
            movable[constraint.index] &= ~constraint.mask
        else:
            raise ConfigError(
                f"unsupported molecular constraint: {type(constraint).__name__}"
            )
    if movable.all():
        return None
    return movable[:, 0] if (movable == movable[:, :1]).all() else movable


def exact_identity(record: MolecularRecord, *, include_constraints: bool = True) -> str:
    """Hash exact ordered geometry, protected state and external conditions.

    Unknown non-provenance metadata is conservatively treated as an external
    condition. Rotated, permuted, rounded or merely same-graph records are never
    merged by this identity.

    Older saved runs use `include_constraints=False` to reproduce their
    original candidate pool.
    """
    label_info_keys = {
        label.source_key.split(":", 1)[1]
        for label in record.labels.values()
        if label.source_key.startswith("info:")
    }
    conditions = {
        key: value
        for key, value in {**record.record.metadata, **record.atoms.info}.items()
        if key
        not in _PROVENANCE_KEYS
        | label_info_keys
        | {"energy", "free_energy", "forces", "stress"}
    }
    payload = {
        "numbers": record.atoms.numbers,
        "positions": record.atoms.positions,
        "cell": record.atoms.cell.array,
        "pbc": record.atoms.pbc,
        "state": record.state_key,
        "conditions": conditions,
    }
    if include_constraints and (mask := _move_mask(record.atoms)) is not None:
        payload["move_mask"] = mask
    return _digest(payload)


def _label_identity(record: MolecularRecord) -> str:
    return _digest(
        {
            "labels": {
                name: (label.value, label.unit) for name, label in record.labels.items()
            },
            "arrays": {
                name: values
                for name, values in record.atoms.arrays.items()
                if name not in {"numbers", "positions"}
            },
            "results": getattr(record.atoms.calc, "results", {}),
            "info_labels": {
                name: record.atoms.info[name]
                for name in ("energy", "free_energy", "forces", "stress")
                if name in record.atoms.info
            },
            "graph": record.graph,
            "graph_status": record.graph_status,
        }
    )


def _groups(
    records: list[MolecularRecord], include_constraints: bool
) -> tuple[dict[str, str], list[str]]:
    parents = {record.record_id: record.record_id for record in records}

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owners: dict[tuple[str, str], str] = {}
    warnings = []
    for record in records:
        metadata = {**record.record.metadata, **record.atoms.info}
        keys = [
            (key, str(metadata[key]))
            for key in _GROUP_KEYS
            if metadata.get(key) is not None
        ]
        if record.family_id is not None:
            keys.append(("family_id", record.family_id))
        if not keys:
            # Composition is deliberately broad: it cannot split unknown
            # conformers or matched states by charge or geometry.
            keys = [
                ("composition_fallback", str(sorted(record.atoms.numbers.tolist())))
            ]
            warnings.append(
                f"{record.record_id}: missing grouping metadata; "
                "grouped by exact composition"
            )
        # Identical source calculations bridge every declared family alias,
        # including aliases that would disappear when duplicate rows merge.
        keys.append(
            (
                "exact_record",
                exact_identity(record, include_constraints=include_constraints),
            )
        )
        for key in keys:
            if key in owners:
                left, right = root(record.record_id), root(owners[key])
                parents[max(left, right)] = min(left, right)
            else:
                owners[key] = record.record_id
    return {key: root(key) for key in parents}, warnings


def _development_split(
    records: list[MolecularRecord],
    validation_fraction: float,
    seed: int,
    groups: dict[str, str],
    warnings: list[str],
) -> tuple[list[MolecularRecord], list[str]]:
    group_splits: dict[str, set[Split]] = {}
    for record in records:
        group_splits.setdefault(groups[record.record_id], set()).add(record.split)
    for group, splits in sorted(group_splits.items()):
        if len(splits) > 1:
            warnings.append(
                f"{group}: related records span source splits {sorted(splits)}; "
                "source assignments preserved"
            )
    result = []
    for record in records:
        fraction = (
            int(
                hashlib.sha256(
                    f"{seed}:{groups[record.record_id]}".encode()
                ).hexdigest(),
                16,
            )
            / 2**256
        )
        assigned = record
        if record.split is Split.TRAIN and fraction < validation_fraction:
            assigned = replace(
                record, record=replace(record.record, split=Split.VALIDATION)
            )
        result.append(assigned)
    return result, warnings


def prepare_records(
    records: Iterable[MolecularRecord],
    *,
    validation_fraction: float = 0.0,
    seed: int = 20261105,
    include_constraints: bool = True,
) -> PreparedRecords:
    """Validate, deduplicate within source splits and group development folds.

    Conflicting labels are retained and reported, never averaged. Official
    held-out assignments remain intact; only TRAIN records may be assigned to
    development validation. Candidate fitting sees only the remaining TRAIN.
    """
    if not 0.0 <= validation_fraction < 1.0:
        raise ConfigError("validation_fraction must lie in [0, 1)")
    eligible, exclusions = [], []
    seen_ids: set[str] = set()
    for record in sorted(records, key=lambda item: item.record_id):
        if record.record_id in seen_ids:
            raise DuplicateRecordError(
                f"duplicate molecular record ID: {record.record_id}"
            )
        seen_ids.add(record.record_id)
        issues = validate_molecular_record(record)
        failed = [issue for issue in issues if issue.severity is not IssueSeverity.WARN]
        if failed:
            exclusions.append(
                {
                    "record_id": record.record_id,
                    "reasons": [
                        {"code": issue.code, "message": issue.message}
                        for issue in failed
                    ],
                }
            )
        else:
            eligible.append(record)
    groups, warnings = _groups(eligible, include_constraints)
    exact: dict[str, list[tuple[MolecularRecord, str]]] = {}
    lineage: dict[str, list[str]] = {}
    kept, conflicts, cross_split = [], [], []
    for record in eligible:
        identity = exact_identity(record, include_constraints=include_constraints)
        labels = _label_identity(record)
        previous = exact.setdefault(identity, [])
        duplicate = None
        for other, other_labels in previous:
            if record.split is not other.split:
                cross_split.append(
                    {
                        "record_ids": [other.record_id, record.record_id],
                        "splits": [other.split.value, record.split.value],
                        "conflicting_labels": labels != other_labels,
                    }
                )
            elif labels == other_labels:
                duplicate = other.record_id
                break
            else:
                conflicts.append(
                    {
                        "record_ids": [other.record_id, record.record_id],
                        "split": record.split.value,
                        "reason": "same_exact_state_conflicting_labels_or_graph",
                    }
                )
        if duplicate is None:
            previous.append((record, labels))
            kept.append(record)
        else:
            lineage.setdefault(duplicate, []).append(record.record_id)
    split_records, warnings = _development_split(
        kept, validation_fraction, seed, groups, warnings
    )
    return PreparedRecords(
        records=tuple(split_records),
        candidates=tuple(
            record for record in split_records if record.split is Split.TRAIN
        ),
        exclusions=tuple(exclusions),
        duplicate_lineage=lineage,
        conflicts=tuple(conflicts),
        cross_split_duplicates=tuple(cross_split),
        group_warnings=tuple(warnings),
        split_manifest={
            record.record_id: record.split.value for record in split_records
        },
    )


def _export_atoms(record: MolecularRecord) -> Atoms:
    atoms: Atoms = record.atoms.copy()
    atoms.info = deepcopy(record.atoms.info)
    sources = {
        "calc": getattr(record.atoms.calc, "results", {}),
        "info": record.atoms.info,
        "arrays": record.atoms.arrays,
    }
    atoms.info["envtok_original_labels"] = {
        name: {
            "value": _jsonable(
                sources.get(label.source_key.partition(":")[0], {}).get(
                    label.source_key.partition(":")[2], label.value
                )
            ),
            "unit": record.original_label_units.get(name, label.unit),
            "source_key": label.source_key,
        }
        for name, label in record.labels.items()
    }
    for name, value in getattr(record.atoms.calc, "results", {}).items():
        array = np.asarray(value)
        if (
            name in {"forces", "charges", "magmoms", "energies", "stresses"}
            and array.ndim
            and array.shape[0] == record.natoms
        ):
            atoms.arrays[name] = array.copy()
        else:
            atoms.info[name] = _jsonable(value)
    for name, label in record.labels.items():
        if name == "forces":
            atoms.arrays[name] = label.value.copy()
        else:
            atoms.info[name] = _jsonable(label.value)
    atoms.info.update(
        {
            "envtok_record_id": record.record_id,
            "envtok_source_ref": {
                "dataset": record.record.source_ref.dataset,
                "revision": record.record.source_ref.revision,
                "shard": record.record.source_ref.shard,
                "key": record.record.source_ref.key,
                "native_id": record.record.source_ref.native_id,
            },
            "envtok_original_label_units": dict(record.original_label_units),
            "label_namespace": record.label_namespace,
            "split": record.split.value,
        }
    )
    atoms.info["envtok_original_state"] = {
        key: atoms.info[key]
        for key in (
            "total_charge",
            "spin_multiplicity",
            "electronic_state_id",
            "family_id",
        )
        if key in atoms.info
    }
    for key, value in (
        ("total_charge", record.total_charge),
        ("spin_multiplicity", record.spin_multiplicity),
        ("electronic_state_id", record.electronic_state_id),
        ("family_id", record.family_id),
    ):
        if value is not None:
            atoms.info[key] = value
        else:
            atoms.info.pop(key, None)
    move_mask = _move_mask(atoms)
    if move_mask is not None:
        atoms.set_array("move_mask", move_mask)
    return atoms


def _write_frame(handle: TextIO, atoms: Atoms) -> None:
    """Write standard extxyz with 17-digit floats, retaining source precision."""
    if {"species", "pos"} & atoms.arrays.keys():
        raise ConfigError(
            "reserved extxyz array names 'species' and 'pos' cannot be exported"
        )
    arrays = {
        "species": np.asarray(atoms.get_chemical_symbols()),
        "pos": atoms.positions,
    }
    arrays.update(
        {
            key: value
            for key, value in atoms.arrays.items()
            if key not in {"numbers", "positions"}
        }
    )
    columns = []
    properties: list[str] = []
    shapes = {}
    for name, array in arrays.items():
        if array.ndim > 2:  # noqa: PLR2004 - extxyz columns are flat
            shapes[name] = list(array.shape)
        flat = array.reshape(len(atoms), -1)
        dtype = {"f": "R", "i": "I", "u": "I", "b": "L", "U": "S"}.get(array.dtype.kind)
        if dtype is None:
            raise ConfigError(
                f"unsupported extxyz array dtype for {name}: {array.dtype}"
            )
        if dtype == "S" and any(
            not value or any(character.isspace() for character in value)
            for value in array.flat
        ):
            raise ConfigError(
                f"string array {name!r} contains empty or whitespace values"
            )
        properties.extend((name, dtype, str(flat.shape[1])))
        columns.append((flat, dtype))
    info = encode_string_metadata(_jsonable(dict(atoms.info)))
    if shapes:
        info["envtok_array_shapes"] = shapes
    info["Lattice"] = atoms.cell.array.T
    info["pbc"] = atoms.pbc
    handle.write(
        f"{len(atoms)}\nProperties={':'.join(properties)} {key_val_dict_to_str(info)}\n"
    )
    for index in range(len(atoms)):
        values = []
        for array, dtype in columns:
            for value in array[index]:
                if dtype == "R":
                    values.append(format(float(value), ".17g"))
                elif dtype == "L":
                    values.append("T" if value else "F")
                else:
                    values.append(str(value))
        handle.write(" ".join(values) + "\n")


def export_records(
    records: Iterable[MolecularRecord],
    path: str | Path,
    *,
    strict_training_export: bool = True,
) -> Path:
    """Atomically export whole configurations with normalised labels and IDs.

    Strict mode refuses unknown charge/multiplicity. No coordinates or atoms
    are reordered. Higher-rank per-atom tensors are flattened into extxyz
    columns with stored shapes restored by this module's adapters.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                if strict_training_export and (
                    record.total_charge is None or record.spin_multiplicity is None
                ):
                    raise ConfigError(
                        "strict molecular export requires charge and multiplicity: "
                        f"{record.record_id}"
                    )
                failed = [
                    issue.code
                    for issue in validate_molecular_record(record)
                    if issue.severity is not IssueSeverity.WARN
                ]
                if failed:
                    raise ConfigError(
                        f"cannot export invalid record {record.record_id}: {failed}"
                    )
                _write_frame(handle, _export_atoms(record))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target
