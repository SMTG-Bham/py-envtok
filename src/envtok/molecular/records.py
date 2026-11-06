"""Molecular state and label contract without inferred electronic metadata."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import numpy as np
from ase import Atoms, units

from envtok.errors import ConfigError
from envtok.records import (
    IssueSeverity,
    Label,
    Split,
    StructureRecord,
    ValidationIssue,
    validate_record,
)

StateKey: TypeAlias = tuple[str, float | None, int | None, str | None]


@dataclass(frozen=True, slots=True)
class MolecularRecord:
    """An intact source configuration with explicit molecular electronic state.

    ``record.atoms`` retains original coordinates, labels and nested metadata.
    ``labels`` contains energy/forces in eV and eV/Ang, while the source units
    remain in ``original_label_units``. Unknown state is represented by None.
    """

    record: StructureRecord
    total_charge: float | None = None
    spin_multiplicity: int | None = None
    electronic_state_id: str | None = None
    label_namespace: str = "unspecified"
    family_id: str | None = None
    graph: dict[str, Any] | None = None
    graph_status: str = "missing"
    issues: tuple[ValidationIssue, ...] = ()
    original_label_units: Mapping[str, str] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        """Stable namespaced source ID."""
        return self.record.record_id

    @property
    def atoms(self) -> Atoms:
        """Original atom ordering, coordinates and source labels."""
        return self.record.atoms

    @property
    def natoms(self) -> int:
        """Configuration cost including explicit hydrogen atoms."""
        return self.record.natoms

    @property
    def split(self) -> Split:
        """Source or explicitly assigned development split."""
        return self.record.split

    @property
    def labels(self) -> Mapping[str, Label]:
        """Normalised labels; values never influence default selection."""
        return self.record.labels

    @property
    def state_key(self) -> StateKey:
        """Fields that must remain protected in every coverage hierarchy."""
        return (
            self.label_namespace,
            None if self.total_charge is None else (float(self.total_charge) or 0.0),
            None if self.spin_multiplicity is None else int(self.spin_multiplicity),
            self.electronic_state_id,
        )


def _state_number(
    raw: object, field_name: str, issues: list[ValidationIssue]
) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        issues.append(
            ValidationIssue(
                f"invalid_{field_name}",
                IssueSeverity.FAIL,
                f"{field_name} is not numeric",
            )
        )
        return None
    value = float(raw)
    if not math.isfinite(value):
        issues.append(
            ValidationIssue(
                f"invalid_{field_name}",
                IssueSeverity.FAIL,
                f"{field_name} is not finite",
            )
        )
        return None
    return value


def _normalise_labels(
    labels: Mapping[str, Label], issues: list[ValidationIssue]
) -> dict[str, Label]:
    energy_units = {
        "eV": 1.0,
        "Hartree": units.Hartree,
        "Ha": units.Hartree,
        "kcal/mol": units.kcal / units.mol,
        "kJ/mol": units.kJ / units.mol,
    }
    force_units = {
        "eV/Ang": 1.0,
        "eV/angstrom": 1.0,
        "eV/Angstrom": 1.0,
        "Hartree/Bohr": units.Hartree / units.Bohr,
        "Ha/Bohr": units.Hartree / units.Bohr,
        "kcal/mol/Ang": units.kcal / units.mol,
    }
    result = dict(labels)
    for name, label in labels.items():
        if name in {"energy", "free_energy"}:
            factors, target_unit = energy_units, "eV"
        elif name == "forces":
            factors, target_unit = force_units, "eV/Ang"
        else:
            continue
        factor = factors.get(label.unit)
        if factor is None:
            issues.append(
                ValidationIssue(
                    "unsupported_label_unit",
                    IssueSeverity.FAIL,
                    f"{name} has unsupported unit {label.unit!r}",
                )
            )
        else:
            result[name] = Label(
                np.asarray(label.value, dtype=np.float64) * factor,
                target_unit,
                label.source_key,
            )
    return result


def adapt_record(
    record: StructureRecord,
    *,
    label_namespace: str,
    charge_key: str = "total_charge",
    multiplicity_key: str = "spin_multiplicity",
    unknown_policy: str = "separate_and_report",
    family_key: str = "family_id",
    state_key: str = "electronic_state_id",
) -> MolecularRecord:
    """Map one source record without guessing its charge or multiplicity.

    Invalid fields become traceable failed issues. Missing values remain None;
    ``unknown_policy='reject'`` makes them failed rather than warning issues.
    No partial charges, magnetic moments or spin expectations are consulted.
    """
    if unknown_policy not in {"separate_and_report", "reject"}:
        raise ConfigError("unknown_policy must be separate_and_report or reject")
    if not label_namespace:
        raise ConfigError("label_namespace must be non-empty")
    info = {**record.metadata, **record.atoms.info}
    issues: list[ValidationIssue] = []
    charge = _state_number(info.get(charge_key), "total_charge", issues)
    mult_value = _state_number(info.get(multiplicity_key), "spin_multiplicity", issues)
    multiplicity = None
    if mult_value is not None:
        if mult_value >= 1 and mult_value.is_integer():
            multiplicity = int(mult_value)
        else:
            issues.append(
                ValidationIssue(
                    "invalid_spin_multiplicity",
                    IssueSeverity.FAIL,
                    "spin_multiplicity must be a positive integer",
                )
            )
    for name, value in (("total_charge", charge), ("spin_multiplicity", multiplicity)):
        if value is None:
            issues.append(
                ValidationIssue(
                    f"unknown_{name}",
                    IssueSeverity.FAIL
                    if unknown_policy == "reject"
                    else IssueSeverity.WARN,
                    f"{name} is unknown; it is not defaulted",
                )
            )
    namespace = info.get("label_namespace", label_namespace)
    if not isinstance(namespace, str) or not namespace:
        raise ConfigError("record label_namespace must be a non-empty string")
    graph = info.get("graph")
    graph_status = str(
        info.get("graph_status", "supplied" if graph is not None else "missing")
    )
    if graph is not None and not isinstance(graph, dict):
        graph, graph_status = None, "ambiguous"
    family = info.get(family_key, record.group_id)
    electronic_state = info.get(state_key)
    return MolecularRecord(
        record=replace(record, labels=_normalise_labels(record.labels, issues)),
        total_charge=charge,
        spin_multiplicity=multiplicity,
        electronic_state_id=None if electronic_state is None else str(electronic_state),
        label_namespace=namespace,
        family_id=None if family is None else str(family),
        graph=graph,
        graph_status=graph_status,
        issues=tuple(issues),
        original_label_units={
            name: label.unit for name, label in record.labels.items()
        },
    )


def validate_molecular_record(record: MolecularRecord) -> tuple[ValidationIssue, ...]:
    """Check finite nonperiodic geometry, label shapes, units and source status."""
    issues = list(record.issues)
    _state_number(record.total_charge, "total_charge", issues)
    multiplicity = _state_number(record.spin_multiplicity, "spin_multiplicity", issues)
    if multiplicity is not None and (multiplicity < 1 or not multiplicity.is_integer()):
        issues.append(
            ValidationIssue(
                "invalid_spin_multiplicity",
                IssueSeverity.FAIL,
                "spin_multiplicity must be a positive integer",
            )
        )
    if not record.label_namespace:
        issues.append(
            ValidationIssue(
                "missing_label_namespace",
                IssueSeverity.FAIL,
                "label_namespace must be nonempty",
            )
        )
    for issue in validate_record(record.record, require_periodic=False):
        # Molecular multiplicity is independent of site magnetic moments.
        if issue.code not in {"noncollinear_moments", "nonfinite_moments"}:
            severity = (
                IssueSeverity.FAIL
                if issue.code in {"label_shape", "nonfinite_label"}
                else issue.severity
            )
            issues.append(replace(issue, severity=severity))
    if record.atoms.get_pbc().any():
        issues.append(
            ValidationIssue(
                "unsupported_periodicity",
                IssueSeverity.EXCLUDE,
                "molecular_v1 accepts finite nonperiodic configurations only",
            )
        )
    if record.atoms.info.get("calculation_status") in {"failed", "unconverged"}:
        issues.append(
            ValidationIssue(
                "failed_calculation",
                IssueSeverity.EXCLUDE,
                "source declares a failed or unconverged calculation",
            )
        )
    return tuple(issues)
