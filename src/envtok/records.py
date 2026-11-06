"""Typed structure records, source references, labels and validation.

Adapters turn every input configuration into a :class:`StructureRecord`. The
record keeps the original ASE atoms (species, positions, cell, periodicity),
a stable namespaced ID, a :class:`SourceRef` that locates the configuration in
its native source, optional labels and free-form metadata. Validation never
modifies a record: it reports :class:`ValidationIssue` objects, and the caller
decides between exclusion and failure.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from ase import Atoms

from envtok.errors import ConfigError

JSONScalar = str | int | float | bool | None


class Split(enum.StrEnum):
    """Data split of a record."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    UNSPLIT = "unsplit"

    @classmethod
    def parse(cls, value: object) -> Split:
        """Parse a split name, accepting common aliases.

        Args:
            value: A split name such as ``"train"``, ``"val"`` or ``"test"``.

        Returns:
            The split.

        Raises:
            ConfigError: If the value is not a recognised split name.
        """
        text = str(value).strip().lower()
        aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
        text = aliases.get(text, text)
        try:
            return cls(text)
        except ValueError:
            raise ConfigError(
                f"unknown split {value!r}; expected train, validation, test or unsplit"
            ) from None


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where a record lives in its native source.

    Attributes:
        dataset: Dataset name, for example ``"omat24"``.
        revision: Dataset revision or version label.
        shard: Shard, file or database path relative to the source root.
        key: Native row key or frame locator within the shard.
        native_id: Optional dataset-provided identifier such as OMat24 ``sid``.
            It is not assumed to be unique.
    """

    dataset: str
    revision: str
    shard: str
    key: str
    native_id: str | None = None


@dataclass(frozen=True, slots=True)
class Label:
    """One label array with its unit and source key.

    Attributes:
        value: The label values (a 0-d array for scalars).
        unit: Declared unit string, for example ``"eV"``.
        source_key: Where the label was read from, for example ``"calc:energy"``.
    """

    value: npt.NDArray[np.float64]
    unit: str
    source_key: str


@dataclass(frozen=True, slots=True)
class StructureRecord:
    """A validated input configuration.

    Attributes:
        record_id: Stable, globally unique configuration ID.
        source_ref: Location of the configuration in its native source.
        atoms: Species, positions, cell and periodicity of the configuration.
        split: Data split.
        group_id: Parent material, trajectory or calculation family, if known.
        metadata: JSON-scalar provenance such as charge, method or temperature.
        labels: Optional label arrays keyed by name (``energy``, ``forces``,
            ``stress``, ``magmoms``).
    """

    record_id: str
    source_ref: SourceRef
    atoms: Atoms
    split: Split = Split.UNSPLIT
    group_id: str | None = None
    metadata: Mapping[str, JSONScalar] = field(default_factory=dict)
    labels: Mapping[str, Label] = field(default_factory=dict)

    @property
    def natoms(self) -> int:
        """Original atom count, the selection cost of the record."""
        return len(self.atoms)

    def metadata_json(self) -> str:
        """Return the metadata as canonical JSON."""
        return json.dumps(dict(self.metadata), sort_keys=True, default=str)


class IssueSeverity(enum.StrEnum):
    """How a validation issue affects eligibility."""

    FAIL = "fail"
    EXCLUDE = "exclude"
    WARN = "warn"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A problem found while validating a record.

    Attributes:
        code: Stable machine-readable code, for example ``"nonfinite_positions"``.
        severity: ``fail`` quarantines the record, ``exclude`` marks it
            ineligible by policy, ``warn`` keeps it eligible.
        message: Human-readable explanation with a suggested action.
    """

    code: str
    severity: IssueSeverity
    message: str


def _expected_shapes(name: str, natoms: int) -> list[tuple[int, ...]] | None:
    shapes: dict[str, list[tuple[int, ...]]] = {
        "energy": [()],
        "free_energy": [()],
        "forces": [(natoms, 3)],
        "stress": [(6,), (3, 3)],
        "magmoms": [(natoms,)],
    }
    return shapes.get(name)


def validate_record(
    record: StructureRecord, *, require_periodic: bool = True
) -> list[ValidationIssue]:
    """Check a record against the input contract.

    The checks follow the specification's preflight list: positive atom
    count, finite coordinates, valid species, a usable cell for periodic
    profiles, consistent label shapes and collinear initial moments.

    Args:
        record: The record to check.
        require_periodic: Whether the profile needs a fully periodic cell.

    Returns:
        Every issue found, in a deterministic order; empty for a valid record.
    """
    issues: list[ValidationIssue] = []
    atoms = record.atoms
    natoms = len(atoms)
    if natoms == 0:
        return [ValidationIssue("empty_structure", IssueSeverity.FAIL, "no atoms")]
    numbers = atoms.get_atomic_numbers()
    if np.any(numbers < 1) or np.any(numbers > 118):  # noqa: PLR2004 - periodic table
        issues.append(
            ValidationIssue(
                "invalid_species",
                IssueSeverity.FAIL,
                "atomic numbers must lie in 1..118",
            )
        )
    if not np.all(np.isfinite(atoms.get_positions())):
        issues.append(
            ValidationIssue(
                "nonfinite_positions",
                IssueSeverity.FAIL,
                "positions contain NaN or infinity",
            )
        )
    issues.extend(_cell_issues(atoms, require_periodic=require_periodic))
    if atoms.has("initial_magmoms"):
        moments = np.asarray(atoms.get_initial_magnetic_moments())
        if moments.ndim != 1:
            issues.append(
                ValidationIssue(
                    "noncollinear_moments",
                    IssueSeverity.FAIL,
                    "initial magnetic moments are vectors; the reference moment "
                    "field needs collinear scalar moments. Convert them to "
                    "scalars explicitly or set tokenizer.magnetic_policy: disabled",
                )
            )
        elif not np.all(np.isfinite(moments)):
            issues.append(
                ValidationIssue(
                    "nonfinite_moments",
                    IssueSeverity.FAIL,
                    "initial magnetic moments contain NaN or infinity",
                )
            )
    for name, label in sorted(record.labels.items()):
        issues.extend(_label_issues(name, label, natoms))
    return issues


def _cell_issues(atoms: Atoms, *, require_periodic: bool) -> list[ValidationIssue]:
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if require_periodic and not pbc.all():
        return [
            ValidationIssue(
                "unsupported_periodicity",
                IssueSeverity.EXCLUDE,
                f"periodicity {pbc.tolist()} is not supported by this periodic "
                "profile; molecules and slabs need a validated nonperiodic profile",
            )
        ]
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if not np.all(np.isfinite(cell)):
        return [ValidationIssue("nonfinite_cell", IssueSeverity.FAIL, "cell has NaN")]
    if require_periodic and abs(float(np.linalg.det(cell))) < 1e-6:  # noqa: PLR2004
        return [
            ValidationIssue(
                "degenerate_cell",
                IssueSeverity.FAIL,
                "periodic cell has (near) zero volume",
            )
        ]
    return []


def _label_issues(name: str, label: Label, natoms: int) -> list[ValidationIssue]:
    value = np.asarray(label.value)
    expected = _expected_shapes(name, natoms)
    if expected is not None and value.shape not in expected:
        return [
            ValidationIssue(
                "label_shape",
                IssueSeverity.WARN,
                f"label {name!r} has shape {value.shape}; expected one of {expected}",
            )
        ]
    if value.size and not np.all(np.isfinite(value)):
        return [
            ValidationIssue(
                "nonfinite_label",
                IssueSeverity.WARN,
                f"label {name!r} contains NaN or infinity",
            )
        ]
    return []


def scalar_metadata(
    info: Mapping[str, Any], *, skip: frozenset[str]
) -> dict[str, JSONScalar]:
    """Return the JSON-scalar entries of an ASE ``info`` dictionary.

    Args:
        info: ``Atoms.info`` or database key-value pairs.
        skip: Keys to omit (for example keys consumed as labels).

    Returns:
        Scalar metadata with NumPy scalars converted to Python types; arrays,
        nested objects and non-finite floats are omitted.
    """
    metadata: dict[str, JSONScalar] = {}
    for key in sorted(info):
        if key in skip:
            continue
        value = info[key]
        if isinstance(value, np.generic):
            value = value.item()
        finite_float = isinstance(value, float) and math.isfinite(value)
        if isinstance(value, bool | int | str) or value is None or finite_float:
            metadata[str(key)] = value
    return metadata
