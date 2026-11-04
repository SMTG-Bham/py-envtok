"""Label extraction from ASE atoms.

Labels are optional inputs. They are read from the attached (single-point)
calculator results, from ``info`` or from ``arrays``, and keep their declared
units. EnvTok never converts units and never recomputes a label.
"""

from collections.abc import Mapping

import numpy as np
from ase import Atoms

from envtok.errors import ConfigError
from envtok.records import Label

DEFAULT_LABEL_SOURCES = {
    "energy": "calc:energy",
    "forces": "calc:forces",
    "stress": "calc:stress",
}
DEFAULT_UNITS = {
    "energy": "eV",
    "free_energy": "eV",
    "forces": "eV/Ang",
    "stress": "eV/Ang^3",
    "magmoms": "mu_B",
}
_KINDS = ("calc", "info", "arrays")


def _lookup(atoms: Atoms, kind: str, key: str) -> object:
    if kind == "calc":
        calc = atoms.calc
        results = getattr(calc, "results", None) if calc is not None else None
        return None if results is None else results.get(key)
    if kind == "info":
        return atoms.info.get(key)
    return atoms.arrays.get(key)


def extract_labels(
    atoms: Atoms, sources: Mapping[str, str], units: Mapping[str, str]
) -> dict[str, Label]:
    """Read the configured labels that are present on ``atoms``.

    Args:
        atoms: Atoms as read from the source.
        sources: Label name to ``"<kind>:<key>"`` with kind ``calc``, ``info``
            or ``arrays``.
        units: Declared unit per label name (``"unknown"`` when absent).

    Returns:
        The labels found; missing labels are omitted.

    Raises:
        ConfigError: If a source specification is malformed.
    """
    labels: dict[str, Label] = {}
    for name, spec in sorted(sources.items()):
        kind, _, key = spec.partition(":")
        if kind not in _KINDS or not key:
            raise ConfigError(
                f"label source {spec!r} for {name!r} must be 'calc:<key>', "
                "'info:<key>' or 'arrays:<key>'"
            )
        value = _lookup(atoms, kind, key)
        if value is None:
            continue
        labels[name] = Label(
            value=np.array(value, dtype=float),
            unit=units.get(name, "unknown"),
            source_key=spec,
        )
    return labels
