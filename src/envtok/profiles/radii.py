"""Frozen covalent radius table used by the reference profile.

The reference tokeniser looks radii up in ``ase.data.covalent_radii``, which
holds the Cordero et al. (2008) values for Z = 1..96 and pads heavier elements
with a placeholder. The values below are copied from ASE 3.25.0, the version in
the reference environment, so a future ASE release cannot silently change the
compression and asymmetry fields. :func:`table_drift` reports any difference
from the installed ASE table.

Reference: B. Cordero et al., "Covalent radii revisited", Dalton Trans. (2008)
2832-2838.
"""

from dataclasses import dataclass

import numpy as np

_CORDERO_2008_ASE_3_25_0: tuple[float, ...] = (
    0.2, 0.31, 0.28, 1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, 1.66,
    1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76, 1.7, 1.6, 1.53,
    1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22, 1.22, 1.2, 1.19, 1.2, 1.2,
    1.16, 2.2, 1.95, 1.9, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45,
    1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.4, 2.44, 2.15, 2.07, 2.04, 2.03,
    2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.9, 1.87, 1.87,
    1.75, 1.7, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32, 1.45, 1.46, 1.48,
    1.4, 1.5, 1.5, 2.6, 2.21, 2.15, 2.06, 2.0, 1.96, 1.9, 1.87, 1.8, 1.69,
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class RadiusTable:
    """A versioned table of covalent radii in angstrom, indexed by atomic number.

    Attributes:
        name: Stable identifier recorded in cache fingerprints.
        values: Radius for Z = 0..``max_z``; index 0 is a placeholder.
        max_z: Highest atomic number with a tabulated radius.
    """

    name: str
    values: tuple[float, ...]
    max_z: int

    def radius(self, atomic_number: int) -> float | None:
        """Return the tabulated radius, or ``None`` if the element has none.

        Args:
            atomic_number: Atomic number Z.

        Returns:
            The radius in angstrom, or ``None`` for Z <= 0, Z > ``max_z`` and
            non-finite or non-positive entries.
        """
        if atomic_number <= 0 or atomic_number > self.max_z:
            return None
        value = self.values[atomic_number]
        if not np.isfinite(value) or value <= 0.0:
            return None
        return value


CORDERO_2008 = RadiusTable(
    name="cordero2008/ase-3.25.0",
    values=_CORDERO_2008_ASE_3_25_0,
    max_z=96,
)


def table_drift(table: RadiusTable = CORDERO_2008) -> list[int]:
    """Return atomic numbers whose installed ASE radius differs from ``table``.

    Args:
        table: The frozen table to compare.

    Returns:
        Sorted atomic numbers 1..``max_z`` with a different value in
        ``ase.data.covalent_radii``; empty when the tables agree.
    """
    from ase.data import covalent_radii  # noqa: PLC0415 - optional diagnostic

    installed = np.asarray(covalent_radii, dtype=float)
    return [
        z
        for z in range(1, table.max_z + 1)
        if z >= installed.size
        or z >= len(table.values)
        or installed[z] != table.values[z]
    ]
