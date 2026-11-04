"""Raw distance-ratio and asymmetry measurements of one site.

For each contact j of the chosen neighbour set, with displacement ``r_ij``,
distance ``d_ij`` and covalent radii ``R_i``, ``R_j``::

    q_ij  = d_ij / (R_i + R_j)
    rho_i = min_j q_ij
    A_i   = | sum_j exp(-(q_ij - 1) / lambda) * r_ij / d_ij |

``A`` is not normalised by coordination number. Contacts whose element has no
tabulated radius, and degenerate zero-length contacts, are skipped; when no
contact remains the site is undefined. The arithmetic follows the reference
``_distance_ratio_bins`` operation by operation so results are bitwise
identical.

Adapted from em819/omat24-chemenv-subset@723696d (MIT License, Copyright (c)
2026 Evgeny Moerman; see LICENSES/omat24-chemenv-subset-MIT.txt).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from ase.data import atomic_numbers

from envtok.profiles.reference import MeasureSettings


@dataclass(frozen=True, slots=True)
class ContactMeasures:
    """Raw measurements of one site; all ``None`` when undefined.

    Attributes:
        ratio_min: Minimum normalised contact distance (rho).
        ratio_max: Maximum normalised contact distance (diagnostic only).
        asymmetry: Asymmetry magnitude A.
        n_used: Number of contacts that contributed.
    """

    ratio_min: float | None
    ratio_max: float | None
    asymmetry: float | None
    n_used: int


UNDEFINED_MEASURES = ContactMeasures(None, None, None, 0)


def covalent_radius(symbol: str, settings: MeasureSettings) -> float | None:
    """Return the tabulated covalent radius of an element.

    Args:
        symbol: Element symbol.
        settings: Measurement settings holding the radius table.

    Returns:
        The radius in angstrom, or ``None`` for unknown symbols and elements
        without a tabulated value (Z > 96 in the reference table).
    """
    z = atomic_numbers.get(symbol)
    if z is None:
        return None
    return settings.radii.radius(z)


def contact_measures(
    center_symbol: str,
    center: npt.NDArray[np.float64],
    contacts: Sequence[tuple[str, npt.NDArray[np.float64]]],
    settings: MeasureSettings,
) -> ContactMeasures:
    """Measure the compression and asymmetry of one site.

    Args:
        center_symbol: Central element.
        center: Cartesian coordinates of the central site.
        contacts: (element, image-resolved cartesian coordinates) of every
            contact in the chosen neighbour set.
        settings: Measurement settings.

    Returns:
        The raw measurements.
    """
    r_central = covalent_radius(center_symbol, settings)
    if r_central is None:
        return UNDEFINED_MEASURES
    ratios: list[float] = []
    a_vec = np.zeros(3, dtype=float)
    for symbol, coords in contacts:
        r_nb = covalent_radius(symbol, settings)
        if r_nb is None:
            continue
        d0 = r_central + r_nb
        vec = np.array(coords) - center
        dist = float(np.linalg.norm(vec))
        if not np.isfinite(dist) or dist <= 0.0:
            continue
        ratio = dist / d0
        ratios.append(ratio)
        a_vec = a_vec + np.exp(-(ratio - 1.0) / settings.asymmetry_lambda) * (
            vec / dist
        )
    if not ratios:
        return UNDEFINED_MEASURES
    return ContactMeasures(
        ratio_min=float(min(ratios)),
        ratio_max=float(max(ratios)),
        asymmetry=float(np.linalg.norm(a_vec)),
        n_used=len(ratios),
    )
