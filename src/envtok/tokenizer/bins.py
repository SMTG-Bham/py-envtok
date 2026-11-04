"""Binning of raw site measurements into token fields.

Scalar functions reproduce the reference expressions exactly; the array
versions are vectorised equivalents used when deriving tokens from cached raw
measurements. Undefined values are ``None`` (scalars) or ``NaN`` (arrays,
except the CSM, see :func:`csm_bins`).
"""

import math

import numpy as np
import numpy.typing as npt

from envtok.profiles.reference import UNDEFINED_SENTINEL, BinSettings, MagneticPolicy


def csm_bin(csm: float | None, bins: BinSettings) -> int | None:
    """Bin a continuous symmetry measure.

    The comparisons reproduce the reference ``if`` chain: 0 for CSM <= 1,
    1 for <= 5, 2 for < 10, otherwise 3. A ``NaN`` CSM fails every
    comparison and lands in bin 3, as in the reference; only a missing
    (``None``) CSM is undefined.

    Args:
        csm: Raw CSM, or ``None`` when ChemEnv assigned no geometry.
        bins: Bin settings.

    Returns:
        The bin, or ``None`` when undefined.
    """
    if csm is None:
        return None
    first, second, third = bins.csm_edges
    if csm <= first:
        return 0
    if csm <= second:
        return 1
    if csm < third:
        return 2
    return 3


def digitize(value: float | None, edges: tuple[float, ...]) -> int | None:
    """Bin a value with ``numpy.digitize`` semantics (lower bound closed).

    Args:
        value: Raw value, or ``None``/``NaN`` when undefined.
        edges: Increasing bin edges.

    Returns:
        The bin index, or ``None`` when the value is undefined.
    """
    if value is None or math.isnan(value):
        return None
    return int(np.digitize(value, edges))


def moment_bin(moment: float, policy: MagneticPolicy) -> int | None:
    """Return the moment field of one site.

    Args:
        moment: The site's own scalar initial moment (0 when none supplied).
        policy: Moment policy.

    Returns:
        ``round(abs(moment))`` with ties to even, or ``None`` when disabled.
    """
    if policy is MagneticPolicy.DISABLED:
        return None
    return round(abs(moment))


def csm_bins(
    csm: npt.NDArray[np.float64], defined: npt.NDArray[np.bool_], bins: BinSettings
) -> npt.NDArray[np.int64]:
    """Vectorised :func:`csm_bin`.

    Args:
        csm: Raw CSM values (``NaN`` allowed and binned like the reference).
        defined: ``False`` where the CSM is missing (undefined).
        bins: Bin settings.

    Returns:
        Bins with :data:`UNDEFINED_SENTINEL` where undefined.
    """
    first, second, third = bins.csm_edges
    with np.errstate(invalid="ignore"):
        result = np.select(
            [csm <= first, csm <= second, csm < third], [0, 1, 2], default=3
        ).astype(np.int64)
    result[~defined] = UNDEFINED_SENTINEL
    return result


def digitize_array(
    values: npt.NDArray[np.float64], edges: tuple[float, ...]
) -> npt.NDArray[np.int64]:
    """Vectorised :func:`digitize`.

    Args:
        values: Raw values with ``NaN`` where undefined.
        edges: Increasing bin edges.

    Returns:
        Bins with :data:`UNDEFINED_SENTINEL` where undefined.
    """
    result = np.digitize(values, edges).astype(np.int64)
    result[np.isnan(values)] = UNDEFINED_SENTINEL
    return result


def moment_bins(
    moments: npt.NDArray[np.float64], policy: MagneticPolicy
) -> npt.NDArray[np.int64] | None:
    """Vectorised :func:`moment_bin`.

    Args:
        moments: Raw scalar initial moments.
        policy: Moment policy.

    Returns:
        Rounded absolute moments (ties to even), or ``None`` when disabled.
    """
    if policy is MagneticPolicy.DISABLED:
        return None
    return np.rint(np.abs(moments)).astype(np.int64)
