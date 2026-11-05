"""Small NumPy helpers shared by the scale-sensitive stages."""

import numpy as np
import numpy.typing as npt


def sorted_unique(values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Return the sorted distinct values of an integer array.

    A sort followed by a neighbour comparison; markedly faster than
    ``numpy.unique`` for the large int64 key arrays used here.

    Args:
        values: Integer keys.

    Returns:
        Sorted distinct keys.
    """
    ordered = np.sort(values, kind="stable")
    if ordered.size == 0:
        return ordered
    keep = np.empty(ordered.size, dtype=bool)
    keep[0] = True
    np.not_equal(ordered[1:], ordered[:-1], out=keep[1:])
    return ordered[keep]
