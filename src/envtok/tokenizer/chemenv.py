"""Batched continuous symmetry measures for ChemEnv's separation-plane search.

pymatgen's ``LocalGeometryFinder`` scores every candidate ordering of a local
environment against a model polyhedron in its own call to ``symmetry_measure``:
a least-squares rotation from a 3x3 SVD, a scale and a sum of squares over a
few points. With the reference profile that is about 4,700 calls per site, and
most of their cost is per-call overhead rather than arithmetic.

:class:`BatchedLocalGeometryFinder` scores all orderings of one separation
plane as stacked arrays. Each ordering still goes through the same BLAS and
LAPACK routines, on the same memory layout, as pymatgen's per-call code: the
stacked matrix products and SVD call them once per ordering, and every sum is
the same BLAS dot product. Symmetry measures, scales and rotations are
therefore bit-for-bit identical, and so are the environments and tokens.

The override mirrors pymatgen source that may change between releases, so
:func:`make_finder` uses it only when that source is exactly the source it was
checked against, and otherwise returns pymatgen's own finder.

The ordering set-up in ``BatchedLocalGeometryFinder`` is adapted from
pymatgen (MIT License, Copyright (c) 2011-2012 MIT & The Regents of the
University of California, through Lawrence Berkeley National Laboratory; see
LICENSES/pymatgen-MIT.txt).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from typing import Any

import numpy as np
import numpy.typing as npt
from pymatgen.analysis.chemenv.coordination_environments import (
    coordination_geometry_finder as finder_module,
)
from typing_extensions import override

LocalGeometryFinder = finder_module.LocalGeometryFinder

_MIRRORED = {
    "symmetry_measure": (
        "d38596e244b075796ba92c9793f47efe1cd8e418991b49765ce99e05cfb8993c"
    ),
    "find_rotation": "53d3f13c09a12f7ac98ad736694223d4ac2ea9b40a70c0d50edc5cc6010aa2d7",
    "find_scaling_factor": (
        "b4bb728fb97dfc9ccf81bd9c43a69b673bdf960133dd4f584ad90937ee00b30a"
    ),
    "LocalGeometryFinder._cg_csm_separation_plane_optim2": (
        "e6c0af9f94e8600e8b1e3a7fac549db52f04fdbbd9cc5e15ee10b3b944a4d2c2"
    ),
    "AbstractGeometry.points_wcs_ctwcc": (
        "8d77492bdd378b0bf2bc16c7247cfe856def2c5a4c5b67c4b12f0a8c5bff10b5"
    ),
}
"""SHA-256 of the pymatgen source the override reproduces (2026.9.23 to 2026.10.24)."""


def _source_digest(qualname: str) -> str | None:
    target: Any = finder_module
    for part in qualname.split("."):
        target = getattr(target, part, None)
    try:
        return hashlib.sha256(inspect.getsource(target).encode()).hexdigest()
    except (OSError, TypeError):
        return None


@functools.cache
def batched_supported() -> bool:
    """Whether the installed pymatgen has exactly the code the override mirrors."""
    if getattr(finder_module, "svd", None) is not np.linalg.svd:
        return False
    return all(_source_digest(name) == digest for name, digest in _MIRRORED.items())


def make_finder(**options: Any) -> Any:
    """A ``LocalGeometryFinder``, batched when the installed pymatgen allows it."""
    finder = BatchedLocalGeometryFinder if batched_supported() else LocalGeometryFinder
    return finder(**options)


def symmetry_measures(
    distorted: npt.NDArray[np.float64], perfect: npt.NDArray[np.float64]
) -> tuple[list[float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Continuous symmetry measures of stacked orderings against one polyhedron.

    Args:
        distorted: ``(orderings, points, 3)`` local points, one ordering each.
        perfect: ``(points, 3)`` model points.

    Returns:
        Each ordering's symmetry measure, scaling factor and rotation matrix,
        bit-for-bit those of pymatgen's ``symmetry_measure``.
    """
    count, points, _ = distorted.shape
    columns = 3 * points
    # find_rotation: H = distorted.T @ perfect, then Vt.T @ U.T from its SVD.
    u, _, vt = np.linalg.svd(np.matmul(distorted.transpose(0, 2, 1), perfect))
    rotations = np.matmul(vt.transpose(0, 2, 1), u.transpose(0, 2, 1))
    # find_scaling_factor: full contractions are BLAS dot products, as in tensordot.
    rotated = np.matmul(rotations, distorted.transpose(0, 2, 1)).transpose(0, 2, 1)
    flat = rotated.reshape(count, 1, columns)
    numerator = np.matmul(flat, perfect.reshape(columns, 1))[:, 0, 0]
    denominator = np.matmul(flat, flat.transpose(0, 2, 1))[:, 0, 0]
    scales = numerator / denominator
    # symmetry_measure: squared error of the scaled rotation, relative to perfect.
    difference = (perfect - rotated * scales[:, None, None]).reshape(count, 1, columns)
    error = np.matmul(difference, difference.transpose(0, 2, 1))[:, 0, 0]
    norm = float(perfect.ravel().dot(perfect.ravel()))
    measures = [0.0] * count if norm == 0.0 else ((error / norm) * 100.0).tolist()
    return measures, scales, rotations


class BatchedLocalGeometryFinder(LocalGeometryFinder):
    """``LocalGeometryFinder`` scoring each separation plane's orderings at once."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Create the finder; see pymatgen's ``LocalGeometryFinder``."""
        super().__init__(*args, **kwargs)
        self._orderings: dict[int, tuple[Any, npt.NDArray[np.intp]]] = {}
        self._projected_geometry: Any = None
        self._projections: dict[tuple[int, bytes], tuple[Any, Any]] = {}

    def _ordering_array(self, orderings: Any) -> npt.NDArray[np.intp]:
        """The algorithm's orderings as one array, converted once per list."""
        cached = self._orderings.get(id(orderings))
        if cached is None or cached[0] is not orderings:
            cached = (orderings, np.asarray(orderings, dtype=np.intp))
            self._orderings[id(orderings)] = cached
        return cached[1]

    def _anticlockwise(self, local_plane: Any, indices: npt.NDArray[np.intp]) -> Any:
        """Anticlockwise order of the given neighbours around ``local_plane``.

        The same plane and neighbours recur for every geometry sharing a
        separation, so results are kept until the local geometry changes.
        """
        geometry = self.local_geometry
        if self._projected_geometry is not geometry:
            self._projected_geometry = geometry
            self._projections = {}
        key = (id(local_plane), indices.tobytes())
        cached = self._projections.get(key)
        if cached is None or cached[0] is not local_plane:
            points = geometry.coords.take(indices, axis=0)
            cached = (
                local_plane,
                local_plane.project_and_to2dim_ordered_indices(points),
            )
            self._projections[key] = cached
        return cached[1]

    @override
    def _cg_csm_separation_plane_optim2(
        self,
        coordination_geometry: Any,
        sepplane: Any,
        local_plane: Any,
        points_perfect: Any = None,
        separation_indices: Any = None,
    ) -> tuple[list[dict[str, Any]], list[Any], list[str], bool]:
        """Score every ordering of one separation plane; see pymatgen."""
        argref_separation = sepplane.argsorted_ref_separation_perm
        geometry = self.local_geometry
        if sepplane.ordered_plane:
            separation_indices = [arr.astype(np.intp) for arr in separation_indices]
            if sepplane.ordered_point_groups[0]:
                ordind_s0 = self._anticlockwise(local_plane, separation_indices[0])
                sep0 = separation_indices[0].take(ordind_s0)
            else:
                sep0 = separation_indices[0]
            if sepplane.ordered_point_groups[1]:
                ordind_s2 = self._anticlockwise(local_plane, separation_indices[2])
                sep2 = separation_indices[2].take(ordind_s2)
            else:
                sep2 = separation_indices[2]
            ordind = self._anticlockwise(local_plane, separation_indices[1])
            inp1 = separation_indices[1].take(ordind)
            separation_perm = np.concatenate((sep0, inp1, sep2))
        else:
            separation_perm = np.concatenate(separation_indices)

        if self.plane_safe_permutations:
            sep_perms = sepplane.safe_separation_permutations(
                ordered_plane=sepplane.ordered_plane,
                ordered_point_groups=sepplane.ordered_point_groups,
            )
        else:
            sep_perms = sepplane.permutations
        if not len(sep_perms):
            return [], [], [], False

        permutations = separation_perm.take(self._ordering_array(sep_perms)).take(
            argref_separation, axis=1
        )
        # points_wcs_ctwcc(permutation): the centre, then the ordered neighbours.
        distorted = np.empty((len(permutations), permutations.shape[1] + 1, 3))
        distorted[:, 0] = geometry.points_wcs_ctwcc()[0]
        distorted[:, 1:] = geometry.points_wocs_ctwcc().take(permutations, axis=0)
        perfect = np.asarray(points_perfect, dtype=np.float64)
        measures, scales, rotations = symmetry_measures(distorted, perfect)
        translation = geometry.centroid_with_centre
        results = [
            {
                "symmetry_measure": measure,
                "scaling_factor": scale,
                "rotation_matrix": rotation,
                "translation_vector": translation,
            }
            for measure, scale, rotation in zip(
                measures, scales, rotations, strict=True
            )
        ]
        algorithms = [sepplane.algorithm_type] * len(results)
        return results, list(permutations), algorithms, False
