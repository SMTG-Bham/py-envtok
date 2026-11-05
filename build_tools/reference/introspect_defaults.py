"""Print the effective defaults of every library call the EnvTok profile relies on.

Run this in both the pinned reference environment and the development
environment and diff the JSON output. Any difference is a candidate source of
token drift and must either be frozen explicitly in
``envtok.profiles.reference`` or recorded in ``docs/discrepancies.md``.

Usage::

    python build_tools/reference/introspect_defaults.py > defaults.json
"""

import inspect
import json
import sys
from importlib.metadata import version

from pymatgen.analysis.chemenv.coordination_environments import (
    chemenv_strategies,
    coordination_geometry_finder,
)
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

LocalGeometryFinder = coordination_geometry_finder.LocalGeometryFinder
SimplestChemenvStrategy = chemenv_strategies.SimplestChemenvStrategy


def _defaults(func: object) -> dict[str, object]:
    signature = inspect.signature(func)  # type: ignore[arg-type]
    return {
        name: repr(parameter.default)
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def main() -> None:
    """Write the defaults as JSON to standard output."""
    report = {
        "python": sys.version.split()[0],
        "versions": {
            name: version(name)
            for name in ("ase", "pymatgen", "spglib", "numpy", "scipy")
        },
        "LocalGeometryFinder.__init__": _defaults(LocalGeometryFinder.__init__),
        "LocalGeometryFinder.setup_parameters": _defaults(
            LocalGeometryFinder.setup_parameters
        ),
        "LocalGeometryFinder.compute_structure_environments": _defaults(
            LocalGeometryFinder.compute_structure_environments
        ),
        "SimplestChemenvStrategy.__init__": _defaults(SimplestChemenvStrategy.__init__),
        "SimplestChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE": repr(
            getattr(SimplestChemenvStrategy, "DEFAULT_SYMMETRY_MEASURE_TYPE", None)
        ),
        "CrystalNN.__init__": _defaults(CrystalNN.__init__),
        "VoronoiNN.__init__": _defaults(VoronoiNN.__init__),
        "SpacegroupAnalyzer.__init__": _defaults(SpacegroupAnalyzer.__init__),
    }
    try:
        report["pymatgen-core"] = version("pymatgen-core")
    except Exception:  # noqa: BLE001 - absent in the reference environment
        report["pymatgen-core"] = None
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
