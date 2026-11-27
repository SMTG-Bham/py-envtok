"""Provenance records written next to every selection."""

from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from envtok import __version__
from envtok.execution import peak_rss_bytes
from envtok.fingerprint import package_versions
from envtok.profiles.reference import REFERENCE_COMMIT, REFERENCE_REPOSITORY

if TYPE_CHECKING:
    from envtok.selection import SelectionResult

DEPENDENCIES = (
    "ase",
    "duckdb",
    "numpy",
    "pyarrow",
    "pymatgen",
    "pymatgen-core",
    "pyyaml",
    "scipy",
    "spglib",
)


def collect(result: SelectionResult) -> dict[str, Any]:
    """Return the provenance of a selection.

    Args:
        result: The selection.

    Returns:
        Package and dependency versions, platform, layered fingerprints, the
        source description and resource measurements.
    """
    classes = result.classes
    cache = classes.token_cache
    manifest = cache.manifest
    return {
        "envtok": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "dependencies": package_versions(DEPENDENCIES),
        "reference": {"repository": REFERENCE_REPOSITORY, "commit": REFERENCE_COMMIT},
        "profile": classes.profile.name,
        "source": {
            "adapter": manifest["adapter_kind"],
            "fingerprint": manifest["source_fingerprint"],
        },
        "fingerprints": {
            **manifest["keys"],
            "index": cache.fingerprint,
            "classes": classes.fingerprint,
            "selection": result.fingerprint,
        },
        "cache": {
            "directory": str(cache.token_dir),
            "summary": cache.summary,
            "corpus_preset": manifest["corpus_preset"],
            "batch_size": manifest["batch_size"],
        },
        "resources": {"peak_rss_bytes": peak_rss_bytes()},
        "created": datetime.now(UTC).isoformat(),
    }
