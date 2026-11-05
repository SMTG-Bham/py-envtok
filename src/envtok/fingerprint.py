"""Canonical hashing and the layered cache fingerprints.

Every persisted stage records the fingerprint of everything that determines
its content, so a cache is reused only when it is still valid:

* **source**: adapter kind, file identities (size, mtime, optional SHA-256)
  and record options.
* **geometry**: neighbour, polyhedron and symmetry settings, the radius table,
  the asymmetry decay length and backend library versions.
* **corpus**: element exclusions applied before geometry.
* **token**: geometry fingerprint plus bins, moment policy and, when enabled,
  the charge policy and key.
* **classes/selection**: token fingerprint plus eligibility, ladder, T,
  constraints, cost and tie policy (see :mod:`envtok.coarsen`).

Changing T never repeats geometry; changing bins reuses raw measurements.
Python's process-randomised ``hash()`` is never used for persistent identity.
"""

import dataclasses
import enum
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from envtok.profiles import CorpusPolicy, Profile

TOKENIZER_IMPLEMENTATION = 1
"""Bumped whenever the tokeniser code changes results for fixed settings."""

BACKEND_PACKAGES = ("ase", "numpy", "pymatgen", "pymatgen-core", "scipy", "spglib")


def json_default(value: object) -> object:
    """Serialise enums, sets, paths and dataclasses for :mod:`json`."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, frozenset | set):
        return sorted(value)
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    raise TypeError(f"cannot fingerprint {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Serialise ``value`` deterministically (sorted keys, no whitespace).

    Args:
        value: JSON-compatible data, dataclasses, enums, sets or paths.

    Returns:
        The canonical JSON text.
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=json_default
    )


def digest(value: object) -> str:
    """Return the SHA-256 hex digest of the canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def package_versions(
    names: tuple[str, ...] = BACKEND_PACKAGES,
) -> dict[str, str | None]:
    """Return installed versions of the named distributions (``None`` if absent)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def geometry_payload(profile: Profile) -> dict[str, Any]:
    """Return everything that determines raw geometric measurements."""
    radii = profile.measures.radii
    return {
        "reference_commit": profile.reference_commit,
        "geometry": profile.geometry,
        "radii": {"name": radii.name, "max_z": radii.max_z, "values": radii.values},
        "asymmetry_lambda": profile.measures.asymmetry_lambda,
        "backend": package_versions(),
        "implementation": TOKENIZER_IMPLEMENTATION,
    }


def corpus_payload(policy: CorpusPolicy) -> dict[str, Any]:
    """Return the corpus settings applied before geometry (not eligibility)."""
    return {
        "exclude_elements": sorted(policy.exclude_elements),
        "exclude_z_at_least": policy.exclude_z_at_least,
    }


def token_payload(profile: Profile) -> dict[str, Any]:
    """Return everything that determines token keys from raw measurements.

    The charge settings are included only when the charge field is enabled,
    so profiles without one keep the token fingerprint they always had.
    """
    payload: dict[str, Any] = {
        "geometry": digest(geometry_payload(profile)),
        "bins": profile.bins,
        "magnetic_policy": profile.magnetic_policy,
    }
    if profile.charge_enabled:
        payload["charge_policy"] = profile.charge_policy
        payload["charge_key"] = profile.charge_key
    return payload
