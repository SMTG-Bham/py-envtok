"""Strict configuration for the experimental finite-molecule profile."""

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from envtok.errors import ConfigError

MAX_ATOMIC_NUMBER = 118

DEFAULTS: dict[str, dict[str, Any]] = {
    "source": {
        "adapter": "ase_extxyz",
        "path": None,
        "label_namespace": None,
        "dataset": "molecular",
        "revision": "unspecified",
        "split": "train",
        "family_key": "family_id",
        "label_sources": None,
        "label_units": None,
    },
    "state": {
        "charge_key": "total_charge",
        "multiplicity_key": "spin_multiplicity",
        "unknown_policy": "separate_and_report",
        "strict_training_export": True,
    },
    "split": {
        "preserve_source_splits": True,
        "group_related_states": True,
        "group_trajectories_and_scans": True,
        "validation_fraction": 0.0,
    },
    "chemistry": {
        "graph_radius": 2,
        "prefer_supplied_graph": True,
        "allow_validated_perception": False,
        "ambiguous_graph": "geometry_fallback",
    },
    "geometry": {
        "backend": "soap",
        "species": [],
        "r_cut_angstrom": 5.0,
        "n_max": 4,
        "l_max": 4,
        "sigma_angstrom": 0.5,
        "average": "off",
        "normalize": True,
        "cache_dtype": "float32",
        "radial_bins": 32,
        "distance_bin_angstrom": 0.25,
        "torsion_bin_degrees": 30.0,
        "separation_bin_angstrom": 1.0,
    },
    "hierarchy": {
        "min_source_support": 20,
        "min_selected_representatives": 1,
        "protect_charge_and_multiplicity": True,
        "protect_known_central_formal_charge": True,
        "rare_policy": "protect",
    },
    "selection": {
        "mode": "cover",
        "cost": "atoms",
        "atom_fraction": None,
        "atom_budget": None,
        "chemical_core": "greedy_cover",
        "geometric_refinement": "reservoir_fps_regions",
        "prototype_region_radius": 0.20,
        "prototype_cap": 16,
        "reservoir_size": 256,
        "background_reserve_fraction": 0.0,
        "unresolved_constraints": "report_without_relaxing",
        "prune": True,
    },
    "reproducibility": {"seed": 20261105, "stable_tie_break": "record_id"},
    "execution": {"workdir": "runs/molecular", "resume": True},
    "output": {"format": "extxyz"},
}


def _choice(data: dict[str, Any], key: str, choices: tuple[Any, ...]) -> None:
    if data[key] not in choices:
        raise ConfigError(f"{key} must be one of {choices}; got {data[key]!r}")


def _number(
    data: dict[str, Any], key: str, *, minimum: float = 0, integer: bool = False
) -> None:
    value = data[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < minimum
        or (integer and not isinstance(value, int))
    ):
        raise ConfigError(
            f"{key} must be a finite {'integer' if integer else 'number'} >= {minimum}"
        )


def _validate(data: dict[str, Any]) -> None:  # noqa: PLR0912, PLR0915 - explicit schema checks
    source = data["source"]
    _choice(source, "adapter", ("ase_extxyz",))
    for key in ("path", "label_namespace", "dataset", "revision", "family_key"):
        if not isinstance(source[key], str) or not source[key].strip():
            raise ConfigError(f"source.{key} must be a nonempty string")
    _choice(source, "split", ("train", "validation", "test", "unsplit"))
    for key in ("label_sources", "label_units"):
        value = source[key]
        if value is not None and (
            not isinstance(value, dict)
            or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in value.items()
            )
        ):
            raise ConfigError(f"source.{key} must map label names to strings")
    state = data["state"]
    _choice(state, "unknown_policy", ("separate_and_report", "reject"))
    for key in ("charge_key", "multiplicity_key"):
        if not isinstance(state[key], str) or not state[key]:
            raise ConfigError(f"state.{key} must be a nonempty string")
    fixed: dict[str, dict[str, Any]] = {
        "split": {
            "preserve_source_splits": True,
            "group_related_states": True,
            "group_trajectories_and_scans": True,
        },
        "chemistry": {
            "prefer_supplied_graph": True,
            "ambiguous_graph": "geometry_fallback",
        },
        "geometry": {"average": "off", "normalize": True, "cache_dtype": "float32"},
        "hierarchy": {
            "protect_charge_and_multiplicity": True,
            "protect_known_central_formal_charge": True,
        },
        "selection": {
            "cost": "atoms",
            "chemical_core": "greedy_cover",
            "unresolved_constraints": "report_without_relaxing",
        },
        "reproducibility": {"stable_tie_break": "record_id"},
    }
    for section, values in fixed.items():
        for key, expected in values.items():
            if (
                type(data[section][key]) is not type(expected)
                or data[section][key] != expected
            ):
                raise ConfigError(
                    f"{section}.{key} must be {expected!r} in molecular_v1"
                )
    for section, key in (
        ("state", "strict_training_export"),
        ("chemistry", "allow_validated_perception"),
        ("execution", "resume"),
        ("selection", "prune"),
    ):
        if not isinstance(data[section][key], bool):
            raise ConfigError(f"{section}.{key} must be boolean")
    for section, keys in {
        "geometry": ("n_max",),
        "hierarchy": ("min_source_support", "min_selected_representatives"),
        "selection": ("prototype_cap", "reservoir_size"),
    }.items():
        for key in keys:
            _number(data[section], key, minimum=1, integer=True)
    _number(data["chemistry"], "graph_radius", integer=True)
    _number(data["geometry"], "radial_bins", minimum=2, integer=True)
    _number(data["geometry"], "l_max", integer=True)
    _number(data["reproducibility"], "seed", integer=True)
    for key in (
        "r_cut_angstrom",
        "sigma_angstrom",
        "distance_bin_angstrom",
        "torsion_bin_degrees",
        "separation_bin_angstrom",
    ):
        _number(data["geometry"], key, minimum=1e-12)
    _choice(data["geometry"], "backend", ("soap", "radial"))
    species = data["geometry"]["species"]
    if not isinstance(species, list) or any(
        type(z) is not int or not 1 <= z <= MAX_ATOMIC_NUMBER for z in species
    ):
        raise ConfigError("geometry.species must be a list of atomic numbers in 1..118")
    _choice(data["hierarchy"], "rare_policy", ("protect", "optional"))
    selection = data["selection"]
    _choice(selection, "mode", ("cover", "budget"))
    _choice(selection, "geometric_refinement", ("reservoir_fps_regions", "none"))
    _number(selection, "prototype_region_radius", minimum=1e-12)
    for section, key in (
        ("split", "validation_fraction"),
        ("selection", "background_reserve_fraction"),
    ):
        _number(data[section], key)
        if data[section][key] >= 1:
            raise ConfigError(f"{section}.{key} must be less than 1")
    budget, fraction = selection["atom_budget"], selection["atom_fraction"]
    if budget is not None:
        _number(selection, "atom_budget", integer=True)
    if fraction is not None:
        _number(selection, "atom_fraction", minimum=1e-12)
        if fraction > 1:
            raise ConfigError("atom_fraction must be at most 1")
    if budget is not None and fraction is not None:
        raise ConfigError("set atom_budget or atom_fraction, not both")
    if selection["mode"] == "budget" and budget is None and fraction is None:
        raise ConfigError("budget mode requires atom_budget or atom_fraction")
    if selection["mode"] == "cover" and (budget is not None or fraction is not None):
        raise ConfigError("cover mode does not accept a budget; use mode: budget")
    if selection["mode"] == "cover" and selection["background_reserve_fraction"]:
        raise ConfigError("a background reserve requires budget mode")
    _choice(data["output"], "format", ("extxyz", "none"))
    if (
        not isinstance(data["execution"]["workdir"], str)
        or not data["execution"]["workdir"]
    ):
        raise ConfigError("execution.workdir must be a nonempty path")


@dataclass(frozen=True, slots=True)
class MolecularConfig:
    """Resolved molecular settings with paths relative to the YAML file."""

    data: dict[str, Any]
    base_dir: Path

    def resolve(self, path: str) -> Path:
        """Return an absolute path relative to this configuration."""
        return (self.base_dir / path).resolve()

    def resolved(self) -> dict[str, Any]:
        """Return all defaults with source and work paths made absolute."""
        data = copy.deepcopy(self.data)
        data["source"]["path"] = str(self.resolve(data["source"]["path"]))
        data["execution"]["workdir"] = str(self.resolve(data["execution"]["workdir"]))
        return data


def load_molecular_config(source: str | Path | Mapping[str, Any]) -> MolecularConfig:
    """Read a strict molecular configuration without silently ignoring settings."""
    base = Path.cwd()
    if isinstance(source, str | Path):
        path = Path(source)
        base = path.resolve().parent
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read molecular configuration: {exc}") from exc
    else:
        raw = dict(source)
    if not isinstance(raw, dict) or raw.get("profile") != "molecular_v1":
        raise ConfigError("molecular configuration requires profile: molecular_v1")
    unknown = set(raw) - {"profile", *DEFAULTS}
    if unknown:
        raise ConfigError(f"unknown molecular sections: {sorted(unknown)}")
    data: dict[str, Any] = {"profile": "molecular_v1"}
    for name, defaults in DEFAULTS.items():
        given = raw.get(name, {})
        if not isinstance(given, dict) or set(given) - set(defaults):
            raise ConfigError(f"invalid or unknown keys in molecular section {name!r}")
        data[name] = copy.deepcopy(defaults) | given
    _validate(data)
    return MolecularConfig(data, base)
