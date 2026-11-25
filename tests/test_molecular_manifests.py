"""Columnar molecular manifests: round trips, legacy layouts and tampering."""

import hashlib
import json
import shutil
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from envtok.errors import ConfigError, StaleCacheError
from envtok.molecular.geometry import GeometryRegions
from envtok.molecular.manifests import (
    TARGET_TABLES,
    TARGETS,
    Built,
    FrozenTargets,
    write_targets,
)
from envtok.molecular.pipeline import MolecularPipeline
from envtok.molecular.selection import Candidate, HierarchyResult, Target

EXAMPLE = Path(__file__).parents[1] / "examples" / "molecular"
TABLES = (
    "tokens.parquet",
    "incidence.parquet",
    "regions.parquet",
    "hierarchy.parquet",
    "classes.parquet",
    "requirements.parquet",
    "geometry.parquet",
    "prototypes.npy",
    "overflow.parquet",
    "atom_classes.npy",
)


@pytest.fixture
def example_config(tmp_path: Path) -> dict[str, Any]:
    """The bundled molecular example, which has rare classes and overflow."""
    config: dict[str, Any] = yaml.safe_load((EXAMPLE / "config.yaml").read_text())
    shutil.copy(EXAMPLE / "molecules.extxyz", tmp_path / "molecules.extxyz")
    config["source"]["path"] = str(tmp_path / "molecules.extxyz")
    config["execution"]["workdir"] = str(tmp_path / "runs")
    # Radial descriptors need no optional DScribe install.
    config["geometry"]["backend"] = "radial"
    config["selection"]["prototype_region_radius"] = 0.01
    config["selection"]["prototype_cap"] = 1
    config["hierarchy"]["min_source_support"] = 8
    config["hierarchy"]["rare_policy"] = "optional"
    return config


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_document(built: Built, signatures: dict[str, str]) -> dict[str, Any]:
    """``targets.json`` exactly as runs before molecular_v1.6 wrote it."""
    candidates, mandatory, hierarchy, regions = built
    return {
        "mapping": hierarchy.mapping,
        "lineage": hierarchy.lineage,
        "support": hierarchy.support,
        "group_support": hierarchy.group_support,
        "source_support": hierarchy.source_support,
        "source_group_support": hierarchy.source_group_support,
        "rare_targets": sorted(hierarchy.rare_targets),
        "optional_targets": sorted(hierarchy.optional_targets),
        "requirements": {key: asdict(value) for key, value in mandatory.items()},
        "prototypes": {
            key: value.tolist() for key, value in regions.prototypes.items()
        },
        "region_targets": {
            key: sorted(value) for key, value in regions.targets.items()
        },
        "overflow": regions.overflow,
        "region_radius": regions.radius,
        "sampled_counts": regions.sampled_counts,
        "signatures": signatures,
        "incidence": [
            {
                "record_id": row.record_id,
                "cost": row.cost,
                "group_id": row.group_id,
                "targets": sorted(row.targets),
                "refinement_targets": sorted(row.refinement_targets),
            }
            for row in candidates
        ],
    }


def write_legacy(directory: Path, built: Built, signatures: dict[str, str]) -> None:
    (directory / TARGETS).write_text(
        json.dumps(legacy_document(built, signatures), sort_keys=True, indent=2) + "\n"
    )
    shutil.rmtree(directory / TARGET_TABLES, ignore_errors=True)


def as_compared(built: Built) -> list[Any]:
    """Every loaded value with its dict order and Python types."""
    candidates, mandatory, hierarchy, regions = built

    def typed(value: Any) -> Any:
        if isinstance(value, dict):
            return [(key, typed(item)) for key, item in value.items()]
        if isinstance(value, frozenset):
            return sorted(value)
        if isinstance(value, tuple):
            return [typed(item) for item in value]
        if isinstance(value, np.ndarray):
            return (value.dtype.str, value.shape, value.tobytes())
        if isinstance(value, Candidate | Target):
            return [typed(getattr(value, f.name)) for f in fields(value)]
        return (type(value).__name__, value)

    return [
        typed(candidates),
        typed(mandatory),
        [typed(getattr(hierarchy, f.name)) for f in fields(hierarchy)],
        [typed(getattr(regions, f.name)) for f in fields(regions)],
    ]


def test_columnar_targets_load_exactly_as_the_legacy_document(
    example_config: dict[str, Any], tmp_path: Path
) -> None:
    pipeline = MolecularPipeline(example_config)
    pipeline.select()
    columnar = FrozenTargets(pipeline.directory).load()
    assert columnar[3].overflow
    assert columnar[2].optional_targets < columnar[2].support.keys()
    assert columnar[2].optional_targets == columnar[2].rare_targets
    signatures = {
        token: signature
        for feature in pipeline.featurize()
        for token, signature in feature.signatures.items()
    }
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    write_legacy(legacy, columnar, signatures)
    assert as_compared(FrozenTargets(legacy).load()) == as_compared(columnar)
    # Reloaded candidates share each record's region set as its refinements.
    candidates, _, _, regions = columnar
    assert all(
        row.refinement_targets is regions.targets[row.record_id] for row in candidates
    )


def test_saved_run_with_legacy_targets_audits_and_exports_identically(
    example_config: dict[str, Any], tmp_path: Path
) -> None:
    pipeline = MolecularPipeline(example_config)
    pipeline.run()
    coverage = (pipeline.directory / "coverage.json").read_bytes()
    exported = (pipeline.directory / "subset.extxyz").read_bytes()
    built = FrozenTargets(pipeline.directory).load()
    signatures = {
        token: signature
        for feature in pipeline.featurize()
        for token, signature in feature.signatures.items()
    }
    write_legacy(pipeline.directory, built, signatures)
    provenance_path = pipeline.directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["targets_sha256"] = digest(pipeline.directory / TARGETS)
    provenance_path.write_text(json.dumps(provenance))
    loaded = MolecularPipeline.load_run(pipeline.directory)
    loaded.audit()
    assert (pipeline.directory / "coverage.json").read_bytes() == coverage
    assert loaded.export(tmp_path / "again.extxyz").read_bytes() == exported
    assert as_compared(loaded.build_targets()) == as_compared(built)
    targets = FrozenTargets(pipeline.directory)
    assert targets.overflow_count() == len(built[3].overflow)
    assert list(targets.prototypes()) == list(built[3].prototypes)


@pytest.mark.parametrize("table", TABLES)
@pytest.mark.parametrize("tamper", ["remove", "append"])
def test_every_target_table_is_checked_before_use(
    example_config: dict[str, Any], table: str, tamper: str
) -> None:
    pipeline = MolecularPipeline(example_config)
    pipeline.select()
    path = pipeline.directory / TARGET_TABLES / table
    if tamper == "remove":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\0")
    for action in (pipeline.audit, pipeline.select):
        with pytest.raises(StaleCacheError, match="frozen molecular targets"):
            action()
    with pytest.raises(StaleCacheError, match="frozen molecular targets"):
        MolecularPipeline.load_run(pipeline.directory)


@pytest.mark.parametrize(
    "manifest",
    [
        {"layout": "unknown", "region_radius": 0.2, "files": {}},
        {"layout": "columnar-v1", "region_radius": 0.2, "files": []},
        {"incidence": []},
    ],
)
def test_unrecognised_target_manifests_fail_closed(
    tmp_path: Path, manifest: dict[str, Any]
) -> None:
    (tmp_path / TARGETS).write_text(json.dumps(manifest))
    with pytest.raises(StaleCacheError, match="frozen molecular targets"):
        FrozenTargets(tmp_path)


def minimal_targets(**regions: Any) -> Built:
    candidates = (Candidate("r", 2, frozenset({"t"}), frozenset(), None),)
    hierarchy = HierarchyResult(
        targets={"r": frozenset({"t"})},
        mapping={"t": "t"},
        support={"t": 1},
        group_support={"t": 1},
        rare_targets=frozenset({"t"}),
        optional_targets=frozenset(),
        lineage={"t": ("t",)},
        source_support={"t": 1},
        source_group_support={"t": 1},
    )
    geometry = GeometryRegions(
        targets={"r": frozenset()},
        prototypes={},
        overflow=(),
        sampled_counts={},
        radius=0.2,
    )
    return candidates, {"t": Target()}, hierarchy, replace(geometry, **regions)


ATOMS = (["t"], np.zeros(2, dtype=np.int32))


def test_targets_without_prototypes_round_trip(tmp_path: Path) -> None:
    built = minimal_targets()
    write_targets(tmp_path, built, ATOMS)
    targets = FrozenTargets(tmp_path)
    assert as_compared(targets.load()) == as_compared(built)
    assert np.load(tmp_path / TARGET_TABLES / "prototypes.npy").shape == (0, 0)
    classes = targets.atom_classes()
    assert classes is not None
    assert [classes[0][code] for code in classes[1]] == ["t", "t"]


def test_target_writer_rejects_fractional_costs_and_mixed_dimensions(
    tmp_path: Path,
) -> None:
    candidates, mandatory, hierarchy, regions = minimal_targets()
    fractional = (replace(candidates[0], cost=2.5),)
    with pytest.raises(ConfigError, match="whole-number atom costs"):
        write_targets(tmp_path, (fractional, mandatory, hierarchy, regions), ATOMS)
    mixed = minimal_targets(
        prototypes={"a": np.zeros((1, 2)), "b": np.zeros((1, 3))},
        sampled_counts={"a": 1, "b": 1},
    )
    with pytest.raises(ConfigError, match="one descriptor dimension"):
        write_targets(tmp_path, mixed, ATOMS)
