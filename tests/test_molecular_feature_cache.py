"""Columnar molecular feature caches: round trips, sharing and tampering."""

import hashlib
import json
import shutil
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from envtok.errors import ConfigError, StaleCacheError
from envtok.molecular import feature_cache
from envtok.molecular.feature_cache import RecordSignatures, read_feature_cache
from envtok.molecular.pipeline import MolecularPipeline
from envtok.molecular.representations import (
    MolecularFeatures,
    RepresentationSettings,
    featurize_records,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "molecular"


@pytest.fixture
def example_config(tmp_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((EXAMPLE / "config.yaml").read_text())
    shutil.copy(EXAMPLE / "molecules.extxyz", tmp_path / "molecules.extxyz")
    config["source"]["path"] = str(tmp_path / "molecules.extxyz")
    config["execution"]["workdir"] = str(tmp_path / "runs")
    config["execution"]["resume"] = False
    return config


def described(feature: MolecularFeatures) -> list[Any]:
    values = []
    for field in fields(feature):
        value = getattr(feature, field.name)
        if field.name == "descriptors":
            value = (value.dtype.str, value.shape, value.tobytes())
        elif field.name == "signatures":
            value = list(value.items())
        values.append(value)
    return values


def rewrite(path: Path, data: bytes, pipeline: MolecularPipeline) -> None:
    """Replace a cache file and re-sign the manifest so only its digest differs."""
    path.write_bytes(data)
    manifest_path = pipeline.feature_directory / "features.json"
    manifest = json.loads(manifest_path.read_text())
    digest = hashlib.sha256(data).hexdigest()
    if path.name == "descriptors.npy":
        manifest["arrays_sha256"] = digest
    else:
        manifest["files"][path.name] = digest
    resign(pipeline, manifest)


def resign(pipeline: MolecularPipeline, manifest: dict[str, Any]) -> None:
    manifest_path = pipeline.feature_directory / "features.json"
    manifest_path.write_text(json.dumps(manifest))
    commit = {"features_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    (pipeline.feature_directory / "commit.json").write_text(json.dumps(commit))


def test_cached_features_equal_fresh_features_in_any_batch_size(
    example_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MolecularPipeline(example_config)
    cached = pipeline.featurize()
    saved = json.loads((pipeline.feature_directory / "features.json").read_text())
    settings = RepresentationSettings(
        **saved["settings"] | {"species": tuple(saved["settings"]["species"])}
    )
    fresh = featurize_records(pipeline.ingest().candidates, settings)
    assert [described(f) for f in cached] == [described(f) for f in fresh]
    monkeypatch.setattr(feature_cache, "_BATCH", 3)
    example_config["execution"]["workdir"] = str(tmp_path / "small-batches")
    batched = MolecularPipeline(example_config).featurize()
    assert [described(f) for f in batched] == [described(f) for f in fresh]
    resumed = MolecularPipeline(example_config | {"execution": {"resume": True}})
    assert [described(f) for f in resumed.featurize()] == [described(f) for f in fresh]


def test_loaded_features_share_tokens_paths_and_state(
    example_config: dict[str, Any],
) -> None:
    features = MolecularPipeline(example_config).featurize()
    paths = [path for f in features for path in f.token_hierarchy]
    tokens = [token for path in paths for token in path]
    states = [f.state_key for f in features]
    # One object per distinct value, and atoms do repeat paths.
    for values in (paths, tokens, states):
        assert len({id(value) for value in values}) == len(set(values))
    assert len(set(paths)) < len(paths)


def test_record_signatures_are_read_once_when_first_looked_up(
    example_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(example_config)
    features = pipeline.featurize()
    signatures = features[0].signatures
    assert isinstance(signatures, RecordSignatures)
    tokens = sorted({t for p in features[0].token_hierarchy for t in p})
    assert list(signatures) == sorted({*tokens, *features[0].safeguard_targets})
    assert len(signatures) == len(list(signatures))
    assert tokens[0] in signatures
    assert "missing" not in signatures
    assert repr(signatures) == f"RecordSignatures({len(signatures)} tokens)"
    with pytest.raises(KeyError):
        signatures["missing"]
    table = pipeline.feature_directory / "signatures.parquet"
    table.write_bytes(table.read_bytes() + b"\0")
    with pytest.raises(StaleCacheError, match="feature metadata"):
        signatures[tokens[0]]
    table.unlink()
    with pytest.raises(StaleCacheError, match="feature metadata"):
        signatures[tokens[0]]


def test_signatures_already_read_survive_later_changes(
    example_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(example_config)
    features = pipeline.featurize()
    expected = dict(features[0].signatures)
    (pipeline.feature_directory / "signatures.parquet").unlink()
    assert dict(features[1].signatures).keys() <= {
        t for f in features for t in f.signatures
    }
    assert dict(features[0].signatures) == expected


@pytest.mark.parametrize("name", ["records.parquet", "signatures.parquet"])
@pytest.mark.parametrize("tamper", ["remove", "append"])
def test_feature_tables_are_checked_before_resuming(
    example_config: dict[str, Any], name: str, tamper: str
) -> None:
    pipeline = MolecularPipeline(example_config)
    pipeline.featurize()
    path = pipeline.feature_directory / name
    if tamper == "remove":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"\0")
    example_config["execution"]["resume"] = True
    with pytest.raises(StaleCacheError, match="feature metadata"):
        MolecularPipeline(example_config).featurize()


@pytest.mark.parametrize("change", [{"layout": "unknown"}, {"files": []}])
def test_unrecognised_feature_manifests_fail_closed(
    example_config: dict[str, Any], change: dict[str, Any]
) -> None:
    pipeline = MolecularPipeline(example_config)
    pipeline.featurize()
    manifest = json.loads((pipeline.feature_directory / "features.json").read_text())
    resign(pipeline, manifest | change)
    example_config["execution"]["resume"] = True
    with pytest.raises(StaleCacheError, match="feature metadata"):
        MolecularPipeline(example_config).featurize()


@pytest.mark.parametrize("rows", ["manifest", "records"])
def test_descriptor_rows_must_match_the_records(
    example_config: dict[str, Any], rows: str
) -> None:
    pipeline = MolecularPipeline(example_config)
    pipeline.featurize()
    path = pipeline.feature_directory / "descriptors.npy"
    matrix = np.load(path)
    np.save(path, np.concatenate([matrix, matrix[:1]]))
    rewrite(path, path.read_bytes(), pipeline)
    if rows == "records":
        manifest = json.loads(
            (pipeline.feature_directory / "features.json").read_text()
        )
        manifest["atom_count"] += 1
        resign(pipeline, manifest)
    example_config["execution"]["resume"] = True
    with pytest.raises(StaleCacheError, match="descriptor cache"):
        MolecularPipeline(example_config).featurize()


def test_legacy_cache_without_its_descriptors_fails_closed(tmp_path: Path) -> None:
    manifest = {"records": [], "arrays_sha256": "0" * 64}
    with pytest.raises(StaleCacheError, match="descriptor cache"):
        read_feature_cache(tmp_path, manifest)


@pytest.mark.parametrize(
    "case", ["vector", "dimension", "hierarchy", "too_many", "too_few"]
)
def test_writer_rejects_inconsistent_descriptor_rows(
    example_config: dict[str, Any], case: str
) -> None:
    pipeline = MolecularPipeline(example_config)
    features = list(pipeline.featurize())
    natoms = sum(len(f.token_hierarchy) for f in features)
    first = features[0]
    if case == "vector":
        features[0] = replace(first, descriptors=np.zeros(3, np.float32))
    elif case == "dimension":
        features[1] = replace(
            features[1], descriptors=np.asarray(features[1].descriptors)[:, :-1]
        )
    elif case == "hierarchy":
        features[0] = replace(first, token_hierarchy=first.token_hierarchy[:-1])
    elif case == "too_many":
        natoms -= 1
    else:
        natoms += 1
    settings = RepresentationSettings(backend="soap", species=(1, 6, 8))
    directory = pipeline.feature_directory
    with pytest.raises(ConfigError, match="atom count"):
        feature_cache.write_feature_cache(directory, iter(features), natoms, settings)
    assert not list(directory.glob("*.partial"))
