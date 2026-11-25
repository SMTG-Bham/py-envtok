"""Regression checks for frozen molecular evidence and source integrity."""

import copy
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms

from envtok.errors import ConfigError, StaleCacheError
from envtok.molecular import feature_cache
from envtok.molecular import pipeline as pipeline_module
from envtok.molecular.data import prepare_records
from envtok.molecular.pipeline import MolecularPipeline
from envtok.molecular.representations import MolecularFeatures, RepresentationSettings


@pytest.fixture
def review_config(tmp_path: Path) -> dict[str, Any]:
    frames = []
    for sid, distance in (("a", 0.7), ("b", 1.1)):
        atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]])
        atoms.info.update(total_charge=0.0, spin_multiplicity=1, sid=sid, split="train")
        frames.append(atoms)
    source = tmp_path / "molecules.extxyz"
    ase.io.write(source, frames, format="extxyz")
    return {
        "profile": "molecular_v1",
        "source": {"path": str(source), "label_namespace": "test"},
        "geometry": {"backend": "radial"},
        "execution": {"workdir": str(tmp_path / "runs")},
    }


@pytest.mark.parametrize(
    "stage", ["ingest", "featurize", "build_targets", "select", "audit", "export"]
)
def test_mutated_source_rejected_even_with_cached_state(
    review_config: dict[str, Any], stage: str
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    source = Path(review_config["source"]["path"])
    source.write_text(source.read_text() + "\n")
    with pytest.raises(StaleCacheError, match="source content changed"):
        getattr(pipeline, stage)()
    with pytest.raises(StaleCacheError, match="source content changed"):
        MolecularPipeline.load_run(pipeline.directory)


def test_source_removed_and_mutated_configuration_rejected(
    review_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.settings["hierarchy"]["min_source_support"] = 99
    with pytest.raises(StaleCacheError, match="configuration changed"):
        pipeline.ingest()
    fresh = MolecularPipeline(review_config)
    Path(review_config["source"]["path"]).unlink()
    with pytest.raises(StaleCacheError, match="unavailable"):
        fresh.ingest()


def test_source_change_during_ingestion_rejected(
    review_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MolecularPipeline(review_config)
    source = Path(review_config["source"]["path"])

    def change_source(*args: Any, **kwargs: Any) -> Any:
        prepared = prepare_records(*args, **kwargs)
        source.write_text(source.read_text() + "\n")
        return prepared

    monkeypatch.setattr("envtok.molecular.pipeline.prepare_records", change_source)
    with pytest.raises(StaleCacheError, match="source content changed"):
        pipeline.ingest()
    assert not (pipeline.directory / "ingestion.json").exists()


def test_moved_run_uses_frozen_targets_in_changed_runtime(
    review_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = MolecularPipeline(review_config)
    original.select()
    moved = tmp_path / "moved-selection"
    original.directory.rename(moved)
    changed_software = copy.deepcopy(original.software)
    changed_software["packages"]["unrelated-runtime-package"] = "new"
    monkeypatch.setattr("envtok.molecular.pipeline._software", lambda: changed_software)
    frozen = {
        name: (moved / name).read_bytes()
        for name in (
            "targets.json",
            "ingestion.json",
            "splits.json",
            "software-lock.json",
            "config.resolved.yaml",
        )
    }
    loaded = MolecularPipeline.load_run(moved)
    assert loaded.directory == moved
    assert loaded.fingerprint == original.fingerprint
    assert loaded.build_targets()[1] == original.build_targets()[1]
    assert loaded.featurize()[0].record_id == original.featurize()[0].record_id

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("auditing must not refit representations or targets")

    monkeypatch.setattr(loaded, "build_targets", forbidden)
    monkeypatch.setattr("envtok.molecular.pipeline.iter_features", forbidden)
    report = loaded.audit()
    assert report["complete"]
    assert not report["software_environment_matches_selection"]
    assert report["geometry_audit_status"] == "available"
    assert loaded.export(tmp_path / "moved.extxyz").is_file()
    assert all((moved / name).read_bytes() == value for name, value in frozen.items())


@pytest.mark.parametrize("missing", ["features.json", "descriptors.npy"])
def test_frozen_coverage_and_export_without_geometry_cache(
    review_config: dict[str, Any], missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    (pipeline.feature_directory / missing).unlink()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("missing audit cache must not trigger fitting")

    monkeypatch.setattr("envtok.molecular.pipeline.iter_features", forbidden)
    loaded = MolecularPipeline.load_run(pipeline.directory)
    report = loaded.audit()
    assert report["complete"]
    assert report["nearest_selected_distances"] is None
    assert report["geometry_audit_status"] == "unavailable_feature_cache"
    assert loaded.export().is_file()
    with pytest.raises(StaleCacheError, match="missing or corrupt"):
        loaded.featurize()


@pytest.mark.parametrize("tamper", ["remove", "append"])
def test_frozen_target_tampering_fails_before_audit_or_reselection(
    review_config: dict[str, Any], tamper: str
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    targets = pipeline.directory / "targets.json"
    if tamper == "remove":
        targets.unlink()
    else:
        targets.write_text(targets.read_text() + "\n")
    for action in (pipeline.audit, pipeline.select):
        with pytest.raises(StaleCacheError, match="frozen molecular targets"):
            action()
    with pytest.raises(StaleCacheError, match="frozen molecular targets"):
        MolecularPipeline.load_run(pipeline.directory)


def test_saved_config_tampering_rejected(review_config: dict[str, Any]) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    path = pipeline.directory / "config.resolved.yaml"
    settings = yaml.safe_load(path.read_text())
    settings["hierarchy"]["min_source_support"] += 1
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(StaleCacheError, match="configuration or software lock changed"):
        MolecularPipeline.load_run(pipeline.directory)


@pytest.mark.parametrize("content", [None, "[", "[]", "{}"])
def test_saved_provenance_missing_or_invalid_fails_closed(
    review_config: dict[str, Any], content: str | None
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    path = pipeline.directory / "provenance.json"
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    with pytest.raises(StaleCacheError, match="cannot read molecular manifest"):
        MolecularPipeline.load_run(pipeline.directory)


def test_feature_metadata_tampering_is_detected_before_resuming(
    review_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.featurize()
    path = pipeline.feature_directory / "features.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(StaleCacheError, match="feature metadata"):
        MolecularPipeline(review_config).featurize()


def test_frozen_feature_metadata_must_match_selection(
    review_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    path = pipeline.feature_directory / "features.json"
    path.write_text(path.read_text() + "\n")
    commit = {"features_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (pipeline.feature_directory / "commit.json").write_text(json.dumps(commit))
    with pytest.raises(
        StaleCacheError, match="frozen molecular feature metadata changed"
    ):
        pipeline.audit()


def test_saved_features_cannot_be_changed_for_benchmark_resumption(
    review_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    path = pipeline.feature_directory / "features.json"
    path.write_text(path.read_text() + "\n")
    (pipeline.feature_directory / "commit.json").write_text(
        json.dumps({"features_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    loaded = MolecularPipeline.load_run(pipeline.directory)
    with pytest.raises(
        StaleCacheError, match="frozen molecular feature metadata changed"
    ):
        loaded.featurize()


def test_run_audits_once_and_explicit_export_reaudits(
    review_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[None] = []
    audit = MolecularPipeline.audit

    def counted(self: MolecularPipeline) -> dict[str, Any]:
        calls.append(None)
        return audit(self)

    monkeypatch.setattr(MolecularPipeline, "audit", counted)
    pipeline = MolecularPipeline(review_config)
    pipeline.run()
    assert len(calls) == 1
    path = pipeline.feature_directory / "features.json"
    path.write_text(path.read_text() + "\n")
    (pipeline.feature_directory / "commit.json").write_text(
        json.dumps({"features_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    )
    with pytest.raises(
        StaleCacheError, match="frozen molecular feature metadata changed"
    ):
        pipeline.export()
    assert len(calls) == 2


def counting_source_hashes(source: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    hashed: list[Path] = []
    file_hash = pipeline_module._file_hash

    def counted(path: Path) -> str:
        if path == source:
            hashed.append(path)
        return file_hash(path)

    monkeypatch.setattr(pipeline_module, "_file_hash", counted)
    return hashed


def test_settled_source_hashed_in_full_only_at_boundaries_or_on_stat_change(
    review_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Treat the freshly written fixture as long settled.
    monkeypatch.setattr(pipeline_module, "_RACY_NS", -(10**12))
    source = Path(review_config["source"]["path"])
    hashed = counting_source_hashes(source, monkeypatch)
    pipeline = MolecularPipeline(review_config)
    pipeline.run()
    # Construction, select() and the export; other stages compare stats.
    assert len(hashed) == 3
    hashed.clear()
    pipeline.ingest()
    pipeline.audit()
    assert not hashed
    os.utime(source)
    pipeline.ingest()
    pipeline.featurize()
    assert len(hashed) == 1


def test_same_size_rewrite_soon_after_a_check_is_detected(
    review_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(review_config["source"]["path"])
    hashed = counting_source_hashes(source, monkeypatch)
    pipeline = MolecularPipeline(review_config)
    pipeline.featurize()
    # A source changed this recently is re-hashed rather than trusted by stat.
    hashed.clear()
    pipeline.ingest()
    assert len(hashed) == 1
    text = source.read_text()
    source.write_text(text.replace("sid=a", "sid=z"))
    assert source.stat().st_size == len(text.encode())
    with pytest.raises(StaleCacheError, match="source content changed"):
        pipeline.featurize()


def test_parallel_featurisation_matches_serial_without_new_run(
    review_config: dict[str, Any],
) -> None:
    serial = MolecularPipeline(review_config)
    serial.run()
    parallel_config = copy.deepcopy(review_config)
    parallel_config["execution"] |= {"workers": 2, "threads_per_worker": 1}
    # Worker counts stay out of fingerprints, so this addresses the same run.
    parallel = MolecularPipeline(parallel_config)
    assert parallel.directory == serial.directory
    assert parallel.feature_directory == serial.feature_directory
    parallel_config["execution"]["resume"] = False
    recomputed = MolecularPipeline(parallel_config).featurize()
    assert [
        (f.record_id, f.token_hierarchy, f.signatures, f.descriptors.tobytes())
        for f in recomputed
    ] == [
        (f.record_id, f.token_hierarchy, f.signatures, f.descriptors.tobytes())
        for f in serial.featurize()
    ]


def test_saved_configuration_without_worker_settings_still_loads(
    review_config: dict[str, Any],
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    path = pipeline.directory / "config.resolved.yaml"
    settings = yaml.safe_load(path.read_text())
    for key in ("workers", "threads_per_worker"):
        del settings["execution"][key]
    path.write_text(yaml.safe_dump(settings, sort_keys=True))
    loaded = MolecularPipeline.load_run(pipeline.directory)
    assert loaded.fingerprint == pipeline.fingerprint
    assert loaded.settings["execution"]["workers"] == 1


def test_feature_cache_rejects_rows_that_do_not_match_records(
    review_config: dict[str, Any], tmp_path: Path
) -> None:
    pipeline = MolecularPipeline(review_config)
    settings = RepresentationSettings(backend="radial", species=(1,))
    with pytest.raises(ConfigError, match="atom count"):
        pipeline._write_features(iter(()), 1, settings)
    manifest = {"records": [{"token_hierarchy": [["t"]]}]}
    path = tmp_path / "descriptors.npy"
    for descriptors in (np.zeros((2, 3), np.float32), np.zeros(1, np.float32)):
        np.save(path, descriptors)
        with pytest.raises(StaleCacheError, match="missing or corrupt"):
            feature_cache.legacy_blocks(manifest, path)


def write_legacy_feature_cache(
    pipeline: MolecularPipeline, features: Sequence[MolecularFeatures], layout: str
) -> None:
    """Rewrite a cache in the molecular_v1.5 or molecular_v1.4 layout.

    v1.5 kept JSON records and one signature table; v1.4 also stored every
    record's descriptors as a compressed npz member and its own signatures.
    """
    directory = pipeline.feature_directory
    settings = json.loads((directory / "features.json").read_text())["settings"]
    records: list[dict[str, Any]] = [
        {
            "record_id": f.record_id,
            "family_id": f.family_id,
            "state_key": list(f.state_key),
            "token_hierarchy": [list(path) for path in f.token_hierarchy],
            "graph_status": f.graph_status,
            "safeguard_targets": list(f.safeguard_targets),
            "zero_descriptor_atoms": list(f.zero_descriptor_atoms),
        }
        for f in features
    ]
    table = {token: f.signatures[token] for f in features for token in f.signatures}
    manifest: dict[str, Any] = {"records": records, "settings": settings}
    if layout == "v1.5":
        manifest["signatures"] = dict(sorted(table.items()))
        arrays = directory / "descriptors.npy"
    else:
        for row, feature in zip(records, features, strict=True):
            row["signatures"] = dict(feature.signatures)
        arrays = directory / "descriptors.npz"
        np.savez_compressed(
            arrays,
            allow_pickle=False,
            **{f"f{i}": np.asarray(f.descriptors) for i, f in enumerate(features)},
        )
        (directory / "descriptors.npy").unlink()
    manifest["arrays_sha256"] = hashlib.sha256(arrays.read_bytes()).hexdigest()
    for name in ("records.parquet", "signatures.parquet"):
        (directory / name).unlink()
    (directory / "features.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256((directory / "features.json").read_bytes()).hexdigest()
    (directory / "commit.json").write_text(json.dumps({"features_sha256": digest}))
    provenance = json.loads((pipeline.directory / "provenance.json").read_text())
    provenance["features_sha256"] = digest
    (pipeline.directory / "provenance.json").write_text(json.dumps(provenance))


@pytest.mark.parametrize("layout", ["v1.4", "v1.5"])
def test_saved_run_with_a_legacy_feature_cache_keeps_its_geometry_audit(
    review_config: dict[str, Any], layout: str
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    expected = pipeline.audit()["nearest_selected_distances"]
    features = pipeline.featurize()
    write_legacy_feature_cache(pipeline, features, layout)
    loaded = MolecularPipeline.load_run(pipeline.directory)
    report = loaded.audit()
    assert report["geometry_audit_status"] == "available"
    assert report["nearest_selected_distances"] == expected
    for old, new in zip(features, loaded.featurize(), strict=True):
        assert new.signatures == old.signatures
        assert new.token_hierarchy == old.token_hierarchy
        assert np.array_equal(new.descriptors, old.descriptors)
