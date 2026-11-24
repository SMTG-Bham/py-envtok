"""Regression checks for frozen molecular evidence and source integrity."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import ase.io
import pytest
import yaml
from ase import Atoms

from envtok.errors import StaleCacheError
from envtok.molecular.data import prepare_records
from envtok.molecular.pipeline import MolecularPipeline


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
    monkeypatch.setattr("envtok.molecular.pipeline.featurize_records", forbidden)
    report = loaded.audit()
    assert report["complete"]
    assert not report["software_environment_matches_selection"]
    assert report["geometry_audit_status"] == "available"
    assert loaded.export(tmp_path / "moved.extxyz").is_file()
    assert all((moved / name).read_bytes() == value for name, value in frozen.items())


@pytest.mark.parametrize("missing", ["features.json", "descriptors.npz"])
def test_frozen_coverage_and_export_without_geometry_cache(
    review_config: dict[str, Any], missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MolecularPipeline(review_config)
    pipeline.select()
    (pipeline.feature_directory / missing).unlink()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("missing audit cache must not trigger fitting")

    monkeypatch.setattr("envtok.molecular.pipeline.featurize_records", forbidden)
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


def test_run_audits_once_and_export_reaudits_changed_inputs(
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
    pipeline.export()
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
