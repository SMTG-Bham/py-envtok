"""End-to-end molecular state, budget, cache and command-line contracts."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.cli import main
from envtok.errors import ConfigError, CoverageError, StaleCacheError
from envtok.molecular import MolecularPipeline, feature_cache, load_molecular_config
from envtok.molecular.cli import dispatch, is_molecular
from envtok.molecular.pipeline import audit_selection
from envtok.molecular.selection import Candidate, Target


@pytest.fixture
def molecular_config(tmp_path: Path) -> dict[str, Any]:
    frames = []
    for index, (distance, charge, spin, split) in enumerate(
        [
            (0.7, 0, 1, "train"),
            (0.9, 0, 1, "train"),
            (1.0, 1, 2, "train"),
            (1.1, 0, 3, "train"),
            (3.0, -1, 2, "validation"),
        ]
    ):
        atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]])
        atoms.info.update(
            total_charge=charge,
            spin_multiplicity=spin,
            family_id=f"family-{index}",
            sid=str(index),
            split=split,
        )
        atoms.calc = SinglePointCalculator(
            atoms, energy=distance, forces=np.ones((2, 3))
        )
        frames.append(atoms)
    path = tmp_path / "molecules.extxyz"
    ase.io.write(path, frames, format="extxyz")
    return {
        "profile": "molecular_v1",
        "source": {"path": str(path), "label_namespace": "test/eV"},
        "geometry": {"backend": "radial"},
        "hierarchy": {"min_source_support": 2},
        "execution": {"workdir": str(tmp_path / "runs")},
    }


def test_full_pipeline_restart_export_and_state(
    molecular_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MolecularPipeline(molecular_config)
    directory = pipeline.run()
    report = json.loads((directory / "coverage.json").read_text())
    assert report["complete"]
    assert report["candidate_atoms"] == 8
    assert report["candidate_configurations"] == 4
    selected = ase.io.read(directory / "subset.extxyz", index=":")
    assert {
        (a.info["total_charge"], a.info["spin_multiplicity"]) for a in selected
    } == {(0, 1), (1, 2), (0, 3)}
    assert all(a.get_forces().shape == (2, 3) for a in selected)
    assert (directory / "software-lock.json").exists()
    ids = (directory / "selected_ids.txt").read_text()

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("restart must reuse descriptors")

    monkeypatch.setattr("envtok.molecular.pipeline.iter_features", forbidden)
    resumed = MolecularPipeline(directory / "config.resolved.yaml")
    assert resumed.run() == directory
    assert (directory / "selected_ids.txt").read_text() == ids
    assert resumed.featurize()[0].zero_descriptor_atoms == ()


@pytest.mark.parametrize("budget", [0, 2, 5])
def test_budget_never_relaxes_state_and_frozen_audit(
    molecular_config: dict[str, Any], budget: int
) -> None:
    molecular_config["selection"] = {"mode": "budget", "atom_budget": budget}
    pipeline = MolecularPipeline(molecular_config)
    result = pipeline.select()
    assert result.achieved_cost <= budget
    assert not result.complete
    report = pipeline.audit()
    assert report["status"] == "heuristic_incomplete"
    assert report["unmet_constraints"] == result.unmet
    assert "infeasibility certificate" in report["explanation"]
    pipeline.export()


def test_fraction_token_only_and_no_export(molecular_config: dict[str, Any]) -> None:
    molecular_config["selection"] = {
        "mode": "budget",
        "atom_fraction": 0.25,
        "geometric_refinement": "none",
    }
    molecular_config["output"] = {"format": "none"}
    pipeline = MolecularPipeline(load_molecular_config(molecular_config))
    assert pipeline.atom_budget() == 2
    pipeline.run()
    assert not (pipeline.directory / "subset.extxyz").exists()
    assert all(not c.refinement_targets for c in pipeline.build_targets()[0])


def test_no_training_and_missing_source(molecular_config: dict[str, Any]) -> None:
    path = Path(molecular_config["source"]["path"])
    frames = ase.io.read(path, index=":")
    for atoms in frames:
        atoms.info["split"] = "test"
    ase.io.write(path, frames, format="extxyz")
    with pytest.raises(ConfigError, match="no eligible training"):
        MolecularPipeline(molecular_config).featurize()
    path.unlink()
    with pytest.raises(ConfigError, match="does not exist"):
        MolecularPipeline(molecular_config)


def test_resume_after_interruption_before_feature_commit(
    molecular_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    pipeline = MolecularPipeline(molecular_config)
    writer = feature_cache.write_json

    def interrupt(path: Path, value: Any) -> None:
        if path.name == feature_cache.COMMIT:
            raise InterruptedError
        writer(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(feature_cache, "write_json", interrupt)
        with pytest.raises(InterruptedError):
            pipeline.featurize()
    manifest = pipeline.feature_directory / feature_cache.FEATURES
    first = manifest.read_bytes()
    assert MolecularPipeline(molecular_config).featurize()
    assert manifest.read_bytes() == first
    assert (pipeline.feature_directory / feature_cache.COMMIT).is_file()


@pytest.mark.parametrize("corruption", ["missing", "content"])
def test_cache_corruption_fails_closed(
    molecular_config: dict[str, Any], corruption: str
) -> None:
    pipeline = MolecularPipeline(molecular_config)
    pipeline.featurize()
    path = pipeline.feature_directory / "descriptors.npy"
    if corruption == "missing":
        path.unlink()
    else:
        path.write_bytes(b"broken")
    with pytest.raises(StaleCacheError, match="missing or corrupt"):
        MolecularPipeline(molecular_config).featurize()
    molecular_config["execution"]["resume"] = False
    assert MolecularPipeline(molecular_config).featurize()


def test_source_change_invalidates_cache(molecular_config: dict[str, Any]) -> None:
    before = MolecularPipeline(molecular_config)
    before.featurize()
    path = Path(molecular_config["source"]["path"])
    frames = ase.io.read(path, index=":")
    assert isinstance(frames, list)
    frames[0].positions[1, 0] += 0.1
    ase.io.write(path, frames, format="extxyz")
    after = MolecularPipeline(molecular_config)
    assert after.fingerprint != before.fingerprint
    assert after.feature_directory != before.feature_directory


def test_audit_rejects_manipulated_selection(molecular_config: dict[str, Any]) -> None:
    pipeline = MolecularPipeline(molecular_config)
    with pytest.raises(ConfigError, match="no molecular selection"):
        pipeline.audit()
    candidate = Candidate("a", 2, frozenset({"t"}))
    for ids in (("a", "a"), ("unknown",)):
        with pytest.raises(CoverageError, match="duplicate or non-candidate"):
            audit_selection([candidate], {"t": Target()}, ids)
    with pytest.raises(CoverageError, match="exceeds hard budget"):
        audit_selection([candidate], {"t": Target()}, ["a"], budget=1)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("source", "adapter", "guess"),
        ("source", "path", ""),
        ("source", "label_namespace", None),
        ("source", "label_units", []),
        ("source", "label_sources", {"energy": 1}),
        ("source", "label_units", {1: "eV"}),
        ("state", "charge_key", ""),
        ("state", "multiplicity_key", 3),
        ("state", "strict_training_export", "yes"),
        ("split", "preserve_source_splits", False),
        ("split", "group_related_states", 1),
        ("hierarchy", "min_source_support", True),
        ("hierarchy", "min_source_support", 1.5),
        ("hierarchy", "min_source_support", "2"),
        ("hierarchy", "min_selected_representatives", 0),
        ("geometry", "r_cut_angstrom", float("nan")),
        ("geometry", "sigma_angstrom", float("inf")),
        ("geometry", "species", "H"),
        ("geometry", "species", [True]),
        ("geometry", "species", [119]),
        ("selection", "background_reserve_fraction", 1),
        ("split", "validation_fraction", -0.1),
        ("selection", "atom_fraction", 2),
        ("selection", "atom_fraction", 0),
        ("selection", "atom_budget", 1.5),
        ("execution", "workdir", None),
        ("execution", "workdir", ""),
        ("execution", "workers", 0),
        ("execution", "workers", True),
        ("execution", "threads_per_worker", 1.5),
        ("execution", "memory_budget", True),
        ("execution", "memory_budget", -1),
        ("execution", "memory_budget", 1.5),
        ("execution", "memory_budget", "lots"),
    ],
)
def test_config_rejects_invalid_settings(
    molecular_config: dict[str, Any], section: str, key: str, value: Any
) -> None:
    molecular_config.setdefault(section, {})[key] = value
    with pytest.raises(ConfigError):
        load_molecular_config(molecular_config)


def test_config_paths_defaults_and_shape_errors(
    molecular_config: dict[str, Any], tmp_path: Path
) -> None:
    path = tmp_path / "config.yaml"
    molecular_config["source"]["path"] = "molecules.extxyz"
    molecular_config["source"]["label_units"] = {"energy": "eV"}
    molecular_config["geometry"]["species"] = [1]
    path.write_text(yaml.safe_dump(molecular_config))
    assert load_molecular_config(str(path)).resolved()["source"]["path"] == str(
        tmp_path / "molecules.extxyz"
    )
    assert is_molecular(path)
    for bad in (
        {},
        {"profile": "molecular_v1", "typo": {}},
        {"profile": "molecular_v1", "source": []},
        {"profile": "molecular_v1", "geometry": {"typo": 1}},
    ):
        with pytest.raises(ConfigError):
            load_molecular_config(bad)
    for selection in (
        {"mode": "budget"},
        {"atom_budget": 1, "atom_fraction": 0.5},
        {"atom_budget": 1},
        {"background_reserve_fraction": 0.1},
    ):
        config = copy.deepcopy(molecular_config)
        config["selection"] = selection
        with pytest.raises(ConfigError):
            load_molecular_config(config)
    for text in ("[", "[]"):
        path.write_text(text)
        with pytest.raises(ConfigError):
            load_molecular_config(path)
    path.write_text("[")
    with pytest.raises(ConfigError, match="cannot read"):
        is_molecular(path)
    path.unlink()
    with pytest.raises(ConfigError, match="cannot read"):
        load_molecular_config(path)


def test_molecular_cli_stages(
    molecular_config: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(molecular_config))
    for command in (
        "ingest",
        "split",
        "inspect",
        "featurize",
        "build-targets",
        "index",
        "select",
        "run",
        "audit",
    ):
        assert main([command, "--config", str(path)]) == 0
        assert json.loads(capsys.readouterr().out)
    directory = MolecularPipeline(path).directory
    for command in ("report", "export"):
        assert main([command, "--run", str(directory)]) == 0
        assert json.loads(capsys.readouterr().out)
    assert main(["select", "--config", str(path), "--min-support", "3"]) == 0
    capsys.readouterr()
    assert main(["tokenize", "--config", str(path)]) == 0
    capsys.readouterr()
    assert (
        main(
            ["export", "--config", str(path), "--output", str(tmp_path / "out.extxyz")]
        )
        == 0
    )
    capsys.readouterr()
    for args in (
        ["export", "--config", str(path), "--format", "asedb"],
        ["tokenize", "--config", str(path), "--num-shards", "2"],
        ["tokenize", "--config", str(path), "--shard-index", "1"],
        ["sweep", "--config", str(path), "--min-support", "1", "2"],
    ):
        assert main(args) == 2
        assert capsys.readouterr().err
    path.write_text("profile: other")
    assert main(["export", "--config", str(path)]) == 2
    assert "requires profile" in capsys.readouterr().err


def test_benchmark_cli_and_unsupported_stage(
    molecular_config: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(molecular_config))
    pipeline = MolecularPipeline(path)
    pipeline.select()
    for args in (
        ["benchmark", "--config", str(path)],
        ["baseline", "--run", str(pipeline.directory)],
    ):
        assert main(args) == 0
        summary = Path(json.loads(capsys.readouterr().out)["summary"])
        report = json.loads(summary.read_text())
        assert report["status"] == "model_accuracy_pending"
        assert len(report["runs"]) == 3
        assert {row["fraction"] for row in report["runs"]} == {0.25}
        assert {row["selection_seed"] for row in report["runs"]} == {0}
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(path),
                "--fractions",
                "0.5",
                "1",
                "--seeds",
                "0",
                "1",
            ]
        )
        == 0
    )
    summary = Path(json.loads(capsys.readouterr().out)["summary"])
    report = json.loads(summary.read_text())
    assert len(report["runs"]) == 12
    assert {row["fraction"] for row in report["runs"]} == {0.5, 1.0}
    with pytest.raises(ConfigError, match="not supported"):
        dispatch(argparse.Namespace(command="future", config=path))
