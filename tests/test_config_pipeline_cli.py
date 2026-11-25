"""Configuration loading, stage orchestration and the command-line interface."""

import json
import logging
import runpy
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import ase.db
import ase.io
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from helpers import CORPUS

import envtok
from envtok.cli import build_parser, main
from envtok.config import EnvTokConfig, load_config
from envtok.errors import ConfigError, CoverageError
from envtok.execution import ExecutionSettings
from envtok.pipeline import Pipeline, load_selection, random_baselines
from envtok.profiles import REFERENCE_723696D, MagneticPolicy
from envtok.selection import SelectionResult

SOURCE = {
    "adapter": "extxyz",
    "path": str(CORPUS),
    "revision": "v1",
    "name": "fx",
    "split": "train",
}


def config_for(workdir: Path, **sections: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": 1,
        "source": dict(SOURCE),
        "coverage": {"min_structure_support": 2},
        "execution": {"workdir": str(workdir), "batch_size": 5},
    }
    for name, values in sections.items():
        config.setdefault(name, {}).update(values)
    return config


@pytest.fixture(scope="module")
def shared_workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Work directory whose token cache is shared with the other test modules."""
    return tmp_path_factory.getbasetemp() / "envtok-shared-full"


@pytest.fixture(scope="module")
def selection(shared_workdir: Path) -> tuple[Pipeline, SelectionResult, Path]:
    pipeline = Pipeline(config_for(shared_workdir))
    result, directory = pipeline.select()
    return pipeline, result, directory


def labelled(atoms: Atoms, sid: str, **info: Any) -> Atoms:
    frame: Atoms = atoms.copy()
    frame.calc = SinglePointCalculator(
        frame, energy=-1.0, forces=np.zeros((len(frame), 3)), stress=np.zeros(6)
    )
    frame.info.update(sid=sid, **info)
    return frame


class TestConfig:
    def test_yaml_paths_resolve_against_the_file(self, tmp_path: Path) -> None:
        (tmp_path / "data").mkdir()
        shutil.copy(CORPUS, tmp_path / "data" / "corpus.extxyz")
        document = {
            "schema_version": 1,
            "source": {
                "adapter": "extxyz",
                "path": "data/corpus.extxyz",
                "revision": "r",
            },
            "execution": {"workdir": "runs", "memory_budget": "1GB"},
        }
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(document))
        config = load_config(path)
        assert config.base_dir == tmp_path
        assert config.workdir == tmp_path / "runs"
        corpus = config.corpus_object()
        assert corpus.name == "corpus"
        assert corpus.adapter.options.split.value == "unsplit"
        assert config.execution_settings().memory_budget == 10**9
        assert config.resolve("/abs/x") == Path("/abs/x")

    def test_mapping_and_round_trip(self) -> None:
        config = load_config(config_for(Path("w")))
        assert config.base_dir == Path.cwd()
        resolved = config.resolved()
        assert resolved["schema_version"] == 1
        assert resolved["selection"]["tie_break"] == "natoms_then_record_id"
        assert load_config(resolved).resolved() == resolved
        assert config.profile() is REFERENCE_723696D

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            ({"schema_version": 2}, "schema_version must be 1"),
            ({"schema_version": None}, "schema_version must be 1"),
            ({"extra": {}}, "unknown sections"),
            ({"source": None}, "needs a 'source' section"),
            ({"corpus": [1]}, "must be a mapping"),
            ({"corpus": {"bogus": 1}}, "unknown keys in 'corpus'"),
            ({"source": {"adapter": "extxyz", "path": "x"}}, "section 'source'"),
            ({"corpus": {"deduplicate": True}}, "deduplicate"),
            ({"coverage": {"support_unit": "group"}}, "support_unit"),
            ({"output": {"preserve_labels": False}}, "preserve_labels"),
            ({"output": {"format": "cif"}}, "output.format"),
        ],
    )
    def test_invalid_configurations(self, change: dict[str, Any], match: str) -> None:
        document: dict[str, Any] = {"schema_version": 1, "source": dict(SOURCE)}
        document.update(change)
        if document.get("source") is None:
            del document["source"]
        if document.get("schema_version") is None:
            del document["schema_version"]
        with pytest.raises(ConfigError, match=match):
            load_config(document)

    def test_non_mapping_document_and_unknown_adapter(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config(path)
        config = load_config(
            {"schema_version": 1, "source": {**SOURCE, "adapter": "parquet"}}
        )
        with pytest.raises(ConfigError, match=r"source\.adapter"):
            config.corpus_object()

    def test_profile_variants(self) -> None:
        def profile(**tokenizer: Any) -> Any:
            return load_config(
                {"schema_version": 1, "source": SOURCE, "tokenizer": tokenizer}
            ).profile()

        geometry_only = profile(magnetic_policy="disabled")
        assert geometry_only.name == "reference-723696d+geometry-only"
        assert geometry_only.magnetic_policy is MagneticPolicy.DISABLED
        all_sites = profile(use_symmetry=False)
        assert all_sites.name == "reference-723696d+all-sites"
        assert not all_sites.geometry.use_symmetry
        both = profile(magnetic_policy="disabled", use_symmetry=False)
        assert both.name == "reference-723696d+geometry-only+all-sites"

    def test_every_adapter_and_option(self, tmp_path: Path) -> None:
        database = tmp_path / "data.db"
        handle = ase.db.connect(str(database))
        handle.write(bulk("NaCl", "rocksalt", a=5.64), data={"sid": "x"})
        (tmp_path / "omat").mkdir()
        (tmp_path / "omat" / "data.aselmdb").touch()
        base = {"schema_version": 1}
        extxyz = load_config(
            {
                **base,
                "source": {
                    **SOURCE,
                    "path": str(tmp_path),
                    "compressed": "stage",
                    "staging_dir": "stage",
                    "patterns": ["*.nothing", "*.extxyz"],
                    "label_sources": {"energy": "info:REF_energy"},
                    "label_units": {"energy": "kcal/mol"},
                },
            }
        )
        shutil.copy(CORPUS, tmp_path / "c.extxyz")
        corpus = extxyz.corpus_object()
        assert corpus.adapter.spec()["staging_dir"] == str(Path.cwd() / "stage")
        assert corpus.adapter.options.label_units["energy"] == "kcal/mol"
        assert corpus.adapter.spec()["patterns"] == ["*.nothing", "*.extxyz"]
        asedb = load_config(
            {
                **base,
                "source": {"adapter": "asedb", "path": str(database), "revision": "r"},
            }
        ).corpus_object()
        assert asedb.adapter.kind == "asedb"
        assert asedb.name == "data"
        omat = load_config(
            {
                **base,
                "source": {
                    "adapter": "omat24",
                    "path": str(tmp_path / "omat"),
                    "revision": "r",
                },
                "corpus": {
                    "preset": "omat24-study",
                    "invalid_record_policy": "reference",
                },
            }
        ).corpus_object()
        assert (omat.adapter.kind, omat.name) == ("omat24", "omat24")
        assert omat.policy.invalid_record_policy.value == "reference"

    def test_execution_sizes(self) -> None:
        config = load_config(
            config_for(
                Path("w"),
                execution={"memory_budget": "2GiB", "temporary_disk_budget": 1024},
            )
        )
        settings = config.execution_settings()
        assert isinstance(settings, ExecutionSettings)
        assert (settings.memory_budget, settings.temporary_disk_budget) == (2**31, 1024)
        assert settings.batch_size == 5


class TestPipeline:
    def test_inspect_reports_the_sample(self, tmp_path: Path) -> None:
        nacl = labelled(bulk("NaCl", "rocksalt", a=5.64), "a")
        lan = labelled(bulk("LaN", "rocksalt", a=5.3), "b")
        lan.set_initial_magnetic_moments([1.0, 0.0])
        lan.calc = None
        slab = labelled(bulk("NaCl", "rocksalt", a=5.64), "a")
        slab.pbc = (True, True, False)
        path = tmp_path / "mixed.extxyz"
        ase.io.write(path, [nacl, lan, slab], format="extxyz")
        config = config_for(
            tmp_path / "work",
            source={"path": str(path), "id_policy": "namespaced_native_id"},
            corpus={"preset": "omat24-study"},
        )
        report = Pipeline(config).inspect()
        assert report["sampled"] == 3
        assert report["duplicate_ids"] == 1
        assert report["excluded_elements_present"] == ["La"]
        assert report["issues"] == {"exclude:unsupported_periodicity": 1}
        assert report["labels"] == {"energy": 2, "forces": 2, "stress": 2}
        assert report["records_with_initial_moments"] == 1
        assert report["splits"] == {"train": 3}
        assert report["supports_retrieval"]
        assert Pipeline(config).inspect(sample=1)["sampled"] == 1

    def test_shards_then_index(self, tmp_path: Path) -> None:
        subset = ase.io.read(CORPUS, index=":4")
        assert isinstance(subset, list)
        path = tmp_path / "subset.extxyz"
        ase.io.write(path, subset, format="extxyz")
        pipeline = Pipeline(
            config_for(
                tmp_path / "w", source={"path": str(path)}, execution={"batch_size": 2}
            )
        )
        with pytest.raises(ConfigError, match="nothing tokenised"):
            pipeline.index()
        assert pipeline.tokenize(shard_index=0, num_shards=2) == pipeline.cache_dir()
        pipeline.tokenize(shard_index=1, num_shards=2)
        assert pipeline.index().summary["discovered"] == 4

    def test_select_writes_a_configuration_addressed_run(
        self, selection: tuple[Pipeline, SelectionResult, Path]
    ) -> None:
        pipeline, result, directory = selection
        assert directory.name.startswith("selection-T2-")
        assert directory.parent == pipeline.workdir
        for name in (
            "selected_ids.txt",
            "selected_manifest.parquet",
            "config.resolved.yaml",
            "provenance.json",
            "coverage.json",
            "report.html",
            "run.json",
        ):
            assert (directory / name).exists(), name
        run = json.loads((directory / "run.json").read_text())
        assert run["min_structure_support"] == 2
        assert result.validate()["classes"]["fraction"] == 1.0

    def test_exports_in_every_format(
        self, selection: tuple[Pipeline, SelectionResult, Path], tmp_path: Path
    ) -> None:
        pipeline, result, directory = selection
        assert pipeline.export(result, directory, tmp_path / "a.extxyz") == (
            tmp_path / "a.extxyz"
        )
        frames = ase.io.read(tmp_path / "a.extxyz", index=":")
        assert isinstance(frames, list)
        assert len(frames) == len(result.selected)
        config = pipeline.config.resolved()
        config["output"]["format"] = "asedb"
        database = Pipeline(config).export(result, tmp_path)
        assert database == tmp_path / "subset.db"
        assert ase.db.connect(str(database)).count() == len(result.selected)
        config["output"]["format"] = "none"
        assert Pipeline(config).export(result, tmp_path) is None

    def test_resume_disabled_reuses_an_indexed_cache(
        self, shared_workdir: Path, tmp_path: Path
    ) -> None:
        reuse = Pipeline(config_for(shared_workdir, execution={"resume": False}))
        assert reuse.cache().summary["discovered"] == 20
        subset = ase.io.read(CORPUS, index=":2")
        assert isinstance(subset, list)
        path = tmp_path / "two.extxyz"
        ase.io.write(path, subset, format="extxyz")
        fresh = Pipeline(
            config_for(
                tmp_path / "w", source={"path": str(path)}, execution={"resume": False}
            )
        )
        assert fresh.cache().summary["discovered"] == 2

    def test_run_and_sweep(self, shared_workdir: Path) -> None:
        pipeline = Pipeline(config_for(shared_workdir))
        directory = pipeline.run(3)
        assert (directory / "subset.extxyz").exists()
        summary = json.loads(pipeline.sweep([2, 3]).read_text())
        assert [row["min_structure_support"] for row in summary] == [2, 3]
        assert all(0 < row["atom_retention"] <= 1 for row in summary)
        assert summary[1]["directory"] == str(directory)

    def test_reload_and_random_baselines(
        self,
        selection: tuple[Pipeline, SelectionResult, Path],
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _, result, directory = selection
        _, reloaded = load_selection(directory)
        assert reloaded.record_ids == result.record_ids
        assert [s.reason for s in reloaded.selected] == [
            s.reason for s in result.selected
        ]
        copy = tmp_path / directory.name
        shutil.copytree(directory, copy)
        manifest = pq.read_table(copy / "selected_manifest.parquet")
        pq.write_table(
            manifest.slice(0, manifest.num_rows - 1), copy / "selected_manifest.parquet"
        )
        with caplog.at_level(logging.WARNING, logger="envtok"):
            _, shortened = load_selection(copy)
        assert "different subset" in caplog.text
        assert shortened.record_ids == result.record_ids[:-1]
        summary = json.loads(random_baselines(copy, [0, 1]).read_text())
        budget = summary["atom_budget"]
        assert budget == shortened.natoms
        assert [row["seed"] for row in summary["baselines"]] == [0, 1]
        assert all(row["atoms"] <= budget for row in summary["baselines"])
        assert (copy / "baselines" / "random-seed0" / "coverage.json").exists()

    def test_protected_strata_keep_every_charge_state(self, tmp_path: Path) -> None:
        base = ase.io.read(CORPUS, index=":")
        assert isinstance(base, list)
        chosen = [base[i] for i in (0, 7, 11, 12)]
        frames = []
        for charge in (0, -1):
            for atoms in chosen:
                frame = atoms.copy()
                frame.calc = atoms.calc
                frame.info["charge"] = charge
                frame.info["sid"] = f"{atoms.info['sid']}-q{charge}"
                frames.append(frame)
        path = tmp_path / "charges.extxyz"
        ase.io.write(path, frames, format="extxyz")
        plain = Pipeline(config_for(tmp_path / "plain", source={"path": str(path)}))
        plain_result, _ = plain.select()
        stratified = Pipeline(
            config_for(
                tmp_path / "strata",
                source={"path": str(path)},
                selection={"stratify_by": ["charge"]},
            )
        )
        result, directory = stratified.select()
        assert [stratum for stratum, _ in result.strata] == [
            {"charge": -1},
            {"charge": 0},
        ]
        charges = {
            json.loads(m)["charge"]
            for m in pq.read_table(directory / "selected_manifest.parquet")[
                "metadata_json"
            ].to_pylist()
        }
        assert charges == {0, -1}
        assert len(result.selected) > len(plain_result.selected)
        assert all("@charge=" in s.reason for s in result.selected)
        report = result.validate()
        assert len(report["strata"]) == 2
        _, reloaded = load_selection(directory)
        assert reloaded.record_ids == result.record_ids
        assert reloaded.validate()["strata"] == report["strata"]
        manifest = directory / "selected_manifest.parquet"
        table = pq.read_table(manifest)
        keep = [
            json.loads(m)["charge"] == 0 for m in table["metadata_json"].to_pylist()
        ]
        pq.write_table(table.filter(keep), manifest)
        _, incomplete = load_selection(directory)
        with pytest.raises(CoverageError, match="uncovered"):
            incomplete.validate()
        missing = Pipeline(
            config_for(
                tmp_path / "strata",
                source={"path": str(path)},
                selection={"stratify_by": ["charge"], "required_records": ["fx@v1:no"]},
            )
        )
        with pytest.raises(ConfigError, match="required records are not eligible"):
            missing.fit_and_select(missing.cache())


class TestCli:
    @pytest.fixture
    def config_path(self, shared_workdir: Path, tmp_path: Path) -> Path:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config_for(shared_workdir)))
        return path

    def run(self, argv: list[str], capsys: pytest.CaptureFixture[str]) -> Any:
        assert main(argv) == 0
        return json.loads(capsys.readouterr().out)

    def test_version_and_parser(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert (
            capsys.readouterr().out.strip() == version("envtok") == envtok.__version__
        )
        assert build_parser().prog == "envtok"

    def test_every_command(
        self,
        config_path: Path,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = str(config_path)
        assert (
            self.run(["inspect", "--config", config, "--sample", "3"], capsys)[
                "sampled"
            ]
            == 3
        )
        assert "cache" in self.run(
            ["tokenize", "--config", config, "--shard-index", "0", "--num-shards", "1"],
            capsys,
        )
        assert self.run(["-v", "index", "--config", config], capsys)["discovered"] == 20
        selected = Path(
            self.run(["select", "--config", config, "--min-support", "3"], capsys)[
                "selection"
            ]
        )
        assert selected.name.startswith("selection-T3-")
        run_dir = Path(self.run(["run", "--config", config], capsys)["selection"])
        assert (run_dir / "subset.extxyz").exists()
        sweep = self.run(
            ["sweep", "--config", config, "--min-support", "2", "3"], capsys
        )
        assert Path(sweep["summary"]).exists()
        classes = self.run(["report", "--run", str(run_dir)], capsys)
        assert classes["fraction"] == 1.0
        target = tmp_path / "out.db"
        exported = self.run(
            [
                "export",
                "--run",
                str(run_dir),
                "--format",
                "asedb",
                "--output",
                str(target),
            ],
            capsys,
        )
        assert Path(exported["exported"]) == target
        assert ase.db.connect(str(target)).count() > 0
        default = self.run(["export", "--run", str(run_dir)], capsys)
        assert Path(default["exported"]) == run_dir / "subset.extxyz"
        baseline = self.run(["baseline", "--run", str(run_dir), "--seeds", "4"], capsys)
        assert Path(baseline["summary"]).exists()

    def test_errors_exit_with_status_two(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config_for(tmp_path / "empty")))
        assert main(["index", "--config", str(path)]) == 2
        assert "envtok: error: nothing tokenised" in capsys.readouterr().err

    def test_module_entry_point(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["envtok", "--version"])
        with pytest.raises(SystemExit) as exit_info:
            runpy.run_module("envtok", run_name="__main__")
        assert exit_info.value.code == 0
        assert capsys.readouterr().out.strip() == envtok.__version__


def test_config_type_is_exported() -> None:
    assert isinstance(load_config(config_for(Path("w"))), EnvTokConfig)
