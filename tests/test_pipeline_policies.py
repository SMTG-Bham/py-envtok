"""Pipeline policies: per-stratum required records and the resume switch."""

from pathlib import Path
from typing import Any

import ase.io
import pytest
from ase import Atoms
from helpers import CORPUS

from envtok.errors import ConfigError
from envtok.pipeline import Pipeline


def small_corpus(tmp_path: Path) -> Path:
    frames = ase.io.read(CORPUS, index=":6")
    assert isinstance(frames, list)
    charged: list[Atoms] = []
    for atoms in frames:
        for charge in (0, -1):
            copy = atoms.copy()
            copy.calc = atoms.calc
            copy.info["charge"] = charge
            charged.append(copy)
    path = tmp_path / "charged.extxyz"
    ase.io.write(path, charged, format="extxyz")
    return path


def config(tmp_path: Path, path: Path, **selection: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": {
            "adapter": "extxyz",
            "path": str(path),
            "revision": "1",
            "split": "train",
        },
        "coverage": {"min_structure_support": 2},
        "selection": selection,
        "execution": {"workdir": str(tmp_path / "work"), "batch_size": 6},
        "output": {"format": "none"},
    }


def test_required_records_go_to_their_own_stratum(tmp_path: Path) -> None:
    path = small_corpus(tmp_path)
    required = "charged@1:charged.extxyz:3"  # a charge -1 frame
    pipeline = Pipeline(
        config(tmp_path, path, stratify_by=["charge"], required_records=[required])
    )
    result, directory = pipeline.select()
    reasons = {s.record_id: s.reason for s in result.selected}
    assert reasons[required] == "user@charge=-1"
    assert {stratum["charge"] for stratum, _ in result.strata} == {-1, 0}
    assert (directory / "coverage.json").exists()
    bad = Pipeline(
        config(
            tmp_path,
            path,
            stratify_by=["charge"],
            required_records=["charged@1:nope:0"],
        )
    )
    with pytest.raises(ConfigError, match="not eligible"):
        bad.fit_and_select(pipeline.cache())


def test_resume_disabled_refuses_partial_caches_but_reuses_finished_ones(
    tmp_path: Path,
) -> None:
    path = small_corpus(tmp_path)
    settings = config(tmp_path, path)
    settings["execution"]["resume"] = False
    pipeline = Pipeline(settings)
    first = pipeline.cache()
    assert pipeline.cache().token_dir == first.token_dir  # finished: reused
    (pipeline.cache_dir() / "plan.json").unlink()  # pretend the run was interrupted
    with pytest.raises(ConfigError, match="resume is disabled"):
        pipeline.tokenize()
    with pytest.raises(ConfigError, match="resume is disabled"):
        pipeline.cache()
