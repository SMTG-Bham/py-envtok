"""Molecular memory estimates and the execution.memory_budget policy."""

import os
from pathlib import Path
from typing import Any

import ase.io
import pytest
from ase import Atoms

from envtok import execution
from envtok.errors import ResourcePolicyError
from envtok.molecular import pipeline as pipeline_module
from envtok.molecular.config import load_molecular_config
from envtok.molecular.pipeline import MolecularPipeline, estimate_molecular_bytes


@pytest.fixture
def small_config(tmp_path: Path) -> dict[str, Any]:
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


def test_estimate_grows_with_atoms_and_queued_worker_batches() -> None:
    base = estimate_molecular_bytes(0, 0)
    assert base == pipeline_module.MEMORY_BASE_BYTES
    serial = estimate_molecular_bytes(1000, 30_000)
    assert serial == base + 30_000 * pipeline_module.MEMORY_ATOM_BYTES
    assert estimate_molecular_bytes(1000, 30_000, workers=1) == serial
    parallel = estimate_molecular_bytes(1000, 30_000, workers=4)
    # 9 batches of 63 records with 30 atoms each.
    queued = 9 * 63 * 30 * pipeline_module.MEMORY_QUEUED_ATOM_BYTES
    assert parallel == serial + queued
    assert estimate_molecular_bytes(10**6, 3 * 10**7, workers=128) > 40 * 2**30


def test_memory_budget_refuses_before_featurising(
    small_config: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "resident_bytes", lambda: 2**30)
    needed = 2**30 + estimate_molecular_bytes(2, 4)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("featurisation must not start above the memory budget")

    small_config["execution"]["memory_budget"] = needed - 1
    with monkeypatch.context() as patched:
        patched.setattr(pipeline_module, "iter_features", forbidden)
        with pytest.raises(ResourcePolicyError, match="memory_budget"):
            MolecularPipeline(small_config).featurize()
    small_config["execution"]["memory_budget"] = f"{needed}B"
    assert MolecularPipeline(small_config).select().selected_ids


def test_memory_budget_does_not_change_the_run(small_config: dict[str, Any]) -> None:
    unbounded = MolecularPipeline(small_config)
    small_config["execution"]["memory_budget"] = "1TB"
    bounded = MolecularPipeline(small_config)
    assert bounded.fingerprint == unbounded.fingerprint
    assert bounded.directory == unbounded.directory
    assert load_molecular_config(small_config).data["execution"]["memory_budget"]


def test_resident_bytes_reads_proc_or_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statm = tmp_path / "statm"
    statm.write_text("100 25 0 0 0 0 0\n")
    monkeypatch.setattr(execution, "_STATM", statm)
    assert execution.resident_bytes() == 25 * os.sysconf("SC_PAGE_SIZE")
    statm.write_text("broken")
    monkeypatch.setattr(execution, "peak_rss_bytes", lambda: 7)
    assert execution.resident_bytes() == 7
    monkeypatch.setattr(execution, "_STATM", tmp_path / "missing")
    assert execution.resident_bytes() == 7
