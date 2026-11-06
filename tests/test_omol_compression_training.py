"""Check that held-out geometry and labels cannot influence compression."""

import runpy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.molecular.data import export_records, iter_extxyz
from envtok.molecular.pipeline import MolecularPipeline

ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(
    str(ROOT / "build_tools/benchmark_omol_compression_training.py")
)


def test_heldout_mutation_cannot_change_selected_training_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pipelines: list[MolecularPipeline] = []

    def radial_pipeline(settings: dict[str, Any]) -> MolecularPipeline:
        settings["geometry"]["backend"] = "radial"
        pipeline = MolecularPipeline(settings)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setitem(
        RUNNER["prepare"].__globals__, "MolecularPipeline", radial_pipeline
    )
    source = tmp_path / "sample.extxyz"
    original = list(
        iter_extxyz(
            ROOT / "examples/molecular/omol/sample.extxyz",
            label_namespace=RUNNER["NAMESPACE"],
        )
    )
    export_records(original, source)
    before, heldout_before = RUNNER["prepare"](source, tmp_path / "before")
    heldout_source_ids = {
        before["source_id_mapping"][record_id] for record_id in before["holdout_ids"]
    }
    records = list(iter_extxyz(source, label_namespace=RUNNER["NAMESPACE"]))
    changed = []
    for record in records:
        updated = record
        if record.record_id in heldout_source_ids:
            atoms = record.atoms.copy()
            atoms.positions *= 1.9
            energy = np.asarray(1e8)
            forces = np.full((record.natoms, 3), 1e7)
            atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            labels = {
                **record.labels,
                "energy": replace(record.labels["energy"], value=energy),
                "forces": replace(record.labels["forces"], value=forces),
            }
            updated = replace(
                record, record=replace(record.record, atoms=atoms, labels=labels)
            )
        changed.append(updated)
    export_records(changed, source)
    after, heldout_after = RUNNER["prepare"](source, tmp_path / "after")

    assert len(heldout_source_ids) == len(heldout_before) > 0
    assert before["source_sha256"] != after["source_sha256"]
    assert before["holdout_sha256"] != after["holdout_sha256"]
    assert after["holdout_zero_force_rmse_eV_per_Ang"] == 1e7
    assert before["split_manifest"] == after["split_manifest"]
    assert before["source_id_mapping"] == after["source_id_mapping"]
    assert len(heldout_before) == len(heldout_after)
    holdout_ids = set(before["holdout_ids"])
    for old, new in zip(before["selections"], after["selections"], strict=True):
        assert old["selected_ids"] == new["selected_ids"]
        assert holdout_ids.isdisjoint(new["selected_ids"])
        assert old["training_atoms"] == new["training_atoms"]
        assert old["train_sha256"] == new["train_sha256"]

    for old, new in zip(pipelines[:2], pipelines[2:], strict=True):
        old_features, new_features = old.featurize(), new.featurize()
        assert {row.record_id for row in old_features} == {
            record.record_id for record in old.ingest().candidates
        }
        assert holdout_ids.isdisjoint(row.record_id for row in new_features)
        for old_row, new_row in zip(old_features, new_features, strict=True):
            assert old_row.record_id == new_row.record_id
            assert old_row.token_hierarchy == new_row.token_hierarchy
            np.testing.assert_array_equal(old_row.descriptors, new_row.descriptors)
