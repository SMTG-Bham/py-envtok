"""Verify parent holdouts cannot affect periodic compression or source exports."""

import os
import runpy
from pathlib import Path
from typing import Any, cast

import ase.io
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from envtok import TokenCache
from envtok.adapters import spool_atoms

ROOT = Path(__file__).resolve().parents[1]
RUNNER = runpy.run_path(
    str(ROOT / "build_tools/benchmark_omat_compression_training.py")
)


def _source(path: Path) -> list[Atoms]:
    frames = []
    for index in range(12):
        atoms = bulk("Cu", "fcc", a=3.6 + 0.01 * index, cubic=True)
        atoms.positions[0] += [0.02, 0.01, -0.01]
        atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
        atoms.new_array("source_marker", np.arange(len(atoms)))
        atoms.info.update(
            sid=f"periodic-{index}",
            parent_id=f"parent-{index}",
            source_record_id=f"native-fixture:{index}",
            split="train",
            native_metadata={"index": index, "optional": None},
            empty_metadata=[],
            adjacent_metadata=["must survive empty array metadata"],
        )
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-4.0 * len(atoms) + 0.01 * index,
            forces=np.arange(len(atoms) * 3).reshape(-1, 3) * 0.001,
            stress=np.arange(6) * 0.0001,
        )
        frames.append(atoms)
    spool_atoms(frames, path)
    return cast("list[Atoms]", ase.io.read(path, index=":"))


def _check_exports(experiment: dict[str, Any], originals: list[Atoms]) -> None:
    by_sid = {atoms.info["sid"]: atoms for atoms in originals}
    for selection in experiment["selections"]:
        exported = ase.io.read(ROOT / selection["train_file"], index=":")
        assert [a.info["envtok_record_id"] for a in exported] == list(
            selection["selected_ids"]
        )
        for atoms in exported:
            original = by_sid[atoms.info["sid"]]
            np.testing.assert_array_equal(atoms.cell.array, original.cell.array)
            np.testing.assert_array_equal(atoms.pbc, original.pbc)
            for key, values in original.arrays.items():
                np.testing.assert_array_equal(atoms.arrays[key], values)
            assert atoms.get_potential_energy() == original.get_potential_energy()
            np.testing.assert_array_equal(atoms.get_forces(), original.get_forces())
            np.testing.assert_array_equal(atoms.get_stress(), original.get_stress())
            assert atoms.info["native_metadata"] == original.info["native_metadata"]
            assert atoms.info["source_record_id"] == original.info["source_record_id"]
            np.testing.assert_array_equal(
                atoms.info["empty_metadata"], original.info["empty_metadata"]
            )
            assert atoms.info["adjacent_metadata"] == original.info["adjacent_metadata"]


def test_heldout_mutation_does_not_change_periodic_selection_or_exports(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.extxyz"
    original = _source(source)
    before, heldout_before = RUNNER["prepare"](source, tmp_path / "before", workers=1)
    _check_exports(before, original)
    parents = {record.group_id for record in heldout_before}
    assert parents == {"parent-7", "parent-10"}
    for atoms in original:
        if atoms.info["parent_id"] in parents:
            atoms.set_cell(atoms.cell.array * 1.7, scale_atoms=True)
            atoms.positions[0] += [0.2, -0.1, 0.3]
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=1e8,
                forces=np.full((len(atoms), 3), 1e7),
                stress=np.full(6, 1e6),
            )
    spool_atoms(original, source)
    after, heldout_after = RUNNER["prepare"](source, tmp_path / "after", workers=1)

    assert before["source_sha256"] != after["source_sha256"]
    assert before["holdout_sha256"] != after["holdout_sha256"]
    assert after["holdout_zero_force_rmse_eV_per_Ang"] == 1e7
    assert before["split_manifest"] == after["split_manifest"]
    assert before["holdout_ids"] == after["holdout_ids"]
    assert len(heldout_before) == len(heldout_after) == 2
    holdout_ids = set(before["holdout_ids"])
    training_ids = {
        record_id
        for record_id, row in before["split_manifest"].items()
        if row["split"] == "train"
    }
    for old, new in zip(before["selections"], after["selections"], strict=True):
        assert old["selected_ids"] == new["selected_ids"]
        assert old["train_sha256"] == new["train_sha256"]
        assert old["coverage"] == new["coverage"]
        assert set(new["selected_ids"]) <= training_ids
        assert holdout_ids.isdisjoint(new["selected_ids"])
    old_cache = TokenCache.open(ROOT / before["token_cache"])
    new_cache = TokenCache.open(ROOT / after["token_cache"])
    for cache in (old_cache, new_cache):
        rows = cache.read("structures").to_pylist()
        assert {row["record_id"] for row in rows} == training_ids
        assert {row["split"] for row in rows} == {"train"}
        assert parents.isdisjoint(row["group_id"] for row in rows)
    for table in ("tokens", "structure_tokens", "site_tokens"):
        assert old_cache.read(table).to_pylist() == new_cache.read(table).to_pylist()


def test_identical_export_preserves_cache_input_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "source.extxyz"
    _source(source)
    records = RUNNER["read_source"](source)
    output = tmp_path / "training-pool.extxyz"
    original_hash = RUNNER["export_records"](records, output)
    old_timestamp = 1_600_000_000_000_000_000
    os.utime(output, ns=(old_timestamp, old_timestamp))

    assert RUNNER["export_records"](records, output) == original_hash
    assert output.stat().st_mtime_ns == old_timestamp

    records[0].atoms.info["changed_metadata"] = "requires fresh cache"
    assert RUNNER["export_records"](records, output) != original_hash
    assert output.stat().st_mtime_ns != old_timestamp
