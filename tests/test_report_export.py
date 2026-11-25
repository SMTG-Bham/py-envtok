"""Independent audit, diagnostics, manifests, provenance and structure export."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import ase.db
import ase.io
import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from helpers import CORPUS

from envtok import export, report
from envtok.adapters.asedb import ARRAYS_KEY, row_atoms
from envtok.coarsen import CoverageBuilder, CoverageClasses
from envtok.corpus import Corpus
from envtok.errors import (
    ConfigError,
    CoverageError,
    RetrievalUnavailableError,
    StaleCacheError,
)
from envtok.profiles import REFERENCE_723696D
from envtok.provenance import collect
from envtok.selection import Selected, SelectionResult, Selector
from envtok.store.cache import TokenCache
from envtok.tokenization import Tokenizer


def load_frames() -> list[Atoms]:
    frames = ase.io.read(CORPUS, index=":", format="extxyz")
    assert isinstance(frames, list)
    return frames


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> TokenCache:
    """The whole fixture corpus, shared through the session base directory."""
    workdir = tmp_path_factory.getbasetemp() / "envtok-shared-full"
    corpus = Corpus.from_extxyz(CORPUS, dataset="fx", revision="v1", split="train")
    return Tokenizer().run(corpus, workdir, batch_size=5)


@pytest.fixture(scope="module")
def classes(cache: TokenCache) -> CoverageClasses:
    return CoverageBuilder(min_structure_support=2).fit(cache, "train")


@pytest.fixture(scope="module")
def result(classes: CoverageClasses) -> SelectionResult:
    return Selector().select(classes)


@pytest.fixture(scope="module")
def coverage(result: SelectionResult) -> dict[str, Any]:
    return report.audit(result)


def truncated(result: SelectionResult, keep: int) -> SelectionResult:
    return dataclasses.replace(result, selected=result.selected[:keep])


class TestAudit:
    def test_valid_cover_is_complete(
        self, result: SelectionResult, coverage: dict[str, Any]
    ) -> None:
        assert coverage["classes"]["fraction"] == 1.0
        assert coverage["classes"]["missing_examples"] == []
        assert coverage["structures"]["selected"] == len(result.selected)
        assert coverage["structures"]["eligible"] == 20
        assert coverage["atoms"]["selected"] == result.natoms
        assert coverage["atoms"]["eligible"] == 76
        tokens = coverage["full_tokens"]
        assert tokens["covered"] <= tokens["eligible"] == 33
        assert 0 < tokens["frequency_weighted"] <= 1
        assert tokens["fraction"] == pytest.approx(tokens["covered"] / 33)
        assert set(coverage["presence"]) == {
            "F_elem_poly",
            "F_elem_cn",
            "F_elem_cn_dmin",
            "F_elem_cn_abin",
        }
        assert all(f["fraction"] == 1.0 for f in coverage["presence"].values())
        assert sum(coverage["reasons"].values()) == len(result.selected)
        assert len(coverage["resolution"]) == len(REFERENCE_723696D.ladder) + 1
        assert coverage["held_out"] == {}
        assert coverage["processing"]["records_by_status"] == [
            {"status": "ok", "reason": None, "count": 20}
        ]
        report.enforce(coverage)
        assert result.validate()["classes"] == coverage["classes"]

    def test_missing_classes_and_presence_are_enforced(
        self, result: SelectionResult, classes: CoverageClasses
    ) -> None:
        partial = report.audit(truncated(result, 1))
        assert partial["classes"]["covered"] < partial["classes"]["required"]
        assert partial["classes"]["missing_examples"]
        with pytest.raises(CoverageError, match="classes"):
            report.enforce(partial)
        with pytest.raises(CoverageError, match="classes"):
            truncated(result, 1).validate()

        unrepaired = Selector(presence_constraints="none").select(classes)
        gap = report.audit(unrepaired)
        assert gap["classes"]["fraction"] == 1.0
        assert (
            gap["presence"]["F_elem_poly"]["covered"]
            < (gap["presence"]["F_elem_poly"]["required"])
        )
        assert gap["presence"]["F_elem_poly"]["missing_examples"]
        report.enforce(gap)
        strict = {
            **gap,
            "selector": {**gap["selector"], "presence_constraints": "reference"},
        }
        with pytest.raises(CoverageError, match="F_elem_poly signatures"):
            report.enforce(strict)

    def test_invalid_selections_are_rejected(
        self, result: SelectionResult, classes: CoverageClasses
    ) -> None:
        ids = result.record_ids
        with pytest.raises(CoverageError, match="1 duplicate"):
            report.audit_classes(classes, [*ids, ids[0]])
        with pytest.raises(CoverageError, match="outside"):
            report.audit_classes(classes, [*ids, "elsewhere:1"])

    def test_profiles_without_presence_families(self, cache: TokenCache) -> None:
        profile = dataclasses.replace(REFERENCE_723696D, presence_families=())
        bare = CoverageBuilder(profile, 2).fit(cache)
        selection = Selector().select(bare)
        audited = report.audit_classes(bare, selection.record_ids)
        assert audited["presence"] == {}
        assert audited["classes"]["fraction"] == 1.0

    def test_helpers(self) -> None:
        assert report._ratio(1, 0) is None
        assert report._ratio(1, 4) == 0.25
        assert report._percent(None) == "n/a"
        assert report._percent(0.5) == "50%"
        assert "&lt;b&gt;" in report._table([{"a": "<b>"}], ["a"])


class TestQualityAndHeldOut:
    def test_invalid_sites_change_the_eligible_denominator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pymatgen.analysis.local_env import CrystalNN  # noqa: PLC0415

        frames = load_frames()

        def fail(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("no rescue")

        monkeypatch.setattr(CrystalNN, "get_nn_info", fail)
        corpus = Corpus.from_atoms(
            [frames[0], frames[18]], name="q", revision="r", split="train"
        )
        invalid = Tokenizer().run(corpus, tmp_path)
        monkeypatch.undo()
        tokens = invalid.read("tokens").to_pylist()
        placeholder = next(t for t in tokens if t["geometry"] == "INVALID")
        assert placeholder["neighbours"] is None
        assert placeholder["coordination_number"] is None
        quarantine = CoverageBuilder(min_structure_support=2).fit(invalid)
        assert len(quarantine.pool.record_ids) == 1
        quality = report.processing_quality(quarantine)
        assert quality["eligible_by_policy"] == {
            "reference": 2,
            "quarantine": 1,
            "active": "quarantine",
        }
        assert quality["chemenv_failed_records"] == 1
        assert any(s["status"] == "invalid" for s in quality["sites"])
        assert any(f["stage"] == "crystalnn" for f in quality["failures"])
        reference = CoverageBuilder(
            min_structure_support=2, invalid_record_policy="reference"
        ).fit(invalid)
        assert len(reference.pool.record_ids) == 2

    def test_held_out_records_are_looked_up_not_fitted(self, tmp_path: Path) -> None:
        frames = load_frames()
        chosen = []
        for index, split in (
            (0, "train"),
            (1, "train"),
            (2, "validation"),
            (7, "test"),
        ):
            frame = frames[index].copy()
            frame.calc = frames[index].calc
            frame.info["split"] = split
            chosen.append(frame)
        twin = frames[0].copy()
        twin.info.update(split="test", sid="twin")
        chosen.append(twin)
        path = tmp_path / "splits.extxyz"
        ase.io.write(path, chosen, format="extxyz")
        corpus = Corpus.from_extxyz(path, dataset="s", revision="r", split_key="split")
        mixed = Tokenizer().run(corpus, tmp_path / "mixed")
        fitted = CoverageBuilder(min_structure_support=2).fit(mixed)
        assert len(fitted.pool.record_ids) == 2
        novelty = report.held_out_novelty(fitted)
        assert set(novelty) == {"validation", "test"}
        assert novelty["test"]["sites"] == 4
        assert novelty["test"]["oov_sites"] == 2
        assert novelty["test"]["oov_fraction"] == 0.5
        assert 0 < novelty["validation"]["oov_fraction"] < 1

        train_only = tmp_path / "train.extxyz"
        ase.io.write(train_only, chosen[:2], format="extxyz")
        alone = Tokenizer().run(
            Corpus.from_extxyz(
                train_only, dataset="s", revision="r", split_key="split"
            ),
            tmp_path / "alone",
        )
        reference = CoverageBuilder(min_structure_support=2).fit(alone)
        assert reference.class_signature == fitted.class_signature
        native = [
            [c.pool.native_ids[s.record_index] for s in Selector().select(c).selected]
            for c in (reference, fitted)
        ]
        assert native[0] == native[1]
        assert native[0]


class TestDiagnosticsAndHtml:
    def test_distribution_tables(self, result: SelectionResult) -> None:
        tables = report.diagnostics(result.classes, result.record_ids)
        assert {
            "element_atoms",
            "geometry_sites",
            "compression_bin_sites",
            "asymmetry_bin_sites",
            "csm_bin_sites",
            "elements_per_structure",
            "natoms_per_structure",
            "source_shards",
            "labels",
        } == set(tables)
        assert sum(row["eligible"] for row in tables["element_atoms"]) == 76
        assert sum(row["selected"] for row in tables["element_atoms"]) == result.natoms
        energy = tables["labels"]["energy_per_atom"]
        assert energy["eligible"]["count"] == 20
        assert energy["selected"]["count"] == len(result.selected)
        assert len(energy["eligible"]["q05_q50_q95"]) == 3

    def test_html_report(
        self, result: SelectionResult, coverage: dict[str, Any]
    ) -> None:
        tables = report.diagnostics(result.classes, result.record_ids)
        page = report.render_html(coverage, tables)
        assert page.startswith("<!doctype html>")
        for text in (
            "EnvTok selection report (T=2)",
            "Class coverage",
            "Presence F_elem_poly",
            "Resolution loss by rung",
            "R0_full",
            "element_atoms",
            "Label summaries",
        ):
            assert text in page


class TestManifestAndProvenance:
    def test_manifest_files(self, result: SelectionResult, tmp_path: Path) -> None:
        directory = result.write_manifest(tmp_path / "run", save_classes=False)
        for name in export.MANIFEST_FILES:
            assert (directory / name).exists(), name
        assert not (directory / "classes").exists()
        ids = (directory / "selected_ids.txt").read_text().split()
        assert ids == result.record_ids
        resolved = yaml.safe_load((directory / "config.resolved.yaml").read_text())
        assert resolved["profile"] == "reference-723696d"
        assert resolved["coverage"]["min_structure_support"] == 2
        assert resolved["selection"]["tie_break"] == "natoms_then_record_id"
        coverage = json.loads((directory / "coverage.json").read_text())
        assert "diagnostics" in coverage
        table = export.manifest_table(result)
        assert table["record_id"].to_pylist() == result.record_ids
        assert table["reason"].to_pylist() == [s.reason for s in result.selected]
        assert table["order"].to_pylist() == list(range(len(result.selected)))
        saved = result.write_manifest(tmp_path / "with", config={"custom": 1})
        assert (saved / "classes" / "classes.parquet").exists()
        assert yaml.safe_load((saved / "config.resolved.yaml").read_text()) == {
            "custom": 1
        }

    def test_provenance(self, result: SelectionResult) -> None:
        provenance = collect(result)
        assert provenance["profile"] == "reference-723696d"
        assert provenance["reference"]["commit"].startswith("723696d")
        assert provenance["dependencies"]["duckdb"]
        assert {"source", "geometry", "corpus", "token", "cache"} <= set(
            provenance["fingerprints"]
        )
        assert provenance["fingerprints"]["selection"] == result.fingerprint
        assert provenance["fingerprints"]["classes"] == result.classes.fingerprint
        assert provenance["source"]["adapter"] == "extxyz"
        assert provenance["cache"]["summary"]["discovered"] == 20
        assert provenance["resources"]["peak_rss_bytes"] > 0


def assert_frame_matches(exported: Atoms, source: Atoms) -> None:
    assert exported.get_chemical_symbols() == source.get_chemical_symbols()
    np.testing.assert_allclose(exported.positions, source.positions)
    np.testing.assert_allclose(exported.cell.array, source.cell.array)
    assert list(exported.pbc) == list(source.pbc)
    assert exported.get_potential_energy() == pytest.approx(
        source.get_potential_energy()
    )
    np.testing.assert_allclose(exported.get_forces(), source.get_forces())
    np.testing.assert_allclose(exported.get_stress(), source.get_stress())
    for key in ("sid", "fixture", "charge", "method"):
        assert exported.info[key] == source.info[key]


class TestExport:
    def test_custom_array_database_round_trip(self, tmp_path: Path) -> None:
        frame = load_frames()[0]
        weights = np.arange(len(frame) * 2, dtype=np.float32).reshape(-1, 2)
        frame.new_array("site_weight", weights)
        frame.info[ARRAYS_KEY] = "source metadata"
        corpus = Corpus.from_atoms(
            [frame],
            name="custom",
            split="train",
            label_sources={"site_weight": "arrays:site_weight"},
        )
        cache = Tokenizer().run(corpus, tmp_path)
        result = Selector().select(CoverageBuilder(min_structure_support=2).fit(cache))
        path = result.export(tmp_path / "export.db", format="asedb", corpus=corpus)
        restored = row_atoms(ase.db.connect(str(path)), 1)
        np.testing.assert_array_equal(
            restored.arrays["site_weight"], weights, strict=True
        )
        assert restored.info[ARRAYS_KEY] == "source metadata"
        assert_frame_matches(restored, frame)

    def test_extxyz_round_trip(self, result: SelectionResult, tmp_path: Path) -> None:
        sources = {str(a.info["sid"]): a for a in load_frames()}
        path = result.export(tmp_path / "subset.extxyz")
        frames = ase.io.read(path, index=":", format="extxyz")
        assert isinstance(frames, list)
        assert [f.info["envtok_record_id"] for f in frames] == result.record_ids
        assert [f.info["envtok_reason"] for f in frames] == [
            s.reason for s in result.selected
        ]
        for frame in frames:
            assert isinstance(frame.calc, SinglePointCalculator)
            assert_frame_matches(frame, sources[str(frame.info["sid"])])
        plain = result.export(tmp_path / "plain.extxyz", annotate=False)
        first = ase.io.read(plain, index=0, format="extxyz")
        assert isinstance(first, Atoms)
        assert "envtok_record_id" not in first.info

    def test_ase_database_export(self, result: SelectionResult, tmp_path: Path) -> None:
        sources = {str(a.info["sid"]): a for a in load_frames()}
        path = result.export(tmp_path / "subset.db", format="asedb")
        database = ase.db.connect(str(path))
        rows = list(database.select())
        assert [
            r.key_value_pairs["envtok_record_id"] for r in rows
        ] == result.record_ids
        for row in rows:
            atoms = row.toatoms()
            atoms.info.update(row.data)
            assert_frame_matches(atoms, sources[str(row.data["sid"])])
        with pytest.raises(ConfigError, match="format must be"):
            result.export(tmp_path / "x.cif", format="cif")

    def test_in_memory_and_spooled_sources(self, tmp_path: Path) -> None:
        frames = load_frames()
        chosen = [frames[i] for i in (0, 7)]
        for frame in chosen:
            frame.info.update(
                prototype_error="", following_value=1.25, source_row_key="1"
            )
        memory = Corpus.from_atoms(chosen, name="mem", split="train")
        cache = Tokenizer().run(memory, tmp_path / "mem")
        selection = Selector().select(
            CoverageBuilder(min_structure_support=2).fit(cache)
        )
        with pytest.raises(RetrievalUnavailableError, match="in-memory"):
            selection.export(tmp_path / "x.extxyz")
        written = selection.export(tmp_path / "mem.extxyz", corpus=memory)
        exported = ase.io.read(written, index=":")
        assert isinstance(exported, list)
        assert len(exported) == len(selection.selected)
        for frame in exported:
            assert frame.info["prototype_error"] == ""
            assert frame.info["following_value"] == 1.25
            assert frame.info["source_row_key"] == "1"
        assert all(frame.info["prototype_error"] == "" for frame in chosen)
        assert collect(selection)["source"]["adapter"] == "ase-iterable"

        spooled = Corpus.from_atoms(
            iter(chosen), name="sp", split="train", spool_to=tmp_path / "spool.extxyz"
        )
        spooled_cache = Tokenizer().run(spooled, tmp_path / "sp")
        spooled_selection = Selector().select(
            CoverageBuilder(min_structure_support=2).fit(spooled_cache)
        )
        copy = spooled_selection.export(tmp_path / "sp.extxyz")
        frames_out = ase.io.read(copy, index=":")
        assert isinstance(frames_out, list)
        sources = {str(a.info["sid"]): a for a in chosen}
        for frame in frames_out:
            assert_frame_matches(frame, sources[str(frame.info["sid"])])
            assert frame.info["prototype_error"] == ""
            assert frame.info["following_value"] == 1.25
            assert frame.info["source_row_key"] == "1"

        chosen[0].info["charge"] = 99
        with pytest.raises(StaleCacheError, match="original unchanged corpus"):
            selection.export(tmp_path / "changed.extxyz", corpus=memory)
        assert not (tmp_path / "changed.extxyz").exists()

    def test_export_rejects_a_different_explicit_corpus(
        self, result: SelectionResult, tmp_path: Path
    ) -> None:
        wrong = Corpus.from_atoms(load_frames(), name="different", split="train")
        target = tmp_path / "existing.extxyz"
        target.write_text("existing output\n")
        with pytest.raises(StaleCacheError, match="supplied corpus differs"):
            result.export(target, corpus=wrong)
        assert target.read_text() == "existing output\n"

    def test_calculators_are_frozen(self) -> None:
        copper = bulk("Cu", "fcc", a=3.6)
        copper.calc = EMT()
        energy = copper.get_potential_energy()
        frozen = export._frozen(copper)
        assert isinstance(frozen.calc, SinglePointCalculator)
        assert frozen.get_potential_energy() == pytest.approx(energy)
        fresh = bulk("Cu", "fcc", a=3.6)
        fresh.calc = EMT()
        assert export._frozen(fresh).calc is None
        assert export._frozen(bulk("Cu", "fcc", a=3.6)).calc is None

    def test_jsonable(self) -> None:
        assert export._jsonable(np.arange(2)) == [0, 1]
        assert export._jsonable(np.int64(3)) == 3
        assert export._jsonable("x") == "x"


def test_selected_records_keep_their_index(result: SelectionResult) -> None:
    first: Selected = result.selected[0]
    assert result.classes.pool.record_ids[first.record_index] == first.record_id
