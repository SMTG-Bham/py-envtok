"""Dataset adapters: extxyz, ASE databases, OMat24 ASE-LMDB and in-memory iterables."""

import bz2
import gzip
import io
import lzma
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import ase.db
import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.adapters import (
    AseDbAdapter,
    AseIterableAdapter,
    ExtxyzAdapter,
    IdPolicy,
    OMat24Adapter,
    RecordOptions,
    adapter_from_spec,
    make_record,
    spool_atoms,
)
from envtok.adapters.asedb import close_shared_databases, row_atoms, row_ids
from envtok.adapters.base import Locator, file_fingerprint, sorted_files
from envtok.adapters.extxyz import scan_frames
from envtok.errors import (
    ConfigError,
    RetrievalUnavailableError,
    UnsupportedInputError,
)
from envtok.records import SourceRef, Split

OPTIONS = RecordOptions(dataset="fx", revision="v1", split=Split.TRAIN)


def labelled(atoms: Atoms, energy: float, sid: str) -> Atoms:
    frame: Atoms = atoms.copy()
    frame.calc = SinglePointCalculator(
        frame,
        energy=energy,
        forces=np.full((len(frame), 3), energy),
        stress=np.full(6, energy / 10),
    )
    frame.info["sid"] = sid
    return frame


@pytest.fixture
def frames() -> list[Atoms]:
    nacl = bulk("NaCl", "rocksalt", a=5.64)
    zns = bulk("ZnS", "zincblende", a=5.41)
    frames = [
        labelled(nacl, -1.0, "a"),
        labelled(zns, -2.0, "b"),
        labelled(nacl, -3.0, "a"),
    ]
    frames[1].info["split"] = "test"
    frames[1].info["parent"] = "zns"
    return frames


def write(path: Path, frames: list[Atoms]) -> Path:
    ase.io.write(path, frames, format="extxyz")
    return path


def assert_same(a: Atoms, b: Atoms) -> None:
    assert a.get_chemical_symbols() == b.get_chemical_symbols()
    np.testing.assert_allclose(a.positions, b.positions)
    np.testing.assert_allclose(a.cell.array, b.cell.array)
    assert a.get_potential_energy() == pytest.approx(b.get_potential_energy())


class TestExtxyz:
    def test_full_checksum_replaces_staged_data(self, tmp_path: Path) -> None:
        path = tmp_path / "frames.xyz.gz"
        first = b"1\n\nH 1 0 0\n"
        path.write_bytes(gzip.compress(first, compresslevel=0, mtime=0))
        stat = path.stat()
        adapter = ExtxyzAdapter(
            path,
            OPTIONS,
            compressed="stage",
            staging_dir=tmp_path / "stage",
            checksum="full",
        )
        before = adapter.source_fingerprint()
        assert next(adapter.iter_records()).atoms.positions[0, 0] == 1
        path.write_bytes(
            gzip.compress(first.replace(b"H 1", b"H 2"), compresslevel=0, mtime=0)
        )
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        assert path.stat().st_size == stat.st_size
        assert next(adapter.iter_records()).atoms.positions[0, 0] == 2
        assert adapter.source_fingerprint() != before
        assert adapter.read(next(adapter.locators())).atoms.positions[0, 0] == 2

    def test_records_ids_labels_and_offsets(
        self, tmp_path: Path, frames: list[Atoms]
    ) -> None:
        write(tmp_path / "data.extxyz", frames)
        options = RecordOptions(
            dataset="fx",
            revision="v1",
            split=Split.TRAIN,
            split_key="split",
            group_key="parent",
        )
        adapter = ExtxyzAdapter(tmp_path / "data.extxyz", options)
        locators = list(adapter.locators())
        assert [loc.key for loc in locators] == ["0", "1", "2"]
        assert all(loc.offset is not None for loc in locators)
        records = [adapter.read(loc) for loc in locators]
        streamed = list(adapter.iter_records())
        assert [r.record_id for r in records] == [
            "fx@v1:data.extxyz:0",
            "fx@v1:data.extxyz:1",
            "fx@v1:data.extxyz:2",
        ]
        assert [r.record_id for r in streamed] == [r.record_id for r in records]
        assert records[1].split is Split.TEST
        assert records[0].split is Split.TRAIN
        assert records[1].group_id == "zns"
        assert records[0].source_ref.native_id == "a"
        assert float(records[2].labels["energy"].value) == -3.0
        assert records[0].labels["forces"].unit == "eV/Ang"
        assert records[0].metadata["sid"] == "a"
        assert adapter.supports_random_access
        fingerprint = adapter.source_fingerprint()
        assert fingerprint["files"][0]["shard"] == "data.extxyz"
        refs = [records[2].source_ref, records[0].source_ref]
        retrieved = list(adapter.retrieve(refs))
        assert [ref.key for ref, _ in retrieved] == ["2", "0"]
        assert_same(retrieved[0][1], frames[2])

    def test_directory_sources_are_sorted_and_namespaced(
        self, tmp_path: Path, frames: list[Atoms]
    ) -> None:
        (tmp_path / "b").mkdir()
        write(tmp_path / "b" / "z.xyz", frames[:1])
        write(tmp_path / "a.extxyz", frames[1:])
        adapter = ExtxyzAdapter(tmp_path, OPTIONS)
        with pytest.raises(KeyError):
            adapter._path_of("missing.xyz")
        last = adapter.read(list(adapter.locators())[-1])
        ((_, atoms),) = list(adapter.retrieve([last.source_ref]))
        assert_same(atoms, frames[0])
        assert [(loc.shard, loc.key, loc.ordinal) for loc in adapter.locators()] == [
            ("a.extxyz", "0", 0),
            ("a.extxyz", "1", 1),
            ("b/z.xyz", "0", 2),
        ]
        rebuilt = adapter_from_spec(adapter.spec())
        assert isinstance(rebuilt, ExtxyzAdapter)
        assert rebuilt.source_fingerprint() == adapter.source_fingerprint()

    @pytest.mark.parametrize(
        ("suffix", "opener"),
        [(".gz", gzip.open), (".bz2", bz2.open), (".xz", lzma.open)],
    )
    def test_compressed_files_stream_sequentially(
        self, tmp_path: Path, frames: list[Atoms], suffix: str, opener: object
    ) -> None:
        plain = write(tmp_path / "plain.xyz", frames)
        packed = tmp_path / f"data.xyz{suffix}"
        with plain.open("rb") as source, opener(packed, "wb") as target:  # type: ignore[operator]
            shutil.copyfileobj(source, target)
        plain.unlink()
        adapter = ExtxyzAdapter(packed, OPTIONS)
        assert not adapter.supports_random_access
        assert [loc.offset for loc in adapter.locators()] == [None, None, None]
        records = list(adapter.iter_records())
        assert len(records) == 3
        with pytest.raises(UnsupportedInputError, match="compressed"):
            adapter.read(next(adapter.locators()))
        ((ref, atoms),) = list(adapter.retrieve([records[1].source_ref]))
        assert ref.key == "1"
        assert_same(atoms, frames[1])

    def test_compressed_files_can_be_staged(
        self, tmp_path: Path, frames: list[Atoms]
    ) -> None:
        plain = write(tmp_path / "plain.xyz", frames)
        packed = tmp_path / "data.xyz.gz"
        with plain.open("rb") as source, gzip.open(packed, "wb") as target:
            shutil.copyfileobj(source, target)
        plain.unlink()
        staging = tmp_path / "staging"
        adapter = ExtxyzAdapter(
            packed, OPTIONS, compressed="stage", staging_dir=staging
        )
        assert adapter.supports_random_access
        locators = list(adapter.locators())
        assert locators[2].offset is not None
        assert_same(adapter.read(locators[2]).atoms, frames[2])
        assert len(list(staging.iterdir())) == 1
        again = ExtxyzAdapter(packed, OPTIONS, compressed="stage", staging_dir=staging)
        assert [loc.offset for loc in again.locators()] == [
            loc.offset for loc in locators
        ]
        ((_, atoms),) = list(adapter.retrieve([adapter.read(locators[0]).source_ref]))
        assert_same(atoms, frames[0])

    def test_invalid_configuration(self, tmp_path: Path, frames: list[Atoms]) -> None:
        write(tmp_path / "data.xyz", frames)
        with pytest.raises(ConfigError, match="compressed must be"):
            ExtxyzAdapter(tmp_path / "data.xyz", OPTIONS, compressed="random")
        with pytest.raises(ConfigError, match="staging_dir"):
            ExtxyzAdapter(tmp_path / "data.xyz", OPTIONS, compressed="stage")
        with pytest.raises(ConfigError, match="does not exist"):
            ExtxyzAdapter(tmp_path / "missing", OPTIONS)
        (tmp_path / "empty").mkdir()
        with pytest.raises(ConfigError, match="no files"):
            ExtxyzAdapter(tmp_path / "empty", OPTIONS)

    def test_frame_scanner(self) -> None:
        good = io.BytesIO(b"1\ncomment\nH 0 0 0\n\n2\nc\nH 0 0 0\nH 0 0 1\n")
        assert scan_frames(good, "good") == [0, 19]
        with pytest.raises(UnsupportedInputError, match="integer atom-count"):
            scan_frames(io.BytesIO(b"x\n"), "bad")


class TestRecordOptions:
    def test_native_id_policy(self, frames: list[Atoms]) -> None:
        options = RecordOptions(
            dataset="d", revision="r", id_policy=IdPolicy.NAMESPACED_NATIVE_ID
        )
        assert make_record(frames[0], "s", "0", options).record_id == "d@r:a"
        anonymous = frames[0].copy()
        del anonymous.info["sid"]
        with pytest.raises(ConfigError, match="no 'sid'"):
            make_record(anonymous, "s", "0", options)

    def test_option_validation_and_label_sources(self, frames: list[Atoms]) -> None:
        with pytest.raises(ConfigError, match="non-empty"):
            RecordOptions(dataset="", revision="r")
        with pytest.raises(ConfigError, match="native_id_key"):
            RecordOptions(
                dataset="d",
                revision="r",
                id_policy=IdPolicy.NAMESPACED_NATIVE_ID,
                native_id_key=None,
            )
        atoms = frames[0].copy()
        atoms.calc = None
        atoms.info["REF_energy"] = 1.5
        atoms.arrays["REF_forces"] = np.ones((2, 3))
        options = RecordOptions(
            dataset="d",
            revision="r",
            native_id_key=None,
            label_sources={
                "energy": "info:REF_energy",
                "forces": "arrays:REF_forces",
                "stress": "calc:stress",
            },
        )
        record = make_record(atoms, "s", "0", options)
        assert sorted(record.labels) == ["energy", "forces"]
        assert record.labels["energy"].source_key == "info:REF_energy"
        assert "REF_energy" not in record.metadata
        assert record.source_ref.native_id is None
        bad = RecordOptions(
            dataset="d", revision="r", label_sources={"energy": "energy"}
        )
        with pytest.raises(ConfigError, match="label source"):
            make_record(atoms, "s", "0", bad)

    def test_file_fingerprints_and_globbing(self, tmp_path: Path) -> None:
        path = tmp_path / "x.xyz"
        path.write_text("1\n\nH 0 0 0\n")
        stat = file_fingerprint(path)
        assert set(stat) == {"name", "size", "mtime_ns"}
        assert len(file_fingerprint(path, checksum="full")["sha256"]) == 64
        with pytest.raises(ConfigError, match="checksum"):
            file_fingerprint(path, checksum="md5")
        assert sorted_files(path, ["*.db"]) == [path]

    def test_unknown_adapter_kind(self) -> None:
        with pytest.raises(ConfigError, match="unknown adapter kind"):
            adapter_from_spec({"kind": "parquet", "options": OPTIONS.as_dict()})


def write_database(path: Path, frames: list[Atoms]) -> None:
    database = ase.db.connect(str(path))
    for frame in frames:
        sid = frame.info["sid"]
        database.write(frame, key_value_pairs={"tag": f"t-{sid}"}, data={"sid": sid})
    closer = getattr(database, "close", None)
    if closer is not None:
        closer()


@pytest.fixture(autouse=True)
def _close_lmdb() -> Iterator[None]:
    yield
    close_shared_databases()


class TestAseDatabases:
    def test_sqlite_database(self, tmp_path: Path, frames: list[Atoms]) -> None:
        write_database(tmp_path / "data.db", frames)
        adapter = AseDbAdapter(tmp_path / "data.db", OPTIONS)
        locators = list(adapter.locators())
        assert [loc.key for loc in locators] == ["1", "2", "3"]
        records = [adapter.read(loc) for loc in locators]
        assert records[0].record_id == "fx@v1:data.db:1"
        assert records[1].source_ref.native_id == "b"
        assert records[1].metadata["tag"] == "t-b"
        assert float(records[2].labels["energy"].value) == -3.0
        ((ref, atoms),) = list(adapter.retrieve([records[1].source_ref]))
        assert ref.key == "2"
        assert_same(atoms, frames[1])
        rebuilt = adapter_from_spec(adapter.spec())
        assert isinstance(rebuilt, AseDbAdapter)
        assert rebuilt.source_fingerprint() == adapter.source_fingerprint()
        adapter.close()
        adapter.close()

    def test_omat24_lmdb_shards(self, tmp_path: Path, frames: list[Atoms]) -> None:
        (tmp_path / "rattled-300").mkdir()
        (tmp_path / "aimd-npt").mkdir()
        write_database(tmp_path / "rattled-300" / "data.0000.aselmdb", frames[:2])
        write_database(tmp_path / "aimd-npt" / "data.0000.aselmdb", frames[2:])
        options = RecordOptions(dataset="omat24", revision="v1", split=Split.TRAIN)
        adapter = OMat24Adapter(tmp_path, options)
        assert [(loc.shard, loc.key) for loc in adapter.locators()] == [
            ("aimd-npt/data.0000.aselmdb", "1"),
            ("rattled-300/data.0000.aselmdb", "1"),
            ("rattled-300/data.0000.aselmdb", "2"),
        ]
        records = list(adapter.iter_records())
        assert records[0].record_id == "omat24@v1:aimd-npt/data.0000.aselmdb:1"
        assert records[0].source_ref.native_id == "a"
        assert records[2].metadata["omat_subset"] == "rattled-300"
        np.testing.assert_allclose(
            records[2].labels["forces"].value, frames[1].get_forces()
        )
        assert adapter_from_spec(adapter.spec()).spec() == adapter.spec()
        close_shared_databases()
        single = OMat24Adapter(tmp_path / "rattled-300", options)
        assert (
            single.read(Locator(0, "data.0000.aselmdb", "1")).metadata["omat_subset"]
            == "rattled-300"
        )
        adapter.close()

    def test_capabilities_and_shared_handles(
        self, tmp_path: Path, frames: list[Atoms]
    ) -> None:
        write_database(tmp_path / "data.aselmdb", frames[:1])
        adapter = OMat24Adapter(tmp_path / "data.aselmdb", OPTIONS)
        assert adapter.supports_random_access
        assert adapter.supports_retrieval
        assert adapter.retrieval_limitation is None
        first = adapter.read(next(adapter.locators()))
        again = OMat24Adapter(tmp_path / "data.aselmdb", OPTIONS)
        assert again.read(next(again.locators())).record_id == first.record_id
        adapter.close()
        assert again.read(next(again.locators())).record_id == first.record_id

    def test_database_helpers_without_fast_paths(
        self, tmp_path: Path, frames: list[Atoms]
    ) -> None:
        write_database(tmp_path / "data.db", frames[:1])
        database = ase.db.connect(str(tmp_path / "data.db"))

        class Plain:
            def select(self) -> object:
                return database.select()

            def get(self, id: int) -> object:  # noqa: A002 - ASE keyword name
                return database.get(id=id)

        plain = Plain()
        assert row_ids(plain) == [1]
        assert row_atoms(plain, 1).info["sid"] == "a"

        class NoData:
            def get(self, id: int) -> object:  # noqa: A002 - ASE keyword name
                row = database.get(id=id)
                return SimpleNamespace(
                    toatoms=row.toatoms, key_value_pairs={"k": 1}, data=None
                )

        assert row_atoms(NoData(), 1).info == {"k": 1}

    def test_missing_lmdb_backend_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from envtok.adapters import asedb  # noqa: PLC0415

        def missing(*_args: object, **_kwargs: object) -> None:
            raise ImportError("no ase_db_backends")

        monkeypatch.setattr(ase.db, "connect", missing)
        shard = tmp_path / "x.aselmdb"
        shard.touch()
        with pytest.raises(ConfigError, match=r"envtok\[omat24\]"):
            asedb.open_database(shard)


class TestIterable:
    def test_sequence_source(self, frames: list[Atoms]) -> None:
        adapter = AseIterableAdapter(frames, OPTIONS)
        assert not adapter.supports_random_access
        assert adapter.supports_retrieval
        assert adapter.retrieval_limitation is None
        records = list(adapter.iter_records())
        assert [r.record_id for r in records] == [
            "fx@v1:memory:0",
            "fx@v1:memory:1",
            "fx@v1:memory:2",
        ]
        assert adapter.read(Locator(1, "memory", "1")).record_id == records[1].record_id
        for index, record in enumerate(records):
            assert_same(record.atoms, frames[index])
            assert float(record.labels["energy"].value) == -(index + 1)
            np.testing.assert_array_equal(
                record.labels["forces"].value, frames[index].get_forces()
            )
            assert record.atoms.calc is not frames[index].calc
        assert_same(adapter.read(Locator(1, "memory", "1")).atoms, frames[1])
        assert adapter.source_fingerprint()["content_sha256"]
        assert len(list(adapter.locators())) == 3
        ((_, atoms),) = list(adapter.retrieve([records[2].source_ref]))
        assert_same(atoms, frames[2])
        with pytest.raises(UnsupportedInputError, match="portable spec"):
            adapter.spec()

    def test_factory_source(self, frames: list[Atoms]) -> None:
        adapter = AseIterableAdapter(lambda: iter(frames), OPTIONS)
        assert len(list(adapter.iter_records())) == 3
        assert len(list(adapter.iter_records())) == 3
        assert adapter.source_fingerprint()["content_sha256"]
        with pytest.raises(UnsupportedInputError, match="only sequence"):
            adapter.read(Locator(0, "memory", "0"))
        ref = SourceRef("fx", "v1", "memory", "1")
        assert_same(next(adapter.retrieve([ref]))[1], frames[1])

    @pytest.mark.parametrize("factory", [False, True])
    @pytest.mark.parametrize("changed", ["moments", "metadata", "labels", "arrays"])
    def test_fingerprint_tracks_non_geometry_inputs(
        self, frames: list[Atoms], factory: bool, changed: str
    ) -> None:
        adapter = AseIterableAdapter(
            (lambda: iter(frames)) if factory else frames, OPTIONS
        )
        original = adapter.source_fingerprint()
        if changed == "moments":
            frames[0].set_initial_magnetic_moments([1.0] * len(frames[0]))
        elif changed == "metadata":
            frames[0].info.update(charge=1, split="validation", nested={"v": [1]})
        elif changed == "labels":
            frames[0].calc.results["energy"] = -9.0
        else:
            frames[0].new_array("weights", np.ones(len(frames[0])))
        assert adapter.source_fingerprint() != original

    def test_unlabelled_frames_and_one_shot_cache_identity(self) -> None:
        atoms = Atoms("H")
        sequence = AseIterableAdapter([atoms], OPTIONS)
        assert next(sequence.iter_records()).labels == {}
        assert sequence.source_fingerprint()["content_sha256"]
        first = AseIterableAdapter(iter([atoms]), OPTIONS)
        second = AseIterableAdapter(iter([atoms]), OPTIONS)
        fingerprint = first.source_fingerprint()
        assert fingerprint["content_sha256"] is None
        assert first.source_fingerprint() == fingerprint
        assert second.source_fingerprint() != fingerprint
        assert len(list(first.iter_records())) == 1

    def test_custom_stored_labels_are_copied_without_calculation(
        self, frames: list[Atoms], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        atoms = frames[0]
        values = np.ones((len(atoms), 3))
        atoms.calc.results["custom_forces"] = values

        def forbid_calculation(*_args: object, **_kwargs: object) -> None:
            pytest.fail("the adapter must never request a calculator property")

        monkeypatch.setattr(atoms.calc, "get_property", forbid_calculation)
        options = RecordOptions(
            dataset="custom",
            revision="v1",
            label_sources={"forces": "calc:custom_forces"},
        )
        adapter = AseIterableAdapter([atoms], options)
        record = next(adapter.iter_records())
        np.testing.assert_array_equal(record.labels["forces"].value, values)
        atoms.calc.results["custom_forces"][0, 0] = 9
        assert record.atoms.calc.results["custom_forces"][0, 0] == 1

    def test_one_shot_source_explains_retrieval_limits(
        self, frames: list[Atoms]
    ) -> None:
        adapter = AseIterableAdapter(iter(frames), OPTIONS)
        assert not adapter.supports_retrieval
        assert "one-shot" in str(adapter.retrieval_limitation)
        assert len(list(adapter.iter_records())) == 3
        with pytest.raises(RetrievalUnavailableError, match="already consumed"):
            list(adapter.iter_records())
        with pytest.raises(RetrievalUnavailableError, match="spool"):
            list(adapter.retrieve([SourceRef("fx", "v1", "memory", "0")]))

    def test_spooling(self, tmp_path: Path, frames: list[Atoms]) -> None:
        target = tmp_path / "store" / "spool.extxyz"
        digest = spool_atoms(iter(frames), target)
        assert len(digest) == 64
        assert digest == spool_atoms(iter(frames), target)
        spooled = list(ExtxyzAdapter(target, OPTIONS).iter_records())
        assert [r.source_ref.native_id for r in spooled] == ["a", "b", "a"]
        np.testing.assert_allclose(
            spooled[1].labels["forces"].value, frames[1].get_forces()
        )

    def test_repeated_spooling_preserves_empty_array_and_adjacent_metadata(
        self, tmp_path: Path, frames: list[Atoms]
    ) -> None:
        atoms = frames[0]
        empty = np.empty(0)
        warnings = ["A warning with spaces and punctuation."]
        values = np.asarray([1.5, 2.5])
        atoms.info.update(
            empty_adjustments=empty,
            correction_warnings=warnings,
            adjustment_values=values,
            source_row_key="001",
        )
        current = atoms
        for index in range(3):
            target = tmp_path / f"spool-{index}.extxyz"
            spool_atoms([current], target)
            current = cast("Atoms", ase.io.read(target))
            assert isinstance(current.info["empty_adjustments"], np.ndarray)
            np.testing.assert_array_equal(current.info["empty_adjustments"], empty)
            assert current.info["correction_warnings"] == warnings
            np.testing.assert_array_equal(current.info["adjustment_values"], values)
            assert current.info["source_row_key"] == "001"
            assert current.get_potential_energy() == atoms.get_potential_energy()
            np.testing.assert_array_equal(current.get_forces(), atoms.get_forces())
        assert atoms.info["empty_adjustments"] is empty
        assert atoms.info["correction_warnings"] is warnings
        assert atoms.info["adjustment_values"] is values
