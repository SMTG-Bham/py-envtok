"""Token store: fingerprints, partitions, the executor, resume and the index stage."""

import dataclasses
import enum
import json
import os
import sys
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import ase.db
import ase.io
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ase import Atoms
from ase.build import bulk
from helpers import CORPUS

from envtok import __version__
from envtok.corpus import Corpus
from envtok.errors import (
    ConfigError,
    DuplicateRecordError,
    EnvTokError,
    IncompleteCacheError,
    RetrievalUnavailableError,
    StaleCacheError,
)
from envtok.execution import ExecutionSettings, parse_size, peak_rss_bytes
from envtok.fingerprint import (
    canonical_json,
    digest,
    json_default,
    package_versions,
)
from envtok.profiles import REFERENCE_723696D, MagneticPolicy, corpus_policy
from envtok.records import Split
from envtok.store import executor
from envtok.store.cache import (
    CACHE_FILE,
    INDEX_FILE,
    PLAN_FILE,
    TokenCache,
    site_keys,
)
from envtok.store.duck import connect, sql_literal
from envtok.store.executor import (
    Batch,
    BatchSummary,
    FaultPlan,
    PlanStats,
    TaskContext,
    plan_batches,
    thread_limits,
)
from envtok.store.partitions import (
    COMMIT_MARKER,
    committed_partitions,
    partition_name,
    read_marker,
    remove_temporary,
    write_partition,
)
from envtok.store.schema import SITES, rows_to_table
from envtok.tokenization import Tokenizer
from envtok.tokenizer.pipeline import SiteRow, site_token

PROFILE = REFERENCE_723696D


def frames(indices: list[int] | None = None) -> list[Atoms]:
    loaded = ase.io.read(CORPUS, index=":", format="extxyz")
    assert isinstance(loaded, list)
    return loaded if indices is None else [loaded[i] for i in indices]


def write_frames(path: Path, selected: list[Atoms]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(path, selected, format="extxyz")
    return path


def corpus_of(path: Path, **options: Any) -> Corpus:
    base: dict[str, Any] = {"dataset": "fx", "revision": "v1", "split": "train"}
    base.update(options)
    return Corpus.from_extxyz(path, **base)


def tables(cache: TokenCache) -> dict[str, pa.Table]:
    return {name: cache.read(name) for name in TokenCache.TABLES}


DIAGNOSTIC_COLUMNS: frozenset[str] = frozenset()
"""Columns excluded from run comparisons (none: every column must match)."""


def assert_same_tables(a: TokenCache, b: TokenCache) -> None:
    for name, table in tables(a).items():
        other = b.read(name)
        differing = [
            column
            for column in table.column_names
            if column not in DIAGNOSTIC_COLUMNS
            and not table[column].equals(other[column])
        ]
        assert differing == [], name


@pytest.fixture(scope="module")
def subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Six small Na-Cl fixtures in one file (three batches of two)."""
    root = tmp_path_factory.mktemp("subset")
    return write_frames(root / "subset.extxyz", frames([0, 1, 3, 4, 7, 11]))


@pytest.fixture(scope="module")
def serial(subset: Path, tmp_path_factory: pytest.TempPathFactory) -> TokenCache:
    """Uninterrupted serial reference run of ``subset``."""
    return Tokenizer().run(
        corpus_of(subset), tmp_path_factory.mktemp("serial"), batch_size=2
    )


@pytest.fixture(scope="module")
def full_cache(tmp_path_factory: pytest.TempPathFactory) -> TokenCache:
    """The whole fixture corpus, shared through the session base directory."""
    workdir = tmp_path_factory.getbasetemp() / "envtok-shared-full"
    return Tokenizer().run(corpus_of(CORPUS), workdir, batch_size=5)


class TestFingerprints:
    def test_canonical_json_is_deterministic(self) -> None:
        assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
        assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})

    def test_json_default_handles_package_types(self) -> None:
        class Colour(enum.Enum):
            RED = "red"

        @dataclasses.dataclass
        class Point:
            x: int

        assert json_default(Colour.RED) == "red"
        assert json_default(frozenset({"b", "a"})) == ["a", "b"]
        assert json_default({3, 1}) == [1, 3]
        assert json_default(Path("a/b")) == "a/b"
        assert json_default(Point(1)) == {"x": 1}
        with pytest.raises(TypeError, match="cannot fingerprint"):
            json_default(object())
        with pytest.raises(TypeError):
            json_default(Point)

    def test_package_versions(self) -> None:
        versions = package_versions(("numpy", "envtok", "no-such-package-xyz"))
        assert versions["envtok"] == __version__
        assert versions["numpy"]
        assert versions["no-such-package-xyz"] is None


class TestExecutionSettings:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, None),
            (7, 7),
            ("32GB", 32 * 10**9),
            ("1.5GiB", int(1.5 * 2**30)),
            ("512MiB", 512 * 2**20),
            (" 100 ", 100),
            ("2b", 2),
        ],
    )
    def test_parse_size(self, value: str | int | None, expected: int | None) -> None:
        assert parse_size(value) == expected

    @pytest.mark.parametrize("value", ["abc", "5 XB", "GB"])
    def test_parse_size_rejects_text(self, value: str) -> None:
        with pytest.raises(ConfigError, match="cannot parse size"):
            parse_size(value)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"workers": 0},
            {"batch_size": 0},
            {"threads_per_worker": 0},
            {"max_inflight_batches": 0},
            {"duckdb_memory_fraction": 0.0},
            {"duckdb_memory_fraction": 1.5},
        ],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ConfigError):
            ExecutionSettings(**kwargs)

    def test_inflight_and_duckdb(self, tmp_path: Path) -> None:
        assert ExecutionSettings(workers=3).inflight == 6
        assert ExecutionSettings(workers=3, max_inflight_batches=1).inflight == 1
        settings = ExecutionSettings(
            memory_budget=2**30, temporary_disk_budget=2**31, duckdb_threads=1
        )
        connection = settings.duckdb(tmp_path)
        try:
            limit = connection.execute(
                "SELECT current_setting('memory_limit')"
            ).fetchall()[0][0]
            assert "512" in str(limit) or "536" in str(limit)
            assert connection.execute("SELECT current_setting('threads')").fetchall()[
                0
            ][0] in {1, "1"}
        finally:
            connection.close()
        assert (tmp_path / "tmp" / "duckdb").is_dir()
        plain = ExecutionSettings().duckdb(tmp_path)
        plain.close()

    def test_peak_rss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rss = peak_rss_bytes()
        assert rss is not None
        assert rss > 0
        monkeypatch.setitem(sys.modules, "resource", None)
        assert peak_rss_bytes() is None


class TestDuckAndSchema:
    def test_sql_literal_and_connect(self, tmp_path: Path) -> None:
        assert sql_literal("it's") == "'it''s'"
        connection = connect(
            memory_limit="256MiB",
            threads=1,
            temp_directory=tmp_path / "spill",
            max_temp_directory_size="1GiB",
        )
        try:
            assert connection.execute(
                f"SELECT {sql_literal(Path('a'))}"
            ).fetchall() == [("a",)]
        finally:
            connection.close()
        assert (tmp_path / "spill").is_dir()
        assert isinstance(connect(), duckdb.DuckDBPyConnection)

    def test_empty_rows_keep_the_schema(self) -> None:
        table = rows_to_table([], SITES)
        assert table.num_rows == 0
        assert table.schema == SITES


class TestPartitions:
    def test_commit_and_reuse(self, tmp_path: Path) -> None:
        root = tmp_path / "partitions"
        assert committed_partitions(root, "fp") == {}
        assert remove_temporary(root) == 0
        path = write_partition(root, 3, [], fingerprint="fp", keep_neighbours=False)
        assert path.name == partition_name(3) == "part-000003"
        marker = read_marker(path)
        assert marker is not None
        assert (marker["index"], marker["record_ids"], marker["tables"]) == (
            3,
            [],
            ["failures", "sites", "structures"],
        )
        stamp = (path / COMMIT_MARKER).stat().st_mtime_ns
        again = write_partition(root, 3, [], fingerprint="fp", keep_neighbours=True)
        assert again == path
        assert (path / COMMIT_MARKER).stat().st_mtime_ns == stamp
        (root / "part-000004").mkdir()
        assert read_marker(root / "part-000004") is None
        assert committed_partitions(root, "fp") == {3: path}
        with pytest.raises(StaleCacheError, match="stale"):
            committed_partitions(root, "other")
        (root / ".part-000005.tmp-abc").mkdir()
        assert remove_temporary(root) == 1

    def test_concurrent_commit_keeps_the_winner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "partitions"
        rename = Path.rename

        def racing(_self: Path, target: Path) -> Path:
            winner = Path(target)
            winner.mkdir(parents=True)
            (winner / COMMIT_MARKER).write_text("{}")
            raise OSError("exists")

        monkeypatch.setattr(Path, "rename", racing)
        path = write_partition(root, 0, [], fingerprint="fp", keep_neighbours=False)
        monkeypatch.setattr(Path, "rename", rename)
        assert (path / COMMIT_MARKER).read_text() == "{}"
        assert not list(root.glob(".part-*"))

    def test_rename_failure_without_winner_is_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(_self: Path, _target: Path) -> Path:
            raise OSError("disk")

        monkeypatch.setattr(Path, "rename", broken)
        with pytest.raises(OSError, match="disk"):
            write_partition(tmp_path, 0, [], fingerprint="fp", keep_neighbours=False)


def context_for(tmp_path: Path, faults: FaultPlan | None = None) -> TaskContext:
    return TaskContext(
        adapter_spec=None,
        profile=PROFILE,
        policy=corpus_policy(),
        partitions_dir=str(tmp_path),
        fingerprint="fp",
        keep_neighbours=False,
        faults=faults or FaultPlan(),
    )


class FakePool:
    """In-process stand-in for ProcessPoolExecutor whose first pool breaks."""

    instances = 0

    def __init__(self, max_workers: int, mp_context: object) -> None:
        del max_workers, mp_context
        FakePool.instances += 1
        self.number = FakePool.instances
        self.calls = 0

    def submit(self, function: Any, *args: Any) -> Future[Any]:
        self.calls += 1
        if self.number == 1 and self.calls == 2:
            raise BrokenProcessPool("pool broke while submitting")
        future: Future[Any] = Future()
        future.set_result(function(*args))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        del wait, cancel_futures


def fake_run(context: TaskContext, batch: Batch) -> BatchSummary:
    del context
    return BatchSummary(batch.index, 1, 1, 0, 0)


class TestExecutor:
    def test_plan_batches_counts_everything(self, subset: Path) -> None:
        adapter = corpus_of(subset).adapter
        batches = list(plan_batches(adapter, 4, lambda i: i == 1))
        assert [(b.index, len(b.locators)) for b in batches] == [(1, 2)]
        stats = PlanStats()
        assert len(list(plan_batches(adapter, 4, lambda _: True, stats))) == 2
        assert (stats.n_batches, stats.n_records) == (2, 6)

    def test_thread_limits_restore_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in executor.THREAD_VARIABLES:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "7")
        with thread_limits(2):
            assert os.environ["OMP_NUM_THREADS"] == "7"
            assert os.environ["OPENBLAS_NUM_THREADS"] == "2"
        assert os.environ["OMP_NUM_THREADS"] == "7"
        assert "OPENBLAS_NUM_THREADS" not in os.environ

    def test_crash_hook_only_fires_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        exits: list[int] = []

        class ExitedError(Exception):
            pass

        def fake_exit(code: int) -> None:
            exits.append(code)
            raise ExitedError

        monkeypatch.setattr(os, "_exit", fake_exit)
        executor._maybe_crash(0, FaultPlan())
        executor._maybe_crash(0, FaultPlan(crash_batches=frozenset({0})))
        plan = FaultPlan(crash_batches=frozenset({0}), marker_dir=str(tmp_path / "m"))
        with pytest.raises(ExitedError):
            executor._maybe_crash(0, plan)
        executor._maybe_crash(0, plan)
        assert exits == [17]
        assert (tmp_path / "m" / "crashed-0").exists()

    def test_interruption_in_a_pool_shuts_it_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakePool.instances = 1  # every fake pool works
        monkeypatch.setattr(executor, "ProcessPoolExecutor", FakePool)
        monkeypatch.setattr(executor, "run_batch", fake_run)
        batches = [Batch(index=i) for i in range(3)]
        context = context_for(tmp_path, FaultPlan(raise_after=2))
        with pytest.raises(EnvTokError, match="deliberate interruption"):
            executor.execute(batches, context, workers=2, max_inflight=1)
        done = executor.execute(
            batches, context_for(tmp_path), workers=2, max_inflight=2
        )
        assert [s.index for s in done] == [0, 1, 2]

    def test_batch_is_resubmitted_when_submit_finds_a_broken_pool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        FakePool.instances = 0
        monkeypatch.setattr(executor, "ProcessPoolExecutor", FakePool)
        monkeypatch.setattr(executor, "run_batch", fake_run)
        batches = [Batch(index=i) for i in range(3)]
        done = executor.execute(
            batches, context_for(tmp_path), workers=2, max_inflight=1
        )
        assert [s.index for s in done] == [0, 1, 2]


class TestTokenizeAndResume:
    def test_parallel_run_equals_serial_run(
        self, subset: Path, serial: TokenCache, tmp_path: Path
    ) -> None:
        parallel = Tokenizer().run(corpus_of(subset), tmp_path, batch_size=2, workers=2)
        assert_same_tables(serial, parallel)
        assert serial.summary["discovered"] == 6
        assert serial.summary["ok"] == 6

    def test_interrupted_run_resumes_without_duplicates(
        self, subset: Path, serial: TokenCache, tmp_path: Path
    ) -> None:
        tokenizer = Tokenizer()
        corpus = corpus_of(subset)
        with pytest.raises(EnvTokError, match="deliberate"):
            tokenizer.tokenize(
                corpus, tmp_path, batch_size=2, faults=FaultPlan(raise_after=1)
            )
        cache_dir = next((tmp_path / "cache").iterdir())
        assert len(list((cache_dir / "partitions").glob("part-*"))) == 1
        assert not (cache_dir / PLAN_FILE).exists()
        with pytest.raises(IncompleteCacheError, match="has not finished"):
            tokenizer.index(cache_dir, tmp_path)
        resumed = tokenizer.run(corpus, tmp_path, batch_size=2)
        assert_same_tables(serial, resumed)
        ids = resumed.read("structures", ["record_id"])["record_id"].to_pylist()
        assert len(ids) == len(set(ids)) == 6

    def test_worker_crash_is_retried_in_a_new_pool(
        self, subset: Path, serial: TokenCache, tmp_path: Path
    ) -> None:
        faults = FaultPlan(crash_batches=frozenset({1}), marker_dir=str(tmp_path / "m"))
        tokenizer = Tokenizer()
        cache_dir = tokenizer.tokenize(
            corpus_of(subset), tmp_path, batch_size=2, workers=2, faults=faults
        )
        assert (tmp_path / "m" / "crashed-1").exists()
        assert_same_tables(serial, tokenizer.index(cache_dir, tmp_path))

    def test_crash_budget_exhaustion_keeps_committed_partitions(
        self, subset: Path, serial: TokenCache, tmp_path: Path
    ) -> None:
        faults = FaultPlan(crash_batches=frozenset({0}), marker_dir=str(tmp_path / "m"))
        tokenizer = Tokenizer(
            execution=ExecutionSettings(workers=2, batch_size=2, max_batch_retries=0)
        )
        with pytest.raises(EnvTokError, match="crashed 1 times"):
            tokenizer.tokenize(corpus_of(subset), tmp_path, faults=faults)
        assert_same_tables(serial, tokenizer.run(corpus_of(subset), tmp_path))

    def test_sharded_tokenisation_equals_one_shard(
        self, subset: Path, serial: TokenCache, tmp_path: Path
    ) -> None:
        tokenizer = Tokenizer()
        corpus = corpus_of(subset)
        cache_dir = tokenizer.tokenize(
            corpus, tmp_path, batch_size=2, shard_index=0, num_shards=2
        )
        with pytest.raises(IncompleteCacheError, match="1 of 3 batches"):
            tokenizer.index(cache_dir, tmp_path)
        tokenizer.tokenize(corpus, tmp_path, batch_size=2, shard_index=1, num_shards=2)
        assert_same_tables(serial, tokenizer.index(cache_dir, tmp_path))
        with pytest.raises(ConfigError, match="out of range"):
            tokenizer.tokenize(corpus, tmp_path, shard_index=2, num_shards=2)

    def test_disabled_resume_refuses_an_existing_cache(
        self, subset: Path, tmp_path: Path
    ) -> None:
        tokenizer = Tokenizer()
        corpus = corpus_of(subset)
        tokenizer.tokenize(corpus, tmp_path, batch_size=6, resume=False)
        with pytest.raises(ConfigError, match="resume is disabled"):
            tokenizer.tokenize(corpus, tmp_path, batch_size=6, resume=False)
        assert tokenizer.run(corpus, tmp_path, batch_size=6).summary["ok"] == 6

    def test_stale_and_tampered_caches_are_rejected(
        self, subset: Path, tmp_path: Path
    ) -> None:
        tokenizer = Tokenizer()
        corpus = corpus_of(subset)
        cache_dir = tokenizer.tokenize(corpus, tmp_path, batch_size=3)
        leftover = cache_dir / "partitions" / ".part-000007.tmp-dead"
        leftover.mkdir()
        tokenizer.tokenize(corpus, tmp_path, batch_size=3)
        assert not leftover.exists()

        marker = cache_dir / "partitions" / "part-000000" / COMMIT_MARKER
        original = marker.read_text()
        marker.write_text(original.replace('"fingerprint": "', '"fingerprint": "x'))
        with pytest.raises(StaleCacheError, match="stale"):
            tokenizer.tokenize(corpus, tmp_path, batch_size=3)
        marker.write_text(original)

        manifest = json.loads((cache_dir / CACHE_FILE).read_text())
        manifest["keys"]["cache"] = "0" * 64
        (cache_dir / CACHE_FILE).write_text(json.dumps(manifest))
        with pytest.raises(StaleCacheError, match="different configuration"):
            tokenizer.tokenize(corpus, tmp_path, batch_size=3)

    def test_accounting_mismatch_is_reported(
        self, subset: Path, tmp_path: Path
    ) -> None:
        tokenizer = Tokenizer()
        cache_dir = tokenizer.tokenize(corpus_of(subset), tmp_path, batch_size=3)
        plan = json.loads((cache_dir / PLAN_FILE).read_text())
        plan["n_records"] += 1
        (cache_dir / PLAN_FILE).write_text(json.dumps(plan))
        with pytest.raises(IncompleteCacheError, match="7 records were discovered"):
            tokenizer.index(cache_dir, tmp_path)
        assert not list((cache_dir / "tokens").glob(".staging-*"))

    def test_duplicate_record_ids_are_rejected(self, tmp_path: Path) -> None:
        twins = frames([0, 0])
        path = write_frames(tmp_path / "twins.extxyz", twins)
        corpus = corpus_of(path, id_policy="namespaced_native_id")
        with pytest.raises(DuplicateRecordError, match="fx@v1:fx-000-nacl_prim"):
            Tokenizer().run(corpus, tmp_path)

    def test_unreadable_and_failing_records_are_quarantined(
        self, subset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from envtok.adapters import ExtxyzAdapter  # noqa: PLC0415

        read = ExtxyzAdapter.read
        from envtok.tokenizer.pipeline import tokenize_record  # noqa: PLC0415

        original = tokenize_record

        def flaky_read(self: ExtxyzAdapter, locator: Any) -> Any:
            if locator.key == "1":
                raise ValueError("corrupt frame")
            return read(self, locator)

        def flaky_tokenize(record: Any, *args: Any, **kwargs: Any) -> Any:
            if record.source_ref.key == "2":
                raise RuntimeError("geometry exploded")
            return original(record, *args, **kwargs)

        monkeypatch.setattr(executor, "_ADAPTERS", {})
        monkeypatch.setattr(ExtxyzAdapter, "read", flaky_read)
        monkeypatch.setattr(executor, "tokenize_record", flaky_tokenize)
        cache = Tokenizer().run(corpus_of(subset), tmp_path, batch_size=3)
        assert (cache.summary["ok"], cache.summary["failed"]) == (4, 2)
        structures = cache.read("structures").to_pylist()
        reasons = {r["source_key"]: r["status_reason"] for r in structures}
        assert reasons["1"] == "read_error"
        assert reasons["2"] == "tokenize_error"
        unreadable = next(r for r in structures if r["source_key"] == "1")
        assert unreadable["natoms"] == 0
        failures = {
            (f["stage"], f["category"]) for f in cache.read("failures").to_pylist()
        }
        assert {("read", "ValueError"), ("tokenize", "RuntimeError")} <= failures

    @pytest.mark.parametrize("workers", [1, 2])
    def test_compressed_sources_ship_records(
        self, subset: Path, serial: TokenCache, tmp_path: Path, workers: int
    ) -> None:
        import gzip  # noqa: PLC0415

        packed = tmp_path / "subset.xyz.gz"
        packed.write_bytes(gzip.compress(subset.read_bytes()))
        corpus = corpus_of(packed)
        assert not corpus.adapter.supports_random_access
        cache = Tokenizer().run(corpus, tmp_path, batch_size=2, workers=workers)
        assert cache.read("tokens").equals(serial.read("tokens"))
        assert cache.summary == serial.summary
        assert cache.manifest["adapter_spec"]["kind"] == "extxyz"

    def test_neighbour_vectors_are_optional(self, subset: Path, tmp_path: Path) -> None:
        cache = Tokenizer(cache_raw_measurements=False).run(
            corpus_of(subset), tmp_path, batch_size=6
        )
        partition = cache.partitions_dir / "part-000000"
        assert not (partition / "neighbours.parquet").exists()
        assert (partition / "sites.parquet").exists()
        assert cache.manifest["keep_neighbours"] is False


def test_unrelated_warnings_are_not_recorded(
    subset: Path, serial: TokenCache, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3  # noqa: PLC0415

    from envtok.tokenizer import geometry  # noqa: PLC0415

    original = geometry.symmetry_representatives

    def noisy(*args: Any, **kwargs: Any) -> Any:
        sqlite3.connect(":memory:")  # finalised unclosed: ResourceWarning
        return original(*args, **kwargs)

    monkeypatch.setattr(geometry, "symmetry_representatives", noisy)
    cache = Tokenizer().run(corpus_of(subset), tmp_path, batch_size=2)
    assert cache.read("structures")["warnings_json"].equals(
        serial.read("structures")["warnings_json"]
    )


class TestIndex:
    @pytest.mark.parametrize(
        "policy",
        [MagneticPolicy.SOURCE_ABS_ROUND_ZERO_IF_MISSING, MagneticPolicy.DISABLED],
    )
    def test_vectorised_keys_match_scalar_tokens(
        self, full_cache: TokenCache, policy: MagneticPolicy
    ) -> None:
        profile = dataclasses.replace(PROFILE, magnetic_policy=policy)
        for partition in sorted(full_cache.partitions_dir.glob("part-*")):
            sites = pq.read_table(partition / "sites.parquet")
            keys = site_keys(sites, profile).to_pylist()
            expected = [
                site_token(SiteRow(**row), profile).key() for row in sites.to_pylist()
            ]
            assert keys == expected

    def test_cache_api(self, full_cache: TokenCache) -> None:
        reopened = TokenCache.open(full_cache.token_dir)
        assert reopened.fingerprint == full_cache.fingerprint
        assert reopened.summary == full_cache.summary
        assert reopened.summary["discovered"] == 20
        tokens = reopened.read("tokens", ["token_id", "token_key"])
        assert tokens.column_names == ["token_id", "token_key"]
        assert tokens["token_id"].to_pylist() == list(range(tokens.num_rows))
        keys = tokens["token_key"].to_pylist()
        assert keys == sorted(keys)
        with pytest.raises(KeyError):
            reopened.path("nope")
        assert list(Path("/").glob(reopened.neighbours_glob().lstrip("/")))
        typed = reopened.read("tokens").to_pylist()
        po = next(t for t in typed if t["token_key"].startswith("Po_"))
        assert po["neighbours"] == [{"element": "Po", "count": 6}]
        assert po["coordination_number"] == 6
        rf = next(t for t in typed if t["token_key"].startswith("Rf_"))
        assert (rf["dmin_bin"], rf["asym_bin"]) == (None, None)
        counts = reopened.read("structure_tokens")
        assert pa.compute.sum(counts["multiplicity"]).as_py() == 76
        with pytest.raises(IncompleteCacheError, match="no complete token index"):
            TokenCache.open(full_cache.cache_dir)

    def test_profiles_must_match(self, full_cache: TokenCache) -> None:
        rebinned = dataclasses.replace(
            PROFILE,
            bins=dataclasses.replace(PROFILE.bins, asymmetry_edges=(0.5, 3.0)),
        )
        with pytest.raises(StaleCacheError, match="does not match"):
            TokenCache.open(full_cache.token_dir, rebinned)
        geometry = dataclasses.replace(
            PROFILE, geometry=dataclasses.replace(PROFILE.geometry, use_symmetry=False)
        )
        with pytest.raises(StaleCacheError, match="geometry settings"):
            Tokenizer(geometry).index(full_cache.cache_dir, full_cache.cache_dir)

    def test_rebinning_reuses_partitions(
        self, full_cache: TokenCache, tmp_path: Path
    ) -> None:
        tokenizer = Tokenizer()
        assert tokenizer.index(full_cache.cache_dir, tmp_path).token_dir == (
            full_cache.token_dir
        )
        rebinned = dataclasses.replace(
            PROFILE,
            bins=dataclasses.replace(PROFILE.bins, asymmetry_edges=(0.5, 3.0)),
        )
        other = Tokenizer(rebinned).index(full_cache.cache_dir, tmp_path)
        assert other.token_dir != full_cache.token_dir
        assert other.profile is rebinned
        assert other.summary["sites"] == full_cache.summary["sites"]
        assert (other.token_dir / INDEX_FILE).exists()

    def test_moment_policy_never_shares_partitions(
        self, full_cache: TokenCache, tmp_path: Path
    ) -> None:
        # Disabled moments are stored as zeros, relax validation and change the
        # symmetry orbits, so neither direction may reuse the other's cache.
        geometry_only = dataclasses.replace(
            PROFILE,
            name="reference-723696d+geometry-only",
            magnetic_policy=MagneticPolicy.DISABLED,
        )
        with pytest.raises(StaleCacheError, match="moment policy"):
            Tokenizer(geometry_only).index(full_cache.cache_dir, tmp_path)
        iron = bulk("Fe", "bcc", a=2.87, cubic=True)
        iron.set_initial_magnetic_moments([2.2, 2.2])
        corpus = Corpus.from_atoms([iron], name="fe", split="train")
        plain = Tokenizer(geometry_only).run(corpus, tmp_path)
        with pytest.raises(StaleCacheError, match="moment policy"):
            Tokenizer(PROFILE).index(plain.cache_dir, tmp_path)
        magnetic = Tokenizer(PROFILE).run(corpus, tmp_path)
        assert magnetic.cache_dir != plain.cache_dir
        keys = [magnetic.read("tokens")["token_key"], plain.read("tokens")["token_key"]]
        assert [[k[:5] for k in column.to_pylist()] for column in keys] == [
            ["Fe_2_"],
            ["Fe_NA"],
        ]

    def test_adapter_verification(self, subset: Path, tmp_path: Path) -> None:
        loaded = ase.io.read(subset, index=":")
        assert isinstance(loaded, list)
        copy = write_frames(tmp_path / "copy.extxyz", loaded)
        cache = Tokenizer().run(corpus_of(copy), tmp_path, batch_size=6)
        assert cache.adapter().kind == "extxyz"
        stat = copy.stat()
        os.utime(copy, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        with pytest.raises(StaleCacheError, match="source changed"):
            cache.adapter()
        assert cache.adapter(verify=False).kind == "extxyz"

    def test_in_memory_sources_cannot_be_rebuilt(self, tmp_path: Path) -> None:
        corpus = Corpus.from_atoms(frames([0, 7]), name="mem", split="train")
        cache = Tokenizer().run(corpus, tmp_path)
        assert cache.summary["ok"] == 2
        assert cache.manifest["adapter_spec"] is None
        with pytest.raises(RetrievalUnavailableError, match="in-memory"):
            cache.adapter()

    def test_directory_source_gaining_a_file_is_read_in_process(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "source"
        write_frames(source / "a.extxyz", frames([0]))
        Tokenizer().run(corpus_of(source), tmp_path / "first")
        write_frames(source / "b.extxyz", frames([7]))
        cache = Tokenizer().run(corpus_of(source), tmp_path / "second")
        assert cache.summary["ok"] == 2

    def test_replaced_lmdb_shard_is_reread(self, tmp_path: Path) -> None:
        from envtok.adapters.asedb import close_shared_databases  # noqa: PLC0415

        shard = tmp_path / "omat" / "data.aselmdb"
        shard.parent.mkdir()

        def fill(atoms: list[Atoms]) -> None:
            for leftover in shard.parent.glob("data.aselmdb*"):
                leftover.unlink()
            database = ase.db.connect(str(shard))
            for frame in atoms:
                database.write(frame, data={"sid": frame.info["sid"]})
            database.close()

        try:
            fill(frames([0]))
            first = Tokenizer().run(
                Corpus.from_omat24(shard.parent, revision="v1"), tmp_path / "a"
            )
            assert first.read("structures")["formula"].to_pylist() == ["ClNa"]
            fill(frames([7]))
            second = Tokenizer().run(
                Corpus.from_omat24(shard.parent, revision="v1"), tmp_path / "b"
            )
            assert second.read("structures")["formula"].to_pylist() == ["SZn"]
        finally:
            close_shared_databases()


class TestCorpus:
    def test_from_atoms_variants(self) -> None:
        selected = frames([0, 7])
        listed = Corpus.from_atoms(selected, name="mem", split="train")
        assert listed.name == "mem"
        revision = listed.adapter.options.revision
        assert len(revision) == 16
        assert revision != "unversioned"
        assert listed.adapter.options.split is Split.TRAIN
        factory = Corpus.from_atoms(lambda: iter(selected), name="f")
        assert factory.adapter.options.revision == "unversioned"
        assert factory.adapter.supports_retrieval
        once = Corpus.from_atoms(iter(selected), name="o", revision="r1")
        assert once.adapter.options.revision == "r1"
        assert not once.adapter.supports_retrieval

    def test_spooling_makes_iterables_reloadable(self, tmp_path: Path) -> None:
        selected = frames([0, 7])
        spooled = Corpus.from_atoms(
            lambda: iter(selected), name="sp", spool_to=tmp_path / "s.extxyz"
        )
        assert spooled.adapter.kind == "extxyz"
        assert len(spooled.adapter.options.revision) == 16
        pinned = Corpus.from_atoms(
            iter(selected), name="sp", revision="r2", spool_to=tmp_path / "t.extxyz"
        )
        assert pinned.adapter.options.revision == "r2"
        cache = Tokenizer().run(pinned, tmp_path / "work")
        assert cache.adapter().kind == "extxyz"
        fingerprint = cache.manifest["source_fingerprint"]["files"][0]
        assert "sha256" in fingerprint

    def test_file_corpora(self, tmp_path: Path) -> None:
        extxyz = corpus_of(CORPUS, compressed="sequential", checksum="full")
        assert extxyz.adapter.options.split is Split.TRAIN
        (tmp_path / "data.db").touch()
        database = Corpus.from_ase_db(tmp_path / "data.db", dataset="d", revision="r")
        assert database.adapter.kind == "asedb"
        assert database.adapter.options.split is Split.UNSPLIT
        (tmp_path / "x.aselmdb").touch()
        omat = Corpus.from_omat24(
            tmp_path, revision="r", policy=corpus_policy("omat24-study")
        )
        assert omat.name == "omat24"
        assert omat.policy.preset == "omat24-study"

    def test_from_source(self, tmp_path: Path) -> None:
        config = {
            "schema_version": 1,
            "source": {
                "adapter": "extxyz",
                "path": str(CORPUS),
                "revision": "v1",
                "name": "fx",
            },
        }
        from_mapping = Corpus.from_source(config)
        assert from_mapping.name == "fx"
        path = tmp_path / "config.yaml"
        path.write_text(json.dumps(config))
        assert Corpus.from_source(path).adapter.spec() == from_mapping.adapter.spec()


def test_full_cache_records_every_site(full_cache: TokenCache) -> None:
    sites = full_cache.read("site_tokens")
    assert sites.num_rows == full_cache.summary["sites"] == 76
    token_ids = sites["token_id"].to_numpy()
    assert np.all(token_ids >= 0)
    neighbours = pq.read_table(
        full_cache.partitions_dir / "part-000000" / "neighbours.parquet"
    )
    assert neighbours.num_rows > 0
