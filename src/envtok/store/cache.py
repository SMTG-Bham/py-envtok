"""The token cache: content-addressed partitions and the global token index.

Layout below a run's work directory::

    cache/<cache-key>/
        cache.json                  fingerprints and the adapter description
        plan.json                   discovered record and batch counts
        partitions/part-NNNNNN/     one committed partition per batch
            structures.parquet sites.parquet neighbours.parquet failures.parquet
        tokens/<token-key>/         global index for one token profile
            index.json
            structures.parquet tokens.parquet structure_tokens.parquet
            site_tokens.parquet failures.parquet

``cache-key`` hashes the source, geometry and corpus fingerprints (plus batch
size and neighbour retention), so a changed source or setting can never resume
from an unrelated checkpoint. ``token-key`` hashes the token fingerprint, so
re-binning reuses the partitions without repeating geometry.

Token IDs are ranks of the sorted canonical token keys over the whole cache:
they do not depend on partitioning, worker count or completion order.
"""

import json
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from envtok.adapters import Adapter, adapter_from_spec
from envtok.errors import (
    ConfigError,
    DuplicateRecordError,
    IncompleteCacheError,
    RetrievalUnavailableError,
    StaleCacheError,
)
from envtok.execution import ExecutionSettings
from envtok.fingerprint import (
    corpus_payload,
    digest,
    geometry_payload,
    json_default,
    package_versions,
    token_payload,
)
from envtok.profiles import CorpusPolicy, Profile, get_profile
from envtok.profiles.reference import UNDEFINED_SENTINEL
from envtok.store.duck import sql_literal
from envtok.store.executor import (
    FaultPlan,
    PlanStats,
    TaskContext,
    execute,
    plan_batches,
)
from envtok.store.partitions import committed_partitions, remove_temporary
from envtok.store.schema import SCHEMA_VERSION, TOKENS
from envtok.tokenizer.bins import csm_bins, digitize_array, moment_bins
from envtok.tokenizer.tokens import SiteToken

CACHE_FILE = "cache.json"
PLAN_FILE = "plan.json"
INDEX_FILE = "index.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    text = json.dumps(data, indent=1, sort_keys=True, default=json_default)
    partial.write_text(text + "\n")
    partial.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data


@dataclass(frozen=True, slots=True)
class CacheKeys:
    """The layered fingerprints of a cache."""

    source: str
    geometry: str
    corpus: str
    token: str
    cache: str


def cache_keys(
    adapter: Adapter,
    profile: Profile,
    policy: CorpusPolicy,
    *,
    batch_size: int,
    keep_neighbours: bool,
) -> CacheKeys:
    """Compute the fingerprints that address a cache directory.

    Args:
        adapter: Source adapter.
        profile: Scientific profile.
        policy: Corpus policy.
        batch_size: Records per partition.
        keep_neighbours: Whether neighbour vectors are retained.

    Returns:
        The fingerprints.
    """
    source = digest(adapter.source_fingerprint())
    geometry = digest(geometry_payload(profile))
    corpus = digest(corpus_payload(policy))
    token = digest(token_payload(profile))
    cache = digest(
        {
            "schema": SCHEMA_VERSION,
            "source": source,
            "geometry": geometry,
            "corpus": corpus,
            "batch_size": batch_size,
            "keep_neighbours": keep_neighbours,
        }
    )
    return CacheKeys(source, geometry, corpus, token, cache)


def site_keys(sites: pa.Table, profile: Profile) -> pa.Array:
    """Derive the canonical token key of every site from raw measurements.

    This vectorised path must agree exactly with
    :func:`envtok.tokenizer.pipeline.site_token` (tested).

    Args:
        sites: Table with the ``sites`` schema.
        profile: Scientific profile (bins and moment policy).

    Returns:
        String array of reference token keys.
    """
    bins = profile.bins
    n = sites.num_rows

    def text(values: np.ndarray) -> pa.Array:
        return pa.array(values, type=pa.int64()).cast(pa.string())

    moments = moment_bins(
        sites["moment"].to_numpy(zero_copy_only=False).astype(np.float64),
        profile.magnetic_policy,
    )
    spin = pa.array(["NA"] * n, pa.string()) if moments is None else text(moments)
    csm_column = sites["csm"]
    defined = pc.invert(pc.is_null(csm_column)).to_numpy(zero_copy_only=False)
    csm = pc.fill_null(csm_column, np.nan).to_numpy(zero_copy_only=False)
    ratio = pc.fill_null(sites["ratio_min"], np.nan).to_numpy(zero_copy_only=False)
    asym = pc.fill_null(sites["asymmetry"], np.nan).to_numpy(zero_copy_only=False)
    return pc.binary_join_element_wise(
        sites["element"],
        spin,
        sites["neighbours"],
        sites["geometry"],
        text(csm_bins(csm.astype(np.float64), defined.astype(bool), bins)),
        text(digitize_array(ratio.astype(np.float64), bins.dmin_edges)),
        text(digitize_array(asym.astype(np.float64), bins.asymmetry_edges)),
        "_",
    )


def _typed_tokens(vocabulary: pa.Table) -> pa.Table:
    """Parse the sorted vocabulary into typed token columns."""
    columns: dict[str, list[Any]] = {name: [] for name in TOKENS.names}
    for token_id, key, n_sites, n_records in zip(
        vocabulary["token_id"].to_pylist(),
        vocabulary["token_key"].to_pylist(),
        vocabulary["n_sites"].to_pylist(),
        vocabulary["n_records"].to_pylist(),
        strict=True,
    ):
        token = SiteToken.from_key(key)
        columns["token_id"].append(token_id)
        columns["token_key"].append(key)
        columns["element"].append(token.element)
        columns["moment"].append(token.moment)
        columns["neighbours"].append(
            None
            if token.neighbours is None
            else [{"element": e, "count": c} for e, c in token.neighbours]
        )
        columns["coordination_number"].append(token.coordination_number)
        columns["geometry"].append(token.geometry)
        columns["csm_bin"].append(token.csm_bin)
        columns["dmin_bin"].append(token.dmin_bin)
        columns["asym_bin"].append(token.asym_bin)
        columns["n_sites"].append(n_sites)
        columns["n_records"].append(n_records)
    return pa.table(columns, schema=TOKENS)


class TokenCache:
    """An indexed token cache for one source, profile and corpus policy.

    Use :meth:`envtok.tokenizer.Tokenizer.run` to build one and
    :meth:`open` to reopen it.
    """

    TABLES = ("structures", "tokens", "structure_tokens", "site_tokens", "failures")

    def __init__(self, token_dir: Path, profile: Profile | None = None) -> None:
        """Open an index directory.

        Args:
            token_dir: ``cache/<key>/tokens/<token-key>`` directory.
            profile: The profile the index was built with. Registered profiles
                are resolved by name when omitted.

        Raises:
            IncompleteCacheError: If the directory holds no complete index.
            StaleCacheError: If ``profile`` does not match the index.
        """
        if not (token_dir / INDEX_FILE).exists():
            raise IncompleteCacheError(f"{token_dir} holds no complete token index")
        self.token_dir = token_dir
        self.cache_dir = token_dir.parent.parent
        self.index = _read_json(token_dir / INDEX_FILE)
        self.manifest = _read_json(self.cache_dir / CACHE_FILE)
        resolved = profile or get_profile(self.index["profile"])
        if digest(token_payload(resolved)) != self.index["token_fingerprint"]:
            raise StaleCacheError(
                f"profile {resolved.name!r} does not match the token index "
                f"in {token_dir}"
            )
        self.profile = resolved

    @classmethod
    def open(cls, token_dir: str | Path, profile: Profile | None = None) -> TokenCache:
        """Reopen an indexed cache (see :meth:`__init__`)."""
        return cls(Path(token_dir), profile)

    @property
    def partitions_dir(self) -> Path:
        """Directory of the committed partitions."""
        return self.cache_dir / "partitions"

    def path(self, table: str) -> Path:
        """Return the Parquet file of an indexed table.

        Raises:
            KeyError: For unknown tables.
        """
        if table not in self.TABLES:
            raise KeyError(table)
        return self.token_dir / f"{table}.parquet"

    def read(self, table: str, columns: list[str] | None = None) -> pa.Table:
        """Read an indexed table."""
        return pq.read_table(self.path(table), columns=columns)

    def neighbours_glob(self) -> str:
        """Glob of the partition neighbour tables (absent if not retained)."""
        return (self.partitions_dir / "part-*" / "neighbours.parquet").as_posix()

    @property
    def summary(self) -> dict[str, Any]:
        """Counts and accounting recorded by the index stage."""
        result: dict[str, Any] = self.index["summary"]
        return result

    @property
    def fingerprint(self) -> str:
        """Fingerprint identifying this index (cache plus token profile)."""
        return digest([self.manifest["keys"]["cache"], self.index["token_fingerprint"]])

    def adapter(self, *, verify: bool = True) -> Adapter:
        """Rebuild the source adapter, checking the source is unchanged.

        Args:
            verify: Compare the current source fingerprint with the cached one.

        Returns:
            The adapter.

        Raises:
            RetrievalUnavailableError: If the source was an in-memory iterable.
            StaleCacheError: If the source changed since tokenisation.
        """
        spec = self.manifest["adapter_spec"]
        if spec is None:
            raise RetrievalUnavailableError(
                "this cache was built from an in-memory iterable; pass the original "
                "corpus to export, or spool it with Corpus.from_atoms(spool_to=...)"
            )
        adapter = adapter_from_spec(spec)
        if (
            verify
            and digest(adapter.source_fingerprint()) != self.manifest["keys"]["source"]
        ):
            raise StaleCacheError(
                "the source changed after tokenisation; exported structures would "
                "not match the cached tokens. Re-run tokenize on the current source"
            )
        return adapter


def _cache_dir(workdir: Path, keys: CacheKeys) -> Path:
    return workdir / "cache" / keys.cache[:16]


def tokenize(
    adapter: Adapter,
    profile: Profile,
    policy: CorpusPolicy,
    workdir: Path,
    *,
    execution: ExecutionSettings,
    keep_neighbours: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
    faults: FaultPlan | None = None,
) -> Path:
    """Tokenise every uncommitted batch of this shard; resume what is committed.

    Args:
        adapter: Source adapter.
        profile: Scientific profile.
        policy: Corpus policy.
        workdir: Run work directory.
        execution: Parallelism settings.
        keep_neighbours: Retain image-resolved neighbour vectors.
        shard_index: Shard processed by this invocation.
        num_shards: Number of independent shards (for example array tasks).
        faults: Test-only fault injection.

    Returns:
        The cache directory.

    Raises:
        ConfigError: For an invalid shard selection.
        StaleCacheError: If the cache directory holds another configuration.
    """
    if not 0 <= shard_index < num_shards:
        raise ConfigError(
            f"shard index {shard_index} out of range for {num_shards} shards"
        )
    keys = cache_keys(
        adapter,
        profile,
        policy,
        batch_size=execution.batch_size,
        keep_neighbours=keep_neighbours,
    )
    cache_dir = _cache_dir(workdir, keys)
    partitions = cache_dir / "partitions"
    partitions.mkdir(parents=True, exist_ok=True)
    portable = _portable_spec(adapter)
    spec = portable if adapter.supports_random_access else None
    if not (cache_dir / CACHE_FILE).exists():
        _write_json(
            cache_dir / CACHE_FILE,
            {
                "schema": SCHEMA_VERSION,
                "keys": _keys_dict(keys),
                "adapter_kind": adapter.kind,
                "adapter_spec": portable,
                "source_fingerprint": adapter.source_fingerprint(),
                "profile": profile.name,
                "geometry": geometry_payload(profile),
                "corpus": corpus_payload(policy),
                "corpus_preset": policy.preset,
                "batch_size": execution.batch_size,
                "keep_neighbours": keep_neighbours,
                "created": _now(),
            },
        )
    elif _read_json(cache_dir / CACHE_FILE)["keys"]["cache"] != keys.cache:
        raise StaleCacheError(f"{cache_dir} belongs to a different configuration")
    remove_temporary(partitions)
    committed = committed_partitions(partitions, keys.cache)
    stats = PlanStats()

    def selected(index: int) -> bool:
        return index % num_shards == shard_index and index not in committed

    context = TaskContext(
        adapter_spec=spec,
        profile=profile,
        policy=policy,
        partitions_dir=str(partitions),
        fingerprint=keys.cache,
        keep_neighbours=keep_neighbours,
        faults=faults or FaultPlan(),
    )
    execute(
        plan_batches(adapter, execution.batch_size, selected, stats),
        context,
        workers=execution.workers,
        max_inflight=execution.inflight,
        max_retries=execution.max_batch_retries,
        threads_per_worker=execution.threads_per_worker,
    )
    _write_json(
        cache_dir / PLAN_FILE,
        {
            "n_batches": stats.n_batches,
            "n_records": stats.n_records,
            "batch_size": execution.batch_size,
        },
    )
    return cache_dir


def _keys_dict(keys: CacheKeys) -> dict[str, str]:
    return {
        "source": keys.source,
        "geometry": keys.geometry,
        "corpus": keys.corpus,
        "token": keys.token,
        "cache": keys.cache,
    }


def _portable_spec(adapter: Adapter) -> dict[str, Any] | None:
    try:
        return adapter.spec()
    except Exception:  # noqa: BLE001 - in-memory sources have no portable spec
        return None


def build_index(
    cache_dir: Path, profile: Profile, *, execution: ExecutionSettings, workdir: Path
) -> TokenCache:
    """Merge committed partitions into the global token index for ``profile``.

    Args:
        cache_dir: Cache directory returned by :func:`tokenize`.
        profile: Scientific profile (its token settings select the index).
        execution: Resource settings for DuckDB.
        workdir: Run work directory (DuckDB spill location).

    Returns:
        The indexed cache.

    Raises:
        IncompleteCacheError: If any planned batch is not committed.
        DuplicateRecordError: If two records share an ID.
        StaleCacheError: If ``profile`` has different geometry settings.
    """
    manifest = _read_json(cache_dir / CACHE_FILE)
    if manifest["keys"]["geometry"] != digest(geometry_payload(profile)):
        raise StaleCacheError("the profile's geometry settings differ from this cache")
    if not (cache_dir / PLAN_FILE).exists():
        raise IncompleteCacheError(f"tokenisation of {cache_dir} has not finished")
    plan = _read_json(cache_dir / PLAN_FILE)
    committed = committed_partitions(
        cache_dir / "partitions", manifest["keys"]["cache"]
    )
    missing = sorted(set(range(plan["n_batches"])) - set(committed))
    if missing:
        raise IncompleteCacheError(
            f"{len(missing)} of {plan['n_batches']} batches are not committed "
            f"(first: {missing[:5]}); run tokenize for every shard first"
        )
    token_fingerprint = digest(token_payload(profile))
    token_dir = cache_dir / "tokens" / token_fingerprint[:16]
    if (token_dir / INDEX_FILE).exists():
        return TokenCache(token_dir, profile)
    staging = cache_dir / "tokens" / f".staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        summary = _merge(
            [committed[i] for i in sorted(committed)],
            staging,
            profile,
            connection_factory=lambda: execution.duckdb(workdir),
        )
        if plan["n_records"] != summary["discovered"]:
            raise IncompleteCacheError(
                f"{plan['n_records']} records were discovered but "
                f"{summary['discovered']} are in the committed partitions"
            )
        _write_json(
            staging / INDEX_FILE,
            {
                "schema": SCHEMA_VERSION,
                "profile": profile.name,
                "token_fingerprint": token_fingerprint,
                "token_payload": json.loads(
                    json.dumps(token_payload(profile), default=str)
                ),
                "versions": package_versions(),
                "n_partitions": len(committed),
                "summary": summary,
                "created": _now(),
            },
        )
        token_dir.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(token_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return TokenCache(token_dir, profile)


def _copy(connection: Any, query: str, path: Path) -> None:
    connection.execute(
        f"COPY ({query}) TO {sql_literal(path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def _parquet_list(paths: list[Path], name: str) -> str:
    return "[" + ", ".join(sql_literal(p / f"{name}.parquet") for p in paths) + "]"


def _merge(
    partitions: list[Path],
    out: Path,
    profile: Profile,
    *,
    connection_factory: Callable[[], Any],
) -> dict[str, Any]:
    keys_dir = out / "_keys"
    keys_dir.mkdir()
    for path in partitions:
        sites = pq.read_table(path / "sites.parquet")
        keyed = pa.table(
            {
                "record_id": sites["record_id"],
                "atom_index": sites["atom_index"],
                "token_key": site_keys(sites, profile),
            }
        )
        pq.write_table(keyed, keys_dir / f"{path.name}.parquet")
    connection = connection_factory()
    try:
        connection.execute(
            "CREATE VIEW structures_raw AS SELECT * FROM "
            f"read_parquet({_parquet_list(partitions, 'structures')})"
        )
        duplicates = connection.execute(
            "SELECT record_id, count(*) FROM structures_raw GROUP BY record_id "
            "HAVING count(*) > 1 ORDER BY record_id LIMIT 5"
        ).fetchall()
        if duplicates:
            raise DuplicateRecordError(
                f"duplicate record IDs (first {len(duplicates)}): "
                + ", ".join(f"{rid!r} x{n}" for rid, n in duplicates)
                + "; use id_policy namespaced_source_record or a unique native key"
            )
        key_glob = sql_literal((keys_dir / "*.parquet").as_posix())
        connection.execute(
            f"CREATE VIEW keys AS SELECT * FROM read_parquet({key_glob})"
        )
        connection.execute(
            "CREATE TABLE vocab AS SELECT token_key, "
            "CAST(row_number() OVER (ORDER BY token_key) - 1 AS BIGINT) AS token_id, "
            "n_sites, n_records FROM (SELECT token_key, count(*) AS n_sites, "
            "count(DISTINCT record_id) AS n_records FROM keys GROUP BY token_key)"
        )
        _copy(
            connection,
            "SELECT * FROM structures_raw ORDER BY record_id",
            out / "structures.parquet",
        )
        _copy(
            connection,
            "SELECT k.record_id, v.token_id, "
            "CAST(count(*) AS BIGINT) AS multiplicity "
            "FROM keys k JOIN vocab v USING (token_key) "
            "GROUP BY k.record_id, v.token_id ORDER BY k.record_id, v.token_id",
            out / "structure_tokens.parquet",
        )
        _copy(
            connection,
            "SELECT s.*, v.token_id "
            f"FROM read_parquet({_parquet_list(partitions, 'sites')}) s "
            "JOIN keys k USING (record_id, atom_index) "
            "JOIN vocab v USING (token_key) ORDER BY s.record_id, s.atom_index",
            out / "site_tokens.parquet",
        )
        _copy(
            connection,
            f"SELECT * FROM read_parquet({_parquet_list(partitions, 'failures')}) "
            "ORDER BY record_id, stage, category, atom_index NULLS FIRST, message",
            out / "failures.parquet",
        )
        vocabulary = connection.sql(
            "SELECT token_id, token_key, n_sites, n_records FROM vocab "
            "ORDER BY token_id"
        ).to_arrow_table()
        counts = dict(
            connection.execute(
                "SELECT status, count(*) FROM structures_raw GROUP BY status"
            ).fetchall()
        )
        (ok_atoms,) = connection.execute(
            "SELECT coalesce(sum(natoms), 0) FROM structures_raw WHERE status = 'ok'"
        ).fetchall()[0]
        (n_sites,) = connection.execute("SELECT count(*) FROM keys").fetchall()[0]
    finally:
        connection.close()
    pq.write_table(
        _typed_tokens(vocabulary), out / "tokens.parquet", compression="zstd"
    )
    shutil.rmtree(keys_dir)
    return {
        "discovered": int(sum(counts.values())),
        "ok": int(counts.get("ok", 0)),
        "excluded": int(counts.get("excluded", 0)),
        "failed": int(counts.get("failed", 0)),
        "ok_atoms": int(ok_atoms),
        "sites": int(n_sites),
        "tokens": vocabulary.num_rows,
        "undefined_sentinel": UNDEFINED_SENTINEL,
    }
