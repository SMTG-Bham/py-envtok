"""Exact coverage classes: coarsen full tokens by distinct-record support.

The unresolved vocabulary starts as the full tokens present in the frozen
eligible pool (records of the fit split that are eligible under the invalid
record policy). For each rung, every unresolved token is projected to the
rung's signature, and

    support(signature) = number of distinct eligible records containing at
                         least one unresolved token with that signature.

Every unresolved token whose signature reaches ``T`` is assigned to the class
``(rung, signature)`` and leaves the pool; already-resolved tokens never
support a later rung. Tokens left after the last rung join a per-element
terminal class (signature ``__CATCHALL__|<element>``, rung -1) whatever its
support. Every pool token ends in exactly one class. Atoms are never counted:
two equivalent sites in one record contribute one supporting record.

Two backends implement the same contract: ``memory`` (NumPy over the compact
incidence) and ``duckdb`` (external aggregation over the Parquet incidence).
Records outside the fit split, and failed, excluded or quarantined records,
never influence the classes.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from envtok._arrays import sorted_unique
from envtok.errors import ConfigError
from envtok.execution import ExecutionSettings
from envtok.fingerprint import digest
from envtok.profiles import (
    InvalidRecordPolicy,
    PresenceFamily,
    Profile,
    Rung,
    get_profile,
)
from envtok.records import JSONScalar, Split
from envtok.store.cache import TokenCache
from envtok.tokenizer.tokens import (
    CATCHALL_PREFIX,
    SiteToken,
    signature,
    signature_fields,
)

TERMINAL_RUNG = -1
BACKENDS = ("auto", "memory", "duckdb")
AUTO_MEMORY_EDGE_LIMIT = 50_000_000


@dataclass(frozen=True, slots=True)
class Pool:
    """The frozen eligible pool in canonical (sorted record ID) order.

    Attributes:
        record_ids: Eligible record IDs, sorted.
        natoms: Original atom count of each record.
        native_ids: Native IDs (``None`` where absent).
        edge_record: Record index of every (record, token) incidence edge.
        edge_token: Token ID of every edge.
        edge_multiplicity: Sites of the token in the record.
        token_ids: Distinct pool token IDs, sorted.
        token_keys: Canonical keys aligned with ``token_ids``.
    """

    record_ids: npt.NDArray[np.str_]
    natoms: npt.NDArray[np.int64]
    native_ids: list[str | None]
    edge_record: npt.NDArray[np.int64]
    edge_token: npt.NDArray[np.int64]
    edge_multiplicity: npt.NDArray[np.int64]
    token_ids: npt.NDArray[np.int64]
    token_keys: list[str]


def eligible_mask(
    structures: pa.Table,
    split: Split,
    policy: InvalidRecordPolicy,
    where: Mapping[str, JSONScalar] | None = None,
) -> npt.NDArray[np.bool_]:
    """Return which structure rows belong to the eligible pool.

    Args:
        structures: The indexed ``structures`` table.
        split: Fit split.
        policy: Invalid record policy.
        where: Optional metadata equality filter (protected strata).

    Returns:
        Boolean mask aligned with the table rows.
    """
    status = structures["status"].to_numpy(zero_copy_only=False)
    splits = structures["split"].to_numpy(zero_copy_only=False)
    mask = (status == "ok") & (splits == split.value)
    if policy is InvalidRecordPolicy.QUARANTINE:
        invalid = structures["n_invalid_sites"].to_numpy(zero_copy_only=False)
        mask &= invalid == 0
    if where:
        metadata = [json.loads(m) for m in structures["metadata_json"].to_pylist()]
        mask &= np.array(
            [all(m.get(k) == v for k, v in where.items()) for m in metadata], dtype=bool
        )
    return np.asarray(mask, dtype=bool)


def load_pool(
    cache: TokenCache,
    split: Split,
    policy: InvalidRecordPolicy,
    where: Mapping[str, JSONScalar] | None = None,
) -> Pool:
    """Load the eligible pool and its record-token incidence from a cache.

    Args:
        cache: Indexed token cache.
        split: Fit split.
        policy: Invalid record policy.
        where: Optional metadata equality filter.

    Returns:
        The pool.
    """
    structures = cache.read(
        "structures",
        [
            "record_id",
            "natoms",
            "native_id",
            "split",
            "status",
            "n_invalid_sites",
            "metadata_json",
        ],
    )
    mask = eligible_mask(structures, split, policy, where)
    chosen = structures.filter(pa.array(mask))
    record_ids = np.asarray(chosen["record_id"].to_pylist(), dtype=np.str_)
    position = {rid: i for i, rid in enumerate(record_ids.tolist())}
    incidence = cache.read("structure_tokens")
    keep = pc.is_in(incidence["record_id"], value_set=chosen["record_id"])
    incidence = incidence.filter(keep)
    edge_record = np.fromiter(
        (position[r] for r in incidence["record_id"].to_pylist()),
        dtype=np.int64,
        count=incidence.num_rows,
    )
    edge_token = incidence["token_id"].to_numpy().astype(np.int64)
    token_ids = sorted_unique(edge_token)
    tokens = cache.read("tokens", ["token_id", "token_key"])
    keys = dict(
        zip(
            tokens["token_id"].to_pylist(), tokens["token_key"].to_pylist(), strict=True
        )
    )
    return Pool(
        record_ids=record_ids,
        natoms=chosen["natoms"].to_numpy().astype(np.int64),
        native_ids=chosen["native_id"].to_pylist(),
        edge_record=edge_record,
        edge_token=edge_token,
        edge_multiplicity=incidence["multiplicity"].to_numpy().astype(np.int64),
        token_ids=token_ids.astype(np.int64),
        token_keys=[keys[int(t)] for t in token_ids],
    )


@dataclass(frozen=True, slots=True)
class Assignment:
    """Class assignment of every pool token, aligned with ``Pool.token_ids``.

    Attributes:
        rung: Rung index, or :data:`TERMINAL_RUNG` for terminal residues.
        signature: Class signature.
        support: Distinct-record support of the class when it was formed.
    """

    rung: npt.NDArray[np.int64]
    signature: list[str]
    support: npt.NDArray[np.int64]


def _distinct_support(
    edge_record: npt.NDArray[np.int64],
    edge_group: npt.NDArray[np.int64],
    n_groups: int,
    n_records: int,
) -> npt.NDArray[np.int64]:
    """Count distinct records per group from (record, group) edges."""
    pairs = sorted_unique(edge_group * np.int64(max(n_records, 1)) + edge_record)
    return np.bincount(pairs // max(n_records, 1), minlength=n_groups).astype(np.int64)


def assign_memory(
    pool: Pool,
    fields: Sequence[Mapping[str, str]],
    ladder: Sequence[Rung],
    threshold: int,
) -> Assignment:
    """In-memory exact unresolved-pool coarsening.

    Args:
        pool: Eligible pool.
        fields: Signature fields of every pool token (aligned with
            ``pool.token_ids``).
        ladder: Ordered rungs.
        threshold: Structure-support threshold T.

    Returns:
        The assignment.
    """
    n_tokens = len(pool.token_ids)
    n_records = len(pool.record_ids)
    position = np.searchsorted(pool.token_ids, pool.edge_token)
    rung = np.full(n_tokens, TERMINAL_RUNG, dtype=np.int64)
    support = np.zeros(n_tokens, dtype=np.int64)
    signatures: list[str] = [""] * n_tokens
    unresolved = np.arange(n_tokens)
    for index, step in enumerate(ladder):
        if unresolved.size == 0:
            break
        groups: dict[str, int] = {}
        local = np.array(
            [
                groups.setdefault(signature(fields[t], step.fields), len(groups))
                for t in unresolved.tolist()
            ],
            dtype=np.int64,
        )
        group_of = np.full(n_tokens, -1, dtype=np.int64)
        group_of[unresolved] = local
        edge_group = group_of[position]
        live = edge_group >= 0
        counts = _distinct_support(
            pool.edge_record[live], edge_group[live], len(groups), n_records
        )
        accepted = counts[local] >= threshold
        texts = list(groups)
        for token, group in zip(unresolved[accepted], local[accepted], strict=True):
            rung[token] = index
            support[token] = counts[group]
            signatures[token] = texts[group]
        unresolved = unresolved[~accepted]
    if unresolved.size:
        groups = {}
        local = np.array(
            [
                groups.setdefault(CATCHALL_PREFIX + fields[t]["elem"], len(groups))
                for t in unresolved.tolist()
            ],
            dtype=np.int64,
        )
        group_of = np.full(n_tokens, -1, dtype=np.int64)
        group_of[unresolved] = local
        edge_group = group_of[position]
        live = edge_group >= 0
        counts = _distinct_support(
            pool.edge_record[live], edge_group[live], len(groups), n_records
        )
        texts = list(groups)
        for token, group in zip(unresolved, local, strict=True):
            support[token] = counts[group]
            signatures[token] = texts[group]
    return Assignment(rung=rung, signature=signatures, support=support)


def assign_duckdb(
    pool: Pool,
    fields: Sequence[Mapping[str, str]],
    ladder: Sequence[Rung],
    threshold: int,
    connection_factory: Callable[[], Any],
) -> Assignment:
    """DuckDB exact unresolved-pool coarsening (external aggregation).

    The record-token incidence is aggregated by DuckDB, which can spill to
    disk; only the vocabulary-level signature fields live in Python.

    Args:
        pool: Eligible pool.
        fields: Signature fields of every pool token.
        ladder: Ordered rungs.
        threshold: Structure-support threshold T.
        connection_factory: Returns a configured DuckDB connection.

    Returns:
        The assignment, identical to :func:`assign_memory`.
    """
    names = sorted({name for step in ladder for name in step.fields} | {"elem"})
    vocabulary = pa.table(
        {"token": pa.array(np.arange(len(pool.token_ids)), pa.int64())}
        | {name: pa.array([f[name] for f in fields], pa.string()) for name in names}
    )
    position = np.searchsorted(pool.token_ids, pool.edge_token)
    incidence = pa.table(
        {
            "record": pa.array(pool.edge_record, pa.int64()),
            "token": pa.array(position.astype(np.int64), pa.int64()),
        }
    )
    connection = connection_factory()
    try:
        connection.register("vocabulary", vocabulary)
        connection.register("incidence", incidence)
        connection.execute("CREATE TEMP TABLE unresolved AS SELECT * FROM vocabulary")
        connection.execute(
            "CREATE TEMP TABLE resolved (token BIGINT, rung BIGINT, sig VARCHAR, "
            "support BIGINT)"
        )
        for index, step in enumerate(ladder):
            expression = " || '|' || ".join(f'"{name}"' for name in step.fields)
            connection.execute(
                "CREATE OR REPLACE TEMP TABLE step AS "
                f"SELECT token, {expression} AS sig FROM unresolved"
            )
            connection.execute(
                "CREATE OR REPLACE TEMP TABLE supported AS SELECT s.sig, "
                "count(DISTINCT i.record) AS support FROM incidence i "
                "JOIN step s USING (token) GROUP BY s.sig"
            )
            connection.execute(
                f"INSERT INTO resolved SELECT s.token, {index}, s.sig, p.support "
                "FROM step s JOIN supported p USING (sig) "
                f"WHERE p.support >= {int(threshold)}"
            )
            connection.execute(
                "DELETE FROM unresolved WHERE token IN "
                f"(SELECT token FROM resolved WHERE rung = {index})"
            )
        connection.execute(
            "CREATE OR REPLACE TEMP TABLE step AS SELECT token, "
            f"'{CATCHALL_PREFIX}' || elem AS sig FROM unresolved"
        )
        connection.execute(
            f"INSERT INTO resolved SELECT s.token, {TERMINAL_RUNG}, s.sig, "
            "coalesce(p.support, 0) FROM step s LEFT JOIN (SELECT s2.sig, "
            "count(DISTINCT i.record) AS support FROM incidence i JOIN step s2 "
            "USING (token) GROUP BY s2.sig) p USING (sig)"
        )
        rows = connection.execute(
            "SELECT token, rung, sig, support FROM resolved ORDER BY token"
        ).fetchall()
    finally:
        connection.close()
    n_tokens = len(pool.token_ids)
    rung = np.full(n_tokens, TERMINAL_RUNG, dtype=np.int64)
    support = np.zeros(n_tokens, dtype=np.int64)
    signatures = [""] * n_tokens
    for token, step_index, text, count in rows:
        rung[token] = step_index
        support[token] = count
        signatures[token] = text
    return Assignment(rung=rung, signature=signatures, support=support)


@dataclass(slots=True)
class CoverageClasses:
    """Coverage classes fitted on a frozen eligible pool.

    Attributes:
        cache: The token cache the classes were fitted on (``None`` for an
            in-memory pool).
        profile: Scientific profile.
        min_support: Structure-support threshold T.
        split: Fit split.
        invalid_record_policy: Eligibility policy for invalid sites.
        where: Metadata filter of a protected stratum, if any.
        ladder: Rungs used.
        presence_families: Presence families of the profile.
        pool: The eligible pool.
        token_class: Class ID of every pool token.
        class_rung: Rung of every class (-1 for terminal residues).
        class_signature: Signature of every class.
        class_support: Distinct-record support of every class.
        presence_signatures: Target signatures of every presence family.
        token_presence: Index into the family's targets of every pool token.
        backend: Backend that computed the assignment.
    """

    cache: TokenCache | None
    profile: Profile
    min_support: int
    split: Split
    invalid_record_policy: InvalidRecordPolicy
    where: dict[str, JSONScalar] | None
    ladder: tuple[Rung, ...]
    presence_families: tuple[PresenceFamily, ...]
    pool: Pool
    token_class: npt.NDArray[np.int64]
    class_rung: npt.NDArray[np.int64]
    class_signature: list[str]
    class_support: npt.NDArray[np.int64]
    presence_signatures: dict[str, list[str]]
    token_presence: dict[str, npt.NDArray[np.int64]]
    backend: str = "memory"
    _key_index: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def n_classes(self) -> int:
        """Number of required classes."""
        return len(self.class_signature)

    @property
    def token_cache(self) -> TokenCache:
        """The token cache, required for auditing and export.

        Raises:
            ConfigError: For classes fitted on an in-memory pool.
        """
        if self.cache is None:
            raise ConfigError(
                "these classes were fitted on an in-memory pool; auditing and "
                "export need classes fitted on a token cache"
            )
        return self.cache

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the class universe (cache, pool, ladder, T)."""
        source = (
            self.cache.fingerprint
            if self.cache is not None
            else digest([self.pool.record_ids.tolist(), self.pool.token_keys])
        )
        return digest(
            {
                "cache": source,
                "split": self.split.value,
                "invalid_record_policy": self.invalid_record_policy.value,
                "where": self.where,
                "ladder": [(r.name, r.fields) for r in self.ladder],
                "presence": [(f.name, f.fields) for f in self.presence_families],
                "min_support": self.min_support,
                "support_unit": "record",
            }
        )

    def rung_name(self, rung: int) -> str:
        """Return a rung's name (``"terminal"`` for terminal residues)."""
        return "terminal" if rung == TERMINAL_RUNG else self.ladder[rung].name

    def assign(self, keys: Iterable[str]) -> list[int | None]:
        """Look full tokens up in the frozen mapping (exact match only).

        Held-out tokens never modify the classes: unseen tokens are
        out-of-vocabulary (``None``).

        Args:
            keys: Canonical token keys.

        Returns:
            Class ID per key, or ``None`` when the token is not in the pool.
        """
        if not self._key_index:
            self._key_index = {key: i for i, key in enumerate(self.pool.token_keys)}
        result: list[int | None] = []
        for key in keys:
            position = self._key_index.get(key)
            result.append(None if position is None else int(self.token_class[position]))
        return result

    def rung_statistics(self) -> list[dict[str, Any]]:
        """Tokens, atoms and classes resolved at each rung (resolution loss)."""
        position = np.searchsorted(self.pool.token_ids, self.pool.edge_token)
        atoms_per_token = np.bincount(
            position,
            weights=self.pool.edge_multiplicity,
            minlength=len(self.pool.token_ids),
        )
        token_rung = self.class_rung[self.token_class]
        stats = []
        for rung in [*range(len(self.ladder)), TERMINAL_RUNG]:
            members = token_rung == rung
            stats.append(
                {
                    "rung": rung,
                    "name": self.rung_name(rung),
                    "tokens": int(members.sum()),
                    "atoms": int(atoms_per_token[members].sum()),
                    "classes": int((self.class_rung == rung).sum()),
                }
            )
        return stats

    def save(self, directory: str | Path) -> Path:
        """Persist the mapping, classes and presence targets as Parquet/JSON."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "token_id": pa.array(self.pool.token_ids, pa.int64()),
                    "token_key": pa.array(self.pool.token_keys, pa.string()),
                    "class_id": pa.array(self.token_class, pa.int64()),
                    "rung": pa.array(self.class_rung[self.token_class], pa.int64()),
                }
            ),
            out / "token_classes.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "class_id": pa.array(np.arange(self.n_classes), pa.int64()),
                    "rung": pa.array(self.class_rung, pa.int64()),
                    "rung_name": pa.array(
                        [self.rung_name(int(r)) for r in self.class_rung]
                    ),
                    "signature": pa.array(self.class_signature, pa.string()),
                    "support": pa.array(self.class_support, pa.int64()),
                    "terminal": pa.array(self.class_rung == TERMINAL_RUNG, pa.bool_()),
                }
            ),
            out / "classes.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "family": [
                        f for f, sigs in self.presence_signatures.items() for _ in sigs
                    ],
                    "signature": [
                        s for sigs in self.presence_signatures.values() for s in sigs
                    ],
                }
            ),
            out / "presence_targets.parquet",
        )
        (out / "classes.json").write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "cache": None if self.cache is None else str(self.cache.token_dir),
                    "profile": self.profile.name,
                    "min_support": self.min_support,
                    "split": self.split.value,
                    "invalid_record_policy": self.invalid_record_policy.value,
                    "where": self.where,
                    "backend": self.backend,
                    "n_records": len(self.pool.record_ids),
                    "n_tokens": len(self.pool.token_ids),
                    "n_classes": self.n_classes,
                    "rungs": self.rung_statistics(),
                },
                indent=1,
            )
            + "\n"
        )
        return out


class CoverageBuilder:
    """Fit coverage classes with the profile's ladder and threshold T.

    Args:
        profile: Profile name or instance.
        min_structure_support: Threshold T (distinct records).
        invalid_record_policy: ``quarantine`` or ``reference``.
        backend: ``auto``, ``memory`` or ``duckdb``.
        ladder: Override the profile's rungs (for documented variants and tests).
        execution: Resource settings for the DuckDB backend.
        workdir: Spill directory root for the DuckDB backend (default: the
            token cache directory, else a temporary directory removed after
            the fit).
    """

    def __init__(
        self,
        profile: str | Profile = "reference-723696d",
        min_structure_support: int = 100,
        *,
        invalid_record_policy: InvalidRecordPolicy
        | str = InvalidRecordPolicy.QUARANTINE,
        backend: str = "auto",
        ladder: Sequence[Rung] | None = None,
        execution: ExecutionSettings | None = None,
        workdir: str | Path | None = None,
    ) -> None:
        """Validate the settings."""
        self.profile = get_profile(profile)
        if isinstance(min_structure_support, bool) or not isinstance(
            min_structure_support, int
        ):
            raise ConfigError("min_structure_support must be an integer")
        if min_structure_support < self.profile.min_support:
            raise ConfigError(
                f"min_structure_support={min_structure_support} is below "
                f"{self.profile.min_support}, the minimum of profile "
                f"{self.profile.name!r} (T >= n_coverage + 1 with single coverage)"
            )
        if backend not in BACKENDS:
            raise ConfigError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.min_support = min_structure_support
        self.invalid_record_policy = InvalidRecordPolicy(invalid_record_policy)
        self.backend = backend
        self.ladder = tuple(ladder) if ladder is not None else self.profile.ladder
        self.execution = execution or ExecutionSettings()
        self.workdir = Path(workdir) if workdir is not None else None

    def with_threshold(self, min_structure_support: int) -> CoverageBuilder:
        """Return a copy with another threshold (for T sweeps)."""
        clone = CoverageBuilder.__new__(CoverageBuilder)
        clone.__dict__.update(self.__dict__)
        CoverageBuilder.__init__(
            clone,
            self.profile,
            min_structure_support,
            invalid_record_policy=self.invalid_record_policy,
            backend=self.backend,
            ladder=self.ladder,
            execution=self.execution,
            workdir=self.workdir,
        )
        return clone

    def fit(
        self,
        cache: TokenCache,
        split: Split | str = Split.TRAIN,
        *,
        where: Mapping[str, JSONScalar] | None = None,
    ) -> CoverageClasses:
        """Fit the classes on the eligible pool of ``split``.

        Args:
            cache: Indexed token cache.
            split: Fit split (training data only, to avoid leakage).
            where: Optional metadata equality filter (a protected stratum).

        Returns:
            The fitted classes.

        Raises:
            ConfigError: If the eligible pool is empty.
        """
        split = Split.parse(split)
        pool = load_pool(cache, split, self.invalid_record_policy, where)
        if len(pool.record_ids) == 0:
            raise ConfigError(
                f"no eligible {split.value!r} records"
                + (f" matching {dict(where)}" if where else "")
            )
        return self.fit_pool(pool, cache=cache, split=split, where=where)

    def fit_pool(
        self,
        pool: Pool,
        *,
        cache: TokenCache | None = None,
        split: Split | str = Split.TRAIN,
        where: Mapping[str, JSONScalar] | None = None,
    ) -> CoverageClasses:
        """Fit the classes on an explicit eligible pool.

        Args:
            pool: The eligible pool.
            cache: The cache the pool came from (``None`` for synthetic pools,
                which can be selected from but not audited or exported).
            split: Fit split recorded with the classes.
            where: Metadata filter recorded with the classes.

        Returns:
            The fitted classes.
        """
        split = Split.parse(split)
        fields = [signature_fields(SiteToken.from_key(key)) for key in pool.token_keys]
        backend = self.backend
        if backend == "auto":
            backend = (
                "memory"
                if pool.edge_record.size <= AUTO_MEMORY_EDGE_LIMIT
                else "duckdb"
            )
        if backend == "memory":
            assignment = assign_memory(pool, fields, self.ladder, self.min_support)
        else:
            root = self.workdir or (cache.cache_dir if cache is not None else None)
            with tempfile.TemporaryDirectory(prefix="envtok-") as scratch:
                spill = root or Path(scratch)
                assignment = assign_duckdb(
                    pool,
                    fields,
                    self.ladder,
                    self.min_support,
                    lambda: self.execution.duckdb(spill),
                )
        keys = sorted(
            set(zip(assignment.rung.tolist(), assignment.signature, strict=True))
        )
        class_id = {key: i for i, key in enumerate(keys)}
        token_class = np.array(
            [
                class_id[r, s]
                for r, s in zip(
                    assignment.rung.tolist(), assignment.signature, strict=True
                )
            ],
            dtype=np.int64,
        )
        class_support = np.zeros(len(keys), dtype=np.int64)
        class_support[token_class] = assignment.support
        presence_signatures: dict[str, list[str]] = {}
        token_presence: dict[str, npt.NDArray[np.int64]] = {}
        for family in self.profile.presence_families:
            texts = [signature(f, family.fields) for f in fields]
            targets = sorted(set(texts))
            index = {text: i for i, text in enumerate(targets)}
            presence_signatures[family.name] = targets
            token_presence[family.name] = np.array(
                [index[t] for t in texts], dtype=np.int64
            )
        return CoverageClasses(
            cache=cache,
            profile=self.profile,
            min_support=self.min_support,
            split=split,
            invalid_record_policy=self.invalid_record_policy,
            where=dict(where) if where else None,
            ladder=self.ladder,
            presence_families=self.profile.presence_families,
            pool=pool,
            token_class=token_class,
            class_rung=np.array([k[0] for k in keys], dtype=np.int64),
            class_signature=[k[1] for k in keys],
            class_support=class_support,
            presence_signatures=presence_signatures,
            token_presence=token_presence,
            backend=backend,
        )


def paper_table_s1_ladder(profile: Profile) -> tuple[Rung, ...]:
    """Return the 13-rung ladder enumerated in Table S1 of paper v1.

    This is not the reference profile; it exists to document and test the
    difference (rungs R0_full, R1dA..R8dA, R8A, R8).
    """
    ladder = profile.ladder
    wanted = [r.name for r in ladder[:11]] + ["R8A_elem", "R8_elem"]
    by_name = {r.name: r for r in ladder}
    return tuple(replace(by_name[name]) for name in wanted)
