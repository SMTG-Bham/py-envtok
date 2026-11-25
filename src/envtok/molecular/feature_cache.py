"""Columnar molecular feature caches.

A feature cache holds every candidate descriptor in one float32
``descriptors.npy``, records in order, and each record's metadata in
``records.parquet``. Token signatures are stored once in
``signatures.parquet``, whose row order numbers the tokens that records refer
to. ``features.json`` records the digest of each file and ``commit.json``,
written last, the digest of ``features.json``.

Descriptors are hashed as they are written, so a new cache is not read back.
Loaded features share one string per token and one tuple per distinct token
path, and read signatures only when one is first looked up. Caches written
before ``molecular_v1.6`` are still read.
"""

from __future__ import annotations

import array
import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from typing_extensions import override

from envtok.errors import ConfigError, StaleCacheError
from envtok.molecular.manifests import (
    HashingWriter,
    as_bytes,
    file_hash,
    list_columns,
    list_offsets,
    token_lists,
    write_array_header,
    write_json,
)
from envtok.molecular.representations import MolecularFeatures, RepresentationSettings

DESCRIPTORS = "descriptors.npy"
"""All candidate descriptors as one float32 array, records in manifest order."""
LEGACY_DESCRIPTORS = "descriptors.npz"
"""Caches before ``molecular_v1.5``: one member per record, read for saved runs."""
FEATURES = "features.json"
COMMIT = "commit.json"
RECORDS = "records.parquet"
SIGNATURES = "signatures.parquet"
_LAYOUT = "columnar-v1"
_BATCH = 4096
_MATRIX = 2
_METADATA = "molecular feature metadata is missing or corrupt"
_DESCRIPTORS = "molecular descriptor cache is missing or corrupt"
_ROWS = "descriptor rows do not match the candidate atom count"
_RECORD_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("family_id", pa.string()),
        ("state_key", pa.string()),
        ("graph_status", pa.string()),
        ("atoms", pa.int64()),
        ("token_hierarchy", pa.large_list(pa.large_list(pa.int32()))),
        ("safeguard_targets", pa.large_list(pa.int32())),
        ("zero_descriptor_atoms", pa.large_list(pa.int32())),
    ]
)
_SIGNATURE_SCHEMA = pa.schema([("token", pa.string()), ("signature", pa.string())])


def descriptor_file(directory: Path) -> Path | None:
    """The cache's descriptor array, in the current or pre-v1.5 layout."""
    for name in (DESCRIPTORS, LEGACY_DESCRIPTORS):
        if (directory / name).exists():
            return directory / name
    return None


class SignatureTable:
    """A cache's token signatures, read and checked on first lookup."""

    def __init__(self, path: Path, digest: str, tokens: Sequence[str]) -> None:
        """Refer to ``path``, whose rows are ``tokens`` in order."""
        self.path = path
        self.digest = digest
        self.tokens = tokens

    @cached_property
    def _signatures(self) -> dict[str, str]:
        if not self.path.is_file() or file_hash(self.path) != self.digest:
            raise StaleCacheError(_METADATA)
        values = pq.read_table(self.path, columns=["signature"]).column(0)
        return dict(zip(self.tokens, values.to_pylist(), strict=True))

    def __getitem__(self, token: str) -> str:
        """The canonical signature serialised for ``token``."""
        return self._signatures[token]


class RecordSignatures(Mapping[str, str]):
    """One record's token signatures, in token order, from the pool's table.

    Keys are the tokens of the record's hierarchy and safeguards; values are
    read from the cache only when first needed.
    """

    __slots__ = ("_hierarchy", "_safeguards", "_table")

    def __init__(
        self,
        table: SignatureTable,
        hierarchy: tuple[tuple[str, ...], ...],
        safeguards: tuple[str, ...],
    ) -> None:
        """Describe the tokens used by one record."""
        self._table = table
        self._hierarchy = hierarchy
        self._safeguards = safeguards

    def _used(self) -> set[str]:
        return {token for path in self._hierarchy for token in path}.union(
            self._safeguards
        )

    @override
    def __getitem__(self, token: str) -> str:
        if token not in self._used():
            raise KeyError(token)
        return self._table[token]

    @override
    def __contains__(self, token: object) -> bool:
        return token in self._used()

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._used()))

    @override
    def __len__(self) -> int:
        return len(self._used())

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} tokens)"


class _Batches:
    """Buffer record and new-signature rows and write them in batches."""

    def __init__(self, directory: Path) -> None:
        self.paths = {
            name: directory / (name + ".partial") for name in (RECORDS, SIGNATURES)
        }
        self.records = pq.ParquetWriter(
            self.paths[RECORDS], _RECORD_SCHEMA, compression="zstd"
        )
        self.signatures = pq.ParquetWriter(
            self.paths[SIGNATURES], _SIGNATURE_SCHEMA, compression="zstd"
        )
        self.index: dict[str, int] = {}
        self.count = 0
        self._reset()

    def _reset(self) -> None:
        self.rows: dict[str, list[Any]] = {
            name: [] for name in ("record_id", "family_id", "state_key", "graph_status")
        }
        self.atoms: list[int] = []
        self.path_lengths: list[int] = []
        self.paths_flat: array.array[int] = array.array("i")
        self.safeguard_lengths: list[int] = []
        self.safeguards: array.array[int] = array.array("i")
        self.zero_lengths: list[int] = []
        self.zeros: array.array[int] = array.array("i")
        self.new_tokens: list[str] = []
        self.new_signatures: list[str] = []

    def _id(self, token: str, feature: MolecularFeatures) -> int:
        position = self.index.get(token)
        if position is None:
            position = self.index[token] = len(self.index)
            self.new_tokens.append(token)
            self.new_signatures.append(feature.signatures[token])
        return position

    def add(self, feature: MolecularFeatures) -> None:
        """Buffer one record, numbering tokens in order of first appearance."""
        self.count += 1
        self.rows["record_id"].append(feature.record_id)
        self.rows["family_id"].append(feature.family_id)
        self.rows["state_key"].append(json.dumps(feature.state_key, allow_nan=False))
        self.rows["graph_status"].append(feature.graph_status)
        self.atoms.append(len(feature.token_hierarchy))
        for path in feature.token_hierarchy:
            self.paths_flat.extend(self._id(token, feature) for token in path)
            self.path_lengths.append(len(path))
        self.safeguards.extend(
            self._id(token, feature) for token in feature.safeguard_targets
        )
        self.safeguard_lengths.append(len(feature.safeguard_targets))
        self.zeros.extend(feature.zero_descriptor_atoms)
        self.zero_lengths.append(len(feature.zero_descriptor_atoms))
        if len(self.atoms) >= _BATCH:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows."""
        hierarchy = pa.LargeListArray.from_arrays(
            list_offsets(self.atoms), token_lists(self.path_lengths, self.paths_flat)
        )
        columns = {
            name: pa.array(values, pa.string()) for name, values in self.rows.items()
        }
        self.records.write_table(
            pa.table(
                {
                    **columns,
                    "atoms": pa.array(self.atoms, pa.int64()),
                    "token_hierarchy": hierarchy,
                    "safeguard_targets": token_lists(
                        self.safeguard_lengths, self.safeguards
                    ),
                    "zero_descriptor_atoms": token_lists(self.zero_lengths, self.zeros),
                },
                schema=_RECORD_SCHEMA,
            )
        )
        self.signatures.write_table(
            pa.table(
                {"token": self.new_tokens, "signature": self.new_signatures},
                schema=_SIGNATURE_SCHEMA,
            )
        )
        self._reset()

    def close(self, *, keep: bool) -> dict[str, str]:
        """Finish both tables; keep them under their final names if ``keep``."""
        self.records.close()
        self.signatures.close()
        digests = {}
        for name, partial in self.paths.items():
            if keep:
                partial.replace(partial.with_name(name))
                digests[name] = file_hash(partial.with_name(name))
            else:
                partial.unlink()
        return digests


def write_feature_cache(
    directory: Path,
    features: Iterable[MolecularFeatures],
    natoms: int,
    settings: RepresentationSettings,
) -> dict[str, Any]:
    """Stream features into ``directory`` and return the manifest written last.

    Each record's descriptor rows are written, and hashed, as it arrives, so
    only one batch of descriptors is held at once.

    Raises:
        ConfigError: If the descriptor rows do not match ``natoms`` or a
            record's token hierarchy.
    """
    partial = directory / (DESCRIPTORS + ".partial")
    batches = _Batches(directory)
    rows = 0
    dimension: int | None = None
    try:
        with partial.open("wb") as raw:
            stream = HashingWriter(raw)
            for feature in features:
                block = np.asarray(feature.descriptors)
                if dimension is None and block.ndim == _MATRIX:
                    dimension = int(block.shape[1])
                    write_array_header(stream, np.float32, (natoms, dimension))
                if (
                    block.ndim != _MATRIX
                    or block.shape[1] != dimension
                    or len(block) != len(feature.token_hierarchy)
                    or rows + len(block) > natoms
                ):
                    raise ConfigError(_ROWS)
                stream.write(as_bytes(block, np.float32))
                rows += len(block)
                batches.add(feature)
            digest = stream.digest.hexdigest()
        if dimension is None or rows != natoms:
            raise ConfigError(_ROWS)
        batches.flush()
    except BaseException:
        batches.close(keep=False)
        partial.unlink()
        raise
    files = batches.close(keep=True)
    partial.replace(directory / DESCRIPTORS)
    manifest = {
        "layout": _LAYOUT,
        "settings": asdict(settings),
        "record_count": batches.count,
        "atom_count": natoms,
        "dimension": dimension,
        "arrays_sha256": digest,
        "files": files,
    }
    write_json(directory / FEATURES, manifest)
    write_json(directory / COMMIT, {"features_sha256": file_hash(directory / FEATURES)})
    return manifest


def _checked(
    directory: Path,
    manifest: Mapping[str, Any],
    names: Sequence[str],
    *,
    verify: bool,
) -> npt.NDArray[np.float32]:
    """Check the named tables and the descriptors, then map the descriptors."""
    files = manifest.get("files")
    if manifest.get("layout") != _LAYOUT or not isinstance(files, dict):
        raise StaleCacheError(_METADATA)
    descriptors = directory / DESCRIPTORS
    if not descriptors.is_file() or (
        verify and file_hash(descriptors) != manifest["arrays_sha256"]
    ):
        raise StaleCacheError(_DESCRIPTORS)
    for name in names:
        path = directory / name
        if not path.is_file() or (verify and file_hash(path) != files.get(name)):
            raise StaleCacheError(_METADATA)
    matrix: npt.NDArray[np.float32] = np.load(
        descriptors, mmap_mode="r", allow_pickle=False
    )
    if matrix.ndim != _MATRIX or len(matrix) != manifest["atom_count"]:
        raise StaleCacheError(_DESCRIPTORS)
    return matrix


def _paths(
    table: pa.Table, tokens: Sequence[str]
) -> tuple[list[tuple[str, ...]], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Distinct token paths, each atom's path, and each record's atom offsets."""
    column = table.column("token_hierarchy").combine_chunks()
    records = column.offsets.to_numpy()
    offsets = column.values.offsets.to_numpy()
    values = column.values.values.to_numpy(zero_copy_only=False)
    lengths = np.diff(offsets)
    path_of = np.zeros(len(lengths), dtype=np.int64)
    paths: list[tuple[str, ...]] = []
    for length in np.unique(lengths).tolist():
        members = np.flatnonzero(lengths == length)
        rows = values[offsets[members, None] + np.arange(length)]
        unique, inverse = np.unique(rows, axis=0, return_inverse=True)
        path_of[members] = len(paths) + inverse.reshape(-1)
        paths.extend(tuple(map(tokens.__getitem__, row)) for row in unique.tolist())
    return paths, path_of, records


def read_feature_cache(
    directory: Path, manifest: Mapping[str, Any], *, verify: bool = True
) -> tuple[tuple[MolecularFeatures, ...], npt.NDArray[np.float32]]:
    """Features, with descriptors as views of the cache's descriptor matrix.

    ``verify`` checks every file against the manifest's digests; skip it only
    for a cache this process has just written.

    Raises:
        StaleCacheError: If a file is missing, changed or inconsistent.
    """
    if isinstance(manifest.get("records"), list):
        return _read_legacy(directory, manifest)
    matrix = _checked(directory, manifest, (RECORDS, SIGNATURES), verify=verify)
    tokens: list[str] = (
        pq.read_table(directory / SIGNATURES, columns=["token"]).column(0).to_pylist()
    )
    digest = manifest["files"][SIGNATURES]
    signatures = SignatureTable(directory / SIGNATURES, digest, tokens)
    table = pq.read_table(directory / RECORDS)
    paths, path_of, records = _paths(table, tokens)
    if records[-1] - records[0] != len(matrix):
        raise StaleCacheError(_DESCRIPTORS)
    safeguard_offsets, safeguard_values = list_columns(table, "safeguard_targets")
    zero_offsets, zero_values = list_columns(table, "zero_descriptor_atoms")
    shared: dict[str, Any] = {}
    first = int(records[0])
    features = []
    for (
        record_id,
        family_id,
        state,
        status,
        (start, stop),
        (safeguard_start, safeguard_stop),
        (zero_start, zero_stop),
    ) in zip(
        table.column("record_id").to_pylist(),
        table.column("family_id").to_pylist(),
        table.column("state_key").to_pylist(),
        table.column("graph_status").to_pylist(),
        itertools.pairwise(records.tolist()),
        itertools.pairwise(safeguard_offsets.tolist()),
        itertools.pairwise(zero_offsets.tolist()),
        strict=True,
    ):
        hierarchy = tuple(map(paths.__getitem__, path_of[start:stop].tolist()))
        safeguards = tuple(
            map(
                tokens.__getitem__,
                safeguard_values[safeguard_start:safeguard_stop].tolist(),
            )
        )
        if state not in shared:
            shared[state] = tuple(json.loads(state))
        features.append(
            MolecularFeatures(
                record_id=record_id,
                family_id=shared.setdefault(family_id, family_id),
                state_key=shared[state],
                token_hierarchy=hierarchy,
                signatures=RecordSignatures(signatures, hierarchy, safeguards),
                descriptors=matrix[start - first : stop - first],
                graph_status=shared.setdefault(status, status),
                safeguard_targets=safeguards,
                zero_descriptor_atoms=tuple(zero_values[zero_start:zero_stop].tolist()),
            )
        )
    return tuple(features), matrix


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    """What an audit needs from a cache: record order, row counts and status."""

    record_ids: list[str]
    atoms: list[int]
    graph_status: list[str]
    zero_descriptor_atoms: list[tuple[int, ...]]
    descriptors: npt.NDArray[np.float32]
    features: tuple[MolecularFeatures, ...] | None = None
    """Full features, already loaded for caches written before ``molecular_v1.6``."""


def read_feature_summary(
    directory: Path, manifest: Mapping[str, Any]
) -> FeatureSummary:
    """Each record's ID, row count, graph status and zero-descriptor atoms.

    Token hierarchies and signatures are not read. The descriptor array and
    record table are checked against the manifest's digests.

    Raises:
        StaleCacheError: If a file is missing, changed or inconsistent.
    """
    if isinstance(manifest.get("records"), list):
        features, matrix = _read_legacy(directory, manifest)
        return FeatureSummary(
            [f.record_id for f in features],
            [len(f.token_hierarchy) for f in features],
            [f.graph_status for f in features],
            [f.zero_descriptor_atoms for f in features],
            matrix,
            features,
        )
    matrix = _checked(directory, manifest, (RECORDS,), verify=True)
    table = pq.read_table(
        directory / RECORDS,
        columns=["record_id", "graph_status", "atoms", "zero_descriptor_atoms"],
    )
    atoms: list[int] = table.column("atoms").to_pylist()
    if sum(atoms) != len(matrix):
        raise StaleCacheError(_DESCRIPTORS)
    zeros: list[list[int]] = table.column("zero_descriptor_atoms").to_pylist()
    return FeatureSummary(
        table.column("record_id").to_pylist(),
        atoms,
        table.column("graph_status").to_pylist(),
        [tuple(values) for values in zeros],
        matrix,
    )


def _read_legacy(
    directory: Path, manifest: Mapping[str, Any]
) -> tuple[tuple[MolecularFeatures, ...], npt.NDArray[np.float32]]:
    """Caches written before ``molecular_v1.6``, with JSON record metadata."""
    path = descriptor_file(directory)
    if path is None or manifest["arrays_sha256"] != file_hash(path):
        raise StaleCacheError(_DESCRIPTORS)
    blocks = legacy_blocks(manifest, path)
    table = manifest.get("signatures", {})
    features = []
    for row, descriptors in zip(manifest["records"], blocks, strict=True):
        hierarchy = tuple(tuple(tokens) for tokens in row["token_hierarchy"])
        safeguards = tuple(row["safeguard_targets"])
        used = {token for tokens in hierarchy for token in tokens} | set(safeguards)
        features.append(
            MolecularFeatures(
                record_id=row["record_id"],
                family_id=row["family_id"],
                state_key=tuple(row["state_key"]),
                token_hierarchy=hierarchy,
                signatures=row.get("signatures")
                or {token: table[token] for token in sorted(used)},
                descriptors=descriptors,
                graph_status=row["graph_status"],
                safeguard_targets=safeguards,
                zero_descriptor_atoms=tuple(row["zero_descriptor_atoms"]),
            )
        )
    matrix = (
        np.concatenate(blocks)
        if path.name == LEGACY_DESCRIPTORS
        else np.load(path, mmap_mode="r", allow_pickle=False)
    )
    return tuple(features), matrix


def legacy_blocks(
    manifest: Mapping[str, Any], path: Path
) -> list[npt.NDArray[np.float32]]:
    """Each legacy record's descriptor rows: mapped views, or npz members."""
    rows = manifest["records"]
    if path.name == LEGACY_DESCRIPTORS:
        with np.load(path, allow_pickle=False) as arrays:
            return [arrays[f"f{index}"] for index in range(len(rows))]
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    sizes = [len(row["token_hierarchy"]) for row in rows]
    if matrix.ndim != _MATRIX or len(matrix) != sum(sizes):
        raise StaleCacheError(_DESCRIPTORS)
    bounds = np.cumsum([0, *sizes]).tolist()
    return [matrix[start:stop] for start, stop in itertools.pairwise(bounds)]
