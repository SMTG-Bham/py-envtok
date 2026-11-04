"""Arrow schemas of the persisted tables.

Counts and offsets are 64-bit. Token and class identifiers are 64-bit on disk;
in-memory selection arrays choose their width from the actual cardinality.
"""

from collections.abc import Sequence
from dataclasses import fields
from typing import Any

import pyarrow as pa

SCHEMA_VERSION = 1

STRUCTURES = pa.schema(
    [
        ("record_id", pa.string()),
        ("dataset", pa.string()),
        ("revision", pa.string()),
        ("shard", pa.string()),
        ("source_key", pa.string()),
        ("native_id", pa.string()),
        ("natoms", pa.int64()),
        ("split", pa.string()),
        ("group_id", pa.string()),
        ("metadata_json", pa.string()),
        ("formula", pa.string()),
        ("elements", pa.list_(pa.string())),
        ("cell_volume", pa.float64()),
        ("status", pa.string()),
        ("status_reason", pa.string()),
        ("n_invalid_sites", pa.int64()),
        ("n_recovered_sites", pa.int64()),
        ("n_extended_sites", pa.int64()),
        ("n_representatives", pa.int64()),
        ("symmetry_status", pa.string()),
        ("chemenv_status", pa.string()),
        ("moment_source", pa.string()),
        ("energy", pa.float64()),
        ("fmax", pa.float64()),
        ("stress_max", pa.float64()),
        ("has_energy", pa.bool_()),
        ("has_forces", pa.bool_()),
        ("has_stress", pa.bool_()),
        ("warnings_json", pa.string()),
    ]
)

SITES = pa.schema(
    [
        ("record_id", pa.string()),
        ("atom_index", pa.int64()),
        ("element", pa.string()),
        ("moment", pa.float64()),
        ("rep_index", pa.int64()),
        ("neighbour_source", pa.string()),
        ("site_status", pa.string()),
        ("neighbours", pa.string()),
        ("geometry", pa.string()),
        ("csm", pa.float64()),
        ("ratio_min", pa.float64()),
        ("ratio_max", pa.float64()),
        ("asymmetry", pa.float64()),
        ("n_contacts", pa.int64()),
    ]
)

NEIGHBOURS = pa.schema(
    [
        ("record_id", pa.string()),
        ("atom_index", pa.int64()),
        ("neighbour_index", pa.int64()),
        ("image_a", pa.int64()),
        ("image_b", pa.int64()),
        ("image_c", pa.int64()),
        ("element", pa.string()),
        ("distance", pa.float64()),
        ("dx", pa.float64()),
        ("dy", pa.float64()),
        ("dz", pa.float64()),
    ]
)

FAILURES = pa.schema(
    [
        ("record_id", pa.string()),
        ("atom_index", pa.int64()),
        ("stage", pa.string()),
        ("category", pa.string()),
        ("message", pa.string()),
        ("fallback", pa.string()),
        ("fatal", pa.bool_()),
        ("retry_outcome", pa.string()),
    ]
)

TOKENS = pa.schema(
    [
        ("token_id", pa.int64()),
        ("token_key", pa.string()),
        ("element", pa.string()),
        ("moment", pa.int64()),
        (
            "neighbours",
            pa.list_(pa.struct([("element", pa.string()), ("count", pa.int64())])),
        ),
        ("coordination_number", pa.int64()),
        ("geometry", pa.string()),
        ("csm_bin", pa.int64()),
        ("dmin_bin", pa.int64()),
        ("asym_bin", pa.int64()),
        ("n_sites", pa.int64()),
        ("n_records", pa.int64()),
    ]
)


def rows_to_table(rows: Sequence[Any], schema: pa.Schema) -> pa.Table:
    """Convert dataclass rows to an Arrow table with ``schema``.

    Args:
        rows: Dataclass instances whose field names match the schema.
        schema: Target schema.

    Returns:
        The table (empty tables keep the schema).
    """
    names = schema.names
    if rows:
        assert [f.name for f in fields(rows[0])] == names  # noqa: S101 - programming error
    columns = {name: [getattr(row, name) for row in rows] for name in names}
    return pa.table(columns, schema=schema)
