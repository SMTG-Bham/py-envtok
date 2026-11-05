"""Dataset adapters: ASE iterables, extxyz, ASE databases and OMat24 ASE-LMDB."""

from collections.abc import Mapping
from typing import Any

from envtok.adapters.ase_iterable import AseIterableAdapter, spool_atoms
from envtok.adapters.asedb import AseDbAdapter, OMat24Adapter
from envtok.adapters.base import Adapter, IdPolicy, Locator, RecordOptions, make_record
from envtok.adapters.extxyz import ExtxyzAdapter
from envtok.errors import ConfigError
from envtok.records import Split

ADAPTER_KINDS = ("extxyz", "asedb", "omat24")


def options_from_dict(data: Mapping[str, Any]) -> RecordOptions:
    """Rebuild :class:`RecordOptions` from :meth:`RecordOptions.as_dict` output.

    Args:
        data: Serialised options.

    Returns:
        The options.
    """
    return RecordOptions(
        dataset=data["dataset"],
        revision=data["revision"],
        id_policy=IdPolicy(data["id_policy"]),
        native_id_key=data["native_id_key"],
        split=Split.parse(data["split"]),
        split_key=data["split_key"],
        group_key=data["group_key"],
        label_sources=dict(data["label_sources"]),
        label_units=dict(data["label_units"]),
    )


def adapter_from_spec(spec: Mapping[str, Any]) -> Adapter:
    """Rebuild a file-backed adapter from its :meth:`Adapter.spec`.

    Args:
        spec: Serialised adapter description.

    Returns:
        A new adapter instance.

    Raises:
        ConfigError: If the adapter kind is unknown.
    """
    kind = spec.get("kind")
    options = options_from_dict(spec["options"])
    if kind == "extxyz":
        return ExtxyzAdapter(
            spec["path"],
            options,
            patterns=spec["patterns"],
            compressed=spec["compressed"],
            staging_dir=spec["staging_dir"],
            checksum=spec["checksum"],
        )
    if kind == "asedb":
        return AseDbAdapter(
            spec["path"], options, patterns=spec["patterns"], checksum=spec["checksum"]
        )
    if kind == "omat24":
        return OMat24Adapter(
            spec["path"], options, patterns=spec["patterns"], checksum=spec["checksum"]
        )
    raise ConfigError(f"unknown adapter kind {kind!r}; expected one of {ADAPTER_KINDS}")


__all__ = [
    "ADAPTER_KINDS",
    "Adapter",
    "AseDbAdapter",
    "AseIterableAdapter",
    "ExtxyzAdapter",
    "IdPolicy",
    "Locator",
    "OMat24Adapter",
    "RecordOptions",
    "adapter_from_spec",
    "make_record",
    "options_from_dict",
    "spool_atoms",
]
