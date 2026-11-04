"""Manifest writing and source-preserving structure export."""

import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ase.db
import ase.io
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.adapters import Adapter
from envtok.errors import ConfigError
from envtok.provenance import collect
from envtok.records import SourceRef
from envtok.report import audit, diagnostics, render_html

if TYPE_CHECKING:
    from envtok.corpus import Corpus
    from envtok.selection import SelectionResult

FORMATS = ("extxyz", "asedb")
MANIFEST_FILES = (
    "selected_ids.txt",
    "selected_manifest.parquet",
    "config.resolved.yaml",
    "provenance.json",
    "coverage.json",
    "report.html",
)


def _atomic_text(path: Path, text: str) -> None:
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    partial.write_text(text)
    partial.replace(path)


def manifest_table(result: SelectionResult) -> pa.Table:
    """Return the selected records with source locators and selection reasons."""
    structures = result.classes.cache.read(
        "structures",
        [
            "record_id",
            "dataset",
            "revision",
            "shard",
            "source_key",
            "native_id",
            "natoms",
            "split",
            "group_id",
            "metadata_json",
            "formula",
        ],
    )
    ids = pa.array(result.record_ids, pa.string())
    rows = structures.filter(pc.is_in(structures["record_id"], value_set=ids))
    position = {rid: i for i, rid in enumerate(rows["record_id"].to_pylist())}
    order = [position[rid] for rid in result.record_ids]
    rows = rows.take(pa.array(order, pa.int64()))
    return rows.append_column(
        "reason", pa.array([s.reason for s in result.selected])
    ).append_column("order", pa.array([s.order for s in result.selected], pa.int64()))


def write_manifest(
    result: SelectionResult,
    directory: Path,
    *,
    config: Mapping[str, Any] | None = None,
    save_classes: bool = True,
) -> Path:
    """Write every output of a successful selection.

    Files: ``selected_ids.txt``, ``selected_manifest.parquet``,
    ``config.resolved.yaml``, ``provenance.json``, ``coverage.json`` (the
    independent audit) and ``report.html``; the fitted classes go to
    ``classes/``.

    Args:
        result: The selection.
        directory: Output directory.
        config: Resolved configuration to record (defaults to the selector
            and class settings).
        save_classes: Also persist the frozen class mapping.

    Returns:
        The output directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        directory / "selected_ids.txt", "".join(f"{r}\n" for r in result.record_ids)
    )
    pq.write_table(manifest_table(result), directory / "selected_manifest.parquet")
    classes = result.classes
    resolved = (
        dict(config)
        if config is not None
        else {
            "profile": classes.profile.name,
            "coverage": {
                "min_structure_support": classes.min_support,
                "support_unit": "record",
                "split": classes.split.value,
                "invalid_record_policy": classes.invalid_record_policy.value,
            },
            "selection": result.settings,
        }
    )
    _atomic_text(
        directory / "config.resolved.yaml", yaml.safe_dump(resolved, sort_keys=True)
    )
    _atomic_text(
        directory / "provenance.json",
        json.dumps(collect(result), indent=1, default=str) + "\n",
    )
    report = audit(result)
    tables = diagnostics(classes, result.record_ids)
    _atomic_text(
        directory / "coverage.json",
        json.dumps({**report, "diagnostics": tables}, indent=1, default=str) + "\n",
    )
    _atomic_text(directory / "report.html", render_html(report, tables))
    if save_classes:
        classes.save(directory / "classes")
    return directory


def _frozen(atoms: Atoms) -> Atoms:
    """Return a copy whose calculator holds stored results only."""
    frozen: Atoms = atoms.copy()
    results: dict[str, Any] = (
        dict(getattr(atoms.calc, "results", {}) or {}) if atoms.calc is not None else {}
    )
    frozen.calc = SinglePointCalculator(frozen, **results) if results else None
    return frozen


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def selected_structures(
    result: SelectionResult, adapter: Adapter, *, annotate: bool = True
) -> Iterator[Atoms]:
    """Yield the original atoms of every selected record, in selection order.

    Args:
        result: The selection.
        adapter: Source adapter (checked against the cache by the caller).
        annotate: Add ``envtok_record_id`` and ``envtok_reason`` to ``info``.

    Yields:
        Atoms with frozen single-point labels.
    """
    table = manifest_table(result)
    refs = [
        SourceRef(d, r, s, k, n)
        for d, r, s, k, n in zip(
            table["dataset"].to_pylist(),
            table["revision"].to_pylist(),
            table["shard"].to_pylist(),
            table["source_key"].to_pylist(),
            table["native_id"].to_pylist(),
            strict=True,
        )
    ]
    reasons = dict(
        zip(table["record_id"].to_pylist(), table["reason"].to_pylist(), strict=True)
    )
    for (_, atoms), record_id in zip(
        adapter.retrieve(refs), result.record_ids, strict=True
    ):
        frame = _frozen(atoms)
        if annotate:
            frame.info["envtok_record_id"] = record_id
            frame.info["envtok_reason"] = reasons[record_id]
        yield frame


def export_structures(
    result: SelectionResult,
    path: Path,
    *,
    format: str = "extxyz",  # noqa: A002 - mirrors ase.io
    corpus: Corpus | None = None,
    annotate: bool = True,
) -> Path:
    """Export the selected original structures with their labels.

    Atom order, cell, periodicity, ``info``, per-atom arrays and stored
    energy/forces/stress are preserved; no live calculator is attached.

    Args:
        result: The selection.
        path: Output file (``.extxyz`` or an ASE database file).
        format: ``"extxyz"`` or ``"asedb"``.
        corpus: Source corpus; by default the adapter is rebuilt from the cache
            and checked against the cached source fingerprint.
        annotate: Add ``envtok_record_id``/``envtok_reason`` to each frame.

    Returns:
        The written file.

    Raises:
        ConfigError: For an unknown format.
    """
    if format not in FORMATS:
        raise ConfigError(f"format must be one of {FORMATS}, got {format!r}")
    adapter = corpus.adapter if corpus is not None else result.classes.cache.adapter()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}{path.suffix}")
    frames = selected_structures(result, adapter, annotate=annotate)
    if format == "extxyz":
        with partial.open("w", encoding="utf-8") as handle:
            for atoms in frames:
                ase.io.write(handle, atoms, format="extxyz")
    else:
        database = ase.db.connect(str(partial), type="db")
        for atoms in frames:
            info = {k: _jsonable(v) for k, v in atoms.info.items()}
            database.write(
                atoms,
                key_value_pairs={
                    "envtok_record_id": str(info.get("envtok_record_id", "")),
                    "envtok_reason": str(info.get("envtok_reason", "")),
                },
                data=info,
            )
    partial.replace(path)
    return path
