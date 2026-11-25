"""Restartable molecular pilot pipeline, manifests and independent auditing.

This implementation is intended for measured pilot pools. Records and incidence
are held in memory; full-dataset sharding is a separate scaling milestone.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
import platform
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, fields, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from envtok.errors import ConfigError, CoverageError, StaleCacheError
from envtok.molecular.config import (
    EXECUTION_ONLY,
    MolecularConfig,
    load_molecular_config,
)
from envtok.molecular.data import (
    PreparedRecords,
    export_records,
    iter_extxyz,
    prepare_records,
)
from envtok.molecular.geometry import (
    GeometryRegions,
    build_geometry_regions,
    nearest_selected_distances,
)
from envtok.molecular.manifests import TARGETS, FrozenTargets, write_targets
from envtok.molecular.manifests import read_json as _read_json
from envtok.molecular.manifests import write_json as _write_json
from envtok.molecular.records import MolecularRecord
from envtok.molecular.representations import (
    MolecularFeatures,
    RepresentationSettings,
    iter_features,
)
from envtok.molecular.selection import (
    Candidate,
    HierarchyResult,
    MolecularSelectionResult,
    Target,
    build_hierarchy,
    select_candidates,
)

IMPLEMENTATION_VERSION = "molecular_v1.6"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


DESCRIPTORS = "descriptors.npy"
"""All candidate descriptors as one float32 array, records in manifest order."""
LEGACY_DESCRIPTORS = "descriptors.npz"
"""Caches before ``molecular_v1.5``: one member per record, read for saved runs."""
_MATRIX = 2


def _features_from_cache(
    manifest: Mapping[str, Any], blocks: Sequence[npt.NDArray[np.float32]]
) -> tuple[MolecularFeatures, ...]:
    """Rebuild features from a manifest and each record's descriptor rows.

    Signatures come from the pool table, or from each record in caches written
    before ``molecular_v1.5``.
    """
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
    return tuple(features)


def _descriptor_blocks(
    manifest: Mapping[str, Any], path: Path
) -> list[npt.NDArray[np.float32]]:
    """Each record's descriptor rows: views of the mapped array, or legacy members."""
    rows = manifest["records"]
    if path.name == LEGACY_DESCRIPTORS:
        with np.load(path, allow_pickle=False) as arrays:
            return [arrays[f"f{index}"] for index in range(len(rows))]
    matrix = np.load(path, mmap_mode="r", allow_pickle=False)
    sizes = [len(row["token_hierarchy"]) for row in rows]
    if matrix.ndim != _MATRIX or len(matrix) != sum(sizes):
        raise StaleCacheError("molecular descriptor cache is missing or corrupt")
    bounds = np.cumsum([0, *sizes]).tolist()
    return [matrix[start:stop] for start, stop in itertools.pairwise(bounds)]


def _fingerprinted(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Settings that address a run; worker counts cannot change its results."""
    execution = {
        key: value
        for key, value in settings["execution"].items()
        if key not in EXECUTION_ONLY
    }
    return {**settings, "execution": execution}


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


_RACY_NS = 2_000_000_000
"""File timestamps are coarse (a few ms on Linux), so a file changed this close
to a check could change again without a new timestamp; such stats are not
trusted until a later full hash, as with git's racily clean entries."""


def _stat_key(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


def _trusted(signature: tuple[int, int, int, int] | None, checked: int) -> bool:
    """Whether the file was last changed well before ``checked`` (ns)."""
    return signature is not None and checked - max(signature[1:3]) > _RACY_NS


def _software() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": IMPLEMENTATION_VERSION,
        "packages": {
            dist.metadata["Name"]: dist.version for dist in metadata.distributions()
        },
    }


def audit_selection(
    candidates: Sequence[Candidate],
    mandatory: Mapping[str, Target],
    selected_ids: Sequence[str],
    *,
    budget: float | None = None,
) -> dict[str, Any]:
    """Independently count distinct selected witnesses and enforce the hard budget."""
    lookup = {row.record_id: row for row in candidates}
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) - set(lookup):
        raise CoverageError("selection contains duplicate or non-candidate IDs")
    selected = [lookup[rid] for rid in selected_ids]
    counts: Counter[str] = Counter(token for row in selected for token in row.targets)
    unmet = {
        token: target.requirement - counts[token]
        for token, target in mandatory.items()
        if counts[token] < target.requirement
    }
    cost = sum(row.cost for row in selected)
    if budget is not None and cost > budget:
        raise CoverageError(f"selected cost {cost} exceeds hard budget {budget}")
    refinements = set().union(*(row.refinement_targets for row in candidates))
    covered = set().union(*(row.refinement_targets for row in selected))
    return {
        "complete": not unmet,
        "unmet_constraints": unmet,
        "achieved_cost": cost,
        "budget": budget,
        "candidate_configurations": len(candidates),
        "selected_configurations": len(selected),
        "candidate_atoms": sum(row.cost for row in candidates),
        "target_counts": {token: counts[token] for token in sorted(mandatory)},
        "mandatory_targets": len(mandatory),
        "covered_targets": len(mandatory) - len(unmet),
        "refinement_targets": len(refinements),
        "covered_refinement_targets": len(covered),
        "status": "complete" if not unmet else "heuristic_incomplete",
        "explanation": "All mandatory constraints achieved."
        if not unmet
        else (
            "The heuristic did not achieve every mandatory constraint. "
            "This is not an infeasibility certificate."
        ),
    }


class MolecularPipeline:
    """Run finite-molecule stages without changing the periodic reference profile."""

    def __init__(
        self, config: str | Path | Mapping[str, Any] | MolecularConfig
    ) -> None:
        """Resolve source and configuration-addressed output directories."""
        self.config = (
            config
            if isinstance(config, MolecularConfig)
            else load_molecular_config(config)
        )
        self.settings = self.config.resolved()
        self.source = Path(self.settings["source"]["path"])
        if not self.source.is_file():
            raise ConfigError(f"molecular source does not exist: {self.source}")
        checked = time.time_ns()
        signature = _stat_key(self.source)
        self.source_hash = _file_hash(self.source)
        self._source_stat = signature if _trusted(signature, checked) else None
        self.software = _software()
        self.runtime_software = self.software
        self._settings_hash = _digest(self.settings)
        self._loaded = False
        fingerprint_data = {
            "config": _fingerprinted(self.settings),
            "source_sha256": self.source_hash,
            "software": self.software,
        }
        self.fingerprint = _digest(fingerprint_data)
        self.directory = Path(self.settings["execution"]["workdir"]) / (
            "molecular-" + self.fingerprint[:16]
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        feature_settings = {
            key: self.settings[key]
            for key in (
                "source",
                "state",
                "split",
                "chemistry",
                "geometry",
                "reproducibility",
            )
        }
        feature_key = _digest(
            {
                "settings": feature_settings,
                "source_sha256": self.source_hash,
                "software": self.software,
            }
        )
        self.feature_directory = self.directory.parent / (
            "features-" + feature_key[:16]
        )
        self.feature_directory.mkdir(parents=True, exist_ok=True)
        self._prepared: PreparedRecords | None = None
        self._features: tuple[MolecularFeatures, ...] | None = None
        self._built: (
            tuple[
                tuple[Candidate, ...],
                dict[str, Target],
                HierarchyResult,
                GeometryRegions,
            ]
            | None
        ) = None
        self.timings: dict[str, float] = {}

    @classmethod
    def load_run(cls, directory: str | Path) -> MolecularPipeline:
        """Bind an existing selection and verify its frozen provenance.

        The saved directory is authoritative, including when moved. A different
        runtime is reported without refitting the saved representation or targets.
        """
        self = cls.__new__(cls)
        self.directory = Path(directory).resolve()
        self.config = load_molecular_config(self.directory / "config.resolved.yaml")
        self.settings = self.config.resolved()
        self._settings_hash = _digest(self.settings)
        provenance = _read_json(
            self.directory / "provenance.json",
            {
                "source_sha256",
                "fingerprint",
                "feature_directory",
                "targets_sha256",
                "features_sha256",
            },
        )
        self.source = Path(self.settings["source"]["path"])
        self.source_hash = provenance["source_sha256"]
        self.fingerprint = provenance["fingerprint"]
        self.feature_directory = Path(provenance["feature_directory"])
        self.software = _read_json(
            self.directory / "software-lock.json",
            {"python", "implementation", "packages"},
        )
        expected = _digest(
            {
                "config": _fingerprinted(self.settings),
                "source_sha256": self.source_hash,
                "software": self.software,
            }
        )
        if expected != self.fingerprint:
            raise StaleCacheError(
                "saved molecular configuration or software lock changed"
            )
        self.runtime_software = _software()
        self._source_stat = None
        self._loaded = True
        self._prepared = None
        self._features = None
        self._built = None
        self.timings = {}
        self._check_source()
        self._load_targets()
        return self

    def _check_source(self, *, full: bool = False) -> None:
        """Reject changed settings or source content.

        The source is hashed in full by ``select()`` and ``export()``, and
        whenever its stat signature (size, mtime, ctime, inode) differs from
        the one recorded at the last verified hash. Otherwise an unchanged
        signature stands in for the hash, provided the source had not changed
        within ``_RACY_NS`` of that verification.
        """
        if _digest(self.settings) != self._settings_hash:
            raise StaleCacheError(
                "molecular configuration changed after initialization"
            )
        checked = time.time_ns()
        signature = _stat_key(self.source)
        if not full and signature is not None and signature == self._source_stat:
            return
        try:
            unchanged = _file_hash(self.source) == self.source_hash
        except OSError as exc:
            raise StaleCacheError("molecular source is unavailable") from exc
        if not unchanged:
            raise StaleCacheError("molecular source content changed; create a new run")
        self._source_stat = signature if _trusted(signature, checked) else None

    def _frozen_targets(self) -> FrozenTargets:
        """Open the frozen targets after checking them against the provenance."""
        provenance = _read_json(self.directory / "provenance.json", {"targets_sha256"})
        path = self.directory / TARGETS
        if not path.exists() or _file_hash(path) != provenance["targets_sha256"]:
            raise StaleCacheError("frozen molecular targets are missing or corrupt")
        return FrozenTargets(self.directory)

    def _load_targets(
        self,
    ) -> tuple[
        tuple[Candidate, ...], dict[str, Target], HierarchyResult, GeometryRegions
    ]:
        return self._frozen_targets().load()

    def _descriptor_file(self) -> Path | None:
        for name in (DESCRIPTORS, LEGACY_DESCRIPTORS):
            if (self.feature_directory / name).exists():
                return self.feature_directory / name
        return None

    def _load_features(self) -> tuple[MolecularFeatures, ...]:
        descriptor_path = self._descriptor_file()
        manifest_path = self.feature_directory / "features.json"
        commit = _read_json(self.feature_directory / "commit.json", {"features_sha256"})
        if (
            not manifest_path.exists()
            or _file_hash(manifest_path) != commit["features_sha256"]
        ):
            raise StaleCacheError("molecular feature metadata is missing or corrupt")
        if self._loaded:
            provenance = _read_json(
                self.directory / "provenance.json", {"features_sha256"}
            )
            if _file_hash(manifest_path) != provenance["features_sha256"]:
                raise StaleCacheError("frozen molecular feature metadata changed")
        manifest = _read_json(manifest_path, {"records", "arrays_sha256"})
        if descriptor_path is None or manifest["arrays_sha256"] != _file_hash(
            descriptor_path
        ):
            raise StaleCacheError("molecular descriptor cache is missing or corrupt")
        return _features_from_cache(
            manifest, _descriptor_blocks(manifest, descriptor_path)
        )

    @property
    def seed(self) -> int:
        """Selection randomness, separate from downstream model seeds."""
        return int(self.settings["reproducibility"]["seed"])

    def ingest(self) -> PreparedRecords:
        """Validate records, protect splits and deduplicate conservatively."""
        self._check_source()
        if self._prepared is not None:
            return self._prepared
        start = time.perf_counter()
        source = {
            key: value
            for key, value in self.settings["source"].items()
            if key not in {"adapter", "path"}
        }
        state = {
            key: value
            for key, value in self.settings["state"].items()
            if key != "strict_training_export"
        }
        self._prepared = prepare_records(
            iter_extxyz(self.source, **source, **state),
            validation_fraction=self.settings["split"]["validation_fraction"],
            seed=self.seed,
        )
        self._check_source()
        prepared = self._prepared
        report = {
            key: getattr(prepared, key)
            for key in (
                "exclusions",
                "duplicate_lineage",
                "conflicts",
                "cross_split_duplicates",
                "group_warnings",
                "split_manifest",
            )
        }
        report.update(
            {
                "eligible_records": len(prepared.records),
                "candidate_records": len(prepared.candidates),
                "source_sha256": self.source_hash,
                "missing_charge": sum(r.total_charge is None for r in prepared.records),
                "missing_multiplicity": sum(
                    r.spin_multiplicity is None for r in prepared.records
                ),
            }
        )
        if not self._loaded:
            _write_json(self.directory / "ingestion.json", report)
            _write_json(self.directory / "splits.json", prepared.split_manifest)
        self.timings["ingest_seconds"] = time.perf_counter() - start
        return prepared

    def featurize(self) -> tuple[MolecularFeatures, ...]:
        """Fit species on training candidates and cache local features."""
        self._check_source()
        if self._features is not None:
            return self._features
        records = self.ingest().candidates
        if not records:
            raise ConfigError(
                "no eligible training candidates; official held-out splits "
                "are never used for fitting"
            )
        start = time.perf_counter()
        manifest_path = self.feature_directory / "features.json"
        if self._loaded or (
            self.settings["execution"]["resume"] and manifest_path.exists()
        ):
            self._features = self._load_features()
        else:
            geometry = {
                key: value
                for key, value in self.settings["geometry"].items()
                if key not in {"average", "normalize", "cache_dtype"}
            }
            geometry["species"] = tuple(geometry["species"]) or tuple(
                sorted({int(z) for record in records for z in record.atoms.numbers})
            )
            settings = RepresentationSettings(
                **geometry,
                graph_radius=self.settings["chemistry"]["graph_radius"],
                allow_validated_perception=self.settings["chemistry"][
                    "allow_validated_perception"
                ],
            )
            execution = self.settings["execution"]
            self._features = self._write_features(
                iter_features(
                    records,
                    settings,
                    workers=execution["workers"],
                    threads_per_worker=execution["threads_per_worker"],
                ),
                sum(record.natoms for record in records),
                settings,
            )
        self._check_source()
        self.timings["featurize_seconds"] = time.perf_counter() - start
        return self._features

    def _write_features(
        self,
        features: Iterator[MolecularFeatures],
        natoms: int,
        settings: RepresentationSettings,
    ) -> tuple[MolecularFeatures, ...]:
        """Stream descriptors into one array file and return memory-mapped features.

        Each record's rows are written as it arrives, so only one batch of
        descriptors is held in memory. Signatures are stored once for the pool.
        """
        descriptor_path = self.feature_directory / DESCRIPTORS
        temporary = descriptor_path.with_suffix(".partial.npy")
        rows: list[dict[str, Any]] = []
        signatures: dict[str, str] = {}
        array: np.memmap[Any, Any] | None = None
        offset = 0
        for feature in features:
            if array is None:
                array = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float32,
                    shape=(natoms, feature.descriptors.shape[1]),
                )
            array[offset : offset + len(feature.descriptors)] = feature.descriptors
            offset += len(feature.descriptors)
            signatures.update(feature.signatures)
            rows.append(
                {
                    field.name: getattr(feature, field.name)
                    for field in fields(feature)
                    if field.name not in {"descriptors", "signatures"}
                }
            )
        if array is None or offset != natoms:
            raise ConfigError("descriptor rows do not match the candidate atom count")
        array.flush()
        del array
        temporary.replace(descriptor_path)
        manifest = {
            "records": rows,
            "signatures": dict(sorted(signatures.items())),
            "settings": asdict(settings),
            "arrays_sha256": _file_hash(descriptor_path),
        }
        manifest_path = self.feature_directory / "features.json"
        _write_json(manifest_path, manifest)
        _write_json(
            self.feature_directory / "commit.json",
            {"features_sha256": _file_hash(manifest_path)},
        )
        return _features_from_cache(
            manifest, _descriptor_blocks(manifest, descriptor_path)
        )

    def build_targets(
        self,
    ) -> tuple[
        tuple[Candidate, ...], dict[str, Target], HierarchyResult, GeometryRegions
    ]:
        """Fit protected chemical hierarchy and bounded geometric prototype regions."""
        self._check_source()
        if self._loaded:
            return self._load_targets()
        if self._built is not None:
            return self._built
        start = time.perf_counter()
        features = self.featurize()
        families = {f.record_id: f.family_id for f in features}
        hierarchy_settings = self.settings["hierarchy"]
        hierarchy = build_hierarchy(
            {f.record_id: f.token_hierarchy for f in features},
            hierarchy_settings["min_source_support"],
            family_ids=families,
            rare_policy=hierarchy_settings["rare_policy"],
        )
        selection = self.settings["selection"]
        regions = build_geometry_regions(
            self.descriptor_rows(hierarchy),
            group_ids=families,
            radius=selection["prototype_region_radius"],
            prototype_cap=selection["prototype_cap"],
            reservoir_size=selection["reservoir_size"],
            seed=self.seed,
        )
        records = {r.record_id: r for r in self.ingest().candidates}
        candidates = tuple(
            Candidate(
                f.record_id,
                records[f.record_id].natoms,
                hierarchy.targets[f.record_id] | frozenset(f.safeguard_targets),
                regions.targets[f.record_id]
                if selection["geometric_refinement"] != "none"
                else frozenset(),
                f.family_id,
            )
            for f in features
        )
        mandatory = {
            token: Target(hierarchy_settings["min_selected_representatives"])
            for row in candidates
            for token in sorted(row.targets - hierarchy.optional_targets)
        }
        write_targets(self.directory, (candidates, mandatory, hierarchy, regions))
        # Hold the written prototypes as file pages rather than a second copy.
        regions = replace(
            regions, prototypes=FrozenTargets.unverified(self.directory).prototypes()
        )
        self._built = candidates, mandatory, hierarchy, regions
        self.timings["targets_seconds"] = time.perf_counter() - start
        return self._built

    def descriptor_rows(
        self, hierarchy: HierarchyResult
    ) -> dict[str, list[tuple[str, Any]]]:
        """Pair each local descriptor with its protected fitted class."""
        return {
            f.record_id: [
                (hierarchy.mapping[path[0]], vector)
                for path, vector in zip(f.token_hierarchy, f.descriptors, strict=True)
            ]
            for f in self.featurize()
        }

    def atom_budget(self) -> int | None:
        """Resolve fractions against deduplicated training atoms, rounding down."""
        selection = self.settings["selection"]
        if selection["mode"] == "cover":
            return None
        if selection["atom_budget"] is not None:
            return int(selection["atom_budget"])
        return math.floor(
            sum(r.natoms for r in self.ingest().candidates)
            * float(selection["atom_fraction"])
        )

    def select(self) -> MolecularSelectionResult:
        """Select whole records, audit independently and write the manifest."""
        self._check_source(full=True)
        if (self.directory / "provenance.json").exists():
            self._frozen_targets()
        candidates, mandatory, _, _ = self.build_targets()
        start = time.perf_counter()
        settings = self.settings["selection"]
        result = select_candidates(
            candidates,
            mandatory,
            mode=settings["mode"],
            budget=self.atom_budget(),
            background_reserve_fraction=settings["background_reserve_fraction"],
            seed=self.seed,
            prune=settings["prune"],
        )
        self.timings["select_seconds"] = time.perf_counter() - start
        (self.directory / "selected_ids.txt").write_text(
            "".join(f"{rid}\n" for rid in result.selected_ids)
        )
        selected = self.selected_records(result.selected_ids)
        rows = [
            {
                "record_id": r.record_id,
                "natoms": r.natoms,
                "total_charge": r.total_charge,
                "spin_multiplicity": r.spin_multiplicity,
                "electronic_state_id": r.electronic_state_id,
                "label_namespace": r.label_namespace,
                "family_id": r.family_id,
                "split": r.split.value,
                "source_ref_json": _json(asdict(r.record.source_ref)),
                "reason": result.reasons[r.record_id],
            }
            for r in selected
        ]
        pq.write_table(
            pa.Table.from_pylist(rows)
            if rows
            else pa.table({"record_id": pa.array([], type=pa.string())}),
            self.directory / "selected_manifest.parquet",
        )
        (self.directory / "config.resolved.yaml").write_text(
            yaml.safe_dump(self.settings, sort_keys=True)
        )
        _write_json(self.directory / "software-lock.json", self.software)
        _write_json(
            self.directory / "provenance.json",
            {
                "fingerprint": self.fingerprint,
                "source_sha256": self.source_hash,
                "split_sha256": _digest(self.ingest().split_manifest),
                "feature_directory": str(self.feature_directory),
                "targets_sha256": _file_hash(self.directory / TARGETS),
                "features_sha256": _file_hash(self.feature_directory / "features.json"),
                "candidate_restriction": "eligible deduplicated training records",
                "timings": self.timings,
                "feature_bytes": sum(f.descriptors.nbytes for f in self.featurize()),
                "background_requested_cost": result.background_requested_cost,
                "background_achieved_cost": result.background_achieved_cost,
            },
        )
        self.audit()
        return result

    def selected_records(
        self, selected_ids: Sequence[str]
    ) -> tuple[MolecularRecord, ...]:
        """Retrieve selected records with their original atom order and labels."""
        lookup = {r.record_id: r for r in self.ingest().candidates}
        return tuple(lookup[rid] for rid in selected_ids)

    def audit(self) -> dict[str, Any]:
        """Audit saved IDs against training-only targets, cost and local distances."""
        self._check_source()
        path = self.directory / "selected_ids.txt"
        if not path.exists():
            raise ConfigError("no molecular selection exists; run select first")
        ids = path.read_text().splitlines()
        targets = self._frozen_targets()
        report = audit_selection(
            targets.candidates(),
            targets.requirements(),
            ids,
            budget=self.atom_budget(),
        )
        manifest_path = self.feature_directory / "features.json"
        if manifest_path.exists() and self._descriptor_file() is not None:
            provenance = _read_json(
                self.directory / "provenance.json", {"features_sha256"}
            )
            if _file_hash(manifest_path) != provenance["features_sha256"]:
                raise StaleCacheError("frozen molecular feature metadata changed")
            features = self._load_features()
            mapping = targets.mapping()
            descriptors = {
                feature.record_id: [
                    (mapping[path[0]], vector)
                    for path, vector in zip(
                        feature.token_hierarchy, feature.descriptors, strict=True
                    )
                ]
                for feature in features
            }
            report["nearest_selected_distances"] = nearest_selected_distances(
                descriptors, ids, seed=self.seed
            )
            report["geometry_audit_status"] = "available"
        else:
            features = ()
            report["nearest_selected_distances"] = None
            report["geometry_audit_status"] = "unavailable_feature_cache"
        report["geometry_overflow_environments"] = targets.overflow_count()
        report["graph_status"] = dict(
            Counter(feature.graph_status for feature in features)
        )
        report["zero_descriptor_atoms"] = {
            feature.record_id: feature.zero_descriptor_atoms
            for feature in features
            if feature.zero_descriptor_atoms
        }
        report["software_environment_matches_selection"] = (
            self.runtime_software == self.software
        )
        selected = self.selected_records(ids)
        distribution_keys: tuple[tuple[str, Callable[[MolecularRecord], str]], ...] = (
            ("state", lambda r: _json(r.state_key)),
            ("size", lambda r: str(r.natoms)),
            ("formula", lambda r: str(r.atoms.get_chemical_formula())),
        )
        report["distributions"] = {
            name: {
                "input": dict(Counter(key(r) for r in self.ingest().candidates)),
                "selected": dict(Counter(key(r) for r in selected)),
            }
            for name, key in distribution_keys
        }
        report["scientific_validation"] = (
            "Selection diagnostics only; downstream accuracy has not been established."
        )
        _write_json(self.directory / "coverage.json", report)
        (self.directory / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'>"
            "<title>Molecular selection audit</title>"
            "<h1>Molecular selection audit</h1><pre>"
            + html.escape(_json(report))
            + "</pre>"
        )
        return report

    def export(self, output: str | Path | None = None) -> Path:
        """Audit and export intact records, enforcing the state policy."""
        self.audit()
        return self._export(output)

    def _export(self, output: str | Path | None = None) -> Path:
        self._check_source(full=True)
        ids = (self.directory / "selected_ids.txt").read_text().splitlines()
        return export_records(
            self.selected_records(ids),
            output or self.directory / "subset.extxyz",
            strict_training_export=self.settings["state"]["strict_training_export"],
        )

    def run(self) -> Path:
        """Select, report and optionally materialise a training subset."""
        self.select()
        if self.settings["output"]["format"] != "none":
            # select() has just audited this selection.
            self._export()
        return self.directory
