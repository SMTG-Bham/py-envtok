"""Restartable molecular pilot pipeline, manifests and independent auditing.

Fresh runs keep full records through featurisation. Target fitting, selection
and audit use compact candidate metadata and in-memory incidence.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import platform
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from envtok.errors import ConfigError, CoverageError, StaleCacheError
from envtok.execution import parse_size, resident_bytes
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
from envtok.molecular.feature_cache import (
    COMMIT,
    FEATURES,
    FeatureSummary,
    descriptor_file,
    read_feature_cache,
    read_feature_summary,
    write_feature_cache,
)
from envtok.molecular.geometry import (
    DescriptorPool,
    GeometryRegions,
    build_geometry_regions,
    nearest_selected_distances,
)
from envtok.molecular.manifests import (
    TARGETS,
    AtomClasses,
    FrozenTargets,
    write_targets,
)
from envtok.molecular.manifests import read_json as _read_json
from envtok.molecular.manifests import write_json as _write_json
from envtok.molecular.records import MolecularRecord
from envtok.molecular.representations import (
    MAX_BATCH,
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
from envtok.records import SourceRef, Split
from envtok.selection.incidence import check_budget

IMPLEMENTATION_VERSION = "molecular_v1.7"
_PRE_CONSTRAINT_VERSIONS = {f"molecular_v1.{minor}" for minor in range(7)}

MEMORY_BASE_BYTES = 512 * 2**20
"""Descriptor, perception and Arrow libraries loaded once featurisation starts."""
MEMORY_ATOM_BYTES = 1536
"""Features, fitted targets, selection and audit state per candidate atom."""
MEMORY_QUEUED_ATOM_BYTES = 4 * 2**10
"""Descriptors and tokens per atom of a worker batch waiting to be written."""


def estimate_molecular_bytes(records: int, atoms: int, *, workers: int = 1) -> int:
    """Estimate the memory the stages after ingestion add to the loaded records.

    Covers featurisation (including queued worker batches), target fitting,
    selection and the audit. Descriptors are memory-mapped and not counted.
    The constants are calibrated, with headroom, on perturbed copies of OMol
    records, whose fine tokens are almost all distinct; pools that repeat
    chemical environments need less.
    """
    queued = 0.0
    if workers > 1 and records:
        batch = min(MAX_BATCH, math.ceil(records / (4 * workers)))
        queued = (2 * workers + 1) * batch * atoms / records * MEMORY_QUEUED_ATOM_BYTES
    return int(MEMORY_BASE_BYTES + MEMORY_ATOM_BYTES * atoms + queued)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _mapped_classes(
    features: Sequence[MolecularFeatures], mapping: Mapping[str, str]
) -> AtomClasses:
    """Each descriptor row's fitted class, from its atom's finest token."""
    names = sorted(set(mapping.values()))
    index = {name: position for position, name in enumerate(names)}
    codes = np.fromiter(
        (index[mapping[path[0]]] for f in features for path in f.token_hierarchy),
        dtype=np.int32,
        count=sum(len(f.token_hierarchy) for f in features),
    )
    return names, codes


def _pool(
    record_ids: Sequence[str],
    atoms: Sequence[int],
    classes: AtomClasses,
    descriptors: npt.NDArray[np.float32],
) -> DescriptorPool:
    """Records' descriptor rows and classes, without per-atom objects."""
    offsets = np.zeros(len(atoms) + 1, dtype=np.int64)
    np.cumsum(atoms, out=offsets[1:])
    return DescriptorPool(record_ids, offsets, classes[0], classes[1], descriptors)


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


@dataclass(frozen=True, slots=True)
class _CandidateInfo:
    """Source fields needed for the selection manifest and audit distributions."""

    natoms: int
    formula: str
    audit_state: str
    total_charge: float | None
    spin_multiplicity: int | None
    electronic_state_id: str | None
    label_namespace: str
    family_id: str | None
    split: str
    source_ref: SourceRef


@dataclass(frozen=True, slots=True)
class _CandidateIndex:
    rows: dict[str, _CandidateInfo]
    split_sha256: str

    @classmethod
    def from_records(
        cls, records: Iterable[MolecularRecord], split_sha256: str
    ) -> _CandidateIndex:
        rows = {
            r.record_id: _CandidateInfo(
                natoms=r.natoms,
                formula=str(r.atoms.get_chemical_formula()),
                audit_state=_json(r.state_key),
                total_charge=r.total_charge,
                spin_multiplicity=r.spin_multiplicity,
                electronic_state_id=r.electronic_state_id,
                label_namespace=r.label_namespace,
                family_id=r.family_id,
                split=r.split.value,
                source_ref=r.record.source_ref,
            )
            for r in records
        }
        return cls(rows, split_sha256)


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
        self._index: _CandidateIndex | None = None
        self._features: tuple[MolecularFeatures, ...] | None = None
        self._descriptors: npt.NDArray[np.float32] | None = None
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
        self._index = None
        self._features = None
        self._descriptors = None
        self._built = None
        self.timings = {}
        self._check_source()
        self._frozen_targets()
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

    def _load_features(self) -> tuple[MolecularFeatures, ...]:
        features, self._descriptors = read_feature_cache(
            self.feature_directory, self._feature_manifest()
        )
        return features

    def _feature_manifest(self) -> dict[str, Any]:
        """The committed feature manifest, matching the frozen run if loaded."""
        manifest_path = self.feature_directory / FEATURES
        commit = _read_json(self.feature_directory / COMMIT, {"features_sha256"})
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
        return _read_json(manifest_path, {"arrays_sha256"})

    @property
    def seed(self) -> int:
        """Selection randomness, separate from downstream model seeds."""
        return int(self.settings["reproducibility"]["seed"])

    def _source_records(
        self, refs: Sequence[SourceRef] | None = None
    ) -> Iterator[MolecularRecord]:
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
        return iter_extxyz(self.source, refs=refs, **source, **state)

    def ingest(self) -> PreparedRecords:
        """Return full prepared records, re-reading them if already released."""
        self._check_source()
        if self._prepared is not None:
            return self._prepared
        start = time.perf_counter()
        prepared = prepare_records(
            self._source_records(),
            validation_fraction=self.settings["split"]["validation_fraction"],
            seed=self.seed,
            include_constraints=self.software["implementation"]
            not in _PRE_CONSTRAINT_VERSIONS,
        )
        self._check_source()
        if self._index is not None:
            return prepared
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
        self._prepared = prepared
        return prepared

    def _candidate_index(self) -> _CandidateIndex:
        if self._index is None and self._loaded and self._prepared is None:
            self._index = self._saved_candidate_index()
        if self._index is None:
            prepared = self.ingest()
            self._index = _CandidateIndex.from_records(
                prepared.candidates, _digest(prepared.split_manifest)
            )
            if self._loaded:
                self._prepared = None
        return self._index

    def _saved_candidate_index(self) -> _CandidateIndex | None:
        """Stream summaries for the training IDs in a verified split manifest."""
        provenance = _read_json(self.directory / "provenance.json", set())
        digest = provenance.get("split_sha256")
        if not digest:
            return None
        try:
            splits = _read_json(self.directory / "splits.json", set())
            verified = _digest(splits) == digest
        except (StaleCacheError, ValueError):
            return None
        if not verified:
            # Runs can be restored without these optional reports. Recompute
            # eligibility from the verified source if the manifest is unusable.
            return None
        candidates = (
            r for r in self._source_records() if splits.get(r.record_id) == Split.TRAIN
        )
        index = _CandidateIndex.from_records(candidates, digest)
        self._check_source()
        return index

    def featurize(self) -> tuple[MolecularFeatures, ...]:
        """Fit species on training candidates and cache local features."""
        self._check_source()
        if self._features is not None:
            return self._features
        prepared = None if self._loaded else self.ingest()
        if prepared is not None:
            if not prepared.candidates:
                raise ConfigError(
                    "no eligible training candidates; official held-out splits "
                    "are never used for fitting"
                )
            self._check_memory(prepared.candidates)
        start = time.perf_counter()
        commit_path = self.feature_directory / COMMIT
        if prepared is None or (
            self.settings["execution"]["resume"] and commit_path.exists()
        ):
            features = self._load_features()
        else:
            records = prepared.candidates
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
            features = self._write_features(
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
        if self._prepared is not None:
            self._candidate_index()
        self._features = features
        self._prepared = None
        self.timings["featurize_seconds"] = time.perf_counter() - start
        return self._features

    def _check_memory(self, records: Sequence[MolecularRecord]) -> None:
        """Refuse to continue if the estimated peak exceeds ``memory_budget``.

        The peak is what this process holds now, with the records loaded, plus
        :func:`estimate_molecular_bytes` for the remaining stages.
        """
        execution = self.settings["execution"]
        budget = parse_size(execution["memory_budget"])
        if budget is None:
            return
        estimate = estimate_molecular_bytes(
            len(records),
            sum(record.natoms for record in records),
            workers=execution["workers"],
        )
        check_budget(
            (resident_bytes() or 0) + estimate,
            budget,
            f"Molecular selection of {len(records)} records",
        )

    def _write_features(
        self,
        features: Iterator[MolecularFeatures],
        natoms: int,
        settings: RepresentationSettings,
    ) -> tuple[MolecularFeatures, ...]:
        """Write the feature cache, then return its memory-mapped features."""
        manifest = write_feature_cache(
            self.feature_directory, features, natoms, settings
        )
        loaded, self._descriptors = read_feature_cache(
            self.feature_directory, manifest, verify=False
        )
        return loaded

    def build_targets(
        self,
    ) -> tuple[
        tuple[Candidate, ...], dict[str, Target], HierarchyResult, GeometryRegions
    ]:
        """Fit protected chemical hierarchy and bounded geometric prototype regions."""
        self._check_source()
        if self._loaded:
            return self._frozen_targets().load()
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
        atom_classes = _mapped_classes(features, hierarchy.mapping)
        assert self._descriptors is not None  # noqa: S101 - set by featurize()
        regions = build_geometry_regions(
            _pool(
                [f.record_id for f in features],
                [len(f.token_hierarchy) for f in features],
                atom_classes,
                self._descriptors,
            ),
            group_ids=families,
            radius=selection["prototype_region_radius"],
            prototype_cap=selection["prototype_cap"],
            reservoir_size=selection["reservoir_size"],
            seed=self.seed,
        )
        records = self._candidate_index().rows
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
        write_targets(
            self.directory, (candidates, mandatory, hierarchy, regions), atom_classes
        )
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
        self._check_source()
        return math.floor(
            sum(r.natoms for r in self._candidate_index().rows.values())
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
        index = self._candidate_index()
        selected = ((rid, index.rows[rid]) for rid in result.selected_ids)
        rows = [
            {
                "record_id": rid,
                "natoms": r.natoms,
                "total_charge": r.total_charge,
                "spin_multiplicity": r.spin_multiplicity,
                "electronic_state_id": r.electronic_state_id,
                "label_namespace": r.label_namespace,
                "family_id": r.family_id,
                "split": r.split,
                "source_ref_json": _json(asdict(r.source_ref)),
                "reason": result.reasons[rid],
            }
            for rid, r in selected
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
                "split_sha256": index.split_sha256,
                "feature_directory": str(self.feature_directory),
                "targets_sha256": _file_hash(self.directory / TARGETS),
                "features_sha256": _file_hash(self.feature_directory / FEATURES),
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
        return tuple(self._iter_selected_records(selected_ids))

    def _iter_selected_records(
        self, selected_ids: Sequence[str]
    ) -> Iterator[MolecularRecord]:
        self._check_source()
        rows = self._candidate_index().rows
        refs = [rows[rid].source_ref for rid in selected_ids]
        yield from self._source_records(refs)
        # Verify the reread before export_records publishes its partial file.
        self._check_source()

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
        manifest_path = self.feature_directory / FEATURES
        summary: FeatureSummary | None = None
        if manifest_path.exists() and descriptor_file(self.feature_directory):
            provenance = _read_json(
                self.directory / "provenance.json", {"features_sha256"}
            )
            if _file_hash(manifest_path) != provenance["features_sha256"]:
                raise StaleCacheError("frozen molecular feature metadata changed")
            summary = read_feature_summary(
                self.feature_directory, self._feature_manifest()
            )
            atom_classes = targets.atom_classes()
            if atom_classes is None:
                features = summary.features or self._load_features()
                atom_classes = _mapped_classes(features, targets.mapping())
            report["nearest_selected_distances"] = nearest_selected_distances(
                _pool(
                    summary.record_ids,
                    summary.atoms,
                    atom_classes,
                    summary.descriptors,
                ),
                ids,
                seed=self.seed,
            )
            report["geometry_audit_status"] = "available"
        else:
            report["nearest_selected_distances"] = None
            report["geometry_audit_status"] = "unavailable_feature_cache"
        report["geometry_overflow_environments"] = targets.overflow_count()
        report["graph_status"] = dict(Counter(summary.graph_status if summary else ()))
        report["zero_descriptor_atoms"] = (
            {
                record_id: zeros
                for record_id, zeros in zip(
                    summary.record_ids, summary.zero_descriptor_atoms, strict=True
                )
                if zeros
            }
            if summary
            else {}
        )
        report["software_environment_matches_selection"] = (
            self.runtime_software == self.software
        )
        records = self._candidate_index().rows
        selected = [records[rid] for rid in ids]
        distribution_keys: tuple[tuple[str, Callable[[_CandidateInfo], str]], ...] = (
            ("state", lambda r: r.audit_state),
            ("size", lambda r: str(r.natoms)),
            ("formula", lambda r: r.formula),
        )
        report["distributions"] = {
            name: {
                "input": dict(Counter(key(r) for r in records.values())),
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
            self._iter_selected_records(ids),
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
