"""Neighbour sets, polyhedra and symmetry reuse for one periodic structure.

This module ports the geometry functions of the reference tokeniser into
small, separately testable pieces:

1. Symmetry representatives (:func:`symmetry_representatives`): one site per
   spglib orbit, falling back to every site when symmetry analysis fails.
   Geometric results are reused across an orbit; moments never are.
2. ChemEnv structure environments are built once per structure on the
   original cell (no refinement, centroid centring, ``csm_wcs_ctwcc``), and
   read with two ``SimplestChemenvStrategy`` cutoffs (1.4 and 1.7).
3. Per site, the wider neighbour set is adopted only when a contact it adds has
   a normalised Voronoi solid-angle weight >= 0.45
   (:meth:`GeometryEngine.select_neighbour_set`). Added contacts are compared
   as in the reference: a multiset of (element, distance rounded to 0.01 A),
   weighted by the largest Voronoi weight of a matching key.
4. The descriptor (:meth:`GeometryEngine.describe_site`) takes the ChemEnv
   geometry and CSM, or recovers the neighbour set with CrystalNN and labels
   it ``UNKNOWN:<cn>``, or marks the site invalid.

Every contact keeps its periodic image, so distinct images of one atom are
distinct contacts. The neighbour composition, geometry and raw measurements of
a site are always computed from the same chosen neighbour set.

Adapted from em819/omat24-chemenv-subset@723696d (MIT License, Copyright (c)
2026 Evgeny Moerman; see LICENSES/omat24-chemenv-subset-MIT.txt).
"""

import enum
import gc
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pymatgen.analysis.chemenv.coordination_environments import (
    chemenv_strategies,
    structure_environments,
)
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from envtok.profiles.reference import GeometrySettings, MeasureSettings
from envtok.tokenizer.chemenv import make_finder
from envtok.tokenizer.measures import (
    UNDEFINED_MEASURES,
    ContactMeasures,
    contact_measures,
)
from envtok.tokenizer.tokens import INVALID, Composition, composition

SimplestChemenvStrategy = chemenv_strategies.SimplestChemenvStrategy
LightStructureEnvironments = structure_environments.LightStructureEnvironments


class NeighbourSource(enum.StrEnum):
    """Which procedure produced a site's neighbour set."""

    CHEMENV_DEFAULT = "chemenv-default"
    CHEMENV_EXTENDED = "chemenv-extended"
    CRYSTALNN = "crystalnn"
    NONE = "none"


class SiteStatus(enum.StrEnum):
    """Quality of a site's neighbour set."""

    OK = "ok"
    RECOVERED = "recovered"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Contact:
    """One image-resolved contact of a central site.

    Attributes:
        index: Atom index of the neighbour in the original structure.
        image: Periodic image translation of the neighbour.
        element: Neighbour element symbol.
        vector: Cartesian displacement from the central site.
    """

    index: int
    image: tuple[int, int, int]
    element: str
    vector: tuple[float, float, float]

    @property
    def distance(self) -> float:
        """Contact length in angstrom."""
        return float(np.linalg.norm(self.vector))


@dataclass(frozen=True, slots=True)
class SiteGeometry:
    """The geometric descriptor of one representative site.

    Attributes:
        source: Procedure that produced the neighbour set.
        status: Quality of the neighbour set.
        neighbours: Canonical composition, ``None`` when invalid.
        geometry: ChemEnv symbol, ``UNKNOWN:<cn>`` or ``INVALID``.
        csm: Raw continuous symmetry measure, ``None`` when undefined.
        measures: Raw distance-ratio and asymmetry measurements.
        contacts: Image-resolved contacts of the chosen neighbour set.
    """

    source: NeighbourSource
    status: SiteStatus
    neighbours: Composition | None
    geometry: str
    csm: float | None
    measures: ContactMeasures
    contacts: tuple[Contact, ...]


INVALID_SITE = SiteGeometry(
    source=NeighbourSource.NONE,
    status=SiteStatus.INVALID,
    neighbours=None,
    geometry=INVALID,
    csm=None,
    measures=UNDEFINED_MEASURES,
    contacts=(),
)


@dataclass(frozen=True, slots=True)
class GeometryEvent:
    """A recoverable problem met while analysing a structure.

    Attributes:
        stage: ``symmetry``, ``chemenv``, ``neighbour_selection`` or ``crystalnn``.
        category: Exception class name or reason code.
        message: Short explanation.
        fallback: What was done instead.
        atom_index: Affected site, or ``None`` for the whole structure.
    """

    stage: str
    category: str
    message: str
    fallback: str
    atom_index: int | None = None


@dataclass(slots=True)
class StructureGeometry:
    """Geometric analysis of one structure.

    Attributes:
        equivalent_atoms: Representative site of every atom.
        symmetry_status: ``ok``, ``fallback`` (all sites analysed after a
            failure) or ``disabled``.
        chemenv_status: ``ok`` or ``failed``.
        sites: Descriptor of every representative site.
        events: Recoverable problems, in the order met.
        warnings: Counts of third-party warnings raised during the analysis.
    """

    equivalent_atoms: list[int]
    symmetry_status: str
    chemenv_status: str
    sites: dict[int, SiteGeometry] = field(default_factory=dict)
    events: list[GeometryEvent] = field(default_factory=list)
    warnings: dict[str, int] = field(default_factory=dict)


def _event(
    stage: str, exc: BaseException, fallback: str, atom: int | None = None
) -> GeometryEvent:
    return GeometryEvent(
        stage=stage,
        category=type(exc).__name__,
        message=str(exc)[:500],
        fallback=fallback,
        atom_index=atom,
    )


def _image_tuple(image: Any) -> tuple[int, int, int]:
    a, b, c = (int(v) for v in np.rint(np.asarray(image, dtype=np.float64)))
    return a, b, c


def symmetry_representatives(
    structure: Structure, settings: GeometrySettings
) -> tuple[list[int], str, GeometryEvent | None]:
    """Return the representative site of every atom.

    Args:
        structure: The original structure (with the ``magmom`` site property
            set, as in the reference).
        settings: Geometry settings (``symprec``, ``angle_tolerance``,
            ``use_symmetry``).

    Returns:
        ``(equivalent_atoms, status, event)``. On failure every atom is its own
        representative, the status is ``fallback`` and the event explains why.
    """
    n_sites = len(structure)
    if not settings.use_symmetry:
        return list(range(n_sites)), "disabled", None
    try:
        dataset = SpacegroupAnalyzer(
            structure,
            symprec=settings.symprec,
            angle_tolerance=settings.angle_tolerance,
        ).get_symmetry_dataset()
        if dataset is None:
            raise ValueError("spglib returned no symmetry dataset")
        equivalent = [int(x) for x in dataset.equivalent_atoms]
        if len(equivalent) != n_sites:
            raise ValueError(
                f"equivalent_atoms length {len(equivalent)} != {n_sites} sites"
            )
    except Exception as exc:  # noqa: BLE001 - spglib raises many types; never abort
        return list(range(n_sites)), "fallback", _event("symmetry", exc, "all_sites")
    return equivalent, "ok", None


class GeometryEngine:
    """Reusable ChemEnv, CrystalNN and VoronoiNN machinery for one profile.

    Constructing the ChemEnv finder is comparatively expensive, so workers keep
    one engine per process and analyse many structures with it.

    Args:
        settings: Geometry settings.
        measures: Raw measurement settings.
    """

    def __init__(self, settings: GeometrySettings, measures: MeasureSettings) -> None:
        """Create the finder and neighbour algorithms with frozen settings."""
        self.settings = settings
        self.measure_settings = measures
        self.finder = make_finder(
            permutations_safe_override=settings.permutations_safe_override,
            plane_ordering_override=settings.plane_ordering_override,
            plane_safe_permutations=settings.plane_safe_permutations,
        )
        self.finder.setup_parameters(
            centering_type=settings.centering_type,
            include_central_site_in_centroid=settings.include_central_site_in_centroid,
            structure_refinement=settings.structure_refinement,
        )
        cnn = settings.crystalnn
        self.crystalnn = CrystalNN(
            weighted_cn=cnn.weighted_cn,
            cation_anion=cnn.cation_anion,
            distance_cutoffs=cnn.distance_cutoffs,
            x_diff_weight=cnn.x_diff_weight,
            porous_adjustment=cnn.porous_adjustment,
            search_cutoff=cnn.search_cutoff,
            fingerprint_length=cnn.fingerprint_length,
        )
        vnn = settings.voronoi
        self.voronoi = VoronoiNN(
            tol=vnn.tol,
            cutoff=vnn.cutoff,
            allow_pathological=vnn.allow_pathological,
            weight=vnn.weight,
            extra_nn_info=vnn.extra_nn_info,
            compute_adj_neighbors=vnn.compute_adj_neighbors,
        )

    def _strategy(self, distance_cutoff: float) -> Any:
        settings = self.settings
        return SimplestChemenvStrategy(
            distance_cutoff=distance_cutoff,
            angle_cutoff=settings.angle_cutoff,
            additional_condition=settings.additional_condition,
            continuous_symmetry_measure_cutoff=settings.continuous_symmetry_measure_cutoff,
            symmetry_measure_type=settings.symmetry_measure_type,
        )

    def light_environments(
        self, structure: Structure, representatives: list[int]
    ) -> tuple[Any, Any, GeometryEvent | None]:
        """Build ChemEnv environments once and read them at both cutoffs.

        Args:
            structure: The original structure.
            representatives: Sites to analyse; the neighbour search still uses
                the full cell.

        Returns:
            ``(default, extended, event)``; both environments are ``None`` and
            the event explains why when ChemEnv fails for the structure.
        """
        settings = self.settings
        try:
            self.finder.setup_structure(structure)
            environments = self.finder.compute_structure_environments(
                only_cations=settings.only_cations,
                only_indices=representatives,
                maximum_distance_factor=settings.maximum_distance_factor,
                minimum_angle_factor=settings.minimum_angle_factor,
                voronoi_normalized_distance_tolerance=(
                    settings.voronoi_normalized_distance_tolerance
                ),
                voronoi_normalized_angle_tolerance=settings.voronoi_normalized_angle_tolerance,
                optimization=settings.optimization,
            )
            default = LightStructureEnvironments.from_structure_environments(
                strategy=self._strategy(settings.distance_cutoff_default),
                structure_environments=environments,
            )
            extended = LightStructureEnvironments.from_structure_environments(
                strategy=self._strategy(settings.distance_cutoff_extended),
                structure_environments=environments,
            )
        except Exception as exc:  # noqa: BLE001 - ChemEnv/Voronoi failures are data
            return None, None, _event("chemenv", exc, "crystalnn")
        return default, extended, None

    def _neighbour_keyset(
        self, structure: Structure, site_index: int, sites: list[Any]
    ) -> Counter[tuple[str, float]]:
        center = np.array(structure[site_index].coords)
        keys: Counter[tuple[str, float]] = Counter()
        for neighbour in sites:
            distance = round(
                float(np.linalg.norm(np.array(neighbour.coords) - center)),
                self.settings.neighbour_key_decimals,
            )
            keys[neighbour.specie.symbol, distance] += 1
        return keys

    def select_neighbour_set(  # noqa: PLR0911 - mirrors the reference exits
        self, default: Any, extended: Any, structure: Structure, site_index: int
    ) -> tuple[Any, GeometryEvent | None]:
        """Choose the default or extended ChemEnv environments for one site.

        Args:
            default: Environments at the default cutoff, or ``None``.
            extended: Environments at the extended cutoff, or ``None``.
            structure: The original structure.
            site_index: Representative site.

        Returns:
            ``(environments, event)``: the extended environments when an added
            contact is prominent, otherwise the default ones; ``event`` is set
            when a failure forced the default choice.
        """
        if default is None or extended is None:
            return default, None
        try:
            ns_default = default.neighbors_sets[site_index]
            ns_extended = extended.neighbors_sets[site_index]
        except Exception as exc:  # noqa: BLE001 - reference falls back to default
            return default, _event(
                "neighbour_selection", exc, "default_cutoff", site_index
            )
        if not ns_extended:
            return default, None
        k_default = (
            self._neighbour_keyset(structure, site_index, ns_default[0].neighb_sites)
            if ns_default
            else Counter()
        )
        k_extended = self._neighbour_keyset(
            structure, site_index, ns_extended[0].neighb_sites
        )
        added = [
            key
            for key in set(k_extended) | set(k_default)
            if k_extended.get(key, 0) > k_default.get(key, 0)
        ]
        if not added:
            return default, None
        try:
            center = np.array(structure[site_index].coords)
            weights: dict[tuple[str, float], float] = {}
            for info in self.voronoi.get_nn_info(structure, site_index):
                distance = round(
                    float(np.linalg.norm(np.array(info["site"].coords) - center)),
                    self.settings.neighbour_key_decimals,
                )
                key = (info["site"].specie.symbol, distance)
                if info["weight"] > weights.get(key, 0.0):
                    weights[key] = info["weight"]
        except Exception as exc:  # noqa: BLE001 - reference falls back to default
            return default, _event(
                "neighbour_selection", exc, "default_cutoff", site_index
            )
        prominence = max((weights.get(key, 0.0) for key in added), default=0.0)
        if prominence >= self.settings.voronoi_prominence_threshold:
            return extended, None
        return default, None

    def _contacts(
        self, structure: Structure, site_index: int, entries: list[tuple[int, Any, Any]]
    ) -> tuple[tuple[Contact, ...], ContactMeasures]:
        center = np.array(structure[site_index].coords)
        contacts = tuple(
            Contact(
                index=int(index),
                image=_image_tuple(image),
                element=site.specie.symbol,
                vector=tuple(float(x) for x in np.array(site.coords) - center),  # type: ignore[arg-type]
            )
            for index, image, site in entries
        )
        measures = contact_measures(
            structure[site_index].specie.symbol,
            center,
            [(site.specie.symbol, np.array(site.coords)) for _, _, site in entries],
            self.measure_settings,
        )
        return contacts, measures

    def describe_site(
        self,
        environments: Any,
        source: NeighbourSource,
        structure: Structure,
        site_index: int,
    ) -> tuple[SiteGeometry, GeometryEvent | None]:
        """Build the descriptor of one representative site.

        Args:
            environments: The chosen ChemEnv environments, or ``None``.
            source: Which ChemEnv cutoff ``environments`` corresponds to.
            structure: The original structure.
            site_index: Representative site.

        Returns:
            ``(descriptor, event)``; ``event`` records a CrystalNN failure.
        """
        neighbours_info = None
        coord_env: Any = None
        if environments is not None:
            coord_env = environments.coordination_environments[site_index]
            neighbours_info = environments.neighbors_sets[site_index]
        if (
            environments is not None
            and neighbours_info
            and coord_env is not None
            and len(coord_env) > 0
        ):
            chosen = neighbours_info[0]
            sites = list(chosen.neighb_sites)
            images = chosen.neighb_indices_and_images
            entries = [
                (item["index"], item["image_cell"], site)
                for item, site in zip(images, sites, strict=True)
            ]
            contacts, measures = self._contacts(structure, site_index, entries)
            csm = coord_env[0]["csm"]
            return (
                SiteGeometry(
                    source=source,
                    status=SiteStatus.OK,
                    neighbours=composition([site.specie.symbol for site in sites]),
                    geometry=str(coord_env[0]["ce_symbol"]),
                    csm=None if csm is None else float(csm),
                    measures=measures,
                    contacts=contacts,
                ),
                None,
            )
        try:
            info = self.crystalnn.get_nn_info(structure, site_index)
            if info:
                entries = [(n["site_index"], n["image"], n["site"]) for n in info]
                contacts, measures = self._contacts(structure, site_index, entries)
                return (
                    SiteGeometry(
                        source=NeighbourSource.CRYSTALNN,
                        status=SiteStatus.RECOVERED,
                        neighbours=composition([n["site"].specie.symbol for n in info]),
                        geometry=f"UNKNOWN:{len(info)}",
                        csm=None,
                        measures=measures,
                        contacts=contacts,
                    ),
                    None,
                )
        except Exception as exc:  # noqa: BLE001 - irrecoverable sites become INVALID
            return INVALID_SITE, _event("crystalnn", exc, "invalid_token", site_index)
        return INVALID_SITE, GeometryEvent(
            stage="crystalnn",
            category="no_neighbours",
            message="CrystalNN found no neighbours",
            fallback="invalid_token",
            atom_index=site_index,
        )

    def analyse(self, structure: Structure) -> StructureGeometry:
        """Analyse every representative site of a structure.

        Third-party warnings (for example CrystalNN's missing oxidation states)
        are captured and counted rather than propagated, so a caller's warning
        filters cannot change the result. Resource warnings from finalised
        objects are not attributed to the record.

        Args:
            structure: The original structure with the ``magmom`` site property.

        Returns:
            The geometric analysis.
        """
        # Finalise unrelated garbage first, and ignore ResourceWarnings, so
        # warnings raised by other code cannot leak into this record.
        gc.collect()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("ignore", ResourceWarning)
            result = self._analyse(structure)
        counts = Counter(
            f"{w.category.__name__}: {str(w.message).splitlines()[0][:160]}"
            for w in caught
        )
        result.warnings = dict(sorted(counts.items()))
        return result

    def _analyse(self, structure: Structure) -> StructureGeometry:
        equivalent, symmetry_status, event = symmetry_representatives(
            structure, self.settings
        )
        representatives = list(dict.fromkeys(equivalent))
        default, extended, chemenv_event = self.light_environments(
            structure, representatives
        )
        result = StructureGeometry(
            equivalent_atoms=equivalent,
            symmetry_status=symmetry_status,
            chemenv_status="failed" if chemenv_event else "ok",
        )
        result.events.extend(e for e in (event, chemenv_event) if e is not None)
        for rep in representatives:
            chosen, selection_event = self.select_neighbour_set(
                default, extended, structure, rep
            )
            source = (
                NeighbourSource.CHEMENV_EXTENDED
                if chosen is not None and chosen is extended
                else NeighbourSource.CHEMENV_DEFAULT
            )
            site, site_event = self.describe_site(chosen, source, structure, rep)
            result.sites[rep] = site
            result.events.extend(
                e for e in (selection_event, site_event) if e is not None
            )
        return result
