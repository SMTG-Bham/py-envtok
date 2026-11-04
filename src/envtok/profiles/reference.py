"""The ``reference-723696d`` scientific profile.

This module freezes every setting that defines EnvTok tokens and coverage
classes as implemented in ``em819/omat24-chemenv-subset`` at commit
``723696de2c26584b8a907ef900690c0caf3c7b68``. Settings that the reference left
to library defaults are written out explicitly. The values come from
introspecting the reference environment (pymatgen 2025.6.14, see
``tests/fixtures/reference/defaults-723696d.json``), so a future library
release cannot silently change the profile.

The profile identifies scientific behaviour. It does not promise that a
modern implementation reproduces every historical selected ID: the package
breaks ties by stable record IDs instead of the reference's internal row
order.
"""

import enum
from dataclasses import dataclass, field

from envtok.profiles.radii import CORDERO_2008, RadiusTable

REFERENCE_COMMIT = "723696de2c26584b8a907ef900690c0caf3c7b68"
REFERENCE_REPOSITORY = "https://github.com/em819/omat24-chemenv-subset"
UNDEFINED_SENTINEL = 999


class MagneticPolicy(enum.StrEnum):
    """How the site moment field is derived.

    Attributes:
        SOURCE_ABS_ROUND_ZERO_IF_MISSING: ``round(abs(initial_magmom))`` with
            Python's ties-to-even rounding, and 0 for records without initial
            moments (the reference behaviour).
        DISABLED: The field is ``NA`` for every site of the run (a
            geometry-only token).
    """

    SOURCE_ABS_ROUND_ZERO_IF_MISSING = "source_abs_round_zero_if_missing"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class CrystalNNSettings:
    """Constructor arguments of the CrystalNN rescue (reference defaults)."""

    weighted_cn: bool = False
    cation_anion: bool = False
    distance_cutoffs: tuple[float, float] = (0.5, 1.0)
    x_diff_weight: float = 3.0
    porous_adjustment: bool = True
    search_cutoff: float = 7.0
    fingerprint_length: int | None = None


@dataclass(frozen=True, slots=True)
class VoronoiSettings:
    """Constructor arguments of the VoronoiNN prominence test."""

    tol: float = 0.0
    cutoff: float = 13.0
    allow_pathological: bool = False
    weight: str = "solid_angle"
    extra_nn_info: bool = True
    compute_adj_neighbors: bool = True


@dataclass(frozen=True, slots=True)
class GeometrySettings:
    """Neighbour, polyhedron and symmetry settings.

    Attributes:
        distance_cutoff_default: Normalised distance cutoff of the default
            neighbour set.
        distance_cutoff_extended: Normalised distance cutoff of the wider set.
        angle_cutoff: Normalised angle cutoff of both strategies.
        voronoi_prominence_threshold: Minimum normalised Voronoi solid-angle
            weight of an added contact for the wider set to be adopted.
        neighbour_key_decimals: Rounding of distances in the reference's
            added-contact comparison (element, rounded distance).
        centering_type: ChemEnv centring convention.
        include_central_site_in_centroid: ChemEnv centroid option.
        structure_refinement: Must be ``"none"`` so site indices stay aligned.
        only_cations: Analyse every site, not only cations.
        maximum_distance_factor: ChemEnv Voronoi distance factor.
        minimum_angle_factor: ChemEnv Voronoi angle factor.
        voronoi_normalized_distance_tolerance: ChemEnv tolerance.
        voronoi_normalized_angle_tolerance: ChemEnv tolerance.
        optimization: ChemEnv optimisation level.
        additional_condition: SimplestChemenvStrategy additional condition.
        continuous_symmetry_measure_cutoff: SimplestChemenvStrategy CSM cutoff.
        symmetry_measure_type: Continuous symmetry measure.
        permutations_safe_override: LocalGeometryFinder option.
        plane_ordering_override: LocalGeometryFinder option.
        plane_safe_permutations: LocalGeometryFinder option.
        symprec: SpacegroupAnalyzer distance tolerance for symmetry reuse.
        angle_tolerance: SpacegroupAnalyzer angle tolerance in degrees.
        use_symmetry: Analyse one representative per symmetry orbit.
        crystalnn: Settings of the CrystalNN rescue.
        voronoi: Settings of the VoronoiNN prominence test.
    """

    distance_cutoff_default: float = 1.4
    distance_cutoff_extended: float = 1.7
    angle_cutoff: float = 0.3
    voronoi_prominence_threshold: float = 0.45
    neighbour_key_decimals: int = 2
    centering_type: str = "centroid"
    include_central_site_in_centroid: bool = True
    structure_refinement: str = "none"
    only_cations: bool = False
    maximum_distance_factor: float = 2.0
    minimum_angle_factor: float = 0.05
    voronoi_normalized_distance_tolerance: float = 0.05
    voronoi_normalized_angle_tolerance: float = 0.03
    optimization: int = 2
    additional_condition: int = 1
    continuous_symmetry_measure_cutoff: float = 10.0
    symmetry_measure_type: str = "csm_wcs_ctwcc"
    permutations_safe_override: bool = False
    plane_ordering_override: bool = True
    plane_safe_permutations: bool = False
    symprec: float = 0.01
    angle_tolerance: float = 5.0
    use_symmetry: bool = True
    crystalnn: CrystalNNSettings = field(default_factory=CrystalNNSettings)
    voronoi: VoronoiSettings = field(default_factory=VoronoiSettings)


@dataclass(frozen=True, slots=True)
class MeasureSettings:
    """Raw distance-ratio and asymmetry measurement settings.

    Attributes:
        radii: Versioned covalent radius table.
        asymmetry_lambda: Decay length of the asymmetry weight
            ``exp(-(q - 1) / lambda)``.
    """

    radii: RadiusTable = CORDERO_2008
    asymmetry_lambda: float = 0.10


@dataclass(frozen=True, slots=True)
class BinSettings:
    """Bin edges of the CSM, compression and asymmetry fields.

    The CSM bins mix closed and open bounds: 0 for CSM <= ``csm_edges[0]``,
    1 for <= ``csm_edges[1]``, 2 for < ``csm_edges[2]`` and 3 otherwise.
    The other fields use ``numpy.digitize`` semantics (lower bound closed).

    Attributes:
        csm_edges: CSM thresholds.
        dmin_edges: Edges of the minimum normalised contact distance.
        dmax_edges: Edges of the maximum normalised contact distance (stored
            for diagnostics; not part of the token).
        asymmetry_edges: Edges of the asymmetry magnitude.
    """

    csm_edges: tuple[float, float, float] = (1.0, 5.0, 10.0)
    dmin_edges: tuple[float, ...] = (0.60, 0.70, 0.85, 1.15)
    dmax_edges: tuple[float, ...] = (0.85, 1.15, 1.50)
    asymmetry_edges: tuple[float, ...] = (1.0, 3.0)


SIGNATURE_FIELDS = (
    "elem",
    "spin",
    "nb",
    "poly",
    "f5",
    "dmin",
    "a_bin",
    "poly_reliable",
    "nb_binned",
    "nbset",
    "cn_sig",
)
"""Field names usable in rung and presence signatures (reference column names)."""


@dataclass(frozen=True, slots=True)
class Rung:
    """One coarsening rung: a named projection of the full token.

    Attributes:
        name: Reference rung name, for example ``"R2bdA_binned_mult"``.
        fields: Token fields joined with ``|`` to form the signature.
    """

    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PresenceFamily:
    """A rare-presence constraint family repaired after the cover.

    Attributes:
        name: Reference floor name, for example ``"F_elem_cn_dmin"``.
        fields: Token fields joined with ``|`` to form the signature.
    """

    name: str
    fields: tuple[str, ...]


# (prefix, description, identity fields) in the reference order.
_IDENTITY_LEVELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("R0", "", ("elem", "spin", "nb", "poly", "f5")),
    ("R1", "drop_f5", ("elem", "spin", "nb", "poly")),
    ("R2", "drop_spin", ("elem", "nb", "poly")),
    ("R2a", "unrel_poly", ("elem", "nb", "poly_reliable")),
    ("R2b", "binned_mult", ("elem", "nb_binned", "poly_reliable", "cn_sig")),
    ("R3", "nbset_poly", ("elem", "nbset", "poly_reliable", "cn_sig")),
    ("R4", "nbset_cn", ("elem", "nbset", "cn_sig")),
    ("R5", "nbset", ("elem", "nbset")),
    ("R6", "elem_poly", ("elem", "poly_reliable", "cn_sig")),
    ("R7", "elem_cn", ("elem", "cn_sig")),
    ("R8", "elem", ("elem",)),
)

# (suffix, modifier fields), finest first. The modifier loop is the OUTER loop.
_MODIFIER_CHAIN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dA", ("dmin", "a_bin")),
    ("A", ("a_bin",)),
    ("", ()),
)


def build_reference_ladder() -> tuple[Rung, ...]:
    """Return the 33 reference rungs, finest first.

    Every identity level is offered with both modifiers attached, then with
    the asymmetry bin only, then bare. The first rung is the full token and
    is named ``R0_full`` as in the reference.

    Returns:
        The ordered rungs.
    """
    rungs = []
    for suffix, modifiers in _MODIFIER_CHAIN:
        for prefix, description, identity in _IDENTITY_LEVELS:
            name = f"{prefix}{suffix}_{description}" if description else prefix + suffix
            rungs.append(Rung(name=name, fields=identity + modifiers))
    rungs[0] = Rung(name="R0_full", fields=rungs[0].fields)
    return tuple(rungs)


REFERENCE_LADDER = build_reference_ladder()

REFERENCE_PRESENCE_FAMILIES: tuple[PresenceFamily, ...] = (
    PresenceFamily("F_elem_poly", ("elem", "poly_reliable")),
    PresenceFamily("F_elem_cn", ("elem", "cn_sig")),
    PresenceFamily("F_elem_cn_dmin", ("elem", "cn_sig", "dmin")),
    PresenceFamily("F_elem_cn_abin", ("elem", "cn_sig", "a_bin")),
)


@dataclass(frozen=True, slots=True)
class Profile:
    """A complete, versioned scientific profile.

    Attributes:
        name: Profile identifier recorded in every manifest.
        reference_commit: Upstream commit whose behaviour the profile encodes.
        geometry: Neighbour, polyhedron and symmetry settings.
        measures: Raw measurement settings.
        bins: Field bin edges.
        magnetic_policy: Moment field policy.
        ladder: Ordered coarsening rungs.
        presence_families: Ordered rare-presence repair families.
        min_support: Smallest allowed structure-support threshold T
            (``n_coverage + 1`` with single coverage).
    """

    name: str
    reference_commit: str
    geometry: GeometrySettings
    measures: MeasureSettings
    bins: BinSettings
    magnetic_policy: MagneticPolicy
    ladder: tuple[Rung, ...]
    presence_families: tuple[PresenceFamily, ...]
    min_support: int = 2


REFERENCE_723696D = Profile(
    name="reference-723696d",
    reference_commit=REFERENCE_COMMIT,
    geometry=GeometrySettings(),
    measures=MeasureSettings(),
    bins=BinSettings(),
    magnetic_policy=MagneticPolicy.SOURCE_ABS_ROUND_ZERO_IF_MISSING,
    ladder=REFERENCE_LADDER,
    presence_families=REFERENCE_PRESENCE_FAMILIES,
)
