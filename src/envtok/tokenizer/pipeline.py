"""Tokenise one structure record into structure, site, neighbour and failure rows.

The rows hold raw measurements, not bins: bins and token keys are derived
from them at index time (:func:`site_token`), so changing bin edges never
repeats the geometry pass. Each original atom keeps its own initial moment;
only geometric results are reused across a symmetry orbit.
"""

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from pymatgen.core.structure import Structure

from envtok.errors import ConfigError
from envtok.profiles import CorpusPolicy, MagneticPolicy, Profile
from envtok.records import IssueSeverity, StructureRecord, validate_record
from envtok.tokenizer.bins import charge_bin, csm_bin, digitize, moment_bin
from envtok.tokenizer.geometry import GeometryEngine, SiteStatus
from envtok.tokenizer.tokens import SiteToken, neighbour_string, parse_neighbours


class RecordStatus:
    """Record status values of the ``structures`` table."""

    OK = "ok"
    EXCLUDED = "excluded"
    FAILED = "failed"


@dataclass(slots=True)
class SiteRow:
    """Raw per-atom measurements (one row of the ``sites`` table)."""

    record_id: str
    atom_index: int
    element: str
    moment: float
    rep_index: int
    neighbour_source: str
    site_status: str
    neighbours: str
    geometry: str
    csm: float | None
    ratio_min: float | None
    ratio_max: float | None
    asymmetry: float | None
    n_contacts: int


@dataclass(slots=True)
class NeighbourRow:
    """One image-resolved contact of a representative site."""

    record_id: str
    atom_index: int
    neighbour_index: int
    image_a: int
    image_b: int
    image_c: int
    element: str
    distance: float
    dx: float
    dy: float
    dz: float


@dataclass(slots=True)
class FailureRow:
    """A failure, exclusion or recoverable fallback (``failures`` table)."""

    record_id: str
    atom_index: int | None
    stage: str
    category: str
    message: str
    fallback: str
    fatal: bool
    retry_outcome: str | None = None


@dataclass(slots=True)
class StructureRow:
    """One row of the ``structures`` table."""

    record_id: str
    dataset: str
    revision: str
    shard: str
    source_key: str
    native_id: str | None
    natoms: int
    split: str
    group_id: str | None
    metadata_json: str
    formula: str
    elements: list[str]
    cell_volume: float | None
    status: str
    status_reason: str | None
    n_invalid_sites: int = 0
    n_recovered_sites: int = 0
    n_extended_sites: int = 0
    n_representatives: int = 0
    symmetry_status: str | None = None
    chemenv_status: str | None = None
    moment_source: str = "defaulted"
    energy: float | None = None
    fmax: float | None = None
    stress_max: float | None = None
    has_energy: bool = False
    has_forces: bool = False
    has_stress: bool = False
    warnings_json: str = "{}"


@dataclass(slots=True)
class TokenizedRecord:
    """All rows produced for one record."""

    structure: StructureRow
    sites: list[SiteRow] = field(default_factory=list)
    neighbours: list[NeighbourRow] = field(default_factory=list)
    failures: list[FailureRow] = field(default_factory=list)


def stable_seed(record_id: str) -> int:
    """Return a process-independent 32-bit seed derived from a record ID.

    Args:
        record_id: Stable record ID.

    Returns:
        A seed for NumPy's legacy global generator.
    """
    digest = hashlib.blake2b(record_id.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def _label_summary(record: StructureRecord, row: StructureRow) -> None:
    labels = record.labels
    if "energy" in labels:
        row.has_energy = True
        row.energy = float(np.asarray(labels["energy"].value).reshape(()))
    if "forces" in labels:
        forces = np.asarray(labels["forces"].value, dtype=float).reshape(-1, 3)
        row.has_forces = True
        row.fmax = float(np.linalg.norm(forces, axis=1).max()) if forces.size else None
    if "stress" in labels:
        stress = np.asarray(labels["stress"].value, dtype=float)
        row.has_stress = True
        row.stress_max = float(np.abs(stress).max()) if stress.size else None


def _base_row(record: StructureRecord) -> StructureRow:
    atoms = record.atoms
    ref = record.source_ref
    symbols = atoms.get_chemical_symbols()
    try:
        volume: float | None = float(abs(atoms.get_volume()))
    except ValueError:
        volume = None
    row = StructureRow(
        record_id=record.record_id,
        dataset=ref.dataset,
        revision=ref.revision,
        shard=ref.shard,
        source_key=ref.key,
        native_id=ref.native_id,
        natoms=len(atoms),
        split=record.split.value,
        group_id=record.group_id,
        metadata_json=record.metadata_json(),
        formula=atoms.get_chemical_formula() if len(atoms) else "",
        elements=sorted(set(symbols)),
        cell_volume=volume if volume is not None and np.isfinite(volume) else None,
        status=RecordStatus.OK,
        status_reason=None,
    )
    _label_summary(record, row)
    return row


def geometry_structure(atoms: Atoms, moments: np.ndarray) -> Structure:
    """Build the pymatgen structure analysed by ChemEnv.

    Only species, positions, cell and periodicity are transferred; labels and
    calculators are deliberately dropped so they cannot influence geometry.
    The ``magmom`` site property is set as in the reference.

    Args:
        atoms: Input atoms.
        moments: Scalar initial moments in atom order.

    Returns:
        The structure, with site order identical to ``atoms``.
    """
    clean = Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )
    structure = Structure.from_ase_atoms(clean)
    structure.add_site_property("magmom", [float(m) for m in moments])
    return structure


def tokenize_record(
    record: StructureRecord,
    profile: Profile,
    policy: CorpusPolicy,
    engine: GeometryEngine,
    *,
    keep_neighbours: bool = True,
) -> TokenizedRecord:
    """Validate, filter and analyse one record.

    Args:
        record: The input record.
        profile: Scientific profile.
        policy: Corpus eligibility policy.
        engine: Geometry engine built for ``profile``.
        keep_neighbours: Whether to emit image-resolved neighbour rows.

    Returns:
        The rows for every table. Records failing validation have status
        ``failed``; records excluded by policy have status ``excluded``.
    """
    row = _base_row(record)
    result = TokenizedRecord(structure=row)
    moments_disabled = profile.magnetic_policy is MagneticPolicy.DISABLED
    for issue in validate_record(record, require_periodic=True):
        if moments_disabled and issue.code in {
            "noncollinear_moments",
            "nonfinite_moments",
        }:
            continue
        if issue.severity is IssueSeverity.WARN:
            result.failures.append(
                FailureRow(
                    record.record_id,
                    None,
                    "validate",
                    issue.code,
                    issue.message,
                    "none",
                    False,
                )
            )
            continue
        row.status = (
            RecordStatus.FAILED
            if issue.severity is IssueSeverity.FAIL
            else RecordStatus.EXCLUDED
        )
        row.status_reason = issue.code
        result.failures.append(
            FailureRow(
                record.record_id,
                None,
                "validate",
                issue.code,
                issue.message,
                "none",
                True,
            )
        )
        return result
    offending = policy.excluded_symbols(record.atoms.get_chemical_symbols())
    if offending:
        row.status = RecordStatus.EXCLUDED
        row.status_reason = "excluded_elements"
        result.failures.append(
            FailureRow(
                record.record_id,
                None,
                "corpus",
                "excluded_elements",
                f"contains {', '.join(offending)} (corpus preset {policy.preset!r})",
                "none",
                True,
            )
        )
        return result
    atoms = record.atoms
    if moments_disabled:
        moments = np.zeros(len(atoms))
        row.moment_source = "disabled"
    else:
        moments = np.asarray(atoms.get_initial_magnetic_moments(), dtype=float)
        row.moment_source = "supplied" if atoms.has("initial_magmoms") else "defaulted"
    _analyse(record, moments, engine, result, keep_neighbours=keep_neighbours)
    return result


def _analyse(
    record: StructureRecord,
    moments: np.ndarray,
    engine: GeometryEngine,
    result: TokenizedRecord,
    *,
    keep_neighbours: bool,
) -> None:
    row = result.structure
    rng_state = np.random.get_state()  # noqa: NPY002 - restored below
    np.random.seed(stable_seed(record.record_id))  # noqa: NPY002 - ChemEnv uses the global RNG
    try:
        structure = geometry_structure(record.atoms, moments)
        analysis = engine.analyse(structure)
    finally:
        np.random.set_state(rng_state)  # noqa: NPY002
    row.warnings_json = json.dumps(analysis.warnings)
    row.symmetry_status = analysis.symmetry_status
    row.chemenv_status = analysis.chemenv_status
    row.n_representatives = len(analysis.sites)
    for event in analysis.events:
        result.failures.append(
            FailureRow(
                record.record_id,
                event.atom_index,
                event.stage,
                event.category,
                event.message,
                event.fallback,
                False,
            )
        )
    for rep, site in sorted(analysis.sites.items()):
        if keep_neighbours:
            result.neighbours.extend(
                NeighbourRow(
                    record.record_id,
                    rep,
                    contact.index,
                    *contact.image,
                    contact.element,
                    contact.distance,
                    *contact.vector,
                )
                for contact in site.contacts
            )
    for atom_index, rep in enumerate(analysis.equivalent_atoms):
        site = analysis.sites[rep]
        if site.status is SiteStatus.INVALID:
            row.n_invalid_sites += 1
        elif site.status is SiteStatus.RECOVERED:
            row.n_recovered_sites += 1
        if site.source == "chemenv-extended":
            row.n_extended_sites += 1
        result.sites.append(
            SiteRow(
                record_id=record.record_id,
                atom_index=atom_index,
                element=structure[atom_index].specie.symbol,
                moment=float(moments[atom_index]),
                rep_index=rep,
                neighbour_source=site.source.value,
                site_status=site.status.value,
                neighbours=neighbour_string(site.neighbours),
                geometry=site.geometry,
                csm=site.csm,
                ratio_min=site.measures.ratio_min,
                ratio_max=site.measures.ratio_max,
                asymmetry=site.measures.asymmetry,
                n_contacts=site.measures.n_used,
            )
        )


def site_token(
    site: SiteRow, profile: Profile, net_charge: float | None = None
) -> SiteToken:
    """Derive the full token of one site from its raw measurements.

    Args:
        site: Raw site row.
        profile: Scientific profile (bins, moment and charge policies).
        net_charge: The record's resolved net charge (see
            :func:`envtok.tokenizer.charge.resolve_net_charge`); required when
            the profile has a charge field, ignored otherwise.

    Returns:
        The typed token.

    Raises:
        ConfigError: If the profile has a charge field and no charge is given.
    """
    bins = profile.bins
    charge = None
    if profile.charge_enabled:
        if net_charge is None:
            raise ConfigError(
                f"profile {profile.name!r} has a charge field; pass the record's "
                "net charge"
            )
        charge = charge_bin(net_charge, profile.charge_policy)
    return SiteToken(
        element=site.element,
        moment=moment_bin(site.moment, profile.magnetic_policy),
        neighbours=parse_neighbours(site.neighbours),
        geometry=site.geometry,
        csm_bin=csm_bin(site.csm, bins),
        dmin_bin=digitize(site.ratio_min, bins.dmin_edges),
        asym_bin=digitize(site.asymmetry, bins.asymmetry_edges),
        charge=charge,
    )
