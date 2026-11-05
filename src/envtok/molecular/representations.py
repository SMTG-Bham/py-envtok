"""Versioned, state-conditioned graph tokens and finite molecular descriptors.

Graph dictionaries use original zero-based atom indices in ``bonds`` entries
``[i, j, type]`` (an optional fourth entry is bond stereochemistry). Optional
per-atom lists are ``formal_charges``, ``radicals``, ``stereochemistry``,
``aromatic`` and ``rings``. Missing fields remain unknown. Graph signatures
encode the induced rooted neighbourhood, not a folded fingerprint bit.

``soap`` is the scientific descriptor backend. ``radial`` is an explicitly
simpler, angle-blind smoke-test backend and is not a substitute for validation
with SOAP. All descriptors have one row per original atom.
"""

import importlib
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

import numpy as np
import numpy.typing as npt

from envtok.errors import ConfigError
from envtok.molecular.records import MolecularRecord

StateKey = tuple[str, float | None, int | None, str | None]
FloatArray = npt.NDArray[np.float64]
_BOND_TYPES = {
    "1": "single",
    "1.0": "single",
    "2": "double",
    "2.0": "double",
    "3": "triple",
    "3.0": "triple",
    "1.5": "aromatic",
}
_ATOM_FIELDS = ("formal_charges", "radicals", "stereochemistry", "aromatic", "rings")
_INFERENCE_SPECIES = frozenset((1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53))
_MAX_ATOMIC_NUMBER = 118
_STEREO_BOND_FIELDS = 4
_TORSION_ATOMS = 4
_MATRIX_DIMENSIONS = 2


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _valid_integer(value: object, minimum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


@dataclass(frozen=True, slots=True)
class RepresentationSettings:
    """Settings defining one compatible molecular representation space."""

    backend: str = "soap"
    species: tuple[int, ...] = ()
    graph_radius: int = 2
    r_cut_angstrom: float = 5.0
    n_max: int = 4
    l_max: int = 4
    sigma_angstrom: float = 0.5
    allow_validated_perception: bool = False
    distance_bin_angstrom: float = 0.25
    radial_bins: int = 32
    torsion_bin_degrees: float = 30.0
    separation_bin_angstrom: float = 1.0

    def __post_init__(self) -> None:
        """Reject incompatible or invalid descriptor settings."""
        if self.backend not in {"soap", "radial"}:
            raise ConfigError("molecular geometry.backend must be soap or radial")
        integer_bounds = (
            (self.graph_radius, 0),
            (self.n_max, 1),
            (self.l_max, 0),
            (self.radial_bins, 2),
        )
        if any(not _valid_integer(value, minimum) for value, minimum in integer_bounds):
            raise ConfigError(
                "graph radius and descriptor sizes must be valid integers"
            )
        lengths = (
            self.r_cut_angstrom,
            self.sigma_angstrom,
            self.distance_bin_angstrom,
            self.torsion_bin_degrees,
            self.separation_bin_angstrom,
        )
        if any(not math.isfinite(value) or value <= 0 for value in lengths):
            raise ConfigError(
                "descriptor lengths and bin widths must be positive and finite"
            )
        if self.backend == "soap" and self.r_cut_angstrom <= 1:
            raise ConfigError("SOAP gto basis requires r_cut_angstrom > 1")
        if any(
            not _valid_integer(z, 1) or z > _MAX_ATOMIC_NUMBER for z in self.species
        ):
            raise ConfigError("geometry.species must contain atomic numbers in 1..118")
        if len(set(self.species)) != len(self.species):
            raise ConfigError("geometry.species must not contain duplicates")


@dataclass(frozen=True, slots=True)
class MolecularFeatures:
    """Per-atom hierarchy, descriptors and explicit whole-record safeguards."""

    record_id: str
    family_id: str | None
    state_key: StateKey
    token_hierarchy: tuple[tuple[str, ...], ...]
    signatures: dict[str, str]
    descriptors: npt.NDArray[np.float32]
    graph_status: str
    safeguard_targets: tuple[str, ...] = ()
    zero_descriptor_atoms: tuple[int, ...] = ()


@dataclass(slots=True)
class SignatureRegistry:
    """Retain canonical signatures and detect digest collisions."""

    signatures: dict[str, str] = field(default_factory=dict)

    def register(self, signature: Any, channel: str) -> str:
        """Return a digest only if its canonical payload is unambiguous."""
        serialized = _json(signature)
        token = f"molecular_v1:{channel}:{sha256(serialized.encode()).hexdigest()}"
        previous = self.signatures.setdefault(token, serialized)
        if previous != serialized:
            raise ConfigError(f"molecular token digest collision: {token}")
        return token


@dataclass(frozen=True, slots=True)
class _Graph:
    labels: tuple[tuple[Any, ...], ...]
    edges: dict[tuple[int, int], str]

    def neighbors(self, atom: int) -> list[int]:
        return [j if i == atom else i for i, j in self.edges if atom in (i, j)]


def _atom_fields(graph: dict[str, Any], count: int) -> list[list[Any]]:
    fields: list[list[Any]] = []
    for name in _ATOM_FIELDS:
        values = graph.get(name, [None] * count)
        if not isinstance(values, list | tuple) or len(values) != count:
            raise ValueError(f"invalid graph {name} array")
        for value in values:
            if value is None:
                continue
            if name in {"formal_charges", "radicals"}:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"invalid graph {name} value")
                if name == "radicals" and value < 0:
                    raise ValueError("negative radical count")
            elif name == "stereochemistry":
                if not isinstance(value, str):
                    raise ValueError("stereochemistry must be a string or unknown")
            elif not isinstance(value, bool):
                raise ValueError(f"invalid graph {name} flag")
        fields.append(list(values))
    return fields


def _parse_graph(record: MolecularRecord, raw: dict[str, Any]) -> _Graph:
    fields = _atom_fields(raw, record.natoms)
    bonds = raw.get("bonds")
    if not isinstance(bonds, list | tuple):
        raise ValueError("graph needs an atom-mapped bonds list")
    edges: dict[tuple[int, int], str] = {}
    for bond in bonds:
        if not isinstance(bond, list | tuple) or len(bond) not in {3, 4}:
            raise ValueError("graph bonds must have two endpoints and a type")
        i, j, bond_type = bond[:3]
        if (
            any(
                isinstance(k, bool)
                or not isinstance(k, int)
                or not 0 <= k < record.natoms
                for k in (i, j)
            )
            or i == j
        ):
            raise ValueError("graph bond endpoint is not an original atom index")
        key = (min(i, j), max(i, j))
        if key in edges:
            raise ValueError("duplicate graph bond")
        normalized = _BOND_TYPES.get(str(bond_type), str(bond_type).lower())
        if normalized not in {"single", "double", "triple", "aromatic"}:
            raise ValueError("unsupported graph bond type")
        stereo = bond[3] if len(bond) == _STEREO_BOND_FIELDS else None
        if stereo is not None and not isinstance(stereo, str):
            raise ValueError("invalid bond stereochemistry")
        edges[key] = _json((normalized, stereo))
    ring_atoms = _ring_atoms(edges)
    aromatic_atoms = {
        atom
        for pair, bond in edges.items()
        if json.loads(bond)[0] == "aromatic"
        for atom in pair
    }
    for atom in range(record.natoms):
        if fields[3][atom] is None:
            fields[3][atom] = atom in aromatic_atoms
        if fields[4][atom] is None:
            fields[4][atom] = atom in ring_atoms
    if (
        record.total_charge is not None
        and all(value is not None for value in fields[0])
        and sum(fields[0]) != record.total_charge
    ):
        raise ValueError("graph formal charges disagree with total_charge")
    labels = tuple(
        (int(z), *(values[index] for values in fields))
        for index, z in enumerate(record.atoms.numbers)
    )
    return _Graph(labels, edges)


def _ring_atoms(edges: dict[tuple[int, int], str]) -> set[int]:
    """Find cycle membership from all supplied bonds, beyond the token radius."""
    neighbors: dict[int, set[int]] = {}
    for left, right in edges:
        neighbors.setdefault(left, set()).add(right)
        neighbors.setdefault(right, set()).add(left)
    ring_atoms: set[int] = set()
    for left, right in edges:
        visited = {left}
        frontier = [left]
        while frontier:
            atom = frontier.pop()
            candidates = neighbors[atom] - visited
            if atom == left:
                candidates.discard(right)
            visited.update(candidates)
            frontier.extend(candidates)
        if right in visited:
            ring_atoms.update((left, right))
    return ring_atoms


def _inferred_graph(record: MolecularRecord) -> dict[str, Any] | None:
    """Perceive only the explicitly supported closed-shell main-group domain."""
    charge = record.total_charge
    if (
        charge is None
        or not float(charge).is_integer()
        or record.spin_multiplicity != 1
        or not set(record.atoms.numbers.tolist()) <= _INFERENCE_SPECIES
    ):
        return None
    try:
        chem = importlib.import_module("rdkit.Chem")
        determine = importlib.import_module("rdkit.Chem.rdDetermineBonds")
    except ImportError as exc:
        raise ConfigError(
            "graph perception requires the envtok[molecular] RDKit extra"
        ) from exc
    xyz = f"{record.natoms}\n\n" + "\n".join(
        f"{symbol} {x:.17g} {y:.17g} {z:.17g}"
        for symbol, (x, y, z) in zip(
            record.atoms.get_chemical_symbols(), record.atoms.positions, strict=True
        )
    )
    molecule = chem.MolFromXYZBlock(xyz)
    if molecule is None:
        return None
    determine.DetermineBonds(
        molecule,
        charge=int(charge),
        allowChargedFragments=True,
        embedChiral=True,
        useAtomMap=False,
    )
    atoms = list(molecule.GetAtoms())
    if any(atom.GetNumRadicalElectrons() for atom in atoms):
        return None
    return {
        "bonds": [
            [
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond.GetBondTypeAsDouble(),
                str(bond.GetStereo()),
            ]
            for bond in molecule.GetBonds()
        ],
        "formal_charges": [atom.GetFormalCharge() for atom in atoms],
        "radicals": [atom.GetNumRadicalElectrons() for atom in atoms],
        "stereochemistry": [
            atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
            for atom in atoms
        ],
        "aromatic": [atom.GetIsAromatic() for atom in atoms],
        "rings": [atom.IsInRing() for atom in atoms],
    }


def _get_graph(  # noqa: PLR0911 - explicit graph-status outcomes
    record: MolecularRecord, settings: RepresentationSettings
) -> tuple[_Graph | None, str]:
    raw = record.graph
    status = record.graph_status
    if raw is not None and status in {"supplied", "validated", "inferred"}:
        try:
            return _parse_graph(record, raw), status
        except (ValueError, TypeError) as exc:
            return None, f"fallback_invalid_graph:{exc}"
    if raw is not None or status == "ambiguous":
        return None, "fallback_ambiguous_graph"
    if settings.allow_validated_perception:
        try:
            inferred = _inferred_graph(record)
            if inferred is not None:
                return _parse_graph(record, inferred), "inferred_rdkit"
        except ConfigError:
            raise
        except (ValueError, RuntimeError) as exc:
            return None, f"fallback_perception_failed:{exc}"
        return None, "fallback_perception_unsupported"
    return None, "fallback_missing_graph"


def _refine_partition(partition: list[list[int]], graph: _Graph) -> list[list[int]]:
    while True:
        refined: list[list[int]] = []
        for cell in partition:
            groups: dict[str, list[int]] = {}
            for node in cell:
                signature = _json(
                    [
                        sorted(
                            graph.edges.get((min(node, other), max(node, other)), "")
                            for other in peers
                        )
                        for peers in partition
                    ]
                )
                groups.setdefault(signature, []).append(node)
            refined.extend(groups[key] for key in sorted(groups))
        if refined == partition:
            return refined
        partition = refined


def _twins(left: int, right: int, nodes: list[int], graph: _Graph) -> bool:
    return all(
        graph.edges.get((min(left, other), max(left, other)))
        == graph.edges.get((min(right, other), max(right, other)))
        for other in nodes
        if other not in {left, right}
    )


def _canonical_graph(graph: _Graph, center: int, radius: int) -> str:
    neighborhood = {center}
    frontier = {center}
    for _ in range(radius):
        frontier = {
            other for atom in frontier for other in graph.neighbors(atom)
        } - neighborhood
        neighborhood.update(frontier)
    groups: dict[str, list[int]] = {}
    for atom in sorted(neighborhood):
        groups.setdefault(_json((atom == center, graph.labels[atom])), []).append(atom)
    initial = [groups[key] for key in sorted(groups)]
    nodes = sorted(neighborhood)

    def canonical(partition: list[list[int]]) -> str:
        partition = _refine_partition(partition, graph)
        for index, cell in enumerate(partition):
            if len(cell) > 1:
                representatives: list[int] = []
                for node in cell:
                    if not any(
                        _twins(node, previous, nodes, graph)
                        for previous in representatives
                    ):
                        representatives.append(node)
                return min(
                    canonical(
                        [
                            *partition[:index],
                            [node],
                            [other for other in cell if other != node],
                            *partition[index + 1 :],
                        ]
                    )
                    for node in representatives
                )
        order = [cell[0] for cell in partition]
        positions = {atom: index for index, atom in enumerate(order)}
        bonds = sorted(
            (min(positions[i], positions[j]), max(positions[i], positions[j]), bond)
            for (i, j), bond in graph.edges.items()
            if i in positions and j in positions
        )
        return _json(
            {
                "root": positions[center],
                "atoms": [graph.labels[atom] for atom in order],
                "bonds": bonds,
            }
        )

    return canonical(initial)


def _protected_fallback(record: MolecularRecord, atom: int) -> tuple[Any, ...]:
    fields: list[Any] = []
    raw = record.graph or {}
    for name in ("formal_charges", "radicals"):
        values = raw.get(name)
        value = (
            values[atom]
            if isinstance(values, list | tuple) and len(values) == record.natoms
            else None
        )
        fields.append(
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and (name != "radicals" or value >= 0)
            else None
        )
    return (int(record.atoms.numbers[atom]), *fields)


def _bin(distance: float, width: float) -> int:
    return math.floor(round(distance / width, 10))


def _fallback_hierarchy(
    record: MolecularRecord,
    atom: int,
    settings: RepresentationSettings,
    registry: SignatureRegistry,
) -> tuple[str, ...]:
    distances = np.linalg.norm(
        record.atoms.positions - record.atoms.positions[atom], axis=1
    )
    neighbors = [
        (int(record.atoms.numbers[j]), float(distance))
        for j, distance in enumerate(distances)
        if j != atom and round(float(distance), 10) <= settings.r_cut_angstrom
    ]
    base = {
        "version": 1,
        "state": record.state_key,
        "central": _protected_fallback(record, atom),
        "unavailable": "validated_connectivity",
    }
    tokens = []
    for level in range(settings.graph_radius, -1, -1):
        width = settings.distance_bin_angstrom * 2 ** (settings.graph_radius - level)
        context = (
            sorted((z, _bin(distance, width)) for z, distance in neighbors)
            if level
            else []
        )
        tokens.append(
            registry.register(
                {
                    **base,
                    "level": level,
                    "cutoff": settings.r_cut_angstrom,
                    "bin_width": width,
                    "neighbors": context,
                },
                "fallback",
            )
        )
    return tuple(tokens)


def normalize_descriptors(
    values: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float32], tuple[int, ...]]:
    """Normalize nonzero descriptor rows and report explicit zero-vector atoms."""
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != _MATRIX_DIMENSIONS
        or not matrix.shape[1]
        or not np.isfinite(matrix).all()
    ):
        raise ConfigError("local descriptors must be a finite two-dimensional matrix")
    scale = np.max(np.abs(matrix), axis=1)
    zero = scale == 0
    scaled = np.divide(
        matrix, scale[:, None], out=np.zeros_like(matrix), where=~zero[:, None]
    )
    norms = np.linalg.norm(scaled, axis=1)
    normalized = np.divide(
        scaled, norms[:, None], out=np.zeros_like(matrix), where=~zero[:, None]
    )
    return normalized.astype(np.float32), tuple(np.flatnonzero(zero).tolist())


def _radial_descriptors(
    record: MolecularRecord, settings: RepresentationSettings
) -> FloatArray:
    grid = np.linspace(0, settings.r_cut_angstrom, settings.radial_bins)
    output = np.zeros((record.natoms, len(settings.species) * settings.radial_bins))
    for atom, position in enumerate(record.atoms.positions):
        distances = np.linalg.norm(record.atoms.positions - position, axis=1)
        for species_index, species in enumerate(settings.species):
            selected = distances[
                (record.atoms.numbers == species)
                & (distances <= settings.r_cut_angstrom)
            ]
            density = np.exp(
                -0.5 * ((selected[:, None] - grid) / settings.sigma_angstrom) ** 2
            ).sum(axis=0)
            start = species_index * settings.radial_bins
            output[atom, start : start + settings.radial_bins] = density
    return output


def _soap_descriptors(
    record: MolecularRecord, settings: RepresentationSettings
) -> FloatArray:
    try:
        soap = importlib.import_module("dscribe.descriptors").SOAP
    except ImportError as exc:
        raise ConfigError(
            "SOAP requires the envtok[molecular] DScribe extra; "
            "radial is an explicit smoke-test alternative"
        ) from exc
    backend = soap(
        species=list(settings.species),
        periodic=False,
        r_cut=settings.r_cut_angstrom,
        n_max=settings.n_max,
        l_max=settings.l_max,
        sigma=settings.sigma_angstrom,
        average="off",
        sparse=False,
        dtype="float64",
    )
    return np.asarray(backend.create(record.atoms, centers=None), dtype=np.float64)


def _fragment_targets(
    record: MolecularRecord,
    settings: RepresentationSettings,
    registry: SignatureRegistry,
) -> list[str]:
    fragments = record.atoms.info.get(
        "fragment_ids", record.atoms.arrays.get("fragment_ids")
    )
    if fragments is None:
        return []
    if (
        not isinstance(fragments, list | tuple | np.ndarray)
        or len(fragments) != record.natoms
    ):
        raise ConfigError("fragment_ids must map every original atom")
    groups: dict[str, list[int]] = {}
    for atom, fragment in enumerate(fragments):
        if not isinstance(fragment, str | int | np.integer):
            raise ConfigError("fragment_ids must be strings or integers")
        identity = _json(
            int(fragment) if isinstance(fragment, np.integer) else fragment
        )
        groups.setdefault(identity, []).append(atom)
    targets = []
    for (first, left), (second, right) in itertools.combinations(
        sorted(groups.items()), 2
    ):
        separation = min(
            float(np.linalg.norm(record.atoms.positions[i] - record.atoms.positions[j]))
            for i in left
            for j in right
        )
        signature = {
            "state": record.state_key,
            "family": record.family_id,
            "fragment_pair": (first, second),
            "compositions": [
                sorted(record.atoms.numbers[left].tolist()),
                sorted(record.atoms.numbers[right].tolist()),
            ],
            "distance_bin": _bin(separation, settings.separation_bin_angstrom),
            "width": settings.separation_bin_angstrom,
        }
        targets.append(registry.register(signature, "fragment_separation"))
    return targets


def _torsion_targets(
    record: MolecularRecord,
    settings: RepresentationSettings,
    registry: SignatureRegistry,
) -> list[str]:
    torsions = record.atoms.info.get("torsions", [])
    if not isinstance(torsions, list | tuple):
        raise ConfigError("torsions must be named, family-matched atom-index records")
    targets = []
    for entry in torsions:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or record.family_id is None
        ):
            raise ConfigError("torsion targets require an id and a known family_id")
        indices = entry.get("atoms")
        if (
            not isinstance(indices, list | tuple)
            or len(indices) != _TORSION_ATOMS
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < record.natoms
                for i in indices
            )
            or len(set(indices)) != _TORSION_ATOMS
        ):
            raise ConfigError("torsion atoms must be four distinct original indices")
        symmetry = entry.get("symmetry", 1)
        if isinstance(symmetry, bool) or not isinstance(symmetry, int) or symmetry < 1:
            raise ConfigError("torsion symmetry must be a positive integer")
        try:
            angle = float(
                record.atoms.get_dihedral(
                    indices[0], indices[1], indices[2], indices[3], mic=False
                )
            )
        except ZeroDivisionError as exc:
            raise ConfigError(
                "supplied torsion is undefined for collinear atoms"
            ) from exc
        period = 360.0 / symmetry
        # Round at the periodic seam so numerical rotations cannot split 0/360.
        wrapped = round(angle % period, 8) % period
        signature = {
            "state": record.state_key,
            "family": record.family_id,
            "torsion": entry["id"],
            "symmetry": symmetry,
            "angle_bin": _bin(wrapped, settings.torsion_bin_degrees),
            "width": settings.torsion_bin_degrees,
        }
        targets.append(registry.register(signature, "torsion"))
    return targets


def featurize(
    record: MolecularRecord,
    settings: RepresentationSettings,
    *,
    registry: SignatureRegistry | None = None,
) -> MolecularFeatures:
    """Represent an intact nonperiodic record using a fixed species vocabulary."""
    if record.atoms.pbc.any():
        raise ConfigError("molecular_v1 requires nonperiodic configurations")
    if record.natoms == 0 or not np.isfinite(record.atoms.positions).all():
        raise ConfigError("molecular features require nonempty finite coordinates")
    if not settings.species or not set(record.atoms.numbers.tolist()) <= set(
        settings.species
    ):
        raise ConfigError(
            "all molecular elements must be in the fixed geometry.species vocabulary"
        )
    registry = SignatureRegistry() if registry is None else registry
    graph, status = _get_graph(record, settings)
    hierarchy = []
    for atom in range(record.natoms):
        if graph is None:
            hierarchy.append(_fallback_hierarchy(record, atom, settings, registry))
        else:
            hierarchy.append(
                tuple(
                    registry.register(
                        {
                            "version": 1,
                            "state": record.state_key,
                            "radius": radius,
                            "graph": _canonical_graph(graph, atom, radius),
                        },
                        "graph",
                    )
                    for radius in range(settings.graph_radius, -1, -1)
                )
            )
    values = (
        _soap_descriptors(record, settings)
        if settings.backend == "soap"
        else _radial_descriptors(record, settings)
    )
    descriptors, zeros = normalize_descriptors(values)
    if descriptors.shape[0] != record.natoms:
        raise ConfigError("descriptor backend did not return one row per original atom")
    safeguards = tuple(
        sorted(
            set(
                _fragment_targets(record, settings, registry)
                + _torsion_targets(record, settings, registry)
            )
        )
    )
    used = {token for row in hierarchy for token in row} | set(safeguards)
    signatures = {token: registry.signatures[token] for token in sorted(used)}
    return MolecularFeatures(
        record.record_id,
        record.family_id,
        record.state_key,
        tuple(hierarchy),
        signatures,
        descriptors,
        status,
        safeguards,
        zeros,
    )


def featurize_records(
    records: Sequence[MolecularRecord], settings: RepresentationSettings
) -> list[MolecularFeatures]:
    """Represent a training pool with one registry and frozen species vocabulary.

    If no vocabulary is configured, infer it from this supplied pool only. The
    caller must not include held-out records when fitting this vocabulary.
    """
    if not settings.species:
        settings = replace(
            settings,
            species=tuple(
                sorted({int(z) for record in records for z in record.atoms.numbers})
            ),
        )
    registry = SignatureRegistry()
    return [featurize(record, settings, registry=registry) for record in records]
