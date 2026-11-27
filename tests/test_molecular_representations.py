"""Scientific invariance and failure contracts for the molecular profile."""

import copy
import json
from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from envtok.errors import ConfigError
from envtok.molecular import representations as rep
from envtok.molecular.records import MolecularRecord
from envtok.records import SourceRef, Split, StructureRecord


def _record(atoms: Atoms | None = None, **kwargs: Any) -> MolecularRecord:
    if atoms is None:
        atoms = molecule("H2O")
    base = StructureRecord("r", SourceRef("test", "v1", "s", "0"), atoms, Split.TRAIN)
    return MolecularRecord(
        base, total_charge=0, spin_multiplicity=1, label_namespace="test", **kwargs
    )


def _water_graph() -> dict[str, Any]:
    return {
        "bonds": [[0, 1, "single"], [0, 2, 1]],
        "formal_charges": [0, 0, 0],
        "radicals": [0, 0, 0],
    }


def _radial(**kwargs: Any) -> rep.RepresentationSettings:
    return rep.RepresentationSettings(backend="radial", species=(1, 8), **kwargs)


def _permute(record: MolecularRecord, permutation: list[int]) -> MolecularRecord:
    graph = copy.deepcopy(record.graph)
    if graph:
        inverse = {atom: position for position, atom in enumerate(permutation)}
        graph["bonds"] = [
            [inverse[i], inverse[j], *rest] for i, j, *rest in graph["bonds"]
        ]
        for key in rep._ATOM_FIELDS:
            if key in graph:
                graph[key] = [graph[key][i] for i in permutation]
    return replace(
        record,
        record=replace(record.record, atoms=record.atoms[permutation]),
        graph=graph,
    )


@pytest.mark.parametrize("graph", [None, _water_graph()])
def test_rotation_translation_and_atom_permutation(
    graph: dict[str, Any] | None,
) -> None:
    record = _record(graph=graph, graph_status="supplied")
    settings = _radial()
    original = rep.featurize(record, settings)
    transformed = record.atoms.copy()
    transformed.rotate(73, (1, 2, 3))
    transformed.translate((15, -12, 0.7))
    rotated = rep.featurize(
        replace(record, record=replace(record.record, atoms=transformed)), settings
    )
    assert original.token_hierarchy == rotated.token_hierarchy
    np.testing.assert_allclose(original.descriptors, rotated.descriptors, atol=1e-7)
    permuted = rep.featurize(_permute(record, [2, 0, 1]), settings)
    assert permuted.token_hierarchy == tuple(
        original.token_hierarchy[i] for i in [2, 0, 1]
    )
    np.testing.assert_allclose(
        permuted.descriptors, original.descriptors[[2, 0, 1]], atol=1e-7
    )
    assert original.descriptors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(original.descriptors, axis=1), 1)


def test_canonical_graph_distinguishes_second_shell_and_cycles() -> None:
    # A six-cycle and two disconnected triangles have identical degree/labels,
    # but different rooted induced graphs; a folded WL fingerprint can miss this.
    atoms = Atoms("C6", positions=np.arange(18).reshape(6, 3))
    ring = {"bonds": [[i, (i + 1) % 6, "single"] for i in range(6)]}
    triangles = {
        "bonds": [[0, 1, 1], [1, 2, 1], [2, 0, 1], [3, 4, 1], [4, 5, 1], [5, 3, 1]]
    }
    settings = rep.RepresentationSettings(backend="radial", species=(6,))
    record = _record(atoms, graph=ring, graph_status="validated")
    first = rep.featurize(record, settings)
    second = rep.featurize(replace(record, graph=triangles), settings)
    assert first.token_hierarchy[0][0] != second.token_hierarchy[0][0]
    reordered = rep.featurize(_permute(record, [5, 1, 3, 0, 4, 2]), settings)
    assert sorted(first.token_hierarchy) == sorted(reordered.token_hierarchy)


@pytest.mark.parametrize(
    "changed",
    [
        {"total_charge": 1},
        {"spin_multiplicity": 3},
        {"electronic_state_id": "excited"},
        {"label_namespace": "other"},
        {"total_charge": None},
        {"spin_multiplicity": None},
    ],
)
def test_state_protection_survives_every_hierarchy_level(
    changed: dict[str, Any],
) -> None:
    record = _record()
    first = rep.featurize(record, _radial())
    second = rep.featurize(replace(record, **changed), _radial())
    assert all(
        a != b
        for first_atom, second_atom in zip(
            first.token_hierarchy, second.token_hierarchy, strict=True
        )
        for a, b in zip(first_atom, second_atom, strict=True)
    )


def test_central_formal_charge_and_radicals_are_never_removed() -> None:
    base = _record(graph=_water_graph(), graph_status="supplied")
    changed = copy.deepcopy(base.graph)
    assert changed is not None
    changed["formal_charges"] = [-1, 1, 0]
    changed["radicals"] = [1, 1, 0]
    for status in ["supplied", "ambiguous"]:
        first = rep.featurize(replace(base, graph_status=status), _radial())
        second = rep.featurize(
            replace(base, graph=changed, graph_status=status), _radial()
        )
        assert all(
            a != b
            for a, b in zip(
                first.token_hierarchy[0], second.token_hierarchy[0], strict=True
            )
        )


def test_graph_and_fallback_cannot_cover_each_other() -> None:
    record = _record(graph=_water_graph(), graph_status="supplied")
    first = rep.featurize(record, _radial())
    fallback = rep.featurize(replace(record, graph_status="ambiguous"), _radial())
    assert first.graph_status == "supplied"
    assert fallback.graph_status == "fallback_ambiguous_graph"
    assert not set(first.token_hierarchy[0]) & set(fallback.token_hierarchy[0])
    signature = json.loads(fallback.signatures[fallback.token_hierarchy[0][0]])
    assert signature["unavailable"] == "validated_connectivity"


@pytest.mark.parametrize(
    "graph",
    [
        {},
        {"bonds": None},
        {"bonds": [[0, 1]]},
        {"bonds": [[0, 0, 1]]},
        {"bonds": [[-1, 2, 1]]},
        {"bonds": [[0.0, 1, 1]]},
        {"bonds": [[True, 1, 1]]},
        {"bonds": [[0, 1, 1], [1, 0, 1]]},
        {"bonds": [[0, 1, "coordination"]]},
        {"bonds": [[0, 1, 1, 123]]},
        {"bonds": [], "formal_charges": [0]},
        {"bonds": [], "formal_charges": [False, 0, 0]},
        {"bonds": [], "radicals": [-1, 0, 0]},
        {"bonds": [], "stereochemistry": [1, None, None]},
        {"bonds": [], "rings": [1, None, None]},
        {"bonds": [], "formal_charges": [1, 0, 0]},
    ],
)
def test_invalid_graphs_fall_back_without_discarding_configuration(
    graph: dict[str, Any],
) -> None:
    result = rep.featurize(_record(graph=graph, graph_status="supplied"), _radial())
    assert result.graph_status.startswith("fallback_invalid_graph:")
    assert result.descriptors.shape[0] == 3


def test_known_graph_attributes_and_float_bonds_are_serialized() -> None:
    graph = {
        "bonds": [[0, 1, 2.0, "STEREOE"], [0, 2, 1.5]],
        "formal_charges": [None, 0, 0],
        "radicals": [None, 0, 0],
        "stereochemistry": ["R", None, None],
        "aromatic": [True, False, True],
        "rings": [False, False, False],
    }
    result = rep.featurize(_record(graph=graph, graph_status="supplied"), _radial())
    signature = result.signatures[result.token_hierarchy[0][0]]
    assert "STEREOE" in signature
    assert "aromatic" in signature


@pytest.mark.parametrize(
    "settings",
    [
        {"backend": "unknown"},
        {"graph_radius": -1},
        {"graph_radius": True},
        {"radial_bins": 1},
        {"n_max": 0},
        {"l_max": -1},
        {"r_cut_angstrom": float("nan")},
        {"sigma_angstrom": 0},
        {"backend": "soap", "r_cut_angstrom": 1},
        {"species": (0,)},
        {"species": (True,)},
        {"species": (119,)},
        {"species": (1, 1)},
    ],
)
def test_representation_settings_are_validated(settings: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        rep.RepresentationSettings(**settings)


@pytest.mark.parametrize("values", [[1, 2], [[float("nan")]], [[float("inf")]]])
def test_descriptor_validation(values: Any) -> None:
    with pytest.raises(ConfigError, match="finite two-dimensional"):
        rep.normalize_descriptors(values)


def test_zero_descriptor_is_reported() -> None:
    descriptors, zeros = rep.normalize_descriptors([[0, 0], [3, 4]])
    assert zeros == (0,)
    np.testing.assert_allclose(descriptors, [[0.0, 0.0], [0.6, 0.8]])


def test_descriptor_normalization_avoids_overflow_and_underflow() -> None:
    descriptors, zeros = rep.normalize_descriptors([[1e308, 1e308], [1e-308, 0]])
    assert zeros == ()
    np.testing.assert_allclose(descriptors, [[2**-0.5, 2**-0.5], [1.0, 0.0]])
    with pytest.raises(ConfigError, match="two-dimensional"):
        rep.normalize_descriptors(np.zeros((3, 0)))


def test_ring_context_is_derived_beyond_the_local_radius() -> None:
    ring = {"bonds": [[i, (i + 1) % 6, "single"] for i in range(6)]}
    atoms = Atoms("C6", positions=np.arange(18).reshape(6, 3))
    record = _record(atoms, graph=ring, graph_status="supplied")
    graph = rep._parse_graph(record, ring)
    assert all(label[-1] is True for label in graph.labels)
    assert rep._ring_atoms({(0, 1): "single", (1, 2): "single"}) == set()


def test_hash_collision_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_digest = MagicMock()
    mock_digest.hexdigest.return_value = "constant"
    monkeypatch.setattr(rep, "sha256", lambda _: mock_digest)
    registry = rep.SignatureRegistry()
    assert registry.register({"a": 1}, "graph") == registry.register({"a": 1}, "graph")
    with pytest.raises(ConfigError, match="collision"):
        registry.register({"a": 2}, "graph")


def test_fixed_species_vocabulary_and_input_validation() -> None:
    record = _record()
    with pytest.raises(ConfigError, match="vocabulary"):
        rep.featurize(record, rep.RepresentationSettings(backend="radial"))
    with pytest.raises(ConfigError, match="vocabulary"):
        rep.featurize(
            record, rep.RepresentationSettings(backend="radial", species=(1,))
        )
    record.atoms.pbc = True
    with pytest.raises(ConfigError, match="nonperiodic"):
        rep.featurize(record, _radial())
    record.atoms.pbc = False
    record.atoms.positions[0, 0] = np.nan
    with pytest.raises(ConfigError, match="finite coordinates"):
        rep.featurize(record, _radial())
    with pytest.raises(ConfigError, match="nonempty"):
        rep.featurize(_record(Atoms()), _radial())


def test_training_pool_fits_species_and_detects_geometry_changes() -> None:
    first = _record(graph=_water_graph(), graph_status="supplied")
    atoms = first.atoms.copy()
    atoms.positions[1] += (0.6, -0.4, 0.9)
    second = replace(
        first, record=replace(first.record, record_id="stretched", atoms=atoms)
    )
    features = rep.featurize_records(
        [first, second], rep.RepresentationSettings(backend="radial")
    )
    assert features[0].token_hierarchy == features[1].token_hierarchy
    assert np.linalg.norm(features[0].descriptors - features[1].descriptors) > 0.2
    assert rep.featurize_records([], _radial()) == []
    assert len(rep.featurize_records([first], _radial())) == 1


def test_missing_optional_backends_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(
        "envtok.molecular.representations.importlib.import_module", missing
    )
    with pytest.raises(ConfigError, match=r"py-envtok\[molecular\] DScribe"):
        rep.featurize(_record(), rep.RepresentationSettings(species=(1, 8)))
    with pytest.raises(ConfigError, match=r"py-envtok\[molecular\] RDKit"):
        rep.featurize(_record(), _radial(allow_validated_perception=True))


def test_soap_configuration_and_backend_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MagicMock()
    backend.create.return_value = np.ones((3, 4))
    constructor = MagicMock(return_value=backend)
    monkeypatch.setattr(
        "envtok.molecular.representations.importlib.import_module",
        lambda _: type("Descriptors", (), {"SOAP": constructor}),
    )
    record = _record()
    features = rep.featurize(record, rep.RepresentationSettings(species=(1, 8)))
    assert features.descriptors.shape == (3, 4)
    constructor.assert_called_once_with(
        species=[1, 8],
        periodic=False,
        r_cut=5.0,
        n_max=4,
        l_max=4,
        sigma=0.5,
        average="off",
        sparse=False,
        dtype="float64",
    )
    backend.create.assert_called_once_with(record.atoms, centers=None)
    backend.create.return_value = np.ones((1, 4))
    with pytest.raises(ConfigError, match="one row"):
        rep.featurize(record, rep.RepresentationSettings(species=(1, 8)))


def test_perception_failures_and_unsupported_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _radial(allow_validated_perception=True)
    changes: list[dict[str, Any]] = [
        {"total_charge": None},
        {"total_charge": 0.5},
        {"spin_multiplicity": 2},
    ]
    for change in changes:
        assert (
            rep.featurize(replace(_record(), **change), settings).graph_status
            == "fallback_perception_unsupported"
        )
    metal = _record(Atoms("Fe", positions=[[0, 0, 0]]))
    assert (
        rep.featurize(metal, replace(settings, species=(26,))).graph_status
        == "fallback_perception_unsupported"
    )
    monkeypatch.setattr(rep, "_inferred_graph", lambda _: _water_graph())
    assert rep.featurize(_record(), settings).graph_status == "inferred_rdkit"

    def broken(_: MolecularRecord) -> None:
        raise ValueError("ambiguous valence")

    monkeypatch.setattr(rep, "_inferred_graph", broken)
    assert "perception_failed" in rep.featurize(_record(), settings).graph_status


def test_perception_invalid_xyz_and_radicals(monkeypatch: pytest.MonkeyPatch) -> None:
    chem = MagicMock()
    determine = MagicMock()
    monkeypatch.setattr(
        "envtok.molecular.representations.importlib.import_module",
        lambda name: determine if name.endswith("rdDetermineBonds") else chem,
    )
    chem.MolFromXYZBlock.return_value = None
    assert rep._inferred_graph(_record()) is None
    atom = MagicMock()
    atom.GetNumRadicalElectrons.return_value = 1
    chem.MolFromXYZBlock.return_value = MagicMock()
    chem.MolFromXYZBlock.return_value.GetAtoms.return_value = [atom]
    assert rep._inferred_graph(_record()) is None
    determine.DetermineBonds.assert_called_once()
    atom.GetNumRadicalElectrons.return_value = 0
    atom.GetFormalCharge.return_value = 0
    atom.HasProp.return_value = True
    atom.GetProp.return_value = "R"
    atom.GetIsAromatic.return_value = False
    atom.IsInRing.return_value = False
    other = MagicMock()
    other.GetNumRadicalElectrons.return_value = 0
    other.GetFormalCharge.return_value = 0
    other.HasProp.return_value = False
    other.GetIsAromatic.return_value = False
    other.IsInRing.return_value = False
    bond = MagicMock()
    bond.GetBeginAtomIdx.return_value = 0
    bond.GetEndAtomIdx.return_value = 1
    bond.GetBondTypeAsDouble.return_value = 1.0
    bond.GetStereo.return_value = "STEREONONE"
    chem.MolFromXYZBlock.return_value.GetAtoms.return_value = [atom, other]
    chem.MolFromXYZBlock.return_value.GetBonds.return_value = [bond]
    assert rep._inferred_graph(_record()) == {
        "bonds": [[0, 1, 1.0, "STEREONONE"]],
        "formal_charges": [0, 0],
        "radicals": [0, 0],
        "stereochemistry": ["R", None],
        "aromatic": [False, False],
        "rings": [False, False],
    }


@pytest.mark.parametrize(
    "value", [6.6760000000000005e-05, -6.590999999999999e-05, 3.45e-06, 5e-324, 1.5e20]
)
def test_xyz_coordinates_avoid_exponents_and_round_trip(value: float) -> None:
    text = rep._xyz_coordinate(value)
    assert "e" not in text.lower()
    assert float(text) == value


@pytest.mark.parametrize("value", [0.0, -0.0, 1e-4, 0.119262, -0.763239, 12.5])
def test_xyz_coordinates_keep_existing_positional_text(value: float) -> None:
    assert rep._xyz_coordinate(value) == f"{value:.17g}"


def test_real_rdkit_perceives_near_zero_coordinates() -> None:
    pytest.importorskip("rdkit")
    atoms = molecule("H2O")
    atoms.positions[:, 0] += 3.45e-06
    record = _record(atoms)
    graph = rep._inferred_graph(record)
    assert graph is not None
    assert sorted(sorted(bond[:2]) for bond in graph["bonds"]) == [[0, 1], [0, 2]]
    settings = _radial(allow_validated_perception=True)
    assert rep.featurize(record, settings).graph_status == "inferred_rdkit"


def test_real_soap_and_rdkit_optional() -> None:
    pytest.importorskip("dscribe")
    pytest.importorskip("rdkit")
    record = _record()
    settings = rep.RepresentationSettings(
        species=(1, 8), allow_validated_perception=True
    )
    first = rep.featurize(record, settings)
    assert first.graph_status == "inferred_rdkit"
    atoms = record.atoms.copy()
    atoms.rotate(44, (2, 1, 7))
    second = rep.featurize(
        replace(record, record=replace(record.record, atoms=atoms)), settings
    )
    np.testing.assert_allclose(first.descriptors, second.descriptors, atol=1e-7)
    assert first.token_hierarchy == second.token_hierarchy


def test_real_soap_detects_butane_conformers_with_identical_graphs() -> None:
    pytest.importorskip("dscribe")
    pytest.importorskip("rdkit")
    record = _record(molecule("trans-butane"), family_id="butane")
    graph = rep._inferred_graph(record)
    assert graph is not None
    record = replace(record, graph=graph, graph_status="supplied")
    settings = rep.RepresentationSettings(species=(1, 6))
    trans = rep.featurize(record, settings)
    moving = {2}
    for _ in range(record.natoms):
        moving.update(
            j if i in moving else i
            for i, j, *_ in graph["bonds"]
            if (i in moving or j in moving) and {i, j} != {1, 2}
        )
    atoms = record.atoms.copy()
    atoms.set_dihedral(0, 1, 2, 3, 60, indices=sorted(moving))
    gauche = rep.featurize(
        replace(record, record=replace(record.record, atoms=atoms)), settings
    )
    assert trans.token_hierarchy == gauche.token_hierarchy
    assert np.linalg.norm(trans.descriptors - gauche.descriptors) > 0.05


def test_fragment_separation_beyond_local_cutoff() -> None:
    atoms = Atoms("HH", positions=[[0, 0, 0], [8, 0, 0]], info={"fragment_ids": [0, 1]})
    record = _record(atoms)
    settings = rep.RepresentationSettings(backend="radial", species=(1,))
    close = rep.featurize(record, settings)
    atoms.positions[1, 0] = 11
    far = rep.featurize(record, settings)
    assert close.token_hierarchy == far.token_hierarchy
    np.testing.assert_array_equal(close.descriptors, far.descriptors)
    assert close.safeguard_targets != far.safeguard_targets
    atoms.info.pop("fragment_ids")
    atoms.new_array("fragment_ids", np.array([0, 1]))
    assert rep.featurize(record, settings).safeguard_targets == far.safeguard_targets


def test_distinct_integer_and_string_fragment_ids_are_not_merged() -> None:
    atoms = Atoms(
        "HH", positions=[[0, 0, 0], [8, 0, 0]], info={"fragment_ids": [1, "1"]}
    )
    features = rep.featurize(
        _record(atoms), rep.RepresentationSettings(backend="radial", species=(1,))
    )
    assert len(features.safeguard_targets) == 1


@pytest.mark.parametrize("fragments", [[0], 5, [[], []]])
def test_invalid_fragment_maps(fragments: Any) -> None:
    atoms = Atoms(
        "HH", positions=[[0, 0, 0], [8, 0, 0]], info={"fragment_ids": fragments}
    )
    with pytest.raises(ConfigError, match="fragment_ids"):
        rep.featurize(
            _record(atoms), rep.RepresentationSettings(backend="radial", species=(1,))
        )


def test_periodic_torsion_targets_and_symmetry() -> None:
    atoms = Atoms("CCCC", positions=[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]])
    atoms.info["torsions"] = [{"id": "central", "atoms": [0, 1, 2, 3], "symmetry": 2}]
    record = _record(atoms, family_id="butane")
    settings = rep.RepresentationSettings(backend="radial", species=(6,))
    initial = rep.featurize(record, settings)
    atoms.set_dihedral(0, 1, 2, 3, 180)
    equivalent = rep.featurize(record, settings)
    assert initial.safeguard_targets == equivalent.safeguard_targets
    atoms.set_dihedral(0, 1, 2, 3, 90)
    rotated = rep.featurize(record, settings)
    assert initial.safeguard_targets != rotated.safeguard_targets


@pytest.mark.parametrize(
    "torsions",
    [
        1,
        [1],
        [{"atoms": [0, 1, 2, 3]}],
        [{"id": "t", "atoms": [0, 1, 2]}],
        [{"id": "t", "atoms": [0, 1, 2, 2]}],
        [{"id": "t", "atoms": [0, 1, 2, -1]}],
        [{"id": "t", "atoms": [0, 1, 2, []]}],
        [{"id": "t", "atoms": [0, 1, 2, 3], "symmetry": 0}],
    ],
)
def test_invalid_torsion_metadata(torsions: Any) -> None:
    atoms = Atoms("CCCC", positions=np.zeros((4, 3)), info={"torsions": torsions})
    with pytest.raises(ConfigError, match="torsion"):
        rep.featurize(
            _record(atoms, family_id="family"),
            rep.RepresentationSettings(backend="radial", species=(6,)),
        )


def test_torsions_require_matching_families_and_defined_geometry() -> None:
    atoms = Atoms(
        "CCCC",
        positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
        info={"torsions": [{"id": "t", "atoms": [0, 1, 2, 3]}]},
    )
    settings = rep.RepresentationSettings(backend="radial", species=(6,))
    with pytest.raises(ConfigError, match="family_id"):
        rep.featurize(_record(atoms), settings)
    with pytest.raises(ConfigError, match="undefined"):
        rep.featurize(_record(atoms, family_id="family"), settings)


def test_parallel_featurisation_matches_serial() -> None:
    atoms = [molecule(name) for name in ("H2O", "CH4", "NH3", "C2H6", "CH3OH")]
    records = [
        replace(_record(item), record=replace(_record(item).record, record_id=f"r{i}"))
        for i, item in enumerate(atoms * 2)
    ]
    settings = rep.RepresentationSettings(backend="radial")
    serial = rep.featurize_records(records, settings)
    parallel = rep.featurize_records(records, settings, workers=2)
    assert len(parallel) == len(records)
    for left, right in zip(serial, parallel, strict=True):
        assert left.record_id == right.record_id
        assert left.token_hierarchy == right.token_hierarchy
        assert left.signatures == right.signatures
        assert np.array_equal(left.descriptors, right.descriptors)


def test_registry_merge_rejects_conflicting_payloads() -> None:
    registry = rep.SignatureRegistry()
    token = registry.register({"a": 1}, "graph")
    registry.merge({token: registry.signatures[token]})
    with pytest.raises(ConfigError, match="digest collision"):
        registry.merge({token: '{"a":2}'})


def test_materialized_features_share_repeated_signature_text() -> None:
    record = _record()
    first, second = rep.featurize_records([record, record], _radial())
    assert first.signatures is not second.signatures
    assert all(first.signatures[t] is second.signatures[t] for t in first.signatures)


@pytest.mark.parametrize("workers", [1, 2])
def test_stream_rejects_a_token_collision_between_records(
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    # Radius zero and one atom give each record a single token. The conflict
    # is observable only after the first record's local registry is gone.
    digest = MagicMock()
    digest.hexdigest.return_value = "constant"
    monkeypatch.setattr(rep, "sha256", lambda _: digest)
    settings = _radial(graph_radius=0)
    hydrogen = _record(Atoms("H"))
    oxygen = _record(Atoms("O"))
    # Run the worker function locally so it sees the forced hash collision.
    # With three records and two workers, each submitted batch has one record.
    pool = MagicMock()
    pool.return_value.__enter__.return_value.submit.side_effect = lambda func, *args: (
        MagicMock(result=lambda: func(*args))
    )
    monkeypatch.setattr(rep, "ProcessPoolExecutor", pool)
    with pytest.raises(ConfigError, match="digest collision"):
        rep._featurize_batch([hydrogen, oxygen], settings)
    features = rep.iter_features(
        [hydrogen, hydrogen, oxygen], settings, workers=workers
    )
    assert next(features).signatures == next(features).signatures
    with pytest.raises(ConfigError, match="digest collision"):
        next(features)


def test_worker_batches_match_serial_features() -> None:
    records = [
        replace(_record(item), record=replace(_record(item).record, record_id=f"r{i}"))
        for i, item in enumerate([molecule("H2O"), molecule("CH4")])
    ]
    settings = rep.RepresentationSettings(backend="radial", species=(1, 6, 8))
    records[0].atoms.info["fragment_ids"] = [0, 0, 1]
    batch = rep._featurize_batch(records, settings)
    serial = rep.featurize_records(records, settings)
    assert serial[0].safeguard_targets
    assert batch.signatures == {t: s for f in serial for t, s in f.signatures.items()}
    assert [f.safeguard_targets for f in batch] == [f.safeguard_targets for f in serial]
    assert [f.token_hierarchy for f in batch] == [f.token_hierarchy for f in serial]
    assert [f.signatures for f in batch] == [f.signatures for f in serial]
