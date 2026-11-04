"""Fallback and failure paths of the geometry port, exercised with fakes."""

from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from helpers import record_of

from envtok.profiles import REFERENCE_723696D, MagneticPolicy, corpus_policy
from envtok.tokenizer import geometry
from envtok.tokenizer.geometry import GeometryEngine, NeighbourSource, SiteStatus
from envtok.tokenizer.pipeline import (
    RecordStatus,
    geometry_structure,
    site_token,
    stable_seed,
    tokenize_record,
)

PROFILE = REFERENCE_723696D


def nacl_structure() -> Any:
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    return geometry_structure(atoms, np.zeros(len(atoms)))


def fake_site(symbol: str, coords: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        coords=np.array(coords), specie=SimpleNamespace(symbol=symbol)
    )


def fake_environments(*sets: list[SimpleNamespace] | None) -> SimpleNamespace:
    neighbour_sets = [
        None if s is None else [SimpleNamespace(neighb_sites=s)] for s in sets
    ]
    return SimpleNamespace(neighbors_sets=neighbour_sets)


class Raises:
    def __getitem__(self, index: int) -> None:
        raise IndexError(index)


@pytest.fixture
def structure() -> Any:
    return nacl_structure()


class TestSymmetry:
    def test_disabled_symmetry_analyses_every_site(self, structure: Any) -> None:
        settings = replace(PROFILE.geometry, use_symmetry=False)
        assert geometry.symmetry_representatives(structure, settings) == (
            [0, 1],
            "disabled",
            None,
        )

    @pytest.mark.parametrize(
        "dataset",
        [None, SimpleNamespace(equivalent_atoms=[0]), RuntimeError("spglib")],
    )
    def test_failures_fall_back_to_all_sites(
        self, structure: Any, dataset: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeAnalyzer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def get_symmetry_dataset(self) -> object:
                if isinstance(dataset, Exception):
                    raise dataset
                return dataset

        monkeypatch.setattr(geometry, "SpacegroupAnalyzer", FakeAnalyzer)
        reps, status, event = geometry.symmetry_representatives(
            structure, PROFILE.geometry
        )
        assert (reps, status) == ([0, 1], "fallback")
        assert event is not None
        assert event.stage == "symmetry"
        assert event.fallback == "all_sites"


class TestNeighbourSelection:
    def test_missing_environments_return_default(
        self, engine: GeometryEngine, structure: Any
    ) -> None:
        assert engine.select_neighbour_set(None, None, structure, 0) == (None, None)
        default = fake_environments([])
        assert engine.select_neighbour_set(default, None, structure, 0) == (
            default,
            None,
        )

    def test_indexing_failure_returns_default_with_event(
        self, engine: GeometryEngine, structure: Any
    ) -> None:
        default = SimpleNamespace(neighbors_sets=Raises())
        chosen, event = engine.select_neighbour_set(default, default, structure, 0)
        assert chosen is default
        assert event is not None
        assert event.stage == "neighbour_selection"

    def test_empty_extended_set_or_nothing_added_returns_default(
        self, engine: GeometryEngine, structure: Any
    ) -> None:
        shell = [fake_site("Cl", [2.82, 0, 0])]
        default = fake_environments(shell)
        assert engine.select_neighbour_set(
            default, fake_environments(None), structure, 0
        ) == (default, None)
        assert engine.select_neighbour_set(
            default, fake_environments(shell), structure, 0
        ) == (default, None)

    @pytest.mark.parametrize(
        ("weight", "extended_chosen"), [(0.44, False), (0.45, True)]
    )
    def test_prominence_threshold_decides(
        self,
        engine: GeometryEngine,
        structure: Any,
        weight: float,
        extended_chosen: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        near = fake_site("Cl", [2.82, 0, 0])
        far = fake_site("Na", [3.99, 0, 0])
        default = fake_environments(None)  # no default set: every contact is "added"
        extended = fake_environments([near, far])

        def nn_info(_structure: object, _index: int) -> list[dict[str, object]]:
            return [
                {"site": far, "weight": weight},
                {"site": far, "weight": 0.1},
                {"site": near, "weight": 0.2},
            ]

        monkeypatch.setattr(engine.voronoi, "get_nn_info", nn_info)
        chosen, event = engine.select_neighbour_set(default, extended, structure, 0)
        assert event is None
        assert (chosen is extended) is extended_chosen

    def test_voronoi_failure_returns_default(
        self, engine: GeometryEngine, structure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object) -> None:
            raise ValueError("voronoi")

        monkeypatch.setattr(engine.voronoi, "get_nn_info", boom)
        default = fake_environments([fake_site("Cl", [2.82, 0, 0])])
        extended = fake_environments(
            [fake_site("Cl", [2.82, 0, 0]), fake_site("Na", [4, 0, 0])]
        )
        chosen, event = engine.select_neighbour_set(default, extended, structure, 0)
        assert chosen is default
        assert event is not None
        assert event.category == "ValueError"


@pytest.mark.filterwarnings("ignore::UserWarning")
class TestDescriptor:
    def test_chemenv_without_environment_uses_crystalnn_rescue(
        self, engine: GeometryEngine, structure: Any
    ) -> None:
        environments = SimpleNamespace(
            coordination_environments=[None, None],
            neighbors_sets=[[SimpleNamespace()], None],
        )
        site, event = engine.describe_site(
            environments, NeighbourSource.CHEMENV_DEFAULT, structure, 0
        )
        assert event is None
        assert site.source is NeighbourSource.CRYSTALNN
        assert site.status is SiteStatus.RECOVERED
        assert site.geometry == "UNKNOWN:6"
        assert site.csm is None
        assert site.neighbours == (("Cl", 6),)

    def test_crystalnn_failure_gives_invalid_site(
        self, engine: GeometryEngine, structure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object) -> None:
            raise RuntimeError("cnn")

        monkeypatch.setattr(engine.crystalnn, "get_nn_info", boom)
        site, event = engine.describe_site(None, NeighbourSource.NONE, structure, 0)
        assert site is geometry.INVALID_SITE
        assert event is not None
        assert (event.stage, event.fallback) == ("crystalnn", "invalid_token")

    def test_crystalnn_without_neighbours_gives_invalid_site(
        self, engine: GeometryEngine, structure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(engine.crystalnn, "get_nn_info", lambda *_a: [])
        site, event = engine.describe_site(None, NeighbourSource.NONE, structure, 1)
        assert site.status is SiteStatus.INVALID
        assert event is not None
        assert event.category == "no_neighbours"

    def test_chemenv_failure_is_recorded(
        self, engine: GeometryEngine, structure: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("chemenv")

        monkeypatch.setattr(engine.finder, "compute_structure_environments", boom)
        analysis = engine.analyse(structure)
        assert analysis.chemenv_status == "failed"
        assert [e.stage for e in analysis.events] == ["chemenv"]
        assert any("oxidation" in key for key in analysis.warnings)
        assert {s.source for s in analysis.sites.values()} == {
            NeighbourSource.CRYSTALNN
        }


def corpus_atoms(corpus_frames: list[Atoms], name: str) -> Atoms:
    atoms: Atoms = next(a for a in corpus_frames if a.info["fixture"] == name)
    copy: Atoms = atoms.copy()
    return copy


class TestTokenizeRecord:
    def test_irrecoverable_sites_become_invalid_tokens(
        self,
        engine: GeometryEngine,
        corpus_frames: list[Atoms],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*_args: object) -> None:
            raise RuntimeError("cnn")

        monkeypatch.setattr(engine.crystalnn, "get_nn_info", boom)
        sheet = corpus_atoms(corpus_frames, "graphene_vacuum")
        result = tokenize_record(record_of(sheet), PROFILE, corpus_policy(), engine)
        assert result.structure.status == RecordStatus.OK
        assert result.structure.n_invalid_sites == 2
        keys = {site_token(s, PROFILE).key() for s in result.sites}
        assert keys == {"C_0_INVALID_INVALID_999_999_999"}
        assert {f.stage for f in result.failures} == {"chemenv", "crystalnn"}
        assert not any(f.fatal for f in result.failures)

    def test_recovered_sites_and_warnings_are_recorded(
        self, tokenized_corpus: dict[str, Any]
    ) -> None:
        sheet = tokenized_corpus["fx-018-graphene_vacuum"]
        assert sheet.structure.n_recovered_sites == 2
        assert sheet.structure.chemenv_status == "failed"
        assert "UserWarning" in sheet.structure.warnings_json
        cubic = tokenized_corpus["fx-009-srtio3_cubic"]
        assert cubic.structure.n_extended_sites == 3
        assert cubic.structure.n_representatives == 3

    @pytest.mark.parametrize(
        ("preset", "status", "reason"),
        [("omat24-study", RecordStatus.EXCLUDED, "excluded_elements")],
    )
    def test_corpus_exclusion_happens_before_geometry(
        self, engine: GeometryEngine, preset: str, status: str, reason: str
    ) -> None:
        atoms = bulk("LaN", "rocksalt", a=5.3)
        result = tokenize_record(
            record_of(atoms), PROFILE, corpus_policy(preset), engine
        )
        assert (result.structure.status, result.structure.status_reason) == (
            status,
            reason,
        )
        assert result.sites == []
        assert "La" in result.failures[0].message

    def test_validation_outcomes(self, engine: GeometryEngine) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        slab = nacl.copy()
        slab.pbc = (True, True, False)
        excluded = tokenize_record(record_of(slab), PROFILE, corpus_policy(), engine)
        assert excluded.structure.status == RecordStatus.EXCLUDED
        assert excluded.structure.status_reason == "unsupported_periodicity"

        flat = nacl.copy()
        flat.cell[2] = 0.0
        failed = tokenize_record(record_of(flat), PROFILE, corpus_policy(), engine)
        assert failed.structure.status == RecordStatus.FAILED
        assert failed.structure.cell_volume is None
        assert failed.failures[0].fatal

        empty = tokenize_record(record_of(Atoms()), PROFILE, corpus_policy(), engine)
        assert empty.structure.formula == ""
        assert empty.structure.status_reason == "empty_structure"

    def test_noncollinear_moments_fail_unless_moments_disabled(
        self, engine: GeometryEngine
    ) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        nacl.set_initial_magnetic_moments(np.ones((2, 3)))
        failed = tokenize_record(record_of(nacl), PROFILE, corpus_policy(), engine)
        assert failed.structure.status_reason == "noncollinear_moments"
        geometry_only = replace(PROFILE, magnetic_policy=MagneticPolicy.DISABLED)
        result = tokenize_record(
            record_of(nacl), geometry_only, corpus_policy(), engine
        )
        assert result.structure.status == RecordStatus.OK
        assert result.structure.moment_source == "disabled"
        assert {site_token(s, geometry_only).key()[:5] for s in result.sites} == {
            "Na_NA",
            "Cl_NA",
        }

    def test_labels_rng_and_neighbour_rows(self, engine: GeometryEngine) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        from ase.calculators.singlepoint import SinglePointCalculator  # noqa: PLC0415

        nacl.calc = SinglePointCalculator(
            nacl,
            energy=-2.0,
            forces=[[3.0, 4.0, 0.0], [0, 0, 0]],
            stress=np.ones(6) * -2,
        )
        before = np.random.get_state(legacy=False)["state"]["key"].copy()  # noqa: NPY002
        result = tokenize_record(
            record_of(nacl), PROFILE, corpus_policy(), engine, keep_neighbours=False
        )
        after = np.random.get_state(legacy=False)["state"]["key"]  # noqa: NPY002
        assert np.array_equal(after, before)
        row = result.structure
        assert (row.energy, row.fmax, row.stress_max) == (-2.0, 5.0, 2.0)
        assert (row.has_energy, row.has_forces, row.has_stress) == (True, True, True)
        assert result.neighbours == []
        assert stable_seed("a") == stable_seed("a") != stable_seed("b")

    def test_label_warnings_are_not_fatal(self, engine: GeometryEngine) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        nacl.info["REF_forces"] = np.zeros((0, 3))
        record = record_of(nacl, label_sources={"forces": "info:REF_forces"})
        result = tokenize_record(record, PROFILE, corpus_policy(), engine)
        assert result.structure.status == RecordStatus.OK
        assert result.structure.fmax is None
        assert [f.category for f in result.failures] == ["label_shape"]


def test_geometry_events_iterate(structure: Any, engine: GeometryEngine) -> None:
    analysis = engine.analyse(structure)
    events: Iterator[object] = iter(analysis.events)
    assert list(events) == []
    assert analysis.symmetry_status == "ok"
