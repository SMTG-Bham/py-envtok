"""Frozen profile settings, corpus presets, records and validation."""

import json
from dataclasses import fields

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from helpers import FIXTURES

from envtok.errors import ConfigError
from envtok.profiles import (
    OMAT24_STUDY_EXCLUDED,
    REFERENCE_723696D,
    REFERENCE_LADDER,
    REFERENCE_PRESENCE_FAMILIES,
    SIGNATURE_FIELDS,
    InvalidRecordPolicy,
    corpus_policy,
    get_profile,
)
from envtok.records import (
    IssueSeverity,
    Label,
    SourceRef,
    Split,
    StructureRecord,
    scalar_metadata,
    validate_record,
)

DEFAULTS = json.loads((FIXTURES / "reference" / "defaults-723696d.json").read_text())


class TestLadder:
    def test_modifier_outer_rule_generates_33_named_rungs(self) -> None:
        names = [rung.name for rung in REFERENCE_LADDER]
        assert len(names) == 33
        assert names[:3] == ["R0_full", "R1dA_drop_f5", "R2dA_drop_spin"]
        assert names[10] == "R8dA_elem"
        assert names[11] == "R0A"
        assert names[21] == "R8A_elem"
        assert names[22] == "R0"
        assert names[-1] == "R8_elem"
        assert REFERENCE_LADDER[0].fields == (
            "elem",
            "spin",
            "nb",
            "poly",
            "f5",
            "dmin",
            "a_bin",
        )
        assert len(set(names)) == 33

    def test_every_rung_keeps_the_element_and_known_fields(self) -> None:
        groups = [rung.fields for rung in REFERENCE_LADDER]
        groups += [family.fields for family in REFERENCE_PRESENCE_FAMILIES]
        for names in groups:
            assert names[0] == "elem"
            assert set(names) <= set(SIGNATURE_FIELDS)

    def test_coordination_dropped_only_at_r5_and_r8(self) -> None:
        dropping = [
            rung.name
            for rung in REFERENCE_LADDER
            if "cn_sig" not in rung.fields and "nb" not in rung.fields
        ]
        assert dropping == [
            "R5dA_nbset",
            "R8dA_elem",
            "R5A_nbset",
            "R8A_elem",
            "R5_nbset",
            "R8_elem",
        ]

    def test_presence_families_in_reference_order(self) -> None:
        assert [f.name for f in REFERENCE_PRESENCE_FAMILIES] == [
            "F_elem_poly",
            "F_elem_cn",
            "F_elem_cn_dmin",
            "F_elem_cn_abin",
        ]


class TestFrozenSettings:
    def test_geometry_settings_equal_reference_library_defaults(self) -> None:
        geometry = REFERENCE_723696D.geometry
        compute = DEFAULTS["LocalGeometryFinder.compute_structure_environments"]
        strategy = DEFAULTS["SimplestChemenvStrategy.__init__"]
        finder = DEFAULTS["LocalGeometryFinder.__init__"]
        sga = DEFAULTS["SpacegroupAnalyzer.__init__"]
        assert (
            float(compute["maximum_distance_factor"])
            == geometry.maximum_distance_factor
        )
        assert float(compute["minimum_angle_factor"]) == geometry.minimum_angle_factor
        assert float(compute["voronoi_normalized_distance_tolerance"]) == (
            geometry.voronoi_normalized_distance_tolerance
        )
        assert float(compute["voronoi_normalized_angle_tolerance"]) == (
            geometry.voronoi_normalized_angle_tolerance
        )
        assert int(compute["optimization"]) == geometry.optimization
        assert int(strategy["additional_condition"]) == geometry.additional_condition
        assert float(strategy["continuous_symmetry_measure_cutoff"]) == (
            geometry.continuous_symmetry_measure_cutoff
        )
        assert strategy["symmetry_measure_type"] == repr(geometry.symmetry_measure_type)
        assert finder["permutations_safe_override"] == "False"
        assert finder["plane_ordering_override"] == "True"
        assert float(sga["symprec"]) == geometry.symprec
        assert float(sga["angle_tolerance"]) == geometry.angle_tolerance
        crystalnn = DEFAULTS["CrystalNN.__init__"]
        for item in fields(geometry.crystalnn):
            value = getattr(geometry.crystalnn, item.name)
            expected = eval(crystalnn[item.name])  # noqa: S307 - trusted fixture repr
            assert value == (tuple(expected) if isinstance(value, tuple) else expected)
        voronoi = DEFAULTS["VoronoiNN.__init__"]
        assert float(voronoi["cutoff"]) == geometry.voronoi.cutoff
        assert voronoi["weight"] == repr(geometry.voronoi.weight)

    def test_get_profile(self) -> None:
        assert get_profile("reference-723696d") is REFERENCE_723696D
        assert get_profile(REFERENCE_723696D) is REFERENCE_723696D
        with pytest.raises(ConfigError, match="unknown profile"):
            get_profile("paper-v1")


class TestCorpusPolicy:
    def test_general_preset_excludes_nothing(self) -> None:
        policy = corpus_policy()
        assert policy.excluded_symbols(["La", "U", "Og"]) == []
        assert policy.invalid_record_policy is InvalidRecordPolicy.QUARANTINE

    def test_study_preset_is_exactly_la_lu_and_ac_lr(self) -> None:
        policy = corpus_policy("omat24-study", invalid_record_policy="reference")
        assert len(OMAT24_STUDY_EXCLUDED) == 30
        assert policy.excluded_symbols(["La", "Lr", "Rf", "O"]) == ["La", "Lr"]
        assert policy.invalid_record_policy is InvalidRecordPolicy.REFERENCE

    def test_explicit_exclusions(self) -> None:
        policy = corpus_policy(exclude_elements=["Pb"], exclude_z_at_least=89)
        assert policy.excluded_symbols(["Pb", "Ac", "Rf", "Bi"]) == ["Ac", "Pb", "Rf"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"preset": "nope"}, "unknown corpus preset"),
            ({"exclude_elements": ["Qq"]}, "Qq"),
        ],
    )
    def test_invalid_policies(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            corpus_policy(**kwargs)  # type: ignore[arg-type]


def record(atoms: Atoms, **labels: Label) -> StructureRecord:
    return StructureRecord(
        record_id="t:1",
        source_ref=SourceRef("t", "1", "s", "0"),
        atoms=atoms,
        labels=labels,
    )


class TestRecords:
    @pytest.mark.parametrize(
        ("text", "split"),
        [("train", Split.TRAIN), ("val", Split.VALIDATION), (" TEST ", Split.TEST)],
    )
    def test_split_parsing(self, text: str, split: Split) -> None:
        assert Split.parse(text) is split

    def test_unknown_split(self) -> None:
        with pytest.raises(ConfigError, match="unknown split"):
            Split.parse("holdout")

    def test_valid_record_has_no_issues(self) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        rec = record(
            nacl,
            energy=Label(np.array(-1.0), "eV", "calc:energy"),
            forces=Label(np.zeros((2, 3)), "eV/Ang", "calc:forces"),
        )
        assert validate_record(rec) == []
        assert rec.natoms == 2
        assert rec.metadata_json() == "{}"

    def test_invalid_records_report_actionable_issues(self) -> None:
        assert [i.code for i in validate_record(record(Atoms()))] == ["empty_structure"]

        nacl = bulk("NaCl", "rocksalt", a=5.64)
        bad = nacl.copy()
        bad.positions[0, 0] = np.nan
        bad.numbers[1] = 0
        codes = {i.code for i in validate_record(record(bad))}
        assert codes == {"nonfinite_positions", "invalid_species"}

        slab = nacl.copy()
        slab.pbc = (True, True, False)
        (issue,) = validate_record(record(slab))
        assert (issue.code, issue.severity) == (
            "unsupported_periodicity",
            IssueSeverity.EXCLUDE,
        )
        assert validate_record(record(slab), require_periodic=False) == []

        flat = nacl.copy()
        flat.cell[2] = 0.0
        assert [i.code for i in validate_record(record(flat))] == ["degenerate_cell"]
        broken = nacl.copy()
        broken.cell[2, 2] = np.inf
        assert [i.code for i in validate_record(record(broken))] == ["nonfinite_cell"]

    def test_moment_validation(self) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        vector = nacl.copy()
        vector.set_initial_magnetic_moments(np.ones((2, 3)))
        (issue,) = validate_record(record(vector))
        assert issue.code == "noncollinear_moments"
        assert "magnetic_policy: disabled" in issue.message
        nan = nacl.copy()
        nan.set_initial_magnetic_moments([np.nan, 0.0])
        assert [i.code for i in validate_record(record(nan))] == ["nonfinite_moments"]

    def test_label_warnings(self) -> None:
        nacl = bulk("NaCl", "rocksalt", a=5.64)
        rec = record(
            nacl,
            forces=Label(np.zeros((3, 3)), "eV/Ang", "calc:forces"),
            energy=Label(np.array(np.nan), "eV", "calc:energy"),
            custom=Label(np.zeros(4), "x", "info:custom"),
        )
        issues = validate_record(rec)
        assert [(i.code, i.severity) for i in issues] == [
            ("nonfinite_label", IssueSeverity.WARN),
            ("label_shape", IssueSeverity.WARN),
        ]

    def test_scalar_metadata(self) -> None:
        info = {
            "charge": np.int64(-1),
            "method": "PBE",
            "temperature": 300.0,
            "bad": float("nan"),
            "array": np.zeros(3),
            "flag": True,
            "none": None,
            "skip": 1,
        }
        assert scalar_metadata(info, skip=frozenset({"skip"})) == {
            "charge": -1,
            "flag": True,
            "method": "PBE",
            "none": None,
            "temperature": 300.0,
        }
