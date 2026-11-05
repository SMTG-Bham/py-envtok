"""Token encoding, signature fields, binning and raw measurements."""

import math

import numpy as np
import pytest

from envtok.errors import UnsupportedInputError
from envtok.profiles import CORDERO_2008, REFERENCE_723696D, MagneticPolicy, table_drift
from envtok.profiles.radii import RadiusTable
from envtok.tokenizer.bins import (
    csm_bin,
    csm_bins,
    digitize,
    digitize_array,
    moment_bin,
    moment_bins,
)
from envtok.tokenizer.measures import contact_measures, covalent_radius
from envtok.tokenizer.tokens import (
    SiteToken,
    composition,
    multiplicity_bin,
    neighbour_string,
    parse_neighbours,
    signature,
    signature_fields,
)

BINS = REFERENCE_723696D.bins
MEASURES = REFERENCE_723696D.measures


def token(**changes: object) -> SiteToken:
    base: dict[str, object] = {
        "element": "Fe",
        "moment": 0,
        "neighbours": (("Cl", 1), ("O", 3)),
        "geometry": "T:4",
        "csm_bin": 3,
        "dmin_bin": 1,
        "asym_bin": 2,
    }
    base.update(changes)
    return SiteToken(**base)  # type: ignore[arg-type]


class TestTokenEncoding:
    def test_reference_key_matches_the_paper_example(self) -> None:
        assert token().key() == "Fe_0_Cl-O-O-O_T:4_3_1_2"

    @pytest.mark.parametrize(
        "key",
        [
            "Fe_0_Cl-O-O-O_T:4_3_1_2",
            "O_0_O-O-Sr-Sr-Sr-Sr-Ti-Ti_BO_3:8_1_3_1",
            "F_0_Ca-Ca-F_BS_2:10_3_3_0",
            "Cs_0_Cl-Cs_UNKNOWN:14_999_2_0",
            "Fe_0_INVALID_INVALID_999_999_999",
            "Na_NA_Cl_O:6_0_3_0",
            "X_0__O:6_0_999_999",
        ],
    )
    def test_round_trip_is_lossless_including_underscore_geometries(
        self, key: str
    ) -> None:
        parsed = SiteToken.from_key(key)
        assert parsed.key() == key

    def test_underscore_geometry_parsed_from_both_ends(self) -> None:
        parsed = SiteToken.from_key("O_0_O-O-Sr_BO_3:8_1_3_1")
        assert parsed.geometry == "BO_3:8"
        assert (parsed.csm_bin, parsed.dmin_bin, parsed.asym_bin) == (1, 3, 1)

    def test_sentinels_are_typed_missing_values(self) -> None:
        parsed = SiteToken.from_key("Fe_NA_INVALID_INVALID_999_999_999")
        assert parsed.moment is None
        assert parsed.neighbours is None
        assert parsed.coordination_number is None
        assert (parsed.csm_bin, parsed.dmin_bin, parsed.asym_bin) == (None, None, None)

    @pytest.mark.parametrize(
        "key",
        [
            "Fe_0_O_T:4_3_1",
            "Fe_0_O__3_1_2",
            "_0_O_T:4_3_1_2",
            "Fe_x_O_T:4_3_1_2",
            "Fe_0_O-Cl_T:4_3_1_2",
        ],
    )
    def test_malformed_keys_are_rejected(self, key: str) -> None:
        with pytest.raises(UnsupportedInputError):
            SiteToken.from_key(key)

    def test_neighbour_composition_helpers(self) -> None:
        assert composition(["O", "Cl", "O"]) == (("Cl", 1), ("O", 2))
        assert composition(None) is None
        assert neighbour_string((("Cl", 1), ("O", 2))) == "Cl-O-O"
        assert neighbour_string(()) == ""
        assert neighbour_string(None) == "INVALID"
        assert parse_neighbours("") == ()
        assert parse_neighbours("INVALID") is None
        assert token(neighbours=()).coordination_number is None
        assert token().coordination_number == 4


class TestSignatureFields:
    def test_table_s1_example_signatures(self) -> None:
        fields = signature_fields(token())
        assert signature(fields, ("elem", "nb_binned", "poly_reliable", "cn_sig")) == (
            "Fe|Cl:1-O:34|UNCL|4"
        )
        assert signature(fields, ("elem", "nbset", "cn_sig")) == "Fe|Cl-O|4"
        assert fields["poly_reliable"] == "UNCL"

    def test_reliable_geometry_kept_for_low_csm(self) -> None:
        assert signature_fields(token(csm_bin=1))["poly_reliable"] == "T:4"
        assert signature_fields(token(csm_bin=None))["poly_reliable"] == "UNCL"

    def test_invalid_placeholder_cannot_masquerade_as_one_coordinate(self) -> None:
        invalid = signature_fields(
            token(neighbours=None, geometry="INVALID", csm_bin=None)
        )
        genuine = signature_fields(token(neighbours=(("O", 1),), geometry="S:1"))
        empty = signature_fields(token(neighbours=()))
        assert invalid["cn_sig"] == "NA"
        assert empty["cn_sig"] == "NA"
        assert genuine["cn_sig"] == "1"
        assert (invalid["nbset"], invalid["nb_binned"]) == ("INVALID", "INVALID:1")
        assert (empty["nbset"], empty["nb_binned"]) == ("", ":1")

    @pytest.mark.parametrize(
        ("count", "label"),
        [(1, "1"), (2, "2"), (3, "34"), (4, "34"), (5, "56"), (6, "56"), (7, "7p")],
    )
    def test_multiplicity_bins(self, count: int, label: str) -> None:
        assert multiplicity_bin(count) == label

    def test_disabled_moment_field(self) -> None:
        assert signature_fields(token(moment=None))["spin"] == "NA"


class TestBins:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, 0),
            (np.nextafter(1.0, 2.0), 1),
            (5.0, 1),
            (np.nextafter(5.0, 6.0), 2),
            (np.nextafter(10.0, 0.0), 2),
            (10.0, 3),
            (float("nan"), 3),
            (None, None),
        ],
    )
    def test_csm_bin_edges(self, value: float | None, expected: int | None) -> None:
        assert csm_bin(value, BINS) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.nextafter(0.6, 0.0), 0),
            (0.6, 1),
            (np.nextafter(0.7, 0.0), 1),
            (0.7, 2),
            (0.85, 3),
            (np.nextafter(1.15, 0.0), 3),
            (1.15, 4),
            (None, None),
            (float("nan"), None),
        ],
    )
    def test_dmin_bin_edges(self, value: float | None, expected: int | None) -> None:
        assert digitize(value, BINS.dmin_edges) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(np.nextafter(1.0, 0.0), 0), (1.0, 1), (np.nextafter(3.0, 0.0), 1), (3.0, 2)],
    )
    def test_asymmetry_bin_edges(self, value: float, expected: int) -> None:
        assert digitize(value, BINS.asymmetry_edges) == expected

    @pytest.mark.parametrize(
        ("moment", "expected"),
        [(2.3, 2), (-2.6, 3), (2.5, 2), (-3.5, 4), (0.5, 0), (-0.49, 0)],
    )
    def test_moment_uses_absolute_value_and_ties_to_even(
        self, moment: float, expected: int
    ) -> None:
        policy = MagneticPolicy.SOURCE_ABS_ROUND_ZERO_IF_MISSING
        assert moment_bin(moment, policy) == expected
        assert moment_bin(moment, MagneticPolicy.DISABLED) is None

    def test_vectorised_bins_match_scalar_bins(self) -> None:
        rng = np.random.default_rng(0)
        values = np.concatenate([rng.uniform(0, 15, 200), [1.0, 5.0, 10.0, np.nan]])
        defined = np.ones(values.size, dtype=bool)
        defined[-2:] = False
        expected = [
            csm_bin(None if not d else v, BINS)
            for v, d in zip(values, defined, strict=True)
        ]
        result = csm_bins(values, defined, BINS)
        assert [999 if e is None else e for e in expected] == result.tolist()

        ratios = np.concatenate([rng.uniform(0.4, 1.5, 200), [np.nan, 0.6, 1.15]])
        scalar = [digitize(float(r), BINS.dmin_edges) for r in ratios]
        assert [999 if s is None else s for s in scalar] == digitize_array(
            ratios, BINS.dmin_edges
        ).tolist()

        moments = np.array([2.5, -3.5, 0.49, -1.51])
        policy = MagneticPolicy.SOURCE_ABS_ROUND_ZERO_IF_MISSING
        vector = moment_bins(moments, policy)
        assert vector is not None
        assert vector.tolist() == [moment_bin(float(m), policy) for m in moments]
        assert moment_bins(moments, MagneticPolicy.DISABLED) is None


class TestMeasures:
    def test_radius_table_matches_installed_ase(self) -> None:
        assert table_drift() == []
        changed = list(CORDERO_2008.values)
        changed[8] = 0.7
        assert table_drift(RadiusTable("x", tuple(changed), max_z=96)) == [8]
        assert table_drift(RadiusTable("x", CORDERO_2008.values, max_z=120))[-1] == 120

    def test_radius_lookup_rejects_untabulated_elements(self) -> None:
        assert covalent_radius("O", MEASURES) == pytest.approx(0.66)
        assert covalent_radius("Cm", MEASURES) == pytest.approx(1.69)
        assert covalent_radius("Bk", MEASURES) is None  # ASE placeholder, Z = 97
        assert covalent_radius("Xx", MEASURES) is None
        assert CORDERO_2008.radius(0) is None
        broken = RadiusTable("broken", (0.2, float("nan"), -1.0), max_z=2)
        assert broken.radius(1) is None
        assert broken.radius(2) is None

    def test_measures_follow_the_specified_formula(self) -> None:
        center = np.zeros(3)
        contacts = [("O", np.array([1.0, 0.0, 0.0])), ("O", np.array([0.0, 2.0, 0.0]))]
        result = contact_measures("Ti", center, contacts, MEASURES)
        d0 = 1.60 + 0.66
        q = np.array([1.0 / d0, 2.0 / d0])
        weights = np.exp(-(q - 1.0) / 0.1)
        assert result.ratio_min == pytest.approx(q.min())
        assert result.ratio_max == pytest.approx(q.max())
        assert result.asymmetry == pytest.approx(math.hypot(*weights))
        assert result.n_used == 2

    def test_unusable_contacts_are_skipped_and_empty_shells_undefined(self) -> None:
        center = np.zeros(3)
        partial = contact_measures(
            "O",
            center,
            [
                ("Bk", np.array([2.0, 0, 0])),
                ("O", center),
                ("Na", np.array([0, 2.0, 0])),
            ],
            MEASURES,
        )
        assert partial.n_used == 1
        assert partial.ratio_min == pytest.approx(2.0 / (0.66 + 1.66))
        undefined = contact_measures("O", center, [("Bk", np.ones(3))], MEASURES)
        assert (undefined.ratio_min, undefined.asymmetry) == (None, None)
        no_center = contact_measures("Bk", center, [("O", np.ones(3))], MEASURES)
        assert no_center.n_used == 0
