"""The net-charge token field: profile, token grammar, bins and resolution."""

import dataclasses

import numpy as np
import pytest

from envtok.errors import ConfigError, UnsupportedInputError
from envtok.fingerprint import digest, token_payload
from envtok.profiles import (
    NET_CHARGE_LADDER,
    REFERENCE_723696D,
    REFERENCE_LADDER,
    SIGNATURE_FIELDS,
    ChargePolicy,
    with_net_charge,
)
from envtok.tokenizer.bins import charge_bin, charge_bins
from envtok.tokenizer.charge import (
    MAX_ABS_CHARGE,
    ChargeSource,
    resolve_net_charge,
)
from envtok.tokenizer.pipeline import SiteRow, site_token
from envtok.tokenizer.tokens import SiteToken, signature, signature_fields

CHARGED = with_net_charge(REFERENCE_723696D)
LENIENT = with_net_charge(REFERENCE_723696D, "net_round_zero_if_missing")


def site_row(**changes: object) -> SiteRow:
    base: dict[str, object] = {
        "record_id": "fx@v1:a:0",
        "atom_index": 0,
        "element": "Fe",
        "moment": 2.2,
        "rep_index": 0,
        "neighbour_source": "chemenv-default",
        "site_status": "ok",
        "neighbours": "Cl-O-O-O",
        "geometry": "T:4",
        "csm": 6.0,
        "ratio_min": 0.65,
        "ratio_max": 0.9,
        "asymmetry": 3.5,
        "n_contacts": 4,
    }
    base.update(changes)
    return SiteRow(**base)  # type: ignore[arg-type]


class TestLadder:
    def test_net_charge_ladder_inserts_one_identity_level(self) -> None:
        names = [rung.name for rung in NET_CHARGE_LADDER]
        assert len(names) == 36
        assert len(set(names)) == 36
        assert names[:4] == [
            "R0_full",
            "R1dA_drop_f5",
            "R1qdA_drop_charge",
            "R2dA_drop_spin",
        ]
        assert names[12:15] == ["R0A", "R1A_drop_f5", "R1qA_drop_charge"]
        assert names[24:27] == ["R0", "R1_drop_f5", "R1q_drop_charge"]
        assert NET_CHARGE_LADDER[0].fields == (
            "elem",
            "spin",
            "charge",
            "nb",
            "poly",
            "f5",
            "dmin",
            "a_bin",
        )

    def test_charge_is_dropped_first_and_never_restored(self) -> None:
        with_charge = [r.name for r in NET_CHARGE_LADDER if "charge" in r.fields]
        assert with_charge == [
            "R0_full",
            "R1dA_drop_f5",
            "R0A",
            "R1A_drop_f5",
            "R0",
            "R1_drop_f5",
        ]
        for rung in NET_CHARGE_LADDER:
            assert set(rung.fields) <= set(SIGNATURE_FIELDS)
            assert rung.fields[0] == "elem"

    def test_without_charge_the_rungs_are_the_reference_rungs(self) -> None:
        stripped = [
            (r.name, tuple(f for f in r.fields if f != "charge"))
            for r in NET_CHARGE_LADDER
            if not r.name.startswith("R1q")
        ]
        assert stripped == [(r.name, r.fields) for r in REFERENCE_LADDER]
        assert not any("charge" in r.fields for r in REFERENCE_LADDER)


class TestProfile:
    def test_reference_profile_has_no_charge_field(self) -> None:
        assert REFERENCE_723696D.charge_policy is ChargePolicy.DISABLED
        assert not REFERENCE_723696D.charge_enabled

    def test_with_net_charge(self) -> None:
        assert CHARGED.name == "reference-723696d+net-charge"
        assert CHARGED.charge_policy is ChargePolicy.NET_ROUND
        assert CHARGED.charge_key == "charge"
        assert CHARGED.ladder == NET_CHARGE_LADDER
        assert CHARGED.bins == REFERENCE_723696D.bins
        assert CHARGED.geometry == REFERENCE_723696D.geometry
        assert LENIENT.charge_policy is ChargePolicy.NET_ROUND_ZERO_IF_MISSING
        keyed = with_net_charge(REFERENCE_723696D, key="total_charge")
        assert keyed.charge_key == "total_charge"

    def test_with_net_charge_rejects_inconsistent_requests(self) -> None:
        with pytest.raises(ConfigError, match="enabled charge policy"):
            with_net_charge(REFERENCE_723696D, ChargePolicy.DISABLED)
        with pytest.raises(ConfigError, match="already has a charge field"):
            with_net_charge(CHARGED)
        custom = dataclasses.replace(REFERENCE_723696D, ladder=REFERENCE_LADDER[:5])
        with pytest.raises(ConfigError, match="custom ladder"):
            with_net_charge(custom)
        explicit = with_net_charge(custom, ladder=NET_CHARGE_LADDER[:3])
        assert explicit.ladder == NET_CHARGE_LADDER[:3]

    def test_ladder_and_policy_must_agree(self) -> None:
        with pytest.raises(ConfigError, match="no rung uses the 'charge' field"):
            dataclasses.replace(CHARGED, ladder=REFERENCE_LADDER)
        with pytest.raises(ConfigError, match="charge policy is disabled"):
            dataclasses.replace(REFERENCE_723696D, ladder=NET_CHARGE_LADDER)
        with pytest.raises(ConfigError, match="non-empty metadata key"):
            dataclasses.replace(CHARGED, charge_key="")
        with pytest.raises(ValueError, match="'net'"):
            with_net_charge(REFERENCE_723696D, "net")

    def test_token_fingerprint_changes_only_with_the_charge_field(self) -> None:
        reference = token_payload(REFERENCE_723696D)
        assert set(reference) == {"geometry", "bins", "magnetic_policy"}
        charged = token_payload(CHARGED)
        assert charged["charge_policy"] is ChargePolicy.NET_ROUND
        assert charged["charge_key"] == "charge"
        digests = {
            digest(token_payload(profile))
            for profile in (
                REFERENCE_723696D,
                CHARGED,
                LENIENT,
                with_net_charge(REFERENCE_723696D, key="q"),
            )
        }
        assert len(digests) == 4


class TestTokenGrammar:
    @pytest.mark.parametrize(
        "key",
        [
            "Fe_0_q-1_Cl-O-O-O_T:4_3_1_2",
            "Fe_2_q0_Cl-O-O-O_T:4_3_1_2",
            "O_0_q3_O-O-Sr-Sr-Sr-Sr-Ti-Ti_BO_3:8_1_3_1",
            "Fe_NA_q-12_INVALID_INVALID_999_999_999",
            "X_0_q2__O:6_0_999_999",
        ],
    )
    def test_charged_keys_round_trip(self, key: str) -> None:
        parsed = SiteToken.from_key(key)
        assert parsed.charge is not None
        assert parsed.key() == key

    def test_charge_field_sits_after_the_moment(self) -> None:
        parsed = SiteToken.from_key("O_0_q-2_O-O-Sr_BO_3:8_1_3_1")
        assert parsed.charge == -2
        assert parsed.moment == 0
        assert parsed.neighbours == (("O", 2), ("Sr", 1))
        assert parsed.geometry == "BO_3:8"
        uncharged = dataclasses.replace(parsed, charge=None)
        assert uncharged.key() == "O_0_O-O-Sr_BO_3:8_1_3_1"
        assert SiteToken.from_key(uncharged.key()).charge is None

    @pytest.mark.parametrize(
        ("key", "match"),
        [
            ("Fe_0_q1.5_O_T:4_3_1_2", "non-integer charge"),
            ("Fe_0_qx_O_T:4_3_1_2", "non-integer charge"),
            ("Fe_0_q_O_T:4_3_1_2", "non-integer charge"),
            ("Fe_0_q+1_O_T:4_3_1_2", "non-canonically"),
            ("Fe_0_q01_O_T:4_3_1_2", "non-canonically"),
            ("Fe_0_q-0_O_T:4_3_1_2", "non-canonically"),
            ("Fe_0_q1_O_T:4_3_1", "too few fields"),
        ],
    )
    def test_malformed_charge_fields_are_rejected(self, key: str, match: str) -> None:
        with pytest.raises(UnsupportedInputError, match=match):
            SiteToken.from_key(key)

    def test_charge_signature_field(self) -> None:
        charged = SiteToken.from_key("Fe_0_q-1_Cl-O-O-O_T:4_3_1_2")
        fields = signature_fields(charged)
        assert fields["charge"] == "q-1"
        assert signature(fields, NET_CHARGE_LADDER[0].fields) == (
            "Fe|0|q-1|Cl-O-O-O|T:4|3|1|2"
        )
        assert signature(fields, NET_CHARGE_LADDER[2].fields) == (
            "Fe|0|Cl-O-O-O|T:4|1|2"
        )
        plain = dataclasses.replace(charged, charge=None)
        assert signature_fields(plain)["charge"] == "NA"


class TestBins:
    @pytest.mark.parametrize(
        ("charge", "expected"),
        [
            (-1.0, -1),
            (2.0, 2),
            (-1.5, -2),
            (-0.5, 0),
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (-2.4, -2),
            (0.6, 1),
        ],
    )
    def test_charge_keeps_its_sign_and_ties_to_even(
        self, charge: float, expected: int
    ) -> None:
        assert charge_bin(charge, ChargePolicy.NET_ROUND) == expected
        assert charge_bin(charge, ChargePolicy.NET_ROUND_ZERO_IF_MISSING) == expected
        assert charge_bin(charge, ChargePolicy.DISABLED) is None

    def test_vectorised_charge_bins_match_scalar_bins(self) -> None:
        rng = np.random.default_rng(1)
        charges = np.concatenate(
            [rng.uniform(-4, 4, 200), [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.0]]
        )
        vector = charge_bins(charges, ChargePolicy.NET_ROUND)
        assert vector is not None
        assert vector.dtype == np.int64
        assert vector.tolist() == [
            charge_bin(float(c), ChargePolicy.NET_ROUND) for c in charges
        ]
        assert charge_bins(charges, ChargePolicy.DISABLED) is None


class TestResolution:
    def test_disabled_profiles_read_nothing(self) -> None:
        resolved = resolve_net_charge({"charge": -1}, REFERENCE_723696D)
        assert resolved.source is ChargeSource.DISABLED
        assert (resolved.raw, resolved.value, resolved.failure) == (None, None, None)

    @pytest.mark.parametrize("value", [-1, 2, 0, -0.75, 1.5])
    def test_supplied_numbers(self, value: float) -> None:
        for profile in (CHARGED, LENIENT):
            resolved = resolve_net_charge({"charge": value, "method": "PBE"}, profile)
            assert resolved.source is ChargeSource.SUPPLIED
            assert resolved.raw == resolved.value == float(value)
            assert resolved.failure is None

    @pytest.mark.parametrize("metadata", [{}, {"charge": None}, {"q": -1}])
    def test_missing_charge(self, metadata: dict[str, object]) -> None:
        strict = resolve_net_charge(metadata, CHARGED)
        assert strict.source is ChargeSource.MISSING
        assert strict.value is None
        assert strict.failure == "missing_charge"
        assert strict.message is not None
        assert "net_round_zero_if_missing" in strict.message
        lenient = resolve_net_charge(metadata, LENIENT)
        assert lenient.source is ChargeSource.DEFAULTED
        assert (lenient.raw, lenient.value, lenient.failure) == (None, 0.0, None)

    @pytest.mark.parametrize(
        "value",
        ["-1", True, False, float("nan"), float("inf"), MAX_ABS_CHARGE, 10**400],
    )
    def test_unusable_values_fail_under_every_policy(self, value: object) -> None:
        for profile in (CHARGED, LENIENT):
            resolved = resolve_net_charge({"charge": value}, profile)
            assert resolved.source is ChargeSource.INVALID
            assert resolved.failure == "invalid_charge"
            assert resolved.value is None
            assert resolved.message is not None
            assert "not a finite number" in resolved.message

    def test_configured_key(self) -> None:
        profile = with_net_charge(REFERENCE_723696D, key="total_charge")
        resolved = resolve_net_charge({"charge": 5, "total_charge": -2}, profile)
        assert resolved.value == -2.0


class TestSiteToken:
    def test_scalar_token_carries_the_rounded_charge(self) -> None:
        token = site_token(site_row(), CHARGED, -1.4)
        assert token.charge == -1
        assert token.key() == "Fe_2_q-1_Cl-O-O-O_T:4_2_1_2"
        assert site_token(site_row(), REFERENCE_723696D).key() == (
            "Fe_2_Cl-O-O-O_T:4_2_1_2"
        )
        assert site_token(site_row(), REFERENCE_723696D, -1.0).charge is None

    def test_charged_profiles_need_the_net_charge(self) -> None:
        with pytest.raises(ConfigError, match="pass the record's net charge"):
            site_token(site_row(), CHARGED)
