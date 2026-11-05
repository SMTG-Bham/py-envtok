"""The net-charge token field: profile, token grammar, bins and resolution."""

import dataclasses
import json
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pyarrow.parquet as pq
import pytest
from ase import Atoms
from helpers import CORPUS

from envtok import CoverageBuilder, TokenCache, Tokenizer
from envtok.config import load_config
from envtok.corpus import Corpus
from envtok.errors import ConfigError, StaleCacheError, UnsupportedInputError
from envtok.fingerprint import digest, token_payload
from envtok.pipeline import Pipeline, load_selection
from envtok.profiles import (
    NET_CHARGE_LADDER,
    REFERENCE_723696D,
    REFERENCE_LADDER,
    SIGNATURE_FIELDS,
    ChargePolicy,
    with_net_charge,
)
from envtok.store.cache import _charged_sites, _record_charges, site_keys
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


# --- Index-time derivation -------------------------------------------------

NACL_PRIM, NACL_COMPRESSED = 0, 3
CHARGES: list[tuple[int, object]] = [
    (NACL_PRIM, 0),
    (NACL_PRIM, -1),
    (NACL_PRIM, 1.5),  # rounds to 2
    (NACL_PRIM, None),  # no charge in info
    (NACL_PRIM, "neutral"),  # present but unusable
    (NACL_COMPRESSED, -1),
]


def record_id(index: int) -> str:
    return f"fx@v1:charged.extxyz:{index}"


@pytest.fixture(scope="module")
def charged_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    loaded = ase.io.read(CORPUS, index=":", format="extxyz")
    assert isinstance(loaded, list)
    written: list[Atoms] = []
    for fixture, charge in CHARGES:
        atoms = loaded[fixture].copy()
        atoms.info.pop("charge")
        if charge is not None:
            atoms.info["charge"] = charge
        written.append(atoms)
    molecule = loaded[NACL_PRIM].copy()
    molecule.pbc = False  # excluded before geometry, and has no charge
    molecule.info.pop("charge")
    written.append(molecule)
    path = tmp_path_factory.mktemp("charged") / "charged.extxyz"
    ase.io.write(path, written, format="extxyz")
    return path


@pytest.fixture(scope="module")
def charged_corpus(charged_path: Path) -> Corpus:
    return Corpus.from_extxyz(charged_path, dataset="fx", revision="v1", split="train")


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("charged-work")


@pytest.fixture(scope="module")
def plain(charged_corpus: Corpus, workdir: Path) -> TokenCache:
    return Tokenizer().run(charged_corpus, workdir, batch_size=3)


@pytest.fixture(scope="module")
def strict(plain: TokenCache, workdir: Path) -> TokenCache:
    return Tokenizer(CHARGED).index(plain.cache_dir, workdir)


@pytest.fixture(scope="module")
def lenient(plain: TokenCache, workdir: Path) -> TokenCache:
    return Tokenizer(LENIENT).index(plain.cache_dir, workdir)


def by_record(cache: TokenCache, *columns: str) -> dict[str, tuple[Any, ...]]:
    table = cache.read("structures", ["record_id", *columns]).to_pylist()
    return {row["record_id"]: tuple(row[c] for c in columns) for row in table}


class TestIndex:
    def test_the_charge_field_reuses_the_geometry_pass(
        self, plain: TokenCache, strict: TokenCache, lenient: TokenCache
    ) -> None:
        assert strict.cache_dir == lenient.cache_dir == plain.cache_dir
        assert len({plain.token_dir, strict.token_dir, lenient.token_dir}) == 3
        assert len(list(plain.partitions_dir.glob("part-*"))) == 3
        assert not list(strict.token_dir.glob("_*"))
        assert strict.profile is CHARGED

    def test_reference_index_is_unchanged(self, plain: TokenCache) -> None:
        assert "charge" not in plain.summary
        rows = by_record(plain, "status", "net_charge", "charge", "charge_source")
        assert {row[1:] for row in rows.values()} == {(None, None, "disabled")}
        assert set(plain.read("tokens")["charge"].to_pylist()) == {None}
        assert not any("_q" in k for k in plain.read("tokens")["token_key"].to_pylist())

    def test_strict_policy_fails_records_without_a_usable_charge(
        self, strict: TokenCache
    ) -> None:
        rows = by_record(
            strict, "status", "status_reason", "net_charge", "charge", "charge_source"
        )
        assert rows == {
            record_id(0): ("ok", None, 0.0, 0, "supplied"),
            record_id(1): ("ok", None, -1.0, -1, "supplied"),
            record_id(2): ("ok", None, 1.5, 2, "supplied"),
            record_id(3): ("failed", "missing_charge", None, None, "missing"),
            record_id(4): ("failed", "invalid_charge", None, None, "invalid"),
            record_id(5): ("ok", None, -1.0, -1, "supplied"),
            record_id(6): (
                "excluded",
                "unsupported_periodicity",
                None,
                None,
                "missing",
            ),
        }
        assert strict.summary["ok"] == 4
        assert strict.summary["failed"] == 2
        assert strict.summary["excluded"] == 1
        assert strict.summary["sites"] == 8
        assert strict.summary["charge"] == {
            "policy": "net_round",
            "key": "charge",
            "records_by_source": {"invalid": 1, "missing": 2, "supplied": 4},
            "failed_records": 2,
        }
        failures = [
            row
            for row in strict.read("failures").to_pylist()
            if row["stage"] == "charge"
        ]
        assert [(f["record_id"], f["category"], f["fatal"]) for f in failures] == [
            (record_id(3), "missing_charge", True),
            (record_id(4), "invalid_charge", True),
        ]
        assert "net_round_zero_if_missing" in failures[0]["message"]
        assert "'neutral'" in failures[1]["message"]
        failed = {record_id(3), record_id(4)}
        for table in ("site_tokens", "structure_tokens"):
            assert failed.isdisjoint(strict.read(table)["record_id"].to_pylist())

    def test_lenient_policy_defaults_missing_charges_to_neutral(
        self, lenient: TokenCache
    ) -> None:
        rows = by_record(lenient, "status", "net_charge", "charge", "charge_source")
        assert rows[record_id(3)] == ("ok", None, 0, "defaulted")
        assert rows[record_id(4)] == ("failed", None, None, "invalid")
        assert lenient.summary["ok"] == 5
        # The excluded record resolves too, but keeps its own status.
        assert rows[record_id(6)] == ("excluded", None, 0, "defaulted")
        assert lenient.summary["charge"]["records_by_source"] == {
            "defaulted": 2,
            "invalid": 1,
            "supplied": 4,
        }

    def test_tokens_carry_the_rounded_charge(
        self, strict: TokenCache, lenient: TokenCache
    ) -> None:
        tokens = strict.read("tokens").to_pylist()
        assert all(t["charge"] is not None for t in tokens)
        sodium = sorted(
            (t["charge"], t["token_key"], t["n_records"])
            for t in tokens
            if t["token_key"].startswith("Na_") and t["n_records"] == 1
        )
        assert [s[0] for s in sodium] == [-1, -1, 0, 2]
        for charge, key, _ in sodium:
            assert SiteToken.from_key(key).charge == charge
            assert key.startswith(f"Na_0_q{charge}_")
        site_records = lenient.read("site_tokens")["record_id"].to_pylist()
        assert record_id(3) in site_records
        keys = dict(
            zip(
                lenient.read("tokens")["token_id"].to_pylist(),
                lenient.read("tokens")["token_key"].to_pylist(),
                strict=True,
            )
        )
        neutral = {
            keys[row["token_id"]]
            for row in lenient.read("site_tokens").to_pylist()
            if row["record_id"] in {record_id(0), record_id(3)}
        }
        assert all("_q0_" in key for key in neutral)
        assert len(neutral) == 2  # Na and Cl, shared by the two neutral records

    def test_vectorised_keys_match_scalar_tokens(self, strict: TokenCache) -> None:
        structures = {
            row["record_id"]: json.loads(row["metadata_json"])
            for row in strict.read("structures").to_pylist()
        }
        for profile in (CHARGED, LENIENT):
            for partition in sorted(strict.partitions_dir.glob("part-*")):
                charges = _record_charges(
                    pq.read_table(partition / "structures.parquet"), profile
                )
                sites, net = _charged_sites(
                    pq.read_table(partition / "sites.parquet"), charges
                )
                expected = []
                for row in sites.to_pylist():
                    resolved = resolve_net_charge(structures[row["record_id"]], profile)
                    expected.append(
                        site_token(SiteRow(**row), profile, resolved.value).key()
                    )
                assert site_keys(sites, profile, net).to_pylist() == expected

    def test_charged_keys_need_one_charge_per_site(self, strict: TokenCache) -> None:
        part = min(strict.partitions_dir.glob("part-*"))
        sites = pq.read_table(part / "sites.parquet")
        with pytest.raises(ConfigError, match="one net charge per site"):
            site_keys(sites, CHARGED)
        with pytest.raises(ConfigError, match="one net charge per site"):
            site_keys(sites, CHARGED, np.zeros(sites.num_rows + 1))
        assert site_keys(sites, REFERENCE_723696D, np.zeros(3)) == site_keys(
            sites, REFERENCE_723696D
        )

    def test_reopening_needs_the_charged_profile(self, strict: TokenCache) -> None:
        assert TokenCache.open(strict.token_dir, CHARGED).summary == strict.summary
        with pytest.raises(StaleCacheError, match="does not match"):
            TokenCache.open(strict.token_dir, LENIENT)
        with pytest.raises(StaleCacheError, match="does not match"):
            TokenCache.open(strict.token_dir, REFERENCE_723696D)


class TestClasses:
    @pytest.mark.parametrize("backend", ["memory", "duckdb"])
    def test_rare_charge_states_merge_at_the_drop_charge_rung(
        self, strict: TokenCache, backend: str
    ) -> None:
        classes = CoverageBuilder(CHARGED, 2, backend=backend).fit(strict)
        assert classes.ladder == NET_CHARGE_LADDER
        keys = classes.pool.token_keys
        prim = [
            i
            for i, key in enumerate(keys)
            if key.startswith("Na_") and "_q-1_" not in key
        ]
        # Na of the q0 and q2 primitive cells: one record each at R0 and R1.
        rungs = {
            classes.rung_name(int(classes.class_rung[classes.token_class[i]]))
            for i in prim
        }
        assert rungs == {"R1qdA_drop_charge"}
        assert len({int(classes.token_class[i]) for i in prim}) == 1

    def test_backends_agree(self, strict: TokenCache) -> None:
        memory = CoverageBuilder(CHARGED, 2, backend="memory").fit(strict)
        duck = CoverageBuilder(CHARGED, 2, backend="duckdb").fit(strict)
        assert memory.class_signature == duck.class_signature
        assert memory.token_class.tolist() == duck.token_class.tolist()


# --- Configuration, pipeline and report ------------------------------------


def pipeline_config(path: Path, workdir: Path, **tokenizer: Any) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": {
            "adapter": "extxyz",
            "path": str(path),
            "name": "fx",
            "revision": "v1",
            "split": "train",
        },
        "tokenizer": tokenizer,
        "coverage": {"min_structure_support": 2},
        "execution": {"workdir": str(workdir), "batch_size": 3},
        "output": {"format": "none"},
    }


class TestConfig:
    def test_charge_variants(self, tmp_path: Path) -> None:
        def profile(**tokenizer: Any) -> Any:
            return load_config(
                pipeline_config(tmp_path / "x.extxyz", tmp_path, **tokenizer)
            ).profile()

        assert profile() is REFERENCE_723696D
        charged = profile(charge_policy="net_round")
        assert charged.name == "reference-723696d+net-charge"
        assert charged.ladder == NET_CHARGE_LADDER
        assert charged.charge_policy is ChargePolicy.NET_ROUND
        both = profile(
            charge_policy="net_round_zero_if_missing",
            charge_key="total_charge",
            magnetic_policy="disabled",
        )
        assert both.name == "reference-723696d+geometry-only+net-charge"
        assert both.charge_key == "total_charge"
        assert profile(charge_key="ignored") is REFERENCE_723696D

    def test_invalid_charge_settings(self, tmp_path: Path) -> None:
        document = pipeline_config(tmp_path / "x.extxyz", tmp_path, charge_policy="q")
        with pytest.raises(ConfigError, match=r"tokenizer\.charge_policy must be"):
            load_config(document)
        document["tokenizer"] = {"charge_policy": "net_round", "charge_key": ""}
        with pytest.raises(ConfigError, match="non-empty metadata key"):
            load_config(document).profile()

    def test_resolved_configuration_records_the_charge_settings(
        self, tmp_path: Path
    ) -> None:
        resolved = load_config(
            pipeline_config(tmp_path / "x.extxyz", tmp_path)
        ).resolved()
        assert resolved["tokenizer"]["charge_policy"] == "disabled"
        assert resolved["tokenizer"]["charge_key"] == "charge"


class TestPipeline:
    def test_inspect_counts_charge_sources(
        self, charged_path: Path, tmp_path: Path
    ) -> None:
        strict = Pipeline(
            pipeline_config(charged_path, tmp_path, charge_policy="net_round")
        )
        assert strict.inspect()["net_charge_sources"] == {
            "supplied": 4,
            "missing": 2,
            "invalid": 1,
        }
        plain = Pipeline(pipeline_config(charged_path, tmp_path))
        assert plain.inspect()["net_charge_sources"] == {"disabled": 7}

    def test_selection_with_the_charge_field(
        self, charged_path: Path, tmp_path: Path
    ) -> None:
        pipeline = Pipeline(
            pipeline_config(charged_path, tmp_path / "w", charge_policy="net_round")
        )
        result, directory = pipeline.select()
        selected = set(result.record_ids)
        assert selected.isdisjoint({record_id(3), record_id(4), record_id(6)})
        report = json.loads((directory / "coverage.json").read_text())
        assert report["profile"] == "reference-723696d+net-charge"
        assert report["classes"]["covered"] == report["classes"]["required"]
        assert [r["name"] for r in report["resolution"][:3]] == [
            "R0_full",
            "R1dA_drop_f5",
            "R1qdA_drop_charge",
        ]
        processing = report["processing"]
        assert processing["charge"] == {
            "policy": "net_round",
            "key": "charge",
            "records_by_source": [
                {"source": "invalid", "count": 1},
                {"source": "missing", "count": 2},
                {"source": "supplied", "count": 4},
            ],
        }
        statuses = {
            (row["status"], row["reason"]): row["count"]
            for row in processing["records_by_status"]
        }
        assert statuses[("failed", "invalid_charge")] == 1
        assert statuses[("failed", "missing_charge")] == 1
        charges = report["diagnostics"]["net_charge_structures"]
        assert [row["value"] for row in charges] == [-1, 0, 2]
        assert [row["eligible"] for row in charges] == [2, 1, 1]
        assert sum(row["selected"] for row in charges) == len(selected)
        html = (directory / "report.html").read_text()
        assert "Net charge (key charge, policy net_round)" in html
        assert "net_charge_structures" in html
        reloaded, again = load_selection(directory)
        assert reloaded.profile.charge_enabled
        assert again.record_ids == result.record_ids

    def test_reference_reports_have_no_charge_sections(
        self, charged_path: Path, tmp_path: Path
    ) -> None:
        _, directory = Pipeline(pipeline_config(charged_path, tmp_path)).select()
        report = json.loads((directory / "coverage.json").read_text())
        assert report["processing"]["charge"] == {
            "policy": "disabled",
            "key": None,
            "records_by_source": [],
        }
        assert "net_charge_structures" not in report["diagnostics"]
        assert "Net charge" not in (directory / "report.html").read_text()
