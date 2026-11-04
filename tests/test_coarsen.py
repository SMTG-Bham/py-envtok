"""Exact coverage classes: reference equivalence, support rules and isolation."""

from dataclasses import replace
from pathlib import Path

import ase.io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ase import Atoms
from helpers import CORPUS, random_pool, random_token
from reference_oracles import (
    FLOOR_SIGS,
    reference_demand_mapping,
    reference_ladder,
    reference_signatures,
)

from envtok.coarsen import (
    TERMINAL_RUNG,
    CoverageBuilder,
    CoverageClasses,
    Pool,
    assign_duckdb,
    assign_memory,
    eligible_mask,
    paper_table_s1_ladder,
)
from envtok.corpus import Corpus
from envtok.errors import ConfigError
from envtok.execution import ExecutionSettings
from envtok.profiles import REFERENCE_723696D, REFERENCE_LADDER, InvalidRecordPolicy
from envtok.records import Split
from envtok.store.cache import TokenCache
from envtok.store.duck import connect
from envtok.tokenization import Tokenizer
from envtok.tokenizer.tokens import SiteToken, signature, signature_fields

PROFILE = REFERENCE_723696D

SPECIAL_KEYS = [
    "Fe_0_INVALID_INVALID_999_999_999",
    "Fe_0_O_S:1_0_3_0",
    "X_0__O:6_0_999_999",
    "O_0_O-O-Sr-Sr-Sr-Sr-Ti-Ti_BO_3:8_1_3_1",
    "Cs_2_Cl-Cl-Cl-Cl-Cl-Cl-Cl-Cl-Cs-Cs_UNKNOWN:10_999_2_0",
    "Na_NA_Br-Cl-Cl-Cl-Cl-Cl-Cl-Cl_O:6_2_4_2",
    "C_0_C-C-C_UNKNOWN:3_999_3_0",
]


def fields_of(pool: Pool) -> list[dict[str, str]]:
    return [signature_fields(SiteToken.from_key(k)) for k in pool.token_keys]


def test_projections_match_reference_sql_for_every_rung_and_floor() -> None:
    rng = np.random.default_rng(7)
    keys = SPECIAL_KEYS + [random_token(rng).key() for _ in range(400)]
    expected = reference_signatures(keys)
    assert [name for name, _ in reference_ladder()] == [
        r.name for r in REFERENCE_LADDER
    ]
    fields = [signature_fields(SiteToken.from_key(k)) for k in keys]
    for rung in REFERENCE_LADDER:
        assert [signature(f, rung.fields) for f in fields] == expected[rung.name], (
            rung.name
        )
    for family, (name, _) in zip(PROFILE.presence_families, FLOOR_SIGS, strict=True):
        assert family.name == name
        assert [signature(f, family.fields) for f in fields] == expected[name], name


def reference_assignment(
    pool: Pool, threshold: int, ladder: object = None
) -> list[tuple[int, str]]:
    incidence = sorted(
        set(zip(pool.edge_record.tolist(), pool.edge_token.tolist(), strict=True))
    )
    mapping = reference_demand_mapping(pool.token_keys, incidence, threshold, ladder)  # type: ignore[arg-type]
    return [mapping[i] for i in range(len(pool.token_keys))]


@pytest.mark.parametrize(
    ("seed", "threshold"), [(0, 2), (1, 3), (2, 5), (3, 2), (4, 8)]
)
def test_both_backends_reproduce_the_reference_demand_mapping(
    seed: int, threshold: int, tmp_path: Path
) -> None:
    pool = random_pool(seed)
    fields = fields_of(pool)
    expected = reference_assignment(pool, threshold)
    memory = assign_memory(pool, fields, REFERENCE_LADDER, threshold)
    duck = assign_duckdb(
        pool,
        fields,
        REFERENCE_LADDER,
        threshold,
        lambda: connect(temp_directory=tmp_path),
    )
    for assignment in (memory, duck):
        assert (
            list(zip(assignment.rung.tolist(), assignment.signature, strict=True))
            == expected
        )
    np.testing.assert_array_equal(memory.support, duck.support)
    assert len({r for r, _ in expected}) > 3  # the ladder is genuinely exercised


def handmade_pool(rows: list[list[str]]) -> Pool:
    keys = sorted({k for row in rows for k in row})
    edges: dict[tuple[int, int], int] = {}
    for record, row in enumerate(rows):
        for key in row:
            edge = (record, keys.index(key))
            edges[edge] = edges.get(edge, 0) + 1
    pairs = sorted(edges.items())
    return Pool(
        record_ids=np.array([f"r{i}" for i in range(len(rows))], dtype=np.str_),
        natoms=np.array([len(row) for row in rows], dtype=np.int64),
        native_ids=[None] * len(rows),
        edge_record=np.array([r for (r, _), _ in pairs], dtype=np.int64),
        edge_token=np.array([t for (_, t), _ in pairs], dtype=np.int64),
        edge_multiplicity=np.array([c for _, c in pairs], dtype=np.int64),
        token_ids=np.arange(len(keys), dtype=np.int64),
        token_keys=keys,
    )


def fit(rows: list[list[str]], threshold: int = 2, **kwargs: object) -> CoverageClasses:
    builder = CoverageBuilder(min_structure_support=threshold, **kwargs)  # type: ignore[arg-type]
    return builder.fit_pool(handmade_pool(rows))


def class_of(classes: CoverageClasses, key: str) -> tuple[str, str]:
    index = classes.pool.token_keys.index(key)
    cls = int(classes.token_class[index])
    return classes.rung_name(int(classes.class_rung[cls])), classes.class_signature[cls]


A = "Fe_0_O-O-O-O_T:4_0_3_0"
A_DMIN = "Fe_0_O-O-O-O_T:4_0_2_0"
SAME_SHELL = "Fe_0_O-O-O-O-O-O_O:6_0_3_0"  # shares A's coarse signatures
OTHER = "Na_0_Cl-Cl-Cl-Cl-Cl-Cl_O:6_0_3_0"  # shares nothing with A


class TestSupportRules:
    def test_sites_in_one_record_count_once(self) -> None:
        classes = fit([[A, A, A, A], [OTHER]])
        assert class_of(classes, A)[0] == "terminal"
        assert classes.class_support[classes.token_class[0]] == 1
        assert class_of(fit([[A], [A]]), A) == ("R0_full", "Fe|0|O-O-O-O|T:4|0|3|0")

    def test_sibling_tokens_in_one_record_count_once(self) -> None:
        # A and A_DMIN share every signature without dmin, but sit in one record.
        classes = fit([[A, A_DMIN], [OTHER]])
        assert class_of(classes, A)[0] == "terminal"
        two_records = fit([[A], [A_DMIN], [OTHER]])
        assert class_of(two_records, A) == class_of(two_records, A_DMIN)
        assert class_of(two_records, A)[0] == "R0A"

    def test_resolved_tokens_never_support_a_later_rung(self) -> None:
        # SAME_SHELL resolves at R0_full in records 0-1. A, alone in record 2,
        # shares SAME_SHELL's coarse signatures but cannot borrow its support.
        classes = fit([[SAME_SHELL], [SAME_SHELL], [A]])
        assert class_of(classes, SAME_SHELL)[0] == "R0_full"
        assert class_of(classes, A)[0] == "terminal"
        together = fit([[SAME_SHELL], [A]])
        assert class_of(together, A) == class_of(together, SAME_SHELL)

    def test_every_token_has_exactly_one_class_and_residues_are_kept(self) -> None:
        pool = random_pool(11)
        classes = CoverageBuilder(min_structure_support=4).fit_pool(pool)
        assert classes.token_class.shape == (len(pool.token_ids),)
        assert set(classes.token_class.tolist()) == set(range(classes.n_classes))
        terminal = classes.class_rung == TERMINAL_RUNG
        assert terminal.any()
        assert all(
            s.startswith("__CATCHALL__|")
            for s, t in zip(classes.class_signature, terminal, strict=True)
            if t
        )
        stats = classes.rung_statistics()
        assert sum(s["tokens"] for s in stats) == len(pool.token_ids)
        assert sum(s["atoms"] for s in stats) == int(pool.edge_multiplicity.sum())
        assert stats[-1]["name"] == "terminal"

    def test_invalid_placeholders_never_join_one_coordinate_classes(self) -> None:
        # With CN_SIG the two only meet at the CN-free R8_elem rung; had the
        # placeholder parsed as CN=1 they would merge at R7_elem_cn ("Fe|1").
        invalid = "Fe_0_INVALID_INVALID_999_999_999"
        genuine = "Fe_0_O_S:1_0_3_0"
        classes = fit([[invalid], [genuine]])
        assert class_of(classes, invalid) == ("R8_elem", "Fe")
        assert class_of(classes, genuine) == ("R8_elem", "Fe")
        pair = fit([[invalid], [invalid]])
        assert class_of(pair, invalid)[1].split("|")[2] == "INVALID"

    def test_paper_table_s1_ladder_differs_from_the_33_rung_ladder(self) -> None:
        paper = paper_table_s1_ladder(PROFILE)
        assert [r.name for r in paper] == [
            "R0_full", "R1dA_drop_f5", "R2dA_drop_spin", "R2adA_unrel_poly",
            "R2bdA_binned_mult", "R3dA_nbset_poly", "R4dA_nbset_cn", "R5dA_nbset",
            "R6dA_elem_poly", "R7dA_elem_cn", "R8dA_elem", "R8A_elem", "R8_elem",
        ]  # fmt: skip
        rows = [[A], [A_DMIN], [OTHER]]
        assert class_of(fit(rows), A) == ("R0A", "Fe|0|O-O-O-O|T:4|0|0")
        assert class_of(fit(rows, ladder=paper), A) == ("R8A_elem", "Fe|0")
        expected = reference_assignment(handmade_pool(rows), 2)
        assert expected[0][0] == 11  # rung index of R0A in the reference ladder


class TestBuilderValidation:
    @pytest.mark.parametrize("threshold", [1, 0, -3])
    def test_threshold_below_profile_minimum_is_rejected(self, threshold: int) -> None:
        with pytest.raises(ConfigError, match="minimum"):
            CoverageBuilder(min_structure_support=threshold)

    def test_other_invalid_settings(self) -> None:
        with pytest.raises(ConfigError, match="integer"):
            CoverageBuilder(min_structure_support=2.5)  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="integer"):
            CoverageBuilder(min_structure_support=True)
        with pytest.raises(ConfigError, match="backend"):
            CoverageBuilder(backend="gpu")

    def test_with_threshold_copies_settings(self) -> None:
        builder = CoverageBuilder(min_structure_support=2, backend="memory")
        other = builder.with_threshold(7)
        assert (other.min_support, other.backend, builder.min_support) == (
            7,
            "memory",
            2,
        )

    def test_auto_backend_switches_to_duckdb_for_large_incidence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from envtok import coarsen  # noqa: PLC0415

        pool = random_pool(5)
        monkeypatch.setattr(coarsen, "AUTO_MEMORY_EDGE_LIMIT", 1)
        builder = CoverageBuilder(min_structure_support=3, workdir=tmp_path)
        duck = builder.fit_pool(pool)
        assert duck.backend == "duckdb"
        memory = CoverageBuilder(min_structure_support=3, backend="memory").fit_pool(
            pool
        )
        np.testing.assert_array_equal(duck.token_class, memory.token_class)
        assert duck.class_signature == memory.class_signature
        unrooted = CoverageBuilder(min_structure_support=3, backend="duckdb")
        assert unrooted.fit_pool(pool).class_signature == memory.class_signature


class TestEligibility:
    def table(self) -> pa.Table:
        return pa.table(
            {
                "status": ["ok", "ok", "ok", "failed", "ok"],
                "split": ["train", "train", "validation", "train", "train"],
                "n_invalid_sites": [0, 2, 0, 0, 0],
                "metadata_json": [
                    '{"q": 0}',
                    '{"q": 0}',
                    '{"q": 0}',
                    "{}",
                    '{"q": -1}',
                ],
            }
        )

    def test_policies_and_strata(self) -> None:
        table = self.table()
        quarantine = eligible_mask(table, Split.TRAIN, InvalidRecordPolicy.QUARANTINE)
        reference = eligible_mask(table, Split.TRAIN, InvalidRecordPolicy.REFERENCE)
        assert quarantine.tolist() == [True, False, False, False, True]
        assert reference.tolist() == [True, True, False, False, True]
        stratum = eligible_mask(
            table, Split.TRAIN, InvalidRecordPolicy.REFERENCE, where={"q": -1}
        )
        assert stratum.tolist() == [False, False, False, False, True]


@pytest.fixture(scope="module")
def mixed_cache(tmp_path_factory: pytest.TempPathFactory) -> TokenCache:
    frames = ase.io.read(CORPUS, index=":")
    assert isinstance(frames, list)
    for i, atoms in enumerate(frames):
        atoms.info["split"] = (
            "validation" if i in {1, 4} else "test" if i == 7 else "train"
        )
    root = tmp_path_factory.mktemp("mixed")
    path = root / "mixed.extxyz"
    ase.io.write(path, frames, format="extxyz")
    corpus = Corpus.from_extxyz(path, dataset="mix", revision="1", split_key="split")
    return Tokenizer().run(corpus, root / "work", batch_size=7)


def without_held_out(cache: TokenCache, tmp_path: Path) -> TokenCache:
    """Copy the index, dropping held-out records entirely."""
    import shutil  # noqa: PLC0415

    target = tmp_path / "cache"
    shutil.copytree(cache.cache_dir, target)
    token_dir = target / "tokens" / cache.token_dir.name
    structures = pq.read_table(token_dir / "structures.parquet")
    keep = pa.array([s == "train" for s in structures["split"].to_pylist()])
    pq.write_table(structures.filter(keep), token_dir / "structures.parquet")
    return TokenCache.open(token_dir)


class TestFittedOnCache:
    def test_fit_uses_only_the_training_split(
        self, mixed_cache: TokenCache, tmp_path: Path
    ) -> None:
        builder = CoverageBuilder(min_structure_support=2)
        classes = builder.fit(mixed_cache, "train")
        assert len(classes.pool.record_ids) == 17
        assert all("validation" not in rid for rid in classes.pool.record_ids)
        alone = builder.fit(without_held_out(mixed_cache, tmp_path), Split.TRAIN)
        assert alone.class_signature == classes.class_signature
        np.testing.assert_array_equal(alone.token_class, classes.token_class)

    def test_backends_agree_on_the_fixture_cache(self, mixed_cache: TokenCache) -> None:
        settings = ExecutionSettings(memory_budget=2**30)
        memory = CoverageBuilder(min_structure_support=2, backend="memory").fit(
            mixed_cache
        )
        duck = CoverageBuilder(
            min_structure_support=2, backend="duckdb", execution=settings
        ).fit(mixed_cache)
        assert duck.class_signature == memory.class_signature
        np.testing.assert_array_equal(duck.class_support, memory.class_support)

    def test_assign_maps_held_out_tokens_or_reports_oov(
        self, mixed_cache: TokenCache
    ) -> None:
        classes = CoverageBuilder(min_structure_support=2).fit(mixed_cache)
        known = classes.pool.token_keys[0]
        assert classes.assign([known, "Xx_0_O_S:1_0_3_0"]) == [
            int(classes.token_class[0]),
            None,
        ]
        assert classes.assign([known]) == [int(classes.token_class[0])]

    def test_empty_pools_and_strata(self, mixed_cache: TokenCache) -> None:
        builder = CoverageBuilder(min_structure_support=2)
        with pytest.raises(ConfigError, match="no eligible 'unsplit'"):
            builder.fit(mixed_cache, "unsplit")
        with pytest.raises(ConfigError, match="matching"):
            builder.fit(mixed_cache, "train", where={"method": "HSE"})
        stratum = builder.fit(mixed_cache, "train", where={"method": "PBE"})
        assert stratum.where == {"method": "PBE"}

    def test_save_and_fingerprint(
        self, mixed_cache: TokenCache, tmp_path: Path
    ) -> None:
        classes = CoverageBuilder(min_structure_support=2).fit(mixed_cache)
        out = classes.save(tmp_path / "classes")
        table = pq.read_table(out / "classes.parquet")
        assert table.num_rows == classes.n_classes
        assert pq.read_table(out / "token_classes.parquet").num_rows == len(
            classes.pool.token_ids
        )
        assert (out / "classes.json").read_text().count('"rung"') == 34
        again = CoverageBuilder(min_structure_support=2).fit(mixed_cache)
        assert again.fingerprint == classes.fingerprint
        assert CoverageBuilder(min_structure_support=3).fit(
            mixed_cache
        ).fingerprint != (classes.fingerprint)
        assert classes.token_cache is mixed_cache


def test_synthetic_classes_cannot_be_audited() -> None:
    classes = CoverageBuilder(min_structure_support=2).fit_pool(random_pool(1))
    with pytest.raises(ConfigError, match="in-memory pool"):
        _ = classes.token_cache
    assert len(classes.fingerprint) == 64
    saved = replace(classes.ladder[0])
    assert saved.name == "R0_full"


def test_frames_fixture_untouched(corpus_frames: list[Atoms]) -> None:
    assert {a.info["split"] for a in corpus_frames} == {"train"}
