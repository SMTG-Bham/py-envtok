"""Golden comparison against the reference tokeniser in its pinned environment.

``tests/fixtures/reference/tokens-723696d.json`` was produced by the vendored
upstream functions running with ASE 3.25.0 and pymatgen 2025.6.14 (see
``build_tools/reference``). Token keys, representatives and neighbour-set
choices must match exactly. Raw values may differ only at floating-point
noise level between library stacks; they are bitwise identical to the
vendored code run on the development stack.
"""

import json
import math
from typing import Any

import numpy as np
import pytest
from ase import Atoms
from helpers import FIXTURES, record_of

from envtok.profiles import REFERENCE_723696D, corpus_policy
from envtok.tokenizer.geometry import GeometryEngine
from envtok.tokenizer.pipeline import TokenizedRecord, site_token, tokenize_record

ALLOWED = json.loads(
    (FIXTURES / "reference" / "allowed_discrepancies.json").read_text()
)
RAW_TOLERANCE = 1e-9


def _close(ours: float | None, theirs: float | None) -> bool:
    if ours is None or theirs is None:
        return ours is None and theirs is None
    return math.isclose(ours, theirs, rel_tol=RAW_TOLERANCE, abs_tol=RAW_TOLERANCE)


def test_golden_file_records_reference_environment() -> None:
    data = json.loads((FIXTURES / "reference" / "tokens-723696d.json").read_text())
    assert data["reference_commit"].startswith("723696d")
    assert data["python"] == "3.10.13"
    assert "pymatgen==2025.6.14" in data["packages"]
    assert "ase==3.25.0" in data["packages"]


def test_every_fixture_site_matches_the_reference(
    tokenized_corpus: dict[str, TokenizedRecord], golden: dict[str, Any]
) -> None:
    mismatches = []
    for sid, result in tokenized_corpus.items():
        reference = golden[sid]
        assert not reference["skipped"]
        assert len(result.sites) == len(reference["sites"])
        for site, expected in zip(result.sites, reference["sites"], strict=True):
            key = site_token(site, REFERENCE_723696D).key()
            extended = site.neighbour_source == "chemenv-extended"
            agrees = (
                key == expected["env_id"]
                and site.rep_index == expected["rep"]
                and extended == (expected["chosen_cutoff"] == "extended")
                and _close(site.csm, expected["sym_measure"])
                and _close(site.ratio_min, expected["dist_ratio_min"])
                and _close(site.ratio_max, expected["dist_ratio_max"])
                and _close(site.asymmetry, expected["a_raw"])
            )
            if not agrees and f"{sid}:{site.atom_index}" not in ALLOWED:
                mismatches.append((sid, site.atom_index, key, expected["env_id"]))
    assert mismatches == []


def test_fixture_corpus_exercises_the_specified_cases(
    tokenized_corpus: dict[str, TokenizedRecord],
) -> None:
    keys = {
        site_token(site, REFERENCE_723696D).key()
        for result in tokenized_corpus.values()
        for site in result.sites
    }
    sources = {
        site.neighbour_source for r in tokenized_corpus.values() for site in r.sites
    }
    assert sources == {"chemenv-default", "chemenv-extended", "crystalnn"}
    assert "O_0_O-O-Sr-Sr-Sr-Sr-Ti-Ti_BO_3:8_1_3_1" in keys  # underscore geometry
    assert "F_0_Ca-Ca-Ca-Ca-F-F-F-F-F-F_BS_2:10_3_3_0" in keys  # CSM bin 3
    assert "C_0_C-C-C_UNKNOWN:3_999_3_0" in keys  # CrystalNN rescue
    assert "Rf_0_O-O-O-O-O-O_O:6_0_999_999" in keys  # missing covalent radius
    assert {k.split("_")[1] for k in keys if k.startswith("Fe_")} == {"2", "3", "4"}
    po = tokenized_corpus["fx-016-po_simple_cubic"]
    assert {n.neighbour_index for n in po.neighbours} == {0}
    assert len({(n.image_a, n.image_b, n.image_c) for n in po.neighbours}) == 6


def test_moments_are_per_atom_not_per_orbit(
    tokenized_corpus: dict[str, TokenizedRecord],
) -> None:
    afm = tokenized_corpus["fx-014-fe_bcc_afm"]
    assert [s.rep_index for s in afm.sites] == [0, 0]
    tokens = [site_token(s, REFERENCE_723696D) for s in afm.sites]
    assert [t.moment for t in tokens] == [2, 3]
    ties = tokenized_corpus["fx-015-fe_bcc_ties"]
    assert [site_token(s, REFERENCE_723696D).moment for s in ties.sites] == [2, 4]
    assert afm.structure.moment_source == "supplied"
    assert tokenized_corpus["fx-000-nacl_prim"].structure.moment_source == "defaulted"


def _keys(atoms: Atoms, engine: GeometryEngine, *, symmetry: bool = True) -> list[str]:
    profile = REFERENCE_723696D
    if not symmetry:
        from dataclasses import replace  # noqa: PLC0415

        profile = replace(
            profile, geometry=replace(profile.geometry, use_symmetry=False)
        )
        engine = GeometryEngine(profile.geometry, profile.measures)
    result = tokenize_record(record_of(atoms), profile, corpus_policy(), engine)
    return [site_token(site, profile).key() for site in result.sites]


INVARIANCE_CASES = ["fx-002-nacl_rattled", "fx-010-srtio3_polar", "fx-013-rutile_tio2"]


@pytest.mark.parametrize("sid", INVARIANCE_CASES)
def test_symmetry_reuse_matches_all_sites_analysis(
    sid: str, corpus_frames: list[Atoms], engine: GeometryEngine
) -> None:
    atoms = next(a for a in corpus_frames if a.info["sid"] == sid)
    assert _keys(atoms, engine) == _keys(atoms, engine, symmetry=False)


@pytest.mark.parametrize("sid", INVARIANCE_CASES)
def test_tokens_invariant_under_rigid_motion_permutation_and_wrapping(
    sid: str, corpus_frames: list[Atoms], engine: GeometryEngine
) -> None:
    atoms = next(a for a in corpus_frames if a.info["sid"] == sid)
    reference = _keys(atoms, engine)

    moved = atoms.copy()
    moved.translate((0.37, -1.21, 2.05))
    moved.wrap()
    assert _keys(moved, engine) == reference

    rotated = atoms.copy()
    rotated.rotate(37.0, (1.0, 2.0, 3.0), rotate_cell=True)
    assert _keys(rotated, engine) == reference

    order = np.random.default_rng(3).permutation(len(atoms))
    permuted = atoms[order]
    assert _keys(permuted, engine) == [reference[i] for i in order]
