"""Native OMol acquisition preserves state, labels and nullable metadata."""

import gzip
import hashlib
import runpy
import tarfile
from pathlib import Path

import ase.db
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.molecular.data import iter_extxyz

SAMPLER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "build_tools/sample_omol.py")
)


def _frame(symbols: str = "NHO") -> Atoms:
    atoms = Atoms(symbols)
    atoms.positions = np.arange(len(atoms) * 3).reshape(-1, 3) * 0.37
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-123.4567890123,
        forces=np.arange(len(atoms) * 3).reshape(-1, 3) * -0.123456789,
    )
    atoms.info.update(
        charge=-1,
        spin=2,
        reference_source=None,
        source="native/source/step1/orca.tar.zst",
        data_id="test",
        electronic_metadata={"nullable": None, "values": [1, None, 3]},
        mulliken_charges=np.linspace(-0.8, 0.2, len(atoms)).tolist(),
        mulliken_spins=np.linspace(0.1, 0.9, len(atoms)).tolist(),
    )
    return atoms


def _write_native(path: Path, frames: list[Atoms]) -> None:
    database = ase.db.connect(str(path))
    try:
        for frame in frames:
            database.write(frame, data=dict(frame.info))
    finally:
        database.close()


def test_native_sample_roundtrip_preserves_null_metadata_state_and_labels(
    tmp_path: Path,
) -> None:
    original = _frame()
    native = tmp_path / "source.aselmdb"
    _write_native(native, [original])
    records, lineage, skipped = SAMPLER["sample"](native, "test-revision", count=1)
    output = tmp_path / "sample"
    hashes = SAMPLER["write_sample"](records, output)
    (exported,) = iter_extxyz(
        output / "sample.extxyz",
        label_namespace=SAMPLER["NAMESPACE"],
        unknown_policy="reject",
    )

    assert skipped == []
    assert lineage[0]["source_row_key"] == "1"
    assert exported.atoms.info["omol_native_metadata"] == original.info
    assert "reference_source" not in exported.atoms.info
    assert exported.total_charge == -1
    assert exported.spin_multiplicity == 2
    assert exported.state_key == records[0].state_key
    assert exported.atoms.info["source_row_key"] == "1"
    np.testing.assert_array_equal(exported.atoms.numbers, original.numbers)
    np.testing.assert_array_equal(exported.atoms.pbc, original.pbc)
    np.testing.assert_allclose(exported.atoms.positions, original.positions, atol=1e-12)
    for name in ("mulliken_charges", "mulliken_spins"):
        np.testing.assert_allclose(
            exported.atoms.arrays[name], original.info[name], atol=1e-12
        )
    np.testing.assert_allclose(
        exported.labels["energy"].value, original.calc.results["energy"], atol=1e-12
    )
    np.testing.assert_allclose(
        exported.labels["forces"].value, original.calc.results["forces"], atol=1e-12
    )
    assert exported.labels["energy"].unit == "eV"
    assert exported.labels["forces"].unit == "eV/Ang"
    assert (
        hashes["sample.extxyz"]
        == hashlib.sha256((output / "sample.extxyz").read_bytes()).hexdigest()
    )


def test_native_sample_accounts_for_exclusions_and_stops_after_requested_count(
    tmp_path: Path,
) -> None:
    unsupported = _frame("Ne2")
    oversized = _frame("H4")
    first = _frame("OH")
    unknown_state = _frame("H2")
    unknown_state.info["spin"] = None
    nonfinite_labels = _frame("H2")
    nonfinite_labels.calc.results["forces"][0, 0] = np.nan
    periodic = _frame("H2")
    periodic.set_cell([5, 5, 5])
    periodic.set_pbc(True)
    second = _frame("NH")
    second.info.update(charge=1, spin=1)
    native = tmp_path / "source.aselmdb"
    _write_native(
        native,
        [
            unsupported,
            oversized,
            first,
            unknown_state,
            nonfinite_labels,
            periodic,
            second,
            _frame("H2"),
        ],
    )

    records, lineage, skipped = SAMPLER["sample"](
        native, "test-revision", count=2, max_atoms=3
    )

    assert [row["source_row_key"] for row in lineage] == ["3", "7"]
    assert [record.total_charge for record in records] == [-1, 1]
    assert [record.spin_multiplicity for record in records] == [2, 1]
    assert {row["source_row_key"]: row["reason"] for row in skipped} == {
        "1": "non_CHNO",
        "2": "over_atom_limit",
        "4": "record_validation:unknown_spin_multiplicity",
        "5": "missing_or_nonfinite_labels",
        "6": "periodic",
    }
    assert len(lineage) + len(skipped) == 7


@pytest.mark.parametrize("kind", ["symlink", "oversized"])
def test_native_member_rejects_unsafe_or_oversized_archive_entry(
    tmp_path: Path, kind: str
) -> None:
    archive = tmp_path / "source.tar.gz"
    member = tarfile.TarInfo(SAMPLER["MEMBER"])
    if kind == "symlink":
        member.type = tarfile.SYMTYPE
        member.linkname = "../outside.aselmdb"
    else:
        member.size = SAMPLER["MAX_MEMBER_BYTES"] + 1
    # A header is sufficient: the sampler must reject before reading payloads.
    archive.write_bytes(gzip.compress(member.tobuf() + b"\0" * 1024))
    staged = tmp_path / "staged.aselmdb"

    with pytest.raises(ValueError, match="regular training shard at most 512 MiB"):
        SAMPLER["stage_member"](archive, staged)
    assert not staged.exists()
