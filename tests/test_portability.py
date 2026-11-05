"""Platform-dependent code paths, exercised on every operating system."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import CORPUS

from envtok.adapters import ExtxyzAdapter, RecordOptions
from envtok.execution import peak_rss_bytes


def fake_resource(peak: int) -> SimpleNamespace:
    return SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _who: SimpleNamespace(ru_maxrss=peak)
    )


@pytest.mark.parametrize(("platform", "scale"), [("linux", 1024), ("darwin", 1)])
def test_peak_rss_units(
    platform: str, scale: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "resource", fake_resource(2048))
    monkeypatch.setattr(sys, "platform", platform)
    assert peak_rss_bytes() == 2048 * scale


def test_peak_rss_unavailable_without_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "resource", None)
    assert peak_rss_bytes() is None


def test_crlf_extxyz_reads_identically(tmp_path: Path) -> None:
    crlf = tmp_path / "crlf.extxyz"
    crlf.write_bytes(CORPUS.read_bytes().replace(b"\n", b"\r\n"))
    options = RecordOptions(dataset="x", revision="1")
    plain = ExtxyzAdapter(CORPUS, options)
    windows = ExtxyzAdapter(crlf, options)
    locators = list(windows.locators())
    assert len(locators) == len(list(plain.locators()))
    for index in (0, 10, len(locators) - 1):
        seeked = windows.read(locators[index])
        reference = plain.read(list(plain.locators())[index])
        assert (
            seeked.atoms.get_chemical_symbols()
            == reference.atoms.get_chemical_symbols()
        )
        np.testing.assert_allclose(seeked.atoms.positions, reference.atoms.positions)
        assert seeked.metadata == reference.metadata
    streamed = [r.atoms.info["sid"] for r in windows.iter_records()]
    assert streamed == [r.atoms.info["sid"] for r in plain.iter_records()]
