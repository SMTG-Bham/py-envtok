"""Shared fixtures: synthetic corpus, golden reference tokens, geometry engine."""

import json
from pathlib import Path
from typing import Any

import ase.io
import pytest
from ase import Atoms
from helpers import CORPUS, GOLDEN, record_of

from envtok.profiles import REFERENCE_723696D, corpus_policy
from envtok.tokenizer.geometry import GeometryEngine
from envtok.tokenizer.pipeline import TokenizedRecord, tokenize_record


@pytest.fixture(scope="session")
def corpus_path() -> Path:
    return CORPUS


@pytest.fixture(scope="session")
def corpus_frames() -> list[Atoms]:
    frames = ase.io.read(CORPUS, index=":", format="extxyz")
    assert isinstance(frames, list)
    return frames


@pytest.fixture(scope="session")
def golden() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GOLDEN.read_text())
    return {entry["sid"]: entry for entry in data["structures"]}


@pytest.fixture(scope="session")
def engine() -> GeometryEngine:
    return GeometryEngine(REFERENCE_723696D.geometry, REFERENCE_723696D.measures)


@pytest.fixture(scope="session")
def tokenized_corpus(
    corpus_frames: list[Atoms], engine: GeometryEngine
) -> dict[str, TokenizedRecord]:
    policy = corpus_policy()
    return {
        str(atoms.info["sid"]): tokenize_record(
            record_of(atoms, str(i)), REFERENCE_723696D, policy, engine
        )
        for i, atoms in enumerate(corpus_frames)
    }
