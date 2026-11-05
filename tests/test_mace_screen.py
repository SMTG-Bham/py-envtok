"""Ensure optional MACE jobs cannot mix held-out IDs or per-method settings."""

import runpy
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

SCREEN = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "build_tools/benchmark_mace.py")
)


def job() -> dict[str, Any]:
    return {
        "train_ids": ["a", "b"],
        "holdout_ids": ["c"],
        "training": deepcopy(SCREEN["TRAINING_SETTINGS"]),
    }


@pytest.mark.parametrize(
    "train_ids", [[], ["a", "a"], ["c"], list(map(str, range(128)))]
)
def test_job_rejects_duplicate_leaking_or_oversized_pool(train_ids: list[str]) -> None:
    config = job()
    config["train_ids"] = train_ids
    with pytest.raises(ValueError, match="disjoint"):
        SCREEN["validate_job"](config)


def test_job_requires_matching_settings_for_every_selector() -> None:
    config = job()
    SCREEN["validate_job"](config)
    config["training"]["steps"] += 1
    with pytest.raises(ValueError, match="fixed training"):
        SCREEN["validate_job"](config)
