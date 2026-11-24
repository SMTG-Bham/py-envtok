"""Optimised molecular selection must reproduce the frozen reference exactly."""

from dataclasses import asdict
from typing import Any

import numpy as np
import pytest
import reference_molecular_selection as reference

from envtok.errors import ConfigError
from envtok.molecular import selection

COSTS = (1.0, 2.0, 3.0, 5.0, 7.0, 0.1, 0.2, 0.25, 1.5, 12.0)
WEIGHTS = (1.0, 1.0, 0.5, 2.5, 10.0, 0.1)


def random_case(seed: int) -> dict[str, Any]:
    """A small pool exercising every selection option and tie-break."""
    rng = np.random.default_rng(seed)
    n_records = int(rng.integers(3, 30))
    vocabulary = [f"t{i}" for i in range(int(rng.integers(2, 12)))]
    regions = [f"g{i}" for i in range(int(rng.integers(0, 10)))]
    groups = [None, "a", "b", "c"]
    records: list[dict[str, Any]] = []
    for index in range(n_records):
        targets = rng.choice(vocabulary, size=int(rng.integers(0, 5))).tolist()
        refinement: list[str] = (
            rng.choice(regions, size=int(rng.integers(0, 4))).tolist()
            if regions
            else []
        )
        records.append(
            {
                "record_id": f"r{index:03d}",
                "cost": float(rng.choice(COSTS)),
                "targets": frozenset(str(t) for t in targets),
                "refinement_targets": frozenset(str(g) for g in refinement),
                "group_id": groups[int(rng.integers(0, len(groups)))],
            }
        )
    tokens = sorted({str(t) for row in records for t in row["targets"]})
    mandatory = None
    if tokens and rng.random() < 0.8:
        chosen = rng.choice(tokens, size=int(rng.integers(1, len(tokens) + 1)))
        mandatory = {
            str(token): (int(rng.integers(1, 4)), float(rng.choice(WEIGHTS)))
            for token in chosen
        }
    ids = [row["record_id"] for row in records]
    bundles = [
        frozenset(str(i) for i in rng.choice(ids, size=int(rng.integers(1, 4))))
        for _ in range(int(rng.integers(0, 3)) if rng.random() < 0.4 else 0)
    ]
    total = sum(row["cost"] for row in records)
    budget_mode = rng.random() < 0.6
    options: dict[str, Any] = {
        "mode": "budget" if budget_mode else "cover",
        "budget": round(float(total * rng.uniform(0.05, 1.0)), 2)
        if budget_mode
        else None,
        "background_reserve_fraction": float(rng.choice([0.0, 0.1, 0.3]))
        if budget_mode
        else 0.0,
        "seed": int(rng.integers(0, 5)),
        "prune": bool(rng.random() < 0.7),
        "bundles": bundles,
    }
    if regions and rng.random() < 0.5:
        options["refinement_weights"] = {
            g: float(rng.choice(WEIGHTS)) for g in regions if rng.random() < 0.5
        }
    return {"records": records, "mandatory": mandatory, "options": options}


def run_both(case: dict[str, Any], **extra: Any) -> tuple[Any, Any]:
    mandatory = case["mandatory"]
    current = selection.select_candidates(
        [selection.Candidate(**row) for row in case["records"]],
        None
        if mandatory is None
        else {t: selection.Target(r, w) for t, (r, w) in mandatory.items()},
        **case["options"],
        **extra,
    )
    frozen = reference.select_candidates(
        [reference.Candidate(**row) for row in case["records"]],
        None
        if mandatory is None
        else {t: reference.Target(r, w) for t, (r, w) in mandatory.items()},
        **case["options"],
    )
    return current, frozen


@pytest.mark.parametrize("algorithm", ["lazy", "transparent"])
@pytest.mark.parametrize("block", range(10))
def test_selection_matches_frozen_reference(block: int, algorithm: str) -> None:
    incomplete = incomplete_budget = 0
    for seed in range(block * 60, (block + 1) * 60):
        current, frozen = run_both(random_case(seed), algorithm=algorithm)
        assert asdict(current) == asdict(frozen), seed
        incomplete += not frozen.complete
        incomplete_budget += not frozen.complete and frozen.mode == "budget"
    # The generator must keep reaching the incomplete-budget branch.
    assert incomplete
    assert incomplete_budget


def test_lazy_matches_on_larger_pools() -> None:
    # Many rounds and stale heap keys, beyond the small randomised pools.
    rng = np.random.default_rng(7)
    rows = [
        selection.Candidate(
            f"r{i:04d}",
            float(rng.integers(5, 61)),
            frozenset(f"t{t}" for t in (rng.zipf(2.0, 20) % 60).tolist()),
            frozenset(f"g{g}" for g in (rng.zipf(1.5, 20) % 300).tolist()),
            f"f{i // 4}",
        )
        for i in range(400)
    ]
    mandatory = {t: selection.Target() for row in rows for t in row.targets}
    budget = 0.5 * sum(row.cost for row in rows)
    results = [
        selection.select_candidates(
            rows,
            mandatory,
            mode="budget",
            budget=budget,
            background_reserve_fraction=0.1,
            algorithm=algorithm,
        )
        for algorithm in ("lazy", "transparent")
    ]
    assert results[0] == results[1]
    assert set(results[0].reasons.values()) == {
        "chemical_core",
        "geometry",
        "background",
    }


def test_unknown_algorithm_rejected() -> None:
    with pytest.raises(ConfigError, match="algorithm"):
        selection.select_candidates([], algorithm="fast")
