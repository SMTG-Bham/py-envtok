import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from ase import Atoms

from envtok.errors import ConfigError
from envtok.molecular.benchmarks import (
    Prediction,
    _random_subset,
    compare_selectors,
    evaluate_predictions,
)
from envtok.molecular.pipeline import MolecularPipeline
from envtok.molecular.records import MolecularRecord
from envtok.molecular.selection import Candidate, Target, build_hierarchy
from envtok.records import Label, SourceRef, Split, StructureRecord


def reference(
    name: str,
    natoms: int = 2,
    *,
    charge: float | None = 0.0,
    multiplicity: int | None = 1,
    family: str | None = None,
) -> MolecularRecord:
    atoms = Atoms(
        numbers=[1] * natoms, positions=np.arange(natoms * 3).reshape(natoms, 3)
    )
    record = StructureRecord(
        name,
        SourceRef("test", "1", "file", name),
        atoms,
        Split.TEST,
        labels={
            "energy": Label(np.asarray(1.0), "eV", "energy"),
            "forces": Label(np.ones((natoms, 3)), "eV/Ang", "forces"),
        },
    )
    return MolecularRecord(
        record, charge, multiplicity, label_namespace="test", family_id=family
    )


def prediction(record: MolecularRecord, error: float) -> Prediction:
    return Prediction(
        record.record_id, 1.0 + error, np.ones((record.natoms, 3)) + error
    )


def pipeline(tmp_path: Path) -> MolecularPipeline:
    records = (
        reference("a", family="f1"),
        reference("b", family="f1"),
        reference("c", charge=1, multiplicity=2, family="f2"),
    )
    candidates = (
        Candidate("a", 2, frozenset({"neutral"}), frozenset({"region1"})),
        Candidate("b", 2, frozenset({"neutral"}), frozenset({"region2"})),
        Candidate("c", 2, frozenset({"charged"}), frozenset({"region3"})),
    )
    hierarchy = build_hierarchy(
        {"a": [("neutral",)], "b": [("neutral",)], "c": [("charged",)]}, 1
    )
    descriptors = {
        "a": [("neutral", np.array([1.0, 0.0]))],
        "b": [("neutral", np.array([0.0, 1.0]))],
        "c": [("charged", np.array([1.0, 0.0]))],
    }
    return cast(
        MolecularPipeline,
        SimpleNamespace(
            directory=tmp_path,
            settings={
                "state": {"strict_training_export": True},
                "selection": {"background_reserve_fraction": 0.1},
            },
            source_hash="source-hash",
            fingerprint="fingerprint",
            build_targets=lambda: (
                candidates,
                {"neutral": Target(), "charged": Target()},
                hierarchy,
                SimpleNamespace(
                    targets={
                        row.record_id: row.refinement_targets for row in candidates
                    }
                ),
            ),
            ingest=lambda: SimpleNamespace(candidates=records),
            descriptor_rows=lambda _: descriptors,
        ),
    )


def test_selector_comparison_writes_audited_manifests_and_exports(
    tmp_path: Path,
) -> None:
    source = pipeline(tmp_path)
    report = compare_selectors(
        source, fractions=(0.25, 1.0), seeds=(0, 42), export_subsets=True
    )
    assert report["status"] == "model_accuracy_pending"
    assert report["full_pool"]["selected_configurations"] == 3
    assert len(report["runs"]) == 12
    for run in report["runs"]:
        assert run["achieved_cost"] <= run["budget"]
        if run["fraction"] == 1:
            assert run["complete"]
            assert run["protected_strata_met"]
            assert run["selected_ids"] == ("a", "b", "c")
        else:
            assert not run["complete"]
            assert not run["protected_strata_met"]
    assert (tmp_path / "benchmarks/full_pool/subset.extxyz").is_file()
    saved = json.loads((tmp_path / "benchmarks/comparison.json").read_text())
    assert saved["source_sha256"] == "source-hash"
    assert saved["model_seeds"] is None
    repeat = compare_selectors(source, fractions=(0.25, 1), seeds=(0, 42))
    assert [row["selected_ids"] for row in repeat["runs"]] == [
        row["selected_ids"] for row in report["runs"]
    ]


@pytest.mark.parametrize("fractions", [(), (0,), (1.1,), (float("nan"),)])
def test_invalid_benchmark_fractions(
    tmp_path: Path, fractions: tuple[float, ...]
) -> None:
    with pytest.raises(ConfigError, match="fractions"):
        compare_selectors(pipeline(tmp_path), fractions=fractions)


@pytest.mark.parametrize("seeds", [(), (-1,), (True,), (1.5,)])
def test_invalid_benchmark_seeds(tmp_path: Path, seeds: tuple[object, ...]) -> None:
    with pytest.raises(ConfigError, match="seeds"):
        compare_selectors(pipeline(tmp_path), seeds=cast(tuple[int, ...], seeds))


def test_stratified_random_skips_oversized_and_exhausted_strata() -> None:
    rows = [
        Candidate("big", 100, frozenset(), group_id="one"),
        Candidate("small", 1, frozenset(), group_id="one"),
        Candidate("other", 1, frozenset(), group_id="two"),
        Candidate("missing_group", 1, frozenset()),
    ]
    assert _random_subset(rows, 3, 0) == ("missing_group", "other", "small")
    assert _random_subset([], 0, 0) == ()


def test_prediction_metrics_units_strata_and_family_bootstrap() -> None:
    records = [
        reference("a", family="family"),
        reference("b", family="family"),
        reference("c", natoms=30, charge=None, multiplicity=None),
    ]
    predictions = {
        record.record_id: prediction(record, float(i))
        for i, record in enumerate(records)
    }
    result = evaluate_predictions(records, predictions, bootstrap_samples=20, seed=4)
    assert result["status"] == "predictions_evaluated"
    assert result["aggregate"]["configurations"] == 3
    assert result["aggregate"]["atoms"] == 34
    assert result["aggregate"]["energy"]["mae"] == pytest.approx(1)
    assert result["aggregate"]["energy"]["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert result["aggregate"]["forces"]["rmse"] == pytest.approx(np.sqrt(122 / 34))
    assert result["aggregate"]["forces"]["p99_vector_error"] == pytest.approx(
        np.sqrt(12)
    )
    assert result["by_charge"]["unknown"]["configurations"] == 1
    assert result["by_multiplicity"]["unknown"]["configurations"] == 1
    assert result["family_bootstrap"]["family_count"] == 2
    assert not result["family_bootstrap"]["families_complete"]
    assert result == evaluate_predictions(
        records, predictions, bootstrap_samples=20, seed=4
    )
    assert "family_bootstrap" not in evaluate_predictions(records, predictions)


def test_prediction_large_size_and_complete_families() -> None:
    records = [
        reference("a", natoms=10, family="a"),
        reference("b", natoms=60, family="b"),
    ]
    result = evaluate_predictions(
        records,
        {record.record_id: prediction(record, 0) for record in records},
        bootstrap_samples=2,
    )
    assert result["by_size"][">=60"]["atoms"] == 60
    assert result["family_bootstrap"]["families_complete"]
    assert result["family_bootstrap"]["force_rmse_95ci"] == [0, 0]


def test_prediction_rejects_empty_and_duplicate_or_missing_ids() -> None:
    record = reference("a")
    with pytest.raises(ConfigError, match="at least one"):
        evaluate_predictions([], {})
    with pytest.raises(ConfigError, match="IDs"):
        evaluate_predictions([record, record], {"a": prediction(record, 0)})
    with pytest.raises(ConfigError, match="IDs"):
        evaluate_predictions([record], {})
    with pytest.raises(ConfigError, match="mapping key"):
        evaluate_predictions(
            [record], {"a": replace(prediction(record, 0), record_id="other")}
        )


@pytest.mark.parametrize("samples", [-1, True, 1.5])
def test_invalid_bootstrap_counts(samples: object) -> None:
    record = reference("a")
    with pytest.raises(ConfigError, match="bootstrap_samples"):
        evaluate_predictions(
            [record], {"a": prediction(record, 0)}, bootstrap_samples=cast(int, samples)
        )


@pytest.mark.parametrize(
    "labels", [{}, {"energy": Label(np.asarray(1.0), "eV", "energy")}]
)
def test_prediction_requires_both_reference_labels(labels: dict[str, Label]) -> None:
    record = reference("a")
    record = replace(record, record=replace(record.record, labels=labels))
    with pytest.raises(ConfigError, match="reference energy and forces"):
        evaluate_predictions([record], {"a": prediction(record, 0)})


@pytest.mark.parametrize(("name", "unit"), [("energy", "Ha"), ("forces", "Ha/Bohr")])
def test_prediction_rejects_noncanonical_units(name: str, unit: str) -> None:
    record = reference("a")
    labels = dict(record.labels)
    labels[name] = replace(labels[name], unit=unit)
    record = replace(record, record=replace(record.record, labels=labels))
    with pytest.raises(ConfigError, match="eV/Ang"):
        evaluate_predictions([record], {"a": prediction(record, 0)})


@pytest.mark.parametrize(
    "case",
    [
        "reference_energy_shape",
        "prediction_energy_shape",
        "reference_force_shape",
        "prediction_force_shape",
        "nonfinite_prediction",
        "nonfinite_reference",
    ],
)
def test_prediction_rejects_shapes_and_nonfinite_labels(case: str) -> None:
    record = reference("a")
    pred = prediction(record, 0)
    labels = dict(record.labels)
    if case == "reference_energy_shape":
        labels["energy"] = replace(labels["energy"], value=np.ones(1))
    elif case == "prediction_energy_shape":
        pred = replace(pred, energy=cast(float, np.ones(1)))
    elif case == "reference_force_shape":
        labels["forces"] = replace(labels["forces"], value=np.ones((1, 3)))
    elif case == "prediction_force_shape":
        pred = replace(pred, forces=np.ones((1, 3)))
    elif case == "nonfinite_prediction":
        pred = replace(pred, energy=float("nan"))
    else:
        labels["energy"] = replace(labels["energy"], value=np.asarray(float("inf")))
    record = replace(record, record=replace(record.record, labels=labels))
    with pytest.raises(ConfigError):
        evaluate_predictions([record], {"a": pred})


def test_comparison_uses_regions_even_when_active_selection_disables_refinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = pipeline(tmp_path)
    candidates, mandatory, hierarchy, regions = source.build_targets()
    monkeypatch.setattr(
        source,
        "build_targets",
        lambda: (
            tuple(replace(row, refinement_targets=frozenset()) for row in candidates),
            mandatory,
            hierarchy,
            regions,
        ),
    )
    result = compare_selectors(source, fractions=(1.0,))
    assert all(run["refinement_targets"] == 3 for run in result["runs"])
