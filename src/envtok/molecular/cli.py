"""Route molecular profile commands through their independent pipeline."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from envtok.errors import ConfigError
from envtok.molecular.pipeline import MolecularPipeline

STAGES = ("ingest", "split", "featurize", "build-targets", "audit", "benchmark")


def is_molecular(path: Path) -> bool:
    """Recognise the molecular profile without changing periodic config parsing."""
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration: {exc}") from exc
    return isinstance(data, dict) and data.get("profile") == "molecular_v1"


def supports(arguments: argparse.Namespace) -> bool:
    """Check an explicit config or a saved run before dispatching."""
    if arguments.command in STAGES:
        return True
    config = getattr(arguments, "config", None)
    if config is not None:
        return is_molecular(config)
    run = getattr(arguments, "run", None)
    return (
        run is not None
        and (run / "config.resolved.yaml").exists()
        and is_molecular(run / "config.resolved.yaml")
    )


def dispatch(arguments: argparse.Namespace) -> Any:  # noqa: PLR0911 - stage dispatch
    """Execute one molecular stage and return a concise machine-readable result."""
    command = arguments.command
    config = getattr(arguments, "config", None)
    pipeline = (
        MolecularPipeline.load_run(arguments.run)
        if config is None
        else MolecularPipeline(config)
    )
    min_support = getattr(arguments, "min_support", None)
    if min_support is not None:
        if isinstance(min_support, list):
            raise ConfigError("molecular sweep is not supported; use benchmark")
        data = pipeline.settings
        data["hierarchy"]["min_source_support"] = min_support
        pipeline = MolecularPipeline(data)
    if command in {"ingest", "split", "inspect"}:
        prepared = pipeline.ingest()
        return {
            "records": len(prepared.records),
            "candidates": len(prepared.candidates),
            "exclusions": len(prepared.exclusions),
            "manifest": pipeline.directory / "ingestion.json",
        }
    if command in {"featurize", "tokenize"}:
        if (
            getattr(arguments, "num_shards", 1) != 1
            or getattr(arguments, "shard_index", 0) != 0
        ):
            raise ConfigError(
                "molecular pilot featurization is serial; sharding is not implemented"
            )
        return {
            "records": len(pipeline.featurize()),
            "features": pipeline.feature_directory,
        }
    if command in {"build-targets", "index"}:
        candidates, mandatory, _, _ = pipeline.build_targets()
        return {
            "candidates": len(candidates),
            "mandatory_targets": len(mandatory),
            "manifest": pipeline.directory / "targets.json",
        }
    if command == "select":
        result = pipeline.select()
        return {
            "selection": pipeline.directory,
            "complete": result.complete,
            "selected_atoms": result.achieved_cost,
            "unmet_constraints": len(result.unmet),
        }
    if command == "run":
        directory = pipeline.run()
        report = json.loads((directory / "coverage.json").read_text())
        return {
            "selection": directory,
            "complete": report["complete"],
            "selected_atoms": report["achieved_cost"],
            "unmet_constraints": len(report["unmet_constraints"]),
        }
    if command in {"audit", "report"}:
        return pipeline.audit()
    if command == "export":
        if arguments.format not in {None, "extxyz"}:
            raise ConfigError("molecular export supports extxyz only")
        return {"exported": pipeline.export(arguments.output)}
    if command in {"benchmark", "baseline"}:
        from envtok.molecular.benchmarks import compare_selectors  # noqa: PLC0415 - optional stage

        summary = compare_selectors(pipeline, seeds=tuple(arguments.seeds))
        destination = pipeline.directory / "benchmarks.json"
        destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return {"summary": destination}
    raise ConfigError(f"command {command!r} is not supported for molecular_v1")
