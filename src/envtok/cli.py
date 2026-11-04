"""Command-line interface: ``envtok <command> --config <file>``."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from envtok.errors import EnvTokError
from envtok.pipeline import Pipeline, load_selection, random_baselines


def _print(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=1, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="envtok",
        description=(
            "Select a smaller set of complete atomistic structures with verified "
            "local-environment coverage."
        ),
    )
    parser.add_argument("--version", action="version", version=version("envtok"))
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    commands = parser.add_subparsers(dest="command", required=True)

    def with_config(name: str, text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=text, description=text)
        sub.add_argument(
            "--config", required=True, type=Path, help="YAML configuration"
        )
        return sub

    inspect = with_config("inspect", "validate a sample of records before a large run")
    inspect.add_argument("--sample", type=int, default=100)
    tokenize = with_config("tokenize", "compute (or resume) the token cache")
    tokenize.add_argument("--shard-index", type=int, default=0)
    tokenize.add_argument("--num-shards", type=int, default=1)
    with_config("index", "merge committed partitions into the global token index")
    for name, text in (
        ("select", "fit classes, select, audit and write the manifest"),
        ("run", "run every stage and export the subset"),
    ):
        sub = with_config(name, text)
        sub.add_argument(
            "--min-support", type=int, help="override coverage.min_structure_support"
        )
    sweep = with_config("sweep", "select for several T, reusing the token cache")
    sweep.add_argument("--min-support", type=int, nargs="+", required=True)

    report = commands.add_parser("report", help="re-run the audit of a selection")
    report.add_argument("--run", required=True, type=Path, help="selection directory")
    export = commands.add_parser("export", help="export the selected structures")
    export.add_argument("--run", required=True, type=Path, help="selection directory")
    export.add_argument("--format", choices=["extxyz", "asedb"], default=None)
    export.add_argument("--output", type=Path, default=None)
    baseline = commands.add_parser("baseline", help="equal-atom random baselines")
    baseline.add_argument("--run", required=True, type=Path, help="selection directory")
    baseline.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser


def _dispatch(arguments: argparse.Namespace) -> Any:  # noqa: PLR0911 - one exit per command
    command = arguments.command
    if command in {"report", "export", "baseline"}:
        if command == "baseline":
            return {"summary": random_baselines(arguments.run, arguments.seeds)}
        pipeline, result = load_selection(arguments.run)
        if command == "report":
            result.write_manifest(arguments.run, config=pipeline.config.resolved())
            return json.loads((arguments.run / "coverage.json").read_text())["classes"]
        fmt = arguments.format or pipeline.config.output.format
        suffix = ".extxyz" if fmt == "extxyz" else ".db"
        target = arguments.output or arguments.run / f"subset{suffix}"
        return {"exported": result.export(target, format=fmt)}
    pipeline = Pipeline(arguments.config)
    if command == "inspect":
        return pipeline.inspect(arguments.sample)
    if command == "tokenize":
        return {
            "cache": pipeline.tokenize(
                shard_index=arguments.shard_index, num_shards=arguments.num_shards
            )
        }
    if command == "index":
        return pipeline.index().summary
    if command == "select":
        _, directory = pipeline.select(arguments.min_support)
        return {"selection": directory}
    if command == "run":
        return {"selection": pipeline.run(arguments.min_support)}
    return {"summary": pipeline.sweep(arguments.min_support)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Arguments (default: ``sys.argv[1:]``).

    Returns:
        Process exit status: 0 on success, 2 for EnvTok errors.
    """
    arguments = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        _print(_dispatch(arguments))
    except EnvTokError as exc:
        sys.stderr.write(f"envtok: error: {exc}\n")
        return 2
    return 0
