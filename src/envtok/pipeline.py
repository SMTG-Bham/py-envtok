"""Stage orchestration used by the command-line interface.

Each selection is written to its own configuration-addressed directory
``<workdir>/selection-T<T>-<fingerprint>``, so runs with different settings
never overwrite or resume each other. ``run.json`` in that directory records
what is needed to reload the selection for reporting and export.
"""

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from envtok.coarsen import CoverageBuilder, CoverageClasses, eligible_mask
from envtok.config import EnvTokConfig, load_config
from envtok.corpus import Corpus
from envtok.errors import ConfigError
from envtok.records import Split, validate_record
from envtok.selection import (
    Selected,
    SelectionResult,
    Selector,
    combine_strata,
    random_subset,
    strata_values,
)
from envtok.store.cache import CACHE_FILE, TokenCache, cache_keys
from envtok.tokenization import Tokenizer

LOGGER = logging.getLogger("envtok")
RUN_FILE = "run.json"


class Pipeline:
    """Run EnvTok stages from a validated configuration.

    Args:
        config: Configuration, YAML path or mapping.
    """

    def __init__(self, config: EnvTokConfig | str | Path | dict[str, Any]) -> None:
        """Load the configuration."""
        self.config = (
            config if isinstance(config, EnvTokConfig) else load_config(config)
        )
        self.profile = self.config.profile()
        self.execution = self.config.execution_settings()
        self._corpus: Corpus | None = None

    @property
    def corpus(self) -> Corpus:
        """The configured corpus (opened lazily)."""
        if self._corpus is None:
            self._corpus = self.config.corpus_object()
        return self._corpus

    @property
    def workdir(self) -> Path:
        """The run work directory."""
        return self.config.workdir

    def tokenizer(self) -> Tokenizer:
        """Return the configured tokenizer."""
        return Tokenizer(
            self.profile,
            cache_raw_measurements=self.config.tokenizer.cache_raw_measurements,
            execution=self.execution,
        )

    def builder(self, min_support: int | None = None) -> CoverageBuilder:
        """Return the configured class builder."""
        coverage = self.config.coverage
        return CoverageBuilder(
            self.profile,
            coverage.min_structure_support if min_support is None else min_support,
            invalid_record_policy=self.config.corpus.invalid_record_policy,
            backend=coverage.backend,
            execution=self.execution,
            workdir=self.workdir,
        )

    def selector(self, required_records: list[str] | None = None) -> Selector:
        """Return the configured selector.

        Args:
            required_records: Override the configured required records (used
                to give each protected stratum only its own records).
        """
        s = self.config.selection
        return Selector(
            s.mode,
            s.cost,
            prune=s.prune,
            presence_constraints=s.presence_constraints,
            tie_break=s.tie_break,
            required_records=s.required_records
            if required_records is None
            else required_records,
            prune_after_repair=s.prune_after_repair,
            algorithm=s.algorithm,
            execution=self.execution,
        )

    def cache_dir(self) -> Path:
        """The cache directory of the configured source and profile."""
        keys = cache_keys(
            self.corpus.adapter,
            self.profile,
            self.corpus.policy,
            batch_size=self.execution.batch_size,
            keep_neighbours=self.config.tokenizer.cache_raw_measurements,
        )
        return self.workdir / "cache" / keys.cache[:16]

    def inspect(self, sample: int = 100) -> dict[str, Any]:
        """Validate a sample of records before scheduling a large run.

        Args:
            sample: Number of leading records to read and validate.

        Returns:
            A summary of the sample and the source fingerprint.
        """
        issues: dict[str, int] = {}
        ids: set[str] = set()
        duplicates = 0
        natoms: list[int] = []
        elements: set[str] = set()
        labels: dict[str, int] = {}
        splits: dict[str, int] = {}
        moments = 0
        for count, record in enumerate(self.corpus.adapter.iter_records()):
            if count >= sample:
                break
            duplicates += record.record_id in ids
            ids.add(record.record_id)
            natoms.append(record.natoms)
            elements.update(record.atoms.get_chemical_symbols())
            moments += bool(record.atoms.has("initial_magmoms"))
            splits[record.split.value] = splits.get(record.split.value, 0) + 1
            for name in record.labels:
                labels[name] = labels.get(name, 0) + 1
            for issue in validate_record(record):
                key = f"{issue.severity.value}:{issue.code}"
                issues[key] = issues.get(key, 0) + 1
        excluded = self.corpus.policy.excluded_symbols(elements)
        return {
            "adapter": self.corpus.adapter.kind,
            "sampled": len(natoms),
            "duplicate_ids": duplicates,
            "natoms": {"min": min(natoms, default=0), "max": max(natoms, default=0)},
            "elements": sorted(elements),
            "excluded_elements_present": excluded,
            "labels": labels,
            "splits": splits,
            "records_with_initial_moments": moments,
            "issues": issues,
            "supports_retrieval": self.corpus.adapter.supports_retrieval,
            "source_fingerprint": self.corpus.adapter.source_fingerprint(),
            "cache_dir": str(self.cache_dir()),
        }

    def tokenize(self, *, shard_index: int = 0, num_shards: int = 1) -> Path:
        """Run (or resume) tokenisation of one shard."""
        return self.tokenizer().tokenize(
            self.corpus,
            self.workdir,
            shard_index=shard_index,
            num_shards=num_shards,
            resume=self.config.execution.resume,
        )

    def index(self) -> TokenCache:
        """Merge every committed partition into the token index."""
        cache_dir = self.cache_dir()
        if not (cache_dir / CACHE_FILE).exists():
            raise ConfigError(
                f"nothing tokenised yet in {cache_dir}; run tokenize first"
            )
        return self.tokenizer().index(cache_dir, self.workdir)

    def cache(self) -> TokenCache:
        """Tokenise (resuming unless disabled) and index the corpus.

        An already indexed cache is reused even with ``resume: false``, so
        selections and sweeps never repeat finished work.
        """
        existing = self.cache_dir()
        if not self.config.execution.resume and (existing / "plan.json").exists():
            return self.index()
        return self.tokenizer().run(
            self.corpus, self.workdir, resume=self.config.execution.resume
        )

    def fit_and_select(
        self, cache: TokenCache, min_support: int | None = None
    ) -> SelectionResult:
        """Fit the classes and select, stratified if configured.

        Args:
            cache: Indexed cache.
            min_support: Override the threshold T.

        Returns:
            The selection.
        """
        builder = self.builder(min_support)
        split = Split.parse(self.config.coverage.split)
        selector = self.selector()
        global_classes = builder.fit(cache, split)
        keys = self.config.selection.stratify_by
        if not keys:
            return selector.select(global_classes)
        structures = cache.read("structures")
        mask = eligible_mask(structures, split, builder.invalid_record_policy)
        metadata = [
            json.loads(m)
            for m, keep in zip(
                structures["metadata_json"].to_pylist(), mask.tolist(), strict=True
            )
            if keep
        ]
        required = set(self.config.selection.required_records)
        placed: set[str] = set()
        parts = []
        for stratum in strata_values(metadata, keys):
            classes = builder.fit(cache, split, where=stratum)
            mine = sorted(required & set(classes.pool.record_ids.tolist()))
            placed.update(mine)
            parts.append((stratum, self.selector(mine).select(classes)))
        if required - placed:
            raise ConfigError(
                f"required records are not eligible: {sorted(required - placed)[:5]}"
            )
        return combine_strata(
            global_classes, parts, {**selector.settings, "stratify_by": list(keys)}
        )

    def selection_dir(self, result: SelectionResult) -> Path:
        """The configuration-addressed output directory of a selection."""
        return (
            self.workdir
            / f"selection-T{result.classes.min_support}-{result.fingerprint[:8]}"
        )

    def select(self, min_support: int | None = None) -> tuple[SelectionResult, Path]:
        """Fit, select, validate and write the manifest.

        Args:
            min_support: Override the threshold T.

        Returns:
            The selection and its output directory.
        """
        cache = self.cache()
        result = self.fit_and_select(cache, min_support)
        directory = self.selection_dir(result)
        resolved = self.config.resolved()
        resolved["coverage"]["min_structure_support"] = result.classes.min_support
        result.write_manifest(directory, config=resolved)
        (directory / RUN_FILE).write_text(
            json.dumps(
                {
                    "config": resolved,
                    "token_dir": str(cache.token_dir),
                    "min_structure_support": result.classes.min_support,
                },
                indent=1,
                default=str,
            )
            + "\n"
        )
        result.validate()
        LOGGER.info("selection written to %s", directory)
        return result, directory

    def export(
        self, result: SelectionResult, directory: Path, path: Path | None = None
    ) -> Path | None:
        """Export the selected structures in the configured format.

        Returns:
            The written file, or ``None`` when ``output.format`` is ``none``.
        """
        output = self.config.output
        if output.format == "none":
            return None
        suffix = ".extxyz" if output.format == "extxyz" else ".db"
        target = path or directory / f"subset{suffix}"
        return result.export(target, format=output.format, annotate=output.annotate)

    def run(self, min_support: int | None = None) -> Path:
        """Run every stage and export the subset.

        Returns:
            The selection directory.
        """
        result, directory = self.select(min_support)
        self.export(result, directory)
        return directory

    def sweep(self, thresholds: Sequence[int]) -> Path:
        """Select for several thresholds, reusing one token cache.

        T is swept empirically: the procedure is not nested, so retained atoms
        need not change monotonically with T.

        Args:
            thresholds: Values of T.

        Returns:
            Path of ``sweep_summary.json``.
        """
        rows = []
        for threshold in thresholds:
            _, directory = self.select(threshold)
            report = json.loads((directory / "coverage.json").read_text())
            rows.append(
                {
                    "min_structure_support": threshold,
                    "directory": str(directory),
                    "classes": report["classes"]["required"],
                    "selected_structures": report["structures"]["selected"],
                    "selected_atoms": report["atoms"]["selected"],
                    "structure_retention": report["structures"]["retention"],
                    "atom_retention": report["atoms"]["retention"],
                    "full_token_coverage": report["full_tokens"]["fraction"],
                    "frequency_weighted_coverage": report["full_tokens"][
                        "frequency_weighted"
                    ],
                }
            )
        summary = self.workdir / "sweep_summary.json"
        summary.write_text(json.dumps(rows, indent=1) + "\n")
        return summary


def load_selection(directory: str | Path) -> tuple[Pipeline, SelectionResult]:
    """Reload a written selection for reporting or export.

    The classes are re-fitted from the cached tokens with the recorded
    configuration (fitting is deterministic), and the selected records and
    reasons are read from the manifest.

    Args:
        directory: A selection directory written by :meth:`Pipeline.select`.

    Returns:
        The pipeline and the reloaded selection.
    """
    run = json.loads((Path(directory) / RUN_FILE).read_text())
    pipeline = Pipeline(run["config"])
    cache = TokenCache.open(run["token_dir"], pipeline.profile)
    result = pipeline.fit_and_select(cache, run["min_structure_support"])
    manifest = pq.read_table(Path(directory) / "selected_manifest.parquet")
    index = {rid: i for i, rid in enumerate(result.classes.pool.record_ids.tolist())}
    stored = [
        Selected(index[rid], rid, reason, order, natoms)
        for rid, reason, order, natoms in zip(
            manifest["record_id"].to_pylist(),
            manifest["reason"].to_pylist(),
            manifest["order"].to_pylist(),
            manifest["natoms"].to_pylist(),
            strict=True,
        )
    ]
    if [s.record_id for s in stored] != result.record_ids:
        LOGGER.warning(
            "re-running the selection gave a different subset; using the manifest"
        )
    result.selected = stored
    return pipeline, result


def random_baselines(directory: str | Path, seeds: Sequence[int]) -> Path:
    """Draw equal-atom random subsets and audit them against the same classes.

    Args:
        directory: A selection directory; its atom total is the budget.
        seeds: Random seeds.

    Returns:
        Path of ``baselines/summary.json`` inside the selection directory.
    """
    _, result = load_selection(directory)
    classes: CoverageClasses = result.classes
    budget = result.natoms
    out = Path(directory) / "baselines"
    out.mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        rows_selected = random_subset(classes, budget, seed)
        baseline = SelectionResult(
            classes=classes,
            settings={"mode": "random", "seed": seed, "atom_budget": budget},
            selected=[
                Selected(
                    r,
                    str(classes.pool.record_ids[r]),
                    "random",
                    i,
                    int(classes.pool.natoms[r]),
                )
                for i, r in enumerate(rows_selected)
            ],
        )
        from envtok.report import audit_classes  # noqa: PLC0415 - avoid cycle

        report = audit_classes(classes, baseline.record_ids)
        target = out / f"random-seed{seed}"
        target.mkdir(exist_ok=True)
        (target / "selected_ids.txt").write_text(
            "".join(f"{r}\n" for r in baseline.record_ids)
        )
        (target / "coverage.json").write_text(json.dumps(report, indent=1) + "\n")
        rows.append(
            {
                "seed": seed,
                "atoms": baseline.natoms,
                "structures": report["structures"]["selected"],
                "class_coverage": report["classes"]["fraction"],
                "full_token_coverage": report["full_tokens"]["fraction"],
            }
        )
    summary = out / "summary.json"
    summary.write_text(
        json.dumps({"atom_budget": budget, "baselines": rows}, indent=1) + "\n"
    )
    return summary
