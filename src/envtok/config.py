"""Versioned YAML configuration (schema 1).

Every section is optional except ``source``; unknown keys are rejected so a
misspelt setting can never be silently ignored. Relative paths are resolved
against the configuration file's directory. ``resolved()`` returns the full
configuration with every default filled in, which is written as
``config.resolved.yaml`` next to each selection.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from envtok.corpus import Corpus
from envtok.errors import ConfigError
from envtok.execution import ExecutionSettings, parse_size
from envtok.profiles import MagneticPolicy, Profile, corpus_policy, get_profile
from envtok.records import Split

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """The ``source`` section."""

    adapter: str
    path: str
    revision: str
    name: str | None = None
    split: str = "unsplit"
    split_key: str | None = None
    group_key: str | None = None
    id_policy: str = "namespaced_source_record"
    native_id_key: str | None = "sid"
    label_sources: dict[str, str] | None = None
    label_units: dict[str, str] | None = None
    patterns: list[str] | None = None
    compressed: str = "sequential"
    staging_dir: str | None = None
    checksum: str = "stat"


@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """The ``corpus`` section."""

    preset: str = "general"
    exclude_elements: list[str] = field(default_factory=list)
    exclude_z_at_least: int | None = None
    deduplicate: bool = False
    invalid_record_policy: str = "quarantine"


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    """The ``tokenizer`` section."""

    profile: str = "reference-723696d"
    magnetic_policy: str = MagneticPolicy.SOURCE_ABS_ROUND_ZERO_IF_MISSING.value
    cache_raw_measurements: bool = True
    use_symmetry: bool = True


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    """The ``coverage`` section."""

    min_structure_support: int = 100
    support_unit: str = "record"
    split: str = "train"
    backend: str = "auto"


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """The ``selection`` section."""

    mode: str = "cover"
    cost: str = "atoms"
    prune: bool = True
    presence_constraints: str = "reference"
    tie_break: str = "natoms_then_record_id"
    required_records: list[str] = field(default_factory=list)
    prune_after_repair: bool = False
    stratify_by: list[str] = field(default_factory=list)
    algorithm: str = "lazy"


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """The ``execution`` section."""

    workdir: str = "runs/envtok"
    workers: int = 1
    batch_size: int = 32
    max_inflight_batches: int | None = None
    threads_per_worker: int = 1
    max_batch_retries: int = 2
    memory_budget: str | int | None = None
    temporary_disk_budget: str | int | None = None
    resume: bool = True


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """The ``output`` section."""

    format: str = "extxyz"
    preserve_labels: bool = True
    annotate: bool = True


def _section[T](cls: type[T], data: object, name: str) -> T:
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"section {name!r} must be a mapping")
    allowed = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"unknown keys in {name!r}: {unknown}; allowed: {sorted(allowed)}"
        )
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"section {name!r}: {exc}") from None


@dataclass(frozen=True, slots=True)
class EnvTokConfig:
    """A validated configuration.

    Attributes:
        source: Input source.
        corpus: Eligibility policy.
        tokenizer: Token profile settings.
        coverage: Class-fitting settings.
        selection: Selector settings.
        execution: Parallelism, resources and work directory.
        output: Structure export settings.
        base_dir: Directory relative paths are resolved against.
    """

    source: SourceConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    base_dir: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        """Check settings that the API objects do not validate themselves."""
        if self.corpus.deduplicate:
            raise ConfigError(
                "corpus.deduplicate is not supported: EnvTok never removes records "
                "silently; set it to false"
            )
        if self.coverage.support_unit != "record":
            raise ConfigError("coverage.support_unit must be 'record'")
        if not self.output.preserve_labels:
            raise ConfigError(
                "output.preserve_labels must be true; labels are never dropped"
            )
        if self.output.format not in {"extxyz", "asedb", "none"}:
            raise ConfigError("output.format must be extxyz, asedb or none")

    def resolve(self, path: str) -> Path:
        """Resolve a configured path against :attr:`base_dir`."""
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self.base_dir / candidate

    @property
    def workdir(self) -> Path:
        """The run work directory."""
        return self.resolve(self.execution.workdir)

    def corpus_object(self) -> Corpus:
        """Build the corpus described by ``source`` and ``corpus``."""
        source = self.source
        options: dict[str, Any] = {
            "dataset": source.name
            or ("omat24" if source.adapter == "omat24" else Path(source.path).stem),
            "revision": source.revision,
            "split": Split.parse(source.split),
            "split_key": source.split_key,
            "group_key": source.group_key,
            "id_policy": source.id_policy,
            "native_id_key": source.native_id_key,
            "checksum": source.checksum,
        }
        if source.label_sources is not None:
            options["label_sources"] = source.label_sources
        if source.label_units is not None:
            options["label_units"] = source.label_units
        if source.patterns is not None:
            options["patterns"] = source.patterns
        policy = corpus_policy(
            self.corpus.preset,
            exclude_elements=self.corpus.exclude_elements,
            exclude_z_at_least=self.corpus.exclude_z_at_least,
            invalid_record_policy=self.corpus.invalid_record_policy,
        )
        path = self.resolve(source.path)
        if source.adapter == "extxyz":
            if source.staging_dir is not None:
                options["staging_dir"] = self.resolve(source.staging_dir)
            return Corpus.from_extxyz(
                path, policy=policy, compressed=source.compressed, **options
            )
        if source.adapter == "asedb":
            return Corpus.from_ase_db(path, policy=policy, **options)
        if source.adapter == "omat24":
            return Corpus.from_omat24(path, policy=policy, **options)
        raise ConfigError(
            f"source.adapter must be extxyz, asedb or omat24, got {source.adapter!r}"
        )

    def profile(self) -> Profile:
        """Resolve the profile and apply the documented tokenizer variants."""
        profile = get_profile(self.tokenizer.profile)
        policy = MagneticPolicy(self.tokenizer.magnetic_policy)
        suffixes = []
        if policy is not profile.magnetic_policy:
            profile = replace(profile, magnetic_policy=policy)
            suffixes.append("geometry-only")
        if self.tokenizer.use_symmetry != profile.geometry.use_symmetry:
            geometry = replace(
                profile.geometry, use_symmetry=self.tokenizer.use_symmetry
            )
            profile = replace(profile, geometry=geometry)
            suffixes.append("all-sites")
        if suffixes:
            profile = replace(profile, name="+".join([profile.name, *suffixes]))
        return profile

    def execution_settings(self) -> ExecutionSettings:
        """Build the execution settings."""
        execution = self.execution
        return ExecutionSettings(
            workers=execution.workers,
            batch_size=execution.batch_size,
            max_inflight_batches=execution.max_inflight_batches,
            threads_per_worker=execution.threads_per_worker,
            max_batch_retries=execution.max_batch_retries,
            memory_budget=parse_size(execution.memory_budget),
            temporary_disk_budget=parse_size(execution.temporary_disk_budget),
        )

    def resolved(self) -> dict[str, Any]:
        """Return the configuration with every default filled in."""
        sections = (
            "source",
            "corpus",
            "tokenizer",
            "coverage",
            "selection",
            "execution",
            "output",
        )
        result: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        for name in sections:
            section = getattr(self, name)
            result[name] = {f.name: getattr(section, f.name) for f in fields(section)}
        result["execution"]["workdir"] = str(self.workdir)
        result["source"]["path"] = str(self.resolve(self.source.path))
        return result


def load_config(source: str | Path | Mapping[str, Any]) -> EnvTokConfig:
    """Load and validate a configuration.

    Args:
        source: YAML file path or an already parsed mapping.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: For unknown keys, missing sections or invalid values.
    """
    if isinstance(source, Mapping):
        data: Any = dict(source)
        base = Path.cwd()
    else:
        path = Path(source)
        data = yaml.safe_load(path.read_text())
        base = path.resolve().parent
    if not isinstance(data, Mapping):
        raise ConfigError("the configuration must be a mapping")
    data = dict(data)
    version = data.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}, got {version!r}")
    sections = {
        "source": SourceConfig,
        "corpus": CorpusConfig,
        "tokenizer": TokenizerConfig,
        "coverage": CoverageConfig,
        "selection": SelectionConfig,
        "execution": ExecutionConfig,
        "output": OutputConfig,
    }
    unknown = sorted(set(data) - set(sections))
    if unknown:
        raise ConfigError(f"unknown sections: {unknown}; allowed: {sorted(sections)}")
    if "source" not in data:
        raise ConfigError("the configuration needs a 'source' section")
    parsed: dict[str, Any] = {
        name: _section(cls, data.get(name), name) for name, cls in sections.items()
    }
    return EnvTokConfig(base_dir=base, **parsed)
