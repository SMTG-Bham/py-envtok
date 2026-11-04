"""The tokenisation stage: compute, cache and index EnvTok tokens of a corpus."""

from dataclasses import replace
from pathlib import Path

from envtok.corpus import Corpus
from envtok.execution import ExecutionSettings
from envtok.profiles import Profile, get_profile
from envtok.store.cache import TokenCache, build_index, tokenize
from envtok.store.executor import FaultPlan


class Tokenizer:
    """Compute and cache EnvTok tokens for a corpus.

    Args:
        profile: Profile name or instance.
        cache_raw_measurements: Keep image-resolved neighbour vectors, which
            are needed to change radii or the asymmetry decay length without
            repeating ChemEnv. Per-site raw scalars (CSM, distance ratios,
            asymmetry, raw moment) are always kept, so bins can always change.
        execution: Parallelism and resource settings.
    """

    def __init__(
        self,
        profile: str | Profile = "reference-723696d",
        *,
        cache_raw_measurements: bool = True,
        execution: ExecutionSettings | None = None,
    ) -> None:
        """Resolve the profile and settings."""
        self.profile = get_profile(profile)
        self.keep_neighbours = cache_raw_measurements
        self.execution = execution or ExecutionSettings()

    def _settings(
        self, workers: int | None, batch_size: int | None
    ) -> ExecutionSettings:
        settings = self.execution
        if workers is not None:
            settings = replace(settings, workers=workers)
        if batch_size is not None:
            settings = replace(settings, batch_size=batch_size)
        return settings

    def tokenize(
        self,
        corpus: Corpus,
        workdir: str | Path,
        *,
        workers: int | None = None,
        batch_size: int | None = None,
        shard_index: int = 0,
        num_shards: int = 1,
        faults: FaultPlan | None = None,
    ) -> Path:
        """Run (or resume) the geometry pass for one shard.

        Args:
            corpus: Input corpus.
            workdir: Run work directory.
            workers: Override the number of worker processes.
            batch_size: Override records per batch.
            shard_index: Shard processed by this call.
            num_shards: Total number of shards.
            faults: Test-only fault injection.

        Returns:
            The cache directory.
        """
        return tokenize(
            corpus.adapter,
            self.profile,
            corpus.policy,
            Path(workdir),
            execution=self._settings(workers, batch_size),
            keep_neighbours=self.keep_neighbours,
            shard_index=shard_index,
            num_shards=num_shards,
            faults=faults,
        )

    def index(self, cache_dir: str | Path, workdir: str | Path) -> TokenCache:
        """Merge committed partitions into the global token index.

        Args:
            cache_dir: Directory returned by :meth:`tokenize`.
            workdir: Run work directory.

        Returns:
            The indexed cache.
        """
        return build_index(
            Path(cache_dir),
            self.profile,
            execution=self.execution,
            workdir=Path(workdir),
        )

    def run(
        self,
        corpus: Corpus,
        workdir: str | Path,
        *,
        workers: int | None = None,
        batch_size: int | None = None,
    ) -> TokenCache:
        """Tokenise (resuming committed batches) and index the whole corpus.

        Args:
            corpus: Input corpus.
            workdir: Run work directory.
            workers: Override the number of worker processes.
            batch_size: Override records per batch.

        Returns:
            The indexed cache.
        """
        cache_dir = self.tokenize(
            corpus, workdir, workers=workers, batch_size=batch_size
        )
        return self.index(cache_dir, workdir)


__all__ = ["TokenCache", "Tokenizer"]
