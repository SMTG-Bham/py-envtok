"""Execution and resource settings shared by every stage."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

from envtok.errors import ConfigError
from envtok.store.duck import connect

_UNITS = {
    "": 1,
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}


def parse_size(value: str | int | None) -> int | None:
    """Parse a size such as ``"32GB"``, ``"1.5GiB"`` or a byte count.

    Args:
        value: Size text, an integer number of bytes, or ``None``.

    Returns:
        The size in bytes, or ``None``.

    Raises:
        ConfigError: If the text is not a size.
    """
    if value is None or isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", value)
    unit = match.group(2).upper() if match else ""
    if match is None or unit not in _UNITS:
        raise ConfigError(f"cannot parse size {value!r}; use e.g. '32GB' or '512MiB'")
    return int(float(match.group(1)) * _UNITS[unit])


def peak_rss_bytes() -> int | None:
    """Return this process's peak resident set size, or ``None`` if unknown."""
    try:
        import resource  # noqa: PLC0415 - POSIX only
    except ImportError:  # pragma: no cover - Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Parallelism and resource policy.

    Attributes:
        workers: Worker processes for tokenisation (1 runs serially in-process).
        batch_size: Records per batch and partition.
        max_inflight_batches: Bound on queued batches (default ``2 * workers``).
        threads_per_worker: Native-library thread cap per worker process.
        max_batch_retries: Pool recreations allowed after worker crashes.
        memory_budget: Bytes the selection working state may use; ``None``
            disables the check.
        temporary_disk_budget: Bytes DuckDB may spill to disk.
        duckdb_memory_fraction: Fraction of ``memory_budget`` given to DuckDB,
            leaving the rest for Python and NumPy.
        duckdb_threads: DuckDB threads (default: DuckDB's own choice).
    """

    workers: int = 1
    batch_size: int = 32
    max_inflight_batches: int | None = None
    threads_per_worker: int = 1
    max_batch_retries: int = 2
    memory_budget: int | None = None
    temporary_disk_budget: int | None = None
    duckdb_memory_fraction: float = 0.5
    duckdb_threads: int | None = None

    def __post_init__(self) -> None:
        """Validate the settings."""
        if self.workers < 1 or self.batch_size < 1 or self.threads_per_worker < 1:
            raise ConfigError("workers, batch_size and threads_per_worker must be >= 1")
        if self.max_inflight_batches is not None and self.max_inflight_batches < 1:
            raise ConfigError("max_inflight_batches must be >= 1")
        if not 0.0 < self.duckdb_memory_fraction <= 1.0:
            raise ConfigError("duckdb_memory_fraction must lie in (0, 1]")

    @property
    def inflight(self) -> int:
        """The effective in-flight bound."""
        return self.max_inflight_batches or 2 * self.workers

    def duckdb(self, workdir: Path) -> duckdb.DuckDBPyConnection:
        """Open a DuckDB connection honouring the resource policy.

        Args:
            workdir: Run directory; spills go to ``workdir/tmp/duckdb``.

        Returns:
            The connection.
        """
        memory = (
            None
            if self.memory_budget is None
            else f"{int(self.memory_budget * self.duckdb_memory_fraction) // 2**20}MiB"
        )
        disk = (
            None
            if self.temporary_disk_budget is None
            else f"{self.temporary_disk_budget // 2**20}MiB"
        )
        return connect(
            memory_limit=memory,
            threads=self.duckdb_threads,
            temp_directory=workdir / "tmp" / "duckdb",
            max_temp_directory_size=disk,
        )
