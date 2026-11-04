"""Bounded, resumable tokenisation of batches over local worker processes.

The batch plan is deterministic: batch ``i`` holds the records with ordinals
``i * batch_size`` to ``(i + 1) * batch_size - 1`` in the adapter's iteration
order. A SLURM array can therefore process ``--shard-index k --num-shards n``
(every batch with ``i % n == k``) independently, and a later ``index`` stage
merges the committed partitions.

Random-access adapters ship locators to workers, which open their own readers;
other adapters are read sequentially by the parent and ship records. At most
``max_inflight`` batches are queued at any time. Native-library threads are
capped per worker. If a worker process dies, the pool is recreated and every
uncommitted batch is resubmitted, up to ``max_retries`` times.
"""

import contextlib
import multiprocessing
import os
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envtok.adapters import Adapter, Locator, adapter_from_spec
from envtok.errors import EnvTokError
from envtok.fingerprint import digest
from envtok.profiles import CorpusPolicy, Profile
from envtok.records import StructureRecord
from envtok.store.partitions import write_partition
from envtok.tokenizer.geometry import GeometryEngine
from envtok.tokenizer.pipeline import (
    FailureRow,
    RecordStatus,
    StructureRow,
    TokenizedRecord,
    tokenize_record,
)

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class FaultPlan:
    """Deliberate failures for testing crash recovery (never set in production).

    Attributes:
        crash_batches: Batch indices whose worker process exits abruptly once.
        marker_dir: Directory recording which crashes already happened.
        raise_after: Raise in the parent after this many committed batches.
    """

    crash_batches: frozenset[int] = frozenset()
    marker_dir: str | None = None
    raise_after: int | None = None


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Everything a worker needs to tokenise a batch.

    Attributes:
        adapter_spec: Portable adapter description, or ``None`` when records
            are shipped by the parent.
        profile: Scientific profile.
        policy: Corpus policy.
        partitions_dir: Directory receiving committed partitions.
        fingerprint: Cache fingerprint written into commit markers.
        keep_neighbours: Whether to write image-resolved neighbour rows.
        faults: Test-only fault injection.
    """

    adapter_spec: dict[str, Any] | None
    profile: Profile
    policy: CorpusPolicy
    partitions_dir: str
    fingerprint: str
    keep_neighbours: bool
    faults: FaultPlan = field(default_factory=FaultPlan)


@dataclass(frozen=True, slots=True)
class Batch:
    """One unit of work.

    Attributes:
        index: Batch index in the deterministic plan.
        locators: Record locators (random-access adapters).
        records: Pre-read records (sequential adapters).
    """

    index: int
    locators: tuple[Locator, ...] = ()
    records: tuple[StructureRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Outcome counts of one committed batch."""

    index: int
    n_records: int
    n_ok: int
    n_excluded: int
    n_failed: int


_ENGINES: dict[str, GeometryEngine] = {}
_ADAPTERS: dict[str, Adapter] = {}


def _engine(profile: Profile) -> GeometryEngine:
    key = digest([profile.geometry, profile.measures])
    if key not in _ENGINES:
        _ENGINES[key] = GeometryEngine(profile.geometry, profile.measures)
    return _ENGINES[key]


def _adapter(spec: dict[str, Any]) -> Adapter:
    key = digest(spec)
    if key not in _ADAPTERS:
        _ADAPTERS[key] = adapter_from_spec(spec)
    return _ADAPTERS[key]


def _failed_read(
    locator: Locator, adapter: Adapter, exc: BaseException
) -> TokenizedRecord:
    options = adapter.options
    record_id = f"{options.dataset}@{options.revision}:{locator.shard}:{locator.key}"
    row = StructureRow(
        record_id=record_id,
        dataset=options.dataset,
        revision=options.revision,
        shard=locator.shard,
        source_key=locator.key,
        native_id=None,
        natoms=0,
        split=options.split.value,
        group_id=None,
        metadata_json="{}",
        formula="",
        elements=[],
        cell_volume=None,
        status=RecordStatus.FAILED,
        status_reason="read_error",
    )
    failure = FailureRow(
        record_id, None, "read", type(exc).__name__, str(exc)[:500], "none", True
    )
    return TokenizedRecord(structure=row, failures=[failure])


def _tokenize_safely(record: StructureRecord, context: TaskContext) -> TokenizedRecord:
    try:
        return tokenize_record(
            record,
            context.profile,
            context.policy,
            _engine(context.profile),
            keep_neighbours=context.keep_neighbours,
        )
    except Exception as exc:  # noqa: BLE001 - quarantine, never abort the run
        from envtok.tokenizer.pipeline import _base_row  # noqa: PLC0415

        row = _base_row(record)
        row.status = RecordStatus.FAILED
        row.status_reason = "tokenize_error"
        message = "".join(traceback.format_exception_only(exc)).strip()[:500]
        failure = FailureRow(
            record.record_id,
            None,
            "tokenize",
            type(exc).__name__,
            message,
            "none",
            True,
        )
        return TokenizedRecord(structure=row, failures=[failure])


def _maybe_crash(index: int, faults: FaultPlan) -> None:
    if index not in faults.crash_batches or faults.marker_dir is None:
        return
    marker = Path(faults.marker_dir) / f"crashed-{index}"
    if marker.exists():
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    os._exit(17)  # pragma: no cover - the process ends here, before coverage saves


def run_batch(context: TaskContext, batch: Batch) -> BatchSummary:
    """Tokenise one batch and commit its partition (runs in a worker).

    Args:
        context: Shared task context.
        batch: The batch.

    Returns:
        Outcome counts of the batch.
    """
    _maybe_crash(batch.index, context.faults)
    results: list[TokenizedRecord] = []
    if batch.locators:
        assert context.adapter_spec is not None  # noqa: S101 - planner invariant
        adapter = _adapter(context.adapter_spec)
        for locator in batch.locators:
            try:
                record = adapter.read(locator)
            except Exception as exc:  # noqa: BLE001 - unreadable records are quarantined
                results.append(_failed_read(locator, adapter, exc))
                continue
            results.append(_tokenize_safely(record, context))
    else:
        results.extend(_tokenize_safely(record, context) for record in batch.records)
    write_partition(
        Path(context.partitions_dir),
        batch.index,
        results,
        fingerprint=context.fingerprint,
        keep_neighbours=context.keep_neighbours,
    )
    statuses = [r.structure.status for r in results]
    return BatchSummary(
        index=batch.index,
        n_records=len(results),
        n_ok=statuses.count(RecordStatus.OK),
        n_excluded=statuses.count(RecordStatus.EXCLUDED),
        n_failed=statuses.count(RecordStatus.FAILED),
    )


def _chunks(items: Iterable[Any], size: int) -> Iterator[list[Any]]:
    chunk: list[Any] = []
    for item in items:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


@dataclass(slots=True)
class PlanStats:
    """Totals of the batch plan; complete once the plan is exhausted."""

    n_batches: int = 0
    n_records: int = 0


def plan_batches(
    adapter: Adapter,
    batch_size: int,
    selected: Callable[[int], bool],
    stats: PlanStats | None = None,
) -> Iterator[Batch]:
    """Yield the selected batches of the deterministic plan.

    Every batch is enumerated (so ``stats`` counts the whole source), but only
    selected ones are yielded.

    Args:
        adapter: The source adapter.
        batch_size: Records per batch.
        selected: Predicate on the batch index (shard selection and resume).
        stats: Optional totals updated during enumeration.

    Yields:
        Batches carrying locators (random access) or records (sequential).
    """
    stats = stats if stats is not None else PlanStats()
    random_access = adapter.supports_random_access
    items = adapter.locators() if random_access else adapter.iter_records()
    for index, chunk in enumerate(_chunks(items, batch_size)):
        stats.n_batches = index + 1
        stats.n_records += len(chunk)
        if selected(index):
            if random_access:
                yield Batch(index=index, locators=tuple(chunk))
            else:
                yield Batch(index=index, records=tuple(chunk))


@contextlib.contextmanager
def thread_limits(threads: int) -> Iterator[None]:
    """Cap native-library threads for worker processes started in this block.

    Variables already set by the user are respected.
    """
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    for name in THREAD_VARIABLES:
        os.environ.setdefault(name, str(threads))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def execute(
    batches: Iterable[Batch],
    context: TaskContext,
    *,
    workers: int,
    max_inflight: int,
    max_retries: int = 2,
    threads_per_worker: int = 1,
) -> list[BatchSummary]:
    """Run batches serially (``workers <= 1``) or in a bounded process pool.

    Args:
        batches: Batches to run, typically lazily planned.
        context: Shared task context.
        workers: Number of worker processes.
        max_inflight: Maximum number of queued batches.
        max_retries: Pool recreations allowed after worker crashes.
        threads_per_worker: Native-library thread cap per worker.

    Returns:
        Summaries of the committed batches, sorted by index.

    Raises:
        EnvTokError: When workers keep crashing, or when the fault plan asks
            for a deliberate interruption.
    """
    summaries: list[BatchSummary] = []
    raise_after = context.faults.raise_after

    def record(summary: BatchSummary) -> None:
        summaries.append(summary)
        if raise_after is not None and len(summaries) >= raise_after:
            raise EnvTokError(f"deliberate interruption after {len(summaries)} batches")

    if workers <= 1:
        for batch in batches:
            record(run_batch(context, batch))
        return sorted(summaries, key=lambda s: s.index)

    pending = iter(batches)
    retries = 0
    queued: dict[Future[BatchSummary], Batch] = {}
    resubmit: list[Batch] = []
    with thread_limits(threads_per_worker):
        while True:
            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
            try:
                while True:
                    while len(queued) < max_inflight:
                        upcoming = resubmit.pop() if resubmit else next(pending, None)
                        if upcoming is None:
                            break
                        queued[pool.submit(run_batch, context, upcoming)] = upcoming
                    if not queued:
                        return sorted(summaries, key=lambda s: s.index)
                    done, _ = wait(queued, return_when=FIRST_COMPLETED)
                    for future in done:
                        batch = queued.pop(future)
                        try:
                            summary = future.result()
                        except BrokenProcessPool:
                            resubmit.append(batch)
                            raise
                        record(summary)
            except BrokenProcessPool as exc:
                retries += 1
                resubmit.extend(queued.values())
                queued.clear()
                if retries > max_retries:
                    raise EnvTokError(
                        f"worker processes crashed {retries} times; committed "
                        "partitions are kept, rerun to resume"
                    ) from exc
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
