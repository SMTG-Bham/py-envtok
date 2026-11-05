"""In-memory ASE iterable adapter and spooling to a managed source store.

This is the smallest integration point for user code. A sequence (``list`` or
``tuple``) of atoms, or a zero-argument factory returning a fresh iterable,
can be iterated again for resume and export. A one-shot iterator cannot:
spool it with :func:`spool_atoms` (or ``Corpus.from_atoms(..., spool_to=...)``)
when the selected structures must be exported later. Otherwise only
manifest-only operation is available.
"""

import hashlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, override

import ase.io
from ase import Atoms

from envtok._extxyz import encode_string_metadata
from envtok.adapters.base import Adapter, Locator, RecordOptions, make_record
from envtok.errors import RetrievalUnavailableError, UnsupportedInputError
from envtok.records import SourceRef, StructureRecord

AtomsSource = Sequence[Atoms] | Callable[[], Iterable[Atoms]] | Iterable[Atoms]
SHARD = "memory"


def _content_digest(frames: Iterable[Atoms]) -> str:
    digest = hashlib.sha256()
    for atoms in frames:
        digest.update(atoms.get_atomic_numbers().tobytes())
        digest.update(atoms.get_positions().tobytes())
        digest.update(atoms.get_cell().array.tobytes())
        digest.update(atoms.get_pbc().tobytes())
    return digest.hexdigest()


class AseIterableAdapter(Adapter):
    """Adapter over atoms supplied by user code.

    Args:
        frames: A sequence, a factory returning a fresh iterable, or a one-shot
            iterable of atoms.
        options: Record options.
    """

    kind = "ase-iterable"

    def __init__(self, frames: AtomsSource, options: RecordOptions) -> None:
        """Wrap the frames."""
        self.options = options
        self._sequence: Sequence[Atoms] | None = None
        self._factory: Callable[[], Iterable[Atoms]] | None = None
        self._once: Iterator[Atoms] | None = None
        self._one_shot = False
        if isinstance(frames, Sequence):
            self._sequence = frames
        elif callable(frames):
            self._factory = frames
        else:
            self._once = iter(frames)
            self._one_shot = True

    @property
    @override
    def supports_random_access(self) -> bool:
        """Records live in this process only, so workers receive them pickled."""
        return False

    @property
    @override
    def supports_retrieval(self) -> bool:
        """Sequences and factories can be re-read; one-shot iterators cannot."""
        return not self._one_shot

    @property
    @override
    def retrieval_limitation(self) -> str | None:
        """Explain why a one-shot iterator cannot be exported."""
        if self.supports_retrieval:
            return None
        return (
            "the corpus came from a one-shot iterator, so selected structures "
            "cannot be retrieved. Pass a list or a factory, or spool the frames "
            "with Corpus.from_atoms(..., spool_to=path); the manifest is still valid"
        )

    def _frames(self) -> Iterable[Atoms]:
        if self._sequence is not None:
            return self._sequence
        if self._factory is not None:
            return self._factory()
        if self._once is None:
            raise RetrievalUnavailableError(
                "the one-shot iterator was already consumed; "
                + str(self.retrieval_limitation)
            )
        once, self._once = self._once, None
        return once

    @override
    def spec(self) -> dict[str, Any]:
        """In-memory sources cannot be rebuilt in another process.

        Raises:
            UnsupportedInputError: Always.
        """
        raise UnsupportedInputError("an in-memory ASE iterable has no portable spec")

    @override
    def source_fingerprint(self) -> dict[str, Any]:
        """Fingerprint the record options and, for sequences, the content."""
        content = (
            _content_digest(self._sequence) if self._sequence is not None else None
        )
        return {
            "kind": self.kind,
            "content_sha256": content,
            "options": self.options.as_dict(),
        }

    @override
    def locators(self) -> Iterator[Locator]:
        """Yield one locator per frame (consumes a one-shot iterator)."""
        for ordinal, _ in enumerate(self._frames()):
            yield Locator(ordinal, SHARD, str(ordinal))

    @override
    def read(self, locator: Locator) -> StructureRecord:
        """Read one frame of a sequence.

        Raises:
            UnsupportedInputError: For factories and iterators, which are only
                streamed.
        """
        if self._sequence is None:
            raise UnsupportedInputError(
                "only sequence sources support read(); stream instead"
            )
        atoms = self._sequence[int(locator.key)]
        return make_record(atoms.copy(), locator.shard, locator.key, self.options)

    @override
    def iter_records(self) -> Iterator[StructureRecord]:
        """Stream every frame as a record."""
        for ordinal, atoms in enumerate(self._frames()):
            yield make_record(atoms.copy(), SHARD, str(ordinal), self.options)

    @override
    def retrieve(self, refs: Sequence[SourceRef]) -> Iterator[tuple[SourceRef, Atoms]]:
        """Yield original frames of a sequence or factory.

        Raises:
            RetrievalUnavailableError: For one-shot iterators.
        """
        if not self.supports_retrieval:
            raise RetrievalUnavailableError(str(self.retrieval_limitation))
        wanted = {int(ref.key) for ref in refs}
        found = {i: atoms for i, atoms in enumerate(self._frames()) if i in wanted}
        for ref in refs:
            yield ref, found[int(ref.key)]


def spool_atoms(frames: Iterable[Atoms], path: str | Path) -> str:
    """Write frames to a managed extxyz source store.

    Args:
        frames: Atoms to store, consumed once.
        path: Destination ``.extxyz`` file (overwritten atomically).

    Returns:
        The SHA-256 digest of the written file, usable as a revision.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as handle:
        for atoms in frames:
            frame = atoms.copy()
            frame.calc = atoms.calc
            frame.info = encode_string_metadata(atoms.info)
            ase.io.write(handle, frame, format="extxyz")
    partial.replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()
