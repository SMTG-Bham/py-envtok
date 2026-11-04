"""A corpus: one source adapter plus its eligibility policy."""

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ase import Atoms

from envtok.adapters import (
    Adapter,
    AseDbAdapter,
    AseIterableAdapter,
    ExtxyzAdapter,
    OMat24Adapter,
    RecordOptions,
    spool_atoms,
)
from envtok.adapters.ase_iterable import AtomsSource
from envtok.profiles import CorpusPolicy, corpus_policy
from envtok.records import Split


@dataclass(slots=True)
class Corpus:
    """Input structures and the policy deciding which of them are eligible.

    Attributes:
        adapter: Source adapter.
        policy: Corpus eligibility policy.
    """

    adapter: Adapter
    policy: CorpusPolicy = field(default_factory=corpus_policy)

    @property
    def name(self) -> str:
        """Dataset name."""
        return self.adapter.options.dataset

    @classmethod
    def from_source(cls, source: str | Path | Mapping[str, Any]) -> Corpus:
        """Build a corpus from a configuration file or mapping.

        Args:
            source: Path to a YAML configuration, or a mapping with ``source``
                and optional ``corpus`` sections.

        Returns:
            The corpus.
        """
        from envtok.config import load_config  # noqa: PLC0415 - avoid import cycle

        return load_config(source).corpus_object()

    @classmethod
    def from_atoms(
        cls,
        frames: AtomsSource,
        *,
        name: str,
        revision: str | None = None,
        split: Split | str = Split.UNSPLIT,
        spool_to: str | Path | None = None,
        policy: CorpusPolicy | None = None,
        **options: Any,
    ) -> Corpus:
        """Wrap atoms supplied by user code.

        Args:
            frames: A sequence, factory or one-shot iterable of atoms.
            name: Dataset name used to namespace record IDs.
            revision: Dataset revision. Defaults to a content digest for
                sequences and spooled data, otherwise ``"unversioned"``.
            split: Split of every record.
            spool_to: Write the frames to this extxyz file and read them back
                from there, making a one-shot iterable exportable and resumable.
            policy: Corpus policy (default: the ``general`` preset).
            **options: Further :class:`RecordOptions` fields.

        Returns:
            The corpus.
        """
        policy = policy or corpus_policy()
        split = Split.parse(split)
        if spool_to is not None:
            source: Iterable[Atoms] = frames() if callable(frames) else frames
            digest = spool_atoms(source, spool_to)
            record_options = RecordOptions(
                dataset=name, revision=revision or digest[:16], split=split, **options
            )
            return cls(ExtxyzAdapter(spool_to, record_options, checksum="full"), policy)
        if revision is None:
            revision = (
                _sequence_digest(frames)
                if isinstance(frames, Sequence)
                else "unversioned"
            )
        record_options = RecordOptions(
            dataset=name, revision=revision, split=split, **options
        )
        return cls(AseIterableAdapter(frames, record_options), policy)

    @classmethod
    def from_extxyz(
        cls, path: str | Path, *, policy: CorpusPolicy | None = None, **kwargs: Any
    ) -> Corpus:
        """Read extended XYZ files (see :class:`ExtxyzAdapter` for keywords)."""
        options, adapter_kwargs = _split_kwargs(kwargs)
        return cls(
            ExtxyzAdapter(path, options, **adapter_kwargs), policy or corpus_policy()
        )

    @classmethod
    def from_ase_db(
        cls, path: str | Path, *, policy: CorpusPolicy | None = None, **kwargs: Any
    ) -> Corpus:
        """Read ASE databases (see :class:`AseDbAdapter` for keywords)."""
        options, adapter_kwargs = _split_kwargs(kwargs)
        return cls(
            AseDbAdapter(path, options, **adapter_kwargs), policy or corpus_policy()
        )

    @classmethod
    def from_omat24(
        cls, path: str | Path, *, policy: CorpusPolicy | None = None, **kwargs: Any
    ) -> Corpus:
        """Read OMat24 ASE-LMDB shards (``dataset`` defaults to ``"omat24"``)."""
        kwargs.setdefault("dataset", "omat24")
        options, adapter_kwargs = _split_kwargs(kwargs)
        return cls(
            OMat24Adapter(path, options, **adapter_kwargs), policy or corpus_policy()
        )


_ADAPTER_KEYWORDS = frozenset({"patterns", "compressed", "staging_dir", "checksum"})


def _split_kwargs(kwargs: dict[str, Any]) -> tuple[RecordOptions, dict[str, Any]]:
    adapter_kwargs = {k: v for k, v in kwargs.items() if k in _ADAPTER_KEYWORDS}
    option_kwargs = {k: v for k, v in kwargs.items() if k not in _ADAPTER_KEYWORDS}
    if isinstance(option_kwargs.get("split"), str):
        option_kwargs["split"] = Split.parse(option_kwargs["split"])
    return RecordOptions(**option_kwargs), adapter_kwargs


def _sequence_digest(frames: Sequence[Atoms]) -> str:
    digest = hashlib.sha256()
    for atoms in frames:
        digest.update(atoms.get_atomic_numbers().tobytes())
        digest.update(atoms.get_positions().tobytes())
        digest.update(atoms.get_cell().array.tobytes())
    return digest.hexdigest()[:16]
