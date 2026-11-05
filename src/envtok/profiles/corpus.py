"""Corpus eligibility presets.

Corpus policy is separate from the descriptor profile: the reference study
excluded lanthanides and actinides, but the general package must not apply
that restriction silently.
"""

import enum
from collections.abc import Iterable
from dataclasses import dataclass

from ase.data import atomic_numbers

from envtok.errors import ConfigError

OMAT24_STUDY_EXCLUDED = frozenset({
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
    "Yb", "Lu", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr",
})  # fmt: skip
"""The exact element set excluded by ``contains_rare_earths`` in the reference."""


class InvalidRecordPolicy(enum.StrEnum):
    """Treatment of records containing sites without any usable neighbour set.

    Attributes:
        QUARANTINE: Such records are not eligible for selection (production
            default). Recovered (CrystalNN) sites remain eligible.
        REFERENCE: Keep them eligible with ``INVALID`` sentinel tokens, as the
            reference corpus did.
    """

    QUARANTINE = "quarantine"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class CorpusPolicy:
    """Which discovered records are eligible.

    Attributes:
        preset: Name of the preset the policy was built from.
        exclude_elements: Records containing any of these symbols are excluded
            before geometry is computed.
        exclude_z_at_least: Records containing an element with Z at least this
            value are excluded (``None`` disables the test).
        invalid_record_policy: Treatment of records with invalid sites.
    """

    preset: str = "general"
    exclude_elements: frozenset[str] = frozenset()
    exclude_z_at_least: int | None = None
    invalid_record_policy: InvalidRecordPolicy = InvalidRecordPolicy.QUARANTINE

    def excluded_symbols(self, symbols: Iterable[str]) -> list[str]:
        """Return the sorted distinct symbols that make a record ineligible.

        Args:
            symbols: Chemical symbols of the record.

        Returns:
            Offending symbols; empty when the record passes the policy.
        """
        found = set()
        for symbol in set(symbols):
            if symbol in self.exclude_elements or (
                self.exclude_z_at_least is not None
                and atomic_numbers[symbol] >= self.exclude_z_at_least
            ):
                found.add(symbol)
        return sorted(found)


PRESETS = ("general", "omat24-study")


def corpus_policy(
    preset: str = "general",
    *,
    exclude_elements: Iterable[str] = (),
    exclude_z_at_least: int | None = None,
    invalid_record_policy: InvalidRecordPolicy | str = InvalidRecordPolicy.QUARANTINE,
) -> CorpusPolicy:
    """Build a corpus policy from a preset plus explicit additions.

    Args:
        preset: ``"general"`` (no exclusions) or ``"omat24-study"`` (the
            reference's La-Lu and Ac-Lr exclusion).
        exclude_elements: Additional excluded symbols.
        exclude_z_at_least: Optional atomic-number exclusion, for example 89
            for a paper-literal "Z >= 89" policy.
        invalid_record_policy: Treatment of records with invalid sites.

    Returns:
        The resolved policy.

    Raises:
        ConfigError: If the preset or an element symbol is unknown.
    """
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown corpus preset {preset!r}; expected one of {PRESETS}"
        )
    extra = frozenset(exclude_elements)
    unknown = sorted(symbol for symbol in extra if symbol not in atomic_numbers)
    if unknown:
        raise ConfigError(f"unknown element symbols in exclude_elements: {unknown}")
    base = OMAT24_STUDY_EXCLUDED if preset == "omat24-study" else frozenset()
    return CorpusPolicy(
        preset=preset,
        exclude_elements=base | extra,
        exclude_z_at_least=exclude_z_at_least,
        invalid_record_policy=InvalidRecordPolicy(invalid_record_policy),
    )
