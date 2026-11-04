"""Typed EnvTok site tokens and their reference interchange representation.

A full site token is the typed tuple

``(central_element, moment, neighbours, geometry, csm_bin, dmin_bin, asym_bin)``

where ``neighbours`` is the canonical neighbour composition as sorted
(element, count) pairs. The historical string form
``E_spin_nb_poly_f5_dmin_abin`` is generated only for interoperability. Some
ChemEnv geometry symbols contain underscores (``BO_1:8``, ``BS_2:10``, ...),
so the string is parsed by anchoring three fields at the start and three at
the end; naive underscore splitting is wrong.

Undefined bins and moments are ``None`` internally and map losslessly to the
reference sentinel ``999`` (bins) or ``NA`` (moment field disabled).
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from envtok.errors import UnsupportedInputError
from envtok.profiles.reference import UNDEFINED_SENTINEL

INVALID = "INVALID"
"""Neighbour/geometry placeholder for sites without any usable neighbour set."""
MOMENT_DISABLED = "NA"
CN_UNKNOWN = "NA"
UNRELIABLE_GEOMETRY = "UNCL"
CATCHALL_PREFIX = "__CATCHALL__|"

Composition = tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True, order=True)
class SiteToken:
    """A full EnvTok site token.

    Attributes:
        element: Central element symbol.
        moment: Rounded absolute initial moment, or ``None`` when the moment
            field is disabled for the run.
        neighbours: Sorted (element, count) composition of the chosen
            neighbour set; ``None`` when no usable neighbour set exists.
        geometry: ChemEnv symbol such as ``"O:6"``, ``"UNKNOWN:<cn>"`` for a
            known neighbour set without a fitted polyhedron, or ``"INVALID"``.
        csm_bin: Continuous-symmetry-measure bin, ``None`` if undefined.
        dmin_bin: Minimum normalised contact distance bin, ``None`` if undefined.
        asym_bin: Asymmetry bin, ``None`` if undefined.
    """

    element: str
    moment: int | None
    neighbours: Composition | None
    geometry: str
    csm_bin: int | None
    dmin_bin: int | None
    asym_bin: int | None

    @property
    def coordination_number(self) -> int | None:
        """Coordination number, or ``None`` for an invalid or empty neighbour set."""
        if not self.neighbours:
            return None
        return sum(count for _, count in self.neighbours)

    @property
    def neighbour_string(self) -> str:
        """Reference neighbour field: sorted elements joined by ``-``."""
        return neighbour_string(self.neighbours)

    def key(self) -> str:
        """Return the reference interchange string (the canonical token key)."""
        spin = MOMENT_DISABLED if self.moment is None else str(self.moment)
        return "_".join(
            (
                self.element,
                spin,
                self.neighbour_string,
                self.geometry,
                _bin_text(self.csm_bin),
                _bin_text(self.dmin_bin),
                _bin_text(self.asym_bin),
            )
        )

    @classmethod
    def from_key(cls, key: str) -> SiteToken:
        """Parse a reference interchange string.

        Args:
            key: A string ``E_spin_nb_poly_f5_dmin_abin``.

        Returns:
            The typed token.

        Raises:
            UnsupportedInputError: If the string does not follow the grammar.
        """
        parts = key.split("_")
        if len(parts) < 7:  # noqa: PLR2004 - seven-field grammar
            raise UnsupportedInputError(f"token {key!r} has fewer than seven fields")
        element, spin, nb = parts[0], parts[1], parts[2]
        geometry = "_".join(parts[3:-3])
        if not element or not geometry:
            raise UnsupportedInputError(
                f"token {key!r} has an empty element or geometry"
            )
        try:
            moment = None if spin == MOMENT_DISABLED else int(spin)
            bins = [_parse_bin(text) for text in parts[-3:]]
        except ValueError:
            raise UnsupportedInputError(
                f"token {key!r} has non-integer fields"
            ) from None
        neighbours = parse_neighbours(nb)
        if neighbour_string(neighbours) != nb:
            raise UnsupportedInputError(
                f"token {key!r} lists its neighbours out of canonical (sorted) order"
            )
        return cls(
            element=element,
            moment=moment,
            neighbours=neighbours,
            geometry=geometry,
            csm_bin=bins[0],
            dmin_bin=bins[1],
            asym_bin=bins[2],
        )


def _bin_text(value: int | None) -> str:
    return str(UNDEFINED_SENTINEL if value is None else value)


def _parse_bin(text: str) -> int | None:
    value = int(text)
    return None if value == UNDEFINED_SENTINEL else value


def composition(elements: list[str] | None) -> Composition | None:
    """Return the sorted (element, count) composition of a neighbour list.

    Args:
        elements: Neighbour element symbols, or ``None`` for no neighbour set.

    Returns:
        The canonical composition, or ``None``.
    """
    if elements is None:
        return None
    return tuple(sorted(Counter(elements).items()))


def neighbour_string(neighbours: Composition | None) -> str:
    """Return the reference neighbour field for a composition.

    Args:
        neighbours: Canonical composition or ``None``.

    Returns:
        ``"INVALID"`` for ``None``, ``""`` for an empty set, otherwise the
        sorted element list joined by ``-`` (``"Cl-Cl-O"``).
    """
    if neighbours is None:
        return INVALID
    return "-".join(element for element, count in neighbours for _ in range(count))


def parse_neighbours(text: str) -> Composition | None:
    """Invert :func:`neighbour_string`.

    Args:
        text: A reference neighbour field.

    Returns:
        The canonical composition, ``()`` for ``""`` and ``None`` for ``INVALID``.
    """
    if text == INVALID:
        return None
    if not text:
        return ()
    return composition(text.split("-"))


def multiplicity_bin(count: int) -> str:
    """Return the reference neighbour-multiplicity bin label.

    Args:
        count: Number of neighbours of one element.

    Returns:
        ``"1"``, ``"2"``, ``"34"``, ``"56"`` or ``"7p"``.
    """
    if count == 1:
        return "1"
    if count == 2:  # noqa: PLR2004 - bin edge
        return "2"
    if count <= 4:  # noqa: PLR2004 - bin edge
        return "34"
    if count <= 6:  # noqa: PLR2004 - bin edge
        return "56"
    return "7p"


def signature_fields(token: SiteToken) -> dict[str, str]:
    """Return every field usable in rung and presence signatures.

    The derived fields reproduce the reference SQL expressions exactly:

    * ``poly_reliable``: the geometry when the CSM bin is 0 or 1, else ``UNCL``
      (including undefined CSM).
    * ``nb_binned``: sorted ``element:multiplicity-bin`` labels.
    * ``nbset``: sorted distinct neighbour elements.
    * ``cn_sig``: the coordination number, or ``NA`` when no usable neighbour
      set exists, so placeholders cannot masquerade as one-coordinate sites.

    The derived fields operate on the reference neighbour string, so the
    ``INVALID`` and empty placeholders project exactly as in the reference
    (``nbset`` = ``INVALID``, ``nb_binned`` = ``INVALID:1``).

    Args:
        token: The full token.

    Returns:
        Field name to signature text.
    """
    nb = token.neighbour_string
    elements = nb.split("-")
    distinct = sorted(set(elements))
    f5 = _bin_text(token.csm_bin)
    csm = token.csm_bin
    return {
        "elem": token.element,
        "spin": MOMENT_DISABLED if token.moment is None else str(token.moment),
        "nb": nb,
        "poly": token.geometry,
        "f5": f5,
        "dmin": _bin_text(token.dmin_bin),
        "a_bin": _bin_text(token.asym_bin),
        "poly_reliable": token.geometry
        if csm is not None and csm < 2  # noqa: PLR2004 - CSM reliability edge
        else UNRELIABLE_GEOMETRY,
        "nb_binned": "-".join(
            sorted(f"{e}:{multiplicity_bin(elements.count(e))}" for e in distinct)
        ),
        "nbset": "-".join(distinct),
        "cn_sig": CN_UNKNOWN if nb in {INVALID, ""} else str(len(elements)),
    }


def signature(fields: Mapping[str, str], names: tuple[str, ...]) -> str:
    """Join the named fields with ``|`` to form a rung or presence signature.

    Args:
        fields: Output of :func:`signature_fields`.
        names: Ordered field names of the rung or family.

    Returns:
        The signature text.
    """
    return "|".join(fields[name] for name in names)
