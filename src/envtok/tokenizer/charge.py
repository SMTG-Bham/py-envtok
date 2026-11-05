"""Net charge of a record for the charge token field (an EnvTok extension).

The reference token has no charge field. A profile with a charge policy adds
the record's net charge to every site token. The charge is read from the
record's scalar metadata (ASE ``info``, or an ASE database row's key-value
pairs) under the profile's ``charge_key``. It is resolved when the token
index is built, from the metadata cached at tokenisation, so enabling,
disabling or re-keying the charge field never repeats the geometry pass.

Every record that reached the geometry pass is accounted for:

* ``supplied``: a finite number was found and is rounded into the token.
* ``defaulted``: no charge was found and the policy treats the record as
  neutral (``net_round_zero_if_missing``).
* ``missing``: no charge was found under ``net_round``. The record fails with
  reason ``missing_charge``.
* ``invalid``: the value is not a finite number (for example a string or a
  boolean). The record fails with reason ``invalid_charge`` under every
  enabled policy: a present but unusable value is a data error, not a
  missing one.
* ``disabled``: the profile has no charge field.

A NaN or infinite charge never reaches the cache, because non-finite floats
are dropped from scalar metadata, so it counts as missing.
"""

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass

from envtok.profiles import ChargePolicy, Profile

MAX_ABS_CHARGE = 2.0**53
"""Largest accepted magnitude: beyond it a float no longer holds integers exactly."""

MISSING_CHARGE = "missing_charge"
INVALID_CHARGE = "invalid_charge"


class ChargeSource(enum.StrEnum):
    """Where a record's charge field came from (``charge_source`` column)."""

    SUPPLIED = "supplied"
    DEFAULTED = "defaulted"
    MISSING = "missing"
    INVALID = "invalid"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class NetCharge:
    """The resolved net charge of one record.

    Attributes:
        raw: The value found in the metadata, as a float, or ``None``.
        value: The net charge used for the token (0 when defaulted), or
            ``None`` when the record has no usable charge or the field is
            disabled.
        source: Where the value came from.
        message: Why the record fails, for ``missing`` and ``invalid``.
    """

    raw: float | None
    value: float | None
    source: ChargeSource
    message: str | None = None

    @property
    def failure(self) -> str | None:
        """The failure category, or ``None`` when the record can be tokenised."""
        if self.source is ChargeSource.MISSING:
            return MISSING_CHARGE
        if self.source is ChargeSource.INVALID:
            return INVALID_CHARGE
        return None


def resolve_net_charge(metadata: Mapping[str, object], profile: Profile) -> NetCharge:
    """Read a record's net charge as the profile's charge policy requires.

    Args:
        metadata: The record's scalar metadata.
        profile: Scientific profile (charge policy and key).

    Returns:
        The resolved charge.
    """
    policy = profile.charge_policy
    if policy is ChargePolicy.DISABLED:
        return NetCharge(None, None, ChargeSource.DISABLED)
    key = profile.charge_key
    value = metadata.get(key)
    if value is None:
        if policy is ChargePolicy.NET_ROUND_ZERO_IF_MISSING:
            return NetCharge(None, 0.0, ChargeSource.DEFAULTED)
        return NetCharge(
            None,
            None,
            ChargeSource.MISSING,
            f"no net charge under metadata key {key!r}; supply it or set "
            "tokenizer.charge_policy: net_round_zero_if_missing to treat records "
            "without one as neutral",
        )
    charge = _finite_number(value)
    if charge is None:
        return NetCharge(
            None,
            None,
            ChargeSource.INVALID,
            f"net charge {value!r} under metadata key {key!r} is not a finite number",
        )
    return NetCharge(charge, charge, ChargeSource.SUPPLIED)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    # Compare the magnitude first: math.isfinite overflows on huge integers.
    if abs(value) >= MAX_ABS_CHARGE or not math.isfinite(value):
        return None
    return float(value)
