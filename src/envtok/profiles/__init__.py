"""Versioned scientific profiles and corpus presets."""

from envtok.errors import ConfigError
from envtok.profiles.corpus import (
    OMAT24_STUDY_EXCLUDED,
    CorpusPolicy,
    InvalidRecordPolicy,
    corpus_policy,
)
from envtok.profiles.radii import CORDERO_2008, RadiusTable, table_drift
from envtok.profiles.reference import (
    NET_CHARGE_LADDER,
    REFERENCE_723696D,
    REFERENCE_COMMIT,
    REFERENCE_LADDER,
    REFERENCE_PRESENCE_FAMILIES,
    SIGNATURE_FIELDS,
    UNDEFINED_SENTINEL,
    BinSettings,
    ChargePolicy,
    GeometrySettings,
    MagneticPolicy,
    MeasureSettings,
    PresenceFamily,
    Profile,
    Rung,
    with_net_charge,
)

_PROFILES = {REFERENCE_723696D.name: REFERENCE_723696D}


def get_profile(profile: str | Profile) -> Profile:
    """Resolve a profile name or pass a profile instance through.

    Args:
        profile: A registered profile name or a :class:`Profile`.

    Returns:
        The profile.

    Raises:
        ConfigError: If the name is not registered.
    """
    if isinstance(profile, Profile):
        return profile
    try:
        return _PROFILES[profile]
    except KeyError:
        raise ConfigError(
            f"unknown profile {profile!r}; available: {sorted(_PROFILES)}"
        ) from None


__all__ = [
    "CORDERO_2008",
    "NET_CHARGE_LADDER",
    "OMAT24_STUDY_EXCLUDED",
    "REFERENCE_723696D",
    "REFERENCE_COMMIT",
    "REFERENCE_LADDER",
    "REFERENCE_PRESENCE_FAMILIES",
    "SIGNATURE_FIELDS",
    "UNDEFINED_SENTINEL",
    "BinSettings",
    "ChargePolicy",
    "CorpusPolicy",
    "GeometrySettings",
    "InvalidRecordPolicy",
    "MagneticPolicy",
    "MeasureSettings",
    "PresenceFamily",
    "Profile",
    "RadiusTable",
    "Rung",
    "corpus_policy",
    "get_profile",
    "table_drift",
    "with_net_charge",
]
