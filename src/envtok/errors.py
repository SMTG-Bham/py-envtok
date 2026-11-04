"""Exception hierarchy for EnvTok.

Every error raised deliberately by the package derives from
:class:`EnvTokError`, so callers can catch package failures without masking
programming errors.
"""


class EnvTokError(Exception):
    """Base class for all errors raised by EnvTok."""


class ConfigError(EnvTokError, ValueError):
    """A configuration value is missing, unknown or inconsistent."""


class UnsupportedInputError(EnvTokError, ValueError):
    """An input record or source cannot be processed by the selected profile."""


class StaleCacheError(EnvTokError):
    """A cached stage does not match the current source or configuration."""


class DuplicateRecordError(EnvTokError):
    """Two input records share one record ID."""


class ResourcePolicyError(EnvTokError):
    """A stage would exceed the configured resource policy."""


class RetrievalUnavailableError(EnvTokError):
    """Selected records cannot be retrieved from their source for export."""


class CoverageError(EnvTokError):
    """An independently audited selection does not meet its coverage contract."""
