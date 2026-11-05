"""Metadata encoding shared by extended-XYZ writers."""

import json
from collections.abc import Mapping
from typing import Any


def encode_string_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    """Use ASE JSON values to preserve strings and adjacent metadata keys.

    ASE otherwise writes ``key= next=value`` and reads ``next=value`` as the
    first key's value. It also coerces numeric-looking strings such as source
    row IDs into numbers. JSON preserves both empty and nonempty string types.
    """
    return {
        key: f"_JSON {json.dumps(value)}" if isinstance(value, str) else value
        for key, value in info.items()
    }
