"""Metadata encoding shared by extended-XYZ writers."""

import json
from collections.abc import Mapping
from typing import Any

import numpy as np


def encode_string_metadata(info: Mapping[str, Any]) -> dict[str, Any]:
    """Use ASE JSON values to preserve strings, empty arrays and adjacent keys.

    ASE otherwise writes ``key= next=value`` and reads ``next=value`` as the
    first key's value. It also coerces numeric-looking strings such as source
    row IDs into numbers. Empty one-dimensional arrays have the same empty
    assignment problem; ASE reads an explicit JSON list back as an empty array.
    JSON also preserves both empty and nonempty string types.
    """
    return {
        key: (
            f"_JSON {json.dumps(value)}"
            if isinstance(value, str)
            else "_JSON []"
            if isinstance(value, np.ndarray) and value.shape == (0,)
            else value
        )
        for key, value in info.items()
    }
