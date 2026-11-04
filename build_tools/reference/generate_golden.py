"""Generate golden reference tokens for the fixture corpus.

Run in the pinned reference environment described in ``README.md``. The output
records, for every atom of every fixture structure, the token and raw fields
produced by the vendored reference functions, together with the exact package
versions and library defaults of the environment that produced them.

Usage::

    REF/bin/python build_tools/reference/generate_golden.py \
        tests/fixtures/corpus.extxyz \
        tests/fixtures/reference/tokens-723696d.json
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import introspect_defaults
import vendored_tokenizer

REFERENCE_COMMIT = "723696de2c26584b8a907ef900690c0caf3c7b68"


def _jsonable(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return value


def main() -> None:
    """Tokenise every frame and write the golden JSON file."""
    corpus, output = Path(sys.argv[1]), Path(sys.argv[2])
    frames = ase.io.read(corpus, index=":", format="extxyz", parallel=False)
    lgf = vendored_tokenizer.make_local_geometry_finder()
    structures = []
    for frame in frames:
        sid = frame.info["sid"]
        np.random.seed(0)  # noqa: NPY002 - the reference uses the legacy global RNG
        rows = vendored_tokenizer.tokenize_atoms(frame, lgf, sid)
        structures.append(
            {
                "sid": sid,
                "skipped": rows is None,
                "sites": None
                if rows is None
                else [{k: _jsonable(v) for k, v in row.items()} for row in rows],
            }
        )
        print(f"{sid}: {'skipped' if rows is None else len(rows)}")
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.split()
    if not freeze:
        freeze = subprocess.run(
            ["uv", "pip", "freeze", "--python", sys.executable],  # noqa: S607
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
    payload = {
        "reference_commit": REFERENCE_COMMIT,
        "corpus": corpus.name,
        "python": sys.version.split()[0],
        "packages": freeze,
        "structures": structures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    if len(sys.argv) != 3:  # noqa: PLR2004
        sys.exit(__doc__)
    del introspect_defaults
    main()
