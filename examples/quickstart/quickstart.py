"""Run the EnvTok pipeline from Python on the fixture corpus.

Usage::

    uv run python examples/quickstart/quickstart.py
"""

import json
from pathlib import Path

from envtok import Corpus, CoverageBuilder, Selector, Tokenizer

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / "runs" / "quickstart-api"


def main() -> None:
    """Tokenise, fit classes, select, validate, write the manifest and export."""
    corpus = Corpus.from_extxyz(
        ROOT / "tests" / "fixtures" / "corpus.extxyz",
        dataset="fixtures",
        revision="v1",
        split="train",
    )
    cache = Tokenizer(profile="reference-723696d").run(corpus, WORKDIR, batch_size=8)
    classes = CoverageBuilder(min_structure_support=2).fit(cache, split="train")
    result = Selector(mode="cover", cost="atoms").select(classes)
    report = result.validate()
    out = result.write_manifest(WORKDIR / "selection")
    result.export(out / "subset.extxyz", format="extxyz")
    summary = {key: report[key] for key in ("structures", "atoms", "classes")}
    print(json.dumps(summary, indent=1))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
