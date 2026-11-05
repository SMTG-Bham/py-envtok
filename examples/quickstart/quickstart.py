"""Run the EnvTok pipeline from Python on the fixture corpus.

Usage::

    uv run python examples/quickstart/quickstart.py
"""

import argparse
import json
from pathlib import Path

import ase.io
import numpy as np
import pyarrow.parquet as pq

from envtok import Corpus, CoverageBuilder, Selector, Tokenizer
from envtok.export import MANIFEST_FILES

ROOT = Path(__file__).resolve().parents[2]
WORKDIR = ROOT / "runs" / "quickstart-api"


def verify_outputs(corpus: Corpus, record_ids: list[str], output: Path) -> None:
    """Check persisted IDs and source geometry, atom order, arrays and labels."""
    missing = [name for name in MANIFEST_FILES if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"missing manifest files: {missing}")
    manifest = pq.read_table(output / "selected_manifest.parquet")
    saved_ids = (output / "selected_ids.txt").read_text().splitlines()
    if (
        not record_ids
        or saved_ids != record_ids
        or manifest["record_id"].to_pylist() != record_ids
    ):
        raise RuntimeError("persisted selection IDs do not match the selection")
    originals = {
        record.record_id: record.atoms for record in corpus.adapter.iter_records()
    }
    frames = ase.io.read(output / "subset.extxyz", index=":")
    for record_id, frame in zip(record_ids, frames, strict=True):
        original = originals[record_id]
        if frame.info.get("envtok_record_id") != record_id:
            raise RuntimeError("exported record identity/order changed")
        np.testing.assert_array_equal(frame.numbers, original.numbers)
        np.testing.assert_array_equal(frame.pbc, original.pbc)
        np.testing.assert_allclose(frame.cell, original.cell, rtol=0, atol=1e-8)
        for key, value in original.arrays.items():
            np.testing.assert_allclose(frame.arrays[key], value, rtol=0, atol=1e-8)
        for key in ("sid", "fixture", "method", "charge", "split"):
            if frame.info.get(key) != original.info.get(key):
                raise RuntimeError(f"exported source metadata changed: {key}")
        for key, value in original.calc.results.items():
            np.testing.assert_allclose(
                frame.calc.results[key], value, rtol=0, atol=1e-8
            )


def main() -> None:
    """Tokenise, fit classes, select, validate, write the manifest and export."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    workdir = parser.parse_args().workdir.resolve()
    corpus = Corpus.from_extxyz(
        ROOT / "tests" / "fixtures" / "corpus.extxyz",
        dataset="fixtures",
        revision="v1",
        split="train",
    )
    cache = Tokenizer(profile="reference-723696d").run(corpus, workdir, batch_size=8)
    classes = CoverageBuilder(min_structure_support=2).fit(cache, split="train")
    result = Selector(mode="cover", cost="atoms").select(classes)
    report = result.validate()
    out = result.write_manifest(workdir / "selection")
    result.export(out / "subset.extxyz", format="extxyz")
    verify_outputs(corpus, result.record_ids, out)
    summary = {key: report[key] for key in ("structures", "atoms", "classes")}
    summary["export_round_trip"] = "passed"
    print(json.dumps(summary, indent=1))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
