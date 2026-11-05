"""Retrieve and reproduce the bounded ColabFit OMol25 training smoke sample.

The public mirror is CC-BY-4.0. No Hugging Face login or gated files are used.
The bundled snapshot is reused by default; --download verifies a fresh preview.
"""

import argparse
import gzip
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from envtok.molecular.data import export_records, from_atoms

REPOSITORY = "colabfit/OMol25_train_4M"
REVISION = "e5f5d2ae9754639feb7f0a1ad46f2ab59955c3e4"
PREVIEW_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=colabfit/OMol25_train_4M"
    "&config=default&split=train&offset=0&length=100"
)
SNAPSHOT_SHA256 = "421f12f9dcf32d0bcaed3f1fe7352ee0f81e8da25d9bdfa49b14a82dcb56b9c7"
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
NAMESPACE = "OMol25:ORCA6:wB97M-V:def2-TZVPD:RI-J:COSX:DEFGRID3"


def _read_url(url: str) -> bytes:
    if not url.startswith("https://"):
        raise ValueError("only HTTPS sources are allowed")
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - HTTPS only
        data = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("sample endpoint exceeded the bounded download size")
    return data


def _frame(entry: dict[str, Any]) -> Atoms:
    row = entry["row"]
    metadata = json.loads(row["property_metadata"])
    if metadata["basis_set"] != "def2-TZVPD":
        raise ValueError("unexpected OMol basis set")
    if metadata["input"] != "RI-J, COSX, DEFGRID3":
        raise ValueError("unexpected OMol calculation settings")
    atoms = Atoms(
        numbers=row["atomic_numbers"],
        positions=row["positions"],
        cell=row["cell"],
        pbc=row["pbc"],
    )
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=row["energy"],
        forces=row["atomic_forces"],
    )
    atoms.info.update(
        {
            "sid": row["property_id"],
            "split": "train",
            "total_charge": metadata["charge"],
            # ColabFit's top-level multiplicity counts duplicate occurrences.
            # OMol's property_metadata.spin is the molecular spin multiplicity.
            "spin_multiplicity": metadata["spin"],
            "label_namespace": NAMESPACE,
            "graph_status": "missing",
            "source_dataset": REPOSITORY,
            "source_revision": REVISION,
            "source_preview_row": entry["row_idx"],
            "omol_metadata": metadata,
            "colabfit_metadata": {
                key: value
                for key, value in row.items()
                if key
                not in {
                    "atomic_numbers",
                    "positions",
                    "cell",
                    "pbc",
                    "energy",
                    "atomic_forces",
                    "property_metadata",
                }
            },
        }
    )
    for scheme in ("lowdin", "mulliken", "nbo"):
        for label in ("charges", "spins"):
            key = f"{scheme}_{label}"
            if key in metadata:
                atoms.new_array(key, np.asarray(metadata[key], dtype=float))
    return atoms


def main() -> None:
    """Build a reproducible, deliberately nonrepresentative training sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/molecular/omol"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "preview.json.gz"
    if args.download:
        revision_info = json.loads(
            _read_url(f"https://huggingface.co/api/datasets/{REPOSITORY}")
        )
        if revision_info["sha"] != REVISION:
            raise ValueError(
                "mirror revision changed; inspect before updating the sample"
            )
        payload = _read_url(PREVIEW_URL)
    else:
        payload = gzip.decompress(snapshot.read_bytes())
    checksum = hashlib.sha256(payload).hexdigest()
    if checksum != SNAPSHOT_SHA256:
        raise ValueError(
            "preview differs from the reviewed snapshot; inspect before use"
        )
    if args.download:
        snapshot.write_bytes(gzip.compress(payload, mtime=0))
    entries = json.loads(payload)["rows"]
    if any(entry["truncated_cells"] for entry in entries):
        raise ValueError("preview contains truncated source fields")
    selected = [
        entry
        for entry in entries
        if (
            set(entry["row"]["atomic_numbers"]) <= {1, 6, 7, 8}
            and entry["row"]["nsites"] <= 60  # noqa: PLR2004 - documented pilot scope
            and entry["row"]["method"] == "ωB97M-V"
            and not any(entry["row"]["pbc"])
        )
    ]
    records = list(
        from_atoms(
            (_frame(entry) for entry in selected),
            label_namespace=NAMESPACE,
            dataset=REPOSITORY,
            revision=REVISION,
        )
    )
    output = export_records(records, args.output / "sample.extxyz")
    manifest = {
        "repository": REPOSITORY,
        "observed_repository_revision": REVISION,
        "retrieved_date": "2026-11-05",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "original_dataset": "https://huggingface.co/facebook/OMol25",
        "mirror_dataset": f"https://huggingface.co/datasets/{REPOSITORY}",
        "citation": (
            "Levine et al., The Open Molecules 2025 Dataset, Evaluations, "
            "and Models, arXiv:2505.08762 (2025)"
        ),
        "mirror_citation_doi": "10.60732/b6f9382a",
        "snapshot_url": PREVIEW_URL,
        "snapshot_sha256": checksum,
        "snapshot_bytes": len(payload),
        "snapshot_rows": len(entries),
        "reproducibility": (
            "Viewer API is not revision pinned. Exact saved response SHA-256 "
            "is authoritative; observed repository commit is recorded separately."
        ),
        "sample_filter": "first 100 preview rows; finite CHNO, <=60 atoms, ωB97M-V",
        "sampling_limitation": (
            "Ordered smoke sample, not random or representative; "
            "no scientific pilot or accuracy claims."
        ),
        "source_split": "train",
        "selected_rows": [entry["row_idx"] for entry in selected],
        "selected_property_ids": [entry["row"]["property_id"] for entry in selected],
        "configurations": len(records),
        "atoms": sum(record.natoms for record in records),
        "state_counts": dict(
            Counter(
                f"charge={record.total_charge:g},multiplicity={record.spin_multiplicity}"
                for record in records
            )
        ),
        "label_namespace": NAMESPACE,
        "field_mapping": {
            "total_charge": "property_metadata.charge",
            "spin_multiplicity": "property_metadata.spin",
            "energy_eV": "energy",
            "forces_eV_per_angstrom": "atomic_forces",
        },
        "grouping": (
            "No verified family mapping; package conservatively groups unknown "
            "families by composition and reports uncertainty."
        ),
        "graph": (
            "No supplied graph; geometry fallback unless independently validated "
            "perception is explicitly enabled."
        ),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (args.output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "configurations": len(records),
                "atoms": manifest["atoms"],
            }
        )
    )


if __name__ == "__main__":
    main()
