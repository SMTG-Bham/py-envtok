"""Measure coarsening and selection time and peak memory on a synthetic pool.

The geometry pass is excluded: this isolates the stages whose memory
behaviour depends on corpus scale (incidence size and vocabulary). Use it to
estimate resources before a pilot, then confirm on real data.

Usage::

    uv run python build_tools/benchmark_selection.py
"""

import argparse
import json
import time

import numpy as np

from envtok.coarsen import CoverageBuilder, Pool
from envtok.execution import peak_rss_bytes
from envtok.selection import Selector, class_incidence
from envtok.selection.incidence import estimate_selection_bytes
from envtok.tokenizer.tokens import SiteToken, composition


def synthetic_pool(n_records: int, n_tokens: int, seed: int) -> Pool:
    """A pool with a Zipf-like token frequency and 2-30 sites per record."""
    rng = np.random.default_rng(seed)
    elements = ["H", "C", "N", "O", "Na", "Mg", "Al", "Si", "Cl", "Fe", "Cu", "Zn"]
    geometries = ["O:6", "T:4", "S:1", "C:8", "TL:3", "BO_1:8", "UNKNOWN:5"]
    keys: set[str] = set()
    while len(keys) < n_tokens:
        drawn = rng.choice(elements, size=int(rng.integers(1, 9)))
        token = SiteToken(
            str(rng.choice(elements)),
            int(rng.integers(0, 4)),
            composition([str(e) for e in drawn]),
            str(rng.choice(geometries)),
            int(rng.integers(0, 4)),
            int(rng.integers(0, 5)),
            int(rng.integers(0, 3)),
        )
        keys.add(token.key())
    ordered = sorted(keys)
    sites = rng.integers(2, 31, size=n_records)
    record = np.repeat(np.arange(n_records), sites)
    token = np.minimum(rng.zipf(1.3, size=record.size) - 1, n_tokens - 1)
    pairs, counts = np.unique(record * n_tokens + token, return_counts=True)
    edge_record, edge_token = pairs // n_tokens, pairs % n_tokens
    used = np.unique(edge_token)
    remap = np.full(n_tokens, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return Pool(
        record_ids=np.array(
            [f"bench:{i:09d}" for i in range(n_records)], dtype=np.str_
        ),
        natoms=sites.astype(np.int64),
        native_ids=[None] * n_records,
        edge_record=edge_record.astype(np.int64),
        edge_token=remap[edge_token],
        edge_multiplicity=counts.astype(np.int64),
        token_ids=np.arange(used.size, dtype=np.int64),
        token_keys=[ordered[i] for i in used.tolist()],
    )


def main() -> None:
    """Run the benchmark and print a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=int, default=1_000)
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--backend", choices=["memory", "duckdb"], default="memory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.perf_counter()
    pool = synthetic_pool(args.records, args.tokens, args.seed)
    built = time.perf_counter()
    builder = CoverageBuilder(
        min_structure_support=args.threshold, backend=args.backend
    )
    classes = builder.fit_pool(pool)
    fitted = time.perf_counter()
    result = Selector().select(classes)
    selected = time.perf_counter()
    incidence = class_incidence(classes)
    peak_rss = peak_rss_bytes()
    print(
        json.dumps(
            {
                "records": args.records,
                "edges": int(pool.edge_record.size),
                "tokens": len(pool.token_ids),
                "classes": classes.n_classes,
                "class_edges": incidence.n_edges,
                "selected_records": len(result.selected),
                "selected_atoms": result.natoms,
                "seconds": {
                    "build_pool": round(built - started, 2),
                    "coarsen": round(fitted - built, 2),
                    "select": round(selected - fitted, 2),
                },
                "estimated_selection_mib": round(
                    estimate_selection_bytes(
                        incidence.n_rows, incidence.n_edges, incidence.n_items
                    )
                    / 2**20,
                    1,
                ),
                "peak_rss_mib": None
                if peak_rss is None
                else round(peak_rss / 2**20, 1),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
