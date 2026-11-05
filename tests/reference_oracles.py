"""Reference oracles adapted from em819/omat24-chemenv-subset@723696d.

MIT License, Copyright (c) 2026 Evgeny Moerman; see
LICENSES/omat24-chemenv-subset-MIT.txt.

* The SQL fragments below (``POLY_RELIABLE``, ``NB_BINNED``, ``CN_SIG``, the
  identity levels, the modifier chain, the floors and the token parse) are
  copied verbatim from ``04_subset_selection/select_subset.py`` and executed
  in DuckDB to check EnvTok's Python projections byte for byte.
* :func:`reference_demand_mapping` reproduces ``build_demand_mapping`` for
  design ``'a'`` and ``support_metric='materials'`` (the reference defaults).
* :func:`reference_selection` is a typed port of ``forced_selection_cascade``,
  the lazy-greedy core of ``parallel_uniform_n_coverage`` (float32 scores,
  key ``(-score, site_count, row)``), ``prune_redundant_materials`` and
  ``topup_floor_coverage`` with ``n_coverage = 1`` and ``cost_alpha = 1``.
"""

import heapq
from collections import defaultdict
from collections.abc import Sequence

import duckdb
import numpy as np
import numpy.typing as npt

# --------------------------------------------------------------- verbatim SQL
POLY_RELIABLE = ("CASE WHEN TRY_CAST(f5 AS INTEGER) IS NULL "
                 "          OR TRY_CAST(f5 AS INTEGER) >= 2 THEN 'UNCL' "
                 "     ELSE poly END")  # fmt: skip

NB_BINNED = """array_to_string(list_sort(list_transform(
        list_distinct(str_split(nb, '-')),
        e -> e || ':' || CASE
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) = 1 THEN '1'
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) = 2 THEN '2'
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) <= 4 THEN '34'
                 WHEN len(list_filter(str_split(nb, '-'), x -> x = e)) <= 6 THEN '56'
                 ELSE '7p' END)), '-')"""  # fmt: skip

CN_SIG = ("CASE WHEN nb IS NULL OR nb IN ('INVALID', '') THEN 'NA' "
          "ELSE CAST(cn AS VARCHAR) END")  # fmt: skip

R0_FULL_RUNG = (
    "R0_full",
    "elem||'|'||spin||'|'||nb||'|'||poly||'|'||f5||'|'||dmin||'|'||a_bin")  # fmt: skip

_IDENTITY_LEVELS = [
    ("R0",  "",             "elem||'|'||spin||'|'||nb||'|'||poly||'|'||f5"),
    ("R1",  "drop_f5",      "elem||'|'||spin||'|'||nb||'|'||poly"),
    ("R2",  "drop_spin",    "elem||'|'||nb||'|'||poly"),
    ("R2a", "unrel_poly",   "elem||'|'||nb||'|'||poly_reliable"),
    ("R2b", "binned_mult",
     f"elem||'|'||nb_binned||'|'||poly_reliable||'|'||{CN_SIG}"),
    ("R3",  "nbset_poly",
     f"elem||'|'||nbset||'|'||poly_reliable||'|'||{CN_SIG}"),
    ("R4",  "nbset_cn",     f"elem||'|'||nbset||'|'||{CN_SIG}"),
    ("R5",  "nbset",        "elem||'|'||nbset"),
    ("R6",  "elem_poly",    f"elem||'|'||poly_reliable||'|'||{CN_SIG}"),
    ("R7",  "elem_cn",      f"elem||'|'||{CN_SIG}"),
    ("R8",  "elem",         "elem"),
]  # fmt: skip

_MODIFIER_CHAIN = [
    ("dA", "||'|'||dmin||'|'||a_bin"),   # every modifier
    ("A",  "||'|'||a_bin"),              # drop dmin (lower priority survives)
    ("",   ""),                          # identity alone
]  # fmt: skip

FLOOR_SIGS = [
    ("F_elem_poly", "elem||'|'||poly_reliable"),
    ("F_elem_cn",   f"elem||'|'||{CN_SIG}"),
    ("F_elem_cn_dmin", f"elem||'|'||{CN_SIG}||'|'||dmin"),
    ("F_elem_cn_abin", f"elem||'|'||{CN_SIG}||'|'||a_bin"),
]  # fmt: skip

PARSE = """
    SELECT env_index, env_id, elem, spin, nb, poly, f5, cn, nbset,
           dmin, a_bin, {POLY_RELIABLE} AS poly_reliable, {NB_BINNED} AS nb_binned
    FROM (
        SELECT env_index, env_id,
               p[1]                          AS elem,
               p[2]                          AS spin,
               p[3]                          AS nb,
               array_to_string(p[4:-4], '_') AS poly,
               p[-3]                         AS f5,
               p[-2]                         AS dmin,
               p[-1]                         AS a_bin,
               len(str_split(p[3], '-'))     AS cn,
               array_to_string(list_sort(list_distinct(str_split(p[3], '-'))), '-') AS nbset
        FROM (SELECT env_index, env_id, str_split(env_id, '_') AS p FROM env_ids)
    )
"""  # fmt: skip
# ------------------------------------------------------------ end verbatim


def reference_ladder() -> list[tuple[str, str]]:
    """The reference ``_build_ladder()`` output with rung 0 renamed R0_full."""
    out = []
    for suffix, mod in _MODIFIER_CHAIN:
        for prefix, desc, ident in _IDENTITY_LEVELS:
            name = f"{prefix}{suffix}_{desc}" if desc else f"{prefix}{suffix}"
            out.append((name, ident + mod))
    out[0] = R0_FULL_RUNG
    return out


def _connect() -> duckdb.DuckDBPyConnection:
    """Open DuckDB accepting the reference's ``x -> expr`` lambdas.

    The reference ran DuckDB 1.4.2; DuckDB 2 deprecates single-arrow lambdas
    and needs this switch to evaluate the verbatim SQL.
    """
    connection = duckdb.connect()
    connection.execute("SET lambda_syntax = 'ENABLE_SINGLE_ARROW'")
    return connection


def _parsed(connection: duckdb.DuckDBPyConnection, keys: Sequence[str]) -> None:
    connection.execute(
        "CREATE OR REPLACE TABLE env_ids (env_index BIGINT, env_id VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO env_ids VALUES (?, ?)", [(i, key) for i, key in enumerate(keys)]
    )
    query = PARSE.replace("{POLY_RELIABLE}", POLY_RELIABLE).replace(
        "{NB_BINNED}", NB_BINNED
    )
    connection.execute(f"CREATE OR REPLACE TABLE v10_tok AS {query}")


def reference_signatures(keys: Sequence[str]) -> dict[str, list[str]]:
    """Evaluate every rung and floor expression on token keys with DuckDB.

    Args:
        keys: Reference token strings.

    Returns:
        Rung or floor name to the signature of every key (in key order).
    """
    connection = _connect()
    try:
        _parsed(connection, keys)
        result = {}
        for name, expr in [*reference_ladder(), *FLOOR_SIGS]:
            rows = connection.execute(
                f"SELECT ({expr}) FROM v10_tok ORDER BY env_index"
            ).fetchall()
            result[name] = [r[0] for r in rows]
        return result
    finally:
        connection.close()


def reference_demand_mapping(
    keys: Sequence[str],
    incidence: Sequence[tuple[int, int]],
    threshold: int,
    ladder: Sequence[tuple[str, str]] | None = None,
) -> dict[int, tuple[int, str]]:
    """Run the reference coarsening loop (design 'a', materials support).

    Args:
        keys: Token strings; their positions are the ``env_index`` values.
        incidence: Distinct (material, env_index) pairs.
        threshold: T.
        ladder: (name, SQL expression) rungs (default: the reference ladder).

    Returns:
        env_index to (rung, signature); rung -1 marks the catch-all.
    """
    connection = _connect()
    try:
        _parsed(connection, keys)
        connection.execute("CREATE TABLE v10_inc (mat_index BIGINT, env_index BIGINT)")
        connection.executemany("INSERT INTO v10_inc VALUES (?, ?)", list(incidence))
        cols = ("env_index, elem, spin, nb, poly, f5, cn, nbset, "
                "dmin, a_bin, poly_reliable, nb_binned, env_id")  # fmt: skip
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE v10_unres AS SELECT {cols} FROM v10_tok"
        )
        connection.execute("CREATE OR REPLACE TEMP TABLE v10_resolved "
                           "(env_index INTEGER, rung INTEGER, sig VARCHAR)")  # fmt: skip
        for i, (_name, expr) in enumerate(ladder or reference_ladder()):
            n_left = connection.execute("SELECT COUNT(*) FROM v10_unres").fetchone()
            if n_left is None or n_left[0] == 0:
                break
            connection.execute(f"""
                CREATE OR REPLACE TEMP TABLE v10_sup AS
                SELECT sig, COUNT(DISTINCT mat_index) AS tot
                FROM (SELECT i.mat_index, ({expr}) AS sig
                      FROM v10_inc i JOIN v10_unres t USING (env_index))
                GROUP BY 1
            """)
            connection.execute(f"""
                CREATE OR REPLACE TEMP TABLE v10_step AS
                SELECT u.*, ({expr}) AS sig FROM v10_unres u
            """)
            connection.execute(f"""
                INSERT INTO v10_resolved
                SELECT s.env_index, {i}, s.sig
                FROM v10_step s JOIN v10_sup p USING (sig)
                WHERE p.tot >= {threshold}
            """)
            connection.execute(f"""
                CREATE OR REPLACE TEMP TABLE v10_unres AS
                SELECT {cols} FROM v10_step s JOIN v10_sup p USING (sig)
                WHERE p.tot < {threshold}
            """)
        connection.execute("INSERT INTO v10_resolved SELECT env_index, "
                           "-1, '__CATCHALL__|' || elem FROM v10_unres")  # fmt: skip
        rows = connection.execute(
            "SELECT env_index, rung, sig FROM v10_resolved"
        ).fetchall()
    finally:
        connection.close()
    return {int(e): (int(r), str(s)) for e, r, s in rows}


def reference_selection(
    matrix: npt.NDArray[np.float32],
    material_sigs: Sequence[dict[str, set[str]]],
    floor_order: Sequence[str],
) -> dict[str, list[int]]:
    """Port of the reference forced/greedy/prune/top-up selection.

    Args:
        matrix: Dense material-by-class site counts (rows in mat_index order).
        material_sigs: Per material, floor name to the floor signatures its
            tokens carry.
        floor_order: Floor names in top-up order.

    Returns:
        Stage name (``forced``, ``greedy``, ``kept``, ``topup``) to rows.
    """
    n_mat = matrix.shape[0]
    col_sums = matrix.sum(axis=0).astype(np.float32)
    target = np.minimum(np.float32(1), col_sums)
    site_counts = matrix.sum(axis=1).round().astype(np.int64)
    cost_pow = np.power(np.maximum(site_counts, 1).astype(np.float64), 1.0).astype(
        np.float32
    )

    # forced_selection_cascade
    coverage = np.zeros(matrix.shape[1], dtype=np.float64)
    available = col_sums.astype(np.float64)
    selected_mask = np.zeros(n_mat, dtype=bool)
    while True:
        slack = available - np.maximum(0.0, target - coverage)
        forced = ((matrix > slack[None, :]) & (matrix > 0)).any(axis=1) & ~selected_mask
        rows = np.flatnonzero(forced)
        if rows.size == 0:
            break
        contrib = matrix[rows].sum(axis=0).astype(np.float64)
        coverage += contrib
        available -= contrib
        selected_mask[rows] = True
    forced_rows = np.flatnonzero(selected_mask).tolist()
    current = coverage.astype(np.float32)

    # lazy greedy
    deficit = np.maximum(0.0, target - current).astype(np.float32)

    def help_of(row: int) -> np.float32:
        return np.float32(np.minimum(matrix[row], deficit).sum(dtype=np.float64))

    candidates = [r for r in range(n_mat) if not selected_mask[r]]
    heap: list[tuple[float, int, int]] = []
    for row in candidates:
        h = help_of(row)
        if h > 0:
            heap.append((float(-(h / cost_pow[row])), int(site_counts[row]), row))
    heapq.heapify(heap)
    removed = np.zeros(n_mat, dtype=bool)
    greedy_rows: list[int] = []
    n_needy = int((current < target).sum())
    while n_needy > 0 and heap:
        best = -1
        while heap:
            _, _, m = heapq.heappop(heap)
            if removed[m]:
                continue
            h = help_of(m)
            if h <= 0:
                removed[m] = True
                continue
            key = (float(-(h / cost_pow[m])), int(site_counts[m]), m)
            if not heap or key <= heap[0]:
                best = m
                break
            heapq.heappush(heap, key)
        if best < 0:
            break
        greedy_rows.append(best)
        removed[best] = True
        old = current.copy()
        current = current + matrix[best]
        deficit = np.maximum(0.0, target - current).astype(np.float32)
        n_needy -= int(((old < target) & (current >= target)).sum())

    # prune_redundant_materials (largest first, then mat_index)
    chosen = forced_rows + greedy_rows
    cov = matrix[chosen].sum(axis=0).astype(np.float64)
    keep = dict.fromkeys(chosen, True)
    order = sorted(chosen, key=lambda r: (-int(site_counts[r]), r))
    while True:
        changed = 0
        for row in order:
            if not keep[row]:
                continue
            mask = matrix[row] > 0
            if np.all(cov[mask] - matrix[row][mask] >= target[mask]):
                cov -= matrix[row]
                keep[row] = False
                changed += 1
        if changed == 0:
            break
    kept = [r for r in chosen if keep[r]]

    # topup_floor_coverage
    selected = list(kept)
    added_all: list[int] = []
    for floor in floor_order:
        everything = set().union(*(m[floor] for m in material_sigs))
        present = (
            set().union(*(material_sigs[r][floor] for r in selected))
            if selected
            else set()
        )
        need = everything - present
        by_mat: dict[int, set[str]] = defaultdict(set)
        for row, sigs in enumerate(material_sigs):
            hits = sigs[floor] & need
            if hits:
                by_mat[row] |= hits
        added = []
        while need and by_mat:
            m, gain = max(
                ((m, len(s & need)) for m, s in by_mat.items()),
                key=lambda kv: (kv[1], -int(site_counts[kv[0]]), -kv[0]),
            )
            if gain == 0:
                break
            added.append(m)
            need -= by_mat.pop(m)
        selected.extend(added)
        added_all.extend(added)
    return {
        "forced": forced_rows,
        "greedy": greedy_rows,
        "kept": kept,
        "topup": added_all,
    }
