"""Independent coverage audit, compression metrics and distribution diagnostics.

The audit recomputes coverage from the selected record IDs, the cached
record-token incidence and the frozen class mapping with DuckDB SQL. It shares
no code with the selector's incidence arrays or greedy loop. It audits
against the class universe fitted on the full eligible pool; classes are never
rebuilt on the selected subset.

Complete class coverage does not imply complete full-token coverage, so both
are reported, along with atom- and structure-based retention, per-rung
resolution loss and processing quality.
"""

import html
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from envtok.errors import CoverageError
from envtok.profiles import InvalidRecordPolicy
from envtok.records import Split
from envtok.store.duck import connect, sql_literal

if TYPE_CHECKING:
    from envtok.coarsen import CoverageClasses
    from envtok.selection import SelectionResult

MISSING_EXAMPLES = 20


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def _scalar(connection: Any, query: str) -> Any:
    return connection.execute(query).fetchall()[0][0]


def audit_classes(classes: CoverageClasses, record_ids: list[str]) -> dict[str, Any]:
    """Recompute coverage of ``classes`` by a list of selected record IDs.

    Args:
        classes: Frozen classes of the eligible pool.
        record_ids: Selected record IDs.

    Returns:
        Coverage metrics (see the module docstring).

    Raises:
        CoverageError: If a selected record is not in the eligible pool or is
            listed twice.
    """
    cache = classes.cache
    pool = classes.pool
    token_class = pa.table(
        {
            "token_id": pa.array(pool.token_ids, pa.int64()),
            "class_id": pa.array(classes.token_class, pa.int64()),
        }
    )
    class_table = pa.table(
        {
            "class_id": pa.array(np.arange(classes.n_classes), pa.int64()),
            "signature": pa.array(classes.class_signature, pa.string()),
        }
    )
    presence = pa.table(
        {
            "family": [f for f in classes.token_presence for _ in pool.token_ids],
            "token_id": np.concatenate([pool.token_ids for _ in classes.token_presence])
            if classes.token_presence
            else np.zeros(0, dtype=np.int64),
            "signature": [
                classes.presence_signatures[f][i]
                for f, index in classes.token_presence.items()
                for i in index.tolist()
            ],
        }
    )
    connection = connect()
    try:
        connection.register(
            "pool_records", pa.table({"record_id": pool.record_ids.tolist()})
        )
        connection.register(
            "selected", pa.table({"record_id": pa.array(record_ids, pa.string())})
        )
        connection.register("token_class", token_class)
        connection.register("class_table", class_table)
        connection.register("presence", presence)
        tokens_path = sql_literal(cache.path("structure_tokens"))
        structures_path = sql_literal(cache.path("structures"))
        connection.execute(
            "CREATE VIEW pool_edges AS SELECT st.* FROM read_parquet("
            f"{tokens_path}) st JOIN pool_records USING (record_id)"
        )
        connection.execute(
            "CREATE VIEW selected_edges AS SELECT st.* FROM read_parquet("
            f"{tokens_path}) st JOIN selected USING (record_id)"
        )
        duplicates = _scalar(
            connection, "SELECT count(*) - count(DISTINCT record_id) FROM selected"
        )
        foreign = connection.execute(
            "SELECT record_id FROM selected EXCEPT SELECT record_id FROM pool_records "
            "ORDER BY record_id LIMIT 5"
        ).fetchall()
        if duplicates or foreign:
            raise CoverageError(
                f"selection lists {duplicates} duplicate IDs and records outside the "
                f"eligible pool: {[r[0] for r in foreign]}"
            )
        atoms = connection.execute(
            f"SELECT sum(s.natoms) FILTER (WHERE s.record_id IN (SELECT record_id "
            f"FROM selected)), sum(s.natoms) FROM read_parquet({structures_path}) s "
            "JOIN pool_records USING (record_id)"
        ).fetchall()[0]
        covered_classes = connection.execute(
            "SELECT DISTINCT tc.class_id FROM selected_edges e JOIN token_class tc "
            "USING (token_id)"
        ).fetchall()
        missing_classes = connection.execute(
            "SELECT signature FROM class_table WHERE class_id NOT IN (SELECT DISTINCT "
            "tc.class_id FROM selected_edges e JOIN token_class tc USING (token_id)) "
            f"ORDER BY class_id LIMIT {MISSING_EXAMPLES}"
        ).fetchall()
        n_pool_tokens = _scalar(
            connection, "SELECT count(DISTINCT token_id) FROM pool_edges"
        )
        n_selected_tokens = _scalar(
            connection, "SELECT count(DISTINCT token_id) FROM selected_edges"
        )
        weighted = connection.execute(
            "SELECT sum(multiplicity) FILTER (WHERE token_id IN (SELECT token_id FROM "
            "selected_edges)), sum(multiplicity) FROM pool_edges"
        ).fetchall()[0]
        families = {}
        for family in classes.presence_signatures:
            name = sql_literal(family)
            required = _scalar(
                connection,
                f"SELECT count(DISTINCT signature) FROM presence WHERE family = {name}",
            )
            present = connection.execute(
                "SELECT DISTINCT p.signature FROM selected_edges e JOIN presence p "
                f"USING (token_id) WHERE p.family = {name}"
            ).fetchall()
            absent = connection.execute(
                f"SELECT DISTINCT signature FROM presence WHERE family = {name} "
                "EXCEPT SELECT DISTINCT p.signature FROM selected_edges e "
                f"JOIN presence p USING (token_id) WHERE p.family = {name} ORDER BY 1 "
                f"LIMIT {MISSING_EXAMPLES}"
            ).fetchall()
            families[family] = {
                "required": int(required),
                "covered": len(present),
                "fraction": _ratio(len(present), required),
                "missing_examples": [r[0] for r in absent],
            }
    finally:
        connection.close()
    selected_atoms = int(atoms[0] or 0)
    pool_atoms = int(atoms[1] or 0)
    return {
        "structures": {
            "selected": len(record_ids),
            "eligible": len(pool.record_ids),
            "retention": _ratio(len(record_ids), len(pool.record_ids)),
        },
        "atoms": {
            "selected": selected_atoms,
            "eligible": pool_atoms,
            "retention": _ratio(selected_atoms, pool_atoms),
        },
        "classes": {
            "required": classes.n_classes,
            "covered": len(covered_classes),
            "fraction": _ratio(len(covered_classes), classes.n_classes),
            "missing_examples": [r[0] for r in missing_classes],
        },
        "full_tokens": {
            "eligible": int(n_pool_tokens),
            "covered": int(n_selected_tokens),
            "fraction": _ratio(n_selected_tokens, n_pool_tokens),
            "frequency_weighted": _ratio(weighted[0] or 0, weighted[1] or 0),
        },
        "presence": families,
    }


def processing_quality(classes: CoverageClasses) -> dict[str, Any]:
    """Summarise exclusions, failures, fallbacks and eligibility policies.

    Args:
        classes: Fitted classes (their cache and fit split are used).

    Returns:
        Processing-quality metrics.
    """
    cache = classes.cache
    connection = connect()
    split = sql_literal(classes.split.value)
    structures = sql_literal(cache.path("structures"))
    try:
        status = connection.execute(
            f"SELECT status, coalesce(status_reason, ''), count(*) FROM read_parquet("
            f"{structures}) GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        failures = connection.execute(
            "SELECT stage, category, fallback, fatal, count(*) FROM read_parquet("
            f"{sql_literal(cache.path('failures'))}) GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        sites = connection.execute(
            "SELECT site_status, neighbour_source, count(*) FROM read_parquet("
            f"{sql_literal(cache.path('site_tokens'))}) GROUP BY ALL ORDER BY ALL"
        ).fetchall()
        records = connection.execute(
            "SELECT count(*) FILTER (WHERE symmetry_status = 'fallback'), "
            "count(*) FILTER (WHERE chemenv_status = 'failed'), "
            "count(*) FILTER (WHERE moment_source = 'defaulted'), "
            "count(*) FILTER (WHERE NOT has_energy), "
            "count(*) FILTER (WHERE NOT has_forces), "
            f"count(*) FILTER (WHERE status = 'ok' AND split = {split}), "
            f"count(*) FILTER (WHERE status = 'ok' AND split = {split} "
            "AND n_invalid_sites = 0) "
            f"FROM read_parquet({structures})"
        ).fetchall()[0]
    finally:
        connection.close()
    return {
        "records_by_status": [
            {"status": s, "reason": r or None, "count": int(n)} for s, r, n in status
        ],
        "failures": [
            {
                "stage": a,
                "category": b,
                "fallback": c,
                "fatal": bool(d),
                "count": int(n),
            }
            for a, b, c, d, n in failures
        ],
        "sites": [
            {"status": a, "neighbour_source": b, "count": int(n)} for a, b, n in sites
        ],
        "symmetry_fallback_records": int(records[0]),
        "chemenv_failed_records": int(records[1]),
        "defaulted_moment_records": int(records[2]),
        "records_missing_energy": int(records[3]),
        "records_missing_forces": int(records[4]),
        "eligible_by_policy": {
            InvalidRecordPolicy.REFERENCE.value: int(records[5]),
            InvalidRecordPolicy.QUARANTINE.value: int(records[6]),
            "active": classes.invalid_record_policy.value,
        },
    }


def held_out_novelty(classes: CoverageClasses) -> dict[str, Any]:
    """Out-of-vocabulary rates of held-out splits under exact token lookup.

    Evaluation records are mapped onto the frozen classes; they never modify
    them.

    Args:
        classes: Fitted classes.

    Returns:
        Per held-out split: records, sites and the OOV site fraction.
    """
    cache = classes.cache
    site_tokens = sql_literal(cache.path("site_tokens"))
    structures = sql_literal(cache.path("structures"))
    tokens = sql_literal(cache.path("tokens"))
    connection = connect()
    try:
        rows = connection.execute(
            "SELECT s.split, t.token_key, count(*) AS sites, "
            f"count(DISTINCT s.record_id) FROM read_parquet({site_tokens}) st "
            f"JOIN read_parquet({structures}) s USING (record_id) "
            f"JOIN read_parquet({tokens}) t USING (token_id) "
            f"WHERE s.status = 'ok' AND s.split <> {sql_literal(classes.split.value)} "
            "GROUP BY ALL ORDER BY ALL"
        ).fetchall()
    finally:
        connection.close()
    result: dict[str, Any] = {}
    for split in (s.value for s in Split if s is not classes.split):
        entries = [(key, int(n)) for sp, key, n, _ in rows if sp == split]
        if not entries:
            continue
        assigned = classes.assign([key for key, _ in entries])
        sites = sum(n for _, n in entries)
        oov = sum(
            n for (_, n), cls in zip(entries, assigned, strict=True) if cls is None
        )
        result[split] = {
            "sites": sites,
            "oov_sites": oov,
            "oov_fraction": _ratio(oov, sites),
        }
    return result


def _distribution(connection: Any, query: str) -> list[dict[str, Any]]:
    return [
        {"value": value, "eligible": int(a), "selected": int(b)}
        for value, a, b in connection.execute(query).fetchall()
    ]


def diagnostics(classes: CoverageClasses, record_ids: list[str]) -> dict[str, Any]:
    """Compare input and selected distributions.

    A cover deliberately does not preserve the sampling distribution; these
    tables show how it differs.

    Args:
        classes: Fitted classes.
        record_ids: Selected record IDs.

    Returns:
        Distribution tables and label summaries.
    """
    cache = classes.cache
    connection = connect()
    try:
        connection.register(
            "pool_records", pa.table({"record_id": classes.pool.record_ids.tolist()})
        )
        connection.register(
            "selected", pa.table({"record_id": pa.array(record_ids, pa.string())})
        )
        structures = sql_literal(cache.path("structures"))
        connection.execute(
            "CREATE VIEW records AS SELECT s.*, s.record_id IN (SELECT record_id FROM "
            f"selected) AS chosen FROM read_parquet({structures}) s "
            "JOIN pool_records USING (record_id)"
        )
        connection.execute(
            "CREATE VIEW sites AS SELECT st.*, t.dmin_bin, t.asym_bin, t.csm_bin, "
            "r.chosen FROM read_parquet("
            f"{sql_literal(cache.path('site_tokens'))}) st JOIN read_parquet("
            f"{sql_literal(cache.path('tokens'))}) t USING (token_id) "
            "JOIN records r USING (record_id)"
        )
        counts = "count(*), count(*) FILTER (WHERE chosen)"
        tables = {
            "element_atoms": _distribution(
                connection, f"SELECT element, {counts} FROM sites GROUP BY 1 ORDER BY 1"
            ),
            "geometry_sites": _distribution(
                connection,
                f"SELECT geometry, {counts} FROM sites GROUP BY 1 ORDER BY 2 DESC, 1 "
                "LIMIT 25",
            ),
            "compression_bin_sites": _distribution(
                connection,
                f"SELECT dmin_bin, {counts} FROM sites GROUP BY 1 ORDER BY 1",
            ),
            "asymmetry_bin_sites": _distribution(
                connection,
                f"SELECT asym_bin, {counts} FROM sites GROUP BY 1 ORDER BY 1",
            ),
            "csm_bin_sites": _distribution(
                connection, f"SELECT csm_bin, {counts} FROM sites GROUP BY 1 ORDER BY 1"
            ),
            "elements_per_structure": _distribution(
                connection,
                f"SELECT len(elements), {counts} FROM records GROUP BY 1 ORDER BY 1",
            ),
            "natoms_per_structure": _distribution(
                connection,
                "SELECT CASE WHEN natoms <= 8 THEN '1-8' WHEN natoms <= 32 THEN '9-32' "
                "WHEN natoms <= 128 THEN '33-128' ELSE '>128' END, "
                f"{counts} FROM records GROUP BY 1 ORDER BY min(natoms)",
            ),
            "source_shards": _distribution(
                connection,
                f"SELECT dataset || ':' || shard, {counts} FROM records GROUP BY 1 "
                "ORDER BY 1 LIMIT 50",
            ),
        }
        labels = {}
        for column in ("energy / natoms", "fmax", "stress_max"):
            row = connection.execute(
                f"SELECT count({column}), quantile_cont({column}, [0.05, 0.5, 0.95]), "
                f"count({column}) FILTER (WHERE chosen), "
                f"quantile_cont({column}, [0.05, 0.5, 0.95]) FILTER (WHERE chosen) "
                "FROM records"
            ).fetchall()[0]
            labels[column.replace(" / natoms", "_per_atom")] = {
                "eligible": {"count": int(row[0]), "q05_q50_q95": row[1]},
                "selected": {"count": int(row[2]), "q05_q50_q95": row[3]},
            }
    finally:
        connection.close()
    tables["labels"] = labels  # type: ignore[assignment]
    return tables


def audit(result: SelectionResult) -> dict[str, Any]:
    """Build the complete coverage report of a selection.

    Args:
        result: The selection.

    Returns:
        The ``coverage.json`` content.
    """
    classes = result.classes
    report: dict[str, Any] = {
        "profile": classes.profile.name,
        "min_structure_support": classes.min_support,
        "split": classes.split.value,
        "invalid_record_policy": classes.invalid_record_policy.value,
        "stratum": classes.where,
        "selector": result.settings,
        **audit_classes(classes, result.record_ids),
        "reasons": _reason_counts(result),
        "pruned": len(result.pruned),
        "unreachable_presence": result.unreachable,
        "resolution": classes.rung_statistics(),
        "processing": processing_quality(classes),
        "held_out": held_out_novelty(classes),
    }
    if result.strata:
        report["strata"] = [
            {"stratum": stratum, **audit_classes(part.classes, part.record_ids)}
            for stratum, part in result.strata
        ]
    return report


def _reason_counts(result: SelectionResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in result.selected:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return dict(sorted(counts.items()))


def _complete(section: dict[str, Any]) -> list[str]:
    problems = []
    if section["classes"]["covered"] != section["classes"]["required"]:
        problems.append(
            f"{section['classes']['required'] - section['classes']['covered']} classes"
        )
    problems.extend(
        f"{family['required'] - family['covered']} {name} signatures"
        for name, family in section["presence"].items()
        if family["covered"] != family["required"]
    )
    return problems


def enforce(report: dict[str, Any]) -> None:
    """Raise unless every class and presence signature is covered.

    For a stratified selection every stratum must be complete against its own
    classes; global coverage is informational.

    Args:
        report: Output of :func:`audit`.

    Raises:
        CoverageError: If the contract is not met.
    """
    sections = report["strata"] if report.get("strata") else [report]
    if report["selector"].get("presence_constraints", "reference") != "reference":
        sections = [{**s, "presence": {}} for s in sections]
    problems = [p for s in sections for p in _complete(s)]
    if problems:
        raise CoverageError("selection leaves uncovered: " + "; ".join(problems))


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in columns)
        + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.3g}%"


def render_html(report: dict[str, Any], diagnostics_tables: dict[str, Any]) -> str:
    """Render a self-contained HTML report.

    Args:
        report: Output of :func:`audit`.
        diagnostics_tables: Output of :func:`diagnostics`.

    Returns:
        The HTML document.
    """

    def row(metric: str, value: float | None, detail: str) -> dict[str, str]:
        return {"metric": metric, "value": _percent(value), "detail": detail}

    structures, atoms = report["structures"], report["atoms"]
    classes, tokens = report["classes"], report["full_tokens"]
    summary = [
        row(
            "Structure retention",
            structures["retention"],
            f"{structures['selected']} / {structures['eligible']}",
        ),
        row(
            "Atom retention",
            atoms["retention"],
            f"{atoms['selected']} / {atoms['eligible']}",
        ),
        row(
            "Class coverage",
            classes["fraction"],
            f"{classes['covered']} / {classes['required']}",
        ),
        row(
            "Full-token coverage",
            tokens["fraction"],
            f"{tokens['covered']} / {tokens['eligible']}",
        ),
        row(
            "Frequency-weighted full-token coverage",
            tokens["frequency_weighted"],
            "share of eligible atoms",
        ),
    ] + [
        row(f"Presence {name}", f["fraction"], f"{f['covered']} / {f['required']}")
        for name, f in report["presence"].items()
    ]
    sections = [
        "<h2>Coverage and compression</h2>",
        _table(summary, ["metric", "value", "detail"]),
        "<h2>Selection reasons</h2>",
        _table(
            [{"reason": k, "records": v} for k, v in report["reasons"].items()],
            ["reason", "records"],
        ),
        "<h2>Resolution loss by rung</h2>",
        _table(report["resolution"], ["rung", "name", "tokens", "atoms", "classes"]),
        "<h2>Processing quality</h2>",
        _table(
            report["processing"]["records_by_status"], ["status", "reason", "count"]
        ),
        _table(
            report["processing"]["failures"],
            ["stage", "category", "fallback", "fatal", "count"],
        ),
        _table(report["processing"]["sites"], ["status", "neighbour_source", "count"]),
    ]
    for name, rows in diagnostics_tables.items():
        if name == "labels":
            continue
        sections += [
            f"<h2>{html.escape(name)}</h2>",
            _table(rows, ["value", "eligible", "selected"]),
        ]
    sections += [
        "<h2>Label summaries</h2>",
        f"<pre>{html.escape(json.dumps(diagnostics_tables['labels'], indent=1))}</pre>",
    ]
    title = f"EnvTok selection report (T={report['min_structure_support']})"
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;max-width:70rem;color:#1b1b1b;"
        "background:#fff}table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
        "th,td{border:1px solid #ccc;padding:.25rem .6rem;text-align:left}"
        "th{background:#f2f2f2}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p>Profile {html.escape(report['profile'])}, "
        f"split {html.escape(report['split'])}, invalid-record policy "
        f"{html.escape(report['invalid_record_policy'])}. A diversity cover does not "
        "preserve the sampling distribution and is not a guarantee of equivalent "
        "model accuracy.</p>" + "".join(sections) + "</body></html>\n"
    )
