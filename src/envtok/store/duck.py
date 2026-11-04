"""DuckDB connections with an explicit resource policy.

DuckDB processes larger-than-memory joins and aggregations by spilling to a
temporary directory. Its memory limit does not cap allocations made by Python
or NumPy in the same process, so the limit should leave room for them.
"""

from pathlib import Path

import duckdb


def sql_literal(text: str | Path) -> str:
    """Quote a string (for example a file path) as a SQL literal."""
    return "'" + str(text).replace("'", "''") + "'"


def connect(
    *,
    memory_limit: str | None = None,
    threads: int | None = None,
    temp_directory: Path | None = None,
    max_temp_directory_size: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the given resource settings.

    Args:
        memory_limit: DuckDB memory limit, for example ``"8GB"``.
        threads: Worker threads.
        temp_directory: Spill directory.
        max_temp_directory_size: Spill budget, for example ``"200GB"``.

    Returns:
        The connection.
    """
    connection = duckdb.connect(":memory:")
    if memory_limit is not None:
        connection.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
    if threads is not None:
        connection.execute(f"SET threads = {int(threads)}")
    if temp_directory is not None:
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {sql_literal(temp_directory)}")
    if max_temp_directory_size is not None:
        connection.execute(
            f"SET max_temp_directory_size = {sql_literal(max_temp_directory_size)}"
        )
    return connection
