"""
Stores the project's country-day data, reference data and scoring bounds
in a SQLite database.

SQLite stores the database in a single file and is included with Python,
so no separate database software is needed.

The database contains six tables:
    
    1. country_day: daily totals and measurements for each country
    
    2. event_code_day: daily event counts grouped by event code, relation
       and whether the event targeted the media
    
    3. pull_log: a record of each day's data download, including file
       counts, dropped rows and unmatched actor codes
    
    4. reference: yearly V-Dem and WGI indicator values and uncertainty
       information
    
    5. reference_file: the file used for each reference source, its
       SHA-256 checksum and loading details
    
    6. norm_bounds: fixed normalisation bounds for each criterion and
       settings variant

Important details:
    - Saving a day replaces any existing data for that day, rather
      than creating duplicates.
    - Only country-level totals and grouped event counts are stored.
      Article text and addresses are not saved.
    - Dates are stored as YYYY-MM-DD text.
    - Database tables are checked against the current code. If their
      columns do not match, the loader raises SchemaMismatch.

The functions in this module create and connect to the database, save
and load daily data, record reference files, and save and load
normalisation bounds. Data changes are saved in transactions so a
failed write does not leave only part of an update in the database.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.reference.common import REFERENCE_COLUMNS
from src.transform.aggregate import (
    EVENT_CODE_COLUMNS,
    QUAD_CLASSES,
    ROOT_CODES,
    criterion_columns,
)

DB_PATH = Path("data/processed/country_day.sqlite")
FILES_PER_DAY = 96 # one file every 15 minutes (24 x 4 = 96)


# SQLite column names and types for the country-day table
MEASURE_COLUMNS: dict[str, str] = {
    "n_events": "INTEGER", # INTEGER = whole number
    "n_mentions": "INTEGER",
    "n_articles": "INTEGER",
    "tone_mean": "REAL", # REAL = decimal number
    "tone_mean_weighted": "REAL",
    "goldstein_mean": "REAL",
    "root_other": "INTEGER",
}

# Add the four event quadrants
for q in QUAD_CLASSES:
    MEASURE_COLUMNS[f"quad_{q}"] = "INTEGER"

# Add the event root-code columns
for code in ROOT_CODES:
    MEASURE_COLUMNS[f"root_{code}"] = "INTEGER"

# Add the inputs used by the scoring criteria
for name in criterion_columns():
    if name.endswith("_goldstein"):
        MEASURE_COLUMNS[name] = "REAL"
    else:
        MEASURE_COLUMNS[name] = "INTEGER"


TABLE_COLUMNS: list[str] = ["iso3", "date", *MEASURE_COLUMNS]


LOG_COLUMNS: list[str] = [
    "date", "pulled_at", "files_listed", "files_read", "complete",
    "rows_in", "duplicate_event", "no_country", "bad_date", "rows_used",
    "actor_unmatched",
]


REFERENCE_FILE_COLUMNS: list[str] = [
    "source", "file_name", "sha256", "loaded_at", "rows",
]


BOUNDS_COLUMNS: list[str] = [
    "criterion", "variant", "lower", "upper", "mean", "sd", "method",
    "reference_start", "reference_end", "computed_at",
]


class SchemaMismatch(RuntimeError):
    """
    The database's columns no longer match.
    """


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """
    Open the database, creating the file and all tables if they do not exist.
    """
    # Used by tests so they never touch the actual DB (they just one in-memory)
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path))

    measures = ",\n    ".join(
        f"{name} {kind}" for name, kind in MEASURE_COLUMNS.items()
    )
    
    # Create country_day
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS country_day (
            iso3 TEXT NOT NULL,
            date TEXT NOT NULL,
            {measures},
            PRIMARY KEY (iso3, date)
        )
    """)

    # Create event_code_day
    connection.execute("""
        CREATE TABLE IF NOT EXISTS event_code_day (
            iso3 TEXT NOT NULL,
            date TEXT NOT NULL,
            event_code TEXT NOT NULL,
            relation TEXT NOT NULL,
            media_target INTEGER NOT NULL,
            n_events INTEGER NOT NULL,
            n_articles INTEGER NOT NULL,
            goldstein_sum REAL NOT NULL,
            PRIMARY KEY (iso3, date, event_code, relation, media_target)
        )
    """)

    # Create pull_log
    connection.execute("""
        CREATE TABLE IF NOT EXISTS pull_log (
            date TEXT PRIMARY KEY,
            pulled_at TEXT NOT NULL,
            files_listed INTEGER NOT NULL,
            files_read INTEGER NOT NULL,
            complete INTEGER NOT NULL,
            rows_in INTEGER NOT NULL,
            duplicate_event INTEGER NOT NULL,
            no_country INTEGER NOT NULL,
            bad_date INTEGER NOT NULL,
            rows_used INTEGER NOT NULL,
            actor_unmatched INTEGER NOT NULL
        )
    """)

    # Create reference
    connection.execute("""
        CREATE TABLE IF NOT EXISTS reference (
            iso3 TEXT NOT NULL,
            year INTEGER NOT NULL,
            variable TEXT NOT NULL,
            source TEXT NOT NULL,
            value REAL NOT NULL,
            sd REAL,
            low REAL,
            high REAL,
            interval REAL,
            PRIMARY KEY (iso3, year, variable)
        )
    """)

    # Create reference_file
    connection.execute("""
        CREATE TABLE IF NOT EXISTS reference_file (
            source TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            loaded_at TEXT NOT NULL,
            rows INTEGER NOT NULL
        )
    """)

    # Create norm_bounds
    connection.execute("""
        CREATE TABLE IF NOT EXISTS norm_bounds (
            criterion TEXT NOT NULL,
            variant TEXT NOT NULL,
            lower REAL NOT NULL,
            upper REAL NOT NULL,
            mean REAL NOT NULL,
            sd REAL NOT NULL,
            method TEXT NOT NULL,
            reference_start TEXT NOT NULL,
            reference_end TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (criterion, variant)
        )
    """)

    connection.commit()

    # Check for outdated columns
    expected = {
        "country_day": TABLE_COLUMNS,
        "event_code_day": EVENT_CODE_COLUMNS,
        "pull_log": LOG_COLUMNS,
        "reference": REFERENCE_COLUMNS,
        "reference_file": REFERENCE_FILE_COLUMNS,
        "norm_bounds": BOUNDS_COLUMNS,
    }
    for table_name, columns in expected.items():
        found = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})")
        ]
        if found != columns:
            connection.close()
            raise SchemaMismatch(
                f"{path}: table {table_name} has different columns from "
                "the current code. Delete the database file and run "
                "src.pull_day again to rebuild it from the cached downloads."
            )
    return connection


def _plain_rows(table: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """
    SQLite needs plain Python values. astype(object) converts pandas numbers
    into Python numbers, and missing values become None.
    """
    rows = table.assign(date=dates).astype(object)
    return rows.where(table.notna(), None)


def _check_one_day(table: pd.DataFrame, day_text: str, name: str) -> None:
    """
    Refuse a table holding rows for any day other than day_text.
    """
    dates = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
    other_days = sorted(set(dates) - {day_text})

    if other_days:
        raise ValueError(
            f"{name} holds rows for other days than {day_text}: {other_days}"
        )


def save_day(
        connection: sqlite3.Connection,
        table: pd.DataFrame,
        day: date,
        files_listed: int,
        files_read: int,
        codes: pd.DataFrame | None = None,
    ) -> int:
    """
    Store one day's country-day table, its event code table and its pull
    record.

    If the day exists, its old rows are removed first. If codes is None, 
    the day's event code rows are removed and none are written.

    Returns the number of country rows written.
    """
    if list(table.columns) != TABLE_COLUMNS:
        raise ValueError(
            "table columns do not match the database. "
            f"Missing: {sorted(set(TABLE_COLUMNS) - set(table.columns))}, "
            f"unexpected: {sorted(set(table.columns) - set(TABLE_COLUMNS))}"
        )
    
    if codes is not None and list(codes.columns) != EVENT_CODE_COLUMNS:
        raise ValueError(
            f"event code table columns should be {EVENT_CODE_COLUMNS}, "
            f"got {list(codes.columns)}"
        )

    dropped = table.attrs.get("dropped")
    if dropped is None:
        raise ValueError(
            "table has no record of dropped rows (table.attrs['dropped'])"
        )
    
    unmatched = table.attrs.get("actor_codes_unmatched", {})

    # Check that every row belongs to the day being saved
    day_text = day.isoformat()
    _check_one_day(table, day_text, "table")
    rows = _plain_rows(table, pd.Series(day_text, index=table.index))

    code_rows = None
    if codes is not None:
        _check_one_day(codes, day_text, "event code table")
        code_rows = _plain_rows(codes, pd.Series(day_text, index=codes.index))

    log_row = (
        day_text,
        datetime.now(UTC).isoformat(timespec="seconds"),
        files_listed,
        files_read,
        int(files_read == FILES_PER_DAY),
        dropped["rows_in"],
        dropped["duplicate_event"],
        dropped["no_country"],
        dropped["bad_date"],
        dropped["rows_used"],
        sum(unmatched.values()),
    )

    def insert(table_name: str, columns: list[str]) -> str:
        return (
            f"INSERT INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )

    # "with connection" ensures atomicity
    with connection:

        for table_name in ("country_day", "event_code_day"):
            connection.execute(
                f"DELETE FROM {table_name} WHERE date = ?", (day_text,)
            )

        connection.executemany(
            insert("country_day", TABLE_COLUMNS),
            rows.itertuples(index=False, name=None),
        )

        if code_rows is not None:
            connection.executemany(
                insert("event_code_day", EVENT_CODE_COLUMNS),
                code_rows.itertuples(index=False, name=None),
            )

        connection.execute(
            insert("pull_log", LOG_COLUMNS).replace(
                "INSERT", "INSERT OR REPLACE", 1
            ),
            log_row,
        )

    return len(rows)


def _where(
        start: date | None,
        end: date | None,
        iso3: str | None,
    ) -> tuple[str, list[str]]:
    """
    Build the WHERE part of a query from the optional filters.

    To prevent SQLi, the values are passed separately and are never pasted into
    the query text.
    """
    conditions: list[str] = []
    values: list[str] = []

    if start is not None:
        conditions.append("date >= ?")
        values.append(start.isoformat())

    if end is not None:
        conditions.append("date <= ?")
        values.append(end.isoformat())

    if iso3 is not None:
        conditions.append("iso3 = ?")
        values.append(iso3)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    return where, values


def load_country_day(
        connection: sqlite3.Connection,
        start: date | None = None,
        end: date | None = None,
        iso3: str | None = None,
    ) -> pd.DataFrame:
    """
    Read stored country-day rows back, sorted by date then country.
    """
    where, values = _where(start, end, iso3)
    query = (
        f"SELECT {', '.join(TABLE_COLUMNS)} FROM country_day {where} "
        "ORDER BY date, iso3"
    )

    table = pd.read_sql_query(query, connection, params=values)
    table["date"] = pd.to_datetime(table["date"], format="%Y-%m-%d")

    return table


def load_event_codes(
        connection: sqlite3.Connection,
        start: date | None = None,
        end: date | None = None,
        iso3: str | None = None,
    ) -> pd.DataFrame:
    """
    Read stored event code rows back, sorted by date, country, event code
    and relation. The filters work as in load_country_day().
    """
    where, values = _where(start, end, iso3)
    query = (
        f"SELECT {', '.join(EVENT_CODE_COLUMNS)} FROM event_code_day "
        f"{where} ORDER BY date, iso3, event_code, relation, media_target"
    )

    table = pd.read_sql_query(query, connection, params=values)
    table["date"] = pd.to_datetime(table["date"], format="%Y-%m-%d")

    return table


def load_pull_log(connection: sqlite3.Connection) -> pd.DataFrame:
    """
    Read the record of every day pulled, oldest first.
    """
    query = f"SELECT {', '.join(LOG_COLUMNS)} FROM pull_log ORDER BY date"

    return pd.read_sql_query(query, connection)


def save_reference(
        connection: sqlite3.Connection,
        table: pd.DataFrame,
        source: str,
        file_name: str,
        sha256: str,
    ) -> int:
    """
    Replace every stored row of one source ("vdem" or "wgi") with table,
    and record which file it came from.

    Returns the number of rows written.
    """
    if list(table.columns) != REFERENCE_COLUMNS:
        raise ValueError(
            f"reference table columns should be {REFERENCE_COLUMNS}, "
            f"got {list(table.columns)}"
        )
    
    other = sorted(set(table["source"]) - {source})

    if other:
        raise ValueError(f"table holds rows for other sources: {other}")

    rows = table.astype(object).where(table.notna(), None)
    file_row = (
        source, file_name, sha256,
        datetime.now(UTC).isoformat(timespec="seconds"), len(rows),
    )

    def insert(table_name: str, columns: list[str], verb: str) -> str:
        return (
            f"{verb} INTO {table_name} ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' for _ in columns)})"
        )

    with connection:
        connection.execute("DELETE FROM reference WHERE source = ?", (source,))

        connection.executemany(
            insert("reference", REFERENCE_COLUMNS, "INSERT"),
            rows.itertuples(index=False, name=None),
        )

        connection.execute(
            insert("reference_file", REFERENCE_FILE_COLUMNS,
                   "INSERT OR REPLACE"),
            file_row,
        )

    return len(rows)


def load_reference(
        connection: sqlite3.Connection,
        variables: list[str] | None = None,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> pd.DataFrame:
    """
    Read stored reference rows back, optionally for some variables and
    years only (both years included).
    """
    conditions: list[str] = []
    values: list[object] = []

    if variables is not None:
        conditions.append(
            f"variable IN ({', '.join('?' for _ in variables)})"
        )
        values.extend(variables)

    if start_year is not None:
        conditions.append("year >= ?")
        values.append(start_year)

    if end_year is not None:
        conditions.append("year <= ?")
        values.append(end_year)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = (
        f"SELECT {', '.join(REFERENCE_COLUMNS)} FROM reference {where} "
        "ORDER BY variable, year, iso3"
    )

    return pd.read_sql_query(query, connection, params=values)


def load_reference_files(connection: sqlite3.Connection) -> pd.DataFrame:
    """
    Which file each reference source was loaded from.
    """
    query = (
        f"SELECT {', '.join(REFERENCE_FILE_COLUMNS)} FROM reference_file "
        "ORDER BY source"
    )

    return pd.read_sql_query(query, connection)


def save_bounds(connection: sqlite3.Connection, table: pd.DataFrame) -> int:
    """
    Store normalisation bounds. A row replaces any stored row for the same
    criterion and variant; other rows are kept.
    """
    if list(table.columns) != BOUNDS_COLUMNS:
        raise ValueError(
            f"bounds table columns should be {BOUNDS_COLUMNS}, "
            f"got {list(table.columns)}"
        )
    
    bad = table[table["upper"] <= table["lower"]]
    if not bad.empty:
        raise ValueError(
            "upper bound not above lower bound:\n"
            f"{bad[['criterion', 'lower', 'upper']].to_string(index=False)}"
        )
    
    rows = table.astype(object).where(table.notna(), None)

    query = (
        f"INSERT OR REPLACE INTO norm_bounds ({', '.join(BOUNDS_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in BOUNDS_COLUMNS)})"
    )

    with connection:
        connection.executemany(query, rows.itertuples(index=False, name=None))

    return len(rows)


def load_bounds(
        connection: sqlite3.Connection,
        variants: list[str] | None = None,
    ) -> pd.DataFrame:
    """
    Stored bounds, optionally for some variants only.
    """
    where, values = "", []
    if variants is not None:
        where = f"WHERE variant IN ({', '.join('?' for _ in variants)})"
        values = list(variants)

    query = (
        f"SELECT {', '.join(BOUNDS_COLUMNS)} FROM norm_bounds {where} "
        "ORDER BY criterion, variant"
    )

    return pd.read_sql_query(query, connection, params=values)
