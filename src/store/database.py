"""
Stores the project's country-day data, reference data and scoring bounds in a
SQLite database.

SQLite stores the database in a single file and is included with Python, so no
separate database software is needed.

The database contains eight tables:
    
    1. country_day: daily totals and measurements for each country
    
    2. event_code_day: daily event counts grouped by event code, relation and
        whether the event targeted the media
    
    3. pull_log: a record of each day's data download, including file counts, 
        dropped rows and unmatched actor codes
    
    4. reference: yearly V-Dem and WGI indicator values and uncertainty info
    
    5. reference_file: the file used for each reference source, its SHA-256 
        checksum and loading details
    
    6. norm_bounds: fixed normalisation bounds for each criterion and settings 
        variant

    7. band_cuts: the frozen cut points between bands, one row per score
        settings variant and percentile (decision log 14.1)

    8. gdelt_file: the manifest of GDELT files behind each stored day: the
        file's address, GDELT's published MD5 checksum, its size and whether
        it was read. This is what makes the download cache replaceable: any
        file can be fetched again from GDELT and checked against its MD5, so
        the cache can be deleted at the end of the project (decision log
        18.1, 26.1). It lives in this database, not in the cache folder.

Important details:
    - Saving a day replaces any existing data for that day, rather than
        creating duplicates.
    - Only country-level totals and grouped event counts are stored. Article
        text and addresses are not saved.
    - Dates are stored as YYYY-MM-DD text.
    - Database tables are checked against the current code. If their columns do
        not match, the loader raises SchemaMismatch.

The functions in this module create and connect to the database, save and load
daily data, record reference files, and save and load normalisation bounds.

Data changes are saved in transactions so a failed write does not leave only
part of an update in the database (atomicity).
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

BAND_CUT_COLUMNS: list[str] = [
    "variant", "percentile", "value", "reference_start", "reference_end",
    "computed_at",
]

# One row per GDELT file: the file list and checksums behind each day
MANIFEST_COLUMNS: list[str] = [
    "date", "url", "md5", "size_bytes", "read_ok",
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
            is_root INTEGER NOT NULL,
            n_events INTEGER NOT NULL,
            n_articles INTEGER NOT NULL,
            goldstein_sum REAL NOT NULL,
            PRIMARY KEY (iso3, date, event_code, relation, media_target,
                         is_root)
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

    # Create band_cuts
    connection.execute("""
        CREATE TABLE IF NOT EXISTS band_cuts (
            variant TEXT NOT NULL,
            percentile REAL NOT NULL,
            value REAL NOT NULL,
            reference_start TEXT NOT NULL,
            reference_end TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (variant, percentile)
        )
    """)

    # Create gdelt_file (the file manifest)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS gdelt_file (
            date TEXT NOT NULL,
            url TEXT PRIMARY KEY,
            md5 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            read_ok INTEGER NOT NULL
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
        "band_cuts": BAND_CUT_COLUMNS,
        "gdelt_file": MANIFEST_COLUMNS,
    }
    for table_name, columns in expected.items():
        found = [
            row[1]
            for row in connection.execute(f"PRAGMA table_info({table_name})")
        ]
        if sorted(found) != sorted(columns):
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
        manifest: pd.DataFrame | None = None,
    ) -> int:
    """
    Store one day's country-day table, its event code table, its pull
    record and, if given, its file manifest (MANIFEST_COLUMNS).

    If the day exists, its old rows are removed first. If codes is None, 
    the day's event code rows are removed and none are written. If manifest
    is None, the day's stored manifest is left as it is.

    Returns the number of country rows written.
    """
    if (sorted(table.columns) != sorted(TABLE_COLUMNS)
        or len(table.columns) != len(TABLE_COLUMNS)):
        raise ValueError(
            "Table columns do not match the database. "
            f"Missing: {sorted(set(TABLE_COLUMNS) - set(table.columns))}, "
            f"unexpected: {sorted(set(table.columns) - set(TABLE_COLUMNS))}"
        )
    
    attrs = table.attrs
    table = table[TABLE_COLUMNS]
    table.attrs = attrs
    
    if codes is not None and list(codes.columns) != EVENT_CODE_COLUMNS:
        raise ValueError(
            f"Event code table columns should be {EVENT_CODE_COLUMNS}, "
            f"got {list(codes.columns)}"
        )

    dropped = table.attrs.get("dropped")
    if dropped is None:
        raise ValueError(
            "Table has no record of dropped rows (table.attrs['dropped'])"
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

    if manifest is not None:
        _check_manifest(manifest, day_text)

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

        if manifest is not None:
            _write_manifest(connection, manifest, day_text)

    return len(rows)


def _check_manifest(manifest: pd.DataFrame, day_text: str) -> None:
    """
    Refuse a manifest with the wrong columns or rows for another day.
    """
    if list(manifest.columns) != MANIFEST_COLUMNS:
        raise ValueError(
            f"Manifest columns should be {MANIFEST_COLUMNS}, "
            f"got {list(manifest.columns)}"
        )

    other = set(manifest["date"].astype(str)) - {day_text}
    if other:
        raise ValueError(
            f"Manifest for {day_text} holds rows for other dates: "
            f"{sorted(other)}"
        )


def _write_manifest(
        connection: sqlite3.Connection,
        manifest: pd.DataFrame,
        day_text: str,
    ) -> None:
    """
    Replace one day's manifest rows. Call inside a transaction.
    """
    connection.execute("DELETE FROM gdelt_file WHERE date = ?", (day_text,))

    rows = manifest.astype(object).where(manifest.notna(), None)
    rows["read_ok"] = [int(bool(x)) for x in manifest["read_ok"]]

    connection.executemany(
        f"INSERT OR REPLACE INTO gdelt_file ({', '.join(MANIFEST_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in MANIFEST_COLUMNS)})",
        rows.itertuples(index=False, name=None),
    )


def save_manifest(
        connection: sqlite3.Connection,
        day: date,
        manifest: pd.DataFrame,
    ) -> int:
    """
    Store one day's file manifest on its own, replacing any stored rows for
    that day. Used to record the manifest of days pulled before the manifest
    existed (python -m src.pull_day --manifest-only).
    """
    day_text = day.isoformat()
    _check_manifest(manifest, day_text)

    with connection:
        _write_manifest(connection, manifest, day_text)

    return len(manifest)


def load_manifest(
        connection: sqlite3.Connection,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
    """
    The stored file manifest, optionally for a date range only (both ends
    included).
    """
    conditions, values = [], []
    if start is not None:
        conditions.append("date >= ?")
        values.append(start.isoformat())
    if end is not None:
        conditions.append("date <= ?")
        values.append(end.isoformat())

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = (
        f"SELECT {', '.join(MANIFEST_COLUMNS)} FROM gdelt_file {where} "
        "ORDER BY date, url"
    )

    return pd.read_sql_query(query, connection, params=values)


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
        f"{where} ORDER BY date, iso3, event_code, relation, media_target, "
        "is_root"
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
    Replace every stored row of one source ("vdem" or "wgi") with table, and
    record which file it came from.

    Returns the number of rows written.
    """
    if list(table.columns) != REFERENCE_COLUMNS:
        raise ValueError(
            f"Reference table columns should be {REFERENCE_COLUMNS}, "
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
    Read stored reference rows back, optionally for some variables and years
    only (both years included).
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
            f"Bounds table columns should be {BOUNDS_COLUMNS}, "
            f"got {list(table.columns)}"
        )
    
    bad = table[table["upper"] <= table["lower"]]
    if not bad.empty:
        raise ValueError(
            "Upper bound not above lower bound:\n"
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


def save_band_cuts(connection: sqlite3.Connection, table: pd.DataFrame) -> int:
    """
    Store band cut points. A row replaces any stored row for the same
    variant and percentile; other rows are kept.
    """
    if list(table.columns) != BAND_CUT_COLUMNS:
        raise ValueError(
            f"Band cut table columns should be {BAND_CUT_COLUMNS}, "
            f"got {list(table.columns)}"
        )

    rows = table.astype(object).where(table.notna(), None)

    query = (
        f"INSERT OR REPLACE INTO band_cuts ({', '.join(BAND_CUT_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in BAND_CUT_COLUMNS)})"
    )

    with connection:
        connection.executemany(query, rows.itertuples(index=False, name=None))

    return len(rows)


def load_band_cuts(
        connection: sqlite3.Connection,
        variants: list[str] | None = None,
    ) -> pd.DataFrame:
    """
    Stored band cut points, optionally for some variants only.
    """
    where, values = "", []
    if variants is not None:
        where = f"WHERE variant IN ({', '.join('?' for _ in variants)})"
        values = list(variants)

    query = (
        f"SELECT {', '.join(BAND_CUT_COLUMNS)} FROM band_cuts {where} "
        "ORDER BY variant, percentile"
    )

    return pd.read_sql_query(query, connection, params=values)
