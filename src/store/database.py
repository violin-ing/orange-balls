"""
Stores the country-day tables in a SQLite database.

SQLite keeps a whole database in one file. It is built into Python, so
nothing extra needs installing.

Three tables:
    country_day     one row per country per day (from to_country_day)
    event_code_day  one row per country, day, event code and relation
                    (from to_event_code_day)
    pull_log        one row per day pulled: when, how many files, what was
                    dropped, and how many actor codes were not countries

Things to note:
    - Saving a day that is already stored replaces it. Nothing is duplicated.
    - Only country-level totals are stored. No article addresses or text.
    - Dates are stored as text in the form YYYY-MM-DD.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.transform.aggregate import (
    EVENT_CODE_COLUMNS,
    QUAD_CLASSES,
    ROOT_CODES,
    criterion_columns,
)

DB_PATH = Path("data/processed/country_day.sqlite")
FILES_PER_DAY = 96  # one file every 15 minutes

# Column name -> SQLite type, in the order to_country_day() produces them
# INTEGER is a whole number, REAL is a decimal number
MEASURE_COLUMNS: dict[str, str] = {
    "n_events": "INTEGER",
    "n_mentions": "INTEGER",
    "n_articles": "INTEGER",
    "tone_mean": "REAL",
    "tone_mean_weighted": "REAL",
    "goldstein_mean": "REAL",
    **{f"quad_{q}": "INTEGER" for q in QUAD_CLASSES},
    **{f"root_{code}": "INTEGER" for code in ROOT_CODES},
    "root_other": "INTEGER",
    # Criteria inputs: 
    # - Events and articles are whole numbers
    # - Goldstein sums are decimals
    **{
        name: "REAL" if name.endswith("_goldstein") else "INTEGER"
        for name in criterion_columns()
    },
}

TABLE_COLUMNS: list[str] = ["iso3", "date", *MEASURE_COLUMNS]

LOG_COLUMNS: list[str] = [
    "date", "pulled_at", "files_listed", "files_read", "complete",
    "rows_in", "duplicate_event", "no_country", "bad_date", "rows_used",
    "actor_unmatched",
]


class SchemaMismatch(RuntimeError):
    """
    The database file was made by an older version of the code and its
    columns no longer match.
    """


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    """
    Open the database, creating the file and all tables if they do not exist.

    Pass ":memory:" to get a temporary database that lives only in memory.
    The tests use this so they never touch the real file.
    """
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path))

    measures = ",\n    ".join(
        f"{name} {kind}" for name, kind in MEASURE_COLUMNS.items()
    )
    # PRIMARY KEY (iso3, date) means the database itself refuses a second
    # row for the same country and day
    connection.execute(f"""
        CREATE TABLE IF NOT EXISTS country_day (
            iso3 TEXT NOT NULL,
            date TEXT NOT NULL,
            {measures},
            PRIMARY KEY (iso3, date)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS event_code_day (
            iso3 TEXT NOT NULL,
            date TEXT NOT NULL,
            event_code TEXT NOT NULL,
            relation TEXT NOT NULL,
            n_events INTEGER NOT NULL,
            n_articles INTEGER NOT NULL,
            goldstein_sum REAL NOT NULL,
            PRIMARY KEY (iso3, date, event_code, relation)
        )
    """)
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
    connection.commit()

    # CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a file
    # made by older code keeps its old columns
    # Note: check for that here, where the message can say what to do, instead 
    # of failing later on insert
    expected = {
        "country_day": TABLE_COLUMNS,
        "event_code_day": EVENT_CODE_COLUMNS,
        "pull_log": LOG_COLUMNS,
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
                "the current code. Nothing is lost: delete the database "
                "file and run src.pull_day again to rebuild it from the "
                "cached downloads."
            )
    return connection


def _plain_rows(table: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """
    SQLite needs plain Python values. astype(object) turns pandas numbers
    into Python numbers, and missing values become None (NULL in SQL).
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
            f"{name} holds rows for other days than {day_text}: "
            f"{other_days}"
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

    If the day is already stored, its old rows are removed first, so running
    the same day twice gives the same result as running it once. If codes
    is None, the day's event code rows are removed and none are written.

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

    # Every row must belong to the day being saved
    # A row from another day would be written under the wrong pull record
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

    # "with connection" makes every statement one transaction => either all
    # of them happen or none do, so a crash cannot leave half a day
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

    The values are passed separately (the ? marks), never pasted into the
    query text. This is what prevents SQL injection.
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

    All three filters are optional. start and end are both included.
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
        f"{where} ORDER BY date, iso3, event_code, relation"
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
