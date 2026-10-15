"""
Tests for src/store/database.py -- run from the repository root.

Unit tests (use a temporary in-memory database, not the real file):

    python3 -m tests.store_test

Show what is in the real database:

    python3 -m tests.store_test --real
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date

import pandas as pd

from src.store.database import (
    DB_PATH,
    TABLE_COLUMNS,
    SchemaMismatch,
    connect,
    load_country_day,
    load_event_codes,
    load_pull_log,
    save_day,
)
from src.transform.aggregate import to_country_day, to_event_code_day
from tests.aggregate_test import make_events

DAY_1 = date(2026, 10, 5)
DAY_2 = date(2026, 10, 6)


def make_day(day: date) -> pd.DataFrame:
    """
    The country-day rows for one day, built from the hand-written events.
    """
    full = to_country_day(make_events())
    table = full[full["date"] == pd.Timestamp(day)].reset_index(drop=True)
    table.attrs["dropped"] = full.attrs["dropped"]
    table.attrs["actor_codes_unmatched"] = full.attrs["actor_codes_unmatched"]
    return table


def make_codes(day: date) -> pd.DataFrame:
    """
    The event code rows for one day, from the hand-written events.
    """
    full = to_event_code_day(make_events())
    return full[full["date"] == pd.Timestamp(day)].reset_index(drop=True)


def test_tables_are_created() -> None:
    connection = connect(":memory:")
    names = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )}
    assert names == {"country_day", "event_code_day", "pull_log"}, names
    connection.close()


def test_save_then_load_gives_the_same_table() -> None:
    connection = connect(":memory:")
    original = make_day(DAY_1)
    written = save_day(connection, original, DAY_1, 96, 96)
    loaded = load_country_day(connection)

    # DEU, FRA, GBR, NGA and USA
    assert written == 5 and len(loaded) == 5
    assert list(loaded.columns) == TABLE_COLUMNS
    pd.testing.assert_frame_equal(
        loaded, original, check_dtype=False, check_names=False
    )
    connection.close()


def test_saving_the_same_day_twice_does_not_duplicate() -> None:
    connection = connect(":memory:")
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)
    assert len(load_country_day(connection)) == 5
    assert len(load_pull_log(connection)) == 1
    connection.close()


def test_saving_again_replaces_old_rows() -> None:
    """
    A country missing from the second run must not be left behind.
    """
    connection = connect(":memory:")
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)

    only_usa = make_day(DAY_1)
    dropped = only_usa.attrs["dropped"]
    only_usa = only_usa[only_usa["iso3"] == "USA"].reset_index(drop=True)
    only_usa.attrs["dropped"] = dropped
    save_day(connection, only_usa, DAY_1, 96, 96)

    assert load_country_day(connection)["iso3"].tolist() == ["USA"]
    connection.close()


def test_a_second_day_is_added_alongside_the_first() -> None:
    connection = connect(":memory:")
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)
    save_day(connection, make_day(DAY_2), DAY_2, 96, 96)

    assert len(load_country_day(connection)) == 6
    assert len(load_country_day(connection, start=DAY_2)) == 1
    assert len(load_country_day(connection, end=DAY_1)) == 5
    assert len(load_country_day(connection, iso3="NGA")) == 2
    one = load_country_day(connection, start=DAY_1, end=DAY_1, iso3="NGA")
    assert len(one) == 1
    connection.close()


def test_pull_log_records_the_run() -> None:
    connection = connect(":memory:")
    save_day(
        connection, make_day(DAY_1), DAY_1, files_listed=96, files_read=95
    )
    log = load_pull_log(connection).iloc[0]

    assert log["date"] == "2026-10-05"
    assert log["files_listed"] == 96 and log["files_read"] == 95
    assert log["complete"] == 0 # 95 of 96 is not a full day
    assert log["rows_in"] == 9 and log["no_country"] == 2
    assert log["rows_used"] == 6
    assert log["actor_unmatched"] == 1 # the EUR region code
    assert log["pulled_at"].startswith("20") # a UTC timestamp
    connection.close()


def test_missing_values_survive_the_round_trip() -> None:
    connection = connect(":memory:")
    # DEU, FRA and GBR appear only as actors, so they have no tone
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)
    loaded = load_country_day(connection)
    assert loaded["iso3"].tolist() == ["DEU", "FRA", "GBR", "NGA", "USA"]
    assert loaded["tone_mean"].isna().tolist() == [
        True, True, True, False, False,
    ]
    connection.close()


def test_rows_from_another_day_are_refused() -> None:
    connection = connect(":memory:")
    try:
        save_day(connection, make_day(DAY_2), DAY_1, 96, 96)
    except ValueError as error:
        assert "2026-10-06" in str(error)
    else:
        raise AssertionError("rows for the wrong day should raise")
    assert len(load_country_day(connection)) == 0
    connection.close()


def test_wrong_columns_are_refused() -> None:
    connection = connect(":memory:")
    table = make_day(DAY_1)
    dropped = table.attrs["dropped"]
    table = table.drop(columns="unrest_events")
    table.attrs["dropped"] = dropped
    try:
        save_day(connection, table, DAY_1, 96, 96)
    except ValueError as error:
        assert "unrest_events" in str(error)
    else:
        raise AssertionError("a missing column should raise")
    connection.close()


def test_database_refuses_duplicate_country_day() -> None:
    """
    The primary key is a second line of defence under save_day.
    """
    connection = connect(":memory:")
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)
    try:
        connection.execute(
            "INSERT INTO country_day (iso3, date) VALUES ('USA', '2026-10-05')"
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("the primary key should refuse a duplicate")
    connection.close()


def test_event_codes_are_saved_and_replaced() -> None:
    connection = connect(":memory:")
    original = make_codes(DAY_1)
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96, codes=original)
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96, codes=original)
    loaded = load_event_codes(connection)

    assert len(loaded) == 5                  # not 10: replaced, not added
    pd.testing.assert_frame_equal(
        loaded, original, check_dtype=False, check_names=False
    )
    assert len(load_event_codes(connection, iso3="NGA")) == 1

    # Saving the day again without codes removes the old code rows
    save_day(connection, make_day(DAY_1), DAY_1, 96, 96)
    assert len(load_event_codes(connection)) == 0
    connection.close()


def test_event_codes_from_another_day_are_refused() -> None:
    connection = connect(":memory:")
    try:
        save_day(
            connection, make_day(DAY_1), DAY_1, 96, 96,
            codes=make_codes(DAY_2),
        )
    except ValueError as error:
        assert "2026-10-06" in str(error)
    else:
        raise AssertionError("code rows for the wrong day should raise")
    assert len(load_country_day(connection)) == 0
    connection.close()


def test_database_from_older_code_is_refused() -> None:
    """
    An old file must give a clear message, not a confusing SQL error.
    """
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE country_day (iso3 TEXT, date TEXT, n_events INTEGER)"
    )
    old.commit()
    old.close()
    try:
        connect(path)
    except SchemaMismatch as error:
        assert "delete the database file" in str(error)
    else:
        raise AssertionError("an old database should be refused")


def check_real_database() -> None:
    """
    Print what is stored in the real database file.
    """
    if not DB_PATH.exists():
        raise SystemExit(
            f"{DB_PATH} does not exist yet, run src.pull_day first"
        )

    connection = connect(DB_PATH)
    log = load_pull_log(connection)
    table = load_country_day(connection)
    codes = load_event_codes(connection)
    connection.close()

    print(f"database: {DB_PATH} ({DB_PATH.stat().st_size / 1024:.0f} KB)")
    print(f"days stored: {len(log)}, country-day rows: {len(table)}")

    print("\n--- pull log ---")
    print(log.to_string(index=False))

    per_day = table.groupby("date").agg(
        countries=("iso3", "count"), events=("n_events", "sum")
    )
    print("\n--- stored rows per day ---")
    print(per_day.to_string())

    # The events stored for a day must equal the rows_used in its pull record
    stored = per_day["events"].to_numpy()
    logged = log["rows_used"].to_numpy()
    assert (stored == logged).all(), "stored events do not match the pull log"

    duplicates = table.duplicated(subset=["iso3", "date"]).sum()
    assert duplicates == 0, f"{duplicates} duplicate country-day rows"

    # The event code rows must add up to the same located events
    code_totals = codes.groupby("date")["n_events"].sum()
    print(f"\nevent code rows: {len(codes):,}")
    assert (code_totals.to_numpy() == logged).all(), (
        "event code rows do not match the pull log"
    )

    print("\nREAL DATABASE CHECK OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="show the contents of the real database instead")
    args = parser.parse_args()

    if args.real:
        check_real_database()
        return

    tests = [
        test_tables_are_created,
        test_save_then_load_gives_the_same_table,
        test_saving_the_same_day_twice_does_not_duplicate,
        test_saving_again_replaces_old_rows,
        test_a_second_day_is_added_alongside_the_first,
        test_pull_log_records_the_run,
        test_missing_values_survive_the_round_trip,
        test_rows_from_another_day_are_refused,
        test_wrong_columns_are_refused,
        test_database_refuses_duplicate_country_day,
        test_event_codes_are_saved_and_replaced,
        test_event_codes_from_another_day_are_refused,
        test_database_from_older_code_is_refused,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
