"""
Tests for src/transform/aggregate.py -- run from the repository root.

Unit tests (no network, no data files needed):

    python3 -m tests.aggregate_test

Aggregate the real GDELT files already downloaded into the cache:

    python3 -m tests.aggregate_test --real
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.gdelt.download import read_events
from src.transform.aggregate import to_country_day

CACHE_DIR = Path("data/cache/gdelt")


def make_events() -> pd.DataFrame:
    """
    Eight test hand-written events:

    id  country  recorded     root  quad  goldstein  articles  tone
    1   US       2026-10-05   01    1      2.0        1         1.0
    2   US       2026-10-05   14    3     -6.0        3        -3.0
    3   US       2026-10-05   14    3     -4.0        1        -1.0
    4   NI       2026-10-05   19    4    -10.0        2        -5.0
    5   NI       2026-10-06   01    1      0.0        4         2.0
    6   (none)   2026-10-05   01    1      1.0        1         0.0   dropped
    7   OS       2026-10-05   01    1      1.0        1         0.0   dropped
    4   NI       (repeat of event 4)                                  dropped
    """
    rows = [
        # id, DATEADDED,       FIPS, root, quad, gold, mentions, articles, tone
        (1, "20261005193000", "US", "01", 1,   2.0, 2, 1,  1.0),
        (2, "20261005194500", "US", "14", 3,  -6.0, 6, 3, -3.0),
        (3, "20261005200000", "US", "14", 3,  -4.0, 2, 1, -1.0),
        (4, "20261005200000", "NI", "19", 4, -10.0, 4, 2, -5.0),
        (5, "20261006001500", "NI", "01", 1,   0.0, 8, 4,  2.0),
        (6, "20261005193000", None, "01", 1,   1.0, 1, 1,  0.0),
        (7, "20261005193000", "OS", "01", 1,   1.0, 1, 1,  0.0),
        (4, "20261005200000", "NI", "19", 4, -10.0, 4, 2, -5.0),
    ]
    frame = pd.DataFrame(rows, columns=[
        "GlobalEventID", "DATEADDED", "ActionGeo_CountryCode",
        "EventRootCode", "QuadClass", "GoldsteinScale",
        "NumMentions", "NumArticles", "AvgTone",
    ])
    # Day is deliberately a year earlier. It must not affect the result.
    frame["Day"] = "20251005"
    frame["SOURCEURL"] = "http://example.com/some-persons-name"
    text_columns = (
        "DATEADDED", "ActionGeo_CountryCode", "EventRootCode", "Day",
    )
    for column in text_columns:
        frame[column] = frame[column].astype("string")
    return frame


def get_row(table: pd.DataFrame, iso3: str, day: str) -> pd.Series:
    same_country = table["iso3"] == iso3
    same_day = table["date"] == pd.Timestamp(day)
    match = table[same_country & same_day]
    assert len(match) == 1, f"expected one row, got {len(match)}"
    return match.iloc[0]


def test_one_row_per_country_day() -> None:
    table = to_country_day(make_events())
    found = [(r.iso3, str(r.date.date())) for r in table.itertuples()]
    assert found == [
        ("NGA", "2026-10-05"),
        ("USA", "2026-10-05"),
        ("NGA", "2026-10-06"),
    ], found


def test_totals_and_averages() -> None:
    usa = get_row(to_country_day(make_events()), "USA", "2026-10-05")

    assert usa["n_events"] == 3
    assert usa["n_mentions"] == 10          # 2 + 6 + 2
    assert usa["n_articles"] == 5           # 1 + 3 + 1
    assert abs(usa["tone_mean"] - (-1.0)) < 1e-9         # (1 - 3 - 1) / 3
    # (1*1 + -3*3 + -1*1) / (1 + 3 + 1) = -9 / 5
    assert abs(usa["tone_mean_weighted"] - (-1.8)) < 1e-9
    assert abs(usa["goldstein_mean"] - (-8.0 / 3)) < 1e-9  # (2 - 6 - 4) / 3


def test_counts_per_code() -> None:
    usa = get_row(to_country_day(make_events()), "USA", "2026-10-05")

    assert usa["root_01"] == 1 and usa["root_14"] == 2
    assert usa["root_19"] == 0              # absent codes are 0, not missing
    assert usa["quad_1"] == 1 and usa["quad_3"] == 2 and usa["quad_4"] == 0
    assert usa["root_other"] == 0


def test_all_code_columns_always_exist() -> None:
    table = to_country_day(make_events())
    for number in range(1, 21):
        assert f"root_{number:02d}" in table.columns
    for number in range(1, 5):
        assert f"quad_{number}" in table.columns


def test_counts_add_up() -> None:
    """
    Every event lands in exactly one root column and one quad column.
    """
    table = to_country_day(make_events())
    root_columns = [c for c in table.columns if c.startswith("root_")]
    quad_columns = [c for c in table.columns if c.startswith("quad_")]
    assert (table[root_columns].sum(axis=1) == table["n_events"]).all()
    assert (table[quad_columns].sum(axis=1) == table["n_events"]).all()


def test_unusual_root_code_is_counted_as_other() -> None:
    events = make_events()
    events.loc[events["GlobalEventID"] == 1, "EventRootCode"] = "--"
    usa = get_row(to_country_day(events), "USA", "2026-10-05")
    assert usa["root_01"] == 0 and usa["root_other"] == 1
    assert usa["n_events"] == 3


def test_dropped_rows_are_recorded() -> None:
    table = to_country_day(make_events())
    assert table.attrs["dropped"] == {
        "rows_in": 8,
        "duplicate_event": 1,   # event 4 appears twice
        "no_country": 2,        # event 6 (empty) and event 7 (oceans)
        "bad_date": 0,
        "rows_used": 5,
    }, table.attrs["dropped"]


def test_date_comes_from_dateadded_not_day() -> None:
    table = to_country_day(make_events())
    assert table["date"].dt.year.eq(2026).all()


def test_no_source_url_in_output() -> None:
    table = to_country_day(make_events())
    assert "SOURCEURL" not in table.columns


def test_input_is_not_changed() -> None:
    events = make_events()
    before = events.copy()
    to_country_day(events)
    pd.testing.assert_frame_equal(events, before)


def check_real_files() -> None:
    """
    Aggregate every cached GDELT file and print the result.
    """
    paths = sorted(CACHE_DIR.glob("*.export.CSV.zip"))
    if not paths:
        raise SystemExit(
            f"no cached files in {CACHE_DIR}, run the fetch test first"
        )

    events = pd.concat([read_events(p) for p in paths], ignore_index=True)
    table = to_country_day(events)

    print(f"{len(paths)} files -> {len(table)} country-day rows, "
          f"{table['iso3'].nunique()} countries, "
          f"{table['date'].nunique()} day(s)")
    print(f"dropped: {table.attrs['dropped']}")

    used = table.attrs["dropped"]["rows_used"]
    assert table["n_events"].sum() == used, "events lost or double counted"

    print("\n--- ten country-days with the most events ---")
    columns = ["iso3", "date", "n_events", "n_articles", "tone_mean",
               "tone_mean_weighted", "goldstein_mean", "root_14", "root_19"]
    top = table.sort_values("n_events", ascending=False).head(10)
    print(top[columns].to_string(index=False, float_format="{:.2f}".format))

    print("\n--- events per QuadClass, all countries ---")
    print(table[[f"quad_{q}" for q in range(1, 5)]].sum().to_string())
    print(f"\nevents with an unusual root code: {table['root_other'].sum()}")
    print("\nREAL DATA CHECK OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="aggregate the cached GDELT files instead")
    args = parser.parse_args()

    if args.real:
        check_real_files()
        return

    tests = [
        test_one_row_per_country_day,
        test_totals_and_averages,
        test_counts_per_code,
        test_all_code_columns_always_exist,
        test_counts_add_up,
        test_unusual_root_code_is_counted_as_other,
        test_dropped_rows_are_recorded,
        test_date_comes_from_dateadded_not_day,
        test_no_source_url_in_output,
        test_input_is_not_changed,
    ]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\nALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
