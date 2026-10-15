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
from src.transform.aggregate import (
    UNREST_ROOT_CODES,
    to_country_day,
    to_event_code_day,
)

CACHE_DIR = Path("data/cache/gdelt")


def make_events() -> pd.DataFrame:
    """
    Hand-written events, small enough to work out the answers by hand.

    id place day   code quad articles tone  actor1 actor2  target
    1  US    5 Oct 010  1    1         1.0  USA    GBR             (a)
    2  US    5 Oct 141  3    3        -3.0  USA    -               (b)
    3  US    5 Oct 145  3    1        -1.0  -      -               (c)
    4  NI    5 Oct 190  4    2        -5.0  NGA    USA             (d)
    5  NI    6 Oct 010  1    4         2.0  EUR    NGA             (e)
    6  none  5 Oct 010  1    1         0.0  FRA    DEU             (f)
    7  OS    5 Oct 010  1    1         0.0  -      -               (g)
    9  US    5 Oct 172  4    2        -4.0  USA    USA     media   (h)
    4  (repeat of event 4, must be ignored)

    (a) cooperation between two countries: counts for USA and for GBR
    (b) unrest, strict and loose: actor 1 is local, actor 2 is absent
    (c) unrest, loose only: no actor has a country
    (d) conflict between two countries: counts for NGA and for USA.
        Not unrest, because actor 2 is foreign.
    (e) EUR is a region, not a country, so only NGA takes part
    (f) no location, so no location measures, but FRA and DEU still
        get a cooperation event as actors
    (g) oceans: no country at all
    (h) pressure on media, and also unrest (both actors are local)

    Codes: 141 is "demonstrate", 145 is "violent protest or riot", so the
    two protests share root 14 but not their full code.
    """
    rows = [
        # id, DATEADDED, FIPS, root, quad, goldstein, mentions, articles,
        # tone, actor1, actor2, actor2 type
        (1, "20261005193000", "US", "01", 1, 2.0, 2, 1, 1.0,
         "USA", "GBR", None),
        (2, "20261005194500", "US", "14", 3, -6.0, 6, 3, -3.0,
         "USA", None, None),
        (3, "20261005200000", "US", "14", 3, -4.0, 2, 1, -1.0,
         None, None, None),
        (4, "20261005200000", "NI", "19", 4, -10.0, 4, 2, -5.0,
         "NGA", "USA", None),
        (5, "20261006001500", "NI", "01", 1, 0.0, 8, 4, 2.0,
         "EUR", "NGA", None),
        (6, "20261005193000", None, "01", 1, 1.0, 1, 1, 0.0,
         "FRA", "DEU", None),
        (7, "20261005193000", "OS", "01", 1, 1.0, 1, 1, 0.0,
         None, None, None),
        (9, "20261005201500", "US", "17", 4, -5.0, 2, 2, -4.0,
         "USA", "USA", "MED"),
        (4, "20261005200000", "NI", "19", 4, -10.0, 4, 2, -5.0,
         "NGA", "USA", None),
    ]

    frame = pd.DataFrame(rows, columns=[
        "GlobalEventID", "DATEADDED", "ActionGeo_CountryCode",
        "EventRootCode", "QuadClass", "GoldsteinScale",
        "NumMentions", "NumArticles", "AvgTone",
        "Actor1CountryCode", "Actor2CountryCode", "Actor2Type1Code",
    ])

    frame["EventCode"] = frame["GlobalEventID"].map({
        1: "010", 2: "141", 3: "145", 4: "190",
        5: "010", 6: "010", 7: "010", 9: "172",
    })

    frame["Actor2Type2Code"] = None
    frame["Actor2Type3Code"] = None

    # Day is deliberately a year earlier (it must not affect the result)
    frame["Day"] = "20251005"

    frame["SOURCEURL"] = "http://example.com/some-persons-name"

    text_columns = (
        "DATEADDED", "ActionGeo_CountryCode", "EventCode",
        "EventRootCode", "Day",
        "Actor1CountryCode", "Actor2CountryCode",
        "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
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
        ("DEU", "2026-10-05"),
        ("FRA", "2026-10-05"),
        ("GBR", "2026-10-05"),
        ("NGA", "2026-10-05"),
        ("USA", "2026-10-05"),
        ("NGA", "2026-10-06"),
    ], found


def test_totals_and_averages() -> None:
    usa = get_row(to_country_day(make_events()), "USA", "2026-10-05")

    assert usa["n_events"] == 4 # events 1, 2, 3 and 9
    assert usa["n_mentions"] == 12 # 2 + 6 + 2 + 2
    assert usa["n_articles"] == 7 # 1 + 3 + 1 + 2
    # (1 - 3 - 1 - 4) / 4
    assert abs(usa["tone_mean"] - (-1.75)) < 1e-9
    # (1*1 + -3*3 + -1*1 + -4*2) / (1 + 3 + 1 + 2) = -17 / 7
    assert abs(usa["tone_mean_weighted"] - (-17 / 7)) < 1e-9
    # (2 - 6 - 4 - 5) / 4
    assert abs(usa["goldstein_mean"] - (-3.25)) < 1e-9


def test_counts_per_code() -> None:
    usa = get_row(to_country_day(make_events()), "USA", "2026-10-05")

    assert usa["root_01"] == 1 and usa["root_14"] == 2
    assert usa["root_17"] == 1
    assert usa["root_19"] == 0 # absent codes are 0, not missing
    assert usa["quad_1"] == 1 and usa["quad_3"] == 2 and usa["quad_4"] == 1
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
    assert usa["n_events"] == 4


def test_unrest_strict_and_loose() -> None:
    table = to_country_day(make_events())
    usa = get_row(table, "USA", "2026-10-05")
    nga = get_row(table, "NGA", "2026-10-05")

    # Strict: events 2 and 9 (actor 1 is local, actor 2 local or absent)
    assert usa["unrest_events"] == 2
    assert usa["unrest_articles"] == 5           # 3 + 2
    # Loose also takes event 3, where no actor has a country
    assert usa["unrest_loose_events"] == 3
    assert usa["unrest_loose_articles"] == 6     # 3 + 1 + 2
    # Event 4 is a fight in Nigeria, but actor 2 is the United States
    assert nga["unrest_events"] == 0 and nga["unrest_loose_events"] == 0


def test_media_pressure() -> None:
    table = to_country_day(make_events())
    usa = get_row(table, "USA", "2026-10-05")
    assert usa["media_pressure_events"] == 1     # event 9
    assert usa["media_pressure_articles"] == 2
    assert get_row(table, "NGA", "2026-10-05")["media_pressure_events"] == 0


def test_media_code_in_second_type_slot_counts() -> None:
    events = make_events()
    is_nine = events["GlobalEventID"] == 9
    events.loc[is_nine, "Actor2Type1Code"] = "GOV"
    events.loc[is_nine, "Actor2Type2Code"] = "MED"
    usa = get_row(to_country_day(events), "USA", "2026-10-05")
    assert usa["media_pressure_events"] == 1


def test_international_conflict_and_cooperation() -> None:
    table = to_country_day(make_events())
    usa = get_row(table, "USA", "2026-10-05")
    nga = get_row(table, "NGA", "2026-10-05")

    # USA takes part in events 1, 2, 4 and 9
    # Event 9 has USA as both actors and is counted once
    assert usa["actor_events"] == 4
    assert usa["actor_articles"] == 8 # 1 + 3 + 2 + 2
    assert usa["intl_coop_events"] == 1 # event 1, with GBR
    assert usa["intl_coop_articles"] == 1
    assert usa["intl_conflict_events"] == 1 # event 4, with NGA
    assert usa["intl_conflict_articles"] == 2

    # The same conflict event also counts for Nigeria
    assert nga["actor_events"] == 1 and nga["intl_conflict_events"] == 1
    assert nga["intl_coop_events"] == 0


def test_country_seen_only_as_actor_gets_a_row() -> None:
    table = to_country_day(make_events())
    gbr = get_row(table, "GBR", "2026-10-05")
    assert gbr["n_events"] == 0 and gbr["actor_events"] == 1
    assert gbr["intl_coop_events"] == 1
    assert pd.isna(gbr["tone_mean"]) # no located events

    # Event 6 has no location but still counts for its two actors
    fra = get_row(table, "FRA", "2026-10-05")
    deu = get_row(table, "DEU", "2026-10-05")
    assert fra["intl_coop_events"] == 1 and deu["intl_coop_events"] == 1


def test_region_code_is_not_a_country() -> None:
    table = to_country_day(make_events())
    assert "EUR" not in table["iso3"].tolist()
    assert table.attrs["actor_codes_unmatched"] == {"EUR": 1}
    # Event 5: EUR is not a country, so nothing is cross-border
    nga = get_row(table, "NGA", "2026-10-06")
    assert nga["actor_events"] == 1 and nga["intl_coop_events"] == 0


def test_goldstein_sums() -> None:
    table = to_country_day(make_events())
    usa = get_row(table, "USA", "2026-10-05")
    nga = get_row(table, "NGA", "2026-10-05")

    assert usa["unrest_goldstein"] == -11.0 # events 2 and 9
    assert usa["unrest_loose_goldstein"] == -15.0 # plus event 3
    assert usa["media_pressure_goldstein"] == -5.0 # event 9
    assert usa["actor_goldstein"] == -19.0 # 1, 2, 4 and 9
    assert usa["intl_coop_goldstein"] == 2.0 # event 1
    assert usa["intl_conflict_goldstein"] == -10.0 # event 4
    assert nga["intl_conflict_goldstein"] == -10.0 # same event
    # A criterion with no events has a sum of 0, not a missing value
    assert nga["unrest_goldstein"] == 0.0


def test_event_code_table() -> None:
    codes = to_event_code_day(make_events())
    found = [
        (r.iso3, str(r.date.date()), r.event_code, r.relation,
         r.n_events, r.n_articles, r.goldstein_sum)
        for r in codes.itertuples()
    ]
    assert found == [
        ("NGA", "2026-10-05", "190", "foreign", 1, 2, -10.0),
        ("NGA", "2026-10-06", "010", "foreign", 1, 4, 0.0),
        ("USA", "2026-10-05", "010", "foreign", 1, 1, 2.0),
        ("USA", "2026-10-05", "141", "domestic", 1, 3, -6.0),
        ("USA", "2026-10-05", "145", "no_country", 1, 1, -4.0),
        ("USA", "2026-10-05", "172", "domestic", 1, 2, -5.0),
    ], found


def test_event_code_table_agrees_with_country_day() -> None:
    """
    The two tables are built from the same events and must agree.
    """
    events = make_events()
    table = to_country_day(events).set_index(["iso3", "date"])
    codes = to_event_code_day(events)
    keys = ["iso3", "date"]

    totals = codes.groupby(keys)["n_events"].sum()
    located = table.loc[table["n_events"] > 0, "n_events"]
    assert totals.sort_index().equals(located.sort_index())

    is_unrest = codes["event_code"].str[:2].isin(UNREST_ROOT_CODES)
    strict = codes[is_unrest & (codes["relation"] == "domestic")]
    loose = codes[is_unrest & (codes["relation"] != "foreign")]
    for subset, column in ((strict, "unrest"), (loose, "unrest_loose")):
        sums = subset.groupby(keys)[["n_events", "goldstein_sum"]].sum()
        for key, row in sums.iterrows():
            assert table.loc[key, f"{column}_events"] == row["n_events"]
            assert table.loc[key, f"{column}_goldstein"] == (
                row["goldstein_sum"]
            )


def test_dropped_rows_are_recorded() -> None:
    table = to_country_day(make_events())
    assert table.attrs["dropped"] == {
        "rows_in": 9,
        "duplicate_event": 1, # event 4 appears twice
        "no_country": 2, # event 6 (empty) and event 7 (oceans)
        "bad_date": 0,
        "rows_used": 6,
    }, table.attrs["dropped"]
    assert table["n_events"].sum() == 6


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


def share(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "n/a"


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

    totals = table.sum(numeric_only=True)
    located = int(totals["n_events"])
    acting = int(totals["actor_events"])

    print("\n--- criteria inputs, all countries and days together ---")
    print(f"unrest (strict):      {int(totals['unrest_events']):>8,}  "
          f"{share(int(totals['unrest_events']), located)} of located")
    print(f"unrest (loose):       {int(totals['unrest_loose_events']):>8,}  "
          f"{share(int(totals['unrest_loose_events']), located)} of located")
    print(f"pressure on media:    {int(totals['media_pressure_events']):>8,}"
          f"  {share(int(totals['media_pressure_events']), located)} "
          "of located")
    print(f"intl conflict:        {int(totals['intl_conflict_events']):>8,}"
          f"  {share(int(totals['intl_conflict_events']), acting)} "
          "of actor events")
    print(f"intl cooperation:     {int(totals['intl_coop_events']):>8,}  "
          f"{share(int(totals['intl_coop_events']), acting)} "
          "of actor events")

    with_media = int((table["media_pressure_events"] > 0).sum())
    print(f"country-days with any pressure on media: {with_media} "
          f"of {len(table)}")

    print("\n--- ten country-days with the most events ---")
    columns = ["iso3", "date", "n_events", "unrest_events",
               "unrest_loose_events", "media_pressure_events",
               "actor_events", "intl_conflict_events", "intl_coop_events"]
    top = table.sort_values("n_events", ascending=False).head(10)
    print(top[columns].to_string(index=False))

    codes = to_event_code_day(events)
    is_unrest = codes["event_code"].str[:2].isin(UNREST_ROOT_CODES)
    domestic = codes[is_unrest & (codes["relation"] == "domestic")]
    by_code = domestic.groupby("event_code").agg(
        events=("n_events", "sum"), goldstein=("goldstein_sum", "sum")
    )
    by_code["goldstein_per_event"] = by_code["goldstein"] / by_code["events"]
    print(f"\n--- event code table: {len(codes):,} rows ---")
    print("most common full codes within strict unrest:")
    top_codes = by_code.sort_values("events", ascending=False).head(12)
    print(top_codes.to_string(float_format="{:.1f}".format))

    unmatched = table.attrs["actor_codes_unmatched"]
    print(f"\n--- actor codes that are not a known country "
          f"({len(unmatched)} codes, {sum(unmatched.values()):,} uses) ---")
    print(dict(list(unmatched.items())[:25]))

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
        test_unrest_strict_and_loose,
        test_media_pressure,
        test_media_code_in_second_type_slot_counts,
        test_international_conflict_and_cooperation,
        test_country_seen_only_as_actor_gets_a_row,
        test_region_code_is_not_a_country,
        test_goldstein_sums,
        test_event_code_table,
        test_event_code_table_agrees_with_country_day,
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
