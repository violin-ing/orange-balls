"""
One-off checks raised by the research chat. Run from the repo root:

    python -m tests.criteria_checks              (all checks)
    python -m tests.criteria_checks --only 3     (one check)

Check 1 -- actor coding coverage (reads the database)
    Unrest requires actor 1 to be from the country ("domestic"), but the
    share's denominator so far is ALL located events. If the fraction of
    events with that kind of actor coding varies a lot between countries,
    the share partly measures coding quality. This prints how much it
    varies, and the unrest share under both denominators.

Check 2 -- are domestic root-19 events really domestic? (reads the cache)
    "Use conventional military force" events may be located in the acting
    military's own country even when the action was abroad. This prints,
    for a few large countries, how many domestic root-19 events have no
    second actor, and a sample of them to judge by eye.

Check 3 -- how many events a day does a country have? (reads the database)
    Shares from countries with few events are noisy. This prints, per
    country, the median number of events a day over all complete days,
    for each denominator in use, and how many countries fall below 50.
    Also the share of each country's located events with usable actor
    coding (domestic or foreign, i.e. not "no_country").

Source addresses are printed to the screen to help judge plausibility.
Nothing is stored.

Not a unit test: pytest does not collect this file.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.gdelt.download import read_events
from src.gdelt.schema import USED_COLUMNS
from src.store.database import (
    DB_PATH,
    connect,
    load_country_day,
    load_event_codes,
    load_pull_log,
)
from src.transform.aggregate import (
    RELATION_DOMESTIC,
    RELATION_NO_COUNTRY,
    UNREST_ROOT_CODES,
    _prepare,
)

CACHE_DIR = Path("data/cache/gdelt")
MIN_EVENTS = 200          # countries with fewer located events are skipped
ROOT19_COUNTRIES = ["USA", "GBR", "RUS", "ISR", "IND", "NGA"]
SAMPLE = 8
LOW_DATA = 50             # events a day, matches Settings.low_data_events


def check_actor_coding() -> None:
    print("=" * 72)
    print("CHECK 1: how much actor coding varies between countries")
    print("=" * 72)

    connection = connect(DB_PATH)
    codes = load_event_codes(connection)
    connection.close()
    if codes.empty:
        print("no event code rows stored; run src.pull_day first")
        return

    days = sorted(codes["date"].dt.date.unique())
    print(f"days in the database: {len(days)} ({days[0]} to {days[-1]})\n")

    by_relation = codes.pivot_table(
        index="iso3", columns="relation", values="n_events",
        aggfunc="sum", fill_value=0,
    )
    by_relation["total"] = by_relation.sum(axis=1)
    big = by_relation[by_relation["total"] >= MIN_EVENTS].copy()
    big["domestic_share"] = big["domestic"] / big["total"]

    print(f"countries with at least {MIN_EVENTS} located events: {len(big)}")
    print("share of located events coded 'domestic' (actor 1 local):")
    print(big["domestic_share"].describe(
        percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
    ).round(3).to_string())

    columns = ["total", "domestic", "domestic_share"]
    print("\nlowest 10:")
    print(big.sort_values("domestic_share")[columns].head(10)
          .round(3).to_string())
    print("\nhighest 10:")
    print(big.sort_values("domestic_share", ascending=False)[columns]
          .head(10).round(3).to_string())

    # Unrest share under the two denominators
    root = codes["event_code"].str[:2]
    unrest = codes[
        root.isin(UNREST_ROOT_CODES)
        & (codes["relation"] == RELATION_DOMESTIC)
        & (codes["media_target"] == 0)
    ].groupby("iso3")["n_events"].sum()
    big["unrest"] = unrest.reindex(big.index, fill_value=0)
    big["share_all"] = big["unrest"] / big["total"]
    big["share_domestic"] = big["unrest"] / big["domestic"]
    rank_all = big["share_all"].rank(ascending=False)
    rank_domestic = big["share_domestic"].rank(ascending=False)
    big["rank_change"] = (rank_all - rank_domestic).abs()

    print("\nstrict unrest share, two denominators (top 15 by events):")
    top = big.sort_values("total", ascending=False).head(15)
    print(top[["total", "unrest", "share_all", "share_domestic"]]
          .round(3).to_string())
    correlation = big["share_all"].corr(
        big["share_domestic"], method="spearman"
    )
    print(f"\nrank correlation between the two shares: {correlation:.2f}")
    print(f"median rank change: {big['rank_change'].median():.0f} places, "
          f"largest: {big['rank_change'].max():.0f} "
          f"(out of {len(big)} countries)")


def check_root19() -> None:
    print("\n" + "=" * 72)
    print("CHECK 2: domestic root-19 events, are they really domestic?")
    print("=" * 72)

    paths = sorted(CACHE_DIR.glob("*.export.CSV.zip"))
    if not paths:
        print(f"no cached files in {CACHE_DIR}")
        return

    columns = USED_COLUMNS + (
        "Actor1Name", "Actor2Name", "ActionGeo_FullName",
    )
    events = pd.concat(
        [read_events(p, columns) for p in paths], ignore_index=True
    )
    work = _prepare(events).work
    names = events.set_index("GlobalEventID")[
        ["Actor1Name", "Actor2Name", "ActionGeo_FullName", "SOURCEURL"]
    ]
    names = names[~names.index.duplicated()]
    work = work.join(names, on="GlobalEventID")
    print(f"{len(paths)} cached files, {len(events):,} events\n")

    fights = work[
        (work["EventRootCode"] == "19")
        & (work["relation"] == RELATION_DOMESTIC)
        & ~work["media_target"]
    ]
    for iso3 in ROOT19_COUNTRIES:
        rows = fights[fights["iso3"] == iso3]
        if rows.empty:
            print(f"{iso3}: no domestic root-19 events")
            continue
        no_target = rows["Actor2CountryCode"].fillna("").str.strip() == ""
        print(f"{iso3}: {len(rows)} domestic root-19 events, "
              f"{int(no_target.sum())} with no second actor "
              f"({no_target.mean():.0%})")
        print(f"  codes: {rows['EventCode'].value_counts().head(5).to_dict()}")

        sample = rows[no_target].head(SAMPLE)
        for r in sample.itertuples():
            domain = urlparse(str(r.SOURCEURL)).netloc
            path = urlparse(str(r.SOURCEURL)).path[-60:]
            print(f"    {r.EventCode}  {str(r.Actor1Name)[:22]:<22} "
                  f"-> {str(r.Actor2Name)[:18]:<18} "
                  f"at {str(r.ActionGeo_FullName)[:30]:<30} "
                  f"{domain}{path}")
        print()


def check_daily_counts() -> None:
    print("\n" + "=" * 72)
    print("CHECK 3: events a day per country, and usable actor coding")
    print("=" * 72)

    connection = connect(DB_PATH)
    log = load_pull_log(connection)
    days = sorted(pd.to_datetime(log.loc[log["complete"] == 1, "date"]))
    if not days:
        connection.close()
        print("no complete days stored; run src.pull_day first")
        return
    country_day = load_country_day(connection)
    codes = load_event_codes(connection)
    connection.close()
    country_day = country_day[country_day["date"].isin(days)]
    codes = codes[codes["date"].isin(days)]
    print(f"complete days: {len(days)} ({days[0].date()} to "
          f"{days[-1].date()})\n")

    def daily(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        """country x day, zero where a country had no row that day"""
        return frame.pivot_table(
            index="iso3", columns="date", values=column, aggfunc="sum",
            fill_value=0,
        ).reindex(columns=days, fill_value=0)

    denominators = {
        "all located events (media share)": daily(country_day, "n_events"),
        "domestic-coded events (unrest)": daily(
            codes[codes["relation"] == RELATION_DOMESTIC], "n_events"),
        "events with the country as an actor (international)": daily(
            country_day, "actor_events"),
    }
    for label, table in denominators.items():
        medians = table.median(axis=1)
        print(f"{label}: {len(medians)} countries")
        print("  median events a day, across countries: "
              + ", ".join(
                  f"{name} {medians.quantile(q):,.0f}"
                  for name, q in (("10%", 0.1), ("25%", 0.25),
                                  ("median", 0.5), ("75%", 0.75),
                                  ("90%", 0.9))
              ))
        low = (medians < LOW_DATA).sum()
        print(f"  countries with a median below {LOW_DATA} a day: {low} "
              f"of {len(medians)} ({low / len(medians):.0%})")
        zero_days = (table == 0).mean(axis=1)
        print(f"  countries with no events on half the days or more: "
              f"{(zero_days >= 0.5).sum()}\n")

    by_relation = codes.pivot_table(
        index="iso3", columns="relation", values="n_events",
        aggfunc="sum", fill_value=0,
    )
    total = by_relation.sum(axis=1)
    big = by_relation[total >= MIN_EVENTS * len(days)]
    usable = 1 - big.get(RELATION_NO_COUNTRY, 0) / big.sum(axis=1)
    print(f"share of located events with usable actor coding (countries "
          f"with at least {MIN_EVENTS} events a day, n = {len(big)}):")
    print(usable.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
          .round(3).to_string())
    print("\nlowest 10:")
    print(usable.sort_values().head(10).round(3).to_string())


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    checks = {1: check_actor_coding, 2: check_root19,
              3: check_daily_counts}
    for number, check in checks.items():
        if args.only in (None, number):
            check()


if __name__ == "__main__":
    main()
