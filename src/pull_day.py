"""
Pulls GDELT events for one day or a range of days, aggregates them, and
saves the country-day table. Run from the repo root:

    python3 -m src.pull_day                     (yesterday, in UTC)
    python3 -m src.pull_day --date 2026-10-09   (a chosen day)
    python3 -m src.pull_day --from 2026-10-01 --to 2026-10-09
    python3 -m src.pull_day --days 90           (the last 90 full days)

Other options:
    --force     pull a day again even if it is already stored
    --no-save   print the summary without saving to the database

What it does for each day:
    1. Selects the export files for that day (a full day is 96 files)
    2. Downloads each one into the cache, skipping any that fail
    3. Reads them and builds the country-day table
    4. Prints a summary (a full one for one day, one line per day for
       a range)
    5. Saves the table to the database

A day that is already stored as complete is skipped.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from src.countries.actor_codes import clean_actor_codes
from src.countries.codes import (
    NON_COUNTRY_CODES,
    UnknownCountryCode,
    load_crosswalk,
)
from src.gdelt.download import ChecksumMismatch, download, read_events
from src.gdelt.filelist import GdeltFile, fetch_index, select_day
from src.gdelt.schema import USED_COLUMNS
from src.store.database import (
    DB_PATH,
    FILES_PER_DAY,
    connect,
    load_pull_log,
    save_day,
)
from src.transform.aggregate import (
    UNREST_LOOSE_ROOT_CODES,
    to_country_day,
    to_event_code_day,
)

CACHE_DIR = Path("data/cache/gdelt")

# The names are only read so that an unknown code can be identified
# They are not used by the aggregation and are never stored
READ_COLUMNS: tuple[str, ...] = USED_COLUMNS + (
    "ActionGeo_FullName",
    "Actor1Name",
    "Actor2Name",
)


def explain_unknown_codes(
        events: pd.DataFrame,
        error: UnknownCountryCode,
    ) -> None:
    """Print each unknown country code with the places GDELT attached to it."""
    print(f"\nSTOPPED: {error}")
    print("\nMost common place names for each unknown code:")

    recognised = set(load_crosswalk()) | NON_COUNTRY_CODES | {""}
    codes = events["ActionGeo_CountryCode"].fillna("").str.strip().str.upper()
    for code in sorted(set(codes.unique()) - recognised):
        names = events.loc[codes == code, "ActionGeo_FullName"]
        places = names.value_counts().head(3)
        print(f"  {code}: {list(places.index)}")

    print("\nDownloaded files cached.")


def print_unmatched_actor_codes(
        events: pd.DataFrame,
        unmatched: dict[str, int],
        limit: int = 15,
    ) -> None:
    """
    Print the most common actor codes that are not a known country, each
    with the actor name seen most often.

    Most of these are regions or groups and need no action. A real country
    in this list should be added to src/countries/cameo_iso3.csv.
    """
    total = sum(unmatched.values())
    print(f"\n--- actor codes that are not a known country "
          f"({len(unmatched)} codes, {total:,} uses) ---")

    code_columns = {
        "Actor1CountryCode": "Actor1Name",
        "Actor2CountryCode": "Actor2Name",
    }
    for code, count in list(unmatched.items())[:limit]:
        names: list[pd.Series] = []
        for code_column, name_column in code_columns.items():
            matches = clean_actor_codes(events[code_column]) == code
            names.append(events.loc[matches, name_column])
        common = pd.concat(names).value_counts().head(2)
        print(f"  {code}: {count:>6,}  e.g. {list(common.index)}")


def print_unrest_by_code(codes: pd.DataFrame, limit: int = 10) -> None:
    """
    Print the most common full event codes within loose unrest (domestic,
    not aimed at media), with their average Goldstein value. This shows
    the detail that the root code alone hides, e.g. demonstrations (141)
    against riots (145), or arrests (173) inside root 17.
    """
    is_unrest = codes["event_code"].str[:2].isin(UNREST_LOOSE_ROOT_CODES)
    domestic = codes[
        is_unrest
        & (codes["relation"] == "domestic")
        & (codes["media_target"] == 0)
    ]
    by_code = domestic.groupby("event_code").agg(
        events=("n_events", "sum"), goldstein=("goldstein_sum", "sum")
    )
    by_code["per_event"] = by_code["goldstein"] / by_code["events"]
    top = by_code.sort_values("events", ascending=False).head(limit)

    print(f"\n--- event code table: {len(codes):,} rows; most common full "
          "codes in loose unrest ---")
    print(top.to_string(float_format="{:.1f}".format))


def pull_one_day(
        index: list[GdeltFile],
        target: date,
        connection: sqlite3.Connection | None,
        verbose: bool,
    ) -> str:
    """
    Pull, aggregate and save one day.

    Returns "saved", "incomplete", "empty" or "stopped". "stopped" means
    an unknown country code was found and the crosswalk needs a new row.
    """
    started = time.perf_counter()

    # 1. Which files exist for this day
    files = select_day(index, target)
    if not files:
        print(f"{target}: no files listed (is the date in the future?)")
        return "empty"

    # 2. Download + skip files that fail
    paths: list[Path] = []
    skipped: list[str] = []
    for file in files:
        try:
            paths.append(download(file, CACHE_DIR))
        except (httpx.HTTPError, ChecksumMismatch) as error:
            skipped.append(file.filename)
            if verbose:
                print(f"  SKIPPED {file.filename}: {error}")

    if not paths:
        print(f"{target}: nothing could be downloaded")
        return "empty"

    # 3. Read and aggregate
    events = pd.concat(
        [read_events(path, READ_COLUMNS) for path in paths], ignore_index=True
    )
    try:
        table = to_country_day(events)
        codes = to_event_code_day(events)
    except UnknownCountryCode as error:
        print(f"{target}: {len(paths)} files, {len(events):,} event rows")
        explain_unknown_codes(events, error)
        return "stopped"

    complete = len(paths) == FILES_PER_DAY
    dropped = table.attrs["dropped"]
    unmatched = table.attrs["actor_codes_unmatched"]
    countries = table["iso3"].nunique()

    # 4. Print summary
    if verbose:
        print(f"{target}: {len(files)} of {FILES_PER_DAY} export files listed")
        print(f"files ready: {len(paths)}, skipped: {len(skipped)}")
        print(f"event rows read: {len(events):,}")
        print(f"dropped: {dropped}")
        no_country = dropped["no_country"] / dropped["rows_in"]
        print(f"rows with no usable country: {no_country:.1%}")
        print(f"country-day rows: {len(table)}, countries: {countries}")

        print("\n--- rows per date (expect one date) ---")
        per_date = table.groupby("date")["n_events"].agg(["count", "sum"])
        print(per_date.to_string())

        print("\n--- ten countries with the most events ---")
        columns = ["iso3", "n_events", "tone_mean", "unrest_events",
                   "unrest_goldstein", "media_pressure_events",
                   "intl_conflict_events", "intl_conflict_goldstein",
                   "intl_coop_events"]
        top = table.sort_values("n_events", ascending=False).head(10)
        print(top[columns].to_string(
            index=False, float_format="{:.1f}".format
        ))

        print_unrest_by_code(codes)

        print_unmatched_actor_codes(events, unmatched)
        print(f"\nevents with an unusual root code: "
              f"{table['root_other'].sum()}")
        print(f"total time: {time.perf_counter() - started:.0f}s")
        if complete:
            print("\nFULL DAY PULL OK")
        else:
            print("\nNOTE: this is not a complete day, so the totals "
                  "are low.")

    # 5. Save
    status = "saved" if complete else "incomplete"
    if connection is not None:
        written = save_day(
            connection, table, target, len(files), len(paths), codes=codes
        )
        saved_text = f"saved {written} country rows"
    else:
        saved_text = "not saved (--no-save)"

    label = "complete day" if complete else "INCOMPLETE day"
    if verbose and connection is not None:
        print(f"{saved_text} for {target} to {DB_PATH} ({label})")
    elif verbose:
        print(saved_text)
    else:
        print(
            f"{target}: {len(paths)}/{FILES_PER_DAY} files, "
            f"{len(events):,} rows, {countries} countries, "
            f"{dropped['no_country'] / dropped['rows_in']:.1%} no country, "
            f"{saved_text} ({label}, "
            f"{time.perf_counter() - started:.0f}s)"
        )
    return status


def days_to_pull(args: argparse.Namespace) -> list[date]:
    """
    Work out the list of days asked for on the command line.
    """
    yesterday = datetime.now(UTC).date() - timedelta(days=1)

    if args.days is not None:
        if args.days < 1:
            raise SystemExit("--days must be 1 or more")
        first, last = yesterday - timedelta(days=args.days - 1), yesterday
    elif args.start is not None or args.end is not None:
        if args.start is None:
            raise SystemExit("--to needs --from as well")
        first, last = args.start, args.end or yesterday
    else:
        first = last = args.date or yesterday

    if first > last:
        raise SystemExit(f"the first day ({first}) is after the last ({last})")

    count = (last - first).days + 1
    return [first + timedelta(days=offset) for offset in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull one day, or a range of days, of GDELT events."
    )
    parser.add_argument(
        "--date", type=date.fromisoformat,
        help="one day to pull, as YYYY-MM-DD (default: yesterday in UTC)",
    )
    parser.add_argument(
        "--from", dest="start", type=date.fromisoformat,
        help="first day of a range, as YYYY-MM-DD",
    )
    parser.add_argument(
        "--to", dest="end", type=date.fromisoformat,
        help="last day of a range (default: yesterday in UTC)",
    )
    parser.add_argument(
        "--days", type=int,
        help="pull the last N full days, ending yesterday",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="pull a day again even if it is already stored as complete",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="print the summary without saving to the database",
    )
    args = parser.parse_args()

    chosen = [args.date is not None, args.start is not None or
              args.end is not None, args.days is not None]
    if sum(chosen) > 1:
        raise SystemExit("use only one of --date, --from/--to, --days")

    # WARNING level keeps the output short; httpx logs every request at INFO
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(message)s"
    )

    days = days_to_pull(args)
    verbose = len(days) == 1

    connection = None if args.no_save else connect(DB_PATH)
    try:
        # Days already stored as complete do not need pulling again
        already: set[str] = set()
        if connection is not None and not args.force:
            log = load_pull_log(connection)
            already = set(log.loc[log["complete"] == 1, "date"])

        todo = [day for day in days if day.isoformat() not in already]
        if len(todo) < len(days):
            print(f"{len(days) - len(todo)} day(s) already stored, skipped "
                  "(use --force to pull them again)")
        if not todo:
            return

        # The master file list is fetched once and shared by every day
        index = fetch_index()

        results: dict[str, int] = {}
        for day in todo:
            status = pull_one_day(index, day, connection, verbose)
            results[status] = results.get(status, 0) + 1
            if status == "stopped":
                break

        if not verbose:
            print(f"\nfinished: {results}")
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
