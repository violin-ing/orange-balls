
"""
Pull GDELT events for one day or a date range, aggregate them and
save the results to the database

Run from the project root:

    python3 -m src.pull_day
    python3 -m src.pull_day --date 2026-10-09
    python3 -m src.pull_day --from 2026-10-01 --to 2026-10-09
    python3 -m src.pull_day --days 90
    python3 -m src.pull_day --from 2026-05-14 --to 2026-11-10 --manifest-only

Options:
    --force          pull days again even if already stored
    --no-save        print summaries without saving to the database
    --manifest-only  record the file manifest of days already pulled, from
                     GDELT's index and the cache, without re-aggregating

For each day, the script:
    1. Finds the available GDELT export files (96 for a full day)
    2. Downloads the files, skipping any that fail
    3. Reads and aggregates the events
    4. Prints a summary of the processed data
    5. Saves the results, unless --no-save is used, together with the day's
       file manifest: every file's address, GDELT's published MD5 checksum,
       its size and whether it was read (the gdelt_file table)

Days already stored as complete are skipped unless --force is used

The manifest is what allows the download cache to be deleted at the end of
the project (decision log 18.1): every file can be fetched again from GDELT
and checked against its recorded MD5. Run --manifest-only for days pulled
before the manifest existed BEFORE deleting the cache: it checks each file
against the cache, so after deletion every file would be marked unread.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import textwrap
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
    MANIFEST_COLUMNS,
    connect,
    load_pull_log,
    save_day,
    save_manifest,
)
from src.transform.aggregate import (
    UNREST_LOOSE_ROOT_CODES,
    to_country_day,
    to_event_code_day,
)

# === Settings ===

CACHE_DIR = Path("data/cache/gdelt")

# Additional columns are read to help investigate unfamiliar country
# and actor codes using the associated place and actor names
# These columns are only used for diagnostics and are not stored
READ_COLUMNS: tuple[str, ...] = USED_COLUMNS + (
    "ActionGeo_FullName",
    "Actor1Name",
    "Actor2Name",
)


# === 1. Explain unknown location country codes ===

def explain_unknown_codes(
    events: pd.DataFrame,
    error: UnknownCountryCode,
) -> None:
    """
    Explain which unknown location country codes caused processing
    to stop, using the place names supplied by GDELT
    """

    print(f"\nSTOPPED: {error}")
    print("\nMost common place names for each unknown code:")

    # Build a set of recognised country codes, including non-country codes
    recognised = set(load_crosswalk()) | NON_COUNTRY_CODES | {""}

    # Standardise the location codes before comparing them
    codes = (
        events["ActionGeo_CountryCode"]
        .fillna("")
        .str.strip()
        .str.upper()
    )

    # Identify codes that are not included in the recognised set
    for code in sorted(set(codes.unique()) - recognised):

        names = events.loc[codes == code, "ActionGeo_FullName"]

        # Show the three most frequently occurring place names
        places = names.value_counts().head(3)
        print(f"  {code}: {list(places.index)}")

    print("\nDownloaded files cached.")


# === 2. Print unmatched actor country codes ===

def print_unmatched_actor_codes(
    events: pd.DataFrame,
    unmatched: dict[str, int],
    limit: int = 15,
) -> None:
    """
    Show the most common actor country codes that are not recognised
    countries, together with examples of the associated actor names
    """

    total = sum(unmatched.values())

    if not unmatched:
        print("\n5. Actor codes that are not a country: none")
        return

    print(
        f"\n5. Actor codes that are not a country "
        f"({len(unmatched)} codes, {total:,} uses; left out of the totals)"
    )

    # Link each actor country-code column to its corresponding name column
    code_columns = {
        "Actor1CountryCode": "Actor1Name",
        "Actor2CountryCode": "Actor2Name",
    }

    # Examine the most frequently occurring unmatched codes
    for code, count in list(unmatched.items())[:limit]:

        names: list[pd.Series] = []

        # Check both actor columns because either actor may have
        # an unmatched country code
        for code_column, name_column in code_columns.items():

            # Clean actor codes before comparing them
            matches = clean_actor_codes(events[code_column]) == code

            names.append(events.loc[matches, name_column])

        # Combine the names and identify the two most common examples
        common = pd.concat(names).value_counts().head(2)

        print(f"   {code:<5} {count:>7,}   e.g. {', '.join(common.index)}")


# === 3. Print domestic unrest by event code ===

def print_unrest_by_code(codes: pd.DataFrame, limit: int = 10) -> None:
    """
    Show the most common full event codes classified as domestic unrest

    Also calculates average Goldstein scores and the share of events
    originating from an article's lead paragraph
    """

    # Identify event codes whose first two digits belong to the unrest 
    # categories
    is_unrest = codes["event_code"].str[:2].isin(UNREST_LOOSE_ROOT_CODES)

    # Restrict the analysis to domestic events that are not media-target events
    domestic = codes[
        is_unrest
        & (codes["relation"] == "domestic")
        & (codes["media_target"] == 0)
    ]

    # Calculate total events and combined Goldstein scores for each code
    by_code = domestic.groupby("event_code").agg(
        events=("n_events", "sum"),
        goldstein=("goldstein_sum", "sum"),
    )

    # Calculate the average Goldstein score per event code
    by_code["per_event"] = by_code["goldstein"] / by_code["events"]

    # Identify events originating from an article's lead paragraph
    lead = domestic[domestic["is_root"] == 1]

    # Calculate the proportion of events for each code that come
    # from a lead paragraph
    by_code["lead_share"] = (
        lead.groupby("event_code")["n_events"].sum()
        .reindex(by_code.index, fill_value=0) / by_code["events"]
    )

    # Rank event codes by frequency and retain the most common ones
    top = by_code.sort_values("events", ascending=False).head(limit)

    print(f"\n4. Domestic unrest: the {limit} most common event codes")

    # Readable column names for the printed table
    table = pd.DataFrame({
        "event code": top.index,
        "events": top["events"].to_numpy(),
        "avg Goldstein": top["per_event"].round(1).to_numpy(),
        "from lead paragraph": (top["lead_share"] * 100).round(0)
            .astype(int).astype(str).to_numpy(),
    })
    table["from lead paragraph"] += "%"

    print(textwrap.indent(table.to_string(index=False), "   "))


# === 4. Record which files a day was built from ===

def manifest_for(
        files: list[GdeltFile],
        target: date,
        read: set[str],
    ) -> pd.DataFrame:
    """
    One row per listed file: address, published MD5, size, and whether it
    was read (read holds the file names that were read).
    """
    return pd.DataFrame(
        [
            (target.isoformat(), f.url, f.md5, f.size_bytes,
             f.filename in read)
            for f in files
        ],
        columns=MANIFEST_COLUMNS,
    )


def cached_and_intact(file: GdeltFile) -> bool:
    """
    True when the file is in the cache and matches GDELT's published MD5.
    """
    path = CACHE_DIR / file.filename
    if not path.exists():
        return False

    return hashlib.md5(path.read_bytes()).hexdigest() == file.md5


def record_manifests(
        index: list[GdeltFile],
        days: list[date],
        connection: sqlite3.Connection,
    ) -> None:
    """
    Record the manifest of days that were pulled before the manifest
    existed, without downloading or aggregating anything. A file counts as
    read when the cache holds an intact copy (MD5 matches), which is what
    the stored totals were built from.
    """
    stored = set(load_pull_log(connection)["date"])

    for day in days:
        if day.isoformat() not in stored:
            print(f"{day}: not pulled yet, skipped")
            continue

        files = select_day(index, day)
        read = {f.filename for f in files if cached_and_intact(f)}
        save_manifest(connection, day, manifest_for(files, day, read))

        note = "" if len(read) == len(files) else (
            f" ({len(files) - len(read)} missing from the cache or not "
            "matching their MD5)"
        )
        print(f"{day}: {len(files)} files recorded, {len(read)} intact in "
              f"the cache{note}")


# === 5. Download and process one day of GDELT data ===

def pull_one_day(
    index: list[GdeltFile],
    target: date,
    connection: sqlite3.Connection | None,
    verbose: bool,
) -> str:
    """
    Download and process GDELT events for one day

    Returns "saved", "incomplete", "empty" or "stopped"
    "stopped" means an unknown location country code was found
    """

    started = time.perf_counter()

    # === 5.1 Find the files available for the requested date ===

    files = select_day(index, target)

    if not files:
        print(f"{target}: no files listed (is the date in the future?)")
        return "empty"

    # === 5.2 Download the available files ===

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

    # === 5.3 Read and aggregate the downloaded event data ===

    # Combine all successfully downloaded files into one DataFrame
    events = pd.concat(
        [read_events(path, READ_COLUMNS) for path in paths],
        ignore_index=True,
    )

    try:
        # Aggregate events into country-day and event-code-day tables
        table = to_country_day(events)
        codes = to_event_code_day(events)

    except UnknownCountryCode as error:
        print(f"{target}: {len(paths)} files, {len(events):,} event rows")
        explain_unknown_codes(events, error)
        return "stopped"

    # Check whether all expected export files were downloaded
    complete = len(paths) == FILES_PER_DAY

    dropped = table.attrs["dropped"]
    unmatched = table.attrs["actor_codes_unmatched"]
    countries = table["iso3"].nunique()

    # === 5.4 Print diagnostic summaries ===

    if verbose:

        print(f"\n=== GDELT pull for {target} ===")

        print("\n1. Download")
        print(f"   export files listed      {len(files)} of {FILES_PER_DAY}")
        print(f"   downloaded               {len(paths)} "
              f"({len(skipped)} skipped)")
        print(f"   event rows read          {len(events):,}")

        # Share of input rows without a usable country
        no_country = dropped["no_country"] / dropped["rows_in"]

        print("\n2. Clean and match events to countries")
        print(f"   duplicate events removed {dropped['duplicate_event']:,}")
        print(f"   rows with a bad date     {dropped['bad_date']:,}")
        print(f"   rows with no country     {dropped['no_country']:,} "
              f"({no_country:.1%})")
        print(f"   rows used                {dropped['rows_used']:,}")
        unusual = int(table["root_other"].sum())
        print(f"   unusual root codes       {unusual:,}")
        print(f"   countries                {countries}")

        # Every row should belong to the day being pulled
        dates = table["date"].nunique()
        if dates != 1:
            print(f"   NOTE: the table covers {dates} dates (expected 1)")

        print("\n3. Country totals for the day (the 10 countries with "
              "the most events)")

        # Readable column names for the printed table
        columns = {
            "iso3": "country",
            "n_events": "events",
            "tone_mean": "avg tone",
            "unrest_events": "unrest",
            "unrest_goldstein": "unrest Goldstein",
            "media_pressure_events": "media pressure",
            "intl_conflict_events": "intl conflict",
            "intl_conflict_goldstein": "conflict Goldstein",
            "intl_coop_events": "intl cooperation",
        }

        # Rank countries by event frequency and retain the top ten
        top = table.sort_values("n_events", ascending=False).head(10)

        print(textwrap.indent(
            top[list(columns)].rename(columns=columns).to_string(
                index=False, float_format="{:.1f}".format),
            "   ",
        ))

        print_unrest_by_code(codes)
        print_unmatched_actor_codes(events, unmatched)

    # === 5.5 Save the aggregated results and the file manifest ===

    status = "saved" if complete else "incomplete"

    if connection is not None:

        # Save the country-level and event-code-level tables
        # The file counts are also passed so that completeness can be recorded
        read = {path.name for path in paths}
        written = save_day(
            connection, table, target, len(files), len(paths), codes=codes,
            manifest=manifest_for(files, target, read),
        )

        saved_text = f"saved {written} country rows"

    else:
        # If --no-save was specified, process and summarise the data
        # without writing the results to the database
        saved_text = "not saved (--no-save)"

    label = "complete day" if complete else "INCOMPLETE day"

    if verbose:
        print("\n6. Save")
        print(f"   {saved_text}" + (f" to {DB_PATH}" if connection else ""))
        print(f"   time taken: {time.perf_counter() - started:.0f}s")

        # Flag whether the downloaded data represents a complete day
        if complete:
            print("\nDone: full day pulled.")
        else:
            print("\nDone, but this is NOT a complete day, so the totals "
                  "are low.")

    else:
        print(
            f"{target}: {len(paths)}/{FILES_PER_DAY} files, "
            f"{len(events):,} rows, {countries} countries, "
            f"{dropped['no_country'] / dropped['rows_in']:.1%} no country, "
            f"{saved_text} ({label}, "
            f"{time.perf_counter() - started:.0f}s)"
        )

    return status


# === 6. Determine which dates to pull ===

def days_to_pull(args: argparse.Namespace) -> list[date]:
    """
    Return the dates requested by the command-line arguments

    Supports a single date, a date range or the last N full days
    """

    # Use yesterday because the current UTC day may not have a complete
    # set of GDELT export files yet
    yesterday = datetime.now(UTC).date() - timedelta(days=1)

    # Option 1: retrieve the last N full days, ending yesterday
    if args.days is not None:

        if args.days < 1:
            raise SystemExit("--days must be 1 or more")

        # Calculate the first and last dates in the requested period
        first, last = yesterday - timedelta(days=args.days - 1), yesterday

    # Option 2: retrieve a specified date range
    elif args.start is not None or args.end is not None:

        if args.start is None:
            raise SystemExit("--to needs --from as well")

        # If no end date is supplied, use yesterday
        first, last = args.start, args.end or yesterday

    # Option 3: retrieve a single date
    else:
        first = last = args.date or yesterday

    if first > last:
        raise SystemExit(f"the first day ({first}) is after the last ({last})")

    count = (last - first).days + 1

    # Return every date in the requested period, including both endpoints
    return [
        first + timedelta(days=offset)
        for offset in range(count)
    ]


# === 7. Main function ===

def main() -> None:
    """
    Main entry point for the GDELT data-pulling script

    Handles command-line arguments, determines which dates need
    processing and coordinates the processing of each requested day
    """

    # === 7.1 Define command-line arguments ===

    parser = argparse.ArgumentParser(
        description="Pull one day, or a range of days, of GDELT events."
    )

    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="one day to pull, as YYYY-MM-DD (default: yesterday in UTC)",
    )

    parser.add_argument(
        "--from",
        dest="start",
        type=date.fromisoformat,
        help="first day of a range, as YYYY-MM-DD",
    )

    parser.add_argument(
        "--to",
        dest="end",
        type=date.fromisoformat,
        help="last day of a range (default: yesterday in UTC)",
    )

    parser.add_argument(
        "--days",
        type=int,
        help="pull the last N full days, ending yesterday",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="pull a day again even if it is already stored as complete",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="print the summary without saving to the database",
    )

    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="record the file manifest of days already pulled, without "
             "pulling them again",
    )

    args = parser.parse_args()

    # === 7.2 Validate the date selection ===

    # Record which date-selection options have been supplied
    chosen = [
        args.date is not None,
        args.start is not None or args.end is not None,
        args.days is not None,
    ]

    # Only one date-selection method can be used at a time
    if sum(chosen) > 1:
        raise SystemExit("use only one of --date, --from/--to, --days")

    if args.manifest_only and (args.no_save or args.force):
        raise SystemExit("--manifest-only cannot be used with --no-save "
                         "or --force")

    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(message)s"
    )

    days = days_to_pull(args)

    # Use detailed diagnostic output when processing only one day
    verbose = len(days) == 1

    # === 7.3 Connect to the database and identify dates to process ===

    connection = None if args.no_save else connect(DB_PATH)

    try:
        if args.manifest_only:
            record_manifests(fetch_index(), days, connection)
            return

        already: set[str] = set()

        # Retrieve the database pull log and identify dates already
        # stored as complete, unless --force was specified
        if connection is not None and not args.force:
            log = load_pull_log(connection)
            already = set(log.loc[log["complete"] == 1, "date"])

        # Keep only dates that have not already been stored as complete
        todo = [
            day for day in days
            if day.isoformat() not in already
        ]

        if len(todo) < len(days):
            print(
                f"{len(days) - len(todo)} day(s) already stored, skipped "
                "(use --force to pull them again)"
            )

        if not todo:
            return

        # === 7.4 Retrieve the GDELT file index ===

        # Download the master index once and reuse it for every requested day
        index = fetch_index()

        # Count how many days finish with each processing status
        results: dict[str, int] = {}

        # === 7.5 Process each requested date ===

        for day in todo:

            status = pull_one_day(index, day, connection, verbose)

            # Increment the count for the returned processing status
            results[status] = results.get(status, 0) + 1

            # Stop processing if an unknown location country code is found
            if status == "stopped":
                break

        if not verbose:
            print(f"\nfinished: {results}")

    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
