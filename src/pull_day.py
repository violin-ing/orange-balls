"""
Pulls one full day of GDELT events and aggregates it. Run from the repo root:

    python3 -m src.pull_day                     (yesterday, in UTC)
    python3 -m src.pull_day --date 2026-10-09   (a chosen day)
    python3 -m src.pull_day --no-save           (print the summary only)

What it does:
    1. Fetches the master file list
    2. Selects the export files for that day (a full day is 96 files)
    3. Downloads each one into the cache, skipping any that fail
    4. Reads them and builds the country-day table
    5. Prints a summary
    6. Saves the table to the database (skip this with --no-save)

Pulling a day that is already stored replaces it, so it is safe to run twice.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
from src.store.database import DB_PATH, FILES_PER_DAY, connect, save_day

from src.countries.codes import (
    NON_COUNTRY_CODES,
    UnknownCountryCode,
    load_crosswalk,
)
from src.gdelt.download import ChecksumMismatch, download, read_events
from src.gdelt.filelist import fetch_index, select_day
from src.gdelt.schema import USED_COLUMNS
from src.transform.aggregate import to_country_day

CACHE_DIR = Path("data/cache/gdelt")

# The place name is read so that an unknown country code can be identified
READ_COLUMNS: tuple[str, ...] = USED_COLUMNS + ("ActionGeo_FullName",)


def explain_unknown_codes(
        events: pd.DataFrame,
        error: UnknownCountryCode,
    ) -> None:
    """
    Print each unknown country code with the places GDELT attached to it.
    """
    print(f"\nSTOPPED: {error}")
    print("\nMost common place names for each unknown code:")

    recognised = set(load_crosswalk()) | NON_COUNTRY_CODES | {""}
    codes = events["ActionGeo_CountryCode"].fillna("").str.strip().str.upper()
    for code in sorted(set(codes.unique()) - recognised):
        names = events.loc[codes == code, "ActionGeo_FullName"]
        places = names.value_counts().head(3)
        print(f"  {code}: {list(places.index)}")

    print("\nDownloaded files cached.")


def pull_day(target: date, save: bool = True) -> pd.DataFrame | None:
    started = time.perf_counter()

    # Find out which files exist for the target date
    index = fetch_index()
    files = select_day(index, target)
    print(f"{target}: {len(files)} of {FILES_PER_DAY} export files listed")
    if not files:
        print("No files for this day. Is the date in the future?")
        return None

    # Download files (skip those that fail)
    paths: list[Path] = []
    skipped: list[str] = []
    for number, file in enumerate(files, start=1):
        try:
            paths.append(download(file, CACHE_DIR))
        except (httpx.HTTPError, ChecksumMismatch) as error:
            skipped.append(file.filename)
            print(f"  SKIPPED {file.filename}: {error}")
        if number % 24 == 0:
            print(f"  {number} files checked")

    print(f"files ready: {len(paths)}, skipped: {len(skipped)}")
    if not paths:
        print("Nothing could be downloaded.")
        return None

    # Read and aggregate
    events = pd.concat(
        [read_events(path, READ_COLUMNS) for path in paths], ignore_index=True
    )
    print(f"event rows read: {len(events):,}")

    try:
        table = to_country_day(events)
    except UnknownCountryCode as error:
        explain_unknown_codes(events, error)
        return None

    # Print summary
    dropped = table.attrs["dropped"]
    no_country = dropped["no_country"] / dropped["rows_in"]
    print(f"dropped: {dropped}")
    print(f"rows with no usable country: {no_country:.1%}")

    countries = table["iso3"].nunique()
    print(f"country-day rows: {len(table)}, countries: {countries}")

    print("\n--- rows per date (expect one date) ---")
    print(table.groupby("date")["n_events"].agg(["count", "sum"]).to_string())

    print("\n--- top ten countries with the most events ---")
    columns = ["iso3", "n_events", "n_articles", "tone_mean",
               "tone_mean_weighted", "goldstein_mean", "root_14", "root_19"]
    top = table.sort_values("n_events", ascending=False).head(10)
    print(top[columns].to_string(index=False, float_format="{:.2f}".format))

    print(f"\nevents with an unusual root code: {table['root_other'].sum()}")
    print(f"total time: {time.perf_counter() - started:.0f}s")

    complete = len(paths) == FILES_PER_DAY
    if complete:
        print("\nFULL DAY PULL OK")
    else:
        print("\nNOTE: this is not a complete day, so the totals are low.")

    # Save file
    if save:
        connection = connect(DB_PATH)
        try:
            written = save_day(
                connection, table, target, len(files), len(paths)
            )
        finally:
            connection.close()
        status = "complete" if complete else "INCOMPLETE"
        print(
            f"saved {written} country rows for {target} "
            f"to {DB_PATH} ({status} day)"
        )
    else:
        print("not saved (--no-save)")

    return table


def main() -> None:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)

    parser = argparse.ArgumentParser(
        description="Pull one full day of GDELT events."
    )
    parser.add_argument(
        "--date", type=date.fromisoformat, default=yesterday,
        help="day to pull, as YYYY-MM-DD (default: yesterday in UTC)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="print the summary without saving to the database",
    )
    args = parser.parse_args()

    # WARNING level keeps the output short. httpx logs every request at INFO.
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(message)s"
    )

    pull_day(args.date, save=not args.no_save)


if __name__ == "__main__":
    main()
