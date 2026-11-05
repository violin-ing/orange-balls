"""
Smoke test for the GDELT fetch code (under gdelt/ from repo root).

Usage: (run from repo root)
    python3 -m tests.gdelt_fetch_test --step 1
    python3 -m tests.gdelt_fetch_test --step 2
    python3 -m tests.gdelt_fetch_test --step 3 --limit 8

Step 1: Parse three sample index lines (no network needed)
Step 2: Fetch lastupdate.txt (small network call)
Step 3: Fetch the master list, download --limit export files, read them
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from src.gdelt.download import ChecksumMismatch, download, read_events
from src.gdelt.filelist import LAST_UPDATE, fetch_index, parse_index

CACHE_DIR = Path("data/cache/gdelt")

# Example lines from the GDELT index, plus a blank line and a broken line
BASE = "http://data.gdeltproject.org/gdeltv2/20150218230000"

SAMPLE_INDEX = "\n".join([
    f"150383 297a16b493de7cf6ca809a7cc31d0b93 {BASE}.export.CSV.zip",
    f"318084 bb27f78ba45f69a17ea6ed7755e9f8ff {BASE}.mentions.CSV.zip",
    f"10768507 ea8dde0beb0ba98810a92db068c0ce99 {BASE}.gkg.csv.zip",
    "",
    "this line is broken on purpose",
])


def step_1() -> None:
    """
    Parse sample lines. The blank line and the broken line must be skipped.
    """
    files = parse_index(SAMPLE_INDEX)
    for f in files:
        print(f.stamp, f.stream, f.size_bytes, f.day, f.filename)

    assert len(files) == 3, f"expected 3 records, got {len(files)}"
    assert [f.stream for f in files] == ["export", "mentions", "gkg"]
    assert all(f.stamp == "20150218230000" for f in files)

    print("STEP 1 OK")


def step_2() -> None:
    """
    Fetch the small 'latest three files' index from the live service.
    """
    files = fetch_index(LAST_UPDATE, timeout=30.0)
    for f in files:
        print(f.stamp, f.stream, f.size_bytes, f.url)

    assert len(files) == 3, f"expected 3 records, got {len(files)}"
    assert {f.stream for f in files} == {"export", "mentions", "gkg"}

    print("STEP 2 OK")


def step_3(limit: int) -> None:
    """
    Download and read the most recent `limit` export files.
    """
    started = time.perf_counter()
    files = fetch_index() # the full master list
    print(f"master list: {len(files):,} records "
          f"in {time.perf_counter() - started:.0f}s")

    exports = sorted(
        (f for f in files if f.stream == "export"), key=lambda f: f.stamp
    )
    chosen = exports[-limit:]

    frames: list[pd.DataFrame] = []
    for f in chosen:
        try:
            path = download(f, CACHE_DIR)
        except (httpx.HTTPStatusError, ChecksumMismatch) as error:
            # One bad file must not stop the whole run
            print(f"SKIPPED {f.filename}: {error}")
            continue
        frame = read_events(path)
        print(f"{f.filename}: {len(frame):,} rows")
        frames.append(frame)

    if not frames:
        raise SystemExit("no files could be read")

    events = pd.concat(frames, ignore_index=True)

    print("\n--- column types ---")
    print(events.dtypes)

    print("\n--- EventRootCode values (expect 01 to 20, zeros kept) ---")
    print(sorted(events["EventRootCode"].dropna().unique()))

    missing = events["ActionGeo_CountryCode"].isna().mean()
    print(f"\nrows with no ActionGeo_CountryCode: {missing:.1%}")

    print("\n--- ten most common ActionGeo_CountryCode values ---")
    print(events["ActionGeo_CountryCode"].value_counts().head(10))

    print("\n--- Day values (event dates, not file dates) ---")
    print(events["Day"].value_counts().head(5))

    print(f"\nSTEP 3 DONE: {len(frames)}/{len(chosen)} files, "
          f"{len(events):,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--step", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()

    # Show warnings from download.py
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.step == 1:
        step_1()
    elif args.step == 2:
        step_2()
    else:
        step_3(args.limit)


if __name__ == "__main__":
    main()
