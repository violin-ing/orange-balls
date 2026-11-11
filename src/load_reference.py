
"""
Load V-Dem and WGI data into the database.

Run from the project root:

    python3 -m src.load_reference
    python3 -m src.load_reference --year 2024

By default, reads one CSV from data/reference/vdem/, one Excel file from
data/reference/wgi/ and, if available, the World Bank GDP per capita files
from data/reference/worldbank/.

Use --vdem, --wgi and --wb to specify different file paths.

The script:
    1. Calculates each file's SHA-256 checksum and checks it against 
        docs/VERSIONS.md
    2. Loads the variables listed in src/reference/variables.py
    3. Checks country codes against the ISO3 list
    4. Replaces the source's existing database records
    5. Reports country coverage by year and reference-data coverage for GDELT
       countries in the selected year
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.reference.common import UnknownReferenceCode
from src.reference.select import for_year, join_reference
from src.reference.vdem import load_vdem
from src.reference.wgi import load_wgi
from src.reference.worldbank import load_worldbank
from src.store.database import (
    DB_PATH,
    connect,
    load_country_day,
    load_pull_log,
    save_reference,
)

# === Settings ===

VDEM_DIR = Path("data/reference/vdem")
WGI_DIR = Path("data/reference/wgi")
WB_DIR = Path("data/reference/worldbank")
VERSIONS = Path("docs/VERSIONS.md")


# === 1. Calculate file checksum ===

def sha256(path: Path) -> str:
    """
    Calculate the SHA-256 checksum of a file

    Used to identify the exact version of each reference dataset
    """

    digest = hashlib.sha256()

    # Read the file in 1 MB blocks to avoid loading the entire file into memory
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


# === 2. Identify the reference data files ===

def only_file(folder: Path, pattern: str) -> Path:
    """
    Find the only file matching the specified pattern

    Stops with an error if zero or multiple matching files are found
    """

    found = sorted(folder.glob(pattern))

    if len(found) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} file in {folder}, found "
            f"{[p.name for p in found]}. Use --vdem / --wgi to choose."
        )

    return found[0]


# === 3. Check recorded file versions ===

def check_recorded(path: Path, checksum: str) -> None:
    """
    Check whether the file's checksum is recorded in VERSIONS.md

    Provides a warning if the checksum has not been documented
    """

    if not VERSIONS.exists():
        print(f"  WARNING: {VERSIONS} not found, cannot check the version")
        return

    # Check whether the calculated checksum appears in the version record
    if checksum not in VERSIONS.read_text(encoding="utf-8"):
        print(
            f"  WARNING: {path.name}'s checksum is not in {VERSIONS}. "
            "Record it there before using results from this file."
        )
    else:
        print(f"  checksum matches {VERSIONS}")


# === 4. Calculate reference-data coverage by year ===

def coverage_by_year(table: pd.DataFrame, years: range) -> pd.DataFrame:
    """
    Count the number of countries with data for each variable and year

    Returns a table with variables as rows and years as columns
    """

    # Restrict the data to the requested years
    recent = table[table["year"].isin(list(years))]

    # Count unique countries for each variable and year
    return recent.pivot_table(
        index="variable", columns="year", values="iso3", aggfunc="nunique"
    ).fillna(0).astype(int)


# === 5. Main function ===

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load V-Dem and WGI into the database."
    )

    # Allow alternative input files to be specified from the command line
    parser.add_argument("--vdem", type=Path, help="path to the V-Dem CSV")
    parser.add_argument("--wgi", type=Path, help="path to the WGI Excel")
    parser.add_argument(
        "--wb",
        type=Path,
        help="path to the World Bank API_*.csv (optional)"
    )

    # Select the reference year used when checking coverage against GDELT data
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="reference year for the GDELT coverage report (default 2025)",
    )

    args = parser.parse_args()

    # === 5.1 Identify input files ===

    # Use explicitly provided paths or locate the default files
    files = {
        "vdem": args.vdem or only_file(VDEM_DIR, "*.csv"),
        "wgi": args.wgi or only_file(WGI_DIR, "*.xlsx"),
    }

    # World Bank data is optional, so continue if no file is available
    wb = args.wb or next(iter(sorted(WB_DIR.glob("API_*.csv"))), None)

    if wb is not None:
        files["wb"] = wb
    else:
        print(f"no World Bank file in {WB_DIR}; GDP per capita not loaded")

    # Match each reference source to its corresponding loading function
    loaders = {
        "vdem": load_vdem,
        "wgi": load_wgi,
        "wb": load_worldbank,
    }

    # === 5.2 Load and validate reference data ===

    # Store the processed tables separately by data source
    tables: dict[str, pd.DataFrame] = {}

    for source, path in files.items():
        print(f"{source}: {path}")

        # Calculate a checksum to identify the exact input file version
        checksum = sha256(path)
        print(f"  sha256 {checksum}")

        # Check whether the input file version is documented
        check_recorded(path, checksum)

        try:
            # Load and validate the source using its corresponding function
            # Includes checking country codes against recognised ISO3 codes
            tables[source] = loaders[source](path)

        except UnknownReferenceCode as error:
            raise SystemExit(f"\nSTOPPED: {error}") from None

        # Attach the checksum to the table so it can be recorded in the DB
        tables[source].attrs["sha256"] = checksum

        print(f"  {len(tables[source]):,} rows read")

    # === 5.3 Save reference data to the database ===

    connection = connect(DB_PATH)

    try:
        for source, table in tables.items():

            # Replace the existing records for this source and save the
            # filename and checksum to maintain a record of the data version
            written = save_reference(
                connection,
                table,
                source,
                files[source].name,
                table.attrs["sha256"],
            )

            print(f"saved {written:,} {source} rows to {DB_PATH}")

        # Combine all reference sources into one table for the coverage report
        reference = pd.concat(tables.values(), ignore_index=True)

        # === 5.4 Report reference-data coverage ===

        # Show the number of countries with data for each variable across the
        # years 2019 to 2025
        print("\n--- countries with a value, 2019 to 2025 ---")
        print(coverage_by_year(reference, range(2019, 2026)).to_string())

        # Retrieve the GDELT pull log to identify the latest available date
        log = load_pull_log(connection)

        if log.empty:
            print("\nno GDELT days stored yet, so no join report")
            return

        # Select the latest date with GDELT data available
        last_day = pd.Timestamp(log["date"].max()).date()

        # Load country-level GDELT data for that date only
        days = load_country_day(
            connection,
            start=last_day,
            end=last_day,
        )

    finally:
        connection.close()

    # === 5.5 Join GDELT data with reference data ===

    # Select the reference observations corresponding to the requested year
    # and join them to the country-level GDELT data
    joined = join_reference(days, for_year(reference, args.year))

    # Retrieve statistics describing how much reference data is available
    coverage = joined.attrs["reference_coverage"]

    print(
        f"\n--- join: GDELT {last_day} with reference year "
        f"{args.year} ---"
    )
    print(coverage)

    # Calculate the proportion of located GDELT events occurring in countries
    # where some reference data is missing
    share = coverage["events_without_reference"] / max(
        coverage["events_total"], 1
    )

    print(
        f"located events in countries missing some reference data: "
        f"{share:.1%}"
    )

    # Identify countries missing either WGI government effectiveness or the 
    # V-Dem executive corruption measure
    missing = joined[
        joined["wgi_ge"].isna() | joined["v2mecenefm"].isna()
    ]

    # Rank countries by event frequency and retain the ten largest
    top = missing.sort_values("n_events", ascending=False).head(10)

    print("\nlargest GDELT countries missing some reference data:")
    print(
        top[
            [
                "iso3",
                "n_events",
                "wgi_ge",
                "v2mecenefm",
                "v2smgovshutcap",
            ]
        ].to_string(index=False)
    )

    print("\nREFERENCE LOAD OK")


if __name__ == "__main__":
    main()
