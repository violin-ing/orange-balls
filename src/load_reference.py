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
       docs/VERSIONS.md.
    2. Loads the variables listed in src/reference/variables.py.
    3. Checks country codes against the ISO3 list.
    4. Replaces the source's existing database records.
    5. Reports country coverage by year and reference-data coverage for GDELT 
       countries in the selected year.
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

VDEM_DIR = Path("data/reference/vdem")
WGI_DIR = Path("data/reference/wgi")
WB_DIR = Path("data/reference/worldbank")
VERSIONS = Path("docs/VERSIONS.md")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def only_file(folder: Path, pattern: str) -> Path:
    """
    Find the only file matching pattern, or stop with an error.
    """
    found = sorted(folder.glob(pattern))
    if len(found) != 1:
        raise SystemExit(
            f"expected exactly one {pattern} file in {folder}, found "
            f"{[p.name for p in found]}. Use --vdem / --wgi to choose."
        )
    
    return found[0]


def check_recorded(path: Path, checksum: str) -> None:
    """
    Check whether the file's checksum is recorded in VERSIONS.md.
    """
    if not VERSIONS.exists():
        print(f"  WARNING: {VERSIONS} not found, cannot check the version")
        return
    
    if checksum not in VERSIONS.read_text(encoding="utf-8"):
        print(f"  WARNING: {path.name}'s checksum is not in {VERSIONS}. "
              "Record it there before using results from this file.")
    else:
        print(f"  checksum matches {VERSIONS}")


def coverage_by_year(table: pd.DataFrame, years: range) -> pd.DataFrame:
    """
    Count countries with data for each variable and year.
    """
    recent = table[table["year"].isin(list(years))]

    return recent.pivot_table(
        index="variable", columns="year", values="iso3", aggfunc="nunique"
    ).fillna(0).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load V-Dem and WGI into the database."
    )
    parser.add_argument("--vdem", type=Path, help="path to the V-Dem CSV")
    parser.add_argument("--wgi", type=Path, help="path to the WGI Excel")
    parser.add_argument("--wb", type=Path,
                        help="path to the World Bank API_*.csv (optional)")
    parser.add_argument(
        "--year", type=int, default=2025,
        help="reference year for the GDELT coverage report (default 2025)",
    )
    args = parser.parse_args()

    files = {
        "vdem": args.vdem or only_file(VDEM_DIR, "*.csv"),
        "wgi": args.wgi or only_file(WGI_DIR, "*.xlsx"),
    }

    wb = args.wb or next(iter(sorted(WB_DIR.glob("API_*.csv"))), None)
    if wb is not None:
        files["wb"] = wb
    else:
        print(f"no World Bank file in {WB_DIR}; GDP per capita not loaded")

    loaders = {"vdem": load_vdem, "wgi": load_wgi, "wb": load_worldbank}

    tables: dict[str, pd.DataFrame] = {}
    for source, path in files.items():
        print(f"{source}: {path}")

        checksum = sha256(path)
        print(f"  sha256 {checksum}")

        check_recorded(path, checksum)
        
        try:
            tables[source] = loaders[source](path)
        except UnknownReferenceCode as error:
            raise SystemExit(f"\nSTOPPED: {error}") from None
        
        tables[source].attrs["sha256"] = checksum

        print(f"  {len(tables[source]):,} rows read")

    connection = connect(DB_PATH)
    try:
        for source, table in tables.items():
            written = save_reference(
                connection, table, source, files[source].name,
                table.attrs["sha256"],
            )
            print(f"saved {written:,} {source} rows to {DB_PATH}")

        reference = pd.concat(tables.values(), ignore_index=True)
        print("\n--- countries with a value, 2019 to 2025 ---")
        print(coverage_by_year(reference, range(2019, 2026)).to_string())

        log = load_pull_log(connection)

        if log.empty:
            print("\nno GDELT days stored yet, so no join report")
            return
        
        last_day = pd.Timestamp(log["date"].max()).date()
        days = load_country_day(connection, start=last_day, end=last_day)

    finally:
        connection.close()

    joined = join_reference(days, for_year(reference, args.year))
    coverage = joined.attrs["reference_coverage"]

    print(f"\n--- join: GDELT {last_day} with reference year "
          f"{args.year} ---")
    print(coverage)

    share = coverage["events_without_reference"] / max(
        coverage["events_total"], 1
    )

    print(f"located events in countries missing some reference data: "
          f"{share:.1%}")

    missing = joined[joined["wgi_ge"].isna() | joined["v2mecenefm"].isna()]
    top = missing.sort_values("n_events", ascending=False).head(10)
    
    print("\nlargest GDELT countries missing some reference data:")
    print(top[["iso3", "n_events", "wgi_ge", "v2mecenefm",
               "v2smgovshutcap"]].to_string(index=False))
    print("\nREFERENCE LOAD OK")


if __name__ == "__main__":
    main()
