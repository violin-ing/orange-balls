"""
Calculates and saves the fixed limits used to make country indicators
comparable across countries and assessment dates.

Run from the repository root:

    python -m src.compute_bounds

    python -m src.compute_bounds --slow-years 2000 2024

    python -m src.compute_bounds --fast-from 2026-08-01 --fast-to 2026-10-31

This script sets the reference scale used by the scoring system.

It calculates limits for two types of indicators:

    1. Structural indicators

       Uses the natural 0-100 scale for WGI variables.
       For V-Dem variables, uses the lowest and highest country
       values found over the selected reference years.

    2. GDELT indicators

       Calculates the four GDELT scores for each country and
       assessment date in the selected period.

       Days without enough historical data are skipped.
       The upper limit is set at the 99th percentile to prevent
       unusually high values from stretching the scale too far.

The calculated limits are saved in the database, replacing
previous limits for the same indicators and settings.

Default periods (the reference periods decided in decision log 11.13):

    Structural data: 2000 to the latest year of structural data loaded.
    GDELT data:      All completely pulled days. A 180-day pull gives about
                     90 usable reference days, since each needs about 90
                     days of history before it. Fewer than
                     MIN_FAST_REFERENCE_DAYS prints a warning.

Options:

    --slow-years FIRST LAST
        Select the reference years for structural indicators.

    --fast-from DATE
        First date to include for GDELT indicators.

    --fast-to DATE
        Last date to include for GDELT indicators.

    --skip-fast
        Calculate structural limits only.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.reference.variables import with_role
from src.score.bounds import (
    fast_bounds,
    fast_reference_values,
    fast_variant,
    slow_bounds,
)
from src.score.fast import load_fast_inputs
from src.score.settings import Settings
from src.store.database import (
    DB_PATH,
    connect,
    load_pull_log,
    load_reference,
    save_bounds,
)

SLOW_REFERENCE_FIRST_YEAR = 2000
MIN_FAST_REFERENCE_DAYS = 60


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and save fixed normalisation bounds."
    )
    parser.add_argument("--slow-years", type=int, nargs=2,
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--fast-from", type=date.fromisoformat)
    parser.add_argument("--fast-to", type=date.fromisoformat)
    parser.add_argument("--skip-fast", action="store_true",
                        help="structural bounds only")
    args = parser.parse_args()
    settings = Settings().validate()

    connection = connect(DB_PATH)

    try:
        
        # === 1. Calculate structural indicator bounds ===

        reference = load_reference(connection)

        if reference.empty:
            raise SystemExit("no reference data: run src.load_reference")

        # Exclude validation variables when determining the default year range
        # Validation data such as GDP may extend beyond the scored variables
        scored = reference[~reference["variable"].isin(
            with_role("validation"))]
        first, last = args.slow_years or (
            SLOW_REFERENCE_FIRST_YEAR, int(scored["year"].max()))
        slow = slow_bounds(reference, first, last)

        print(f"structural bounds, years {first} to {last}:")

        print(slow[["criterion", "method", "lower", "upper"]]
              .round(3).to_string(index=False))

        save_bounds(connection, slow)

        if args.skip_fast:
            return


        # === 2. Identify the complete GDELT reference period ===

        log = load_pull_log(connection)
        complete = sorted(
            pd.to_datetime(log.loc[log["complete"] == 1, "date"]).dt.date
        )

        if not complete:
            raise SystemExit("no complete GDELT days: run src.pull_day")

        # Restrict the reference period to days with completed pulls
        start = args.fast_from or complete[0]
        end = args.fast_to or complete[-1]
        days = [d for d in complete if start <= d <= end]

        # Load the history needed to calculate scores for the selected days
        inputs = load_fast_inputs(connection, end, settings,
                                  first_date=start)


        # === 3. Calculate GDELT indicator bounds ===

        # Calculate country-level scores and exclude days wout enough history
        values, skipped = fast_reference_values(inputs, days, settings)
        fast = fast_bounds(values, settings)

        print(f"\nGDELT bounds, settings variant {fast_variant(settings)}, "
              f"{len(days) - len(skipped)} days used, {len(skipped)} "
              "skipped (not enough history before them):")

        print(fast[["criterion", "method", "lower", "upper"]]
              .round(4).to_string(index=False))

        # Warn when the percentile limits are based on a small reference sample
        used = len(days) - len(skipped)
        if used < MIN_FAST_REFERENCE_DAYS:
            print(f"\nWARNING: only {used} reference days (fewer than "
                  f"{MIN_FAST_REFERENCE_DAYS}). The percentile caps rest on "
                  "little data; pull more days (about 180 in total) and "
                  "run this again.")


        # === 4. Save GDELT indicator bounds ===

        save_bounds(connection, fast)

    finally:
        connection.close()

    print("\nBOUNDS SAVED")


if __name__ == "__main__":
    main()