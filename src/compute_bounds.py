"""
Calculates and saves the fixed limits used to make country indicators
comparable across countries and assessment dates.

Run from the repository root:

    python -m src.compute_bounds

    python -m src.compute_bounds --slow-years 2000 2024

    python -m src.compute_bounds --fast-from 2026-08-01 --fast-to 2026-10-31

This script sets the reference scale used by the scoring system.
It should be run once the reference data has been loaded, and only
run again when there is a deliberate decision to change that scale.

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

Default periods:

    Structural data: 2000 to the latest year available.
    GDELT data:      All completely pulled days.

Options:

    --slow-years FIRST LAST
        Select the reference years for structural indicators.

    --fast-from DATE
        First date to include for GDELT indicators.

    --fast-to DATE
        Last date to include for GDELT indicators.

    --skip-fast
        Calculate structural limits only.

If different GDELT scoring settings are used for sensitivity
testing, separate limits must be calculated for each set of
settings. These are identified by a variant code.

Important:
Changing the reference limits can change every country's score.
Keep the reference periods fixed and record them in the project
methodology so that results remain comparable over time.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

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
        reference = load_reference(connection)

        if reference.empty:
            raise SystemExit("no reference data: run src.load_reference")
        
        first, last = args.slow_years or (2000, int(reference["year"].max()))
        slow = slow_bounds(reference, first, last)

        print(f"structural bounds, years {first} to {last}:")

        print(slow[["criterion", "method", "lower", "upper"]]
              .round(3).to_string(index=False))
        
        save_bounds(connection, slow)

        if args.skip_fast:
            return
        
        log = load_pull_log(connection)
        complete = sorted(
            pd.to_datetime(log.loc[log["complete"] == 1, "date"]).dt.date
        )

        if not complete:
            raise SystemExit("no complete GDELT days: run src.pull_day")
        
        start = args.fast_from or complete[0]
        end = args.fast_to or complete[-1]
        days = [d for d in complete if start <= d <= end]

        inputs = load_fast_inputs(connection, end, settings,
                                  first_date=start)
        
        values, skipped = fast_reference_values(inputs, days, settings)
        fast = fast_bounds(values, settings)

        print(f"\nGDELT bounds, settings variant {fast_variant(settings)}, "
              f"{len(days) - len(skipped)} days used, {len(skipped)} "
              "skipped (not enough history before them):")
        
        print(fast[["criterion", "method", "lower", "upper"]]
              .round(4).to_string(index=False))
        
        save_bounds(connection, fast)

    finally:
        connection.close()
        
    print("\nBOUNDS SAVED")


if __name__ == "__main__":
    main()
