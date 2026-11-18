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

    3. Band cut points (decision log 14.1 and 14.2)

       Scores every country on every usable reference day with the
       limits above, then takes the 5th, 20th, 50th and 80th
       percentiles of all those scores as the cut points between bands
       1 to 5. Also stores equal fifths (20th, 40th, 60th, 80th) for the
       comparison run, and the same two sets for the geometric variant.
       Prints how many countries each split puts in each band on the
       last reference day.

The calculated limits and cut points are saved in the database, replacing
previous ones for the same indicators and settings. They are then frozen:
run this again only on purpose, since new values change every band.

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
        Calculate structural limits only (no GDELT limits, no cut points).
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import replace
from datetime import date

import pandas as pd

from src.reference.variables import with_role
from src.score.bands import band_of, reference_cut_rows
from src.score.bounds import (
    fast_bounds,
    fast_reference_values,
    fast_variant,
    slow_bounds,
)
from src.score.fast import load_fast_inputs
from src.score.run import ScoreInputs, reference_scores
from src.score.settings import EQUAL_FIFTHS, Settings, geometric_variant
from src.store.database import (
    DB_PATH,
    connect,
    load_bounds,
    load_pull_log,
    load_reference,
    save_band_cuts,
    save_bounds,
)

SLOW_REFERENCE_FIRST_YEAR = 2000
MIN_FAST_REFERENCE_DAYS = 60


def show_bounds(bounds: pd.DataFrame, digits: int) -> None:
    """
    Print the lower and upper limit of each criterion, indented.
    """
    table = bounds[["criterion", "method", "lower", "upper"]].round(digits)
    print(textwrap.indent(table.to_string(index=False), "   "))


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

    print("\n=== Fixed scales and band cut points ===")

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

        print(f"\n1. Structural criteria: limits over the years {first} "
              f"to {last}")
        show_bounds(slow, 3)

        save_bounds(connection, slow)

        if args.skip_fast:
            print("\nDone: structural limits saved (GDELT skipped).")
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

        used = len(days) - len(skipped)

        print(f"\n2. GDELT criteria: limits over {used} reference days "
              f"({days[0]} to {days[-1]})")
        print(f"   {len(skipped)} days skipped (not enough history before "
              f"them); settings variant {fast_variant(settings)}")
        show_bounds(fast, 4)

        # Warn when the percentile limits are based on a small reference sample
        if used < MIN_FAST_REFERENCE_DAYS:
            print(f"\n   WARNING: only {used} reference days (fewer than "
                  f"{MIN_FAST_REFERENCE_DAYS}). The percentile caps rest on "
                  "little data; pull more days (about 180 in total) and "
                  "run this again.")


        # === 4. Save GDELT indicator bounds ===

        save_bounds(connection, fast)


        # === 5. Band cut points from reference-period scores ===

        # Everything scoring needs for the whole period, loaded once
        score_inputs = ScoreInputs(
            fast=inputs,
            reference=reference,
            bounds=load_bounds(connection),
        )

        print("\n3. Band cut points (from every country's score on every "
              "reference day)")

        for label, base in (("arithmetic mean (the method)", settings),
                            ("geometric mean (variant)",
                             geometric_variant(settings))):
            scores, _ = reference_scores(score_inputs, days, base)
            last_day = scores[scores["date"] == scores["date"].max()]

            print(f"\n   {label}: {scores['date'].nunique()} days, "
                  f"{len(scores):,} country-day scores")

            for name, split in (("decided split", base.band_percentiles),
                                ("equal fifths", EQUAL_FIFTHS)):
                rows = reference_cut_rows(
                    scores, replace(base, band_percentiles=split))
                save_band_cuts(connection, rows)

                cuts = rows["value"].to_numpy()
                counts = pd.Series(band_of(last_day["score"], cuts)) \
                    .value_counts().reindex(range(1, len(cuts) + 2),
                                            fill_value=0)

                percentiles = "/".join(str(int(p)) for p in split)
                print(f"     {name} (percentiles {percentiles})")
                print("       cut points       "
                      + "  ".join(f"{float(c):.3f}" for c in cuts))
                print(f"       countries per band on "
                      f"{last_day['date'].max()}: "
                      + " | ".join(f"band {b}: {n}"
                                   for b, n in counts.items()))

    finally:
        connection.close()

    print("\nDone: limits and cut points saved. They stay fixed until "
          "this is run again on purpose.")


if __name__ == "__main__":
    main()