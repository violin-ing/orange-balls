
"""
Runs the weights sensitivity analysis for one assessment date.

Run from the repository root:

    python -m src.run_sensitivity --date 2026-10-18
    python -m src.run_sensitivity --date 2026-10-18 --concentration 10
    python -m src.run_sensitivity --date 2026-10-18 --geometric

What it does:
    1. Loads the data for the specified date once and scores every country
       using the default settings (weighted arithmetic mean, equal weights).
    2. Draws random sets of weights (--draws, default 1,000, fixed seed)
       and re-scores every country using each set of weights.
    3. Calculates how often countries change their score band when the
       weights are varied.
    4. Identifies the countries whose bands are most sensitive to changes
       in weights.
    5. Saves the country-level results to a CSV file.

The --geometric option repeats the analysis using the geometric aggregation
variant (with a min-max floor of 0.1). This is reported as a variant rather
than a separate method.

Requires the normalisation bounds (src.compute_bounds) to be calculated first.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.score.run import load_score_inputs, score_countries
from src.score.sensitivity import weights_monte_carlo
from src.score.settings import Settings, geometric_variant
from src.store.database import DB_PATH, connect

OUTPUT_DIR = Path("data/processed")

# Number of countries to display
SHOW = 15


def main() -> None:

    # === 1. Set up command-line args ===

    parser = argparse.ArgumentParser(
        description="How often do countries change band under random weights?"
    )

    # Assessment date: required argument, parsed into a Python date object
    parser.add_argument("--date", type=date.fromisoformat, required=True)

    # Number of random weight combinations to generate (default: 1,000)
    parser.add_argument("--draws", type=int, default=1000)

    # Controls how concentrated the randomly generated weights are
    # (lower = more variation; higher = more equal weights)
    parser.add_argument("--concentration", type=float, default=1.0)

    # Random seed to ensure that the same random weights are generated
    parser.add_argument("--seed", type=int, default=0)

    # Optional flag to run the analysis using geometric aggregation
    parser.add_argument(
        "--geometric",
        action="store_true",
        help="the geometric aggregation variant"
    )

    args = parser.parse_args()


    # === 2. Configure the scoring method ===

    settings = Settings().validate()

    # If requested, replace the default aggregation settings with the
    # geometric variant and validate the resulting configs
    if args.geometric:
        settings = geometric_variant(settings).validate()


    # === 3. Load the assessment data ===

    connection = connect(DB_PATH)

    try:
        # Load all scoring inputs for the specified assessment date
        inputs = load_score_inputs(connection, args.date, settings)
    finally:
        connection.close()


    # === 4. Calculate the baseline country scores ===

    # Score every country using the selected settings
    result = score_countries(inputs, args.date, settings)


    # === 5. Run the Monte Carlo weights sensitivity analysis ===

    # Generate random combinations of indicator weights and re-score
    # every country using each combination.
    #
    # The function compares the resulting country score bands with
    # the baseline bands to establish how sensitive each country's
    # classification is to changes in indicator weights.
    #
    # draws         = number of random weight combinations
    # concentration = controls the distribution of the random weights
    # seed          = ensures reproducibility of the random draws
    mc = weights_monte_carlo(
        result,
        settings,
        draws=args.draws,
        concentration=args.concentration,
        seed=args.seed
    )


    # === 6. Print a summary of the analysis ===

    s = mc.summary

    # Identify aggregation method used
    method = "geometric" if args.geometric else "arithmetic"

    print(
        f"{args.date}, {method} mean, {s['countries']} countries, "
        f"{s['draws']} weight draws (concentration "
        f"{s['concentration']}, seed {s['seed']})"
    )

    # Avg ppt of random weight combinations where countrues change their bands
    # (relative to baseline)
    print(
        f"  average share of draws in which a country changes band: "
        f"{s['mean_changed_share']:.1%}"
    )


    for key, value in s.items():

        # Only process summary stats relating to countries changing bands
        if key.startswith("countries_changing_"):

            # Extract the percentage threshold from the summary key
            share = key.removeprefix("countries_changing_").removesuffix(
                "pc"
            )

            # Number of countries exceeding this threshold
            print(
                f"  countries changing band in at least {share}% of "
                f"draws: {value}"
            )

    # Display the 15 countries whose score bands change most frequently (3dp)
    print(f"\nthe {SHOW} countries that change band most often:")
    print(mc.countries.head(SHOW).round(3).to_string())


    # === 7. Save the country-level sensitivity results ===

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_geometric" if args.geometric else ""
    path = OUTPUT_DIR / f"sensitivity_weights_{args.date}{suffix}.csv"
    mc.countries.to_csv(path, index_label="iso3")

    print(f"\nsaved {path}")

if __name__ == "__main__":
    main()
