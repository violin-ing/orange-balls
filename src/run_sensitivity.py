
"""
Runs the weights sensitivity analysis for one assessment date.

Run from the repository root:

    python -m src.run_sensitivity --date 2026-10-18
    python -m src.run_sensitivity --date 2026-10-18 --concentration 10
    python -m src.run_sensitivity --date 2026-10-18 --geometric
    python -m src.run_sensitivity --date 2026-10-18 --fifths
    python -m src.run_sensitivity --date 2026-10-18 --country BRA

What it does:
    1. Loads the data for the specified date once and scores every country
       using the default settings (weighted arithmetic mean, equal weights).
    2. Prints the band cut points, how many countries are in each band, how
       many have a band range spanning more than one band (band_uncertain),
       and the five lowest and five highest scores.
    3. With --country, prints that country's breakdown: each criterion's
       raw value and its value on the 0-1 scale, then its score and band.
    4. Draws random sets of weights (--draws, default 1,000, fixed seed)
       and re-scores every country using each set of weights.
    5. Calculates how often countries change their score band when the
       weights are varied, and lists the countries that change most often.
    6. Saves the country-level results to a CSV file.

The --geometric option repeats the analysis using the geometric aggregation
variant (with a min-max floor of 0.1). This is reported as a variant rather
than a separate method.

The --fifths option uses equal fifths (20th, 40th, 60th, 80th percentiles)
as the band cut points instead of the decided 5th, 20th, 50th and 80th: the
comparison run asked for in decision log 14.2.

Requires the normalisation bounds and band cut points (src.compute_bounds)
to be calculated first.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

from src.score.bands import check_descriptions, describe, load_descriptions
from src.score.drivers import (
    check_criterion_texts,
    compose,
    load_criterion_texts,
    select_drivers,
)
from src.score.run import ScoreResult, load_score_inputs, score_countries
from src.score.sensitivity import weights_monte_carlo
from src.score.settings import EQUAL_FIFTHS, Settings, geometric_variant
from src.store.database import DB_PATH, connect

OUTPUT_DIR = Path("data/processed")

# Number of countries to display
SHOW = 15

# Readable names for the criteria in the printed tables
CRITERION_NAMES = {
    "wgi_ge": "Government effectiveness",
    "wgi_pv": "Political stability",
    "v2mecenefm": "Media censorship",
    "v2meharjrn": "Harassment of journalists",
    "v2smgovshutcap": "Internet shutdown capacity",
    "v2smgovdom": "Government disinformation",
    "unrest": "Domestic unrest",
    "media_pressure": "Pressure on media",
    "intl_conflict": "International conflict",
    "intl_coop": "International cooperation",
}


def indent(text: str) -> str:
    return textwrap.indent(text, "   ")


def show_country(result: ScoreResult, iso3: str, settings: Settings) -> None:
    """
    Print one country's criteria (raw and on the 0-1 scale), score and band,
    then the text a planner would read: the band's description and one line
    for each of its weakest criteria (decision log 24.4).
    """
    print(f"\n2. Breakdown for {iso3}")

    if iso3 not in result.values.index:
        print("   no data for this country on this date")
        return

    raw = result.values.loc[iso3]
    scaled = result.norm.loc[iso3] if iso3 in result.norm.index else None

    table = pd.DataFrame({
        "criterion": [CRITERION_NAMES.get(c, c) for c in raw.index],
        "raw value": raw.round(3).to_numpy(),
        "0-1 scale (1 = best)": (scaled.round(2).to_numpy()
                                 if scaled is not None else None),
    })
    print(indent(table.to_string(index=False, na_rep="missing")))

    row = result.scores.loc[iso3] if iso3 in result.scores.index else None
    if row is None or pd.isna(row["score"]):
        print("\n   not scored: one or more criteria are missing")
        return

    low, high = int(row["band_low"]), int(row["band_high"])
    span = f"band {low} only" if low == high else f"bands {low}-{high}"
    print(f"\n   score {row['score']:.3f} +/- {1.96 * row['score_se']:.3f}"
          f"  ->  band {int(row['band'])} (range: {span})")

    # What a planner reads: band description + one line per weak criterion
    scored = result.norm.dropna()
    drivers = select_drivers(scored, settings)
    texts = check_criterion_texts(load_criterion_texts(),
                                  list(scored.columns))
    bands = check_descriptions(load_descriptions(), settings.bands)
    band_row = describe(result.scores.loc[[iso3]], bands).loc[iso3]

    named = drivers[drivers["iso3"] == iso3]
    if named.empty:
        why = "no criterion is notably below average"
    else:
        why = ", ".join(
            f"{CRITERION_NAMES.get(r.criterion, r.criterion)} "
            f"({r.value:.2f} against an average of {r.average:.2f})"
            for r in named.itertuples()
        )
    print(f"   main drivers: {why}")

    print("\n   what a planner reads (draft wording):")
    for line in compose(iso3, band_row, drivers, texts):
        print(indent(textwrap.fill(line, 72, subsequent_indent="  ")))


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

    # Optional flag to use equal fifths as the band cut points
    parser.add_argument(
        "--fifths",
        action="store_true",
        help="equal fifths as band cut points (comparison run, 14.2)"
    )

    # Optional country to show in detail, as an ISO3 code
    parser.add_argument(
        "--country",
        type=str.upper,
        help="also print one country's breakdown, e.g. BRA"
    )

    args = parser.parse_args()


    # === 2. Configure the scoring method ===

    settings = Settings().validate()

    # If requested, replace the default aggregation settings with the
    # geometric variant and validate the resulting configs
    if args.geometric:
        settings = geometric_variant(settings).validate()

    # If requested, cut the bands at equal fifths instead
    if args.fifths:
        settings = replace(settings, band_percentiles=EQUAL_FIFTHS).validate()


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


    # === 6. Print the scores, then the sensitivity summary ===

    s = mc.summary
    scores = result.scores.dropna(subset=["score"])

    method = "geometric" if args.geometric else "arithmetic"
    percentiles = "/".join(str(int(p)) for p in settings.band_percentiles)

    print(f"\n=== Scores and weights test for {args.date} ===")
    print(f"   {method} mean, equal weights; bands cut at percentiles "
          f"{percentiles} of the reference period")

    # 6.1 Scores and bands
    print(f"\n1. Scores and bands ({len(scores)} countries scored)")
    print("   cut points        "
          + "  ".join(f"{float(c):.3f}" for c in result.cuts))

    # How many countries are in each band, and how many are not pinned to
    # one band by their data (band range spans more than one band)
    bands = scores["band"].value_counts().reindex(
        range(1, settings.bands + 1), fill_value=0)
    print("   countries         "
          + " | ".join(f"band {b}: {n}" for b, n in bands.items()))
    print(f"   band uncertain    {int(scores['band_uncertain'].sum())} "
          "countries (their range spans more than one band)")

    ranked = scores.sort_values("score")
    shown = pd.DataFrame({
        "country": ranked.index,
        "score": ranked["score"].round(3).to_numpy(),
        "+/-": (1.96 * ranked["score_se"]).round(3).to_numpy(),
        "band": ranked["band"].astype(int).to_numpy(),
    })
    print("\n   five lowest scores:")
    print(indent(shown.head(5).to_string(index=False)))
    print("\n   five highest scores:")
    print(indent(shown.tail(5).iloc[::-1].to_string(index=False)))

    # 6.2 One country in detail
    if args.country:
        show_country(result, args.country, settings)

    # 6.3 Weights sensitivity
    step = 3 if args.country else 2
    print(f"\n{step}. Weights test: {s['draws']:,} random weight sets "
          f"(concentration {s['concentration']:g}, seed {s['seed']})")

    # Average share of random weight sets in which a country's band differs
    # from its band under equal weights
    print(f"   average share of sets in which a country changes band: "
          f"{s['mean_changed_share']:.1%}")

    for key, value in s.items():

        # Only process summary stats relating to countries changing bands
        if key.startswith("countries_changing_"):

            # Extract the percentage threshold from the summary key
            share = key.removeprefix("countries_changing_").removesuffix(
                "pc"
            )

            # Number of countries at or above this threshold
            print(f"   countries changing band in at least {share}% of "
                  f"sets: {value}")

    # The countries whose bands change most often
    top = mc.countries.head(SHOW)
    shown = pd.DataFrame({
        "country": top.index,
        "band": top["base_band"].astype(int).to_numpy(),
        "changes in": [f"{x:.1%}" for x in top["changed_share"]],
        "lowest": top["lowest_band"].astype(int).to_numpy(),
        "highest": top["highest_band"].astype(int).to_numpy(),
    })
    print(f"\n   the {SHOW} countries that change band most often:")
    print(indent(shown.to_string(index=False)))


    # === 7. Save the country-level sensitivity results ===

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_geometric" if args.geometric else "") \
        + ("_fifths" if args.fifths else "")
    path = OUTPUT_DIR / f"sensitivity_weights_{args.date}{suffix}.csv"
    mc.countries.to_csv(path, index_label="iso3")

    print(f"\nDone: full results saved to {path}")

if __name__ == "__main__":
    main()
