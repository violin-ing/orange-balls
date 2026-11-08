"""
Runs the weights sensitivity analysis for one assessment date. Run from
the repository root:

    python -m src.run_sensitivity --date 2026-10-18
    python -m src.run_sensitivity --date 2026-10-18 --concentration 10
    python -m src.run_sensitivity --date 2026-10-18 --geometric

What it does:
    1. Loads the data for the date once and scores every country with the
       default settings (weighted arithmetic mean, equal weights).
    2. Draws --draws random sets of weights (default 1,000, fixed seed)
       and re-scores every country with each (src/score/sensitivity.py).
    3. Prints how often countries change band, and the countries that
       change most.
    4. Saves the per-country table to
       data/processed/sensitivity_weights_<date>[_geometric].csv
       (country-level shares only).

--geometric repeats the analysis for the geometric aggregation variant
(min-max floor 0.1), which is reported as a variant, not the method.
Needs the normalisation bounds (src.compute_bounds) to be computed first.
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
SHOW = 15


def main() -> None:
    parser = argparse.ArgumentParser(
        description="How often do countries change band under random "
                    "weights?"
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--concentration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometric", action="store_true",
                        help="the geometric aggregation variant")
    args = parser.parse_args()

    settings = Settings().validate()
    if args.geometric:
        settings = geometric_variant(settings).validate()

    connection = connect(DB_PATH)
    try:
        inputs = load_score_inputs(connection, args.date, settings)
    finally:
        connection.close()

    result = score_countries(inputs, args.date, settings)
    mc = weights_monte_carlo(result, settings, draws=args.draws,
                             concentration=args.concentration,
                             seed=args.seed)

    s = mc.summary
    method = "geometric" if args.geometric else "arithmetic"
    print(f"{args.date}, {method} mean, {s['countries']} countries, "
          f"{s['draws']} weight draws (concentration "
          f"{s['concentration']}, seed {s['seed']})")
    print(f"  average share of draws in which a country changes band: "
          f"{s['mean_changed_share']:.1%}")
    for key, value in s.items():
        if key.startswith("countries_changing_"):
            share = key.removeprefix("countries_changing_").removesuffix(
                "pc")
            print(f"  countries changing band in at least {share}% of "
                  f"draws: {value}")

    print(f"\nthe {SHOW} countries that change band most often:")
    print(mc.countries.head(SHOW).round(3).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_geometric" if args.geometric else ""
    path = OUTPUT_DIR / f"sensitivity_weights_{args.date}{suffix}.csv"
    mc.countries.to_csv(path, index_label="iso3")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
