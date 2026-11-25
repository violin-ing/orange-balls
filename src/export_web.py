"""
Writes the scores for the web front end (src/web/) as JSON files.

Run from the repository root:

    python -m src.export_web                       # the latest complete day
    python -m src.export_web --date 2026-10-31
    python -m src.export_web --from 2026-10-01 --to 2026-10-31   # for Replay

For each date it writes two files to data/web/<date>/:

    international.json   the International View: every scored country's
                         band, score, band range, its eleven criteria (0-1,
                         1 = best), the criteria that pull it down most, and
                         the band and criterion texts
    bilateral.json       the Bilateral View: for every country, each other
                         country it has had contact with in the last 90 days,
                         with the relation band and score (src/score/pairs.py)

and updates data/web/index.json, the list of exported dates, which the front
end reads to fill its date menu (the latest date is the Live view).

Only country-level totals are written, never articles (decision log 1.7).
data/ is not committed to git.

To look at the result, serve the repository root and open the front end:

    python -m http.server 8000
    then open http://localhost:8000/src/web/ in a browser

Needs the bounds and band cut points first (python -m src.compute_bounds).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.score.bands import (
    MissingBandCuts,
    check_descriptions,
    load_descriptions,
)
from src.score.combine import MissingBounds
from src.score.drivers import (
    check_criterion_texts,
    load_criterion_texts,
    select_drivers,
)
from src.score.fast import InsufficientHistory
from src.score.names import CRITERION_GROUPS, GROUPS, name
from src.score.pairs import (
    load_relation_bands,
    pair_scores,
    stored_pair_cuts,
)
from src.score.run import ScoreResult, load_score_inputs, score_countries
from src.score.settings import Settings
from src.store.database import (
    DB_PATH,
    connect,
    load_band_cuts,
    load_pair_day,
    load_pull_log,
)

WEB_DIR = Path("data/web")


def _round(x: float, digits: int = 3) -> float | None:
    """
    A plain rounded number for JSON, or None for a missing value.
    """
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), digits)


def _sentence(text: str) -> str:
    """
    One action from a band description, trimmed, with a capital letter.
    """
    text = text.strip()
    return text[:1].upper() + text[1:]


def international_json(
        day: date,
        result: ScoreResult,
        settings: Settings,
    ) -> dict:
    """
    The International View's data for one date, as a plain dictionary.
    """
    scores = result.scores.dropna(subset=["score"])
    criteria = list(result.values.columns)
    scored = result.norm.dropna()

    texts = check_criterion_texts(load_criterion_texts(), criteria)
    internal = texts[texts["audience"] == "internal"].set_index("criterion")
    bands = check_descriptions(load_descriptions(), settings.bands)
    internal_bands = bands[bands["audience"] == "internal"]
    drivers = select_drivers(scored, settings)
    named = drivers.groupby("iso3")["criterion"].apply(list).to_dict()

    countries = {}
    for iso3, row in scores.iterrows():
        countries[iso3] = {
            "score": _round(row["score"]),
            "se": _round(row["score_se"], 4),
            "band": int(row["band"]),
            "band_low": int(row["band_low"]),
            "band_high": int(row["band_high"]),
            "values": {c: _round(result.norm.at[iso3, c], 3)
                       for c in criteria},
            "raw": {c: _round(result.values.at[iso3, c], 4)
                    for c in criteria},
            "drivers": named.get(iso3, []),
        }

    # Countries with some data but not every criterion: listed so the
    # front end can say what is missing rather than just "no data"
    partial = {}
    for iso3, row in result.values.iterrows():
        if iso3 in countries:
            continue
        missing = [c for c in criteria if pd.isna(row[c])]
        if len(missing) < len(criteria):
            partial[iso3] = missing

    return {
        "view": "international",
        "date": day.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_year": result.reference_year,
        "score_variant": result.score_variant,
        "cuts": [_round(c, 4) for c in result.cuts],
        "bands": [
            {
                "band": int(r.band), "label": r.label,
                "actions": [_sentence(a) for a in r.action.split(";")],
                "status": r.status,
            }
            for r in internal_bands.itertuples()
        ],
        "groups": [{"key": k, "name": v} for k, v in GROUPS.items()],
        "criteria": [
            {
                "key": c, "name": name(c),
                "group": CRITERION_GROUPS.get(c, "structural"),
                "label": internal.at[c, "label"],
                "text": internal.at[c, "text"],
            }
            for c in criteria
        ],
        "countries": countries,
        "partial": partial,
    }


def bilateral_json(
        day: date,
        pairs: pd.DataFrame,
        cuts: np.ndarray | None,
        settings: Settings,
    ) -> dict:
    """
    The Bilateral View's data for one date: for every country, each other
    country it had contact with, as [band, score, events, conflictual
    events, events it started, events the other started].
    """
    labels = load_relation_bands()
    partners: dict[str, dict[str, list]] = {}

    for r in pairs.itertuples():
        partners.setdefault(r.a, {})[r.b] = [
            int(r.band), _round(r.value), int(r.n), int(r.conflict),
            int(r.a_to_b), int(r.b_to_a),
        ]
        partners.setdefault(r.b, {})[r.a] = [
            int(r.band), _round(r.value), int(r.n), int(r.conflict),
            int(r.b_to_a), int(r.a_to_b),
        ]

    return {
        "view": "bilateral",
        "date": day.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_days": settings.baseline_days,
        "min_events": settings.prior_strength,
        "cuts": None if cuts is None else [_round(c, 4) for c in cuts],
        "fields": ["band", "score", "events", "conflict", "out", "in"],
        "bands": [
            {"band": int(r.band), "label": r.label,
             "description": r.description, "status": r.status}
            for r in labels.itertuples()
        ],
        "partners": partners,
    }


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")),
                    encoding="utf-8")


def update_index(web_dir: Path) -> list[str]:
    """
    Rewrite index.json from the date folders present; returns the dates.
    """
    dates = sorted(
        p.name for p in web_dir.iterdir()
        if p.is_dir() and (p / "international.json").exists()
    ) if web_dir.exists() else []

    _write(web_dir / "index.json", {
        "dates": dates,
        "latest": dates[-1] if dates else None,
    })

    return dates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write scores as JSON for the web front end"
    )
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--from", dest="start", type=date.fromisoformat)
    parser.add_argument("--to", dest="end", type=date.fromisoformat)
    args = parser.parse_args()

    settings = Settings().validate()
    connection = connect(DB_PATH)

    try:
        log = load_pull_log(connection)
        complete = sorted(
            pd.to_datetime(log.loc[log["complete"] == 1, "date"]).dt.date
        )
        if not complete:
            raise SystemExit("no complete GDELT days: run src.pull_day")

        if args.date:
            days = [args.date]
        elif args.start:
            end = args.end or complete[-1]
            days = [d for d in complete if args.start <= d <= end]
        else:
            days = [complete[-1]]

        pair_cuts = stored_pair_cuts(load_band_cuts(connection), settings)
        if pair_cuts is None:
            print("note: no relation band cut points stored, so pairs get "
                  "no band; run python -m src.compute_bounds")

        print(f"\n=== Export for the web front end: {len(days)} date(s) ===")

        for day in days:
            try:
                inputs = load_score_inputs(connection, day, settings)
                result = score_countries(inputs, day, settings)
            except (InsufficientHistory, MissingBandCuts,
                    MissingBounds) as error:
                print(f"   {day}: skipped ({error})")
                continue

            folder = WEB_DIR / day.isoformat()
            international = international_json(day, result, settings)
            _write(folder / "international.json", international)

            window_start = day - timedelta(days=settings.baseline_days)
            pair_day = load_pair_day(connection, window_start, day)
            if pair_day.empty:
                pairs_note = "no pair data (pull again with --force)"
                pairs = pd.DataFrame(columns=["a", "b"])
            else:
                try:
                    pairs = pair_scores(pair_day, day, settings, complete,
                                        pair_cuts)
                    pairs_note = (f"{len(pairs):,} pairs, "
                                  f"{int((pairs['band'] > 0).sum()):,} "
                                  "with enough contact")
                except InsufficientHistory as error:
                    pairs = pd.DataFrame(columns=["a", "b"])
                    pairs_note = f"pairs skipped ({error})"

            _write(folder / "bilateral.json",
                   bilateral_json(day, pairs, pair_cuts, settings))

            print(f"   {day}: {len(international['countries'])} countries "
                  f"scored, {pairs_note}")

        dates = update_index(WEB_DIR)

    finally:
        connection.close()

    print(f"\nDone: {len(dates)} date(s) in {WEB_DIR}/index.json. To view: "
          "python -m http.server 8000, then open "
          "http://localhost:8000/src/web/")


if __name__ == "__main__":
    main()
