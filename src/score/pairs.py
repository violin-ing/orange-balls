"""
Scores the relationship between every pair of countries, for the Bilateral
View (decision log 25 and 28).

=== The relation score ===
    For a pair of countries A and B, over the baseline_days days (90) up to
    and including the assessment date, using every event between them in
    either direction (A acting on B and B acting on A):

        coop      cooperative events (QuadClass 1 and 2)
        n         cooperative + conflictual events (QuadClass 1 to 4)

        value     = (coop + a * p0) / (n + a)
        se        = sqrt(value * (1 - value) / (n + a))

    p0 is the cooperative share over all pairs in the same window and a is
    prior_strength (20), the same smoothing as the GDELT criteria (decision
    log 9.2). 1 = all cooperation, 0 = all conflict.

    Symmetric: one score per pair, so "UK-Russia" is the same from either
    side. The stored table keeps the direction, so a directed reading can be
    added later without re-pulling (decision log 25.6).

=== Too little contact is not good relations (decision log 25.3) ===
    Most pairs of countries hardly ever appear in the news together. With
    the smoothing above, a pair with no events would get p0, the average,
    which would look like an ordinary relationship. So a pair is only given
    a band when its own events outweigh the smoothing: n >= a (20 events in
    90 days). Below that it is shown as "too little contact to judge". This
    adds no new constant: it is the point at which the pair's own data
    counts for more than the all-pair average.

=== Bands ===
    Five bands, cut at the same percentiles as the country bands (5, 20, 50
    and 80; Settings.band_percentiles) of every pair score with enough data
    on every reference day, computed once by compute_bounds and then frozen
    (decision log 14.1). Band 1 = the most conflictual. Labels and
    descriptions are in relation_bands.csv (draft wording).

Main functions:
    pair_window_counts()  symmetric counts per pair over one window
    pair_scores()         value, se, enough data, band, for one date
    pair_reference_values()  every pair score on every reference day, for
                             the cut points
    pair_cut_rows()       the cut points as band_cuts rows
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.score.bands import band_of
from src.score.fast import InsufficientHistory
from src.score.settings import Settings
from src.store.database import BAND_CUT_COLUMNS

LABELS_PATH = Path(__file__).with_name("relation_bands.csv")
LABEL_COLUMNS: list[str] = ["band", "label", "description", "status"]

SCORE_COLUMNS: list[str] = [
    "a", "b", "n", "coop", "conflict", "material_conflict",
    "a_to_b", "b_to_a", "value", "se", "enough", "band",
]

# Settings that change a pair score
PAIR_SETTINGS: tuple[str, ...] = (
    "baseline_days", "prior_strength", "min_baseline_coverage",
)


def pair_variant(settings: Settings) -> str:
    """
    A code naming the settings behind pair scores, used to store their cut
    points in the band_cuts table next to the country ones ("pairs-...").
    """
    described = {name: getattr(settings, name) for name in PAIR_SETTINGS}
    text = json.dumps(described, sort_keys=True)
    return "pairs-" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_relation_bands(path: Path = LABELS_PATH) -> pd.DataFrame:
    """
    The relation band labels: band, label, description, status.
    """
    table = pd.read_csv(path, keep_default_na=False)
    if list(table.columns) != LABEL_COLUMNS:
        raise ValueError(f"{path.name}: columns should be {LABEL_COLUMNS}")
    return table


def _symmetric(pair_day: pd.DataFrame) -> pd.DataFrame:
    """
    The pair table with each pair written one way round (a < b), keeping
    which direction each event went.
    """
    swap = pair_day["actor1"] > pair_day["actor2"]
    frame = pair_day.assign(
        a=pair_day["actor1"].where(~swap, pair_day["actor2"]),
        b=pair_day["actor2"].where(~swap, pair_day["actor1"]),
        a_to_b=pair_day["n_events"].where(~swap, 0),
        b_to_a=pair_day["n_events"].where(swap, 0),
    )
    frame["n"] = frame["n_coop"] + frame["n_conflict"]
    return frame


def _window(complete: list[date], end: date, settings: Settings) -> list[date]:
    """
    The complete days in the baseline_days days up to and including end.
    """
    start = end - timedelta(days=settings.baseline_days - 1)
    days = [d for d in complete if start <= d <= end]
    needed = settings.min_baseline_coverage * settings.baseline_days

    if len(days) < needed:
        raise InsufficientHistory(
            f"pairs: {len(days)} complete days between {start} and {end}, "
            f"need at least {int(np.ceil(needed))}"
        )

    return days


def pair_window_counts(
        pair_day: pd.DataFrame,
        days: Iterable[date],
    ) -> pd.DataFrame:
    """
    Symmetric totals per pair over the given days: n, coop, conflict,
    material_conflict, a_to_b, b_to_a (events each way).
    """
    wanted = set(days)
    rows = pair_day[pd.to_datetime(pair_day["date"]).dt.date.isin(wanted)]
    frame = _symmetric(rows).rename(columns={
        "n_coop": "coop", "n_conflict": "conflict",
        "n_material_conflict": "material_conflict",
    })
    sums = ["n", "coop", "conflict", "material_conflict", "a_to_b", "b_to_a"]
    return frame.groupby(["a", "b"])[sums].sum().reset_index()


def _score(counts: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Add value, se and enough to window counts.
    """
    a = settings.prior_strength
    total = counts["n"].sum()
    if total == 0:
        raise InsufficientHistory("pairs: no events between countries")

    prior = counts["coop"].sum() / total
    value = (counts["coop"] + a * prior) / (counts["n"] + a)
    se = np.sqrt(value * (1 - value) / (counts["n"] + a))

    return counts.assign(value=value, se=se, enough=counts["n"] >= a)


def pair_scores(
        pair_day: pd.DataFrame,
        assessment_date: date,
        settings: Settings,
        complete_days: list[date],
        cuts: np.ndarray | None,
    ) -> pd.DataFrame:
    """
    One row per pair with events in the window (SCORE_COLUMNS). band is 0
    for pairs without enough data, and for every pair when cuts is None.
    """
    settings.validate()
    if assessment_date not in complete_days:
        raise InsufficientHistory(
            f"{assessment_date} has not been pulled completely"
        )

    days = _window(complete_days, assessment_date, settings)
    scored = _score(pair_window_counts(pair_day, days), settings)

    band = np.zeros(len(scored), dtype=int)
    if cuts is not None:
        band = np.where(scored["enough"], band_of(scored["value"], cuts), 0)
    scored["band"] = band

    for column in ("n", "coop", "conflict", "material_conflict",
                   "a_to_b", "b_to_a"):
        scored[column] = scored[column].astype("int64")

    return scored[SCORE_COLUMNS].sort_values(["a", "b"], ignore_index=True)


def pair_reference_values(
        pair_day: pd.DataFrame,
        days: Iterable[date],
        settings: Settings,
        complete_days: list[date],
    ) -> tuple[pd.DataFrame, list[date]]:
    """
    Every pair score with enough data on every given day (date, a, b,
    value), for the cut points. Days without enough history are skipped and
    returned separately.

    Uses running totals over the days, so a reference period of a hundred
    days costs about as much as one day.
    """
    complete = set(complete_days)
    frame = _symmetric(pair_day)
    frame["day"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[frame["day"].isin(complete)]

    daily = frame.groupby(["a", "b", "day"])[["n", "n_coop"]].sum()
    if daily.empty:
        raise InsufficientHistory("pairs: no events between countries")

    # One column per calendar day, zeros where a pair had no events
    first = min(daily.index.get_level_values("day"))
    last = max(daily.index.get_level_values("day"))
    calendar = [first + timedelta(days=i)
                for i in range((last - first).days + 1)]
    n = daily["n"].unstack("day").reindex(columns=calendar, fill_value=0)
    coop = daily["n_coop"].unstack("day").reindex(columns=calendar,
                                                  fill_value=0)
    n, coop = n.fillna(0).to_numpy(), coop.fillna(0).to_numpy()
    pairs = daily["n"].unstack("day").index

    # Running totals: the window ending on day i is cum[i] - cum[i - W]
    zero = np.zeros((len(pairs), 1))
    cum_n = np.hstack([zero, n.cumsum(axis=1)])
    cum_coop = np.hstack([zero, coop.cumsum(axis=1)])
    width = settings.baseline_days
    a = settings.prior_strength

    frames: list[pd.DataFrame] = []
    skipped: list[date] = []

    for day in days:
        try:
            _window(complete_days, day, settings)
        except InsufficientHistory:
            skipped.append(day)
            continue
        if day < first or day > last:
            skipped.append(day)
            continue

        end = (day - first).days + 1
        start = max(end - width, 0)
        n_w = cum_n[:, end] - cum_n[:, start]
        coop_w = cum_coop[:, end] - cum_coop[:, start]
        if n_w.sum() == 0:
            skipped.append(day)
            continue

        prior = coop_w.sum() / n_w.sum()
        value = (coop_w + a * prior) / (n_w + a)
        enough = n_w >= a

        frames.append(pd.DataFrame({
            "date": day,
            "a": pairs.get_level_values("a")[enough],
            "b": pairs.get_level_values("b")[enough],
            "value": value[enough],
        }))

    if not frames:
        raise InsufficientHistory(
            "pairs: no reference day could be scored, so the relation band "
            "cut points cannot be worked out"
        )

    return pd.concat(frames, ignore_index=True), skipped


def pair_cut_rows(values: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    The relation band cut points: percentiles (settings.band_percentiles) of
    every pair score with enough data over the reference days, as rows for
    the band_cuts table.
    """
    x = values["value"].to_numpy(dtype=float)
    now = datetime.now(UTC).isoformat(timespec="seconds")

    rows = [
        {
            "variant": pair_variant(settings),
            "percentile": float(p),
            "value": float(np.percentile(x, p)),
            "reference_start": str(values["date"].min()),
            "reference_end": str(values["date"].max()),
            "computed_at": now,
        }
        for p in settings.band_percentiles
    ]

    return pd.DataFrame(rows, columns=BAND_CUT_COLUMNS)


def stored_pair_cuts(band_cuts: pd.DataFrame,
                     settings: Settings) -> np.ndarray | None:
    """
    The stored relation cut points for these settings, or None.
    """
    rows = band_cuts[
        (band_cuts["variant"] == pair_variant(settings))
        & band_cuts["percentile"].isin(
            [float(p) for p in settings.band_percentiles])
    ].sort_values("percentile")

    if len(rows) != len(settings.band_percentiles):
        return None

    return rows["value"].to_numpy(dtype=float)
