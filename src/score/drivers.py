"""
Names the criteria that pull a country's score down most, and composes the
text a planner reads for that country (decision log 24.3 and 24.4).

Two countries in the same band can be there for different reasons, so each
country gets its band's description plus one line per driving criterion.
There is one text per criterion, written once (criterion_texts.csv), never a
text per combination of criteria: the two weakest of eleven criteria can be
any of 55 pairs.

=== Which criteria are named (select_drivers) ===
    For each criterion, a country's normalised value (0 to 1, 1 = best) is
    compared with the average of all scored countries on that date, in
    standard deviations of that criterion across those countries:

        gap = (country's value - average) / standard deviation

    A criterion can be named only if the gap is at most -driver_floor_sd
    (default 0.5): notably below average. Of those, the driver_count (default
    2) most negative are named, weakest first. A country with no notably
    weak criterion gets none named.

    Why standard deviations rather than the raw difference: the criteria are
    spread very differently. V-Dem values span most of the 0 to 1 range,
    while the GDELT criteria sit close together around each country's own
    normal. A raw difference would therefore name V-Dem criteria almost
    every time and a real spike in unrest almost never. This is the
    "criterion value against the global mean" method of decision log 17.2,
    applied to one country instead of a whole band.

=== Criterion texts (criterion_texts.csv) ===
    Columns: criterion, audience, label, text, status (same shape as
    band_descriptions.csv). One internal row per criterion; the
    per-criterion breakdown is internal only (decision log 11.3).

    Texts are phrased as consequences for the communication task, never as
    a judgement of the government: "official statements are less likely to
    be taken at face value", not "the government spreads disinformation"
    (decision log 24.3). That rule cannot be checked by code; the structure
    can, so check_criterion_texts() requires one row per criterion, no empty
    label or text, and a known status.

Main functions:
    load_criterion_texts(), check_criterion_texts()
    select_drivers()   the named criteria for every scored country
    compose()          band description + one line per driver, for one
                       country
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.score.settings import Settings

TEXTS_PATH = Path(__file__).with_name("criterion_texts.csv")
TEXT_COLUMNS: list[str] = ["criterion", "audience", "label", "text", "status"]
STATUSES: tuple[str, ...] = ("draft", "final")

DRIVER_COLUMNS: list[str] = [
    "iso3", "rank", "criterion", "value", "average", "gap_sd",
]


class BadCriterionTexts(ValueError):
    """
    criterion_texts.csv is missing rows, has empty text, or the wrong shape.
    """


def load_criterion_texts(path: Path = TEXTS_PATH) -> pd.DataFrame:
    """
    Read criterion_texts.csv.
    """
    table = pd.read_csv(path, keep_default_na=False)

    if list(table.columns) != TEXT_COLUMNS:
        raise BadCriterionTexts(
            f"{path.name}: columns should be {TEXT_COLUMNS}, got "
            f"{list(table.columns)}"
        )

    return table


def check_criterion_texts(
        table: pd.DataFrame,
        criteria: list[str],
    ) -> pd.DataFrame:
    """
    Check there is exactly one internal text for each criterion scored, and
    return the table. Raises BadCriterionTexts listing every problem.
    """
    problems: list[str] = []
    internal = table[table["audience"] == "internal"]

    missing = sorted(set(criteria) - set(internal["criterion"]))
    if missing:
        problems.append(f"no internal text for {missing}")

    twice = sorted(internal.loc[internal["criterion"].duplicated(),
                                "criterion"])
    if twice:
        problems.append(f"more than one internal text for {twice}")

    empty = table[(table["label"].str.strip() == "")
                  | (table["text"].str.strip() == "")]
    if not empty.empty:
        problems.append(f"empty label or text for {list(empty['criterion'])}")

    unknown = sorted(set(table["status"]) - set(STATUSES))
    if unknown:
        problems.append(f"unknown status {unknown}; use one of {STATUSES}")

    if problems:
        raise BadCriterionTexts("; ".join(problems))

    return table


def select_drivers(norm: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    The criteria that pull each country down most (DRIVER_COLUMNS), weakest
    first, at most settings.driver_count per country.

    norm: the normalised values of the scored countries (country x
    criterion, higher = better), e.g. ScoreResult.norm.dropna(). The
    averages and spreads are taken over exactly these countries.
    """
    settings.validate()

    average = norm.mean()
    spread = norm.std()

    # A criterion with no spread cannot single out any country
    spread = spread.where(spread > 0)
    gap = (norm - average) / spread

    rows = []
    for iso3, gaps in gap.iterrows():
        weak = gaps[gaps <= -settings.driver_floor_sd].sort_values()

        for rank, (criterion, value) in enumerate(
                weak.head(settings.driver_count).items(), start=1):
            rows.append((iso3, rank, criterion, norm.at[iso3, criterion],
                         average[criterion], value))

    return pd.DataFrame(rows, columns=DRIVER_COLUMNS)


def compose(
        iso3: str,
        band_row: pd.Series,
        drivers: pd.DataFrame,
        texts: pd.DataFrame,
    ) -> list[str]:
    """
    The lines a planner reads for one country: the band's label and action,
    then one line per driving criterion ("label: text").

    band_row: the country's row from bands.describe() (label, action).
    """
    lines = [f"{band_row['label']}: {band_row['action']}"]

    internal = texts[texts["audience"] == "internal"].set_index("criterion")
    named = drivers[drivers["iso3"] == iso3].sort_values("rank")

    for criterion in named["criterion"]:
        row = internal.loc[criterion]
        lines.append(f"{row['label']}: {row['text']}")

    return lines
