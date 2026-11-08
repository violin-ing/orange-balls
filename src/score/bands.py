"""
Turns country scores into bands, and bands into descriptions.

=== Bands ===
    A score is placed in one of Settings.bands internal bands (five by
    default). Band 1 holds the lowest scores, i.e. the most difficult
    conditions, as in DEFCON, where 1 is the most severe.

    Cut points come from Settings.band_cuts, or, when it is empty, divide the
    min-max range [minmax_floor, 1] into equal widths. Because the
    normalisation bounds are fixed (src/score/bounds.py), the cut points are
    fixed too, so a band means the same thing on every date. z-scores have no
    fixed range: without explicit cut points they get no band.

    The score's uncertainty is shown as a band range, with the bands of:
        band_low    score - 1.96 x se
        band_high   score + 1.96 x se

=== Descriptions ===
    One row per band and audience, with columns:
        band            the band number for that audience
        audience        "internal" or "public"
        internal_bands  the internal band(s) the row covers, e.g. "1" or "1;2";
                        for internal rows, the row's own band
        label           a short name
        action          internal: what the responsible bodies do;
                        public: the observable consequence of that action
        status          "draft" until the wording is agreed

    The public row is derived from the internal row(s) it covers, not written
    separately: it states the observable consequence of the internal action
    (internal "coordinate with local law enforcement" becomes public "expect 
    heightened law enforcement activity"). That cannot be checked by code, but
    the structure can, so check_descriptions() requires:
        - exactly one internal row per internal band;
        - public bands numbered 1, 2, ... with no gaps;
        - every internal band covered by exactly one public row, each public
            row covering a run of consecutive internal bands, in order. So 
            there may be fewer public bands than internal ones.

        ^ Note: Yet to decide if the public scale should have fewer bands
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.score.settings import Settings

DESCRIPTIONS_PATH = Path(__file__).with_name("band_descriptions.csv")
DESCRIPTION_COLUMNS: list[str] = [
    "band", "audience", "internal_bands", "label", "action", "status",
]
AUDIENCES = ("internal", "public")
Z_95 = 1.959963984540054


class BadDescriptions(ValueError):
    """
    The band description table breaks one of its rules.
    """


def cut_points(settings: Settings) -> np.ndarray | None:
    """
    The cut points between internal bands, lowest first, or None when the
    scores have no fixed range (z-scores without explicit cut points).
    """
    settings.validate()

    if settings.band_cuts:
        return np.asarray(settings.band_cuts, dtype=float)
    
    if settings.normalisation != "minmax":
        return None
    
    floor = settings.minmax_floor
    steps = np.arange(1, settings.bands) / settings.bands

    return floor + (1 - floor) * steps


def band_of(scores: np.ndarray | pd.Series, cuts: np.ndarray) -> np.ndarray:
    """
    Internal band (1 to len(cuts) + 1) of each score. A score exactly on
    a cut point goes to the higher band.
    """
    return np.digitize(np.asarray(scores, dtype=float), cuts) + 1


def assign_bands(scores: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """
    Add band, band_low and band_high to a table with score and score_se
    columns (as made by combine.aggregate).
    """
    cuts = cut_points(settings)
    out = scores.copy()
    if cuts is None:
        for column in ("band", "band_low", "band_high"):
            out[column] = pd.array([pd.NA] * len(out), dtype="Int64")

        return out
    
    spread = Z_95 * out["score_se"]
    out["band"] = band_of(out["score"], cuts)
    out["band_low"] = band_of(out["score"] - spread, cuts)
    out["band_high"] = band_of(out["score"] + spread, cuts)

    for column in ("band", "band_low", "band_high"):
        out[column] = out[column].astype("int64")

    return out


def _covered(text: str) -> list[int]:
    return [int(part) for part in str(text).split(";") if part.strip()]


def load_descriptions(path: Path = DESCRIPTIONS_PATH) -> pd.DataFrame:
    """
    Read band_descriptions.csv.
    """
    table = pd.read_csv(path, dtype={"internal_bands": "string"},
                        keep_default_na=False)
    
    if list(table.columns) != DESCRIPTION_COLUMNS:
        raise BadDescriptions(
            f"{path.name}: columns should be {DESCRIPTION_COLUMNS}, got "
            f"{list(table.columns)}"
        )
    
    return table


def check_descriptions(table: pd.DataFrame, bands: int) -> pd.DataFrame:
    """
    Check the table's structure for this number of internal bands and
    return it. Raises BadDescriptions listing every problem.
    """
    problems: list[str] = []
    unknown = sorted(set(table["audience"]) - set(AUDIENCES))

    if unknown:
        problems.append(f"unknown audiences {unknown}")

    empty = table[(table["label"].str.strip() == "")
                  | (table["action"].str.strip() == "")]
    
    if not empty.empty:
        problems.append(
            "empty label or action in rows "
            f"{list(zip(empty['audience'], empty['band'], strict=True))}"
        )

    internal = table[table["audience"] == "internal"]
    if sorted(internal["band"]) != list(range(1, bands + 1)):
        problems.append(
            f"internal bands should be 1 to {bands}, once each; found "
            f"{sorted(internal['band'])}"
        )

    for row in internal.itertuples():
        if _covered(row.internal_bands) != [row.band]:
            problems.append(
                f"internal band {row.band}: internal_bands should be "
                f"'{row.band}'"
            )

    public = table[table["audience"] == "public"].sort_values("band")
    if public.empty:
        problems.append("no public rows")

    elif list(public["band"]) != list(range(1, len(public) + 1)):
        problems.append(
            f"public bands should be 1, 2, ... with no gaps; found "
            f"{list(public['band'])}"
        )

    else:
        covered = [b for text in public["internal_bands"]
                   for b in _covered(text)]
        
        if covered != list(range(1, bands + 1)):
            problems.append(
                "public rows must cover internal bands 1 to "
                f"{bands} once each, in order, in runs of consecutive "
                f"bands; they cover {covered}"
            )

        for row in public.itertuples():
            run = _covered(row.internal_bands)

            if run and run != list(range(run[0], run[-1] + 1)):
                problems.append(
                    f"public band {row.band} covers {run}, which is not "
                    "a run of consecutive internal bands"
                )

    if problems:
        raise BadDescriptions("; ".join(problems))
    
    return table


def public_band(internal_band: pd.Series, table: pd.DataFrame) -> pd.Series:
    """
    The public band that each internal band falls in.
    """
    public = table[table["audience"] == "public"]
    lookup = {b: row.band for row in public.itertuples()
              for b in _covered(row.internal_bands)}
    
    return internal_band.map(lookup)


def describe(
        banded: pd.DataFrame,
        table: pd.DataFrame,
        audience: str = "internal",
    ) -> pd.DataFrame:
    """
    The band, label and action for each country, for one audience.
    banded: a table with an internal band column (from assign_bands).
    """
    if audience not in AUDIENCES:
        raise ValueError(f"audience must be one of {AUDIENCES}")
    
    band = banded["band"]
    if audience == "public":
        band = public_band(band, table)

    rows = table[table["audience"] == audience].set_index("band")

    return pd.DataFrame(
        {
            "band": band,
            "label": band.map(rows["label"]),
            "action": band.map(rows["action"]),
        }, index=banded.index
    )
