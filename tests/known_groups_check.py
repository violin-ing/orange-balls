"""
One-off check: when GDELT codes an actor as a known group (e.g. the UN, NATO,
the African Union, the Taliban etc.), what does it put in that actor's
country code field? 

    python3 -m tests.known_groups_check

Reads the GDELT files already in the cache. Needs no network.

Why it matters: international conflict and cooperation count an event when
the other actor is a different country or a group of countries (a code such
as WST for NATO). If GDELT leaves the country field BLANK for some groups,
those events are currently missed, and organisations need separate
handling. The answer decides whether that is a filter change or a bigger
one.

Second question: does GDELT fill in the actor TYPE codes IGO
(intergovernmental organisation) and NGO (non-governmental organisation)
for these groups? If so, "the other actor is a different country, or is
typed IGO" can replace the region-code rule. The last two sections show
what that change would gain and lose.

Not a unit test: pytest does not collect this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.countries.actor_codes import actor_to_iso3, clean_actor_codes
from src.gdelt.download import read_events

CACHE_DIR = Path("data/cache/gdelt")

TYPE_SLOTS = ("Type1Code", "Type2Code", "Type3Code")

COLUMNS: tuple[str, ...] = (
    "GlobalEventID",
    "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    *(f"Actor1{slot}" for slot in TYPE_SLOTS),
    "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    *(f"Actor2{slot}" for slot in TYPE_SLOTS),
)


def country_field_kind(codes: pd.Series) -> pd.Series:
    """
    Label each country code as 'blank', 'country' or 'group/region'.
    """
    cleaned = clean_actor_codes(codes)
    kind = pd.Series("group/region", index=codes.index)
    kind[actor_to_iso3(codes).notna()] = "country"
    kind[cleaned == ""] = "blank"
    return kind


def main() -> None:
    paths = sorted(CACHE_DIR.glob("*.export.CSV.zip"))
    if not paths:
        raise SystemExit(f"no cached files in {CACHE_DIR}")

    events = pd.concat(
        [read_events(p, COLUMNS) for p in paths], ignore_index=True
    )
    print(f"{len(paths)} files, {len(events):,} events\n")

    # Stack both actor slots into one table: one row per actor
    sides = []
    for n in ("1", "2"):
        side = pd.DataFrame({
            "slot": f"actor {n}",
            "name": events[f"Actor{n}Name"],
            "country_code": clean_actor_codes(events[f"Actor{n}CountryCode"]),
            "group_code": clean_actor_codes(
                events[f"Actor{n}KnownGroupCode"]
            ),
            "kind": country_field_kind(events[f"Actor{n}CountryCode"]),
            # All type codes of this actor, joined, e.g. "IGO" or "GOV MIL"
            "types": (
                events[[f"Actor{n}{slot}" for slot in TYPE_SLOTS]]
                .fillna("")
                .astype(str)
                .apply(lambda row: " ".join(c for c in row if c), axis=1)
            ),
        })
        sides.append(side)
    actors = pd.concat(sides, ignore_index=True)
    grouped = actors[actors["group_code"] != ""]

    print(f"actor slots with a known group code: {len(grouped):,} "
          f"of {len(actors):,}\n")

    print("--- what the country field holds when a group code is set ---")
    print(pd.crosstab(grouped["slot"], grouped["kind"], margins=True))

    print("\n--- 25 most common group codes ---")
    print("(blank / country / group = how often each country field kind "
          "appears)")
    top = grouped["group_code"].value_counts().head(25).index
    for code in top:
        rows = grouped[grouped["group_code"] == code]
        kinds = rows["kind"].value_counts()
        countries = rows.loc[rows["kind"] != "blank", "country_code"]
        names = rows["name"].value_counts().head(2).index.tolist()
        types = rows["types"].replace("", "(none)").value_counts()
        print(
            f"  {code:<4} {len(rows):>6,}   "
            f"blank {kinds.get('blank', 0):>5,}  "
            f"country {kinds.get('country', 0):>5,}  "
            f"group {kinds.get('group/region', 0):>5,}   "
            f"codes {countries.value_counts().head(3).index.tolist()}  "
            f"e.g. {names}"
        )
        top_types = {k: int(v) for k, v in types.head(3).items()}
        print(f"         types {top_types}")

    blank = grouped[grouped["kind"] == "blank"]
    print(f"\ngroup actors with a BLANK country field: {len(blank):,} "
          f"({len(blank) / max(len(grouped), 1):.0%} of group actors)")
    print("These are the events the international criteria currently miss.")

    is_igo = actors["types"].str.split().apply(lambda t: "IGO" in t)
    is_ngo = actors["types"].str.split().apply(lambda t: "NGO" in t)
    print(f"\nactor slots typed IGO: {int(is_igo.sum()):,}, "
          f"typed NGO: {int(is_ngo.sum()):,}")

    print("\n--- what the IGO rule would GAIN: typed IGO, blank country ---")
    gain = actors[is_igo & (actors["kind"] == "blank")]
    summary = gain.groupby("group_code").agg(
        slots=("name", "size"), example=("name", "first")
    ).sort_values("slots", ascending=False).head(15)
    print(f"{len(gain):,} slots")
    print(summary.to_string())

    print("\n--- what it would LOSE: region code in the country field, "
          "but not typed IGO ---")
    lose = actors[~is_igo & (actors["kind"] == "group/region")]
    summary = lose.groupby("country_code").agg(
        slots=("name", "size"),
        example=("name", lambda n: list(n.value_counts().head(3).index)),
    ).sort_values("slots", ascending=False).head(15)
    print(f"{len(lose):,} slots")
    print(summary.to_string())

    print("\n--- typed IGO but holding a country code (check these) ---")
    odd = actors[is_igo & (actors["kind"] == "country")]
    print(f"{len(odd):,} slots; most common: "
          f"{odd['name'].value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
