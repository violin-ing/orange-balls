"""
The structural (slow-layer) variables this project uses, in one place.

Each variable records where it comes from and which way it points. The
direction is taken from the source's own codebook and must not be guessed:
a reversed direction would silently invert a criterion in every score.

    higher_is_better=True   a high value means better conditions for
                            receiving an announcement
    higher_is_better=False  a high value means worse conditions

Each also has a role:
    "criterion"   scored by default
    "candidate"   loaded and available, but scored only when named in
                  Settings.criteria (awaiting a decision)
    "validation"  never scored; used for checks such as the correlation
                  with GDP per capita

and, where the scale has fixed ends, its bounds (used to normalise).

Sources for the directions:
    V-Dem v16 codebook (Pemstein et al. 2026), answer scales:
        v2mecenefm      0 = censorship is direct and routine
                        4 = government rarely attempts to censor
        v2meharjrn      0 = no journalist dares offend powerful actors
                        4 = journalists are never harassed
        v2smgovshutcap  0 = government lacks the capacity to shut down
                            domestic Internet access
                        4 = government can shut down all, or almost all
    Digital Society Project codebook v8 (March 2026, V-Dem v16):
        v2smgovdom      0 = government disseminates false information on
                            all key political issues ("extremely often")
                        4 = never, or almost never
        v2smpardom      0 = major parties and candidates do so extremely
                            often; 4 = never, or almost never
        v2smpolsoc      0 = serious polarization of society
                        4 = no polarization
    WGI: the governance score runs from 0 (worst) to 100 (best), anchored
    to fixed benchmarks so that it is comparable across years.

V-Dem values are the measurement model's estimates, not the 0-4 answer
scale: they have no fixed ends, so their normalisation bounds come from a
reference period (src/score/bounds.py). The 0-4 directions above still
hold for them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    name: str                  # name used in this project's tables
    source: str                # "vdem" or "wgi"
    source_name: str           # column (V-Dem) or sheet (WGI) in the file
    label: str
    higher_is_better: bool
    role: str = "criterion"
    bounds: tuple[float, float] | None = None


VARIABLES: dict[str, Variable] = {
    v.name: v
    for v in (
        Variable(
            "wgi_ge", "wgi", "ge",
            "Government effectiveness (score 0-100)",
            higher_is_better=True, bounds=(0.0, 100.0),
        ),
        Variable(
            "wgi_pv", "wgi", "pv",
            "Political stability and absence of violence/terrorism "
            "(score 0-100)",
            higher_is_better=True, bounds=(0.0, 100.0),
        ),
        Variable(
            "v2mecenefm", "vdem", "v2mecenefm",
            "Government censorship effort, media (high = less censorship)",
            higher_is_better=True,
        ),
        Variable(
            "v2meharjrn", "vdem", "v2meharjrn",
            "Harassment of journalists (high = less harassment)",
            higher_is_better=True,
        ),
        Variable(
            "v2smgovshutcap", "vdem", "v2smgovshutcap",
            "Government Internet shut down capacity (high = more capacity)",
            higher_is_better=False,
        ),
        Variable(
            "v2smgovdom", "vdem", "v2smgovdom",
            "Government dissemination of false information, domestic "
            "(high = less)",
            higher_is_better=True, role="candidate",
        ),
        Variable(
            "v2smpardom", "vdem", "v2smpardom",
            "Party dissemination of false information, domestic "
            "(high = less)",
            higher_is_better=True, role="candidate",
        ),
        Variable(
            "v2smpolsoc", "vdem", "v2smpolsoc",
            "Polarization of society (high = less polarized)",
            higher_is_better=True, role="candidate",
        ),
        Variable(
            "gdp_pc", "wb", "NY.GDP.PCAP.PP.KD",
            "GDP per capita, PPP (constant 2021 international $)",
            higher_is_better=True, role="validation",
        ),
    )
}

ROLES = ("criterion", "candidate", "validation")


def variables_from(source: str) -> list[Variable]:
    """The variables that come from one source, in a fixed order."""
    return [v for v in VARIABLES.values() if v.source == source]


def with_role(role: str) -> list[str]:
    """Names of the variables with one role, in a fixed order."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    return [n for n, v in VARIABLES.items() if v.role == role]
