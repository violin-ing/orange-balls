"""
The structural (slow-layer) variables this project uses, in one place.

Each variable records where it comes from and which way it points. The
direction is taken from the source's own codebook and must not be guessed:
a reversed direction would silently invert a criterion in every score.

    higher_is_better=True   a high value means better conditions for
                            receiving an announcement
    higher_is_better=False  a high value means worse conditions

Sources for the directions:
    V-Dem v16 codebook (Pemstein et al. 2026), answer scales:
        v2mecenefm      0 = censorship is direct and routine
                        4 = government rarely attempts to censor
        v2meharjrn      0 = no journalist dares offend powerful actors
                        4 = journalists are never harassed
        v2smgovshutcap  0 = government lacks the capacity to shut down
                            domestic Internet access
                        4 = government can shut down all, or almost all
    WGI: estimates run from about -2.5 (weak) to +2.5 (strong)
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


VARIABLES: dict[str, Variable] = {
    v.name: v
    for v in (
        Variable(
            "wgi_ge", "wgi", "ge",
            "Government effectiveness", higher_is_better=True,
        ),
        Variable(
            "wgi_pv", "wgi", "pv",
            "Political stability and absence of violence/terrorism",
            higher_is_better=True,
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
    )
}


def variables_from(source: str) -> list[Variable]:
    """The variables that come from one source, in a fixed order."""
    return [v for v in VARIABLES.values() if v.source == source]
