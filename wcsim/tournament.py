"""Tournament-simulation engine. Supports WC 2018/2022 (8 groups x 4 = 32
teams, R16 first round, 3rd-place playoff) and WC 2026 (12 groups x 4 = 48,
R32 first round, no 3rd-place playoff). Dispatched by team count."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .model import sample_match
from .ratings.base import RatingSystem
from .types import MatchResult, Params, Team


@dataclass(frozen=True)
class TournamentStructure:
    name: str
    groups_count: int
    group_size: int
    top_per_group: int
    best_thirds: int
    knockout_stages: list[str] = field(default_factory=list)
    third_place_playoff: bool = False


STRUCTURE_2018_2022 = TournamentStructure(
    name="WC2018-2022",
    groups_count=8, group_size=4,
    top_per_group=2, best_thirds=0,
    knockout_stages=["R16", "QF", "SF", "Final"],
    third_place_playoff=True,
)

STRUCTURE_2026 = TournamentStructure(
    name="WC2026",
    groups_count=12, group_size=4,
    top_per_group=2, best_thirds=8,
    knockout_stages=["R32", "R16", "QF", "SF", "Final"],
    third_place_playoff=False,
)


def _structure_for(team_count: int) -> TournamentStructure:
    if team_count == 32:
        return STRUCTURE_2018_2022
    if team_count == 48:
        return STRUCTURE_2026
    raise ValueError(
        f"Unsupported tournament size: {team_count} teams "
        "(supported: 32 for WC 2018/2022, 48 for WC 2026)"
    )


# ---------------------------------------------------------------------------
# Group-stage simulation
# ---------------------------------------------------------------------------


def _points_from_score(home: int, away: int) -> tuple[int, int]:
    if home > away:
        return (3, 0)
    if home < away:
        return (0, 3)
    return (1, 1)


def _rank_group(standings: list[dict], rng: np.random.Generator) -> list[dict]:
    """Sort by (points desc, gd desc, gf desc). Ties broken by random shuffle
    within tied blocks (head-to-head omitted for simplicity)."""
    standings = sorted(standings, key=lambda s: (-s["points"], -s["gd"], -s["gf"]))
    out: list[dict] = []
    i = 0
    while i < len(standings):
        j = i + 1
        while j < len(standings) and (
            standings[j]["points"] == standings[i]["points"]
            and standings[j]["gd"] == standings[i]["gd"]
            and standings[j]["gf"] == standings[i]["gf"]
        ):
            j += 1
        block = standings[i:j]
        if len(block) > 1:
            block = list(block)
            rng.shuffle(block)
        out.extend(block)
        i = j
    return out


def simulate_group_stage(
    teams: dict[str, Team], draw: dict[str, list[str]],
    *, rating: RatingSystem, params: Params, rng: np.random.Generator,
    hosts: set[str],
) -> tuple[list[MatchResult], dict[str, int]]:
    """Simulate round-robin for each group. Returns (matches, positions)
    where positions[iso3] in {1..group_size}."""
    all_matches: list[MatchResult] = []
    positions: dict[str, int] = {}

    for group_letter, group_iso3s in sorted(draw.items()):
        standings = {t: {"team": t, "points": 0, "gd": 0, "gf": 0}
                     for t in group_iso3s}
        for a_iso3, b_iso3 in combinations(group_iso3s, 2):
            m = sample_match(
                teams[a_iso3], teams[b_iso3],
                rating=rating, params=params,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
                rng=rng, stage="group",
            )
            all_matches.append(m)
            pa, pb = _points_from_score(m.home_goals, m.away_goals)
            standings[a_iso3]["points"] += pa
            standings[b_iso3]["points"] += pb
            standings[a_iso3]["gf"] += m.home_goals
            standings[b_iso3]["gf"] += m.away_goals
            standings[a_iso3]["gd"] += m.home_goals - m.away_goals
            standings[b_iso3]["gd"] += m.away_goals - m.home_goals

        ranked = _rank_group(list(standings.values()), rng=rng)
        for pos, entry in enumerate(ranked, start=1):
            positions[entry["team"]] = pos

    return all_matches, positions
