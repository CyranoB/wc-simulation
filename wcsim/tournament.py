"""Tournament-simulation engine. Supports WC 2018/2022 (8 groups x 4 = 32
teams, R16 first round, 3rd-place playoff) and WC 2026 (12 groups x 4 = 48,
R32 first round, no 3rd-place playoff). Dispatched by team count."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .model import sample_match
from .ratings.base import RatingSystem
from .types import MatchResult, Params, Team, TournamentResult


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


# ---------------------------------------------------------------------------
# Best third-place selection
# ---------------------------------------------------------------------------


def best_third_place_teams(candidates: list[dict], n: int) -> list[str]:
    """Rank third-place teams by (points, gd, gf) and return top n iso3s."""
    if n == 0:
        return []
    ranked = sorted(candidates, key=lambda s: (-s["points"], -s["gd"], -s["gf"]))
    return [s["team"] for s in ranked[:n]]


# ---------------------------------------------------------------------------
# Knockout seeding
# ---------------------------------------------------------------------------

# Classic crosswise R16 bracket for WC 2018/2022 format.
_R16_PAIRS_2018_2022 = [
    (0, 1), (2, 3), (4, 5), (6, 7),  # top half
    (1, 0), (3, 2), (5, 4), (7, 6),  # bottom half (reversed group pairs)
]


def seed_knockout(
    structure: TournamentStructure,
    group_winners: list[str], group_runners_up: list[str],
    best_thirds: list[str],
) -> list[str]:
    """Return iso3 codes in bracket order -- pairs (i, i+1) play in round 1."""
    if structure.name == "WC2018-2022":
        # Classic crosswise: 1A vs 2B, 1C vs 2D, ..., 1B vs 2A, 1D vs 2C, ...
        out = []
        for w_idx, r_idx in _R16_PAIRS_2018_2022:
            out.append(group_winners[w_idx])
            out.append(group_runners_up[r_idx])
        return out
    elif structure.name == "WC2026":
        # Simplified sequential arrangement for 32-slot R32 bracket.
        # Top 2 per group (24) in group order, then best 8 thirds appended.
        slot_iso3s = []
        for i in range(structure.groups_count):
            slot_iso3s.append(group_winners[i])
            slot_iso3s.append(group_runners_up[i])
        slot_iso3s.extend(best_thirds)
        if len(slot_iso3s) != 32:
            raise ValueError(f"WC2026 expected 32 slots, got {len(slot_iso3s)}")
        return slot_iso3s
    raise ValueError(f"Unknown structure: {structure.name}")


# ---------------------------------------------------------------------------
# Knockout simulation
# ---------------------------------------------------------------------------


def simulate_knockout(
    seeded: list[str], teams: dict[str, Team], structure: TournamentStructure,
    *, rating: RatingSystem, params: Params, rng: np.random.Generator,
    hosts: set[str],
) -> tuple[list[MatchResult], dict[str, str]]:
    """Play knockout rounds. Returns (matches, {iso3 -> exit_stage or 'Champion'})."""
    matches: list[MatchResult] = []
    placements: dict[str, str] = {}
    current = list(seeded)
    semi_losers: list[str] = []

    for stage in structure.knockout_stages:
        next_round: list[str] = []
        for i in range(0, len(current), 2):
            a_iso3, b_iso3 = current[i], current[i + 1]
            m = sample_match(
                teams[a_iso3], teams[b_iso3],
                rating=rating, params=params,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
                rng=rng, stage=stage,
            )
            matches.append(m)
            if m.went_to_pens:
                winner = m.pen_winner
            elif m.home_goals > m.away_goals:
                winner = m.home
            else:
                winner = m.away
            loser = b_iso3 if winner == a_iso3 else a_iso3
            placements[loser] = stage
            if stage == "SF":
                semi_losers.append(loser)
            next_round.append(winner)
        current = next_round

    champion = current[0]
    placements[champion] = "Champion"

    if structure.third_place_playoff and len(semi_losers) == 2:
        a_iso3, b_iso3 = semi_losers
        m = sample_match(
            teams[a_iso3], teams[b_iso3],
            rating=rating, params=params,
            a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
            rng=rng, stage="3rd",
        )
        matches.append(m)

    return matches, placements


# ---------------------------------------------------------------------------
# Top-level tournament simulation
# ---------------------------------------------------------------------------


def simulate_tournament(
    teams: dict[str, Team], draw: dict[str, list[str]], hosts: set[str],
    *, rating: RatingSystem, params: Params | None = None, seed: int,
) -> TournamentResult:
    """Top-level deterministic entry point."""
    p = params if params is not None else Params()
    structure = _structure_for(len(teams))
    rng = np.random.default_rng(seed)

    group_matches, positions = simulate_group_stage(
        teams=teams, draw=draw, rating=rating, params=p, rng=rng, hosts=hosts,
    )

    group_letters = sorted(draw.keys())
    group_winners = [next(t for t in draw[g] if positions[t] == 1) for g in group_letters]
    group_runners_up = [next(t for t in draw[g] if positions[t] == 2) for g in group_letters]

    # Third-place candidates.
    third_place_iso3s = [
        next(t for t in draw[g] if positions[t] == 3) for g in group_letters
    ]
    # Build standings for those teams from group_matches.
    standings_by_team = {iso3: {"team": iso3, "points": 0, "gd": 0, "gf": 0}
                         for iso3 in third_place_iso3s}
    for m in group_matches:
        for iso3 in (m.home, m.away):
            if iso3 not in standings_by_team:
                continue
            if iso3 == m.home:
                gf, ga = m.home_goals, m.away_goals
            else:
                gf, ga = m.away_goals, m.home_goals
            if gf > ga:
                standings_by_team[iso3]["points"] += 3
            elif gf == ga:
                standings_by_team[iso3]["points"] += 1
            standings_by_team[iso3]["gd"] += gf - ga
            standings_by_team[iso3]["gf"] += gf

    best_thirds = best_third_place_teams(
        list(standings_by_team.values()), n=structure.best_thirds,
    )

    seeded = seed_knockout(structure, group_winners, group_runners_up, best_thirds)
    knockout_matches, knockout_placements = simulate_knockout(
        seeded=seeded, teams=teams, structure=structure,
        rating=rating, params=p, rng=rng, hosts=hosts,
    )

    placements: dict[str, str] = {}
    advancing = set(seeded)
    for iso3 in teams:
        if iso3 not in advancing:
            placements[iso3] = "GroupOut"
    placements.update(knockout_placements)

    final_ratings = {iso3: rating.rating_of(t) for iso3, t in teams.items()}

    return TournamentResult(
        seed=seed, rating_mode=rating.name,
        matches=group_matches + knockout_matches,
        placements=placements, final_ratings=final_ratings,
    )
