"""Tests for tournament structures and the team-count dispatcher."""
from __future__ import annotations
import pytest


def test_structure_2018_2022_dimensions():
    from wcsim.tournament import STRUCTURE_2018_2022
    s = STRUCTURE_2018_2022
    assert s.groups_count == 8
    assert s.group_size == 4
    assert s.top_per_group == 2
    assert s.best_thirds == 0
    assert s.knockout_stages == ["R16", "QF", "SF", "Final"]
    assert s.third_place_playoff is True


def test_structure_2026_dimensions():
    from wcsim.tournament import STRUCTURE_2026
    s = STRUCTURE_2026
    assert s.groups_count == 12
    assert s.group_size == 4
    assert s.top_per_group == 2
    assert s.best_thirds == 8
    assert s.knockout_stages == ["R32", "R16", "QF", "SF", "Final"]
    assert s.third_place_playoff is False


def test_structure_for_32_teams():
    from wcsim.tournament import _structure_for, STRUCTURE_2018_2022
    assert _structure_for(32) is STRUCTURE_2018_2022


def test_structure_for_48_teams():
    from wcsim.tournament import _structure_for, STRUCTURE_2026
    assert _structure_for(48) is STRUCTURE_2026


def test_structure_for_unsupported_raises():
    from wcsim.tournament import _structure_for
    with pytest.raises(ValueError, match="Unsupported tournament size"):
        _structure_for(24)


def test_simulate_group_stage_returns_correct_counts(default_params):
    """8 groups × C(4,2) = 48 matches; each team gets position 1-4."""
    from wcsim.tournament import simulate_group_stage
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    import numpy as np

    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(8)}
    rng = np.random.default_rng(42)
    live_ratings = {iso3: t.elo for iso3, t in teams.items()}
    matches, positions = simulate_group_stage(
        teams=teams, draw=draw,
        rating=EloRating(default_params), params=default_params,
        rng=rng, hosts=set(), live_ratings=live_ratings,
    )
    assert len(matches) == 48
    assert len(positions) == 32
    for iso3, pos in positions.items():
        assert pos in {1, 2, 3, 4}
    for g in range(8):
        group_teams = [f"T{g*4+j:02d}" for j in range(4)]
        group_positions = sorted(positions[t] for t in group_teams)
        assert group_positions == [1, 2, 3, 4]


def test_simulate_group_stage_is_deterministic(default_params):
    from wcsim.tournament import simulate_group_stage
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team
    import numpy as np

    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(8)}

    def run(seed):
        rng = np.random.default_rng(seed)
        live_ratings = {iso3: t.elo for iso3, t in teams.items()}
        return simulate_group_stage(
            teams=teams, draw=draw,
            rating=EloRating(default_params), params=default_params,
            rng=rng, hosts=set(), live_ratings=live_ratings,
        )
    m1, p1 = run(42)
    m2, p2 = run(42)
    assert p1 == p2
    assert len(m1) == len(m2)


def test_rank_group_sorts_by_points_gd_gf():
    from wcsim.tournament import _rank_group
    import numpy as np
    standings = [
        {"team": "A", "points": 9, "gd": 0, "gf": 5},
        {"team": "B", "points": 4, "gd": 3, "gf": 7},
        {"team": "C", "points": 3, "gd": 0, "gf": 3},
        {"team": "D", "points": 0, "gd": -3, "gf": 1},
    ]
    rng = np.random.default_rng(1)
    ranked = _rank_group(standings, rng=rng)
    assert [s["team"] for s in ranked] == ["A", "B", "C", "D"]


def test_rank_group_gd_breaks_tie():
    from wcsim.tournament import _rank_group
    import numpy as np
    standings = [
        {"team": "A", "points": 6, "gd": 2, "gf": 4},
        {"team": "B", "points": 6, "gd": 5, "gf": 8},
        {"team": "C", "points": 6, "gd": -1, "gf": 3},
    ]
    rng = np.random.default_rng(1)
    ranked = _rank_group(standings, rng=rng)
    assert [s["team"] for s in ranked] == ["B", "A", "C"]


def test_best_third_place_teams_returns_empty_for_n_zero():
    from wcsim.tournament import best_third_place_teams
    candidates = [{"team": "T1", "points": 3, "gd": 0, "gf": 2}]
    assert best_third_place_teams(candidates, n=0) == []


def test_best_third_place_teams_ranks_correctly():
    from wcsim.tournament import best_third_place_teams
    candidates = [
        {"team": "T0", "points": 6, "gd": 4, "gf": 7},
        {"team": "T1", "points": 4, "gd": 2, "gf": 5},
        {"team": "T2", "points": 4, "gd": 1, "gf": 5},
        {"team": "T3", "points": 3, "gd": 0, "gf": 3},
    ]
    top2 = best_third_place_teams(candidates, n=2)
    assert top2 == ["T0", "T1"]


def test_simulate_tournament_32_teams_end_to_end(default_params):
    """32-team tournament runs and produces 64 matches + 1 champion."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team

    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(8)}

    result = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=EloRating(default_params), params=default_params, seed=42,
    )
    assert result.seed == 42
    assert result.rating_mode == "elo"
    # 48 group + 8 R16 + 4 QF + 2 SF + 1 Final + 1 third-place = 64
    assert len(result.matches) == 64
    champions = [iso for iso, stage in result.placements.items() if stage == "Champion"]
    assert len(champions) == 1
    assert set(result.placements.keys()) == set(teams.keys())


def test_simulate_tournament_48_teams_end_to_end(default_params):
    """48-team tournament runs and produces 103 matches + 1 champion."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team

    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(48)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(12)}

    result = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=EloRating(default_params), params=default_params, seed=42,
    )
    assert result.seed == 42
    # 72 group + 16 R32 + 8 R16 + 4 QF + 2 SF + 1 Final = 103 (no 3rd place)
    assert len(result.matches) == 103
    champions = [iso for iso, stage in result.placements.items() if stage == "Champion"]
    assert len(champions) == 1
    assert set(result.placements.keys()) == set(teams.keys())


def test_simulate_tournament_is_deterministic(default_params):
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Team

    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="UNK", elo=1500.0 + i * 10)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(8)}
    r1 = simulate_tournament(teams=teams, draw=draw, hosts=set(),
                             rating=EloRating(default_params), params=default_params, seed=42)
    r2 = simulate_tournament(teams=teams, draw=draw, hosts=set(),
                             rating=EloRating(default_params), params=default_params, seed=42)
    assert r1.placements == r2.placements


# ---------------------------------------------------------------------------
# WC 2022 / 2026 PIN TESTS (Tasks 19-20)
# ---------------------------------------------------------------------------

import json
from pathlib import Path

WC2022_DRAW = {
    "A": ["QAT", "ECU", "SEN", "NED"],
    "B": ["ENG", "IRN", "USA", "WAL"],
    "C": ["ARG", "SAU", "MEX", "POL"],
    "D": ["FRA", "AUS", "DEN", "TUN"],
    "E": ["ESP", "CRC", "GER", "JPN"],
    "F": ["BEL", "CAN", "MAR", "CRO"],
    "G": ["BRA", "SRB", "SUI", "CMR"],
    "H": ["POR", "GHA", "URU", "KOR"],
}


def _load_teams_from_elo(bundled_elo_history, snapshot_date: str, iso3_set: set[str]):
    """Build a dict of {iso3: Team} from the bundled Elo CSV for teams in iso3_set."""
    from wcsim.types import Team
    from name_to_iso3 import to_iso3

    df = bundled_elo_history[bundled_elo_history["date"] == snapshot_date]
    teams = {}
    for _, row in df.iterrows():
        try:
            iso3 = to_iso3(row["team"])
        except KeyError:
            continue
        if iso3 in iso3_set:
            teams[iso3] = Team(name=row["team"], iso3=iso3, confederation="UNK", elo=float(row["rating"]))
    return teams


def test_wc_2022_tournament_pin(bundled_elo_history, default_params):
    """Pin the WC 2022 simulation at seed=42."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating

    all_iso3s = {iso3 for group in WC2022_DRAW.values() for iso3 in group}
    teams = _load_teams_from_elo(bundled_elo_history, "2022-11-19", all_iso3s)
    assert len(teams) == 32, f"Expected 32 teams, got {len(teams)}: missing {all_iso3s - set(teams)}"

    result = simulate_tournament(
        teams=teams, draw=WC2022_DRAW, hosts={"QAT"},
        rating=EloRating(default_params), params=default_params, seed=42,
    )

    champion = [iso for iso, stage in result.placements.items() if stage == "Champion"][0]
    assert champion in all_iso3s

    # Teams that made it past the group stage
    advancers = frozenset(iso for iso, stage in result.placements.items() if stage != "GroupOut")
    assert len(advancers) == 16  # top 2 per group x 8 groups

    # LOCKED (re-pinned after live_ratings plumbing into sample_match)
    assert champion == "ARG"
    assert advancers == frozenset({
        "ARG", "AUS", "BRA", "CRO", "ESP", "FRA", "GER", "IRN",
        "MAR", "NED", "POL", "POR", "SEN", "SRB", "URU", "USA",
    })


def test_wc_2026_tournament_pin(bundled_elo_history, default_params):
    """Pin the WC 2026 simulation at seed=42."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.elo import EloRating

    draw_path = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw" / "wc2026_draw.json"
    with draw_path.open() as f:
        wc2026_draw = json.load(f)

    all_iso3s = {iso3 for group in wc2026_draw.values() for iso3 in group}
    teams = _load_teams_from_elo(bundled_elo_history, "2026-06-10", all_iso3s)
    assert len(teams) == 48, f"Expected 48 teams, got {len(teams)}: missing {all_iso3s - set(teams)}"

    result = simulate_tournament(
        teams=teams, draw=wc2026_draw, hosts={"USA", "MEX", "CAN"},
        rating=EloRating(default_params), params=default_params, seed=42,
    )

    assert len(result.matches) == 103  # 72 group + 31 knockout, no 3rd place
    champion = [iso for iso, stage in result.placements.items() if stage == "Champion"][0]
    assert champion in all_iso3s

    advancers = frozenset(iso for iso, stage in result.placements.items() if stage != "GroupOut")
    assert len(advancers) == 32  # top 2 per group (24) + 8 best thirds

    # LOCKED (re-pinned after live_ratings plumbing into sample_match)
    assert champion == "ITA"
    assert advancers == frozenset({
        "ARG", "AUS", "AUT", "BRA", "CAN", "COL", "CRO",
        "ECU", "EGY", "ENG", "ESP", "FRA", "GER", "IRN", "ITA", "JOR",
        "KOR", "MAR", "MEX", "NED", "NOR", "PAN", "PAR", "POR",
        "SAU", "SCO", "SEN", "SUI", "TUN", "URU", "USA", "UZB",
    })
