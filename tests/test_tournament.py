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
    matches, positions = simulate_group_stage(
        teams=teams, draw=draw,
        rating=EloRating(default_params), params=default_params,
        rng=rng, hosts=set(),
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
        return simulate_group_stage(
            teams=teams, draw=draw,
            rating=EloRating(default_params), params=default_params,
            rng=rng, hosts=set(),
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
