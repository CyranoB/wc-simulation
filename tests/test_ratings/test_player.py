"""Tests for PlayerRating."""
from __future__ import annotations

import math
import statistics

import pytest

from wcsim.ratings.player import PlayerRating
from wcsim.types import Params, Team


@pytest.fixture
def three_team_squads():
    return {
        "AAA": {"total_value_eur": 1_000_000_000},
        "BBB": {"total_value_eur": 100_000_000},
        "CCC": {"total_value_eur": 10_000_000},
    }


@pytest.fixture
def player_rating(three_team_squads):
    return PlayerRating(Params(), three_team_squads)


def test_player_rating_attributes(player_rating):
    assert player_rating.name == "player"
    assert player_rating.scale == 400.0
    assert player_rating.c == 300.0
    assert player_rating.home_bonus == 100.0


def test_player_rating_of_returns_normalized_value(three_team_squads):
    pr = PlayerRating(Params(), three_team_squads)
    log_vals = [math.log10(v) for v in [1e9, 1e8, 1e7]]
    log_mean = statistics.mean(log_vals)
    log_std = statistics.stdev(log_vals)
    alpha = 190.0 / log_std

    expected_aaa = 1500.0 + alpha * (math.log10(1e9) - log_mean)
    expected_ccc = 1500.0 + alpha * (math.log10(1e7) - log_mean)

    team_a = Team(name="A", iso3="AAA", confederation="X", elo=1500.0)
    team_c = Team(name="C", iso3="CCC", confederation="X", elo=1500.0)
    assert abs(pr.rating_of(team_a) - expected_aaa) < 1e-10
    assert abs(pr.rating_of(team_c) - expected_ccc) < 1e-10


def test_player_rating_of_missing_squad_raises(player_rating):
    team = Team(name="X", iso3="ZZZ", confederation="X", elo=1500.0)
    with pytest.raises(ValueError, match="no squad data"):
        player_rating.rating_of(team)


def test_player_rating_spread_matches_target(three_team_squads):
    pr = PlayerRating(Params(), three_team_squads)
    teams = [
        Team(name="A", iso3="AAA", confederation="X", elo=1500.0),
        Team(name="B", iso3="BBB", confederation="X", elo=1500.0),
        Team(name="C", iso3="CCC", confederation="X", elo=1500.0),
    ]
    ratings = [pr.rating_of(t) for t in teams]
    std = statistics.stdev(ratings)
    assert abs(std - 190.0) < 1.0


def test_player_update_is_no_op(player_rating):
    assert player_rating.update(1600.0, 0.6, 3, 1) == 1600.0
    assert player_rating.update(1400.0, 0.4, 0, 2) == 1400.0


def test_player_rating_diff_includes_home_bonus(player_rating):
    team_a = Team(name="A", iso3="AAA", confederation="X", elo=1500.0)
    team_b = Team(name="B", iso3="BBB", confederation="X", elo=1500.0)
    neutral = player_rating.rating_diff(team_a, team_b, False, False)
    home_a = player_rating.rating_diff(team_a, team_b, True, False)
    assert abs(home_a - neutral - 100.0) < 1e-10


def test_player_win_expectation_symmetric(player_rating):
    assert abs(player_rating.win_expectation(0.0) - 0.5) < 1e-10
    we_pos = player_rating.win_expectation(100.0)
    we_neg = player_rating.win_expectation(-100.0)
    assert abs(we_pos + we_neg - 1.0) < 1e-10


def test_player_lambdas_positive(player_rating):
    lam_a, lam_b = player_rating.lambdas(200.0, 1.35, 0.05)
    assert lam_a > lam_b > 0.05


def test_simulate_tournament_with_player_rating():
    """End-to-end smoke test: player rating produces a champion."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.player import PlayerRating

    squad_data = {f"T{i:02d}": {"total_value_eur": (50 - i) * 10_000_000}
                  for i in range(32)}
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="X", elo=1500.0)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(8)}

    pr = PlayerRating(Params(), squad_data)
    result = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=pr, params=Params(), seed=42,
    )
    champions = [iso for iso, stage in result.placements.items() if stage == "Champion"]
    assert len(champions) == 1
    assert len(result.matches) == 64


def test_player_final_ratings_unchanged_after_tournament():
    """PlayerRating.update() is a no-op, so final_ratings == initial ratings."""
    from wcsim.tournament import simulate_tournament
    from wcsim.ratings.player import PlayerRating

    squad_data = {f"T{i:02d}": {"total_value_eur": (50 - i) * 10_000_000}
                  for i in range(32)}
    teams = {f"T{i:02d}": Team(name=f"T{i:02d}", iso3=f"T{i:02d}",
                               confederation="X", elo=1500.0)
             for i in range(32)}
    draw = {chr(ord("A") + g): [f"T{g*4+j:02d}" for j in range(4)]
            for g in range(8)}

    pr = PlayerRating(Params(), squad_data)
    initial = {iso3: pr.rating_of(teams[iso3]) for iso3 in teams}
    result = simulate_tournament(
        teams=teams, draw=draw, hosts=set(),
        rating=pr, params=Params(), seed=42,
    )
    for iso3, final_r in result.final_ratings.items():
        assert abs(final_r - initial[iso3]) < 1e-10
