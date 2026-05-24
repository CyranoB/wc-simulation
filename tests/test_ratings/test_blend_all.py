"""Tests for BlendAllRating (three-way Elo + FIFA + Player)."""
from __future__ import annotations

import pytest

from wcsim.ratings.blend import BlendRating
from wcsim.ratings.blend_all import BlendAllRating
from wcsim.ratings.player import PlayerRating
from wcsim.types import Params, Team


@pytest.fixture
def squad_data():
    return {
        "BRA": {"total_value_eur": 1_200_000_000},
        "FRA": {"total_value_eur": 1_500_000_000},
    }


@pytest.fixture
def team_brazil():
    return Team(name="Brazil", iso3="BRA", confederation="CONMEBOL",
                elo=2000.0, fifa_points=1800.0, fifa_rank=3)


@pytest.fixture
def team_france():
    return Team(name="France", iso3="FRA", confederation="UEFA",
                elo=1950.0, fifa_points=1850.0, fifa_rank=2)


def test_blend_all_at_zero_player_weight_matches_blend(squad_data, team_brazil):
    params = Params(blend_player=0.0)
    blend_all = BlendAllRating(params, squad_data)
    blend = BlendRating(params)
    assert abs(blend_all.rating_of(team_brazil) - blend.rating_of(team_brazil)) < 1e-10


def test_blend_all_at_full_player_weight_matches_player(squad_data, team_brazil):
    params = Params(blend_player=1.0)
    blend_all = BlendAllRating(params, squad_data)
    player = PlayerRating(params, squad_data)
    assert abs(blend_all.rating_of(team_brazil) - player.rating_of(team_brazil)) < 1e-10


def test_blend_all_intermediate_weight(squad_data, team_brazil):
    params = Params(blend_player=0.3)
    blend_all = BlendAllRating(params, squad_data)
    blend = BlendRating(params)
    player = PlayerRating(params, squad_data)
    expected = 0.3 * player.rating_of(team_brazil) + 0.7 * blend.rating_of(team_brazil)
    assert abs(blend_all.rating_of(team_brazil) - expected) < 1e-10


def test_blend_all_update_returns_changed_value(squad_data):
    params = Params(blend_player=0.3)
    blend_all = BlendAllRating(params, squad_data)
    updated = blend_all.update(1600.0, 0.6, 3, 0)
    assert updated != 1600.0


def test_blend_all_attributes(squad_data):
    params = Params(blend_player=0.3)
    blend_all = BlendAllRating(params, squad_data)
    assert blend_all.name == "blend_all"
    assert blend_all.scale == 400.0
    assert blend_all.c == 300.0
    assert blend_all.home_bonus == 100.0


def test_blend_all_rating_diff_includes_home_bonus(squad_data, team_brazil, team_france):
    params = Params(blend_player=0.3)
    blend_all = BlendAllRating(params, squad_data)
    neutral = blend_all.rating_diff(team_brazil, team_france, False, False)
    home = blend_all.rating_diff(team_brazil, team_france, True, False)
    assert abs(home - neutral - 100.0) < 1e-10
