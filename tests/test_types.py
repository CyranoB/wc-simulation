"""Tests for wcsim.types dataclasses."""
from __future__ import annotations

from datetime import date

import pytest


def test_team_is_frozen_dataclass():
    from wcsim.types import Team
    t = Team(name="Brazil", iso3="BRA", confederation="CONMEBOL", elo=2141.0)
    with pytest.raises(Exception):
        t.elo = 1.0


def test_team_fifa_fields_default_to_none():
    from wcsim.types import Team
    t = Team(name="Brazil", iso3="BRA", confederation="CONMEBOL", elo=2141.0)
    assert t.fifa_points is None
    assert t.fifa_rank is None
    assert t.fifa_updated is None
    assert t.elo_updated is None


def test_team_with_full_fields(sample_team_brazil):
    assert sample_team_brazil.iso3 == "BRA"
    assert sample_team_brazil.elo == 2141.0
    assert sample_team_brazil.fifa_points == 1431.0
    assert sample_team_brazil.fifa_updated == date(2018, 6, 7)


def test_match_result_required_fields():
    from wcsim.types import MatchResult
    m = MatchResult(
        home="ARG", away="FRA",
        home_goals=3, away_goals=3,
        stage="Final", neutral=True,
        extra_time=True, went_to_pens=True, pen_winner="ARG",
        home_rating_before=2143.0, away_rating_before=2004.0,
    )
    assert m.pen_winner == "ARG"
    assert m.extra_time is True


def test_tournament_result_fields():
    from wcsim.types import TournamentResult, MatchResult
    r = TournamentResult(
        seed=42, rating_mode="elo",
        matches=[],
        placements={"ARG": "Champion", "FRA": "Final"},
        final_ratings={"ARG": 2167.5},
    )
    assert r.seed == 42
    assert r.placements["ARG"] == "Champion"


def test_params_defaults_match_prd_v17():
    from wcsim.types import Params
    p = Params()
    assert p.c_elo == 300.0
    assert p.c_fifa == 450.0
    assert p.mu == 1.35
    assert p.lambda_min == 0.05
    assert p.blend_w == 0.7
    assert p.e0 == 1500.0
    assert p.f0 == 1300.0
    assert p.home_bonus_elo == 100.0
    assert p.home_bonus_fifa == 150.0
    assert p.rho == 0.0
    assert p.k_elo == 60.0
    assert p.k_fifa == 60.0


def test_params_is_frozen():
    from wcsim.types import Params
    import pytest
    p = Params()
    with pytest.raises(Exception):
        p.mu = 2.0


def test_public_api_reexports():
    """wcsim.X imports work for the documented public surface."""
    import wcsim
    assert hasattr(wcsim, "Team")
    assert hasattr(wcsim, "MatchResult")
    assert hasattr(wcsim, "TournamentResult")
    assert hasattr(wcsim, "Params")
    assert hasattr(wcsim, "EloRating")
    assert hasattr(wcsim, "FifaRating")
    assert hasattr(wcsim, "BlendRating")
    assert hasattr(wcsim, "predict_match")
    assert hasattr(wcsim, "sample_match")
    assert hasattr(wcsim, "simulate_tournament")
