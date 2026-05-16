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
