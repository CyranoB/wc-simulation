"""Tests for FifaRating. FIFA update: R' = R + I * (W - W_e), no goal-margin multiplier."""
from __future__ import annotations

import math

import pytest


def test_fifa_attributes():
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    assert r.name == "fifa"
    assert r.scale == 600.0
    assert r.c == 450.0
    assert r.home_bonus == 150.0


def test_rating_of_returns_fifa_points(sample_team_brazil):
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    assert r.rating_of(sample_team_brazil) == 1431.0


def test_rating_of_raises_if_fifa_missing():
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params, Team
    elo_only = Team(name="X", iso3="XXX", confederation="UEFA", elo=1500.0)
    r = FifaRating(Params())
    with pytest.raises(ValueError, match="fifa_points"):
        r.rating_of(elo_only)


def test_rating_diff_with_host_uses_fifa_bonus(sample_team_brazil, sample_team_france):
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    diff = r.rating_diff(sample_team_brazil, sample_team_france, a_is_host=True, b_is_host=False)
    assert diff == (1431.0 + 150.0) - 1198.0


def test_win_expectation_uses_scale_600():
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    r = FifaRating(Params())
    assert math.isclose(r.win_expectation(600.0), 1.0 / (1.0 + 10**-1.0), abs_tol=1e-12)


def test_lambdas_uses_c_450():
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    p = Params()
    r = FifaRating(p)
    lam_a, lam_b = r.lambdas(450.0, mu=p.mu, lambda_min=p.lambda_min)
    assert math.isclose(lam_a, p.mu + 0.5, abs_tol=1e-12)
    assert math.isclose(lam_b, p.mu - 0.5, abs_tol=1e-12)


def test_update_uses_I_no_goal_margin():
    from wcsim.ratings.fifa import FifaRating
    from wcsim.types import Params
    p = Params()
    r = FifaRating(p)
    new_rating = r.update(before=1500.0, expected=0.6, score_home=3, score_away=0)
    assert math.isclose(new_rating, 1500.0 + 60.0 * 0.4, abs_tol=1e-12)
