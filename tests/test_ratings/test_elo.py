"""Tests for EloRating. Math cross-checked against PRD §5.5."""
from __future__ import annotations

import math


def test_elo_attributes():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert r.name == "elo"
    assert r.scale == 400.0
    assert r.c == 300.0
    assert r.home_bonus == 100.0


def test_rating_of_returns_team_elo(sample_team_brazil):
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert r.rating_of(sample_team_brazil) == 2141.0


def test_rating_diff_neutral_venue(sample_team_brazil, sample_team_france):
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    diff = r.rating_diff(sample_team_brazil, sample_team_france, a_is_host=False, b_is_host=False)
    assert diff == 2141.0 - 1986.0


def test_rating_diff_with_a_host(sample_team_brazil, sample_team_france):
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    diff = r.rating_diff(sample_team_brazil, sample_team_france, a_is_host=True, b_is_host=False)
    assert diff == 2141.0 + 100.0 - 1986.0


def test_win_expectation_neutral_zero_diff():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert math.isclose(r.win_expectation(0.0), 0.5, abs_tol=1e-12)


def test_win_expectation_400_point_advantage():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    r = EloRating(Params())
    assert math.isclose(r.win_expectation(400.0), 1.0 / (1.0 + 10**-1.0), abs_tol=1e-12)


def test_lambdas_zero_diff_returns_mu():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    lam_a, lam_b = r.lambdas(0.0, mu=p.mu, lambda_min=p.lambda_min)
    assert lam_a == p.mu
    assert lam_b == p.mu


def test_lambdas_positive_diff():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    lam_a, lam_b = r.lambdas(300.0, mu=p.mu, lambda_min=p.lambda_min)
    assert math.isclose(lam_a, p.mu + 0.5, abs_tol=1e-12)
    assert math.isclose(lam_b, p.mu - 0.5, abs_tol=1e-12)


def test_lambdas_floor_applied():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    lam_a, lam_b = r.lambdas(-2000.0, mu=p.mu, lambda_min=p.lambda_min)
    assert lam_a == p.lambda_min


def test_update_draw_1_1():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    new_rating = r.update(before=2000.0, expected=0.5, score_home=1, score_away=1)
    assert math.isclose(new_rating, 2000.0 + 60.0 * 1.0 * (0.5 - 0.5), abs_tol=1e-12)


def test_update_2_goal_win_uses_gm_1_5():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    new_rating = r.update(before=2000.0, expected=0.6, score_home=2, score_away=0)
    assert math.isclose(new_rating, 2000.0 + 60.0 * 1.5 * 0.4, abs_tol=1e-12)


def test_update_3_goal_margin_uses_gm_formula():
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = Params()
    r = EloRating(p)
    new_rating = r.update(before=2000.0, expected=0.6, score_home=3, score_away=0)
    assert math.isclose(new_rating, 2000.0 + 60.0 * 1.75 * 0.4, abs_tol=1e-12)
