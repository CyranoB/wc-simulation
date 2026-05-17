"""Tests for BlendRating. Blended rating = w*Elo + (1-w)*FIFA*E0/F0, in Elo space."""
from __future__ import annotations
import math


def test_blend_attributes():
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    assert r.name == "blend"
    assert r.scale == 400.0
    assert r.c == 300.0
    assert r.home_bonus == 100.0


def test_rating_of_combines_elo_and_fifa(sample_team_brazil):
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    expected = 0.7 * 2141.0 + 0.3 * 1431.0 * 1500.0 / 1300.0
    assert math.isclose(r.rating_of(sample_team_brazil), expected, abs_tol=1e-9)


def test_rating_diff_uses_blended_ratings(sample_team_brazil, sample_team_france):
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    diff = r.rating_diff(sample_team_brazil, sample_team_france, a_is_host=False, b_is_host=False)
    assert math.isclose(diff, r.rating_of(sample_team_brazil) - r.rating_of(sample_team_france), abs_tol=1e-12)


def test_win_expectation_uses_elo_scale():
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    r = BlendRating(Params())
    assert math.isclose(r.win_expectation(400.0), 1.0 / (1.0 + 10**-1.0), abs_tol=1e-12)


def test_lambdas_uses_c_elo():
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    p = Params()
    r = BlendRating(p)
    lam_a, lam_b = r.lambdas(300.0, mu=p.mu, lambda_min=p.lambda_min)
    assert math.isclose(lam_a - lam_b, 1.0, abs_tol=1e-12)


def test_update_returns_changed_rating():
    from wcsim.ratings.blend import BlendRating
    from wcsim.types import Params
    p = Params()
    r = BlendRating(p)
    new_rating = r.update(before=2000.0, expected=0.6, score_home=2, score_away=0)
    assert isinstance(new_rating, float)
    assert new_rating != 2000.0
