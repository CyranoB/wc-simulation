"""Tests for the match model: Poisson PMF, Dixon-Coles τ, predict_match."""
from __future__ import annotations
import math
import numpy as np


def test_poisson_pmf_sums_to_one_for_large_grid():
    from wcsim.model import _poisson_pmf
    pmf = _poisson_pmf(1.35, 20)
    assert math.isclose(pmf.sum(), 1.0, abs_tol=1e-6)


def test_poisson_pmf_correct_for_lambda_zero():
    from wcsim.model import _poisson_pmf
    pmf = _poisson_pmf(0.0, 8)
    assert math.isclose(pmf[0], 1.0, abs_tol=1e-12)
    assert pmf[1:].sum() == 0.0


def test_apply_tau_no_op_at_rho_zero():
    from wcsim.model import _apply_tau
    grid_before = np.ones((9, 9)) / 81.0
    grid_after = _apply_tau(grid_before.copy(), 1.35, 1.35, 0.0)
    np.testing.assert_array_almost_equal(grid_before, grid_after)


def test_apply_tau_positive_rho_suppresses_low_score_draws():
    from wcsim.model import _apply_tau
    grid = np.ones((9, 9)) / 81.0
    adjusted = _apply_tau(grid.copy(), 1.35, 1.35, 0.2)
    assert adjusted[0, 0] < grid[0, 0]
    assert adjusted[1, 1] < grid[1, 1]


def test_predict_match_returns_three_probs_summing_to_one(sample_team_brazil, sample_team_france):
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = predict_match(sample_team_brazil, sample_team_france, rating=EloRating(Params()))
    assert len(p) == 3
    assert math.isclose(sum(p), 1.0, abs_tol=1e-9)


def test_predict_match_is_symmetric(sample_team_brazil, sample_team_france):
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    p = predict_match(sample_team_brazil, sample_team_france, rating=EloRating(Params()))
    q = predict_match(sample_team_france, sample_team_brazil, rating=EloRating(Params()))
    assert math.isclose(p[0], q[2], abs_tol=1e-12)
    assert math.isclose(p[1], q[1], abs_tol=1e-12)
    assert math.isclose(p[2], q[0], abs_tol=1e-12)


def test_predict_match_with_rho_changes_outputs(sample_team_brazil, sample_team_france):
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    base = predict_match(sample_team_brazil, sample_team_france, rating=EloRating(Params(rho=0.0)))
    with_rho = predict_match(sample_team_brazil, sample_team_france, rating=EloRating(Params(rho=0.2)), params=Params(rho=0.2))
    assert not math.isclose(base[1], with_rho[1], abs_tol=1e-6)


def test_predict_match_host_bonus_helps_home(sample_team_brazil, sample_team_france):
    from wcsim.model import predict_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    neutral = predict_match(sample_team_france, sample_team_brazil, rating=EloRating(Params()))
    home = predict_match(sample_team_france, sample_team_brazil, rating=EloRating(Params()), a_is_host=True)
    assert home[0] > neutral[0]


def test_sample_match_returns_match_result(sample_team_brazil, sample_team_france):
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params, MatchResult
    import numpy as np
    rng = np.random.default_rng(42)
    m = sample_match(sample_team_brazil, sample_team_france, rating=EloRating(Params()), rng=rng, stage="group")
    assert isinstance(m, MatchResult)
    assert m.home == "BRA"
    assert m.away == "FRA"
    assert m.stage == "group"
    assert m.extra_time is False
    assert m.went_to_pens is False
    assert m.pen_winner is None
    assert isinstance(m.home_goals, int)
    assert isinstance(m.away_goals, int)
    assert m.home_rating_before == 2141.0
    assert m.away_rating_before == 1986.0


def test_sample_match_is_deterministic_with_seed(sample_team_brazil, sample_team_france):
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    import numpy as np
    def run(seed):
        rng = np.random.default_rng(seed)
        return sample_match(sample_team_brazil, sample_team_france, rating=EloRating(Params()), rng=rng, stage="group")
    assert run(123).home_goals == run(123).home_goals
    assert run(123).away_goals == run(123).away_goals


def test_sample_match_knockout_handles_extra_time_when_tied(sample_team_brazil, sample_team_france):
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    import numpy as np
    saw_et = False
    for seed in range(100):
        rng = np.random.default_rng(seed)
        m = sample_match(sample_team_brazil, sample_team_france, rating=EloRating(Params()), rng=rng, stage="R16")
        if m.extra_time:
            saw_et = True
            assert (m.home_goals != m.away_goals) or m.went_to_pens
            break
    assert saw_et, "no extra-time match found in 100 seeds"


def test_sample_match_pen_winner_is_one_of_teams(sample_team_brazil, sample_team_france):
    from wcsim.model import sample_match
    from wcsim.ratings.elo import EloRating
    from wcsim.types import Params
    import numpy as np
    for seed in range(300):
        rng = np.random.default_rng(seed)
        m = sample_match(sample_team_brazil, sample_team_france, rating=EloRating(Params()), rng=rng, stage="Final")
        if m.went_to_pens:
            assert m.pen_winner in {"BRA", "FRA"}
            break
