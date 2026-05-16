"""Tests for report formatters and Wilson CI."""
import math


def test_wilson_ci_at_p_half_n_100():
    from wcsim.report import wilson_ci
    lo, hi = wilson_ci(0.5, 100)
    assert 0.40 < lo < 0.42
    assert 0.58 < hi < 0.60


def test_wilson_ci_at_p_zero():
    from wcsim.report import wilson_ci
    lo, hi = wilson_ci(0.0, 1000)
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_ci_at_p_one():
    from wcsim.report import wilson_ci
    lo, hi = wilson_ci(1.0, 1000)
    assert lo < 1.0
    assert hi == 1.0


def test_format_csv_produces_header_and_rows():
    from wcsim.report import format_csv
    from wcsim.types import SimulationResult
    result = SimulationResult(
        n=100, seed=42,
        probabilities={"ARG": {"Win": 0.28, "GroupOut": 0.05}},
        ci_lo={"ARG": {"Win": 0.20, "GroupOut": 0.02}},
        ci_hi={"ARG": {"Win": 0.36, "GroupOut": 0.10}},
        mean_goals_for={"ARG": 2.1},
        mean_goals_against={"ARG": 0.8},
    )
    csv_str = format_csv(result)
    assert "team" in csv_str.split("\n")[0]
    assert "ARG" in csv_str


def test_format_table_returns_string_with_percentages():
    from wcsim.report import format_table
    from wcsim.types import SimulationResult
    result = SimulationResult(
        n=100, seed=42,
        probabilities={"ARG": {"Win": 0.28, "GroupOut": 0.05}},
        ci_lo={"ARG": {"Win": 0.20, "GroupOut": 0.02}},
        ci_hi={"ARG": {"Win": 0.36, "GroupOut": 0.10}},
        mean_goals_for={"ARG": 2.1},
        mean_goals_against={"ARG": 0.8},
    )
    table = format_table(result)
    assert "ARG" in table
    assert "28" in table
