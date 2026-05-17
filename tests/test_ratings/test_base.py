"""Tests for the RatingSystem Protocol."""
from __future__ import annotations


def test_rating_system_is_a_protocol():
    from wcsim.ratings.base import RatingSystem
    assert hasattr(RatingSystem, "__protocol_attrs__") or hasattr(
        RatingSystem, "_is_protocol"
    )


def test_rating_system_required_methods():
    from wcsim.ratings.base import RatingSystem
    for method_name in (
        "rating_of", "rating_diff", "win_expectation", "lambdas", "update",
    ):
        assert hasattr(RatingSystem, method_name), f"missing {method_name}"


def test_rating_system_required_attributes():
    from wcsim.ratings.base import RatingSystem
    hints = RatingSystem.__annotations__
    for attr in ("name", "scale", "c", "home_bonus"):
        assert attr in hints, f"missing attribute {attr}"
