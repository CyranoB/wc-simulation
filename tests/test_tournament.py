"""Tests for tournament structures and the team-count dispatcher."""
from __future__ import annotations
import pytest


def test_structure_2018_2022_dimensions():
    from wcsim.tournament import STRUCTURE_2018_2022
    s = STRUCTURE_2018_2022
    assert s.groups_count == 8
    assert s.group_size == 4
    assert s.top_per_group == 2
    assert s.best_thirds == 0
    assert s.knockout_stages == ["R16", "QF", "SF", "Final"]
    assert s.third_place_playoff is True


def test_structure_2026_dimensions():
    from wcsim.tournament import STRUCTURE_2026
    s = STRUCTURE_2026
    assert s.groups_count == 12
    assert s.group_size == 4
    assert s.top_per_group == 2
    assert s.best_thirds == 8
    assert s.knockout_stages == ["R32", "R16", "QF", "SF", "Final"]
    assert s.third_place_playoff is False


def test_structure_for_32_teams():
    from wcsim.tournament import _structure_for, STRUCTURE_2018_2022
    assert _structure_for(32) is STRUCTURE_2018_2022


def test_structure_for_48_teams():
    from wcsim.tournament import _structure_for, STRUCTURE_2026
    assert _structure_for(48) is STRUCTURE_2026


def test_structure_for_unsupported_raises():
    from wcsim.tournament import _structure_for
    with pytest.raises(ValueError, match="Unsupported tournament size"):
        _structure_for(24)
