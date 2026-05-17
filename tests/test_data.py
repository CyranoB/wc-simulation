"""Tests for data loaders."""
from pathlib import Path
import pytest

SPIKE_DATA = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw"


def test_load_teams_returns_dict_of_teams():
    from wcsim.data import load_teams
    teams = load_teams(SPIKE_DATA / "elo_history.csv", snapshot_date="2026-06-10")
    assert len(teams) == 48
    assert "ARG" in teams
    assert teams["ARG"].elo > 2000


def test_load_draw_returns_12_groups():
    from wcsim.data import load_draw
    draw = load_draw(SPIKE_DATA / "wc2026_draw.json")
    assert len(draw) == 12
    assert all(len(v) == 4 for v in draw.values())
    assert "ARG" in draw["J"]


def test_load_teams_raises_on_missing_file():
    from wcsim.data import load_teams
    with pytest.raises(FileNotFoundError):
        load_teams(Path("/nonexistent.csv"), snapshot_date="2026-06-10")
