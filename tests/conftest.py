"""Shared pytest fixtures for the wcsim library suite."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
SPIKE_DIR = REPO_ROOT / "spikes" / "01-validation"
DATA_DIR = SPIKE_DIR / "data" / "raw"


@pytest.fixture(scope="session", autouse=True)
def _setup_paths():
    for p in (REPO_ROOT, SPIKE_DIR):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    yield


@pytest.fixture
def sample_team_brazil():
    from wcsim.types import Team
    return Team(
        name="Brazil", iso3="BRA", confederation="CONMEBOL",
        elo=2141.0, elo_updated=date(2018, 6, 13),
        fifa_points=1431.0, fifa_rank=2, fifa_updated=date(2018, 6, 7),
    )


@pytest.fixture
def sample_team_france():
    from wcsim.types import Team
    return Team(
        name="France", iso3="FRA", confederation="UEFA",
        elo=1986.0, elo_updated=date(2018, 6, 13),
        fifa_points=1198.0, fifa_rank=7, fifa_updated=date(2018, 6, 7),
    )


@pytest.fixture
def default_params():
    from wcsim.types import Params
    return Params()


@pytest.fixture
def bundled_elo_history():
    return pd.read_csv(DATA_DIR / "elo_history.csv")


@pytest.fixture
def bundled_fifa_ranking():
    return pd.read_csv(DATA_DIR / "fifa_ranking.csv")


@pytest.fixture
def bundled_matches():
    return pd.read_csv(DATA_DIR / "matches_history.csv")
