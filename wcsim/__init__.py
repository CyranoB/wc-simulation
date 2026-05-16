"""wcsim — Football Tournament Monte Carlo Simulator (library).

Spike 2 deliverable: validated match math + tournament simulation.
CLI, Monte Carlo runner, and scrapers are deferred to later spikes.
"""
from __future__ import annotations

from .types import Team, MatchResult, TournamentResult, Params
from .ratings.elo import EloRating
from .ratings.fifa import FifaRating
from .ratings.blend import BlendRating
from .model import predict_match, sample_match
from .tournament import simulate_tournament

__all__ = [
    "Team", "MatchResult", "TournamentResult", "Params",
    "EloRating", "FifaRating", "BlendRating",
    "predict_match", "sample_match", "simulate_tournament",
]
