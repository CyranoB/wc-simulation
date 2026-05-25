"""wcsim — Football Tournament Monte Carlo Simulator (library).

Spike 2 deliverable: validated match math + tournament simulation.
CLI, Monte Carlo runner, and scrapers are deferred to later spikes.
"""
from __future__ import annotations

from .model import predict_match, sample_match
from .ratings.blend import BlendRating
from .ratings.blend_all import BlendAllRating
from .ratings.elo import EloRating
from .ratings.fifa import FifaRating
from .ratings.player import PlayerRating
from .sim import run_simulations
from .tournament import simulate_tournament
from .types import MatchResult, Params, SimulationResult, Team, TournamentResult

__all__ = [
    "Team", "MatchResult", "TournamentResult", "Params", "SimulationResult",
    "EloRating", "FifaRating", "BlendRating", "PlayerRating", "BlendAllRating",
    "predict_match", "sample_match", "simulate_tournament",
    "run_simulations",
]
