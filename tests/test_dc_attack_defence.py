"""Tests for the spike-only Dixon-Coles attack/defence backend."""
from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SPIKE_DIR = Path(__file__).parent.parent / "spikes" / "01-validation"
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))


def test_predict_probabilities_sum_to_one():
    from dc_attack_defence import DixonColesAttackDefenseModel

    model = DixonColesAttackDefenseModel(
        teams=["A", "B"],
        attacks={"A": 0.2, "B": -0.1},
        defenses={"A": 0.1, "B": -0.2},
        log_mu=math.log(1.35),
        home_adv=0.0,
        rho=0.05,
    )

    probs = model.predict("A", "B", neutral=True)

    assert len(probs) == 3
    assert math.isclose(sum(probs), 1.0, abs_tol=1e-9)
    assert all(p > 0 for p in probs)


def test_neutral_mirrored_fixtures_are_symmetric():
    from dc_attack_defence import DixonColesAttackDefenseModel

    model = DixonColesAttackDefenseModel(
        teams=["A", "B"],
        attacks={"A": 0.25, "B": -0.15},
        defenses={"A": 0.05, "B": -0.1},
        log_mu=math.log(1.3),
        home_adv=0.2,
        rho=0.0,
    )

    ab = model.predict("A", "B", neutral=True)
    ba = model.predict("B", "A", neutral=True)

    assert math.isclose(ab[0], ba[2], abs_tol=1e-9)
    assert math.isclose(ab[1], ba[1], abs_tol=1e-9)
    assert math.isclose(ab[2], ba[0], abs_tol=1e-9)


def test_positive_rho_suppresses_low_score_draws():
    from dc_attack_defence import score_probability_grid

    base = score_probability_grid(1.2, 1.2, rho=0.0)
    adjusted = score_probability_grid(1.2, 1.2, rho=0.2)

    assert adjusted[0, 0] < base[0, 0]
    assert adjusted[1, 1] < base[1, 1]
    assert math.isclose(float(adjusted.sum()), 1.0, abs_tol=1e-9)


def test_prior_maps_use_strength_for_mapped_teams_and_neutral_for_unmapped():
    from dc_attack_defence import FitConfig, build_prior_maps

    priors = build_prior_maps(
        team_names=["Strong", "Weak", "Unknown"],
        name_to_iso3={"Strong": "STR", "Weak": "WEA"},
        elo_by_iso={"STR": 1850.0, "WEA": 1350.0},
        fifa_by_iso={"STR": 1800.0, "WEA": 1200.0},
        f0=1500.0,
        config=FitConfig(),
    )

    assert priors.attack["Strong"] > priors.attack["Weak"]
    assert priors.defense["Strong"] > priors.defense["Weak"]
    assert priors.attack["Unknown"] == 0.0
    assert priors.defense["Unknown"] == 0.0
    assert priors.sd["Unknown"] < priors.sd["Strong"]
    assert priors.mapped_teams == {"Strong", "Weak"}
    assert priors.unmapped_teams == {"Unknown"}


def test_synthetic_fit_recovers_dominant_team_profile():
    from dc_attack_defence import FitConfig, MatchRecord, build_prior_maps, fit_model

    start = date(2021, 1, 1)
    records = []
    for i in range(8):
        records.append(MatchRecord(start + timedelta(days=i), "Alpha", "Beta", 3, 0, "Friendly", True))
        records.append(MatchRecord(start + timedelta(days=20 + i), "Beta", "Alpha", 0, 2, "Friendly", True))
        records.append(MatchRecord(start + timedelta(days=40 + i), "Alpha", "Gamma", 2, 0, "Friendly", True))
        records.append(MatchRecord(start + timedelta(days=60 + i), "Gamma", "Alpha", 0, 2, "Friendly", True))

    config = FitConfig(maxiter=120)
    priors = build_prior_maps(
        ["Alpha", "Beta", "Gamma"],
        name_to_iso3={},
        elo_by_iso={},
        fifa_by_iso={},
        f0=1500.0,
        config=config,
    )

    model = fit_model(records, priors, config=config, cutoff_date=date(2022, 1, 1))

    assert model.attacks["Alpha"] > model.attacks["Beta"]
    assert model.attacks["Alpha"] > model.attacks["Gamma"]
    assert model.defenses["Alpha"] > model.defenses["Beta"]
    assert model.defenses["Alpha"] > model.defenses["Gamma"]

