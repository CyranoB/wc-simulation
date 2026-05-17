"""Regression: wcsim.predict_match must reproduce validate.predict to 1e-9
across all 128 WC 2018+2022 matches for Elo, FIFA, and Blend modes."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def validate_module():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "spikes" / "01-validation"))
    try:
        import validate
    except (ImportError, ModuleNotFoundError) as e:
        pytest.skip(f"validate module unavailable: {e}")
    return validate


@pytest.fixture(scope="module")
def loaded_data(validate_module):
    v = validate_module
    matches_2018 = v.load_matches(2018)
    matches_2022 = v.load_matches(2022)
    elo_2018 = v.load_elo(v.WC2018[0])
    elo_2022 = v.load_elo(v.WC2022[0])
    fifa_2018, _ = v.load_fifa(v.WC2018[0])
    fifa_2022, _ = v.load_fifa(v.WC2022[0])
    f0_2018 = float(np.mean(list(fifa_2018.values())))
    f0_2022 = float(np.mean(list(fifa_2022.values())))
    return {
        "matches_2018": matches_2018, "matches_2022": matches_2022,
        "elo_2018": elo_2018, "elo_2022": elo_2022,
        "fifa_2018": fifa_2018, "fifa_2022": fifa_2022,
        "f0_2018": f0_2018, "f0_2022": f0_2022,
    }


def _team_from_dicts(iso3, elo_d, fifa_d):
    from wcsim.types import Team
    return Team(
        name=iso3, iso3=iso3, confederation="UNK",
        elo=elo_d[iso3],
        fifa_points=fifa_d.get(iso3),
    )


@pytest.mark.parametrize("mode", ["elo", "fifa", "blend"])
def test_predict_match_matches_validate_py(loaded_data, validate_module, mode):
    from wcsim.model import predict_match
    from wcsim.types import Params
    from wcsim.ratings.elo import EloRating
    from wcsim.ratings.fifa import FifaRating
    from wcsim.ratings.blend import BlendRating

    rating_cls = {"elo": EloRating, "fifa": FifaRating, "blend": BlendRating}[mode]

    for year in (2018, 2022):
        key = str(year)
        elo_d = loaded_data[f"elo_{key}"]
        fifa_d = loaded_data[f"fifa_{key}"]
        f0 = loaded_data[f"f0_{key}"]
        params = Params(f0=f0)
        rating = rating_cls(params)
        hosts = validate_module.HOST_BY_YEAR[year]

        # Ensure validate.py uses the same params
        validate_module.PARAMS["rho"] = params.rho
        validate_module.PARAMS["c_elo"] = params.c_elo
        validate_module.PARAMS["c_fifa"] = params.c_fifa
        validate_module.PARAMS["mu"] = params.mu
        validate_module.PARAMS["lambda_min"] = params.lambda_min
        validate_module.PARAMS["blend_w"] = params.blend_w
        validate_module.PARAMS["e0"] = params.e0
        validate_module.PARAMS["home_bonus_elo"] = params.home_bonus_elo
        validate_module.PARAMS["home_bonus_fifa"] = params.home_bonus_fifa

        for m in loaded_data[f"matches_{key}"]:
            a_iso3, b_iso3 = m["home_iso3"], m["away_iso3"]

            # Skip teams missing from FIFA dict (Elo-only participants)
            if mode in ("fifa", "blend") and (a_iso3 not in fifa_d or b_iso3 not in fifa_d):
                continue

            team_a = _team_from_dicts(a_iso3, elo_d, fifa_d)
            team_b = _team_from_dicts(b_iso3, elo_d, fifa_d)

            wcsim_probs = predict_match(
                team_a, team_b, rating=rating, params=params,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
            )
            validate_probs = validate_module.predict(
                elo_a=elo_d[a_iso3], elo_b=elo_d[b_iso3],
                fifa_a=fifa_d.get(a_iso3, 0.0), fifa_b=fifa_d.get(b_iso3, 0.0),
                f0=f0,
                a_is_host=(a_iso3 in hosts), b_is_host=(b_iso3 in hosts),
                mode=mode,
            )
            np.testing.assert_allclose(
                wcsim_probs, validate_probs, atol=1e-9,
                err_msg=f"{mode} mismatch on {year} {a_iso3} vs {b_iso3}",
            )
