"""Integration tests for the Dixon-Coles validation spike."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SPIKE_DIR = Path(__file__).parent.parent / "spikes" / "01-validation"
if str(SPIKE_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_DIR))


def test_load_international_results_accepts_martj42_fixture(tmp_path):
    from validate_dc import load_international_results

    fixture = tmp_path / "results.csv"
    fixture.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2021-01-01,Alpha,Beta,2,1,Friendly,Paris,France,TRUE\n"
        "2021-02-01,Beta,Alpha,0,0,FIFA World Cup qualification,Rome,Italy,FALSE\n"
    )

    df = load_international_results(fixture)

    assert list(df.columns) == [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ]
    assert len(df) == 2
    assert bool(df.iloc[0]["neutral"]) is True
    assert bool(df.iloc[1]["neutral"]) is False


def test_run_validation_writes_required_json_keys(tmp_path):
    from validate_dc import run_validation

    train_data = tmp_path / "international_results.csv"
    train_data.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2019-03-01,Brazil,Argentina,2,0,Friendly,Madrid,Spain,TRUE\n"
        "2019-06-01,France,Argentina,1,1,Friendly,Paris,France,FALSE\n"
        "2020-03-01,Spain,Germany,2,1,UEFA Nations League,Madrid,Spain,FALSE\n"
        "2020-09-01,Brazil,France,1,0,Friendly,Rome,Italy,TRUE\n"
        "2021-03-01,Argentina,France,0,2,Friendly,Rome,Italy,TRUE\n"
        "2021-09-01,Germany,Brazil,1,3,Friendly,Berlin,Germany,FALSE\n"
        "2022-03-01,Spain,Argentina,2,0,Friendly,Lisbon,Portugal,TRUE\n"
        "2022-09-01,France,Brazil,1,1,Friendly,Lisbon,Portugal,TRUE\n"
    )

    result = run_validation(
        years=[2022],
        train_data_path=train_data,
        out_dir=tmp_path,
        max_train_matches=8,
        forecast_2026=False,
    )

    out_path = tmp_path / "dc_attack_defence.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["decision"] in {"promote_candidate", "keep_as_experiment"}
    assert "rps_90min" in written
    assert "draw_rate" in written
    assert "calibration" in written
    assert "blend_baseline" in written
    assert "fit" in written
    assert result["years"]["2022"]["training_rows"] == 8
