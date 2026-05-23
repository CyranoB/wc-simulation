"""Validate the spike-only Dixon-Coles attack/defence backend."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import validate
from dc_attack_defence import (
    DixonColesAttackDefenseModel,
    FitConfig,
    MatchRecord,
    PriorMaps,
    build_prior_maps,
    fit_model,
)
from name_to_iso3 import NAME_TO_ISO3

HERE = Path(__file__).parent
DATA = HERE / "data" / "raw"
RESULTS = HERE / "results"
DEFAULT_TRAIN_DATA = DATA / "international_results.csv"
EXPECTED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "city", "country", "neutral",
]
TRAIN_WINDOWS = {
    2018: (date(2014, 6, 13), date(2018, 6, 13)),
    2022: (date(2018, 11, 19), date(2022, 11, 19)),
}
SNAPSHOT_BY_YEAR = {2018: validate.WC2018[0], 2022: validate.WC2022[0]}


def load_international_results(path: Path) -> pd.DataFrame:
    """Load martj42-shaped international results with normalized dtypes."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run spikes/01-validation/scrapers/international_results.py first."
        )
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected international-results schema: {list(df.columns)!r}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    before = len(df)
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map(lambda v: str(v).strip().lower() == "true")
    df.attrs["skipped_rows"] = before - len(df)
    return df


def _records_from_frame(df: pd.DataFrame) -> list[MatchRecord]:
    return [
        MatchRecord(
            date=row["date"].date(),
            home_team=str(row["home_team"]),
            away_team=str(row["away_team"]),
            home_goals=int(row["home_score"]),
            away_goals=int(row["away_score"]),
            tournament=str(row["tournament"]),
            neutral=bool(row["neutral"]),
        )
        for _, row in df.iterrows()
    ]


def _training_frame(
    df: pd.DataFrame, year: int, max_train_matches: int | None,
) -> tuple[pd.DataFrame, date]:
    start, cutoff = TRAIN_WINDOWS[year]
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(cutoff))
    out = df[mask].sort_values("date")
    if max_train_matches is not None:
        out = out.tail(max_train_matches)
    return out, cutoff


def _eval_matches_with_names(year: int) -> list[dict]:
    raw = pd.read_csv(DATA / "matches_history.csv")
    raw["date"] = pd.to_datetime(raw["date"])
    if year == 2018:
        frame = raw[raw["date"].between("2018-06-14", "2018-07-15")]
    elif year == 2022:
        frame = raw[raw["date"].between("2022-11-20", "2022-12-18")]
    else:
        raise ValueError(f"Unsupported year: {year}")
    frame = frame[frame["tournament"].eq("FIFA World Cup")].sort_values("date")
    structured = validate.load_matches(year)
    if len(frame) != len(structured):
        raise ValueError(f"Expected {len(structured)} raw eval rows for {year}, got {len(frame)}")

    out: list[dict] = []
    for (_, raw_row), match in zip(frame.iterrows(), structured):
        merged = dict(match)
        merged["home_team"] = str(raw_row["home_team"])
        merged["away_team"] = str(raw_row["away_team"])
        out.append(merged)
    return out


def _historical_prior_maps(year: int, team_names: Iterable[str], config: FitConfig) -> tuple[PriorMaps, float]:
    elo = validate.load_elo(SNAPSHOT_BY_YEAR[year])
    fifa, _ = validate.load_fifa(SNAPSHOT_BY_YEAR[year])
    f0 = float(np.mean(list(fifa.values())))
    priors = build_prior_maps(
        team_names,
        name_to_iso3=NAME_TO_ISO3,
        elo_by_iso=elo,
        fifa_by_iso=fifa,
        f0=f0,
        config=config,
    )
    return priors, f0


def _baseline_blend_predictions(year: int, matches: list[dict], f0: float) -> np.ndarray:
    elo = validate.load_elo(SNAPSHOT_BY_YEAR[year])
    fifa, _ = validate.load_fifa(SNAPSHOT_BY_YEAR[year])
    hosts = validate.HOST_BY_YEAR[year]
    preds = []
    for match in matches:
        preds.append(
            validate.predict(
                elo_a=elo[match["home_iso3"]],
                elo_b=elo[match["away_iso3"]],
                fifa_a=fifa[match["home_iso3"]],
                fifa_b=fifa[match["away_iso3"]],
                f0=f0,
                a_is_host=match["home_iso3"] in hosts,
                b_is_host=match["away_iso3"] in hosts,
                mode="blend",
            )
        )
    return np.array(preds)


def _dc_predictions(model: DixonColesAttackDefenseModel, year: int, matches: list[dict]) -> np.ndarray:
    hosts = validate.HOST_BY_YEAR[year]
    return np.array([
        model.predict(
            match["home_team"],
            match["away_team"],
            neutral=bool(match["is_neutral"]),
            a_is_host=match["home_iso3"] in hosts,
            b_is_host=match["away_iso3"] in hosts,
        )
        for match in matches
    ])


def _calibration(preds: np.ndarray, outcomes: np.ndarray) -> dict:
    mean_p, mean_y, sizes = validate.calibration_buckets(preds, outcomes)
    worst = 0.0
    for p, y, n in zip(mean_p, mean_y, sizes):
        if n >= 8 and not np.isnan(p) and not np.isnan(y):
            worst = max(worst, abs(float(p) - float(y)))
    return {
        "mean_predicted": [None if np.isnan(x) else float(x) for x in mean_p],
        "mean_observed": [None if np.isnan(x) else float(x) for x in mean_y],
        "bucket_sizes": [int(x) for x in sizes],
        "worst_error": worst,
    }


def _draw_rate(matches: list[dict], preds: np.ndarray, outcomes: np.ndarray) -> dict:
    group_mask = np.array([not match["is_knockout"] for match in matches])
    observed = float(np.mean(outcomes[group_mask, 1]))
    simulated = float(np.mean(preds[group_mask, 1]))
    return {
        "observed_group_stage": observed,
        "simulated_group_stage": simulated,
        "delta_pp": (simulated - observed) * 100.0,
    }


def _fit_payload(model: DixonColesAttackDefenseModel, training_rows: int, priors: PriorMaps) -> dict:
    return {
        "rho": model.rho,
        "home_adv": model.home_adv,
        "log_mu": model.log_mu,
        "training_rows": training_rows,
        "mapped_prior_teams": len(priors.mapped_teams),
        "unmapped_prior_teams": len(priors.unmapped_teams),
    }


def evaluate_year(
    year: int,
    train_df: pd.DataFrame,
    *,
    max_train_matches: int | None = None,
    config: FitConfig | None = None,
) -> dict:
    cfg = config or FitConfig()
    training, cutoff = _training_frame(train_df, year, max_train_matches)
    eval_matches = _eval_matches_with_names(year)
    training_team_names = set(training["home_team"]) | set(training["away_team"])
    eval_team_names = {m["home_team"] for m in eval_matches} | {m["away_team"] for m in eval_matches}
    priors, f0 = _historical_prior_maps(year, training_team_names | eval_team_names, cfg)
    model = fit_model(_records_from_frame(training), priors, config=cfg, cutoff_date=cutoff)

    dc_preds = _dc_predictions(model, year, eval_matches)
    blend_preds = _baseline_blend_predictions(year, eval_matches, f0)
    y_90 = np.array([validate.one_hot_90min(match) for match in eval_matches])

    return {
        "year": year,
        "training_rows": int(len(training)),
        "rps_90min": float(validate.rps(dc_preds, y_90)),
        "blend_rps_90min": float(validate.rps(blend_preds, y_90)),
        "draw_rate": _draw_rate(eval_matches, dc_preds, y_90),
        "blend_draw_rate": _draw_rate(eval_matches, blend_preds, y_90),
        "calibration": _calibration(dc_preds, y_90),
        "blend_calibration": _calibration(blend_preds, y_90),
        "fit": _fit_payload(model, len(training), priors),
        "_preds": dc_preds,
        "_blend_preds": blend_preds,
        "_outcomes": y_90,
        "_matches": eval_matches,
    }


def _combine_draw_rate(year_results: list[dict], key: str) -> dict:
    matches = [m for result in year_results for m in result["_matches"]]
    preds = np.vstack([result[key] for result in year_results])
    outcomes = np.vstack([result["_outcomes"] for result in year_results])
    return _draw_rate(matches, preds, outcomes)


def _decision(total: dict, blend: dict, draw: dict, blend_draw: dict, cal: dict, blend_cal: dict) -> str:
    improves_rps = total["total"] < blend["total"]
    draw_ok = abs(draw["delta_pp"]) <= abs(blend_draw["delta_pp"]) + 2.0
    cal_ok = cal["worst_error"] <= blend_cal["worst_error"] + 0.05
    return "promote_candidate" if improves_rps and draw_ok and cal_ok else "keep_as_experiment"


def run_validation(
    *,
    years: list[int],
    train_data_path: Path,
    out_dir: Path,
    max_train_matches: int | None = None,
    forecast_2026: bool = False,
) -> dict:
    train_df = load_international_results(train_data_path)
    skipped_rows = int(train_df.attrs.get("skipped_rows", 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_year(year, train_df, max_train_matches=max_train_matches) for year in years]

    all_dc = np.vstack([result["_preds"] for result in results])
    all_blend = np.vstack([result["_blend_preds"] for result in results])
    all_y = np.vstack([result["_outcomes"] for result in results])
    rps_total = {"total": float(validate.rps(all_dc, all_y))}
    blend_total = {"total": float(validate.rps(all_blend, all_y))}
    draw = _combine_draw_rate(results, "_preds")
    blend_draw = _combine_draw_rate(results, "_blend_preds")
    calibration = _calibration(all_dc, all_y)
    blend_calibration = _calibration(all_blend, all_y)

    payload = {
        "decision": _decision(rps_total, blend_total, draw, blend_draw, calibration, blend_calibration),
        "rps_90min": rps_total | {str(result["year"]): result["rps_90min"] for result in results},
        "draw_rate": draw,
        "calibration": calibration,
        "blend_baseline": {
            "rps_90min": blend_total | {str(result["year"]): result["blend_rps_90min"] for result in results},
            "draw_rate": blend_draw,
            "calibration": blend_calibration,
        },
        "fit": {
            "skipped_rows": skipped_rows,
            "by_year": {str(result["year"]): result["fit"] for result in results},
        },
        "years": {
            str(result["year"]): {
                "training_rows": result["training_rows"],
                "rps_90min": result["rps_90min"],
                "blend_rps_90min": result["blend_rps_90min"],
                "draw_rate": result["draw_rate"],
                "calibration": result["calibration"],
            }
            for result in results
        },
    }
    out_path = out_dir / "dc_attack_defence.json"
    out_path.write_text(json.dumps(payload, indent=2))
    if forecast_2026:
        payload["forecast_2026_path"] = str(run_2026_preview(train_df, out_dir))
    return payload


def _records_for_2026(train_df: pd.DataFrame) -> tuple[pd.DataFrame, date]:
    target = date(2026, 6, 10)
    latest = train_df["date"].max().date()
    cutoff = min(target, latest)
    start = cutoff - timedelta(days=365 * 4)
    mask = (train_df["date"] >= pd.Timestamp(start)) & (train_df["date"] <= pd.Timestamp(cutoff))
    return train_df[mask].sort_values("date"), cutoff


def run_2026_preview(train_df: pd.DataFrame, out_dir: Path) -> Path:
    from wcsim.data import DEFAULT_DRAW_PATH, DEFAULT_TEAMS_PATH, load_draw, load_teams, load_venues
    from wcsim.squad_data import load_squads

    teams = load_teams(DEFAULT_TEAMS_PATH)
    draw = load_draw(DEFAULT_DRAW_PATH)
    venues = load_venues()
    squads = load_squads()
    training, cutoff = _records_for_2026(train_df)
    draw_names = {teams[iso3].name for group in draw.values() for iso3 in group}
    training_names = set(training["home_team"]) | set(training["away_team"])
    fifa = {iso3: team.fifa_points for iso3, team in teams.items() if team.fifa_points is not None}
    f0 = float(np.mean(list(fifa.values())))
    cfg = FitConfig()
    priors = build_prior_maps(
        training_names | draw_names,
        name_to_iso3=NAME_TO_ISO3,
        elo_by_iso={iso3: team.elo for iso3, team in teams.items()},
        fifa_by_iso=fifa,
        f0=f0,
        config=cfg,
        squad_data=squads,
        player_weight=0.5,
    )
    model = fit_model(_records_from_frame(training), priors, config=cfg, cutoff_date=cutoff)

    group_matches = []
    group_venues = venues.group_venues if venues else {}
    for group_letter, iso3s in sorted(draw.items()):
        venue_host = group_venues.get(group_letter)
        for home_iso, away_iso in itertools.combinations(iso3s, 2):
            home_name = teams[home_iso].name
            away_name = teams[away_iso].name
            probs = model.predict(
                home_name,
                away_name,
                neutral=True,
                a_is_host=home_iso == venue_host,
                b_is_host=away_iso == venue_host,
            )
            group_matches.append({
                "group": group_letter,
                "home": home_iso,
                "away": away_iso,
                "home_win": probs[0],
                "draw": probs[1],
                "away_win": probs[2],
            })

    payload = {
        "fit": _fit_payload(model, len(training), priors),
        "cutoff_date": cutoff.isoformat(),
        "teams": {
            iso3: {
                "name": team.name,
                "attack": model.attacks.get(team.name, model.prior_attacks.get(team.name, 0.0) if model.prior_attacks else 0.0),
                "defense": model.defenses.get(team.name, model.prior_defenses.get(team.name, 0.0) if model.prior_defenses else 0.0),
                "prior_strength": priors.strength.get(team.name, 0.0),
            }
            for iso3, team in sorted(teams.items())
        },
        "group_matches": group_matches,
    }
    out_path = out_dir / "dc_2026_preview.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def _parse_years(value: str) -> list[int]:
    years = [int(part.strip()) for part in value.split(",") if part.strip()]
    unsupported = [year for year in years if year not in TRAIN_WINDOWS]
    if unsupported:
        raise ValueError(f"Unsupported years: {unsupported}")
    return years


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2018,2022")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--max-train-matches", type=int, default=None)
    parser.add_argument("--forecast-2026", action="store_true")
    args = parser.parse_args(argv)

    payload = run_validation(
        years=_parse_years(args.years),
        train_data_path=args.train_data,
        out_dir=args.out_dir,
        max_train_matches=args.max_train_matches,
        forecast_2026=args.forecast_2026,
    )
    print(json.dumps({
        "decision": payload["decision"],
        "rps_90min": payload["rps_90min"],
        "blend_rps_90min": payload["blend_baseline"]["rps_90min"],
        "draw_rate": payload["draw_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
