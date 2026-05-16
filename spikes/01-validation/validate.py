"""Spike 1: Model Validation Back-Test for wcsim PRD §5.5.

Loads pre-WC Elo + FIFA ratings and historical match results, predicts every
match under Elo / FIFA / Blend rating modes under two scoring conventions
(90-min and post-ET), computes Brier + draw-rate + Elo-mode calibration,
and emits results/brier.json plus results/calibration.png.

Throwaway. Single-threaded NumPy. No CLI, no package, no unit tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from name_to_iso3 import to_iso3


HERE = Path(__file__).parent
DATA = HERE / "data" / "raw"
RESULTS = HERE / "results"


# (snapshot date, opening match, final) per tournament.
WC2018 = ("2018-06-13", "2018-06-14", "2018-07-15")
WC2022 = ("2022-11-19", "2022-11-20", "2022-12-18")


def load_elo(snapshot_date: str) -> dict[str, float]:
    """Return {iso3: rating} for teams as of snapshot_date.

    Uses 'latest row on or before snapshot_date, per team' so rating-change-only
    datasets work. The scraper writes one row per team per WC snapshot, so this
    reduces to picking the matching snapshot for each team.
    """
    df = pd.read_csv(DATA / "elo_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    target = pd.to_datetime(snapshot_date)
    df = df[df["date"] <= target].sort_values("date")
    latest = df.groupby("team").tail(1)
    out: dict[str, float] = {}
    for _, row in latest.iterrows():
        try:
            out[to_iso3(row["team"])] = float(row["rating"])
        except KeyError:
            # Team not in our ISO3 map — likely a non-WC participant; skip.
            continue
    return out


def load_fifa(snapshot_date: str) -> tuple[dict[str, float], dict[str, int]]:
    """Return ({iso3: fifa_points}, {iso3: fifa_rank}) for the snapshot just on
    or before `snapshot_date`."""
    df = pd.read_csv(DATA / "fifa_ranking.csv")
    df["rank_date"] = pd.to_datetime(df["rank_date"])
    target = pd.to_datetime(snapshot_date)
    eligible = df[df["rank_date"] <= target]
    latest = eligible["rank_date"].max()
    snap = eligible[eligible["rank_date"] == latest]
    points: dict[str, float] = {}
    ranks: dict[str, int] = {}
    for _, row in snap.iterrows():
        try:
            iso3 = to_iso3(row["country_full"])
        except KeyError:
            continue
        points[iso3] = float(row["total_points"])
        ranks[iso3] = int(row["rank"])
    return points, ranks


# (date, home_team, away_team) tuples for every WC 2018+2022 match that went
# to extra time or penalties. load_matches asserts the supplement covers all
# of these — without coverage, the model's gating metric rps_90min would be
# silently corrupted (ET goals would be scored as 90-min goals).
REQUIRED_ET_PENS_MATCHES = {
    ("2018-07-01", "Russia", "Spain"),
    ("2018-07-01", "Croatia", "Denmark"),
    ("2018-07-07", "Russia", "Croatia"),
    ("2018-07-03", "Colombia", "England"),
    ("2018-07-11", "Croatia", "England"),
    ("2022-12-09", "Netherlands", "Argentina"),
    ("2022-12-05", "Japan", "Croatia"),
    ("2022-12-09", "Croatia", "Brazil"),
    ("2022-12-06", "Morocco", "Spain"),
    ("2022-12-18", "Argentina", "France"),
}


def load_matches(year: int) -> list[dict]:
    """Return list of dicts per WC match. Each dict has:
        date, home_iso3, away_iso3,
        regulation_home, regulation_away,
        et_home, et_away,
        pen_winner_iso3 (str | None),
        is_knockout (bool), is_neutral (bool)
    """
    base = pd.read_csv(DATA / "matches_history.csv")
    base["date"] = pd.to_datetime(base["date"])
    if year == 2018:
        window = base[base["date"].between("2018-06-14", "2018-07-15")]
    elif year == 2022:
        window = base[base["date"].between("2022-11-20", "2022-12-18")]
    else:
        raise ValueError(year)
    wc = window[window["tournament"].eq("FIFA World Cup")].sort_values("date")

    supplement_path = DATA / "knockout_supplement.csv"
    if not supplement_path.exists():
        raise FileNotFoundError(
            f"{supplement_path} is required (see Task 2 Step 4 in the plan). "
            "Without it, ET goals get scored as 90-min goals and rps_90min "
            "is silently corrupted."
        )
    supp = pd.read_csv(supplement_path)
    covered = {(r["date"], r["home"], r["away"]) for _, r in supp.iterrows()}
    missing = REQUIRED_ET_PENS_MATCHES - covered
    assert not missing, (
        f"knockout_supplement.csv missing required rows: {missing}. "
        "See Task 2 Step 4 in the plan for the full list."
    )

    out: list[dict] = []
    for _, row in wc.iterrows():
        home, away = row["home_team"], row["away_team"]
        # matches_history.csv stores POST-ET scores in home_score/away_score for
        # knockouts. Use as default; supplement overrides for ET/pens matches.
        et_home = int(row["home_score"])
        et_away = int(row["away_score"])
        reg_home, reg_away = et_home, et_away
        pen_winner_iso3: str | None = None
        match_supp = supp[
            (supp["date"] == row["date"].strftime("%Y-%m-%d"))
            & (supp["home"] == home) & (supp["away"] == away)
        ]
        if not match_supp.empty:
            s = match_supp.iloc[0]
            reg_home = int(s["regulation_home"])
            reg_away = int(s["regulation_away"])
            et_home = int(s["et_home"])
            et_away = int(s["et_away"])
            pen_winner_iso3 = (
                to_iso3(s["pen_winner"]) if pd.notna(s["pen_winner"]) else None
            )
        # Knockout heuristic: dates after group stage end.
        # 2018 group ends 2018-06-28; 2022 group ends 2022-12-02.
        is_knockout = (
            (year == 2018 and row["date"] >= pd.Timestamp("2018-06-30")) or
            (year == 2022 and row["date"] >= pd.Timestamp("2022-12-03"))
        )
        out.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "home_iso3": to_iso3(home),
            "away_iso3": to_iso3(away),
            "regulation_home": reg_home,
            "regulation_away": reg_away,
            "et_home": et_home,
            "et_away": et_away,
            "pen_winner_iso3": pen_winner_iso3,
            "is_knockout": is_knockout,
            "is_neutral": bool(row.get("neutral", True)),
        })
    return out


# Defaults from PRD v1.6 §5.5.
PARAMS = {
    "c_elo": 300.0,
    "c_fifa": 450.0,
    "mu": 1.35,
    "lambda_min": 0.05,
    "blend_w": 0.7,
    "e0": 1500.0,
    "home_bonus_elo": 100.0,
    "home_bonus_fifa": 150.0,
}

SCORE_GRID_MAX = 8   # inclusive => 9x9 grid; covers > 99.99% of mass.


def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    """Discrete Poisson PMF over [0, max_goals]. Drops residual mass."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    log_pmf = k * np.log(max(lmbda, 1e-12)) - lmbda - log_fact
    return np.exp(log_pmf)


def _outcome_probs(lam_a: float, lam_b: float) -> tuple[float, float, float]:
    """Return (P(home_win), P(draw), P(away_win)) from two independent Poissons."""
    pa = _poisson_pmf(lam_a, SCORE_GRID_MAX)
    pb = _poisson_pmf(lam_b, SCORE_GRID_MAX)
    grid = np.outer(pa, pb)
    p_home = float(np.tril(grid, k=-1).sum())
    p_draw = float(np.trace(grid))
    p_away = float(np.triu(grid, k=1).sum())
    s = p_home + p_draw + p_away
    return p_home / s, p_draw / s, p_away / s


def predict(
    *,
    elo_a: float, elo_b: float,
    fifa_a: float, fifa_b: float,
    f0: float,
    a_is_host: bool, b_is_host: bool,
    mode: str,
    params: dict | None = None,
) -> tuple[float, float, float]:
    """Return (P(home), P(draw), P(away)) for team A (home) vs B under `mode`."""
    p = params or PARAMS
    host_diff = float(a_is_host) - float(b_is_host)   # +1, 0, or -1
    mu = p["mu"]
    lam_min = p["lambda_min"]

    if mode == "elo":
        H = p["home_bonus_elo"] * host_diff
        D = (elo_a - elo_b) + H
        c = p["c_elo"]
    elif mode == "fifa":
        H = p["home_bonus_fifa"] * host_diff
        D = (fifa_a - fifa_b) + H
        c = p["c_fifa"]
    elif mode == "blend":
        e0 = p["e0"]
        w = p["blend_w"]
        ra = w * elo_a + (1 - w) * (fifa_a * e0 / f0)
        rb = w * elo_b + (1 - w) * (fifa_b * e0 / f0)
        H = p["home_bonus_elo"] * host_diff   # blend lives in Elo space
        D = (ra - rb) + H
        c = p["c_elo"]
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    lam_a = max(lam_min, mu + D / (2 * c))
    lam_b = max(lam_min, mu - D / (2 * c))
    return _outcome_probs(lam_a, lam_b)


def main() -> None:
    # --- Predictor invariants (Fix 6). These run on synthetic inputs, no data needed. ---
    _p = predict(elo_a=2000, elo_b=1500, fifa_a=1500, fifa_b=1200,
                 f0=1300.0, a_is_host=False, b_is_host=False, mode="elo")
    assert abs(sum(_p) - 1.0) < 1e-9, f"Probs don't sum to 1: {_p}"
    _q = predict(elo_a=1500, elo_b=2000, fifa_a=1200, fifa_b=1500,
                 f0=1300.0, a_is_host=False, b_is_host=False, mode="elo")
    assert abs(_p[0] - _q[2]) < 1e-9 and abs(_p[2] - _q[0]) < 1e-9, (
        f"Predictor not symmetric: {_p} vs {_q}"
    )

    elo_2018 = load_elo(WC2018[0])
    elo_2022 = load_elo(WC2022[0])
    fifa_2018_pts, _ = load_fifa(WC2018[0])
    fifa_2022_pts, _ = load_fifa(WC2022[0])
    m2018 = load_matches(2018)
    m2022 = load_matches(2022)

    print(f"Elo 2018: {len(elo_2018)} teams (sample BRA={elo_2018.get('BRA')})")
    print(f"Elo 2022: {len(elo_2022)} teams (sample ARG={elo_2022.get('ARG')})")
    print(f"FIFA 2018: {len(fifa_2018_pts)} teams (sample BRA={fifa_2018_pts.get('BRA')})")
    print(f"FIFA 2022: {len(fifa_2022_pts)} teams (sample ARG={fifa_2022_pts.get('ARG')})")
    print(f"Matches 2018: {len(m2018)} (expect 64)")
    print(f"Matches 2022: {len(m2022)} (expect 64)")
    print(f"Sample 2022 final: {m2022[-1]}")

    # --- Loader invariants (Fix 6). ---
    assert len(m2018) == 64, f"Expected 64 WC 2018 matches, got {len(m2018)}"
    assert len(m2022) == 64, f"Expected 64 WC 2022 matches, got {len(m2022)}"
    for team_iso in ["ARG", "FRA", "BRA", "GER", "ESP"]:   # spot-check well-known teams
        assert team_iso in elo_2022, f"{team_iso} missing from Elo 2022"
        assert team_iso in fifa_2022_pts, f"{team_iso} missing from FIFA 2022"
    # The 2022 final must have gone to pens with Argentina winning.
    final_2022 = m2022[-1]
    assert final_2022["pen_winner_iso3"] == "ARG", (
        f"2022 final's pen_winner_iso3 should be ARG, got {final_2022['pen_winner_iso3']}"
    )

    # --- Task 5 sanity print: ARG-FRA 2022 final prediction in all 3 modes. ---
    f0_2022 = float(np.mean(list(fifa_2022_pts.values())))
    print(f"\nF0 (mean FIFA points 2022 snapshot) = {f0_2022:.1f}")
    a, b = "ARG", "FRA"
    for mode in ("elo", "fifa", "blend"):
        p = predict(
            elo_a=elo_2022[a], elo_b=elo_2022[b],
            fifa_a=fifa_2022_pts[a], fifa_b=fifa_2022_pts[b],
            f0=f0_2022,
            a_is_host=False, b_is_host=False,    # neutral venue (Qatar hosted)
            mode=mode,
        )
        print(f"ARG-FRA 2022 final ({mode}): P(ARG)={p[0]:.3f} draw={p[1]:.3f} P(FRA)={p[2]:.3f}")


if __name__ == "__main__":
    main()
