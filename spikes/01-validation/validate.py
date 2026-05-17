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


# Defaults from PRD v1.7 §5.5.
PARAMS = {
    "c_elo": 300.0,
    "c_fifa": 450.0,
    "mu": 1.35,
    "lambda_min": 0.05,
    "blend_w": 0.7,
    "e0": 1500.0,
    "home_bonus_elo": 100.0,
    "home_bonus_fifa": 150.0,
    "rho": 0.0,  # Dixon-Coles correlation; 0 reduces to independent Poisson
}

SCORE_GRID_MAX = 8   # inclusive => 9x9 grid; covers > 99.99% of mass.


def _poisson_pmf(lmbda: float, max_goals: int) -> np.ndarray:
    """Discrete Poisson PMF over [0, max_goals]. Drops residual mass."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    log_fact[0] = 0.0
    log_pmf = k * np.log(max(lmbda, 1e-12)) - lmbda - log_fact
    return np.exp(log_pmf)


def _outcome_probs(lam_a: float, lam_b: float, rho: float = 0.0) -> tuple[float, float, float]:
    """Return (P(home_win), P(draw), P(away_win)) from two Poisson marginals with
    a Dixon-Coles τ correlation factor on the four lowest-scoring joint outcomes.
    rho=0 reduces to independent Poisson (the v1.6 baseline)."""
    pa = _poisson_pmf(lam_a, SCORE_GRID_MAX)
    pb = _poisson_pmf(lam_b, SCORE_GRID_MAX)
    grid = np.outer(pa, pb)
    if rho != 0.0:
        # Dixon-Coles τ only adjusts the four cells (0,0), (0,1), (1,0), (1,1).
        # Validity: τ must stay non-negative. For |rho|<=0.2 and typical λ band,
        # the binding constraint is τ(0,0) = 1 − λ_A λ_B ρ; we clamp to 0.
        grid[0, 0] *= max(0.0, 1 - lam_a * lam_b * rho)
        grid[0, 1] *= max(0.0, 1 + lam_a * rho)
        grid[1, 0] *= max(0.0, 1 + lam_b * rho)
        grid[1, 1] *= max(0.0, 1 - rho)
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
    return _outcome_probs(lam_a, lam_b, p.get("rho", 0.0))


HOST_BY_YEAR = {2018: {"RUS"}, 2022: {"QAT"}}


def one_hot_90min(match: dict) -> np.ndarray:
    """Return one-hot (home_win, draw, away_win) under the 90-minute convention.
    Knockout matches that went to ET are counted as 90-min draws (because the
    teams were tied at 90 min before extra time)."""
    h, a = match["regulation_home"], match["regulation_away"]
    if h > a:
        return np.array([1.0, 0.0, 0.0])
    if h < a:
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def one_hot_post_et(match: dict) -> np.ndarray:
    """Return one-hot under the post-ET convention. Penalty winners are
    attributed to that side; 'draw' is impossible for knockout matches."""
    h, a = match["et_home"], match["et_away"]
    if h > a:
        return np.array([1.0, 0.0, 0.0])
    if h < a:
        return np.array([0.0, 0.0, 1.0])
    if match["is_knockout"] and match["pen_winner_iso3"]:
        if match["pen_winner_iso3"] == match["home_iso3"]:
            return np.array([1.0, 0.0, 0.0])
        return np.array([0.0, 0.0, 1.0])
    return np.array([0.0, 1.0, 0.0])


def predict_all_matches(
    matches: list[dict],
    elo: dict[str, float],
    fifa_pts: dict[str, float],
    f0: float,
    hosts: set[str],
) -> dict[str, np.ndarray]:
    """Return {mode: array of shape (N, 3)} with predicted probabilities."""
    preds = {mode: np.zeros((len(matches), 3)) for mode in ("elo", "fifa", "blend")}
    for i, m in enumerate(matches):
        a, b = m["home_iso3"], m["away_iso3"]
        for mode in ("elo", "fifa", "blend"):
            preds[mode][i] = predict(
                elo_a=elo[a], elo_b=elo[b],
                fifa_a=fifa_pts[a], fifa_b=fifa_pts[b],
                f0=f0,
                a_is_host=(a in hosts), b_is_host=(b in hosts),
                mode=mode,
            )
    return preds


def by_bucket_rps(
    matches: list[dict],
    preds_all: dict[str, np.ndarray],
    y_90: np.ndarray,
    elo_by_year: dict[int, dict[str, float]],
) -> dict:
    """Bucket each match by 90-min realised outcome:
    favourite_wins / underdog_wins / draw. Favourite is defined by higher
    pre-match Elo, independent of which mode is scored, so the buckets stay
    comparable across modes."""
    fav_is_home = []
    for m in matches:
        year = 2018 if m["date"].startswith("2018") else 2022
        elo = elo_by_year[year]
        fav_is_home.append(elo[m["home_iso3"]] >= elo[m["away_iso3"]])
    fav_is_home = np.array(fav_is_home)

    labels = []
    for i, m in enumerate(matches):
        outcome = int(np.argmax(y_90[i]))   # 0=home, 1=draw, 2=away
        if outcome == 1:
            labels.append("draw")
        elif (outcome == 0 and fav_is_home[i]) or (outcome == 2 and not fav_is_home[i]):
            labels.append("favourite_wins")
        else:
            labels.append("underdog_wins")
    labels = np.array(labels)

    out: dict = {}
    for tag in ("favourite_wins", "underdog_wins", "draw"):
        mask = labels == tag
        out[tag] = {"n": int(mask.sum())}
        if mask.sum() == 0:
            continue
        for mode in preds_all:
            out[tag][mode] = rps(preds_all[mode][mask], y_90[mask])
    return out


def calibration_buckets(preds: np.ndarray, outcomes: np.ndarray, n_buckets: int = 10):
    """Decile-bucket all (prediction, outcome) pairs flattened across
    home/draw/away. Returns (mean_predicted, mean_observed, bucket_sizes)
    arrays of length n_buckets."""
    p = preds.flatten()
    y = outcomes.flatten()
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    mean_p = np.full(n_buckets, np.nan)
    mean_y = np.full(n_buckets, np.nan)
    sizes = np.zeros(n_buckets, dtype=int)
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_buckets - 1 else (p >= lo) & (p <= hi)
        sizes[i] = int(mask.sum())
        if sizes[i] > 0:
            mean_p[i] = float(p[mask].mean())
            mean_y[i] = float(y[mask].mean())
    return mean_p, mean_y, sizes


def write_calibration_plot(preds: np.ndarray, outcomes: np.ndarray, out_path: Path) -> dict:
    mean_p, mean_y, sizes = calibration_buckets(preds, outcomes)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    valid = sizes > 0
    ax.scatter(
        mean_p[valid], mean_y[valid],
        s=np.maximum(sizes[valid] * 4, 20),
        c="C0", alpha=0.7, edgecolors="C0",
        label="Elo 90-min (point size ∝ n)",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Elo-mode 90-min calibration (deciles)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "mean_predicted": [None if np.isnan(x) else float(x) for x in mean_p],
        "mean_observed":  [None if np.isnan(x) else float(x) for x in mean_y],
        "bucket_sizes":   [int(s) for s in sizes],
    }


def simulated_draw_rate(preds: np.ndarray) -> float:
    """Mean predicted draw probability (column index 1)."""
    return float(np.mean(preds[:, 1]))


def observed_draw_rate(matches: list[dict], y_90: np.ndarray) -> float:
    """Fraction of group-stage matches that ended 90 minutes level."""
    grp_mask = np.array([not m["is_knockout"] for m in matches])
    return float(np.mean(y_90[grp_mask, 1]))


def rps(preds: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score on 3-outcome data. `preds` and `outcomes` are
    both shape (N, 3) with the column order [home_win, draw, away_win] and
    `outcomes` one-hot. Returns mean RPS in [0, 1] where lower is better.

    A uniform predictor on balanced 3-outcome data scores exactly 2/9 ≈ 0.222
    (asserted at startup in main; see Task 6 Step 2)."""
    K = preds.shape[1]
    cum_p = np.cumsum(preds[:, :-1], axis=1)
    cum_y = np.cumsum(outcomes[:, :-1], axis=1)
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / (K - 1)))


def decide(results: dict) -> tuple[str, list[str]]:
    """Apply the spec §6 four-way decision matrix. Returns (label, flags)."""
    elo_90 = results["rps_90min"]["elo"]["total"]
    cal = results["calibration_elo_90min"]

    # Calibration check: every decile with >= 8 matches within +/- 0.05 of diagonal.
    cal_ok = True
    for p, y, n in zip(cal["mean_predicted"], cal["mean_observed"], cal["bucket_sizes"]):
        if n is None or n < 8 or p is None or y is None:
            continue
        if abs(p - y) > 0.05:
            cal_ok = False
            break

    flags: list[str] = []
    if abs(results["draw_rate"]["delta_pp"]["elo"]) > 2:
        flags.append("draw_rate_off")
    if results["et_divergence"]["elo"] > 0.01:
        flags.append("et_divergence_high")

    if elo_90 < 0.215 and cal_ok:
        fifa_b = results["rps_90min"]["fifa"]["total"]
        blend_b = results["rps_90min"]["blend"]["total"]
        if fifa_b > 0.225 or blend_b > 0.225:
            return "pass_with_caveat", flags
        return "pass", flags
    if elo_90 >= 0.222:
        return "hard_fail", flags
    return "soft_fail", flags


def main() -> None:
    # --- Startup invariants (Fix 6). Failures here mean validate.py is broken. ---
    _uniform = np.full((300, 3), 1 / 3)
    _balanced = np.tile(np.eye(3), (100, 1))
    assert abs(rps(_uniform, _balanced) - 2 / 9) < 1e-9, (
        f"RPS broken: uniform predictor should score 2/9, got {rps(_uniform, _balanced)}"
    )

    # --- Predictor invariants (Fix 6). These run on synthetic inputs, no data needed. ---
    _p = predict(elo_a=2000, elo_b=1500, fifa_a=1500, fifa_b=1200,
                 f0=1300.0, a_is_host=False, b_is_host=False, mode="elo")
    assert abs(sum(_p) - 1.0) < 1e-9, f"Probs don't sum to 1: {_p}"
    _q = predict(elo_a=1500, elo_b=2000, fifa_a=1200, fifa_b=1500,
                 f0=1300.0, a_is_host=False, b_is_host=False, mode="elo")
    assert abs(_p[0] - _q[2]) < 1e-9 and abs(_p[2] - _q[0]) < 1e-9, (
        f"Predictor not symmetric: {_p} vs {_q}"
    )

    results: dict = {"params": dict(PARAMS), "n_matches": {}}
    f0_by_year: dict[int, float] = {}

    all_matches: list[dict] = []
    all_preds = {mode: [] for mode in ("elo", "fifa", "blend")}
    all_y_90: list[np.ndarray] = []
    all_y_et: list[np.ndarray] = []
    per_year_indices: dict[int, list[int]] = {}
    elo_by_year: dict[int, dict[str, float]] = {}

    for year, snap in ((2018, WC2018[0]), (2022, WC2022[0])):
        elo = load_elo(snap)
        elo_by_year[year] = elo
        fifa_pts, _ = load_fifa(snap)
        f0 = float(np.mean(list(fifa_pts.values())))
        f0_by_year[year] = f0
        matches = load_matches(year)
        preds = predict_all_matches(matches, elo, fifa_pts, f0, HOST_BY_YEAR[year])
        print(f"WC {year}: {len(matches)} matches, {len(elo)} elo teams, "
              f"{len(fifa_pts)} fifa teams, f0={f0:.1f}")

        # --- Loader invariants (Fix 6) — moved into the per-year loop. ---
        assert len(matches) == 64, f"Expected 64 WC {year} matches, got {len(matches)}"

        start = len(all_matches)
        all_matches.extend(matches)
        for mode in preds:
            all_preds[mode].append(preds[mode])
        for m in matches:
            all_y_90.append(one_hot_90min(m))
            all_y_et.append(one_hot_post_et(m))
        per_year_indices[year] = list(range(start, len(all_matches)))

    y_90 = np.array(all_y_90)
    y_et = np.array(all_y_et)
    preds_all = {mode: np.vstack(all_preds[mode]) for mode in all_preds}

    # Cross-year spot-check invariants.
    for team_iso in ["ARG", "FRA", "BRA", "GER", "ESP"]:
        assert team_iso in elo_by_year[2022], f"{team_iso} missing from Elo 2022"
    final_2022 = all_matches[-1]
    assert final_2022["pen_winner_iso3"] == "ARG", (
        f"2022 final's pen_winner_iso3 should be ARG, got {final_2022['pen_winner_iso3']}"
    )

    # Fix 4: record f0 per tournament (FIFA reformed mid-2018; 2018 and 2022
    # means differ; schema uses f0_by_year, not a single scalar).
    results["params"]["f0_by_year"] = {str(y): v for y, v in f0_by_year.items()}

    n_kn = sum(1 for m in all_matches if m["is_knockout"])
    results["n_matches"] = {
        "total": len(all_matches),
        "group_stage": len(all_matches) - n_kn,
        "knockout": n_kn,
    }
    results["rps_90min"] = {}
    results["rps_post_et"] = {}
    for mode in ("elo", "fifa", "blend"):
        idx_2018 = per_year_indices[2018]
        idx_2022 = per_year_indices[2022]
        results["rps_90min"][mode] = {
            "total": rps(preds_all[mode], y_90),
            "2018": rps(preds_all[mode][idx_2018], y_90[idx_2018]),
            "2022": rps(preds_all[mode][idx_2022], y_90[idx_2022]),
        }
        results["rps_post_et"][mode] = {
            "total": rps(preds_all[mode], y_et),
            "2018": rps(preds_all[mode][idx_2018], y_et[idx_2018]),
            "2022": rps(preds_all[mode][idx_2022], y_et[idx_2022]),
        }
    results["et_divergence"] = {
        mode: results["rps_post_et"][mode]["total"] - results["rps_90min"][mode]["total"]
        for mode in ("elo", "fifa", "blend")
    }

    # --- Task 7: simulated draw rate per mode (informational). ---
    grp_mask = np.array([not m["is_knockout"] for m in all_matches])
    observed = observed_draw_rate(all_matches, y_90)
    sim_per_mode = {
        mode: simulated_draw_rate(preds_all[mode][grp_mask]) for mode in preds_all
    }
    results["draw_rate"] = {
        "observed_group_stage": observed,
        "simulated_group_stage": sim_per_mode,
        "delta_pp": {mode: (sim_per_mode[mode] - observed) * 100 for mode in sim_per_mode},
    }

    # --- Task 8: Elo-mode 90-min calibration plot. ---
    cal = write_calibration_plot(preds_all["elo"], y_90, RESULTS / "calibration.png")
    results["calibration_elo_90min"] = cal
    print(f"\nCalibration plot -> {RESULTS / 'calibration.png'}")

    # --- Task 9: by-outcome-bucket RPS breakdown. ---
    results["by_bucket"] = by_bucket_rps(all_matches, preds_all, y_90, elo_by_year)

    print(json.dumps({
        "n_matches": results["n_matches"],
        "f0_by_year": results["params"]["f0_by_year"],
        "rps_90min": results["rps_90min"],
        "rps_post_et": results["rps_post_et"],
        "et_divergence": results["et_divergence"],
        "draw_rate": results["draw_rate"],
        "by_bucket": results["by_bucket"],
    }, indent=2))

    # --- Task 10: decide + write brier.json + final schema check. ---
    decision, flags = decide(results)
    results["decision"] = decision
    results["informational_flags"] = flags

    # Final schema check (Fix 6) — verify every consumer-visible key is present
    # before we declare success.
    required_top_level = {
        "decision", "params", "n_matches",
        "rps_90min", "rps_post_et", "et_divergence",
        "draw_rate", "calibration_elo_90min", "by_bucket",
        "informational_flags",
    }
    missing_keys = required_top_level - set(results)
    assert not missing_keys, f"brier.json missing required keys: {missing_keys}"
    assert "f0_by_year" in results["params"], "params.f0_by_year missing"

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "brier.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDecision: {decision}")
    if flags:
        print(f"Informational flags: {flags}")
    print(f"Wrote {RESULTS / 'brier.json'}")


if __name__ == "__main__":
    main()
