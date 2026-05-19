"""Data loaders for the CLI. Reads bundled CSVs/JSON into wcsim types."""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .types import Team

SPIKE_DATA = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw"
DEFAULT_TEAMS_PATH = SPIKE_DATA / "elo_history.csv"
DEFAULT_FIFA_PATH = SPIKE_DATA / "fifa_ranking.csv"
DEFAULT_DRAW_PATH = SPIKE_DATA / "wc2026_draw.json"
DEFAULT_VENUES_PATH = SPIKE_DATA / "wc2026_venues.json"

# Import name_to_iso3 from the spike directory.
_spike_dir = str(Path(__file__).parent.parent / "spikes" / "01-validation")
if _spike_dir not in sys.path:
    sys.path.insert(0, _spike_dir)
from name_to_iso3 import to_iso3  # noqa: E402


def _load_fifa_snapshot(
    fifa_path: Path, target: pd.Timestamp,
) -> tuple[dict[str, float], dict[str, int]]:
    """Load FIFA points and ranks from the latest snapshot <= target date."""
    import warnings
    fifa_points: dict[str, float] = {}
    fifa_ranks: dict[str, int] = {}
    if not fifa_path.exists():
        return fifa_points, fifa_ranks
    fdf = pd.read_csv(fifa_path)
    fdf["rank_date"] = pd.to_datetime(fdf["rank_date"])
    eligible = fdf[fdf["rank_date"] <= target]
    if eligible.empty:
        return fifa_points, fifa_ranks
    latest_date = eligible["rank_date"].max()
    staleness_days = (target - latest_date).days
    if staleness_days > 180:
        warnings.warn(
            f"FIFA ranking snapshot is {staleness_days} days stale "
            f"(latest: {latest_date.date()}, target: {target.date()}). "
            f"Results for --rating fifa/blend may be unreliable.",
            stacklevel=3,
        )
    fsnap = eligible[eligible["rank_date"] == latest_date]
    for _, row in fsnap.iterrows():
        try:
            iso3 = to_iso3(row["country_full"])
        except KeyError:
            continue
        fifa_points[iso3] = float(row["total_points"])
        fifa_ranks[iso3] = int(row["rank"])
    return fifa_points, fifa_ranks


def load_teams(
    csv_path: Path, snapshot_date: str = "2026-06-10",
    fifa_path: Path | None = None,
) -> dict[str, Team]:
    """Load teams from elo_history.csv + merge FIFA points from fifa_ranking.csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Teams file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    target = pd.to_datetime(snapshot_date)
    snapshot = df[df["date"] == target]
    if snapshot.empty:
        df = df[df["date"] <= target].sort_values("date")
        snapshot = df.groupby("team").tail(1)

    fifa_points, fifa_ranks = _load_fifa_snapshot(fifa_path or DEFAULT_FIFA_PATH, target)

    teams: dict[str, Team] = {}
    for _, row in snapshot.iterrows():
        try:
            iso3 = to_iso3(row["team"])
        except KeyError:
            continue
        teams[iso3] = Team(
            name=row["team"], iso3=iso3, confederation="UNK",
            elo=float(row["rating"]),
            fifa_points=fifa_points.get(iso3),
            fifa_rank=fifa_ranks.get(iso3),
        )
    return teams


def load_draw(json_path: Path) -> dict[str, list[str]]:
    """Load a draw JSON (group letter -> list of ISO3 codes)."""
    if not json_path.exists():
        raise FileNotFoundError(f"Draw file not found: {json_path}")
    with json_path.open() as f:
        return json.load(f)


@dataclass(frozen=True)
class Venues:
    """Per-match venue mapping: which host country gets the bonus."""
    group_venues: dict[str, str]   # group letter -> host ISO3 ("USA", "MEX", "CAN")
    knockout_venue: str            # ISO3 of the knockout host (all rounds)


def load_venues(json_path: Path | None = None) -> Venues | None:
    """Load venue assignments. Returns None if file doesn't exist (backward compat)."""
    vp = json_path or DEFAULT_VENUES_PATH
    if not vp.exists():
        return None
    with vp.open() as f:
        data = json.load(f)
    return Venues(
        group_venues=data.get("group_venues", {}),
        knockout_venue=data.get("knockout_venue", "USA"),
    )
