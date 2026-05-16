"""Data loaders for the CLI. Reads bundled CSVs/JSON into wcsim types."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
from .types import Team

SPIKE_DATA = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw"
DEFAULT_TEAMS_PATH = SPIKE_DATA / "elo_history.csv"
DEFAULT_DRAW_PATH = SPIKE_DATA / "wc2026_draw.json"

# Import name_to_iso3 from the spike directory.
_spike_dir = str(Path(__file__).parent.parent / "spikes" / "01-validation")
if _spike_dir not in sys.path:
    sys.path.insert(0, _spike_dir)
from name_to_iso3 import to_iso3


def load_teams(csv_path: Path, snapshot_date: str = "2026-06-10") -> dict[str, Team]:
    """Load teams from an elo_history.csv, filtering to a specific snapshot date."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Teams file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    target = pd.to_datetime(snapshot_date)
    # Use only the exact snapshot date's rows (each date represents a tournament cohort).
    snapshot = df[df["date"] == target]
    if snapshot.empty:
        # Fallback: use latest record per team on or before target.
        df = df[df["date"] <= target].sort_values("date")
        snapshot = df.groupby("team").tail(1)
    teams: dict[str, Team] = {}
    for _, row in snapshot.iterrows():
        try:
            iso3 = to_iso3(row["team"])
        except KeyError:
            continue
        teams[iso3] = Team(name=row["team"], iso3=iso3, confederation="UNK", elo=float(row["rating"]))
    return teams


def load_draw(json_path: Path) -> dict[str, list[str]]:
    """Load a draw JSON (group letter -> list of ISO3 codes)."""
    if not json_path.exists():
        raise FileNotFoundError(f"Draw file not found: {json_path}")
    with json_path.open() as f:
        return json.load(f)
