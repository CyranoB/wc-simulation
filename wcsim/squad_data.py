"""Squad data loader: reads bundled squads_2026.json → per-team total values."""
from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SQUAD_PATH = (
    Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw" / "squads_2026.json"
)


def load_squads(path: Path | None = None) -> dict[str, dict]:
    """Load squad data. Returns {iso3: {"total_value_eur": int, ...}}.

    Raises FileNotFoundError if the JSON is missing, ValueError on schema issues."""
    p = path or DEFAULT_SQUAD_PATH
    if not p.exists():
        raise FileNotFoundError(f"Squad data not found at {p}")

    with p.open() as f:
        raw = json.load(f)

    if "squads" not in raw:
        raise ValueError(f"Squad data missing 'squads' key in {p}")

    squads = raw["squads"]
    for iso3, data in squads.items():
        if "total_value_eur" not in data:
            raise ValueError(f"Squad data for {iso3} missing 'total_value_eur'")

    return squads
