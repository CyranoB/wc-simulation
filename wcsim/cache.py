"""Cache: persists last-run results for wcsim bracket (future)."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .types import SimulationResult

DEFAULT_CACHE_DIR = Path.home() / ".wcsim"


def write_cache(result: SimulationResult, meta_det: dict, meta_env: dict,
                cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    """Write last_run.parquet + last_run.meta.json."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"team": iso3, **probs} for iso3, probs in result.probabilities.items()]
    pd.DataFrame(rows).to_parquet(cache_dir / "last_run.parquet", index=False)
    with (cache_dir / "last_run.meta.json").open("w") as f:
        json.dump({"deterministic": meta_det, "environment": meta_env}, f, indent=2)


def read_cache_meta(cache_dir: Path = DEFAULT_CACHE_DIR) -> dict | None:
    """Read last_run.meta.json; returns None if not found."""
    meta_path = cache_dir / "last_run.meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open() as f:
        return json.load(f)
