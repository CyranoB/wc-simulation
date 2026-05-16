"""Tests for cache.py."""
from pathlib import Path
import tempfile


def test_write_and_read_cache():
    from wcsim.cache import write_cache, read_cache_meta
    from wcsim.types import SimulationResult
    result = SimulationResult(
        n=10, seed=42,
        probabilities={"ARG": {"Win": 0.3}},
        ci_lo={"ARG": {"Win": 0.2}},
        ci_hi={"ARG": {"Win": 0.4}},
        mean_goals_for={"ARG": 2.0},
        mean_goals_against={"ARG": 0.8},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        write_cache(result, {"seed": 42}, {"runtime": 1.0}, cache_dir=cache_dir)
        assert (cache_dir / "last_run.parquet").exists()
        assert (cache_dir / "last_run.meta.json").exists()
        meta = read_cache_meta(cache_dir=cache_dir)
        assert meta is not None
        assert meta["deterministic"]["seed"] == 42
