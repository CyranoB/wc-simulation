"""Tests for wcsim.squad_data module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wcsim.squad_data import load_squads


def test_load_squads_returns_all_48_teams():
    squads = load_squads()
    draw_path = Path(__file__).parent.parent / "spikes" / "01-validation" / "data" / "raw" / "wc2026_draw.json"
    with draw_path.open() as f:
        draw = json.load(f)
    all_iso3 = {iso3 for group in draw.values() for iso3 in group}
    for iso3 in all_iso3:
        assert iso3 in squads, f"Missing squad data for {iso3}"
        assert squads[iso3]["total_value_eur"] >= 0


def test_load_squads_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_squads(tmp_path / "nonexistent.json")


def test_load_squads_validates_schema_missing_squads_key(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"provenance": {}}))
    with pytest.raises(ValueError, match="missing 'squads' key"):
        load_squads(bad)


def test_load_squads_validates_schema_missing_total_value(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"squads": {"AAA": {"players": []}}}))
    with pytest.raises(ValueError, match="missing 'total_value_eur'"):
        load_squads(bad)
