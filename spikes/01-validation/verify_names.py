"""Resolve every WC participant name through the ISO3 map and report status.

Run from the spike root: `python verify_names.py`. Fails loudly (exit 1)
if any name in elo_history.csv, fifa_ranking.csv (limited to teams that
also appear in elo or matches), or matches_history.csv can't be mapped to
an ISO3 code. The FIFA dataset contains ~211 teams per snapshot and most
of them aren't WC participants — we only verify the ones we actually need.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from name_to_iso3 import to_iso3

HERE = Path(__file__).parent
DATA = HERE / "data" / "raw"


def names_from(path: Path, cols: list[str]) -> set[str]:
    out: set[str] = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            for c in cols:
                v = row.get(c)
                if v:
                    out.add(v)
    return out


def main() -> int:
    elo_names = names_from(DATA / "elo_history.csv", ["team"])
    matches_names = names_from(DATA / "matches_history.csv", ["home_team", "away_team"])
    fifa_all = names_from(DATA / "fifa_ranking.csv", ["country_full"])

    # Among FIFA's ~211 names, we only need to resolve the ones that overlap
    # with WC participants from the other sources. The rest aren't loaded.
    needed = elo_names | matches_names
    fifa_needed = fifa_all & needed

    # But: some FIFA names use FIFA-specific spellings that aren't in the
    # other sources (e.g., "IR Iran", "Korea Republic", "USA" when matches
    # uses the same "USA" already). Add likely FIFA-variant names of any
    # ISO3 that's already covered.
    fifa_likely_variants = {n for n in fifa_all if n in {"IR Iran", "Korea Republic"}}
    to_check = needed | fifa_needed | fifa_likely_variants

    print(f"Verifying {len(to_check)} unique participant names across 3 sources...")
    print(f"  Elo: {len(elo_names)} names")
    print(f"  Matches: {len(matches_names)} names")
    print(f"  FIFA (participants only): {len(fifa_needed)} names "
          f"({len(fifa_all)} total, {len(fifa_all) - len(fifa_needed)} non-participant)")
    print(f"  FIFA-specific variants: {len(fifa_likely_variants)} names")

    failures: list[str] = []
    for name in sorted(to_check):
        try:
            iso3 = to_iso3(name)
        except KeyError as e:
            failures.append(f"  {name!r}: {e}")

    if failures:
        print(f"\nFAILED: {len(failures)} name(s) have no ISO3 mapping:")
        for f in failures:
            print(f, file=sys.stderr)
        return 1

    print(f"\nOK: all {len(to_check)} names resolved to ISO3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
