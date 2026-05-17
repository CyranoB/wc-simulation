"""Process Kaggle davidcariboo/player-scores players.csv into per-team squad
values for WC 2026 participants. Outputs squads_2026.json."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
PLAYERS_CSV = DATA_DIR / "players.csv"
DRAW_JSON = DATA_DIR / "wc2026_draw.json"
OUTPUT_JSON = DATA_DIR / "squads_2026.json"

TOP_N = 23

CITIZENSHIP_TO_ISO3: dict[str, str] = {
    "Argentina": "ARG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Algeria": "ALG",
    "Belgium": "BEL",
    "Bosnia-Herzegovina": "BIH",
    "Brazil": "BRA",
    "Canada": "CAN",
    "Cape Verde": "CPV",
    "Colombia": "COL",
    "Cote d'Ivoire": "CIV",
    "Croatia": "CRO",
    "Curacao": "CUW",
    "Curaçao": "CUW",
    "Congo DR": "COD",
    "DR Congo": "COD",
    "Czech Republic": "CZE",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Haiti": "HAI",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Japan": "JPN",
    "Jordan": "JOR",
    "Korea, South": "KOR",
    "South Korea": "KOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Panama": "PAN",
    "Paraguay": "PAR",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Saudi Arabia": "SAU",
    "Scotland": "SCO",
    "Senegal": "SEN",
    "South Africa": "RSA",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Ukraine": "UKR",
    "United States": "USA",
    "Uruguay": "URU",
    "Uzbekistan": "UZB",
}


def process() -> dict:
    with DRAW_JSON.open() as f:
        draw = json.load(f)
    wc_iso3s = {iso3 for group in draw.values() for iso3 in group}

    players_by_team: dict[str, list[dict]] = {iso3: [] for iso3 in wc_iso3s}

    with PLAYERS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cit = row["country_of_citizenship"]
            iso3 = CITIZENSHIP_TO_ISO3.get(cit)
            if iso3 is None or iso3 not in wc_iso3s:
                continue
            mv_str = row["market_value_in_eur"]
            if not mv_str:
                continue
            try:
                mv = int(mv_str)
            except ValueError:
                continue
            players_by_team[iso3].append({
                "name": row["name"],
                "position": row.get("position", ""),
                "value_eur": mv,
                "club": row.get("current_club_name", ""),
                "age": row.get("date_of_birth", ""),
            })

    squads: dict[str, dict] = {}
    for iso3 in sorted(wc_iso3s):
        players = sorted(players_by_team[iso3], key=lambda p: -p["value_eur"])
        top = players[:TOP_N]
        squads[iso3] = {
            "total_value_eur": sum(p["value_eur"] for p in top),
            "player_count": len(players),
            "top_n_used": len(top),
            "players": [
                {
                    "name": p["name"],
                    "position": p["position"],
                    "value_eur": p["value_eur"],
                    "club": p["club"],
                }
                for p in top
            ],
        }

    output = {
        "provenance": {
            "source": "Kaggle davidcariboo/player-scores",
            "file": "players.csv",
            "download_date": str(date.today()),
            "description": "Market values from Transfermarkt via Kaggle mirror",
            "top_n": TOP_N,
            "teams": len(squads),
        },
        "squads": squads,
    }
    return output


def main():
    output = process()
    missing = [iso3 for iso3, d in output["squads"].items() if d["top_n_used"] == 0]
    if missing:
        print(f"WARNING: No players found for: {missing}")

    with OUTPUT_JSON.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Written {OUTPUT_JSON}")
    print(f"Teams: {output['provenance']['teams']}")
    for iso3, d in sorted(output["squads"].items(), key=lambda kv: -kv[1]["total_value_eur"])[:10]:
        print(f"  {iso3}: €{d['total_value_eur']/1e6:.0f}M ({d['top_n_used']} players)")

    low = [(iso3, d) for iso3, d in output["squads"].items() if d["top_n_used"] < 15]
    if low:
        print(f"\nLow coverage teams (<15 players):")
        for iso3, d in sorted(low):
            print(f"  {iso3}: {d['top_n_used']} players, €{d['total_value_eur']/1e6:.0f}M")


if __name__ == "__main__":
    main()
