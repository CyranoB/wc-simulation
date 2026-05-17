"""Hand-curated mapping from team-name variants (across Elo / FIFA / matches
sources) to canonical ISO3 codes. Populated by Task 3 from a survey of the
three bundled CSVs in data/raw/.

The 40 unique WC 2018+2022 participants appear under three slightly different
naming conventions:
- eloratings.net (via our scraper) uses canonical English names like
  "Saudi Arabia", "South Korea", "United States".
- openfootball/worldcup.json (via our scraper) uses similar English names but
  with "USA" instead of "United States".
- inside.fifa.com uses FIFA's own conventions: "Korea Republic", "IR Iran",
  "USA". Note: "Korea DPR" is North Korea (not a WC participant in 2018/2022).

Three countries need alias entries to cover all three sources:
- Iran: "Iran" (Elo, matches) + "IR Iran" (FIFA)
- South Korea: "South Korea" (Elo, matches) + "Korea Republic" (FIFA)
- USA: "United States" (Elo) + "USA" (matches, FIFA)
"""

NAME_TO_ISO3: dict[str, str] = {
    # WC 2018 + 2022 participants — all 40 unique countries.
    "Argentina": "ARG",
    "Australia": "AUS",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "Cameroon": "CMR",
    "Canada": "CAN",
    "Colombia": "COL",
    "Costa Rica": "CRC",
    "Croatia": "CRO",
    "Denmark": "DEN",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "England": "ENG",
    "France": "FRA",
    "Germany": "GER",
    "Ghana": "GHA",
    "Iceland": "ISL",
    "Iran": "IRN",
    "IR Iran": "IRN",  # FIFA's spelling
    "Japan": "JPN",
    "Korea Republic": "KOR",  # FIFA's spelling
    "South Korea": "KOR",
    "Mexico": "MEX",
    "Morocco": "MAR",
    "Netherlands": "NED",
    "Nigeria": "NGA",
    "Panama": "PAN",
    "Peru": "PER",
    "Poland": "POL",
    "Portugal": "POR",
    "Qatar": "QAT",
    "Russia": "RUS",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Tunisia": "TUN",
    "United States": "USA",
    "USA": "USA",
    "Uruguay": "URU",
    "Wales": "WAL",
    # WC 2026 additions (countries not in WC 2018/2022):
    "Algeria": "ALG",
    "Austria": "AUT",
    "Bolivia": "BOL",
    "Cape Verde": "CPV",
    "Cape Verde Islands": "CPV",
    "Cabo Verde": "CPV",
    "Curaçao": "CUW",
    "Curacao": "CUW",
    "Haiti": "HAI",
    "Italy": "ITA",
    "Ivory Coast": "CIV",
    "Côte d'Ivoire": "CIV",
    "Cote d'Ivoire": "CIV",
    "Jamaica": "JAM",
    "Jordan": "JOR",
    "New Zealand": "NZL",
    "Norway": "NOR",
    "Paraguay": "PAR",
    "Scotland": "SCO",
    "South Africa": "RSA",
    "Turkey": "TUR",
    "Türkiye": "TUR",
    "Ukraine": "UKR",
    "Uzbekistan": "UZB",
    # WC 2026 corrections (teams that qualified after initial draw file)
    "Bosnia-Herzegovina": "BIH",
    "Bosnia and Herzegovina": "BIH",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Sweden": "SWE",
    "Iraq": "IRQ",
    "DR Congo": "COD",
    "Congo DR": "COD",
    "Dem. Rep. Congo": "COD",
}


def to_iso3(name: str) -> str:
    iso3 = NAME_TO_ISO3.get(name.strip())
    if iso3 is None:
        raise KeyError(f"No ISO3 mapping for {name!r}; add it to name_to_iso3.py")
    return iso3
