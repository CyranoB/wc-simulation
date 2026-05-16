"""Hand-curated mapping from team-name variants (across Elo / FIFA / matches
sources) to canonical ISO3 codes. Populated by Task 3."""

NAME_TO_ISO3: dict[str, str] = {}


def to_iso3(name: str) -> str:
    iso3 = NAME_TO_ISO3.get(name.strip())
    if iso3 is None:
        raise KeyError(f"No ISO3 mapping for {name!r}; add it to name_to_iso3.py")
    return iso3
