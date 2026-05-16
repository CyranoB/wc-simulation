"""Tournament-simulation engine. Supports WC 2018/2022 (8 groups x 4 = 32
teams, R16 first round, 3rd-place playoff) and WC 2026 (12 groups x 4 = 48,
R32 first round, no 3rd-place playoff). Dispatched by team count."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TournamentStructure:
    name: str
    groups_count: int
    group_size: int
    top_per_group: int
    best_thirds: int
    knockout_stages: list[str] = field(default_factory=list)
    third_place_playoff: bool = False


STRUCTURE_2018_2022 = TournamentStructure(
    name="WC2018-2022",
    groups_count=8, group_size=4,
    top_per_group=2, best_thirds=0,
    knockout_stages=["R16", "QF", "SF", "Final"],
    third_place_playoff=True,
)

STRUCTURE_2026 = TournamentStructure(
    name="WC2026",
    groups_count=12, group_size=4,
    top_per_group=2, best_thirds=8,
    knockout_stages=["R32", "R16", "QF", "SF", "Final"],
    third_place_playoff=False,
)


def _structure_for(team_count: int) -> TournamentStructure:
    if team_count == 32:
        return STRUCTURE_2018_2022
    if team_count == 48:
        return STRUCTURE_2026
    raise ValueError(
        f"Unsupported tournament size: {team_count} teams "
        "(supported: 32 for WC 2018/2022, 48 for WC 2026)"
    )
