from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from gamblers.core.grid.geo import Dir8
from gamblers.core.grid.occupancy import Occ
from gamblers.core.grid.tilemap import TileMap
from gamblers.core.types import AgentId, Cell


class NavView(Protocol):
    """pathfinder view of grid"""
    def is_passable(self, cell: Cell) -> bool: ...
    def enter_cost(self, cell: Cell) -> int: ...
    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, Dir8]]: ...

@dataclass(slots=True)
class NavGrid:
    """costs and passability for a single pathfinding query"""
    tilemap: TileMap
    occ: Occ
    # extra cost per agent present in a cell
    # my guess is that this will make agents spread
    # out and route around crowds but we'll see
    congestion_weight: int = 8
    for_agent: AgentId | None = None
    goal: Cell | None = None

    def is_passable(self, cell: Cell) -> bool:
        if not self.tilemap.is_walkable(cell):
            return False
        if cell == self.goal:
            return True
        cap = self.tilemap.cap_at(cell)
        return self.occ.has_room(cell, cap, for_agent=self.for_agent)

    def enter_cost(self, cell: Cell) -> int:
        """entrance fee or something like that"""
        crowds = self.occ.load(cell)
        return self.tilemap.base_cost_at(cell) + self.congestion_weight * crowds

    def neighbours(self, cell: Cell) -> Iterator[tuple[Cell, Dir8]]:
        for nbr, dir in self.tilemap.walkable_neighbours(cell):
            if self.is_passable(nbr):
                yield nbr, dir