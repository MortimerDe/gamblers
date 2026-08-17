from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum

from gamblers.core.grid.geo import DIRS_8, Dir8
from gamblers.core.types import Cell, MachineId
from gamblers.core.versioning import VerStateMixin

BLOCKED_COST: int = -1


class Terrain(IntEnum):
    VOID = 0
    FLOOR = 1
    CARPET = 2
    WALL = 3
    ENTRANCE = 4


@dataclass(frozen=True, slots=True)
class TerrainProps:
    walkable: bool
    base_cost: int
    cap: int


TERRAIN_PROPS: dict[Terrain, TerrainProps] = {
    Terrain.VOID: TerrainProps(walkable=False, base_cost=BLOCKED_COST, cap=0),
    Terrain.WALL: TerrainProps(walkable=False, base_cost=BLOCKED_COST, cap=0),
    Terrain.FLOOR: TerrainProps(walkable=True, base_cost=10, cap=4),
    Terrain.CARPET: TerrainProps(walkable=True, base_cost=8, cap=4),
    Terrain.ENTRANCE: TerrainProps(walkable=True, base_cost=10, cap=6),
}


@dataclass(frozen=True, slots=True)
class Footprint:
    """
    an axis aligned `W x H` rect of tiles, anchored at `origin`
    """

    origin: Cell
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"footprint dims must be positive, got ({self.width}, {self.height})"
            )

    def cells(self) -> Iterator[Cell]:
        ox, oy = self.origin
        for dy in range(self.height):
            for dx in range(self.width):
                yield (ox + dx, oy + dy)

    def contains(self, cell: Cell) -> bool:
        ox, oy = self.origin
        return ox <= cell[0] < ox + self.width and oy <= cell[1] < oy + self.height

    def border_cells(self) -> Iterator[Cell]:
        _ox, _oy = self.origin
        seen: set[Cell] = set()
        for cell in self.cells():
            for direction in DIRS_8:
                neighbour = direction.apply(cell)
                if not self.contains(neighbour) and neighbour not in seen:
                    seen.add(neighbour)
                    yield neighbour


@dataclass(frozen=True, slots=True)
class MachinePlacement:
    """
    where a machine physically sits and where an agent must stant to use it
    """

    machine_id: MachineId
    footprint: Footprint
    interaction_cell: Cell  # todo maybe make this a list of cells

    def __post_init__(self) -> None:
        if self.footprint.contains(self.interaction_cell):
            raise ValueError(
                f"{self.machine_id}: interaction_cell {self.interaction_cell} "
                "is inside the machine footprint (unreachable)"
            )
        if Dir8.between(
            self.interaction_cell, self.footprint.origin
        ) is None and not any(
            Dir8.between(self.interaction_cell, c) is not None
            for c in self.footprint.cells()
        ):
            # not strictly required but a distant interaction cell is almost always a config mistake
            raise ValueError(
                f"{self.machine_id}: interaction_cell {self.interaction_cell} "
                "is not adjacent to the footprint"
            )

class MapValidationError(RuntimeError):
    """
    raised when a map cannot possibly work (unreachable machines, etc.)
    """

class TileMap(VerStateMixin):
    pass
