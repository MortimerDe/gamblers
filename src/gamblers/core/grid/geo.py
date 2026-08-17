from __future__ import annotations

from enum import Enum

from pydantic.dataclasses import dataclass

from gamblers.core.types import Cell

type WorldPos = tuple[float, float]

# A* step costs
STRAIGHT_COST: int = 10
DIAGONAL_COST: int = 14


class Dir8(Enum):
    N = (0, -1)
    NE = (1, -1)
    E = (1, 0)
    SE = (1, 1)
    S = (0, 1)
    SW = (-1, 1)
    W = (-1, 0)
    NW = (-1, -1)

    @property
    def delta(self) -> Cell:
        return self.value

    @property
    def is_diagonal(self) -> bool:
        dx, dy = self.value
        return dx != 0 and dy != 0

    @property
    def step_cost(self) -> int:
        return DIAGONAL_COST if self.is_diagonal else STRAIGHT_COST

    @property
    def opposite(self) -> Dir8:
        dx, dy = self.value
        return _DELTA_TO_DIR8[(-dx, -dy)]

    def apply(self, cell: Cell) -> Cell:
        dx, dy = self.value
        x, y = cell
        return (x + dx, y + dy)

    @staticmethod
    def between(origin: Cell, target: Cell) -> Dir8 | None:
        dx, dy = target[0] - origin[0], target[1] - origin[1]
        return _DELTA_TO_DIR8.get((dx, dy))


_DELTA_TO_DIR8: dict[Cell, Dir8] = {
    (0, -1): Dir8.N,
    (1, -1): Dir8.NE,
    (1, 0): Dir8.E,
    (1, 1): Dir8.SE,
    (0, 1): Dir8.S,
    (-1, 1): Dir8.SW,
    (-1, 0): Dir8.W,
    (-1, -1): Dir8.NW,
}

DIRS_8: tuple[Dir8, ...] = tuple(Dir8)
DIRS_4: tuple[Dir8, ...] = (Dir8.N, Dir8.E, Dir8.S, Dir8.W)


def octile_dist(a: Cell, b: Cell) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return STRAIGHT_COST * (dx + dy) + (DIAGONAL_COST - 2 * STRAIGHT_COST) * min(dx, dy)


def chebyshev_dist(a: Cell, b: Cell) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


@dataclass(frozen=True, slots=True)
class TileGeo:
    tile_size: float = 16.0
    origin: WorldPos = (0.0, 0.0)

    def tile_to_world_center(self, cell: Cell) -> WorldPos:
        x, y = cell
        return (
            self.origin[0] + (x + 0.5) * self.tile_size,
            self.origin[1] + (y + 0.5) * self.tile_size,
        )

    def tile_to_world_corner(self, cell: Cell) -> WorldPos:
        x, y = cell
        return (
            self.origin[0] + x * self.tile_size,
            self.origin[1] + y * self.tile_size,
        )
    
    def world_to_tile(self, pos: WorldPos) -> Cell:
        return (
            int((pos[0] - self.origin[0]) // self.tile_size),
            int((pos[1] - self.origin[1]) // self.tile_size)
        )
