from __future__ import annotations

import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar

import numpy as np

from gamblers.core.grid.geo import DIRS_8, Dir8
from gamblers.core.types import Cell, MachineId
from gamblers.core.utils import todo
from gamblers.core.versioning import StateVerErr, VerStateMixin

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
    state_kind: ClassVar[str] = "tilemap"
    state_version: ClassVar[int] = 1

    def __init__(self, terrain: np.ndarray) -> None:
        if terrain.ndim != 2:
            raise ValueError(f"terrain must be 2D, got {terrain.ndim}D")
        self.terrain: np.ndarray = terrain.astype(np.int8, copy=True)
        self.height: int = int(terrain.shape[0])
        self.width: int = int(terrain.shape[1])

        self._base_cost: np.ndarray = np.zeros(
            (self.height, self.width), dtype=np.int32
        )
        self._cap: np.ndarray = np.zeros((self.height, self.width), dtype=np.int8)
        self._rebuild_derived_layers()

        self._objects: dict[Cell, MachineId] = {}
        self._placements: dict[MachineId, MachinePlacement] = {}

    # construction

    @classmethod
    def empty(cls, width: int, height: int, fill: Terrain = Terrain.FLOOR) -> TileMap:
        return cls(np.full((height, width), int(fill), dtype=np.int8))

    def _rebuild_derived_layers(self) -> None:
        for terr, props in TERRAIN_PROPS.items():
            mask = self.terrain == int(terr)
            self._base_cost[mask] = props.base_cost
            self._cap[mask] = props.cap

    def set_terrain(self, cell: Cell, kind: Terrain) -> None:
        self.require_in_bounds(cell)
        x, y = cell
        props = TERRAIN_PROPS[kind]
        self.terrain[y, x] = int(kind)
        self._base_cost[y, x] = props.base_cost
        self._cap[y, x] = props.cap

    def fill_rect(self, footprint: Footprint, kind: Terrain) -> None:
        for c in footprint.cells():
            self.set_terrain(c, kind)

    # queres

    def in_bounds(self, cell: Cell) -> bool:
        todo()

    def require_in_bounds(self, cell: Cell) -> None:
        todo()

    def terrain_at(self, cell: Cell) -> Terrain:
        todo()

    def is_walkable(self, cell: Cell) -> bool:
        todo()

    def base_cost_at(self, cell: Cell) -> int:
        todo()

    def cap_at(self, cell: Cell) -> int:
        todo()

    def set_cap(self, cell: Cell, cap: int) -> None:
        todo()

    def object_at(self, cell: Cell) -> MachineId | None:
        todo()

    def walkable_neighbours(self, cell: Cell) -> Iterator[tuple[Cell, Dir8]]:
        todo()

    def place_machine(self, placement: MachinePlacement) -> None:
        todo()

    def placement(self, machine_id: MachineId) -> MachinePlacement:
        todo()

    def placements(self) -> Iterator[MachinePlacement]:
        todo()

    # validation

    def reachable_from(self, start: Cell) -> set[Cell]:
        todo()

    def validate(self, spawn_cells: list[Cell]) -> None:
        todo()

    # de/serialization
    def terrain_checksum(self) -> int:
        return zlib.crc32(self.terrain.tobytes())

    def _state_payload(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "terrain_checksum": self.terrain_checksum(),
            "placements": [
                {
                    "machine_id": str(p.machine_id),
                    "origin": list(p.footprint.origin),
                    "size": [p.footprint.width, p.footprint.height],
                    "interaction_cell": list(p.interaction_cell),
                }
                for p in self.placements()
            ],
        }

    def _apply_state(self, payload: dict[str, Any]) -> None:
        if payload["width"] != self.width or payload["height"] != self.height:
            raise StateVerErr(
                f"map size in checkpoint {payload['width']}x{payload['height']} "
                f"differs from config {self.width}x{self.height}"
            )
        if payload["terrain_checksum"] != self.terrain_checksum():
            raise StateVerErr("map terrain changed since the checkpoint was written")

        saved = {p["machine_id"] for p in payload["placements"]}
        current = {str(m) for m in self._placements}
        if saved != current:
            raise StateVerErr(
                f"machine placements differ: checkpoint {sorted(saved)} "
                f"vs config {sorted(current)}"
            )
