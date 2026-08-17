# pyright: reportPrivateUsage=false

from typing import Any, ClassVar

import numpy as np

from gamblers.core.grid.geo import DIRS_8, Dir8, chebyshev_dist
from gamblers.core.grid.tilemap import TileMap
from gamblers.core.types import AgentId, Cell, MachineId

STREAM_QUEUE: str = "queue:{machine_id}"

# relative weights for chosing where the tail grows next
WEIGHT_STRAIGHT: float = 6.0
WEIGHT_SOFT_TURN: float = 2.0  # 45 deg
WEIGHT_HARD_TURN: float = 0.5  # 90 deg


class QueueFullErr(RuntimeError):
    """raised when the tail has nowhere left to grow"""


class SpatialQueue:
    def __init__(
        self,
        machine_id: MachineId,
        interaction_cell: Cell,
        tilemap: TileMap,
        max_length: int = 32,
    ):
        self.machine_id = machine_id
        self.interaction_cell = interaction_cell
        self.tilemap = tilemap
        self.max_length = max_length

        self._slot_cells: list[Cell] = [interaction_cell]
        self._slot_agents: list[AgentId | None] = [None]
        self._agent_slots: dict[AgentId, int] = {}

    # queries
    def __len__(self) -> int:
        return len(self._agent_slots)

    @property
    def tail_cell(self) -> Cell:
        return self._slot_cells[-1]

    def slot_of(self, agent_id: AgentId) -> int | None:
        return self._agent_slots.get(agent_id)

    def cell_of_slot(self, slot: int) -> Cell:
        return self._slot_cells[slot]

    def target_cell(self, agent_id: AgentId) -> Cell | None:
        """where this agent should curr be standing"""
        slot = self._agent_slots.get(agent_id)
        return None if slot is None else self._slot_cells[slot]

    def front_agent(self) -> AgentId | None:
        return self._slot_agents[0]

    def lane_cells(self) -> tuple[Cell, ...]:
        return tuple(self._slot_cells)

    # mutations
    def reserve_slot(self, agent_id: AgentId, rng: np.random.Generator) -> int:
        """give the agent the next free slot"""
        if agent_id in self._agent_slots:
            return self._agent_slots[agent_id]
        slot = self._first_free_slot()
        if slot is None:
            slot = self._grow(rng)
        self._slot_agents[slot] = agent_id
        self._agent_slots[agent_id] = slot
        return slot

    def release(self, agent_id: AgentId) -> None:
        slot = self._agent_slots.pop(agent_id, None)
        if slot is None:
            return
        self._slot_agents[slot] = None
        self._compact()

    def _first_free_slot(self) -> int | None:
        for i, occupant in enumerate(self._slot_agents):
            if occupant is None:
                return i
        return None

    def _compact(self) -> None:
        """shift forward and trim"""
        occupants = [a for a in self._slot_agents if a is not None]
        self._slot_agents = [None] * len(self._slot_cells)
        self._agent_slots.clear()
        for i, agent_id in enumerate(occupants):
            self._slot_agents[i] = agent_id
            self._agent_slots[agent_id] = i

    def _grow(self, rng: np.random.Generator) -> int:
        """append one new slot at the tail"""
        if len(self._slot_agents) >= self.max_length:
            raise QueueFullErr(f"queue {self.machine_id} is full")
        tail = self.tail_cell
        slots_on_tail = sum(1 for cell in self._slot_cells if cell == tail)
        if slots_on_tail < self.tilemap.cap_at(tail):
            self._slot_cells.append(tail)
            self._slot_agents.append(None)
            return len(self._slot_agents) - 1
        next_cell = self._pick_next_cell(rng)
        if next_cell is None:
            raise QueueFullErr(
                f"{self.machine_id}: no free cell to extend the queue past {tail}"
            )
        self._slot_cells.append(next_cell)
        self._slot_agents.append(None)
        return len(self._slot_agents) - 1

    def _pick_next_cell(self, rng: np.random.Generator) -> Cell | None:
        tail = self.tail_cell
        incoming = self._tail_dir()
        cands: list[Cell] = []
        weights: list[float] = []
        used = set(self._slot_cells)

        for nbr, dir in self.tilemap.walkable_neighbours(tail):
            if nbr in used:
                continue
            if chebyshev_dist(nbr, self.interaction_cell) <= chebyshev_dist(
                tail, self.interaction_cell
            ):
                continue
            if self.tilemap.cap_at(nbr) <= 0:
                continue
            cands.append(nbr)
            weights.append(self._growth_weight(incoming, dir))
        if not cands:
            return None
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = int(rng.choice(len(cands), p=probs))
        return cands[chosen]

    def _tail_dir(self) -> Dir8 | None:
        """direction the lane is currently heading in if it has one yet"""
        for index in range(len(self._slot_cells) - 1, 0, -1):
            prev = self._slot_cells[index - 1]
            curr = self._slot_cells[index]
            if prev != curr:
                return Dir8.between(prev, curr)
        return None

    @staticmethod
    def _growth_weight(incoming: Dir8 | None, candidate: Dir8) -> float:
        if incoming is None:
            return WEIGHT_SOFT_TURN
        turn = (DIRS_8.index(candidate) - DIRS_8.index(incoming)) % 8
        turn = min(turn, 8 - turn)
        if turn == 0:
            return WEIGHT_STRAIGHT
        if turn == 1:
            return WEIGHT_SOFT_TURN
        if turn == 2:
            return WEIGHT_HARD_TURN
        return 0.01


class QueueRegistry:
    state_kind: ClassVar[str] = "queue_registry"

    def __init__(self, tilemap: TileMap, max_length: int = 32) -> None:
        self.tilemap = tilemap
        self._queues: dict[MachineId, SpatialQueue] = {
            placement.machine_id: SpatialQueue(
                machine_id=placement.machine_id,
                interaction_cell=placement.interaction_cell,
                tilemap=tilemap,
                max_length=max_length,
            )
            for placement in tilemap.placements()
        }

    def __getitem__(self, machine_id: MachineId) -> SpatialQueue:
        return self._queues[machine_id]

    def all(self) -> list[SpatialQueue]:
        return [self._queues[m] for m in sorted(self._queues)]

    def release_everywhere(self, agent_id: AgentId) -> None:
        for queue in self.all():
            queue.release(agent_id)

    def to_payload(self) -> dict[str, Any]:
        return {
            str(queue.machine_id): {
                "slot_cells": [list(c) for c in queue._slot_cells],
                "slot_agents": [
                    None if a is None else int(a) for a in queue._slot_agents
                ],
            }
            for queue in self.all()
        }

    def from_payload(self, payload: dict[str, Any]) -> None:
        for machine_id_str, data in payload.items():
            queue = self._queues[MachineId(machine_id_str)]
            queue._slot_cells = [(int(x), int(y)) for x, y in data["slot_cells"]]
            queue._slot_agents = [
                None if a is None else AgentId(int(a)) for a in data["slot_agents"]
            ]
            queue._agent_slots = {
                agent_id: index
                for index, agent_id in enumerate(queue._slot_agents)
                if agent_id is not None
            }
