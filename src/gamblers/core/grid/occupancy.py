from typing import Any, ClassVar

from gamblers.core.types import AgentId, Cell
from gamblers.core.utils import todo
from gamblers.core.versioning import VerStateMixin


class OccErr(RuntimeError):
    """
    raised on an inconsistent occupancy transition (prob a logic bug)
    """


class Occ(VerStateMixin):
    state_kind: ClassVar[str] = "occupancy"
    state_version: ClassVar[int] = 1

    def __init__(self) -> None:
        # cell -> [agent1, agent2, ...] (that are physically present in the cell)
        self._present: dict[Cell, list[AgentId]] = {}
        # cell -> [agent1, agent2, ...] (that have reserved the cell but are not yet physically present)
        self._reserved: dict[Cell, list[AgentId]] = {}
        # symmetric to the above, but for fast lookup of which cell an agent is currently present in
        # agent -> cell (that the agent is physically present in)
        self._agent_cell: dict[AgentId, Cell] = {}
        # agent -> cell (that the agent has reserved but is not yet physically present in)
        self._agent_reservation: dict[AgentId, Cell] = {}

    # queries

    def present_count(self, cell: Cell) -> int:
        return len(self._present.get(cell, ()))

    def reserved_count(self, cell: Cell) -> int:
        return len(self._reserved.get(cell, ()))

    def load(self, cell: Cell) -> int:
        return self.present_count(cell) + self.reserved_count(cell)

    def agents_at(self, cell: Cell) -> tuple[AgentId, ...]:
        return tuple(self._present.get(cell, ()))

    def reservation_of(self, agent_id: AgentId) -> Cell | None:
        return self._agent_reservation.get(agent_id)

    def has_room(self, cell: Cell, cap: int, *, for_agent: AgentId | None = None) -> bool:
        if for_agent is not None:
            if self._agent_cell.get(for_agent) == cell:
                return True
            if self._agent_reservation.get(for_agent) == cell:
                return True
        return self.load(cell) < cap

    # mut

    def place(self, agent_id: AgentId, cell: Cell) -> None:
        todo()

    def remove(self, agent_id: AgentId) -> None:
        todo()

    def reserve(self, agent_id: AgentId, cell: Cell) -> None:
        todo()

    def release_reservation(self, agent_id: AgentId) -> None:
        todo()

    def commit_step(self, agent_id: AgentId) -> Cell:
        todo()

    @staticmethod
    def _remove_from(index: dict[Cell, list[AgentId]], cell: Cell, agent_id: AgentId) -> None:
        bucket = index.get(cell)
        if bucket is None or agent_id not in bucket:
            raise OccErr(f"agent {agent_id} is not registered in cell {cell}")
        bucket.remove(agent_id)
        if not bucket:
            del index[cell]

    # de/serialization

    def _state_payload(self) -> dict[str, Any]:
        return {
            "present": [
                [int(agent_id), list(cell)]
                for cell, bucket in sorted(self._present.items())
                for agent_id in bucket
            ],
            "reserved": [
                [int(agent_id), list(cell)]
                for cell, bucket in sorted(self._reserved.items())
                for agent_id in bucket
            ],
        }

    def _apply_state(self, payload: dict[str, Any]) -> None:
        self._present.clear()
        self._reserved.clear()
        self._agent_cell.clear()
        self._agent_reservation.clear()
        for agent_id, cell in payload["present"]:
            self.place(AgentId(int(agent_id)), (int(cell[0]), int(cell[1])))
        for agent_id, cell in payload["reserved"]:
            self.reserve(AgentId(int(agent_id)), (int(cell[0]), int(cell[1])))