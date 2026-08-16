"""
Single source of truth for the simulation state.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import heapq
from typing import Any, ClassVar

from gamblers.core.agents.base import Agent
from gamblers.core.events import EventSink, EventType
from gamblers.core.machines.base import Machine
from gamblers.core.rng import RngHub
from gamblers.core.types import Action, AgentId, Cell, MachineId, Obs, Outcome, Tick
from gamblers.core.versioning import VerStateMixin


class AgentStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    QUEUED = "queued"
    PLAYING = "playing"


@dataclass
class AgentRuntime:
    """
    Runtime state of an agent.
    """

    agent: Agent[Any]
    capital: int
    position: Cell
    prev_pos: Cell  # interpolation
    status: AgentStatus = AgentStatus.IDLE
    target: MachineId | None = None
    path: list[Cell] = field(default_factory=lambda: list[Cell]())
    path_index: int = 0
    ticks_to_next_cell: int = 0
    last_machine: MachineId | None = None
    last_delta: int | None = None

    pending_obs: Obs | None = None
    pending_action: Action | None = None


@dataclass(order=True, slots=True)
class PendingOutcome:
    """
    heap element of pending outcomes
    """

    sort_tick: int
    seq: int  # sequence number to break ties in the heap (just a sequence number assigned to the outcome)
    agent_id: AgentId = field(compare=False)
    machine_id: MachineId = field(compare=False)
    outcome: Outcome = field(compare=False)


class World(VerStateMixin):
    type_name: ClassVar[str] = "world"
    state_kind: ClassVar[str] = "world"
    state_version: ClassVar[int] = 1

    def __init__(
        self,
        machines: dict[MachineId, Machine[Any]],
        runtimes: dict[AgentId, AgentRuntime],
        rng: RngHub,
        event_log: EventSink,
        ticks_per_cell: int = 1,
        config_hash: str = "",
    ):
        self.tick_count: Tick = Tick(0)
        self.machines = machines
        self.runtimes = runtimes
        self.rng = rng
        self.event_log = event_log
        self.ticks_per_cell = ticks_per_cell
        self.config_hash = config_hash

        self.queues: dict[MachineId, deque[AgentId]] = {m: deque() for m in machines}
        self.occupancy: dict[MachineId, int] = {m: 0 for m in machines}

        self._pending: list[PendingOutcome] = [] # heapq
        self._seq: int = 0

        # the order of traversal (both agents and machines) is fixed in
        # advance once to avoid non-reproducibility
        self._agent_order: tuple[AgentId, ...] = tuple(sorted(runtimes))
        self._machine_order: tuple[MachineId, ...] = tuple(sorted(machines))
        self._machine_ids_tuple: tuple[MachineId, ...] = tuple(sorted(machines))

    def tick(self) -> None:
        """
        Advance the world by one tick.
        """
        self.tick_count = Tick(self.tick_count + 1)

        self._resolve_due_outcomes()  # 1. return ready results
        self._tick_machines()  # 2. machines tick their own state
        self._advance_movement()  # 3. move agents along their paths
        self._serve_queues()  # 4. serve agents waiting in queues
        self._decide()  # 5. free agents decide where to go and what to do

    def _resolve_due_outcomes(self) -> None:
        while self._pending and self._pending[0].sort_tick <= self.tick_count:
            item = heapq.heappop(self._pending)
            rt = self.runtimes[item.agent_id]

            cap_before = rt.capital
            rt.capital += item.outcome.delta
            rt.last_machine = item.machine_id
            rt.last_delta = item.outcome.delta
            self.occupancy[item.machine_id] -= 1

            if rt.pending_obs is not None and rt.pending_action is not None:
                rt.agent.observe(rt.pending_obs, rt.pending_action, item.outcome)

            rt.pending_action = None
            rt.pending_obs = None
            rt.status = AgentStatus.IDLE

            self._log(
                rt,
                EventType.RESULT,
                machine_id=item.machine_id,
                delta=item.outcome.delta,
                capital_before=cap_before,
                capital_after=rt.capital,
                extra=item.outcome.extra,
            )
            


    def _tick_machines(self) -> None:
        pass

    def _advance_movement(self) -> None:
        pass

    def _serve_queues(self) -> None:
        pass

    def _start_play(self, agent_id: AgentId, machine: Machine[Any]) -> None:
        pass

    def _decide(self) -> None:
        pass

    def _log(
        self,
        rt: AgentRuntime,
        et: EventType,
        *,
        machine_id: MachineId | None = None,
        delta: int | None = None,
        capital_before: int | None = None,
        capital_after: int | None = None,
        extra: dict[str, Any] | None = None,
        ) -> None:
        pass
