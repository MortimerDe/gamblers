"""
Single source of truth for the simulation state.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from gamblers.core.agents.base import Agent
from gamblers.core.events import EventSink
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

        self._pending: list[PendingOutcome] = []
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

        # todo resolvers for outcomes, machine ticks, advance movevemnt, serve queues, decide next actions, etc.

