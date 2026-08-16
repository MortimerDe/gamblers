"""
Single source of truth for the simulation state.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from gamblers.core.agents.base import Agent
from gamblers.core.events import Event, EventSink, EventType
from gamblers.core.machines.base import Machine
from gamblers.core.rng import RngHub
from gamblers.core.types import (
    Action,
    ActionKind,
    AgentId,
    Cell,
    MachineId,
    Obs,
    Outcome,
    Tick,
)
from gamblers.core.utils import *
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
    prev_position: Cell  # interpolation
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

        self._pending: list[PendingOutcome] = []  # heapq
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
        for machine_id in self._machine_order:
            self.machines[machine_id].tick(self.tick_count)

    def _advance_movement(self) -> None:
        for agent_id in self._agent_order:
            rt = self.runtimes[agent_id]
            if rt.status is not AgentStatus.MOVING:
                continue
            rt.ticks_to_next_cell -= 1
            if rt.ticks_to_next_cell > 0:
                continue
            rt.path_index += 1
            rt.prev_position = rt.position
            rt.position = rt.path[rt.path_index]
            if rt.path_index >= len(rt.path) - 1:
                # we have arrived and are joining the queue.
                # the game will start in the serve_queues phase
                assert rt.target is not None
                rt.status = AgentStatus.QUEUED
                self.queues[rt.target].append(agent_id)
            else:
                rt.ticks_to_next_cell = self.ticks_per_cell

    def _serve_queues(self) -> None:
        for machine_id in self._machine_order:
            machine = self.machines[machine_id]
            queue = self.queues[machine_id]
            while queue:
                cap = machine.cap
                if cap is not None and self.occupancy[machine_id] >= cap:
                    break  # machine is full
                agent_id = queue.popleft()
                self._start_play(agent_id, machine)

    def _start_play(self, agent_id: AgentId, machine: Machine[Any]) -> None:
        rt = self.runtimes[agent_id]
        rng = self.rng.machine_stream(str(machine.machine_id))
        outcome = machine.play(agent_id, rt.capital, rng)
        self.occupancy[machine.machine_id] += 1
        rt.status = AgentStatus.PLAYING

        self._seq += 1
        heapq.heappush(
            self._pending,
            PendingOutcome(
                sort_tick=self.tick_count + outcome.delay_ticks,
                seq=self._seq,
                agent_id=agent_id,
                machine_id=machine.machine_id,
                outcome=outcome,
            ),
        )

        self._log(
            rt,
            EventType.PLAY_START,
            machine_id=machine.machine_id,
            capital_before=rt.capital,
        )

    def _decide(self) -> None:
        for agent_id in self._agent_order:
            rt = self.runtimes[agent_id]
            if rt.status is not AgentStatus.IDLE:
                continue
            obs = self.observe(agent_id)
            act = rt.agent.act(obs, self.rng.agent_stream(str(agent_id)))
            self._apply_action(agent_id, obs, act)

    def observe(self, agent_id: AgentId) -> Obs:
        rt = self.runtimes[agent_id]
        return Obs(
            tick=self.tick_count,
            agent_id=agent_id,
            capital=rt.capital,
            position=rt.position,
            available_machines=self._machine_ids_tuple,
            queue_lengths={m: len(q) for m, q in self.queues.items()},
            last_machine=rt.last_machine,
            last_delta=rt.last_delta,
        )

    def _apply_action(self, agent_id: AgentId, obs: Obs, action: Action) -> None:
        rt = self.runtimes[agent_id]

        if action.kind is ActionKind.IDLE:
            return

        if action.kind is ActionKind.LEAVE_QUEUE:
            for queue in self.queues.values():
                if agent_id in queue:
                    queue.remove(agent_id)
            rt.status = AgentStatus.IDLE
            rt.target = None
            self._log(rt, EventType.LEAVE_QUEUE)
            return

        machine_id = action.machine_id
        # hope it's guaranteed by Actions.__post_init__
        assert machine_id is not None
        machine = self.machines.get(machine_id)
        if machine is None:
            self._log(
                rt,
                EventType.IDLE,
                extra={"reason": "unknown_machine", "requested": str(machine_id)},
            )
            return

        rt.pending_obs = obs
        rt.pending_action = action
        rt.target = machine_id
        rt.path = self.find_path(rt.position, machine.position)
        rt.path_index = 0

        if len(rt.path) <= 1:
            rt.status = AgentStatus.QUEUED
            self.queues[machine_id].append(agent_id)
        else:
            rt.status = AgentStatus.MOVING
            rt.ticks_to_next_cell = self.ticks_per_cell

        self._log(
            rt, EventType.CHOOSE, machine_id=machine_id, capital_before=rt.capital
        )

    def find_path(self, start: Cell, goal: Cell) -> list[Cell]:
        # todo:
        # for now just manhattan
        # later cached A* or something like that
        cells: list[Cell] = [start]
        x, y = start
        gx, gy = goal
        step_x = 1 if gx > x else -1
        while x != gx:
            x += step_x
            cells.append((x, y))
        step_y = 1 if gy > y else -1
        while y != gy:
            y += step_y
            cells.append((x, y))
        return cells

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
        self.event_log.append(
            Event(
                tick=int(self.tick_count),
                agent_id=int(rt.agent.agent_id),
                agent_type=rt.agent.agent_type,
                agent_label=rt.agent.label,
                event=et,
                machine_id=machine_id,
                delta=delta,
                capital_before=capital_before,
                capital_after=capital_after,
                extra=extra or {},
            )
        )
