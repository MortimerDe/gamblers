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
from gamblers.core.grid.nav_grid import NavGrid
from gamblers.core.grid.occupancy import Occ
from gamblers.core.grid.pathfinder import Path, Pathfinder, ReplanPolicy, sidestep
from gamblers.core.grid.queues import STREAM_QUEUE, QueueFullErr, QueueRegistry
from gamblers.core.grid.tilemap import TileMap
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
    action_from_dict,
    action_to_dict,
    observation_from_dict,
    observation_to_dict,
    outcome_from_dict,
    outcome_to_dict,
)
from gamblers.core.versioning import StateVerErr, VerStateMixin


class AgentStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    QUEUED = "queued"
    PLAYING = "playing"


@dataclass
class AgentRuntime:
    agent: Agent[Any]
    capital: int
    position: Cell
    prev_position: Cell  # interpolation

    status: AgentStatus = AgentStatus.IDLE
    target: MachineId | None = None

    path: Path | None = None
    queue_slot: int | None = None
    ticks_to_next_cell: int = 0
    ticks_since_replan: int = 0

    last_machine: MachineId | None = None
    last_delta: int | None = None

    pending_obs: Obs | None = None
    pending_action: Action | None = None


@dataclass(order=True, slots=True)
class PendingOutcome:
    """heap element of pending outcomes"""

    sort_tick: int
    seq: int  # sequence number to break ties in the heap
    agent_id: AgentId = field(compare=False)
    machine_id: MachineId = field(compare=False)
    outcome: Outcome = field(compare=False)


class World(VerStateMixin):
    type_name: ClassVar[str] = "world"
    state_kind: ClassVar[str] = "world"
    state_version: ClassVar[int] = 2

    def __init__(
        self,
        machines: dict[MachineId, Machine[Any]],
        runtimes: dict[AgentId, AgentRuntime],
        rng: RngHub,
        event_log: EventSink,
        tilemap: TileMap,
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

        # spatial layer
        self.tilemap = tilemap
        self.occupancy_grid = Occ()
        self.spatial_queues = QueueRegistry(tilemap)
        self.pathfinder = Pathfinder()
        self.replan_policy = ReplanPolicy()

        self.queues: dict[MachineId, deque[AgentId]] = {m: deque() for m in machines}
        self.occupancy: dict[MachineId, int] = {m: 0 for m in machines}

        self._pending: list[PendingOutcome] = []  # heapq
        self._seq: int = 0

        # the order of traversal (both agents and machines) is fixed in
        # advance once to avoid non-reproducibility
        self._agent_order: tuple[AgentId, ...] = tuple(sorted(runtimes))
        self._machine_order: tuple[MachineId, ...] = tuple(sorted(machines))
        self._machine_ids_tuple: tuple[MachineId, ...] = tuple(sorted(machines))

        for agent_id in self._agent_order:
            self.occupancy_grid.place(agent_id, runtimes[agent_id].position)

    def tick(self) -> None:
        """Advance the world by one tick."""
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
            rt.target = None
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

    def _nav_grid(self, agent_id: AgentId, goal: Cell) -> NavGrid:
        return NavGrid(
            tilemap=self.tilemap,
            occ=self.occupancy_grid,
            for_agent=agent_id,
            goal=goal,
        )

    def _advance_movement(self) -> None:
        for agent_id in self._agent_order:
            rt = self.runtimes[agent_id]
            if rt.status is not AgentStatus.MOVING or rt.path is None:
                continue
            self._advance_one_agent(agent_id, rt)

    def _advance_one_agent(self, agent_id: AgentId, rt: AgentRuntime) -> None:
        assert rt.path is not None
        assert rt.target is not None

        rt.ticks_since_replan += 1

        queue = self.spatial_queues[rt.target]
        desired_goal = queue.target_cell(agent_id)
        if desired_goal is None:
            self._abort_movement(agent_id, rt, reason="slot_lost")
            return

        rt.ticks_to_next_cell -= 1
        if rt.ticks_to_next_cell > 0:
            return

        next_cell = rt.path.peek_next()
        if (
            next_cell is not None
            and self.occupancy_grid.reservation_of(agent_id) == next_cell
        ):
            self.occupancy_grid.commit_step(agent_id)
            rt.prev_position = rt.position
            rt.position = rt.path.advance()

        # are we there yet?
        if rt.position == desired_goal:
            self._arrive(agent_id, rt)
            return

        grid = self._nav_grid(agent_id, desired_goal)
        following = rt.path.peek_next()
        blocked = following is None or not grid.is_passable(following)

        if self.replan_policy.should_replan(
            path=rt.path,
            desired_goal=desired_goal,
            next_cell_blocked=blocked,
            ticks_since_replan=rt.ticks_since_replan,
            jitter=int(agent_id),
        ) and not self._replan(agent_id, rt, grid, desired_goal, following, blocked):
            return

        following = rt.path.peek_next()
        if following is None:
            self._arrive(agent_id, rt)
            return

        self.occupancy_grid.reserve(agent_id, following)
        rt.ticks_to_next_cell = self.ticks_per_cell

    def _replan(
        self,
        agent_id: AgentId,
        rt: AgentRuntime,
        grid: NavGrid,
        goal: Cell,
        blocked_cell: Cell | None,
        blocked: bool,
    ) -> bool:
        assert rt.path is not None

        if blocked and blocked_cell is not None:
            detour = sidestep(rt.position, blocked_cell, grid, goal)
            if detour is not None:
                rt.path = Path(cells=[rt.position, detour])
                rt.ticks_since_replan = 0
                return True

        replanned = self.pathfinder.find(rt.position, goal, grid)
        if replanned is None:
            self._abort_movement(agent_id, rt, reason="unreachable")
            return False

        rt.path = replanned
        rt.ticks_since_replan = 0
        return True

    def _arrive(self, agent_id: AgentId, rt: AgentRuntime) -> None:
        assert rt.target is not None
        self.occupancy_grid.release_reservation(agent_id)
        rt.path = None
        rt.ticks_to_next_cell = 0
        rt.status = AgentStatus.QUEUED
        if agent_id not in self.queues[rt.target]:
            self.queues[rt.target].append(agent_id)

    def _abort_movement(
        self, agent_id: AgentId, rt: AgentRuntime, *, reason: str
    ) -> None:
        machine_id = rt.target
        self.occupancy_grid.release_reservation(agent_id)
        self.spatial_queues.release_everywhere(agent_id)
        if machine_id is not None and agent_id in self.queues[machine_id]:
            self.queues[machine_id].remove(agent_id)

        rt.path = None
        rt.queue_slot = None
        rt.target = None
        rt.ticks_to_next_cell = 0
        rt.ticks_since_replan = 0
        rt.status = AgentStatus.IDLE

        self._log(rt, EventType.IDLE, machine_id=machine_id, extra={"reason": reason})

    def _serve_queues(self) -> None:
        for machine_id in self._machine_order:
            machine = self.machines[machine_id]
            queue = self.queues[machine_id]
            spatial = self.spatial_queues[machine_id]

            while queue:
                cap = machine.cap
                if cap is not None and self.occupancy[machine_id] >= cap:
                    break  # machine is full

                agent_id = queue[0]
                if spatial.slot_of(agent_id) != 0:
                    break
                if not machine.can_play(agent_id, self.tick_count):
                    break

                queue.popleft()
                spatial.release(agent_id)
                self.runtimes[agent_id].queue_slot = None
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
            queue_lengths={m: len(self.spatial_queues[m]) for m in self._machine_order},
            last_machine=rt.last_machine,
            last_delta=rt.last_delta,
        )

    def _apply_action(self, agent_id: AgentId, obs: Obs, action: Action) -> None:
        rt = self.runtimes[agent_id]

        if action.kind is ActionKind.IDLE:
            return

        if action.kind is ActionKind.LEAVE_QUEUE:
            self.occupancy_grid.release_reservation(agent_id)
            self.spatial_queues.release_everywhere(agent_id)
            for queue in self.queues.values():
                if agent_id in queue:
                    queue.remove(agent_id)
            rt.path = None
            rt.queue_slot = None
            rt.target = None
            rt.ticks_to_next_cell = 0
            rt.status = AgentStatus.IDLE
            self._log(rt, EventType.LEAVE_QUEUE)
            return

        machine_id = action.machine_id
        assert machine_id is not None
        machine = self.machines.get(machine_id)
        if machine is None:
            self._log(
                rt,
                EventType.IDLE,
                extra={"reason": "unknown_machine", "requested": str(machine_id)},
            )
            return

        self._start_walking(agent_id, rt, obs, action, machine_id)

    def _start_walking(
        self,
        agent_id: AgentId,
        rt: AgentRuntime,
        obs: Obs,
        action: Action,
        machine_id: MachineId,
    ) -> None:
        """Reserve a queue slot and plan a route to it.

        The slot is reserved BEFORE pathfinding: the agent walks to a fixed
        cell of its own rather than chasing a tail that keeps moving away.
        """
        queue = self.spatial_queues[machine_id]
        queue_rng = self.rng.stream(STREAM_QUEUE.format(machine_id=machine_id))

        try:
            slot = queue.reserve_slot(agent_id, queue_rng)
        except QueueFullErr:
            self._log(
                rt,
                EventType.IDLE,
                machine_id=machine_id,
                extra={"reason": "queue_full"},
            )
            return

        goal = queue.cell_of_slot(slot)
        path = self.pathfinder.find(rt.position, goal, self._nav_grid(agent_id, goal))
        if path is None:
            queue.release(agent_id)
            self._log(
                rt,
                EventType.IDLE,
                machine_id=machine_id,
                extra={"reason": "unreachable", "goal": list(goal)},
            )
            return

        rt.pending_obs = obs
        rt.pending_action = action
        rt.target = machine_id
        rt.queue_slot = slot
        rt.path = path
        rt.ticks_since_replan = 0

        next_cell = path.peek_next()
        if next_cell is None:
            rt.status = AgentStatus.QUEUED
            rt.path = None
            if agent_id not in self.queues[machine_id]:
                self.queues[machine_id].append(agent_id)
        else:
            rt.status = AgentStatus.MOVING
            rt.ticks_to_next_cell = self.ticks_per_cell
            self.occupancy_grid.reserve(agent_id, next_cell)

        self._log(
            rt, EventType.CHOOSE, machine_id=machine_id, capital_before=rt.capital
        )

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

    # de/serialization

    def _state_payload(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick_count),
            "config_hash": self.config_hash,
            "ticks_per_cell": self.ticks_per_cell,
            "seq": self._seq,
            "rng": self.rng.dump_state(),
            "tilemap": self.tilemap.dump_state(),
            "occupancy_grid": self.occupancy_grid.dump_state(),
            "spatial_queues": self.spatial_queues.to_payload(),
            "machines": {
                str(mid): self.machines[mid].dump_state() for mid in self._machine_order
            },
            "queues": {
                str(m): [int(a) for a in self.queues[m]] for m in self._machine_order
            },
            "occupancy": {str(m): int(self.occupancy[m]) for m in self._machine_order},
            "pending": [
                {
                    "sort_tick": p.sort_tick,
                    "seq": p.seq,
                    "agent_id": int(p.agent_id),
                    "machine_id": str(p.machine_id),
                    "outcome": outcome_to_dict(p.outcome),
                }
                for p in sorted(self._pending)
            ],
            "agents": {
                str(int(aid)): self._runtime_payload(self.runtimes[aid])
                for aid in self._agent_order
            },
        }

    @staticmethod
    def _runtime_payload(rt: AgentRuntime) -> dict[str, Any]:
        return {
            "type": rt.agent.agent_type,
            "capital": rt.capital,
            "position": list(rt.position),
            "prev_position": list(rt.prev_position),
            "status": rt.status.value,
            "target": str(rt.target) if rt.target is not None else None,
            "path": rt.path.to_payload() if rt.path is not None else None,
            "queue_slot": rt.queue_slot,
            "ticks_to_next_cell": rt.ticks_to_next_cell,
            "ticks_since_replan": rt.ticks_since_replan,
            "last_machine": str(rt.last_machine)
            if rt.last_machine is not None
            else None,
            "last_delta": rt.last_delta,
            "pending_obs": observation_to_dict(rt.pending_obs)
            if rt.pending_obs
            else None,
            "pending_action": (
                action_to_dict(rt.pending_action) if rt.pending_action else None
            ),
            "brain": rt.agent.dump_state(),
        }

    def _apply_state(self, payload: dict[str, Any]) -> None:
        self.tick_count = Tick(int(payload["tick"]))
        self.ticks_per_cell = int(payload["ticks_per_cell"])
        self._seq = int(payload["seq"])
        self.rng.load_state(payload["rng"])
        self.tilemap.load_state(payload["tilemap"])

        saved_machines = payload["machines"]
        if set(saved_machines) != {str(m) for m in self._machine_order}:
            raise StateVerErr(
                "Set of machines in checkpoint does not match config: "
                f"{sorted(saved_machines)} vs {sorted(map(str, self._machine_order))}"
            )
        for mid_str, envelope in saved_machines.items():
            self.machines[MachineId(mid_str)].load_state(envelope)

        self.queues = {
            MachineId(m): deque(AgentId(int(a)) for a in q)
            for m, q in payload["queues"].items()
        }
        self.occupancy = {MachineId(m): int(v) for m, v in payload["occupancy"].items()}

        self.occupancy_grid.load_state(payload["occupancy_grid"])
        self.spatial_queues.from_payload(payload["spatial_queues"])

        self._pending = [
            PendingOutcome(
                sort_tick=int(p["sort_tick"]),
                seq=int(p["seq"]),
                agent_id=AgentId(int(p["agent_id"])),
                machine_id=MachineId(p["machine_id"]),
                outcome=outcome_from_dict(p["outcome"]),
            )
            for p in payload["pending"]
        ]
        heapq.heapify(self._pending)

        saved_agents = payload["agents"]
        if set(saved_agents) != {str(int(a)) for a in self._agent_order}:
            raise StateVerErr(
                "Set of agents in checkpoint does not match config "
                "(count or group list changed)"
            )
        for aid_str, astate in saved_agents.items():
            self._apply_runtime(AgentId(int(aid_str)), astate)

    def _apply_runtime(self, agent_id: AgentId, data: dict[str, Any]) -> None:
        rt = self.runtimes[agent_id]
        if data["type"] != rt.agent.agent_type:
            raise StateVerErr(
                f"Agent {agent_id}: in checkpoint {data['type']!r}, "
                f"in config {rt.agent.agent_type!r}"
            )
        px, py = data["position"]
        qx, qy = data["prev_position"]
        rt.capital = int(data["capital"])
        rt.position = (int(px), int(py))
        rt.prev_position = (int(qx), int(qy))
        rt.status = AgentStatus(data["status"])
        rt.target = MachineId(data["target"]) if data["target"] else None
        rt.path = Path.from_payload(data["path"]) if data["path"] else None
        rt.queue_slot = data["queue_slot"]
        rt.ticks_to_next_cell = int(data["ticks_to_next_cell"])
        rt.ticks_since_replan = int(data["ticks_since_replan"])
        rt.last_machine = (
            MachineId(data["last_machine"]) if data["last_machine"] else None
        )
        rt.last_delta = data["last_delta"]
        rt.pending_obs = (
            observation_from_dict(data["pending_obs"]) if data["pending_obs"] else None
        )
        rt.pending_action = (
            action_from_dict(data["pending_action"]) if data["pending_action"] else None
        )
        rt.agent.load_state(data["brain"])
