"""
Basic types
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NewType

AgentId = NewType("AgentId", int)
MachineId = NewType("MachineId", str)
Tick = NewType("Tick", int)

type Cell = tuple[int, int]

class ActionKind(Enum):
    """What the agent wants to do at this point in time"""
    IDLE = "idle"                   # skip tick
    GO_TO_MACHINE = "go_to_machine" # go to the machine and join the queue
    LEAVE_QUEUE = "leave_queue"     # leave the queue if agent changes their mind

@dataclass(frozen=True)
class Action:
    """Agent decision. machine_id only for `GO_TO_MACHINE`"""
    kind: ActionKind
    machine_id: MachineId | None = None

    def __post_init__(self) -> None:
        if self.kind is ActionKind.GO_TO_MACHINE and self.machine_id is None:
            raise ValueError("GO_TO_MACHINE requires machine_id")

@dataclass(frozen=True)
class Outcome:
    """
    Result of a single machine play:
    - `delta` - capital change, possibly 0
    - `delay_ticks` - ticks until the result is revealed
    - `extra` - arbitrary log data
    """
    delta: int
    delay_ticks: int = 0
    extra: dict[str, Any] = field(default_factory=dict[str, Any])

@dataclass(frozen=True, slots=True)
class Obs:
    """
    Everything ther agent can observe about the world.
    Its only source of info.

    Indentionally excludes and machine internals.
    """
    tick: Tick
    agent_id: AgentId
    capital: int
    position: Cell
    available_machines: tuple[MachineId, ...]
    queue_lengths: dict[MachineId, int]
    last_machine: MachineId | None
    last_delta: int | None # result of last game

# codecs

def observation_to_dict(obs: Obs) -> dict[str, Any]:
    return {
        "tick": int(obs.tick),
        "agent_id": int(obs.agent_id),
        "capital": obs.capital,
        "position": list(obs.position),
        "available_machines": [str(m) for m in obs.available_machines],
        "queue_lengths": {str(m): v for m, v in obs.queue_lengths.items()},
        "last_machine": str(obs.last_machine) if obs.last_machine is not None else None,
        "last_delta": obs.last_delta,
    }

def observation_from_dict(data: dict[str, Any]) -> Obs:
    x, y = data["position"]
    return Obs(
        tick=Tick(int(data["tick"])),
        agent_id=AgentId(int(data["agent_id"])),
        capital=int(data["capital"]),
        position=(int(x), int(y)),
        available_machines=tuple(MachineId(m) for m in data["available_machines"]),
        queue_lengths={MachineId(m): int(v) for m, v in data["queue_lengths"].items()},
        last_machine=MachineId(data["last_machine"]) if data["last_machine"] else None,
        last_delta=data["last_delta"],
    )

def action_to_dict(action: Action) -> dict[str, Any]:
    return {
        "kind": action.kind.value,
        "machine_id": str(action.machine_id) if action.machine_id is not None else None,
    }

def action_from_dict(data: dict[str, Any]) -> Action:
    mid = data["machine_id"]
    return Action(
        kind=ActionKind(data["kind"]),
        machine_id=MachineId(mid) if mid is not None else None,
    )

def outcome_to_dict(outcome: Outcome) -> dict[str, Any]:
    return {
        "delta": outcome.delta,
        "delay_ticks": outcome.delay_ticks,
        "extra": dict(outcome.extra),
    }


def outcome_from_dict(data: dict[str, Any]) -> Outcome:
    return Outcome(
        delta=int(data["delta"]),
        delay_ticks=int(data["delay_ticks"]),
        extra=dict(data["extra"]),
    )