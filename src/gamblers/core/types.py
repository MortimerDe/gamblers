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
    """What the agent wnats to do at this point in time"""
    IDLE = "idle"                   # skip tick
    GO_TO_MACHINE = "go_to_machine" # go to the machine and join the queue
    LEAVE_QUEUE = "leave_queue"     # leave the queue if agent changes thier mind
    # ...

@dataclass(frozen=True)
class Action:
    """Agent decision. machind_id only for `GO_TO_MACHINE`"""
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

@dataclass(frozen=True)
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
    queue_lengts: dict[MachineId, int]
    last_machine: MachineId | None
    last_delta: int | None # result of last game