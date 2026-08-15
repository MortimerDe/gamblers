"""
Single source of truth for the simulation state.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gamblers.core.agents.base import Agent
from gamblers.core.types import Action, Cell, MachineId, Obs


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
    prev_pos: Cell # interpolation 
    status: AgentStatus = AgentStatus.IDLE
    target: MachineId | None = None
    path: list[Cell] = field(default_factory=lambda: list[Cell]())
    path_index: int = 0
    ticks_to_next_cell: int = 0
    last_machine: MachineId | None = None
    last_delta: int | None = None

    pending_obs: Obs | None = None
    pending_action: Action | None = None

# @dataclass(order=True, slots=True)
# class PendingOutcome:
