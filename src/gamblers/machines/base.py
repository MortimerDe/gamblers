from typing import Any, Protocol, runtime_checkable

import numpy as np

from gamblers.core.types import AgentId, Cell, MachineId, Outcome, Tick


@runtime_checkable
class Machine(Protocol):
    machine_id: MachineId
    position: Cell
    capacity: int | None # None means infinite capacity

    def can_play(self, agent_id: AgentId, tick: Tick) -> bool:
        """
        Whether the machine can accept this agent right now.

        Used by machines with their own cycle. For example, roulette only accepts
        bets during the betting phase and returns False otherwise.
        """
        ...

    def play(self, agent_id: AgentId, capital: int, rng: np.random.Generator) -> Outcome:
        """
        Play single round of the machine
        """
        ...

    def tick(self, tick: Tick) -> None:
        """
        Internal machine state update outside the game (timers, phases etc.)
        """
        ...

    def state_dict(self) -> dict[str, Any]:
        ...

    def load_state_dict(self, data: dict[str, Any]) -> None:
        ...