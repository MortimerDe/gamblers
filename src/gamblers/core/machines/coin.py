from typing import ClassVar

import numpy as np
from pydantic import Field

from gamblers.core.machines.base import Machine, MachineConfig
from gamblers.core.machines.registry import register_machine
from gamblers.core.types import AgentId, Outcome


class CoinFlipperConfig(MachineConfig):
    eps: float = Field(default=0.00, ge=-0.5, le=0.5)
    delay_ticks: int = Field(default=3, ge=0)
    cap: int | None = Field(default=1, ge=1)

@register_machine("coin_flipper")
class CoinFlipper(Machine[CoinFlipperConfig]):
    type_name: ClassVar[str] = "coin_flipper"
    config_cls: ClassVar[type[MachineConfig]] = CoinFlipperConfig
    state_version: ClassVar[int] = 1

    def play(self, agent_id: AgentId, capital: int, rng: np.random.Generator) -> Outcome:
        p_win = 0.5 - self.config.eps
        won = bool(rng.random() < p_win)
        return Outcome(
            delta=1 if won else -1,
            delay_ticks=self.config.delay_ticks,
            extra={"game": "coin_flipper", "won": won}
        )