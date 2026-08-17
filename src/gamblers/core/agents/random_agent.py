from typing import ClassVar

import numpy as np

from gamblers.core.agents.base import Agent, AgentConfig, idle
from gamblers.core.agents.registry import register_agent
from gamblers.core.types import Action, ActionKind, MachineId, Obs


class RandomAgentConfig(AgentConfig):
    pass

@register_agent("random")
class RandomAgent(Agent[RandomAgentConfig]):
    type_name: ClassVar[str] = "random"
    config_cls: ClassVar[type[AgentConfig]] = RandomAgentConfig
    state_version: ClassVar[int] = 1

    def act(self, obs: Obs, rng: np.random.Generator) -> Action:
        if not obs.available_machines:
            return idle()
        index = int(rng.integers(len(obs.available_machines)))
        return Action(ActionKind.GO_TO_MACHINE, obs.available_machines[index])

class SingleMachineConfig(AgentConfig):
    machine_id: MachineId