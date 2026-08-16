from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar, get_args, get_origin

import numpy as np
from pydantic import BaseModel, ConfigDict

from gamblers.core.types import Action, ActionKind, AgentId, Obs, Outcome
from gamblers.core.versioning import VerStateMixin


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: str | None = None

A = TypeVar("A", bound=AgentConfig)

class Agent[A: AgentConfig](VerStateMixin, ABC):
    config_cls: ClassVar[type[AgentConfig]]
    type_name: ClassVar[str]

    def __init__(self, config: A, agent_id: AgentId) -> None:
        self.config: A = config
        self.agent_id: AgentId = agent_id

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if getattr(cls, "__abstractmethods__", None):
            return

        if "type_name" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define type_name")

        type_name = cls.__dict__["type_name"]
        if not isinstance(type_name, str) or not type_name:
            raise TypeError(f"{cls.__name__}.type_name must be a non-empty str")

        if "config_cls" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must define config_cls")

        declared = cls.__dict__["config_cls"]
        if not (isinstance(declared, type) and issubclass(declared, AgentConfig)):
            raise TypeError(f"{cls.__name__}.config_cls must inherit from AgentConfig")

        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Agent:
                (param,) = get_args(base)
                if isinstance(param, type) and param is not declared:
                    raise TypeError(
                        f"{cls.__name__}: config_cls={declared.__name__}, "
                        f"but declared Agent[{param.__name__}]"
                    )

    @property
    def agent_type(self) -> str:
        return self.type_name

    @property
    def label(self) -> str:
        return self.config.label or type(self).type_name

    @abstractmethod
    def act(self, obs: Obs, rng: np.random.Generator) -> Action:
        ...

    def observe(self, obs: Obs, action: Action, outcome: Outcome) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} id={self.agent_id} label={self.label!r}>"

def idle() -> Action:
    """
    Skip the tick
    """
    return Action(ActionKind.IDLE)