from abc import ABC
from typing import (
    Any,
    ClassVar,
    TypeVar,
    get_args,
    get_origin,
)

import numpy as np
from pydantic import BaseModel, ConfigDict

from gamblers.core.types import AgentId, Cell, MachineId, Outcome, Tick
from gamblers.core.versioning import VerStateMixin


class MachineConfig(BaseModel): # = yaml
    model_config = ConfigDict(extra="forbid", frozen=True)

    machine_id: MachineId
    pos: Cell
    cap: int | None = None # None means infinite capacity

C = TypeVar("C", bound=MachineConfig)

class Machine[C: MachineConfig](VerStateMixin, ABC):
    config_cls: ClassVar[type[MachineConfig]]
    type_name: ClassVar[str]

    def __init__(self, config: C) -> None:
        self.config: C = config

    # __init_subclass__: hook called automatically when a subclass is created
    # __abstractmethods__: set of abstract methods that are still not implemented by the class
    # __dict__: namespace containing attributes defined directly on this class/object
    # not inherited from base classes (getattr looks up the parent by mro (method resolution order))
    # __orig_bases__: original base classes as written before generic type info was erased/simplified at runtime
    #
    # getattr: get an attribute by name dyn
    # isinstance: check whether an object is an instance of a typ of its subclasses
    # issubclass: check whether one class inherits from another
    #
    # get_origin: get the base/orig type of a generic type annotation (e.g. Machine[RouletteConfig] -> Machine)
    # get_args: get the type args of a generic type annotation (e.g. Machine[RouletteConfig] -> (RouletteConfig,))
    #   Machine -> get_origin, RouletteConfig -> get_args
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return

        if "config_cls" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} must declare config_cls")

        declared = cls.__dict__["config_cls"]
        if not (isinstance(declared, type) and issubclass(declared, MachineConfig)):
            raise TypeError(f"{cls.__name__}.config_cls must inherit from MachineConfig")

        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Machine:
                (param,) = get_args(base)
                if isinstance(param, type) and param is not declared:
                    raise TypeError(
                        f"{cls.__name__}: config_cls={declared.__name__}, "
                        f"but declared Machine[{param.__name__}]"
                    )

    @property
    def machine_id(self) -> MachineId:
        return self.config.machine_id
    @property
    def pos(self) -> Cell:
        return self.config.pos
    @property
    def cap(self) -> int | None:
        return self.config.cap

    def can_play(self, agent_id: AgentId, tick: Tick) -> bool:
        """
        whether the machine can accept this agent right now
        """
        return True

    def play(self, agent_id: AgentId, capital: int, rng: np.random.Generator) -> Outcome:
        """
        play one round
        """
        ...  # noqa: PIE790

    def tick(self, tick: Tick) -> None:
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.machine_id!r} at {self.pos}>"