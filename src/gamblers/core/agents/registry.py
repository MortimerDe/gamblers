from collections.abc import Callable
from typing import Any, TypeVar

from gamblers.core.agents.base import Agent
from gamblers.core.types import AgentId

G = TypeVar("G", bound=Agent[Any])
_REGISTRY: dict[str, type[Agent[Any]]] = {}

def register_agent(type_name: str) -> Callable[[type[G]], type[G]]:
    def decorator(cls: type[G]) -> type[G]:
        exist = _REGISTRY.get(type_name)
        if exist is not None and exist.type_name != cls.type_name:
            raise ValueError(
                f"Agent type {type_name!r} already registered "
                f"{exist.type_name}"
            )
        cls.type_name = type_name
        if "state_kind" not in cls.__dict__:
            cls.state_kind = f"agent:{type_name}"
        _REGISTRY[type_name] = cls
        return cls

    return decorator

def build_agent(type_name: str, agent_id: AgentId, **kwargs: Any) -> Agent[Any]:
    cls = lookup_agent(type_name)
    return cls(cls.config_cls.model_validate(kwargs), agent_id=agent_id)

def lookup_agent(type_name: str) -> type[Agent[Any]]:
    try:
        return _REGISTRY[type_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown agent type {type_name!r}. Known: {known}") from None

def agent_schema(type_name: str) -> dict[str, Any]:
    return lookup_agent(type_name).config_cls.model_json_schema()

def known_agent_types() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))