from collections.abc import Callable
from typing import Any, TypeVar

from gamblers.core.machines.base import Machine

M = TypeVar("M", bound=Machine[Any])
_REGISTRY: dict[str, type[Machine[Any]]] = {}

def register_machine(type_name: str) -> Callable[[type[M]], type[M]]:
    def decorator(cls: type[M]) -> type[M]:
        exist = _REGISTRY.get(type_name)
        if exist is not None and exist.type_name != cls.type_name:
            raise ValueError(
                f"Machine type {type_name!r} already used "
                f"{exist.type_name}"
            )
        cls.type_name = type_name
        if "state_kind" not in cls.__dict__:
            cls.state_kind = f"machine:{type_name}"
        _REGISTRY[type_name] = cls
        return cls
    return decorator

def build_machine(type_name: str, **kwargs: Any) -> Machine[Any]:
    cls = lookup_machine(type_name)
    return cls(cls.config_cls.model_validate(kwargs))

def build_machine_from_dict(data: dict[str, Any]) -> Machine[Any]: # from yaml
    payload = dict(data)
    try:
        type_name = str(payload.pop("type"))
    except KeyError:
        raise ValueError(f"missing machine `type` in {data!r}") from None
    return build_machine(type_name, **payload)


def lookup_machine(type_name: str) -> type[Machine[Any]]:
    try:
        return _REGISTRY[type_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(f"unknown machine type {type_name!r}, known: {known}")

def machine_schema(type_name: str) -> dict[str, Any]:
    cls = lookup_machine(type_name)
    return cls.config_cls.model_json_schema()

def known_machine_types() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))