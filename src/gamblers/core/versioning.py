"""
Versioning of serialized data.
Every component (machine, agent, RNG, world) has its own state independently of the others.
"""

from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class StateEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    version: int
    payload: dict[str, Any]

class StateVerErr(RuntimeError):
    """raised when a state's version is incompatible with the current code"""

@runtime_checkable
class Ver(Protocol):
    state_kind: ClassVar[str]
    state_version: ClassVar[int]

    def dump_state(self) -> dict[str, Any]: ...
    def load_state(self, state: dict[str, Any]) -> None: ...

class VerStateMixin:
    state_kind: ClassVar[str] = ""
    state_version: ClassVar[int] = 1

    # overridden by subclasses
    def _state_payload(self) -> dict[str, Any]:
        """
        Mutable state of the component
        """
        return {}
    def _apply_state(self, payload: dict[str, Any]) -> None:
        if payload:
            raise StateVerErr(f"{type(self).__name__} does not have state, but got {payload!r}")

    @classmethod
    def _migrate_state(cls, payload: dict[str, Any], from_ver: int) -> dict[str, Any]:
        raise StateVerErr(
            f"{cls.state_kind}: no migration from version {from_ver} to {cls.state_version}, checkpoint is incompatible"
        )

    # public API
    def dump_state(self) -> dict[str, Any]:
        cls = type(self)
        if not cls.state_kind:
            raise StateVerErr(f"{cls.__name__} does not have state")
        return StateEnvelope(
            kind=cls.state_kind,
            version=cls.state_version,
            payload=self._state_payload(),
        ).model_dump()
    
    def load_state(self, envelope: dict[str, Any]) -> None:
        cls = type(self)
        env = StateEnvelope.model_validate(envelope)
        if env.kind != cls.state_kind:
            raise StateVerErr(
                f"state of {env.kind!r} is being loaded into {cls.state_kind!r}"
            )
        payload = env.payload
        if env.version != cls.state_version:
            if env.version > cls.state_version:
                raise StateVerErr(
                    f"{cls.state_kind}: checkpoint version {env.version} is newer than code "
                    f"(v{env.version} > v{cls.state_version}). Update the code."
                )
            payload = cls._migrate_state(payload, env.version)
        self._apply_state(payload)