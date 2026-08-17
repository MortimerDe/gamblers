from typing import Any

from gamblers.core.agents.registry import build_agent
from gamblers.core.config.schema import RunConfig
from gamblers.core.events import EventSink
from gamblers.core.machines.base import Machine
from gamblers.core.machines.registry import build_machine
from gamblers.core.rng import RngHub
from gamblers.core.types import AgentId, MachineId
from gamblers.core.world import AgentRuntime, World


def build_world(config: RunConfig, event_log: EventSink) -> World:
    machines: dict[MachineId, Machine[Any]] = {}
    for spec in config.machines:
        machine = build_machine(spec.type, **spec.to_kwargs())
        machines[machine.machine_id] = machine
    rts: dict[AgentId, AgentRuntime] = {}
    next_id = 0
    for group in config.agents:
        for _ in range(group.count):
            agent_id = AgentId(next_id)
            agent = build_agent(group.type, agent_id, **group.to_kwargs())
            rts[agent_id] = AgentRuntime(
                agent=agent,
                capital=group.start_capital,
                position=group.start_position,
                prev_position=group.start_position,
            )
            next_id += 1
    _validate_machine_refs(config, set(machines))
    return World(
        machines=machines,
        runtimes=rts,
        rng=RngHub(config.seed),
        event_log=event_log,
        ticks_per_cell=config.ticks_per_cell,
        config_hash=config.config_hash
    )
def _validate_machine_refs(config: RunConfig, known: set[MachineId]) -> None:
    for group in config.agents:
        kwargs = group.to_kwargs()
        referenced: list[str] = []
        if "machine_id" in kwargs:
            referenced.append(str(kwargs["machine_id"]))
        referenced.extend(str(m) for m in kwargs.get("order", ()))
        for ref in referenced:
            if MachineId(ref) not in known:
                raise ValueError(
                    f"Agent {group.type!r} refers to machine {ref!r}, "
                    f"which is not in the config. Available: {sorted(map(str, known))}"
                )