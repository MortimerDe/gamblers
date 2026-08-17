import hashlib
from pathlib import Path
from typing import Any

import yaml

from pydantic import BaseModel, ConfigDict, Field, model_validator

class MachineSpec(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    type: str
    machine_id: str
    position: tuple[int, int]

    def to_kwargs(self) -> dict[str, Any]:
        data: dict[str, Any] = self.model_dump()
        data.pop("type")
        return data

class AgentGroupSpec(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    type: str 
    count: int = Field(default=1, ge=1)
    start_capital: int = 0
    start_position: tuple[int, int] = (0, 0)

    def to_kwargs(self) -> dict[str, Any]:
        data: dict[str, Any] = self.model_dump()
        for key in ("type", "count", "start_capital", "start_position"):
            data.pop(key, None)
        return data

class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 42
    ticks: int = Field(default=200_000, ge=1)
    ticks_per_cell: int = Field(default=1, ge=1)
    checkpoint_every: int = Field(default=10_000, ge=0)
    keep_last_checkpoints: int = Field(default=3, ge=0)
    log_batch_size: int = Field(default=10_000, ge=1)
    machines: tuple[MachineSpec, ...] = Field(min_length=1)
    agents: tuple[AgentGroupSpec, ...] = Field(min_length=1)

    config_hash: str = ""

    @model_validator(mode="after")
    def _unique_machine_ids(self) -> RunConfig:
        ids = [m.machine_id for m in self.machines]
        duples = {i for i in ids if ids.count(i) > 1}
        if duples:
            raise ValueError(f"Duplicate machine IDs found: {duples}")
        return self

def load_config(path: Path) -> RunConfig:
    raw_text = path.read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(raw_text) or {}
    hs = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    return RunConfig.model_validate({**raw, "config_hash": hs})