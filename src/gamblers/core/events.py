from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

@dataclass(frozen=True)
class Event:
    """
    One dataset line
    """
    tick: int
    agent_id: int
    agent_type: str
    action: str
    machine: str | None
    delta: int | None
    capital_before: int | None
    capital_after: int | None
    extra: dict[str, Any] = field(default_factory=dict[str, Any])

_SCHEMA = pa.schema(
    [
        pa.field("tick", pa.int64()),
        pa.field("agent_id", pa.int32()),
        pa.field("agent_type", pa.string()),
        pa.field("action", pa.string()),
        pa.field("machine_id", pa.string()),
        pa.field("delta", pa.int32()),
        pa.field("capital_before", pa.int64()),
        pa.field("capital_after", pa.int64()),
        pa.field("extra", pa.string()),
    ]
)
