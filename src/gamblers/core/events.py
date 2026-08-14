from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import pyarrow as pa
from pyarrow import parquet as pq

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

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

class EventLog:
    """
    Buffered writer / Context manager for events. Writes to a Parquet file.
    """
    def __init__(self, path: Path, batch_size: int = 10_000) -> None:
        self._path = path
        self._batch_size = batch_size
        self._buffer: list[Event] = []
        self._writer: pq.ParquetWriter | None = None

    def __enter__(self) -> Self: 
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._path, _SCHEMA, compression="zstd")
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()

    def append(self, event: Event) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass
        