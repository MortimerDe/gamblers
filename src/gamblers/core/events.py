from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import pyarrow as pa
from pyarrow import parquet as pq

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

class EventType(StrEnum):
    CHOOSE = "choose"
    PLAY_START = "play_start"
    RESULT = "result"
    LEAVE_QUEUE = "leave_queue"
    IDLE = "idle"

@dataclass(frozen=True)
class Event:
    tick: int
    agent_id: int
    agent_type: str  
    agent_label: str
    event: EventType
    machine_id: str | None
    delta: int | None
    capital_before: int | None
    capital_after: int | None
    extra: dict[str, Any] = field(default_factory=dict)

_SCHEMA = pa.schema(
    [
        pa.field("tick", pa.int64()),
        pa.field("agent_id", pa.int32()),
        pa.field("agent_type", pa.string()),
        pa.field("agent_label", pa.string()),
        pa.field("event", pa.string()),
        pa.field("machine_id", pa.string()),
        pa.field("delta", pa.int32()),
        pa.field("capital_before", pa.int64()),
        pa.field("capital_after", pa.int64()),
        pa.field("extra", pa.string()),
    ]
)

class EventSink(Protocol):
    def append(self, event: Event) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...

class PqEventLog:
    """
    buffered writer / ctx manager for events. Writes to a Parquet file.
    """
    def __init__(self, path: Path, batch_size: int = 10_000) -> None:
        self.path = path
        self._batch_size = batch_size
        self._buffer: list[Event] = []
        self._writer: pq.ParquetWriter | None = None
        self._rows_written: int = 0

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self.path, _SCHEMA, compression="zstd")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def append(self, event: Event) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> None: # flush the buffer to disk
        if not self._buffer or self._writer is None:
            return
        rows: list[dict[str, Any]] = [
            {
                "tick": e.tick,
                "agent_id": e.agent_id,
                "agent_type": e.agent_type,
                "agent_label": e.agent_label,
                "event": e.event.value,
                "machine_id": e.machine_id,
                "delta": e.delta,
                "capital_before": e.capital_before,
                "capital_after": e.capital_after,
                "extra": json.dumps(e.extra, ensure_ascii=False) if e.extra else "{}",
            }
            for e in self._buffer
        ]
        self._writer.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA))
        self._rows_written += len(self._buffer)
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

class NullEventLog:
    """
    a no-op event
    """
    def append(self, event: Event) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None

class MemoryEventLog:
    """
    stores events in a list in memory
    """
    def __init__(self) -> None:
        self.events: list[Event] = []

    def append(self, event: Event) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None