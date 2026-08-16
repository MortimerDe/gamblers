import zlib
from typing import Any, ClassVar

import numpy as np

from gamblers.core.versioning import VerStateMixin

# stream names convention:
STREAM_MACHINE = "machine:{machine_id}"
STREAM_AGENT = "agent:{agent_id}"
STREAM_WORLD = "world"

def _stable_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))

class RngHub(VerStateMixin):
    state_kind: ClassVar[str] = "rng_hub"
    state_version: ClassVar[int] = 1

    def __init__(self, seed: int) -> None:
        self.seed: int = seed
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        gen = self._streams.get(name)
        if gen is None:
            seq = np.random.SeedSequence(entropy=self.seed, spawn_key=(_stable_key(name),))
            bit_gen = np.random.PCG64(seq)
            gen = np.random.Generator(bit_gen)
            self._streams[name] = gen
        return gen

    def machine_stream(self, machine_id: str) -> np.random.Generator:
        return self.stream(STREAM_MACHINE.format(machine_id=machine_id))

    def agent_stream(self, agent_id: str) -> np.random.Generator:
        return self.stream(STREAM_AGENT.format(agent_id=agent_id))

    def _state_payload(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "streams": {name: gen.bit_generator.state for name, gen in self._streams.items()},
        }

    def _apply_state(self, payload: dict[str, Any]) -> None:
        self.seed = int(payload["seed"])
        self._streams.clear()
        for n, s in payload["streams"].items(): # name / state
            seq = np.random.SeedSequence(entropy=self.seed, spawn_key=(_stable_key(n),))
            bit_gen = np.random.PCG64(seq)
            gen = np.random.Generator(bit_gen)
            gen.bit_generator.state = s
            self._streams[n] = gen