import zlib
from typing import Any

import numpy as np


def _stable_key(name: str) -> int:
    return zlib.crc32(name.encode("utf-8"))

class RngHub:
    def __init__(self, seed: int) -> None:
        self.seed: int = seed
        self._streams: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        gen = self._streams.get(name)
        if gen is None:
            seq = np.random.SeedSequence(entropy=self.seed, spawn_key=(_stable_key(name),))
            gen = np.random.default_rng(seq)
            self._streams[name] = gen
        return gen

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "streams": {name: gen.bit_generator.state for name, gen in self._streams.items()},
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.seed = data["seed"]
        self._streams.clear()
        for name, state in data["streams"].items():
            gen = np.random.default_rng()
            gen.bit_generator.state = state
            self._streams[name] = gen