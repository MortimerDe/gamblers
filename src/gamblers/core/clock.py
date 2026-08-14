from dataclasses import dataclass
from typing import Any


@dataclass
class SpeedCtrl:
    ticks_per_second: float = 10.0
    paused: bool = False

class TickDriver:
    # todo: replace Any with a proper type for the world
    def __init__(self, world: Any, speed: SpeedCtrl):
        self.world = world
        self.speed = speed
        self._acc = 0.0

    def advance(self, dt: float) -> int:
        """Called by the renderer once per frame. Returns the number of ticks"""
        if self.speed.paused:
            return 0

        self._acc += dt * self.speed.ticks_per_second
        n = int(self._acc)
        self._acc -= n
        for _ in range(n):
            self.world.tick()
        return n

    @property
    def alpha(self) -> float:
        """Fraction before the next tick (used for interpolation during rendering)."""
        return self._acc