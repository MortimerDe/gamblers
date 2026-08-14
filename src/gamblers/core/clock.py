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

    def advance(self, dt: float):
        pass
    
    @property
    def alpha(self) -> float:
        """frac before the next tick (for interpolation in render)"""
        return self._acc