from dataclasses import dataclass

from gamblers.core.world import World


@dataclass
class SpeedCtrl:
    ticks_per_second: float = 10.0
    paused: bool = False

    MIN: float = 0.05
    MAX: float = 100_000.0

    def faster(self, by: int = 2) -> None:
        self.ticks_per_second = min(self.ticks_per_second * by, self.MAX)

    def slower(self, by: int = 2) -> None:
        self.ticks_per_second = max(self.ticks_per_second / by, self.MIN)


class TickDriver:
    MAX_TICKS_PER_FRAME: int = 2000

    def __init__(self, world: World, speed: SpeedCtrl) -> None:
        self.world = world
        self.speed = speed
        self._acc = 0.0
        self.last_frame_ticks: int = 0
        self.throttled: bool = False

    def advance(self, dt: float) -> int:
        """Called by the renderer once per frame. Returns the number of ticks"""
        if self.speed.paused:
            return 0
        self._acc += dt * self.speed.ticks_per_second
        n = int(self._acc)
        self._acc -= n
        self.throttled = n > self.MAX_TICKS_PER_FRAME
        if self.throttled:
            n = self.MAX_TICKS_PER_FRAME
            self._acc = 0.0
        for _ in range(n):
            self.world.tick()
        self.last_frame_ticks = n
        return n

    def step_once(self) -> None:
        self.world.tick()

    @property
    def alpha(self) -> float:
        """
        fraction before the next tick (used for interpolation during rendering)
        """
        return self._acc
