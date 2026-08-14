from typing import Any


def run_headless(world: Any, until_tick: int, checkpointer: Any, logger: Any):
    while world.tick_count < until_tick:
        world.tick()