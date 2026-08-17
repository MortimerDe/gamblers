import time
from collections.abc import Callable
from pathlib import Path

from gamblers.core.checkpoints import save_checkpoint
from gamblers.core.world import World

ProgressFunc = Callable[[int, int, float], None]


def default_progress(tick: int, total: int, ticks_per_sec: float) -> None:
    percent = 100.0 * tick / total if total else 0.0
    print(
        f"\r[{percent:5.1f}%] tick {tick}/{total}  ({ticks_per_sec:,.0f} t/s)", flush=True, end=""
    )


def run_headless(
    world: World,
    until_tick: int,
    run_dir: Path,
    checkpoint_every: int = 0,
    keep_last: int = 3,
    report_every: int = 10_000,
    progress: ProgressFunc | None = default_progress,
) -> None:
    started = time.perf_counter()
    last_report_tick = int(world.tick_count)
    last_report_time = started

    while world.tick_count < until_tick:
        world.tick()
        tick = int(world.tick_count)

        if checkpoint_every > 0 and tick % checkpoint_every == 0:
            world.event_log.flush()
            save_checkpoint(world, run_dir, keep_last)
        if progress is not None and report_every > 0 and tick % report_every == 0:
            now = time.perf_counter()
            ticks_per_sec = (tick - last_report_tick) / (now - last_report_time)
            progress(tick, until_tick, ticks_per_sec)
            last_report_tick = tick
            last_report_time = now
    world.event_log.flush()
    save_checkpoint(world, run_dir, keep_last)
