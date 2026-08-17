import argparse
import datetime as dt
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from gamblers.core.checkpoints import CheckpointErr, latest_checkpoint, load_checkpoint
from gamblers.core.config.builder import build_world
from gamblers.core.config.schema import RunConfig, load_config
from gamblers.core.events import PqEventLog
from gamblers.core.runner import run_headless
from gamblers.core.utils import todo
from gamblers.core.world import World

CONFIG_COPY_NAME = "config.yaml"


@dataclass(slots=True)
class Session:
    world: World
    config: RunConfig
    run_dir: Path
    event_log: PqEventLog


def _new_run_dir(root: Path) -> Path:
    ts = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def start_new(config_path: Path, runs_root: Path) -> Session:
    config = load_config(config_path)
    run_dir = _new_run_dir(runs_root)
    shutil.copy(config_path, run_dir / CONFIG_COPY_NAME)
    log = PqEventLog(run_dir / "events.parquet", batch_size=config.log_batch_size)
    log.__enter__()
    return Session(build_world(config, log), config, run_dir, log)


def resume(target: Path) -> Session:
    if target.is_dir():
        ckpt = latest_checkpoint(target)
        if ckpt is None:
            raise CheckpointErr(f"There is no checkpoint in {target}")
    else:
        ckpt = target
    run_dir = ckpt.parent
    config = load_config(run_dir / CONFIG_COPY_NAME)
    meta, state = load_checkpoint(ckpt)
    if meta.config_hash != config.config_hash:
        print(
            "[!] config.yaml has changed since the checkpoint was created - "
            "results are not comparable to the original run.",
            file=sys.stderr,
        )
    log = PqEventLog(
        run_dir / f"events_from_{ckpt.stem}.parquet", batch_size=config.log_batch_size
    )
    log.__enter__()
    world = build_world(config, log)
    world.load_state(state)
    print(f"Loaded checkpoint: tick={meta.tick}, code={meta.code_revision}")
    return Session(world, config, run_dir, log)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gamblers", description="todo: add description"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", type=Path, metavar="CONFIG")
    group.add_argument("--load", type=Path, metavar="CKPT_OR_RUNDIR")
    parser.add_argument(
        "--viz", action="store_true", help="Windowed visualization of the world"
    )
    parser.add_argument(
        "--speed", type=float, default=20.0, help="ticks/sec with --viz"
    )
    parser.add_argument(
        "--ticks", type=int, default=None, help="override the length of the run"
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)

    s = start_new(args.new, args.runs_root) if args.new else resume(args.load)
    until = args.ticks if args.ticks is not None else s.config.ticks

    try:
        if args.viz:
            todo()
        else:
            run_headless(
                s.world,
                until_tick=until,
                run_dir=s.run_dir,
                checkpoint_every=s.config.checkpoint_every,
                keep_last=s.config.keep_last_checkpoints,
            )
    finally:
        s.event_log.close()

    print(f"Run completed. Results are in {s.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
