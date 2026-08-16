import pickle
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from gamblers.core.utils import todo
from gamblers.core.world import World

CHECKPOINT_FORMAT_VERSION: int = 1
CHECKPOINT_GLOB = "chkp_*.pkl"


class ChkpErr(RuntimeError):
    pass


class ChkpMeta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: int
    schema_versions: dict[str, int]  # kind -> version
    config_hash: str
    tick: int
    seed: int
    code_revision: str | None = None  # git sha


def save_checkpoint(world: World, run_dir: Path, keep_last: int = 3) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": ChkpMeta(
            format_version=CHECKPOINT_FORMAT_VERSION,
            schema_versions=collect_versions(world),
            config_hash=world.config_hash,
            tick=int(world.tick_count),
            seed=world.rng.seed,
            code_revision=git_revision(),
        ).model_dump(),
        "state": world.dump_state(),
    }
    path = run_dir / f"ckpt_{int(world.tick_count):012d}.pkl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    _prune(run_dir, keep_last)
    return path


def load_checkpoint(path: Path) -> tuple[ChkpMeta, dict[str, Any]]:
    todo()


def latest_checkpoint(run_dir: Path) -> Path | None:
    fs = sorted(run_dir.glob(CHECKPOINT_GLOB))
    return fs[-1] if fs else None


def collect_versions(world: World) -> dict[str, int]:
    versions: dict[str, int] = {
        type(world).state_kind: type(world).state_version,
        type(world.rng).state_kind: type(world.rng).state_version,
    }
    for m in world.machines.values():
        versions[type(m).state_kind] = type(m).state_version
    for rt in world.runtimes.values():
        versions[type(rt.agent).state_kind] = type(rt.agent).state_version
    return versions


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return out.stdout.strip() or None


def _prune(run_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    for old in sorted(run_dir.glob(CHECKPOINT_GLOB))[:-keep_last]:
        old.unlink(missing_ok=True)
