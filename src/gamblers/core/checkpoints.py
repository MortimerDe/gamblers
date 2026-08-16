import subprocess
from pathlib import Path

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
    schema_versions: dict[str, int] # kind -> version
    config_hash: str
    tick: int
    seed: int 
    code_revision: str | None = None # git sha

def save_checkpoint(world: World, run_dir: Path, keep_last: int = 3) -> Path:
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
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None

def _prune(run_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    for old in sorted(run_dir.glob(CHECKPOINT_GLOB))[:-keep_last]:
        old.unlink(missing_ok=True)