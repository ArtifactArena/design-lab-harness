"""Static configuration for the minimal agentic bot-building harness.

Paths resolve against the self-contained root bundled with this harness (the
folder holding mjarena/ + configs/), so it imports and runs the *vendored*
mjarena package — no outer arena repo required. Falls back to a live repo for dev.
"""
from __future__ import annotations

import os
from pathlib import Path


def _is_repo(d: Path) -> bool:
    return (d / "mjarena").is_dir() and (d / "configs" / "rules" / "rules.yaml").is_file()


def _find_repo_root(start: Path) -> Path:
    """Locate the root that has mjarena/ + configs/rules/rules.yaml.

    This module is also copied into each run workspace (runs/<ts>/...), so the
    root is discovered dynamically:
      1. nearest ancestor that is self-contained (the bundled mjarena + configs);
      2. $ARENA_REPO_ROOT override (dev: point at the live arena repo);
      3. fall back to an importable `mjarena` package's location.
    """
    for d in [start, *start.parents]:
        if _is_repo(d):
            return d
    env = os.environ.get("ARENA_REPO_ROOT")
    if env and _is_repo(Path(env)):
        return Path(env)
    try:
        import mjarena  # noqa: F401
        root = Path(mjarena.__file__).resolve().parent.parent
        if _is_repo(root):
            return root
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(
        f"Could not locate a self-contained root above {start} (needs mjarena/ + "
        "configs/rules/rules.yaml). Set ARENA_REPO_ROOT to a valid repo for dev."
    )


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)

# Ensure the chosen root's mjarena/ wins over any other copy on sys.path.
import sys as _sys  # noqa: E402
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))

# Canonical match settings (arena xml, constraints, physics_mode, score_function)
# are resolved from this tournament config via resolve_match_config().
TOURNAMENT_CONFIG = REPO_ROOT / "configs" / "tournaments" / "base.yaml"

# Model driving the agent loop (forwarded verbatim to configure_lm()).
# Default: the bundled cheap Gemini Flash. Override with $MH_MODEL_CONFIG, which may
# be an absolute path or a path relative to REPO_ROOT (e.g. configs/models/<name>.yaml).
_default_model = REPO_ROOT / "configs" / "models" / "google" / "gemini-2-5-flash.yaml"
_env_model = os.environ.get("MH_MODEL_CONFIG")
if _env_model:
    _p = Path(_env_model)
    MODEL_CONFIG = _p if _p.is_absolute() else (REPO_ROOT / _p)
else:
    MODEL_CONFIG = _default_model

# Default opponent for run_match (relative to the kit/workspace). The agent may
# override with any --opponent_xml / --opponent_controller it wants.
DEFAULT_OPPONENT_XML = "assets/block.xml"
DEFAULT_OPPONENT_CONTROLLER = "assets/block.py"

# Stationary qualifier dummy used by the verify pipeline.
STATIONARY_BLOCK = REPO_ROOT / "mjarena" / "core" / "assets" / "stationary_block_3d.xml"

# Agent loop limits.
MAX_STEPS = 10          # max agent tool-iterations before forced stop
# No fixed tool-result cap. The loop is state-refreshed (a result is shown ONCE then
# drops out), so a result is truncated only to fit the DRIVING MODEL's own context
# window — computed at runtime from litellm per-model info — so each model sees as
# much as it fairly can. MAX_READ only bounds how much read_file pulls off disk into
# a single result (it pages with offset/limit); override with MH_MAX_READ.
def _env_int(name: str, default: int) -> int:
    """Env override as int; empty/unset -> default; non-int -> named error (no silent fallback)."""
    v = os.environ.get(name)
    if not v or not v.strip():
        return default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name} must be an integer (got {v!r})")


MAX_READ = _env_int("MH_MAX_READ", 400000)   # max chars read_file pulls per call
VERIFY_ROLLOUTS = 3     # qualification seeds inside does_bot_qualify (same as the build harness)
MATCH_SEEDS = 3         # default seeds inside run_match


def _qualification_match_time() -> float:
    """rules.controller.match_time from the tournament config (20 s): the qualification
    length, long enough for the 10 s inactivity rule to fire. Missing key -> error."""
    import yaml
    raw = yaml.safe_load(TOURNAMENT_CONFIG.read_text()) or {}
    try:
        return float(raw["rules"]["controller"]["match_time"])
    except (KeyError, TypeError) as exc:
        raise KeyError(f"{TOURNAMENT_CONFIG}: rules.controller.match_time is required") from exc


QUALIFY_MATCH_TIME = _qualification_match_time()  # seconds per qualification seed
JOURNAL_WINDOW = 30     # max journal entries shown in the state block
# Wall-clock guard so a slow/runaway model controller (e.g. while True in
# policy_step) can't hang the whole run. 0 disables. Override with MH_MATCH_TIMEOUT.
MATCH_TIMEOUT = _env_int("MH_MATCH_TIMEOUT", 3600)  # seconds per run_match/qualify (up to 3 parallel seeds, 300 s per sparring match)
