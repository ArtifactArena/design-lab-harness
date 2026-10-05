"""Load match settings from a tournament config (with `base: base.yaml` inheritance).

So the two-stage Elo runs use the EXACT same physics/match settings (arena,
match_time, score_function, contact_fidelity, camera, video, inactivity,
rendering flags) that produced the source LOGS being analyzed. Anything not
pinned by the config falls back to the defaults baked into base.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TwoStageMatchConfig:
    """Subset of tournament config relevant to running matches in two-stage Elo."""

    arena_xml: Path
    constraints_path: Path
    physics_mode: str
    score_function: str
    contact_fidelity: str
    n_rollouts: int
    match_time: float
    save_video: bool
    save_all_videos: bool
    highres: bool
    camera: str
    size_limits: tuple[float, float, float]
    inactivity_timeout: Optional[float] = 10.0
    inactivity_min_displacement: float = 0.5
    rendering_flags: Dict[str, bool] = field(default_factory=dict)
    source_path: Optional[Path] = None



def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge: override wins; lists replaced not appended (matches run_baseline_agent)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_yaml_with_inheritance(config_path: Path) -> Dict[str, Any]:
    """Resolve `base: base.yaml` inheritance the same way run_baseline_agent does."""
    raw: Dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw
    base_path = (config_path.parent / base_ref).resolve()
    base_data = _resolve_yaml_with_inheritance(base_path)
    return _deep_merge(base_data, raw)


def find_logs_config(logs_root: Path) -> Optional[Path]:
    """Find the resolved config.yaml at the LOGS root (used by run_baseline_agent at runtime)."""
    candidate = logs_root / "config.yaml"
    return candidate if candidate.exists() else None


def load_tournament_config(config_path: Path) -> TwoStageMatchConfig:
    """Read a tournament config (with optional `base:`) and extract match settings.

    No silent fallbacks — every value must be present in the resolved YAML.
    `base.yaml` already provides every field, so a config that inherits from
    it (or the resolved logs/config.yaml dump) will always succeed.
    """
    raw = _resolve_yaml_with_inheritance(config_path)
    rules = raw.get("rules", {}) or {}
    tournament = raw.get("tournament", {}) or {}
    match = tournament.get("match", {}) or {}
    rendering = raw.get("rendering", {}) or {}

    def req(d: Dict[str, Any], key: str, where: str) -> Any:
        if key not in d:
            raise KeyError(
                f"missing '{key}' in {where} of {config_path} "
                f"(no fallback — pass a config that inherits from base.yaml)"
            )
        return d[key]

    constraints_path = Path(req(rules, "constraints", "rules"))
    physics_mode = str(req(rules, "physics_mode", "rules"))

    # Runtime size limit: the two-stage Elo runs play the same box the
    # validator (and the tournament/qualification paths) enforce — same
    # pattern as run_baseline_agent.config_to_args.
    from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
    validation_config = ModelValidationConfig(
        constraints_yaml_path=constraints_path,
        physics_mode=physics_mode,
    )
    size_limits = (
        validation_config.max_robot_x_span,
        validation_config.max_robot_y_span,
        validation_config.max_robot_z_span,
    )

    cfg = TwoStageMatchConfig(
        arena_xml=Path(req(rules, "arena", "rules")),
        constraints_path=constraints_path,
        physics_mode=physics_mode,
        score_function=str(req(rules, "off_the_ring_criteria", "rules")),
        contact_fidelity=str(req(rules, "contact_fidelity", "rules")),
        n_rollouts=int(req(match, "n_rollouts", "tournament.match")),
        match_time=float(req(match, "match_time", "tournament.match")),
        save_video=bool(req(match, "save_video", "tournament.match")),
        save_all_videos=bool(req(tournament, "save_all_videos", "tournament")),
        highres=bool(req(tournament, "highres", "tournament")),
        camera=str(req(match, "camera", "tournament.match")),
        size_limits=size_limits,
        inactivity_timeout=req(match, "inactivity_timeout", "tournament.match"),
        inactivity_min_displacement=float(req(match, "inactivity_min_displacement", "tournament.match")),
        rendering_flags={k: bool(v) for k, v in rendering.items()},
        source_path=config_path,
    )
    logger.info("Loaded match config from %s", config_path)
    return cfg


def resolve_match_config(
    logs_root: Path,
    explicit_config: Optional[Path] = None,
) -> TwoStageMatchConfig:
    """Pick the right match config, in order:
      1. --tournament-config <path> (explicit override; resolves base: inheritance)
      2. <logs_root>/config.yaml (the resolved dump that run_baseline_agent wrote)

    Errors if neither is found — no silent defaults.
    """
    if explicit_config is not None:
        return load_tournament_config(explicit_config)
    auto = find_logs_config(logs_root)
    if auto is not None:
        return load_tournament_config(auto)
    raise FileNotFoundError(
        f"No tournament config provided and no {logs_root / 'config.yaml'} found. "
        "Pass --tournament-config <path-to-yaml> (e.g. "
        "configs/tournaments/final-frontier-build50-buildonly.yaml)."
    )
