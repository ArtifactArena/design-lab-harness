"""RulesConfig — game rules and validation parameters.

Separates game rules (arena, constraints, physics, validation gates)
from build-process knobs (k, commit_budget, parallelism, caching).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class MorphologyRules:
    """Which morphology validation checks to run."""
    check_mass: bool = True
    check_size: bool = True
    check_structure: bool = True
    check_actuators: bool = True
    check_root_joint: bool = True  # skipped in 2d mode


@dataclass
class MatchConfig:
    """Match execution parameters — shared between qualification and tournament."""
    match_time: Optional[float] = 30.0
    max_steps: Optional[int] = None  # fallback if match_time is None
    inactivity_timeout: Optional[float] = 10.0
    inactivity_min_displacement: float = 0.5
    n_rollouts: int = 3
    save_video: bool = True
    video_width: int = 1280
    video_height: int = 720
    camera: str = "tracking"


# Keep ControllerRules for backward compat during migration
@dataclass
class ControllerRules:
    """Controller verification parameters (legacy)."""
    n_rollouts: int = 3
    enable_stationary: bool = True
    enable_pusher: bool = False
    save_video: bool = True
    video_width: int = 1280
    video_height: int = 720
    match_time: float = 20.0  # qualification match length (s); > inactivity window


@dataclass
class RulesConfig:
    """All game rules and validation parameters.

    Constructed from the `rules` section of tournament YAML.
    """
    arena: Path = dataclasses.field(
        default_factory=lambda: Path("mjarena/assets/sumo_ring_env_cinematic_3d.xml")
    )
    constraints: Path = dataclasses.field(
        default_factory=lambda: Path("configs/rules/rules.yaml")
    )
    physics_mode: str = "3d"
    off_the_ring_criteria: str = "any"
    morphology: MorphologyRules = dataclasses.field(default_factory=MorphologyRules)
    # Keep controller for backward compat
    controller: ControllerRules = dataclasses.field(default_factory=ControllerRules)

    @classmethod
    def from_yaml(cls, config: Dict) -> "RulesConfig":
        """Construct from full tournament config dict.

        Reads from config["rules"] section (new format).
        Falls back to legacy top-level keys if "rules" section is absent.
        """
        rules = config.get("rules", {})
        build = config.get("build", config.get("models_as_engineers", {}))

        # --- Arena & constraints ---
        arena = Path(
            rules.get("arena")
            or config.get("arena")
            or "mjarena/assets/sumo_ring_env_cinematic_3d.xml"
        )

        physics_mode = (
            rules.get("physics_mode")
            or build.get("physics_mode")
            or "3d"
        )

        constraints = Path(
            rules.get("constraints")
            or config.get("constraints")
            or "configs/rules/rules.yaml"
        )

        off_the_ring_criteria = rules.get("off_the_ring_criteria", "any")

        # --- Morphology rules ---
        morph_cfg = rules.get("morphology", {})
        morph_fields = {f.name for f in dataclasses.fields(MorphologyRules)}
        morph_kwargs = {k: v for k, v in morph_cfg.items() if k in morph_fields}
        morphology = MorphologyRules(**morph_kwargs)

        # --- Controller rules (legacy compat) ---
        ctrl_cfg = rules.get("controller", {})
        if not ctrl_cfg:
            legacy = build.get("controller_validation_params", {})
            ctrl_cfg = {
                "n_rollouts": legacy.get("n_rollouts", 3),
                "enable_stationary": legacy.get("enable_stationary", True),
                "enable_pusher": legacy.get("enable_pusher", False),
                "save_video": legacy.get("save_video", True),
                "video_width": legacy.get("video_width", 1280),
                "video_height": legacy.get("video_height", 720),
            }
        ctrl_fields = {f.name for f in dataclasses.fields(ControllerRules)}
        ctrl_kwargs = {k: v for k, v in ctrl_cfg.items() if k in ctrl_fields}
        controller = ControllerRules(**ctrl_kwargs)

        return cls(
            arena=arena,
            constraints=constraints,
            physics_mode=physics_mode,
            off_the_ring_criteria=off_the_ring_criteria,
            morphology=morphology,
            controller=controller,
        )
