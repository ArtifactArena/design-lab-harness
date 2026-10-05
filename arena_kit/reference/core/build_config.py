"""BuildConfig dataclass — single source of truth for build-phase parameters.

Bundles all ~18 build-phase parameters into one dataclass so they don't have to
thread through 3+ function signatures individually.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ControllerValidationParams:
    """Controls controller verification during the build phase.

    Groups all parameters that govern how generated controllers are tested
    before being accepted.
    """
    n_rollouts: int = 3
    n_parallel_seeds: int = 0  # 0 = auto (run all build-time rollout seeds in parallel)
    seed_parallel_backend: str = "process"
    enable_stationary: bool = True
    enable_pusher: bool = True
    save_video: bool = True
    video_width: int = 1280
    video_height: int = 720
    # Qualification match length in seconds (rules.controller.match_time). Must exceed
    # the inactivity window or the inactivity rule can never fire in qualification.
    match_time: float = 20.0
    # Inactivity rule, copied from tournament.match so qualification and tournament
    # play the same game (run_baseline_agent bridges them). None disables the rule.
    inactivity_timeout: Optional[float] = 10.0
    inactivity_min_displacement: float = 0.5

    @classmethod
    def from_controller_rules(cls, rules: "ControllerRules") -> "ControllerValidationParams":
        """Construct from a ControllerRules instance (backward-compat shim)."""
        from mjarena.core.rules_config import ControllerRules
        return cls(
            n_rollouts=rules.n_rollouts,
            enable_stationary=rules.enable_stationary,
            enable_pusher=rules.enable_pusher,
            save_video=rules.save_video,
            video_width=rules.video_width,
            video_height=rules.video_height,
            match_time=rules.match_time,
        )


@dataclass
class ObservationConfig:
    """Controls which spatial observation features are built for controllers."""
    arena_grid: bool = True
    arena_mass_grid: bool = True
    edge_distance_grid: bool = True
    arena_grid_cell_size: float = 0.25
    bounding_radius: bool = True
    ring_radius: bool = True


@dataclass
class BuildConfig:
    """All tuneable knobs for the build phase (morphology + controller generation).

    Defaults live here — nowhere else. Call ``BuildConfig.from_yaml()`` to construct
    from a tournament YAML, or just instantiate directly for programmatic use.
    """

    # Creator/Critic search
    morphology_k: int = 3       # Parallel candidates
    controller_k: int = 5       # Parallel candidates
    # Commits per build: the engineer loop bound AND the number the prompt tells the model.
    commit_budget: int = 50

    # Controller verification (grouped params)
    controller_validation_params: ControllerValidationParams = dataclasses.field(
        default_factory=ControllerValidationParams
    )

    # Obs/action history lookback for controllers
    max_obs_lookback: int = 20
    max_action_lookback: int = 20

    # Prompt / LLM
    use_generator_name_in_prompt: bool = False
    use_dspy_cache: bool = True
    physics_mode: str = "3d"

    # Parallelism
    n_parallel_builds: int = 1
    raise_on_error: bool = False

    # Observation features for controllers
    observation: ObservationConfig = dataclasses.field(
        default_factory=ObservationConfig
    )

    # Unified generation: single LLM call for morphology + controller
    unified_generation: bool = False

    # Zero-shot: single commit, no iteration, no feedback fields. Forces
    zero_shot_mode: bool = False

    # The run's one consolidated prompt. Relative paths are resolved from the repository
    # root. autoresearch_prompt_path is the iterative (ARH) brief, zero_shot_prompt_path
    # the single-submission (SH) one; a run sets exactly one, matching its mode.
    autoresearch_prompt_path: str = ""
    zero_shot_prompt_path: str = ""

    initial_seed_bot_dir: str = ""

    # Seed
    init_seed: int = 42

    # Replay sample rate for LLM match feedback (Hz). Physics runs at 100Hz;
    # replay_hz=2 means every 50th step, reducing a 3000-step match to ~60 samples.
    replay_hz: float = 2.0

    # Values substituted into the prompt's placeholders (see
    # unified_builder.load_consolidated_prompt); filled by run_baseline_agent.config_to_args.
    game_spec_vars: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Debug
    debug_all: bool = False
    dump_prompt: bool = False
    save_intermediate_candidates: bool = False

    def __post_init__(self) -> None:
        if self.autoresearch_prompt_path and (self.zero_shot_mode or not self.unified_generation):
            raise ValueError("autoresearch_prompt_path requires unified_generation and iterative mode")
        if self.zero_shot_prompt_path and not self.zero_shot_mode:
            raise ValueError("zero_shot_prompt_path requires zero-shot mode (zero_shot_mode: true)")
        # One big prompt per harness: there is no modular fallback to fall back to.
        if self.zero_shot_mode and not self.zero_shot_prompt_path:
            raise ValueError(
                "zero_shot_prompt_path is required: a zero-shot run sends one consolidated "
                "prompt (configs/rules/sampling_prompt.md)"
            )
        if not self.zero_shot_mode and not self.autoresearch_prompt_path:
            raise ValueError(
                "autoresearch_prompt_path is required: an iterative run sends one consolidated "
                "prompt (configs/rules/autoresearch_prompt.md)"
            )

    @classmethod
    def from_yaml(cls, build: Dict, output: Optional[Dict] = None) -> "BuildConfig":
        """Construct from YAML ``build`` or ``models_as_engineers`` section.

        Unknown keys are silently ignored so old configs don't break.
        No ``.get()`` defaults needed — the dataclass already has them.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        legacy = {"morphology_refine_n", "controller_refine_n"} & set(build)
        if legacy:
            raise ValueError(
                f"{sorted(legacy)} were replaced by a single 'commit_budget' key "
                "(models_as_engineers.commit_budget); update the config."
            )
        skip_keys = {"observation"}
        kwargs = {k: v for k, v in build.items() if k in known and k not in skip_keys}
        # Handle nested observation config dict
        obs_cfg = build.get("observation")
        if isinstance(obs_cfg, dict):
            obs_fields = {f.name for f in dataclasses.fields(ObservationConfig)}
            obs_kwargs = {k: v for k, v in obs_cfg.items() if k in obs_fields}
            kwargs["observation"] = ObservationConfig(**obs_kwargs)
        if output:
            for k in ("debug_all", "dump_prompt", "save_intermediate_candidates"):
                if k in output:
                    kwargs[k] = output[k]
        return cls(**kwargs)
