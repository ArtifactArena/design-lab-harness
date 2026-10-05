"""Unified bot builder — single LLM call for morphology + controller.

Contains:
- generate_unified(): Top-level entry point returning GenerationResult.
  Uses L1 Engineer orchestration (single LLM, no Critic).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import dspy

from mjarena.design_shop.utils import _thread_log
from mjarena.agents.types import VerificationResult
from mjarena.core.build_config import BuildConfig
from mjarena.core.build_progress import BuildProgressReporter
from mjarena.core.generation_result import GenerationResult
from mjarena.policy_spec import PolicySpec
from mjarena.dspy_core import (
    create_match_runner,
    get_actuator_names,
    get_actuator_names_from_xml_string,
)
from mjarena.core.qualification_block import block_asset_path, qualification_block_xml
from mjarena.design_shop.agents.L1_engineer_unified import run_baseline_engineer_unified
from mjarena.envs.sumo import compose_sumo_model
from mjarena.utils.file import ensure_dir

logger = logging.getLogger(__name__)

_ASSETS_DIR = Path(__file__).parent / "assets"
_REPO_ROOT = Path(__file__).resolve().parents[2]
def _resolve_optional_bot_dir(bot_dir: str) -> Path:
    path = Path(bot_dir)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


_PROMPT_PLACEHOLDERS = ("mujoco_version", "commit_budget")


def load_consolidated_prompt(cfg: BuildConfig) -> tuple[str, str]:
    """The run's single design document, resolved: ``(filename, text)``.

    One prompt per harness: ``autoresearch_prompt_path`` for an iterative run,
    ``zero_shot_prompt_path`` for a zero-shot one. Every placeholder the document
    uses must be supplied in ``cfg.game_spec_vars``, or the run stops rather than
    shipping a prompt with a hole in it.
    """
    path_str = cfg.zero_shot_prompt_path if cfg.zero_shot_mode else cfg.autoresearch_prompt_path
    if not path_str:
        key = "zero_shot_prompt_path" if cfg.zero_shot_mode else "autoresearch_prompt_path"
        raise ValueError(f"{key} is required: every run sends one consolidated prompt")
    prompt_path = Path(path_str).expanduser()
    if not prompt_path.is_absolute():
        prompt_path = _REPO_ROOT / prompt_path
    text = prompt_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Consolidated prompt is empty: {prompt_path}")
    for key in _PROMPT_PLACEHOLDERS:
        token = "{" + key + "}"
        if token not in text:
            continue
        if key not in cfg.game_spec_vars:
            raise ValueError(f"{prompt_path.name} uses {token} but game_spec_vars has no {key!r}")
        text = text.replace(token, str(cfg.game_spec_vars[key]))
    return prompt_path.name, text


def _load_seed_bot(seed_bot_dir: str) -> dict[str, Any]:
    if not seed_bot_dir:
        return {
            "robot_xml": "",
            "controller_code": "",
            "qualification_passed": False,
            "qualification_score": float("-inf"),
            "bot_name": "",
        }
    resolved = _resolve_optional_bot_dir(seed_bot_dir)
    robot_xml_path = resolved / "robot.xml"
    controller_path = resolved / "controller.py"
    artifact_path = resolved / "bot_artifact.json"
    if not robot_xml_path.is_file() or not controller_path.is_file():
        raise FileNotFoundError(
            f"Seed bot directory must contain robot.xml and controller.py: {resolved}"
        )
    qualification_passed = False
    qualification_score = float("-inf")
    bot_name = resolved.name
    if artifact_path.is_file():
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            qualification_passed = bool(artifact.get("controller_verified", False))
            qualification_score = float(artifact.get("controller_score", float("-inf")))
            bot_name = str(artifact.get("name") or bot_name)
        except Exception:
            pass
    return {
        "robot_xml": robot_xml_path.read_text(encoding="utf-8"),
        "controller_code": controller_path.read_text(encoding="utf-8"),
        "qualification_passed": qualification_passed,
        "qualification_score": qualification_score,
        "bot_name": bot_name,
    }


def generate_unified(
    *,
    generator: str,
    lm: dspy.LM,
    constraints_path: Path,
    output_dir: Path,
    cfg: BuildConfig,
    arena_xml: Optional[Path] = None,
    opponent_xml: Optional[Path] = None,
    eval_dir: Optional[Path] = None,
    match_feedback: str = "",
    season_id: str = "season_00",
    tournament_id: str = "tournament_00",
    verbose: bool = True,
    progress_reporter: Optional[BuildProgressReporter] = None,
) -> GenerationResult:
    """Run unified morphology + controller generation, return GenerationResult.

    Does NOT: save files, debug JSON, prompt dumps, or create artifact.
    All of those happen in bot_builder.build_bot() after this returns.

    Args:
        generator: Generator name (for video overlay).
        lm: Configured DSPy LM instance.
        constraints_path: Path to constraints YAML.
        output_dir: Directory for output files.
        cfg: BuildConfig.
        arena_xml: Path to arena XML.
        opponent_xml: Path to opponent XML.
        eval_dir: Evaluation directory.
        match_feedback: Feedback from previous iterations.
        season_id: Season ID (for video overlay).
        tournament_id: Tournament ID (for video overlay).
        verbose: Print progress.

    Returns:
        GenerationResult.
    """
    # The run's one design document. It is the same text for every build, so
    # run_baseline_agent snapshots it once per run next to config.yaml.
    _, consolidated_prompt = load_consolidated_prompt(cfg)

    cvp = cfg.controller_validation_params
    rollout_dir = Path(eval_dir) if eval_dir else output_dir
    ensure_dir(rollout_dir)

    _has_material_palette = (constraints_path.parent / "materials_store.yaml").exists()
    seed_bot = _load_seed_bot(cfg.initial_seed_bot_dir)
    seed_robot_xml = seed_bot["robot_xml"]
    seed_controller_code = seed_bot["controller_code"]

    # Build run_env_fn closure for unified orchestration
    run_env_fn = None
    _commit_counter = [0]  # mutable counter for commit_num tracking

    if arena_xml is not None:
        from mjarena.design_shop.pipelines.unified_env import run_unified_env
        from mjarena.eval.match_runner import run_seeds, save_matchup_to_disk
        from mjarena.policy_spec import PolicySpec

        # Build stationary match_fn factory for qualification
        stationary_match_fn_factory = None

        if cvp.enable_stationary:
            stat_opponent = block_asset_path(cfg.physics_mode)
            # The block is an arena asset like any other: its mass, friction and contact
            # priority come from the run's material palette, not from numbers written into
            # the XML. Resolve it once per run; each commit writes the exact opponent it
            # fought next to its composed scene.
            stat_block_xml = qualification_block_xml(constraints_path, cfg.physics_mode)
            stat_actuators = get_actuator_names(stat_opponent)
            stat_blue_spec = PolicySpec.zero(stat_actuators)
            # Qualification plays the same runtime size limit as the tournament.
            from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
            cfg_v = ModelValidationConfig(
                constraints_yaml_path=constraints_path,
                physics_mode=cfg.physics_mode,
            )
            stat_size_limits = (
                cfg_v.max_robot_x_span,
                cfg_v.max_robot_y_span,
                cfg_v.max_robot_z_span,
            )

            def _stationary_factory(processed_xml_path):
                commit_num = _commit_counter[0]
                commit_qual_dir = output_dir / "refinement" / f"commit_{commit_num}" / "qualification"
                commit_qual_dir.mkdir(parents=True, exist_ok=True)
                block = commit_qual_dir / "block.xml"
                block.write_text(stat_block_xml, encoding="utf-8")
                composed = commit_qual_dir / "composed.xml"
                compose_sumo_model(
                    env_xml=str(arena_xml),
                    robot_red_xml=str(processed_xml_path),
                    robot_blue_xml=str(block),
                    out_path=str(composed),
                    randomize_spawn_3d=(cfg.physics_mode == "3d"),
                    use_material_palette=_has_material_palette,
                )
                return create_match_runner(
                    composed_xml=composed,
                    blue_policy_spec=stat_blue_spec,
                    out_dir=commit_qual_dir,
                    match_time=cvp.match_time,
                    inactivity_timeout_seconds=cvp.inactivity_timeout,
                    inactivity_min_displacement=cvp.inactivity_min_displacement,
                    save_video_seeds=cvp.n_rollouts if cvp.save_video else 0,
                    video_width=cvp.video_width,
                    video_height=cvp.video_height,
                    quiet=True,
                    max_obs_lookback=cfg.max_obs_lookback,
                    max_action_lookback=cfg.max_action_lookback,
                    gear_clamp_ratio=cfg_v.gear_clamp_ratio,
                    observation_config=cfg.observation,
                    inactivity_exempt_prefixes=["blue_"],
                    size_limits=stat_size_limits,
                )
            stationary_match_fn_factory = _stationary_factory

        def _run_env(
            robot_xml: str,
            controller_code: str,
            *,
            progress_reporter: Optional[BuildProgressReporter] = None,
            progress_key: Optional[str] = None,
            progress_message: str = "",
        ):
            """Closure: run_unified_env with bound infrastructure."""
            commit_num = _commit_counter[0]
            env_result = run_unified_env(
                robot_xml,
                controller_code,
                constraints_yaml_path=constraints_path,
                physics_mode=cfg.physics_mode,
                output_dir=output_dir,
                stationary_match_fn_factory=stationary_match_fn_factory,
                n_rollouts=cvp.n_rollouts,
                n_parallel_seeds=cvp.n_parallel_seeds,
                seed_parallel_backend=cvp.seed_parallel_backend,
                commit_num=commit_num,
                progress_reporter=progress_reporter,
                progress_key=progress_key,
                progress_message=progress_message,
            )
            _commit_counter[0] += 1
            return env_result

        run_env_fn = _run_env

    # Generate unified bot
    start_time = time.time()

    progress_label = (
        "zero-shot submission" if cfg.zero_shot_mode else f"{cfg.commit_budget} commits"
    )
    if progress_reporter:
        progress_reporter.update(
            phase="unified",
            stage="engineer_loop",
            wait="",
            message=progress_label,
        )
    if verbose:
        _thread_log(f"[Unified] Generating bot ({progress_label})...")

    state = run_baseline_engineer_unified(
        lm=lm,
        run_env_fn=run_env_fn,
        refine_n=cfg.commit_budget,
        output_dir=output_dir,
        tournament_data=match_feedback,
        verbose=verbose,
        dump_prompt=cfg.dump_prompt,
        generator=generator,
        progress_reporter=progress_reporter,
        zero_shot_mode=cfg.zero_shot_mode,
        consolidated_prompt=consolidated_prompt,
        seed_robot_xml=seed_robot_xml,
        seed_controller_code=seed_controller_code,
        seed_qualification_passed=bool(seed_bot["qualification_passed"]),
        seed_qualification_score=float(seed_bot["qualification_score"]),
        seed_bot_name=str(seed_bot["bot_name"]),
    )

    generation_time = time.time() - start_time

    # Extract fields from RefinementState
    robot_xml = state.best_content
    controller_code = state.best_controller_code
    best_outputs = state.best_creator_outputs
    overview = best_outputs.get("design_strategy", "")
    design_calculations = best_outputs.get("hardware_plan", "")
    movement = ""
    attack = ""
    defense = ""
    control_calculations = best_outputs.get("combat_plan", "")
    strategy_plan = ""
    bot_design = ""
    reasoning = best_outputs.get("reasoning", "")
    unified_score = float(state.best_scores.get("qualification_score", state.best_score))
    unified_passed = state.best_qualification_passed

    # Get processed XML via validate_morphology
    from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
    from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
    morph_config = ModelValidationConfig(
        constraints_yaml_path=constraints_path,
        physics_mode=cfg.physics_mode,
    )
    morph_result = validate_morphology(robot_xml, morph_config)
    processed_xml = morph_result.processed_xml if morph_result.passed else robot_xml

    # Extract actuator names
    actuator_names = []
    if processed_xml:
        try:
            actuator_names = get_actuator_names_from_xml_string(processed_xml)
        except Exception:
            pass

    if not processed_xml:
        raise RuntimeError(
            f"Unified generation failed: score={unified_score}"
        )

    # Build verification results
    morphology_verification = VerificationResult(
        passed=unified_passed,
        errors=[] if unified_passed else [state.best_feedback],
        details={"score": unified_score, **state.best_scores},
    )
    controller_verification = VerificationResult(
        passed=unified_passed,
        errors=[] if unified_passed else [state.best_feedback],
        details={"score": unified_score, **state.best_scores},
    )

    if progress_reporter:
        progress_reporter.update(
            phase="unified",
            stage="assemble_result",
            wait="",
            message="assembling best unified candidate",
            clear_subtasks=True,
        )
    if verbose:
        _thread_log(
            f"[Unified] Complete! Score: {unified_score:.2f}, "
            f"Passed: {unified_passed}, Time: {generation_time:.1f}s"
        )

    # ── Pack into GenerationResult ───────────────────────────────────────
    return GenerationResult(
        robot_xml=robot_xml,
        processed_xml=processed_xml,
        controller_code=controller_code,
        actuator_names=actuator_names,
        overview=overview,
        design_calculations=design_calculations,
        movement=movement,
        attack=attack,
        defense=defense,
        bot_design=bot_design,
        strategy_plan=strategy_plan,
        control_calculations=control_calculations,
        morphology_score=1.0 if unified_passed else -1.0,
        controller_score=unified_score,
        morphology_verification=morphology_verification,
        controller_verification=controller_verification,
        morphology_generation_time_sec=generation_time,
        controller_generation_time_sec=0.0,
        morph_round_history=[e.__dict__ for e in state.journal],
        morph_lm_history=state.lm_history,
        morph_reasoning=reasoning,
        generation_mode="unified",
    )
