"""
Unified environment — verify morphology + controller together for one commit.

run_unified_env(): morphology validation, controller validation and the
qualification vs the stationary block, returned as one
feedback string plus the qualification result.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable, Optional

from mjarena.core.build_progress import BuildProgressReporter
from mjarena.design_shop.types import UnifiedEnvResult

logger = logging.getLogger(__name__)


def run_unified_env(
    robot_xml: str,
    controller_code: str,
    *,
    constraints_yaml_path: Path,
    physics_mode: str = "3d",
    output_dir: Optional[Path] = None,
    stationary_match_fn_factory: Optional[Callable] = None,
    n_rollouts: int = 3,
    n_parallel_seeds: int = 1,
    seed_parallel_backend: str = "thread",
    commit_num: int = 0,
    progress_reporter: Optional[BuildProgressReporter] = None,
    progress_key: Optional[str] = None,
    progress_message: str = "",
) -> UnifiedEnvResult:
    """Validate the morphology, then the controller, then run qualification.

    Args:
        robot_xml: Raw MJCF XML string from the Engineer.
        controller_code: Python policy_step source code from the Engineer.
        constraints_yaml_path: Path to the rules YAML.
        physics_mode: "2d" or "3d".
        output_dir: Where per-commit files are written.
        stationary_match_fn_factory: Factory callable(processed_xml_path) -> match_fn.
        n_rollouts: Number of qualification seeds.
        commit_num: Current refinement commit number.

    Returns:
        UnifiedEnvResult with the combined feedback, the score (-1 if the
        morphology failed) and the qualification VerifierResult.
    """
    from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
    from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
    from mjarena.design_shop.pipelines.controller_pipeline import validate_controller
    from mjarena.dspy_core import get_actuator_names_from_xml_string

    if progress_reporter and progress_key:
        progress_reporter.update_subtask(
            progress_key,
            phase="unified",
            stage="validate_morphology",
            wait="",
            message=progress_message or f"commit {commit_num + 1}: morphology validation",
        )

    # ── Morphology ─────────────────────────────────────────────────────
    config = ModelValidationConfig(
        constraints_yaml_path=constraints_yaml_path,
        physics_mode=physics_mode,
    )
    morph_result = validate_morphology(robot_xml, config)
    if not morph_result.passed:
        return UnifiedEnvResult(feedback=(
            f"Morphology validation:\n"
            f"{morph_result.feedback}\n\n"
            f"Controller validation:\n"
            f"  (not tested — morphology failed)\n\n"
            f"Qualification:\n"
            f"  (not tested — morphology failed)"
        ))

    processed_xml = morph_result.processed_xml
    morph_block = f"Morphology validation:\n{morph_result.feedback}"

    try:
        actuator_names = get_actuator_names_from_xml_string(processed_xml)
    except Exception as exc:
        return UnifiedEnvResult(feedback=(
            f"{morph_block}\n\n"
            f"Controller validation:\n"
            f"  FAILED: Could not extract actuators from compiled model: {exc}\n\n"
            f"Qualification:\n"
            f"  (not tested)"
        ))
    if not actuator_names:
        return UnifiedEnvResult(feedback=(
            f"{morph_block}\n\n"
            f"Controller validation:\n"
            f"  FAILED: 0 actuators extracted — bot cannot move\n\n"
            f"Qualification:\n"
            f"  (not tested)"
        ))

    # ── Controller checks + qualification ──────────────────────────────
    _processed_xml_path = None
    stationary_match_fn = None
    if processed_xml and stationary_match_fn_factory:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".xml", delete=False, prefix="unified_robot_"
            ) as tmp:
                tmp.write(processed_xml)
                _processed_xml_path = Path(tmp.name)
            stationary_match_fn = stationary_match_fn_factory(_processed_xml_path)
        except Exception as exc:
            # Never a silent pass: without a match runner `validate_controller`
            # takes the static-only path and returns passed=True / score=0.0 /
            # qualification=None, which the ledger records as a validated commit.
            # A commit that could not be qualified is a failed commit, and the
            # reason belongs in the feedback, not only in the log.
            logger.warning(f"Failed to create match runner: {exc}")
            if _processed_xml_path and _processed_xml_path.exists():
                _processed_xml_path.unlink(missing_ok=True)
            return UnifiedEnvResult(feedback=(
                f"{morph_block}\n\n"
                f"Controller validation:\n"
                f"  (not tested — qualification could not be set up)\n\n"
                f"Qualification:\n"
                f"  FAILED: Qualification could not be set up: {exc}"
            ))

    try:
        ctrl_result = validate_controller(
            controller_code,
            actuator_names,
            qualification_match_fn=stationary_match_fn,
            processed_robot_xml=processed_xml,
            n_rollouts=n_rollouts,
            n_parallel_seeds=n_parallel_seeds,
            seed_parallel_backend=seed_parallel_backend,
            output_dir=output_dir,
            commit_num=commit_num,
            progress_reporter=progress_reporter,
            progress_key=progress_key,
            progress_message=progress_message or f"commit {commit_num + 1}: controller validation",
        )
    finally:
        if _processed_xml_path and _processed_xml_path.exists():
            _processed_xml_path.unlink(missing_ok=True)

    return UnifiedEnvResult(
        feedback=f"{morph_block}\n\nController validation and qualification:\n{ctrl_result.feedback}",
        score=ctrl_result.score if morph_result.passed else -1.0,
        qualification=ctrl_result.qualification,
    )
