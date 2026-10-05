"""Controller validation pipeline.

Single entry point: validate_controller(policy_code, actuator_names, ...) -> ControllerResult.
"""
from __future__ import annotations

import re

import logging
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

from mjarena.core.build_progress import BuildProgressReporter
from mjarena.design_shop.types import ControllerResult, VerifierResult
from mjarena.design_shop.rules.software_rules import (
    compile_policy,
    execute_policy,
    exercise_policy,
    validate_actions,
)
from mjarena.policy_spec import PolicySpec

logger = logging.getLogger(__name__)


def _tidy_floats(text: str) -> str:
    """238.68480000000002 -> 238.685: three decimals is all a design decision needs."""
    return re.sub(r"(?<![\w.])(-?\d+\.\d{4,})(?![\w.])", lambda m: f"{float(m.group(1)):.3f}".rstrip("0").rstrip("."), text)


def _format_feedback(steps: List[VerifierResult]) -> str:
    """Concatenate step results into a single feedback string."""
    lines = []
    for step in steps:
        if step.skipped or (step.passed and (step.quiet or not step.message.strip())):
            continue  # checks that do not apply, or passed with nothing to say
        status = "PASSED" if step.passed else "FAILED"
        lines.append(f"── {step.label} ── {status}")
        if step.message:
            for msg_line in step.message.strip().split("\n"):
                lines.append(f"  {_tidy_floats(msg_line)}")
        lines.append("")

    return "\n".join(lines).rstrip()


def validate_controller(
    policy_code: str,
    actuator_names: Sequence[str],
    obs_schema: Optional[Mapping] = None,
    qualification_match_fn: Optional[Callable] = None,
    n_rollouts: int = 3,
    n_parallel_seeds: int = 1,
    seed_parallel_backend: str = "thread",
    output_dir: Optional[Path] = None,
    commit_num: int = 0,
    progress_reporter: Optional[BuildProgressReporter] = None,
    progress_key: Optional[str] = None,
    progress_message: str = "",
    processed_robot_xml: Optional[str] = None,
) -> ControllerResult:
    """Run controller validation: static checks + qualification round.

    1. Static checks (compile, execute, validate actions) — sequential, fail-fast.
    2. If qualification_match_fn provided, run qualify_round.
    3. Format feedback, return ControllerResult.

    Args:
        policy_code: Python source code containing policy_step function.
        actuator_names: List of actuator names that must be controlled.
        obs_schema: Optional observation schema for extended validation.
        qualification_match_fn: Match runner for stationary block qualification.
        n_rollouts: Number of qualification seeds.
        output_dir: Optional directory for saving match data.
        commit_num: Current refinement commit number.

    Returns:
        ControllerResult with passed, score, feedback, step_results, qualification.
    """
    actuator_names = list(actuator_names)
    steps: List[VerifierResult] = []
    static_failed = False
    effective_parallel = n_rollouts if n_parallel_seeds <= 0 else max(1, n_parallel_seeds)

    if progress_reporter and progress_key:
        progress_reporter.update_subtask(
            progress_key,
            phase="controller",
            stage="static_checks",
            wait="",
            message=progress_message or f"commit {commit_num + 1}: static checks",
        )

    # Step 1: Compile
    policy_step, step = compile_policy(policy_code)
    steps.append(step)
    if not step.passed:
        static_failed = True

    # Named joint/geom access must be checked against this robot, not an empty dummy.
    base_observation = None
    if processed_robot_xml and not static_failed:
        try:
            from mjarena.design_shop.controller_observation import initial_controller_observation
            base_observation = initial_controller_observation(processed_robot_xml)
        except Exception as exc:
            steps.append(VerifierResult(
                passed=False, label="Build Observation", message=str(exc),
                name="verifier:build_observation",
            ))
            static_failed = True

    # Step 2: Execute
    raw_action = None
    if not static_failed:
        raw_action, step = execute_policy(policy_step, obs_schema, base_observation)
        steps.append(step)
        if not step.passed:
            static_failed = True
    else:
        steps.append(VerifierResult(
            passed=False, label="Execute", message="", skipped=True,
            name="verifier:execute",
        ))

    # Step 3: Validate Actions
    if not static_failed and raw_action is not None:
        _, step = validate_actions(raw_action, actuator_names)
        steps.append(step)
        if not step.passed:
            static_failed = True
    else:
        steps.append(VerifierResult(
            passed=False, label="Validate Actions", message="", skipped=True,
            name="verifier:validate_actions",
        ))

    # Step 4: Exercise representative branches
    if not static_failed:
        step = exercise_policy(policy_step, actuator_names, obs_schema, base_observation)
        steps.append(step)
        if not step.passed:
            static_failed = True
    else:
        steps.append(VerifierResult(
            passed=False, label="Exercise Branches", message="", skipped=True,
            name="verifier:exercise",
        ))

    # If static checks failed, return early
    if static_failed:
        feedback = _format_feedback(steps)
        return ControllerResult(
            passed=False,
            score=-1.0,
            feedback=feedback,
            step_results=steps,
        )

    # No qualification match fn -> static-only pass
    if not qualification_match_fn:
        feedback = _format_feedback(steps)
        return ControllerResult(
            passed=True,
            score=0.0,
            feedback=feedback,
            step_results=steps,
        )

    # Build policy callable and run qualification
    from mjarena.eval.runtime import build_policy_callable
    from mjarena.design_shop.tools.match_tools import qualify_round

    policy_callable = build_policy_callable(policy_code, actuator_names)
    policy_spec = PolicySpec.controller_code(policy_code, actuator_names)
    qualification = qualify_round(
        policy_callable=policy_callable,
        policy_spec=policy_spec,
        run_match_fn=qualification_match_fn,
        n_rollouts=n_rollouts,
        n_parallel=effective_parallel,
        parallel_backend=seed_parallel_backend,
        output_dir=output_dir,
        commit_num=commit_num,
        progress_reporter=progress_reporter,
        progress_key=progress_key,
        progress_message=progress_message,
    )

    # qualify_round returns VerifierResult — use it directly as a step
    steps.append(qualification)

    feedback = _format_feedback(steps)

    return ControllerResult(
        passed=qualification.passed,
        score=qualification.score or 0.0,
        feedback=feedback,
        step_results=steps,
        qualification=qualification,
    )
