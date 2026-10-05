"""Morphology validation pipeline.

Single entry point: validate_morphology(robot_xml, config) -> MorphologyResult.
Runs all steps, never stops early. Pre-compile failures cause remaining
pre-compile steps to be SKIPPED, but post-compile checks ALL run.
"""
from __future__ import annotations

import re

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from mjarena.design_shop.types import VerifierResult
from mjarena.design_shop.rules.hardware_rules import (
    strip_wrappers,
    sanitize_robot_xml,
    validate_actuator_tags,
    validate_material_attributes,
    apply_material_properties,
    inject_motor_mass,
    validate_moving_bodies_have_geoms,
    mujoco_compile,
    validate_mujoco_model,
)
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig

logger = logging.getLogger(__name__)


@dataclass
class MorphologyResult:
    """Result from the full morphology validation pipeline."""
    passed: bool
    score: float                      # -1 fail, +1 pass
    feedback: str                     # concatenated step messages for LLM
    processed_xml: str = ""           # post-pipeline XML ("" if failed)
    step_results: List[VerifierResult] = field(default_factory=list)


def _tidy_floats(text: str) -> str:
    """238.68480000000002 -> 238.685: three decimals is all a design decision needs."""
    return re.sub(r"(?<![\w.])(-?\d+\.\d{4,})(?![\w.])", lambda m: f"{float(m.group(1)):.3f}".rstrip("0").rstrip("."), text)


def _describe_step_error(exc: Exception, xml: str) -> str:
    """A malformed-XML error names the line and quotes it; anything else is reported as is."""
    import xml.etree.ElementTree as _ET
    err = exc
    while err is not None and not isinstance(err, _ET.ParseError):
        err = err.__cause__ or err.__context__
    if isinstance(err, _ET.ParseError):
        line, col = getattr(err, "position", (None, None))
        lines = xml.splitlines()
        src = lines[line - 1].strip() if line and 0 < line <= len(lines) else ""
        where = f" at line {line}, column {col}" if line else ""
        quoted = f"\n    {src}" if src else ""
        return f"XML is not well-formed{where}: {err.msg if hasattr(err, 'msg') else err}{quoted}"
    return f"Unexpected error: {exc}"


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


def validate_morphology(
    robot_xml: str,
    config: ModelValidationConfig,
    morphology_rules: Optional["MorphologyRules"] = None,
) -> MorphologyResult:
    """Run the full morphology validation pipeline.

    Pre-compile steps run sequentially (each transforms the XML). If any fails,
    remaining pre-compile steps are SKIPPED. Post-compile checks ALL run
    regardless of individual failures. All results concatenated into feedback.

    Args:
        robot_xml: Raw MJCF XML string (possibly with markdown wrappers).
        config: ModelValidationConfig with all constraint values and material palette.

    Returns:
        MorphologyResult with passed, score, feedback string, and processed_xml.
    """
    steps: List[VerifierResult] = []
    xml = robot_xml
    pre_compile_failed = False

    # Step 1: Strip wrappers. An unextractable reply (no block, or several)
    # is feedback, not a crash: skip the rest of the pre-compile chain.
    xml, step = strip_wrappers(xml)
    steps.append(step)
    if not step.passed:
        pre_compile_failed = True

    # Steps 2-6: each depends on previous XML output
    pre_compile_steps = [
        ("Sanitize XML", lambda x: sanitize_robot_xml(x)),
        ("Validate Actuator Tags", lambda x: validate_actuator_tags(x)),
        ("Validate Material Attributes", lambda x: validate_material_attributes(x, config)),
        ("Apply Material Properties", lambda x: apply_material_properties(x, config)),
        ("Inject Motor Mass", lambda x: inject_motor_mass(x, config)),
        ("Validate Moving Bodies Have Geoms", lambda x: validate_moving_bodies_have_geoms(x)),
    ]

    for label, step_fn in pre_compile_steps:
        if pre_compile_failed:
            steps.append(VerifierResult(passed=False, label=label, message="", skipped=True))
            continue
        try:
            xml, step = step_fn(xml)
        except Exception as exc:
            step = VerifierResult(passed=False, label=label, message=_describe_step_error(exc, xml))
        steps.append(step)
        if not step.passed:
            pre_compile_failed = True

    # Step 7: MuJoCo compile
    model = None
    if pre_compile_failed:
        steps.append(VerifierResult(
            passed=False, label="MuJoCo Compile", message="", skipped=True,
        ))
    else:
        model, step = mujoco_compile(xml)
        steps.append(step)
        if not step.passed:
            pre_compile_failed = True

    # Step 7b: Gear clamping (before post-compile checks see the model)
    if model is not None and config.gear_clamp_ratio > 0:
        from mjarena.envs.utils import clamp_actuator_gears
        clamped_msgs = clamp_actuator_gears(model, max_gear_to_inertia_ratio=config.gear_clamp_ratio)
        if clamped_msgs:
            msg = "Clamped actuator gears to safe levels:\n" + "\n".join(f"  {m}" for m in clamped_msgs)
            steps.append(VerifierResult(passed=True, label="Gear Clamping", message=msg))
            logger.info("Gear clamping applied: %d actuators clamped", len(clamped_msgs))

    # Post-compile checks
    if model is None:
        # Add skipped entries for all post-compile checks (validate_mujoco_model's groups)
        post_compile_labels = [
            "Mass Constraints", "Motor Mass", "Size Constraints", "Structural Limits",
            "Actuator Names", "Geom Names", "Control Constraints", "Torque Budget",
            "3D Root Freejoint",
        ]
        for label in post_compile_labels:
            steps.append(VerifierResult(
                passed=False, label=label, message="", skipped=True,
            ))
    else:
        post_results = validate_mujoco_model(model, config, morphology_rules=morphology_rules)
        steps.extend(post_results)

    # Build result
    all_passed = all(s.passed for s in steps if not s.skipped) and not any(s.skipped for s in steps)

    feedback = _format_feedback(steps)
    processed_xml = xml if all_passed else ""
    score = 1.0 if all_passed else -1.0

    return MorphologyResult(
        passed=all_passed,
        score=score,
        feedback=feedback,
        processed_xml=processed_xml,
        step_results=steps,
    )
