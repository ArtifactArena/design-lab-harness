"""Hardware rules — XML/material/compile pass/fail gates.

These are the pre-compile validation steps from the morphology pipeline,
extracted as standalone functions that return (output, VerifierResult).
"""
from __future__ import annotations

import re
from typing import List, Optional

import mujoco

from mjarena.design_shop.types import VerifierResult
from mjarena.design_shop.utils import (
    sanitize_robot_xml_report as _sanitize_robot_xml_report,
    validate_material_attributes as _validate_material_attributes,
    apply_material_properties as _apply_material_properties,
    inject_motor_mass as _inject_motor_mass,
    validate_moving_bodies_have_geoms as _validate_moving_bodies_have_geoms,
)
from mjarena.design_shop.rules.mj_validators import (
    ModelValidationConfig,
    validate_actuator_tags as _validate_actuator_tags,
)
from mjarena.utils.code_blocks import extract_code_block


# <option> would override the arena physics. <asset> is kept for inline meshes; non-mesh
# children are removed in sanitize.
FORBIDDEN_ROOT_TAGS = {"option"}


def strip_wrappers(xml: str) -> tuple[str, VerifierResult]:
    """Strip markdown code blocks, XML declarations, and <mjcf> wrappers."""
    original = xml

    try:
        xml = extract_code_block(xml)
    except ValueError as exc:
        return original, VerifierResult(
            passed=False, label="Strip Wrappers", message=str(exc),
        )

    xml = re.sub(r"<\?xml[^>]*\?>\s*", "", xml)
    xml = re.sub(r"<mjcf[^>]*>\s*", "", xml)
    xml = re.sub(r"\s*</mjcf>", "", xml)
    # a default namespace makes ElementTree emit ns0: prefixes MuJoCo rejects
    xml = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", xml)
    # material=" aluminum" is a typo, not a different material: trim inside the quotes
    xml = re.sub(r'\b(material|name|joint|class)="\s*([^"]*?)\s*"', r'\1="\2"', xml)
    xml = xml.strip()

    if xml and not xml.lstrip().startswith("<mujoco"):
        xml = f"<mujoco>\n{xml}\n</mujoco>"

    changed = xml != original
    msg = "Stripped markdown/XML wrappers." if changed else "No wrappers to strip."
    return xml, VerifierResult(passed=True, label="Strip Wrappers", message=msg, quiet=not changed)


def sanitize_robot_xml(xml: str) -> tuple[str, VerifierResult]:
    """Remove forbidden root tags; report environment-owned XML settings that were overridden."""
    xml, removed_tags, env_notes = _sanitize_robot_xml_report(xml, FORBIDDEN_ROOT_TAGS)
    notes = []
    if removed_tags:
        notes.append(
            f"Removed {', '.join('<' + t + '>' for t in removed_tags)} block(s): "
            "materials come from the palette by name and the arena sets the physics options."
        )
    notes.extend(env_notes)
    changed = bool(notes)
    msg = "\n".join(notes) if changed else "Applied environment-owned XML settings."
    return xml, VerifierResult(passed=True, label="Sanitize XML", message=msg, quiet=not changed)


def validate_actuator_tags(xml: str) -> tuple[str, VerifierResult]:
    """Reject non-<motor> actuator elements (no motor-mass cost = free gear force)."""
    result = _validate_actuator_tags(xml)
    if not result.passed:
        return xml, VerifierResult(
            passed=False,
            label="Validate Actuator Tags",
            message="\n".join(result.errors),
        )
    n = result.details.get("num_actuator_elements", 0)
    return xml, VerifierResult(
        passed=True,
        label="Validate Actuator Tags",
        message=f"All {n} actuator elements are <motor>.",
        quiet=True,
    )


def validate_material_attributes(xml: str, config: ModelValidationConfig) -> tuple[str, VerifierResult]:
    """Validate that every geom has a valid material= attribute."""
    if not config.material_palette:
        return xml, VerifierResult(
            passed=False,
            label="Validate Material Attributes",
            message="Material palette is required but missing. Ensure materials_store.yaml exists.",
        )

    errors, details = _validate_material_attributes(
        xml, config.material_palette,
        default_material=config.default_material,
    )
    if errors:
        return xml, VerifierResult(
            passed=False,
            label="Validate Material Attributes",
            message="\n".join(errors),
        )

    n_geoms = details.get("total_geoms") or len(re.findall(r"<geom\b", xml))
    return xml, VerifierResult(
        passed=True,
        label="Validate Material Attributes",
        message=f"All {n_geoms} geoms have valid material= attributes.",
    )


def apply_material_properties(xml: str, config: ModelValidationConfig) -> tuple[str, VerifierResult]:
    """Compute mass from volume x density, set friction, inject visual assets."""
    xml, errors, info = _apply_material_properties(
        xml, config.material_palette,
        default_material=config.default_material,
    )
    parts = []
    if info:
        parts.extend(f"WARNING: {w}" for w in info)
    if errors:
        parts.extend(errors)
        return xml, VerifierResult(
            passed=False,
            label="Apply Material Properties",
            message="\n".join(parts) if parts else "Failed to apply material properties.",
        )

    msg = "Computed mass and friction from material palette."
    if parts:
        msg += "\n" + "\n".join(parts)
    return xml, VerifierResult(passed=True, label="Apply Material Properties", message=msg, quiet=not parts)


def inject_motor_mass(xml: str, config: ModelValidationConfig) -> tuple[str, VerifierResult]:
    """Add motor mass to geom mass= based on gear values."""
    if config.motor_mass_per_gear <= 0:
        return xml, VerifierResult(
            passed=True,
            label="Inject Motor Mass",
            message="Motor mass injection disabled (motor_mass_per_gear=0).",
            quiet=True,
        )
    try:
        xml = _inject_motor_mass(xml, config.motor_mass_per_gear)
    except RuntimeError as exc:
        return xml, VerifierResult(
            passed=False,
            label="Inject Motor Mass",
            message=str(exc),
        )
    return xml, VerifierResult(
        passed=True,
        label="Inject Motor Mass",
        message=f"Injected motor mass at {config.motor_mass_per_gear} kg per gear unit.",
        quiet=True,
    )


def validate_moving_bodies_have_geoms(xml: str) -> tuple[str, VerifierResult]:
    """Check that each moving assembly contains a rigidly attached geom."""
    errors, details = _validate_moving_bodies_have_geoms(xml)
    if errors:
        return xml, VerifierResult(
            passed=False,
            label="Validate Moving Bodies Have Geoms",
            message="\n".join(errors),
        )
    return xml, VerifierResult(
        passed=True,
        label="Validate Moving Bodies Have Geoms",
        message="All moving bodies have geoms in their rigid assemblies.",
        quiet=True,
    )


def mujoco_compile(xml: str) -> tuple[Optional[mujoco.MjModel], VerifierResult]:
    """Compile XML into MuJoCo model."""
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        return model, VerifierResult(
            passed=True,
            label="MuJoCo Compile",
            message=f"Model compiled: {model.nbody} bodies, {model.ngeom} geoms, {model.nu} actuators.",
        )
    except Exception as exc:
        return None, VerifierResult(
            passed=False,
            label="MuJoCo Compile",
            message=str(exc),
        )


def validate_mujoco_model(
    model: mujoco.MjModel,
    config: ModelValidationConfig,
    morphology_rules: Optional["MorphologyRules"] = None,
) -> List[VerifierResult]:
    """Run all post-compile validators on a compiled MuJoCo model.

    If *morphology_rules* is provided, individual check groups can be
    skipped by setting the corresponding flag to ``False``.  When a flag
    is ``False``, the validators in that group are replaced with
    ``VerifierResult(passed=True, skipped=True)`` entries.

    Returns a list of VerifierResults, one per post-compile check.
    """
    from mjarena.core.rules_config import MorphologyRules
    from mjarena.design_shop.rules.mj_validators import (
        validate_mass_constraints,
        validate_size_constraints,
        validate_control_constraints,
        validate_torque_budget,
        validate_motor_mass as validate_mot_mass,
        validate_structural_constraints,
        validate_root_freejoint,
        validate_actuator_names,
        validate_geom_names,
    )
    from mjarena.agents.types import VerificationResult

    rules = morphology_rules or MorphologyRules()  # all checks enabled by default

    # Map MorphologyRules flags -> (label, validator_fn) groups.
    # Order matches the original POST_COMPILE_CHECKS list.
    CHECK_GROUPS = [
        ("check_mass", [
            ("Mass Constraints", validate_mass_constraints),
            ("Motor Mass", validate_mot_mass),
        ]),
        ("check_size", [
            ("Size Constraints", validate_size_constraints),
        ]),
        ("check_structure", [
            ("Structural Limits", validate_structural_constraints),
        ]),
        ("check_actuators", [
            ("Actuator Names", validate_actuator_names),
            ("Geom Names", validate_geom_names),
            ("Control Constraints", validate_control_constraints),
            ("Torque Budget", validate_torque_budget),
        ]),
        ("check_root_joint", [
            ("3D Root Freejoint", validate_root_freejoint),
        ]),
    ]

    results = []
    for rule_flag, validators in CHECK_GROUPS:
        if not getattr(rules, rule_flag, True):
            # Skip entire group
            for label, _fn in validators:
                results.append(VerifierResult(
                    passed=True, label=label, message="", skipped=True,
                ))
            continue

        for label, validator_fn in validators:
            try:
                result = validator_fn(model, config)
                if result.passed:
                    parts = []
                    if isinstance(result.details, dict):
                        for k, v in result.details.items():
                            if k == "error":
                                continue
                            if isinstance(v, (int, float)):
                                parts.append(f"{k}={v}")
                            elif isinstance(v, str) and len(v) < 150:
                                parts.append(f"{k}={v}")
                            elif isinstance(v, dict):
                                for kk, vv in v.items():
                                    if isinstance(vv, (int, float, str)) and (not isinstance(vv, str) or len(vv) < 100):
                                        parts.append(f"{kk}={vv}")
                    msg = ", ".join(parts) if parts else ""
                    results.append(VerifierResult(passed=True, label=label, message=msg))
                else:
                    results.append(VerifierResult(
                        passed=False,
                        label=label,
                        message="\n".join(result.errors),
                    ))
            except Exception as exc:
                results.append(VerifierResult(passed=False, label=label, message=f"Validator crashed: {exc}"))

    return results
