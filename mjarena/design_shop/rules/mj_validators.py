"""MuJoCo model validation — ModelValidationConfig + post-compile validators. Moved from mjarena.agents.dspy_programs.verifiers.mj_verifier."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import mujoco
from pathlib import Path
from typing import Any, Dict, List, Optional

from mjarena.design_shop.utils import (
    compute_robot_aabb,
    initial_root_frame_extents,
    load_constraints_from_yaml, joint_name,
    validate_single_root_free_joint,
)
from mjarena.agents.types import VerificationResult


# =============================================================================
# Validation Configuration
# =============================================================================

class ModelValidationConfig:
    """
    Configuration for model validation constraints.

    All configuration is loaded from BOT_CONSTRAINTS.yaml. Every required field
    must exist in the YAML — missing keys raise KeyError immediately rather than
    silently falling back to defaults.
    """

    def __init__(
        self,
        constraints_yaml_path: Optional[Path] = None,
        physics_mode: str = "2d",
    ):
        self.physics_mode = physics_mode

        # Load constraints from YAML (required)
        if constraints_yaml_path is not None:
            constraints = load_constraints_from_yaml(constraints_yaml_path)
        else:
            suffix = "3D" if physics_mode == "3d" else "2D"
            default_path = Path(__file__).parents[3] / "configs" / "rules" / f"BOT_CONSTRAINTS_{suffix}.yaml"
            constraints = load_constraints_from_yaml(default_path)

        # --- Material Palette ---
        self.material_palette: Dict = constraints.get("materials", {})
        self.default_material: Optional[str] = constraints.get("default_material")

        # --- Mass ---
        mass_cfg = constraints["mass"]
        self.min_total_mass: float = mass_cfg["min_total_kg"]
        self.max_total_mass: float = mass_cfg["max_total_kg"]

        # --- Size ---
        size_cfg = constraints["size"]
        self.max_robot_x_span: float = size_cfg["max_x_span_m"]
        self.max_robot_y_span: float = size_cfg["max_y_span_m"]
        self.max_robot_z_span: float = size_cfg["max_z_span_m"]

        # --- Control ---
        ctrl_cfg = constraints["control"]
        self.motor_mass_per_gear: float = ctrl_cfg["motor_mass_per_gear"]
        self.max_dof_damping: float = ctrl_cfg["max_dof_damping"]
        self.max_frictionloss: float = ctrl_cfg["max_frictionloss"]
        # Gear clamping: max gear = body_inertia_min × ratio (0 = disabled)
        self.gear_clamp_ratio: float = ctrl_cfg["gear_clamp_ratio"]

        # --- Structure ---
        struct_cfg = constraints["structure"]
        self.max_bodies: int = struct_cfg["max_bodies"]
        self.max_geoms: int = struct_cfg["max_geoms"]
        self.max_actuators: int = struct_cfg["max_actuators"]
        self.min_actuators: int = struct_cfg["min_actuators"]


# =============================================================================
# Core Validation Functions
# =============================================================================

def validate_mass_constraints(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Validate mass-related constraints."""
    errors = []
    details = {}

    # Calculate masses
    body_masses = model.body_mass[1:]  # Skip world body
    total_mass = float(np.sum(body_masses))

    details.update({
        "total_mass": total_mass,
        "num_bodies": len(body_masses)
    })

    # Check constraints
    if total_mass < config.min_total_mass:
        errors.append(f"Total mass {total_mass:.2f} below minimum {config.min_total_mass}")
    if total_mass > config.max_total_mass:
        errors.append(f"Total mass {total_mass:.2f} exceeds maximum {config.max_total_mass}")

    return VerificationResult(passed=not errors, errors=errors, details=details)


def validate_size_constraints(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Validate geometric size constraints."""
    errors = []
    details = {}

    world_axis_over: set = set()
    try:
        mins, maxs = compute_robot_aabb(model)
        spans = maxs - mins

        details.update({
            "x_span": float(spans[0]),
            "y_span": float(spans[1]),
            "z_span": float(spans[2]),
            "bounding_box_min": mins.tolist(),
            "bounding_box_max": maxs.tolist()
        })

        # Check span constraints with actionable error messages
        if spans[0] > config.max_robot_x_span:
            world_axis_over.add("X")
            errors.append(
                f"X-span {spans[0]:.2f}m exceeds {config.max_robot_x_span}m limit. "
                f"Fix: reduce geom size= values or move bodies closer together."
            )
        if spans[1] > config.max_robot_y_span:
            world_axis_over.add("Y")
            errors.append(
                f"Y-span {spans[1]:.2f}m exceeds {config.max_robot_y_span}m limit. "
                f"Fix: reduce geom size= values or move bodies closer together."
            )
        if spans[2] > config.max_robot_z_span:
            world_axis_over.add("Z")
            errors.append(
                f"Z-span {spans[2]:.2f}m exceeds {config.max_robot_z_span}m limit. "
                f"Fix: reduce geom size= values or move bodies closer together."
            )

    except RuntimeError as e:
        errors.append(f"Could not compute robot bounds: {e}")

    # The same box, measured the way the MATCH measures it: in the robot's own
    # root-body frame, every control step. A robot whose root <body> carries
    # euler=/quat= can hide a 2.6 m bar inside a legal world-axis AABB, pass
    # here, and then lose every seed on step 1 to `size_violation`. Both
    # measurements are rules the prompt states, so both are checked.
    # Comparison tolerance is the match's own (`SumoEnv.check_size_limit`:
    # `extents > limits + 1e-6`) so a robot exactly on the line is judged the
    # same way in both places. An axis the world-axis check already reported is
    # not reported a second time.
    limits = (config.max_robot_x_span, config.max_robot_y_span, config.max_robot_z_span)
    try:
        root_spans = initial_root_frame_extents(model)
    except RuntimeError as e:
        errors.append(f"Could not compute root-frame robot bounds: {e}")
    else:
        details["root_frame_spans"] = root_spans.tolist()
        for axis, span, limit in zip(("X", "Y", "Z"), root_spans, limits):
            if span > limit + 1e-6 and axis not in world_axis_over:
                errors.append(
                    f"{axis}-span {span:.2f}m in the robot's own root-body frame exceeds "
                    f"{limit}m limit. The match measures root-frame extents on every "
                    f"control step — a robot extended past the size box "
                    f"({limits[0]} x {limits[1]} x {limits[2]} m) loses the round — so "
                    f"rotating the root body does not make an over-size robot legal. "
                    f"Fix: reduce geom size= values or move bodies closer together."
                )

    return VerificationResult(passed=not errors, errors=errors, details=details)


def validate_control_constraints(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Validate actuator and control constraints."""
    from mjarena.design_shop.utils import validate_internal_actuation, validate_passive_mechanisms
    errors: List[str] = validate_passive_mechanisms(model, config.max_dof_damping, config.max_frictionloss)
    errors.extend(validate_internal_actuation(model))
    details: Dict[str, Any] = {}
    damping = np.asarray(model.dof_damping, dtype=float)
    if damping.size:
        details["max_dof_damping"] = float(damping.max())
        if not np.all(np.isfinite(damping)) or np.any(damping < 0):
            errors.append("Joint damping must be finite and non-negative")
        elif damping.max() > config.max_dof_damping:
            errors.append(f"DOF damping {damping.max():.2f} exceeds maximum {config.max_dof_damping:g}")
        poly = np.asarray(getattr(model, "dof_dampingpoly", np.zeros(0)), dtype=float)
        if poly.size and np.any(poly != 0):
            errors.append("Polynomial (nonlinear) joint damping is not allowed; use scalar damping=")
    max_gear = 0.0
    for i in range(model.nu):
        max_gear = max(max_gear, float(np.linalg.norm(model.actuator_gear[i])))
    details["max_actuator_gear"] = max_gear
    details["num_actuators"] = int(model.nu)
    return VerificationResult(passed=not errors, errors=errors, details=details)



# =============================================================================
# New Validator Functions
# =============================================================================

def validate_torque_budget(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Report total actuator gear and enforce ``min_actuators``."""
    errors = []
    details = {}

    if model.nu == 0:
        if config.min_actuators > 0:
            errors.append(f"Robot has no actuators (min_actuators={config.min_actuators} required)")
        return VerificationResult(passed=not errors, errors=errors, details={"num_actuators": 0})

    # Report the gear the design spends; no arena-imposed cap (the prompt says so).
    actuator_gears = []
    total_gear = 0.0
    for i in range(model.nu):
        gear_norm = np.linalg.norm(model.actuator_gear[i])
        actuator_gears.append(float(gear_norm))
        total_gear += gear_norm

    details.update({
        "total_gear_used": float(total_gear),
        "num_actuators": model.nu,
        "actuator_gears": actuator_gears,
    })

    return VerificationResult(passed=not errors, errors=errors, details=details)


def validate_motor_mass(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Report mass breakdown (geometry vs motor) and check total stays under max_total_kg.

    NOTE: inject_motor_mass() has already added motor mass to geom mass= attributes
    BEFORE MuJoCo compilation. So model.body_mass already includes motor mass.
    We subtract motor mass to show the breakdown (geometry-only vs motor).

    """
    errors = []
    details = {}

    # Compiled total mass (already includes motor mass from inject_motor_mass)
    compiled_total = float(np.sum(model.body_mass[1:]))

    # Motor mass from actuator gear values
    total_abs_gear = 0.0
    for i in range(model.nu):
        total_abs_gear += abs(float(np.linalg.norm(model.actuator_gear[i])))

    motor_mass = total_abs_gear * config.motor_mass_per_gear
    # Subtract motor mass to show geometry-only (for LLM feedback)
    geometry_only = compiled_total - motor_mass

    details.update({
        "geometry_mass_kg": round(geometry_only, 2),
        "total_abs_gear": round(total_abs_gear, 2),
        "motor_mass_per_gear": config.motor_mass_per_gear,
        "motor_mass_kg": round(motor_mass, 2),
        "effective_mass_kg": round(compiled_total, 2),
        "max_total_kg": config.max_total_mass,
    })

    headroom = config.max_total_mass - compiled_total
    if headroom >= 0:
        details["mass_headroom_kg"] = round(headroom, 2)

    if compiled_total > config.max_total_mass:
        over_by = compiled_total - config.max_total_mass
        errors.append(
            f"Effective mass {compiled_total:.1f} kg "
            f"(geometry {geometry_only:.1f} + motor {motor_mass:.1f}) "
            f"exceeds maximum {config.max_total_mass} kg by {over_by:.1f} kg. "
            f"Fix: reduce geometry mass by {over_by:.1f} kg, "
            f"or reduce total |gear| by {over_by / config.motor_mass_per_gear:.0f}, "
            f"or a combination of both."
        )

    return VerificationResult(passed=not errors, errors=errors, details=details)


def validate_structural_constraints(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Validate structural complexity (bodies, geoms, actuators)."""
    errors = []
    details = {}

    # Count robot bodies (exclude world body)
    num_bodies = model.nbody - 1

    # Count robot geoms (exclude world/arena geoms on body 0)
    num_geoms = sum(1 for i in range(model.ngeom) if model.geom_bodyid[i] > 0)

    # Count actuators
    num_actuators = model.nu

    details.update({
        "num_bodies": num_bodies,
        "num_geoms": num_geoms,
        "num_actuators": num_actuators,
        "max_bodies": config.max_bodies,
        "max_geoms": config.max_geoms,
        "max_actuators": config.max_actuators,
        "min_actuators": config.min_actuators
    })

    # Validate counts
    if num_bodies > config.max_bodies:
        errors.append(f"Body count {num_bodies} exceeds maximum {config.max_bodies}")
    if num_geoms > config.max_geoms:
        errors.append(f"Geom count {num_geoms} exceeds maximum {config.max_geoms}")
    if num_actuators > config.max_actuators:
        errors.append(f"Actuator count {num_actuators} exceeds maximum {config.max_actuators}")
    if num_actuators < config.min_actuators:
        errors.append(f"Actuator count {num_actuators} below minimum {config.min_actuators}")

    return VerificationResult(passed=not errors, errors=errors, details=details)


def validate_root_freejoint(model: mujoco.MjModel, config: ModelValidationConfig) -> VerificationResult:
    """Validate that the robot's root body carries a freejoint (3D mode only).

    3D robots need full 6-DOF mobility (translate XYZ + rotate XYZ). In 2D
    mode this check is skipped.
    """
    if config.physics_mode != "3d":
        return VerificationResult(passed=True, errors=[], details={"2d_mode": True, "skipped": True})

    errors = []
    details: Dict[str, object] = {}

    # Body 1 is the first non-world body, i.e. the robot root when the robot
    # XML is compiled standalone.
    root_joints = []
    has_freejoint = False
    for i in range(model.njnt):
        if int(model.jnt_bodyid[i]) != 1:
            continue
        jtype_name = mujoco.mjtJoint(model.jnt_type[i]).name
        root_joints.append(f"{joint_name(model, i)} ({jtype_name})")
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            has_freejoint = True

    details["root_joints"] = root_joints
    details["has_freejoint"] = has_freejoint

    if not has_freejoint:
        found = ", ".join(root_joints) if root_joints else "none"
        errors.append(
            f"[3D MODE] Root body must be mounted on a freejoint (6 DOF: translate XYZ + rotate XYZ). "
            f"Root joints found: {found}. "
            f"Fix: add <freejoint/> (or <joint type=\"free\"/>) as the first child of your root <body>."
        )

    # A robot is one connected assembly: exactly one free joint, on the root
    # body. Any other free joint is a detachable part (a projectile, a second
    # untethered body) and fails validation here too.
    structure_errors = validate_single_root_free_joint(model)
    errors.extend(structure_errors)
    details["structure_errors"] = structure_errors

    return VerificationResult(passed=not errors, errors=errors, details=details)


# =============================================================================
# Actuator Naming Validation
# =============================================================================


def validate_actuator_names(model: mujoco.MjModel,
                            config: Optional[ModelValidationConfig] = None) -> VerificationResult:
    """Verify all actuators have explicit names.

    Controllers reference actuators by name, so unnamed actuators cause
    silent failures downstream.
    """
    errors = []
    details: Dict[str, object] = {"num_actuators": model.nu}
    unnamed = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name:
            unnamed.append(i)
    if unnamed:
        errors.append(
            f"Actuators at indices {unnamed} have no name= attribute. "
            f"Every <motor> must have a unique name so the controller can reference it."
        )
        details["unnamed_indices"] = unnamed
    return VerificationResult(passed=len(errors) == 0, errors=errors, details=details)


def validate_geom_names(model: mujoco.MjModel,
                       config: Optional[ModelValidationConfig] = None) -> VerificationResult:
    """Verify every robot geom has an explicit name.

    The prompt requires a unique name on every geom. Unnamed geoms reach the
    controller's observation records under generated names and carry no design
    intent into the renderer, so the presence check lives next to the actuator one.
    """
    errors = []
    details: Dict[str, object] = {"num_geoms": int(model.ngeom)}
    unnamed = []
    for i in range(model.ngeom):
        if int(model.geom_bodyid[i]) <= 0:
            continue  # world/arena geometry is not the robot's
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i):
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i])) or ""
        unnamed.append((i, body))
    if unnamed:
        # One message per check, as validate_actuator_names does: a robot with 40
        # unnamed geoms should not bury the rest of the report.
        details["unnamed_indices"] = [i for i, _ in unnamed]
        listed = ", ".join(f"Geom #{i} in body '{body}'" for i, body in unnamed)
        errors.append(f"{listed} has no name; every geom needs a unique name="
                      if len(unnamed) == 1 else
                      f"{listed} have no name; every geom needs a unique name=")
    return VerificationResult(passed=not errors, errors=errors, details=details)


def validate_actuator_tags(robot_xml: str) -> VerificationResult:
    """Reject actuator elements that are not <motor> (pre-compile, XML-level).

    Motor mass accounting (inject_motor_mass) only applies to <motor>
    elements, so any other actuator tag (<position>, <velocity>, <general>,
    <intvelocity>, ...) would deliver gear force with no motor-mass cost.
    """
    errors = []
    details: Dict[str, object] = {}

    try:
        root = ET.fromstring(robot_xml)
    except ET.ParseError as exc:
        return VerificationResult(
            passed=False,
            errors=[f"Failed to parse MJCF XML: {exc}"],
            details=details,
        )

    invalid = []
    total = 0
    for actuator_section in root.iter("actuator"):
        for child in actuator_section:
            total += 1
            if child.tag.lower() != "motor":
                invalid.append(f"<{child.tag} name=\"{child.get('name', '(unnamed)')}\">")

    details["num_actuator_elements"] = total
    if invalid:
        details["invalid_actuators"] = invalid
        errors.append(
            f"Only <motor> actuators are allowed; found {', '.join(invalid)}. "
            f"Motor mass (sum(|gear|) x motor_mass_per_gear) is only charged for <motor> "
            f"elements, so other actuator tags would get gear force for free. "
            f"Fix: replace them with <motor> and implement any servo/velocity behavior "
            f"in your controller code."
        )

    return VerificationResult(passed=not errors, errors=errors, details=details)


__all__ = [
    "ModelValidationConfig",
    "validate_mass_constraints",
    "validate_size_constraints",
    "validate_control_constraints",
    "validate_torque_budget",
    "validate_motor_mass",
    "validate_structural_constraints",
    "validate_root_freejoint",
    "validate_actuator_names",
    "validate_geom_names",
    "validate_actuator_tags",
]
