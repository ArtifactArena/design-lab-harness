"""Design shop rules — pass/fail validation gates."""
from mjarena.design_shop.rules.hardware_rules import (
    sanitize_robot_xml,
    validate_material_attributes,
    apply_material_properties,
    inject_motor_mass,
    validate_moving_bodies_have_geoms,
    mujoco_compile,
)
from mjarena.design_shop.rules.software_rules import (
    compile_policy,
    execute_policy,
    validate_actions,
)
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig

__all__ = [
    "sanitize_robot_xml", "validate_material_attributes", "apply_material_properties",
    "inject_motor_mass", "validate_moving_bodies_have_geoms", "mujoco_compile",
    "compile_policy", "execute_policy", "validate_actions", "ModelValidationConfig",
]
