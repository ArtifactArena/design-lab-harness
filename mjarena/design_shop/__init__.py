"""Design shop — validation rules, the build pipeline, and the engineer signatures.

- rules/: pass/fail validation gates (hardware, software)
- pipelines/: morphology validation, controller validation + qualification, the
  unified per-commit environment
- agents/: the unified Engineer loop and its DSPy signatures
"""
from mjarena.design_shop.types import (
    VerifierResult,
    ControllerResult,
    CommitEntry,
    RoundEntry,  # backwards compat alias for CommitEntry
    BuildMorphologyResult,
    BuildControllerResult,
    GameRecord,
)
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
from mjarena.design_shop.rules.hardware_rules import (
    sanitize_robot_xml,
    validate_material_attributes,
    apply_material_properties,
    inject_motor_mass,
    validate_moving_bodies_have_geoms,
    mujoco_compile,
    validate_mujoco_model,
)
from mjarena.design_shop.rules.software_rules import (
    compile_policy,
    execute_policy,
    validate_actions,
)
from mjarena.design_shop.pipelines.morphology_pipeline import (
    validate_morphology,
    MorphologyResult,
)
from mjarena.design_shop.pipelines.controller_pipeline import validate_controller
from mjarena.design_shop.pipelines.unified_env import run_unified_env
from mjarena.design_shop.agents.signatures.baseline_unified import BaselineUnifiedEngineer
from mjarena.design_shop.refinement_state import RefinementState
from mjarena.design_shop.policy_base import compile_policy_function
from mjarena.design_shop.utils import _thread_log

__all__ = [
    "VerifierResult", "ControllerResult", "RoundEntry", "BuildMorphologyResult",
    "BuildControllerResult", "GameRecord",
    "ModelValidationConfig", "sanitize_robot_xml", "validate_material_attributes",
    "apply_material_properties", "inject_motor_mass", "validate_moving_bodies_have_geoms",
    "mujoco_compile", "validate_mujoco_model", "compile_policy", "execute_policy",
    "validate_actions",
    "validate_morphology", "MorphologyResult", "validate_controller", "run_unified_env",
    "BaselineUnifiedEngineer", "RefinementState", "compile_policy_function", "_thread_log",
]
