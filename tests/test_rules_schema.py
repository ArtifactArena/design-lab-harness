# tests/test_rules_schema.py
"""rules.yaml carries exactly the rules the prompt states; the validator config reads them."""
import mjarena.core.unified_builder  # noqa: F401  (import order: avoids the dspy_core/design_shop cycle)
from pathlib import Path

import yaml

from mjarena.design_shop.rules.mj_validators import ModelValidationConfig

RULES = Path("configs/rules/rules.yaml")


def test_rules_yaml_matches_prompt_numbers():
    robot = yaml.safe_load(RULES.read_text())["robot"]
    assert robot["mass"] == {"min_total_kg": 25.0, "max_total_kg": 800.0}
    assert robot["control"]["max_dof_damping"] == 600.0
    assert robot["control"]["max_frictionloss"] == 600.0
    assert robot["control"]["motor_mass_per_gear"] == 0.01
    assert robot["structure"] == {"max_bodies": 2000, "max_geoms": 2000, "max_actuators": 2000, "min_actuators": 1}
    assert robot["size"] == {"max_x_span_m": 2.44, "max_y_span_m": 2.44, "max_z_span_m": 3.05}
    assert "bounding_box_3d" not in robot      # one size block only (Task 15, item P)
    assert "material" not in robot             # friction caps deleted (item O)
    assert "expected_physics" not in robot     # gravity backstop deleted (item M)


def test_validation_config_has_no_per_body_or_min_geom_limits():
    cfg = ModelValidationConfig(RULES, physics_mode="3d")
    assert not hasattr(cfg, "max_single_body_mass")
    assert not hasattr(cfg, "min_geom_mass")
    assert cfg.max_dof_damping == 600.0
    assert cfg.max_frictionloss == 600.0
    assert cfg.max_bodies == 2000 and cfg.max_geoms == 2000 and cfg.max_actuators == 2000
    assert cfg.max_robot_z_span == 3.05
