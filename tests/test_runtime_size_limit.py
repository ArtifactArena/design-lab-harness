"""A robot whose root-frame extents exceed the size box during a match loses on that step."""
import mjarena.core.unified_builder  # noqa: F401  (import order: avoids the dspy_core/design_shop cycle)
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjarena.agents.runtime import BotRuntime
from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
from mjarena.envs.sumo import SumoEnv, compose_sumo_model
from mjarena.policy_spec import PolicySpec
from mjarena.runner.episode import _mj_load, run_match

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "mjarena/assets/sumo_ring_env_studio_3d.xml"
BLOCK = ROOT / "mjarena/core/assets/stationary_block_3d.xml"
RULES = ROOT / "configs/rules/rules.yaml"

# Legal at rest (2.0 m long), but a slide joint can push a plate 1.0 m further out.
TELESCOPE = """<mujoco>
  <worldbody>
    <body name="chassis" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom name="hull" type="box" size="1.0 0.5 0.2" material="foam"/>
      <body name="ram" pos="0.9 0 0">
        <joint name="extend" type="slide" axis="1 0 0" range="0 1.0"/>
        <geom name="plate" type="box" size="0.1 0.4 0.15" material="aluminum"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="push" joint="extend" gear="4000"/></actuator>
</mujoco>"""


def _compose(tmp_path, robot_xml=TELESCOPE):
    cfg = ModelValidationConfig(constraints_yaml_path=RULES, physics_mode="3d")
    res = validate_morphology(robot_xml, cfg)
    assert res.passed, res.feedback
    red = tmp_path / "robot.xml"
    red.write_text(res.processed_xml)
    out = tmp_path / "composed.xml"
    compose_sumo_model(env_xml=str(ARENA), robot_red_xml=str(red),
                       robot_blue_xml=str(BLOCK), out_path=str(out),
                       randomize_spawn_3d=True, spawn_seed=0)
    return out, cfg


def _limits(cfg):
    return (cfg.max_robot_x_span, cfg.max_robot_y_span, cfg.max_robot_z_span)


def _env(composed, limits):
    model, data = _mj_load(composed)
    red = BotRuntime(model=model, data=data, policy_callable=lambda obs: {}, prefix="red_")
    blue = BotRuntime(model=model, data=data, policy_callable=lambda obs: {}, prefix="blue_")
    return SumoEnv(model=model, data=data, xml_path=str(composed),
                   red_contender=red, blue_contender=blue, size_limits=limits)


def _push(actuator="push"):
    return PolicySpec.constant([actuator], {actuator: 1.0}).build_callable()


def _run(tmp_path, composed, red_policy, match_time, **kwargs):
    return run_match(
        composed_xml=composed, red_policy_py=red_policy, blue_policy_py=lambda obs: {},
        out_dir=tmp_path, max_steps=None, use_gui=False, save_video=False,
        camera_mode="tracking", quiet=True, seed=0, match_time=match_time,
        inactivity_exempt_prefixes=["blue_"], **kwargs,
    )


def test_extents_are_measured_in_the_root_frame_and_ignore_yaw(tmp_path):
    composed, cfg = _compose(tmp_path)
    env = _env(composed, _limits(cfg))
    env.reset(seed=0)
    rest = env.robot_extents("red_")
    root = env._root_body_ids["red_"]
    adr = env.model.jnt_qposadr[env.model.body_jntadr[root]]
    env.data.qpos[adr + 3:adr + 7] = [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]   # yaw 45°
    mujoco.mj_forward(env.model, env.data)
    assert np.allclose(env.robot_extents("red_"), rest, atol=1e-6)
    # hull spans x in [-1.0, 1.0]; the retracted plate sits inside it
    assert rest[0] == pytest.approx(2.0, abs=0.05)
    assert env.check_size_limit() is None
    env.close()


def test_extending_past_the_box_is_detected_in_the_root_frame(tmp_path):
    composed, cfg = _compose(tmp_path)
    env = _env(composed, _limits(cfg))
    env.reset(seed=0)
    slide = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "red_extend")
    env.data.qpos[env.model.jnt_qposadr[slide]] = 1.0
    mujoco.mj_forward(env.model, env.data)
    assert env.robot_extents("red_")[0] == pytest.approx(3.0, abs=0.05)
    assert env.check_size_limit() == "red"
    env.close()


def test_both_sides_over_the_box_is_a_tie(tmp_path):
    composed, _ = _compose(tmp_path)
    env = _env(composed, (0.1, 0.1, 0.1))   # a box neither robot can fit in
    env.reset(seed=0)
    assert env.check_size_limit() == "both"
    env.close()


def test_extending_past_the_box_ends_the_match_as_a_loss(tmp_path):
    composed, cfg = _compose(tmp_path)
    limits = _limits(cfg)
    rec = _run(tmp_path, composed, lambda obs: {}, match_time=5.0, size_limits=limits)
    assert rec.termination_reason == "timeout"          # idle policy: the ram stays retracted
    rec = _run(tmp_path, composed, _push(), match_time=5.0, size_limits=limits)
    assert rec.termination_reason == "size_violation" and rec.winner == "blue"
    assert rec.num_steps < 500


def test_rule_is_off_when_no_limits_are_given(tmp_path):
    composed, _ = _compose(tmp_path)
    rec = _run(tmp_path, composed, _push(), match_time=2.0)
    assert rec.termination_reason != "size_violation"


def test_two_stage_config_carries_size_limits():
    """The two-stage runners (cross_model.py, intra_model.py) must arm the same
    runtime size box as the tournament and qualification paths."""
    from mjarena.two_stage.match_config import load_tournament_config

    cfg = load_tournament_config(ROOT / "configs/tournaments/arh.yaml")
    assert cfg.size_limits == (2.44, 2.44, 3.05)


# ---------------------------------------------------------------------------
# The same box, at authoring time. `validate_size_constraints` used to measure
# ONLY the world-axis AABB, so a robot whose root <body> carried euler=/quat=
# passed authoring and then lost every seed on step 1 to `size_violation`.
# ---------------------------------------------------------------------------

def _bar(half_length: float, yaw: float = 0.0) -> str:
    """A straight bar `2 × half_length` long, optionally yawed at the root."""
    euler = f' euler="0 0 {yaw}"' if yaw else ""
    return f"""<mujoco>
  <worldbody>
    <body name="chassis" pos="0 0 0.4"{euler}>
      <freejoint name="root"/>
      <geom name="hull" type="box" size="{half_length} 0.2 0.15" material="foam"/>
      <body name="arm" pos="0 0 0.2">
        <joint name="spin" type="hinge" axis="0 0 1"/>
        <geom name="pad" type="box" size="0.1 0.1 0.05" material="aluminum"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="turn" joint="spin" gear="100"/></actuator>
</mujoco>"""


def _validate(xml: str):
    return validate_morphology(xml, ModelValidationConfig(constraints_yaml_path=RULES, physics_mode="3d"))


def _size_step(res):
    return [s for s in res.step_results if s.label == "Size Constraints"][0]


def test_the_world_axis_aabb_alone_would_wave_the_yawed_bar_through():
    """The premise of the bug: yaw shrinks the world AABB of an over-size bar."""
    from mjarena.design_shop.utils import compute_robot_aabb
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody>'
        '<body name="chassis" pos="0 0 0.4" euler="0 0 0.7853981633974483"><freejoint/>'
        '<geom name="hull" type="box" size="1.3 0.2 0.15" mass="60"/>'
        '</body></worldbody></mujoco>'
    )
    mins, maxs = compute_robot_aabb(model)
    assert (maxs - mins)[0] < 2.44 and (maxs - mins)[1] < 2.44, maxs - mins


def test_a_yawed_bar_that_hides_inside_the_world_aabb_fails_authoring():
    """2.6 m bar yawed 45°: world AABB 2.12 × 2.12 (legal), root frame 2.6 (not)."""
    res = _validate(_bar(1.3, yaw=np.pi / 4))
    size = _size_step(res)
    assert not res.passed, res.feedback
    assert not size.passed, size.message
    assert "root" in size.message.lower(), size.message
    assert "2.60" in size.message, size.message
    # It is the root-frame measurement that rejects it, not the world AABB.
    assert "X-span 2.60m exceeds" not in size.message, size.message


def test_a_straight_bar_inside_the_box_passes_both_measurements():
    res = _validate(_bar(1.2))
    assert res.passed, res.feedback


def test_a_yawed_bar_inside_the_box_still_passes():
    """Rotating the root is legal — it is only over-size in the root frame that loses."""
    res = _validate(_bar(1.2, yaw=np.pi / 4))
    assert res.passed, res.feedback


def test_authoring_and_runtime_measure_the_same_spans(tmp_path):
    """One helper, two callers: the authoring number is the number the match uses."""
    from mjarena.design_shop.rules.mj_validators import validate_size_constraints
    from mjarena.design_shop.utils import initial_root_frame_extents

    robot = _bar(1.2, yaw=np.pi / 4)
    cfg = ModelValidationConfig(constraints_yaml_path=RULES, physics_mode="3d")
    res = validate_morphology(robot, cfg)
    assert res.passed, res.feedback
    model = mujoco.MjModel.from_xml_string(res.processed_xml)
    authored = validate_size_constraints(model, cfg).details["root_frame_spans"]
    assert np.allclose(authored, initial_root_frame_extents(model), atol=1e-12)

    composed, cfg = _compose(tmp_path, robot_xml=robot)
    env = _env(composed, _limits(cfg))
    env.reset(seed=3)
    assert np.allclose(env.robot_extents("red_"), authored, atol=1e-6)
    env.close()
