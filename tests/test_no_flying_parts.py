"""A robot may have exactly one free joint, on its root body; detachable parts are rejected."""
import mjarena.core.unified_builder  # noqa: F401
from pathlib import Path

import pytest
from lxml import etree as ET

from mjarena.agents.runtime import BotRuntime
from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
from mjarena.envs.sumo import SumoEnv, compose_sumo_model
from mjarena.envs.utils import _mj_load

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "configs/rules/rules.yaml"
ARENA = ROOT / "mjarena/assets/sumo_ring_env_studio_3d.xml"
BLOCK = ROOT / "mjarena/core/assets/stationary_block_3d.xml"

CONNECTED = """<mujoco><worldbody>
  <body name="chassis" pos="0 0 0.3"><freejoint name="root"/>
    <geom name="hull" type="box" size="0.5 0.4 0.2" material="foam"/>
    <body name="arm" pos="0.5 0 0"><joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="arm_g" type="box" size="0.2 0.1 0.1" material="aluminum"/></body>
  </body></worldbody>
  <actuator><motor name="m" joint="hinge" gear="200"/></actuator></mujoco>"""

# Same robot plus a "shell" body that floats beside it, tethered by nothing: a
# detachable part / projectile. MuJoCo only permits a free joint on a top-level
# body, so the extra body must be a SIBLING of chassis (not nested inside it) --
# nesting it would fail to *compile* at all ("free joint can only be used on top
# level"), which would test MuJoCo's own restriction rather than this new rule.
# Placed within the size box (2.44 x 2.44 x 3.05 m) so Size Constraints stays
# green and this rule is the one that fails the model.
FLYING = CONNECTED.replace(
    '</worldbody>',
    '<body name="shell" pos="0 -0.9 0.3"><freejoint name="shell_free"/>'
    '<geom name="shell_g" type="sphere" size="0.1" material="steel"/></body>'
    '</worldbody>')


def _cfg():
    return ModelValidationConfig(constraints_yaml_path=RULES, physics_mode="3d")


def test_connected_robot_passes():
    assert validate_morphology(CONNECTED, _cfg()).passed


def test_second_free_joint_fails_authoring_validation():
    res = validate_morphology(FLYING, _cfg())
    assert not res.passed
    assert "free joint" in res.feedback.lower() and "shell" in res.feedback


def _compose_connected(tmp_path):
    # A valid, connected raw robot (density instead of materials to bypass
    # authoring validation on purpose -- compose_sumo_model itself already
    # refuses a raw robot XML with a disconnected top-level body, via
    # _validate_robot_composition, so that path can never reach SumoEnv).
    raw = (CONNECTED.replace('material="foam"', 'density="200"')
                     .replace('material="aluminum"', 'density="2700"'))
    red_xml = tmp_path / "connected_robot.xml"
    red_xml.write_text(raw)
    out = tmp_path / "composed.xml"
    compose_sumo_model(str(ARENA), str(red_xml), str(BLOCK), str(out),
                        randomize_spawn_3d=True, spawn_seed=0)
    return out


def _env_from(out):
    model, data = _mj_load(str(out))
    red = BotRuntime(model=model, data=data, policy_callable=lambda obs: {}, prefix="red_")
    blue = BotRuntime(model=model, data=data, policy_callable=lambda obs: {}, prefix="blue_")
    return model, data, red, blue


def _make_shell(free_name="red_shell_free"):
    shell = ET.Element("body", {"name": "red_shell", "pos": "0 0 5"})
    ET.SubElement(shell, "freejoint", {"name": free_name})
    ET.SubElement(shell, "geom", {"name": "red_shell_g", "type": "sphere", "size": "0.1", "density": "7800"})
    return shell


def test_runtime_rejects_a_preprocessed_flying_part(tmp_path):
    # Hand-edit the ALREADY-COMPOSED XML -- as a preprocessing step would -- to
    # smuggle in a second, untethered free-jointed body sitting directly under
    # <worldbody>, alongside (and after) the real robot root. Only the runtime
    # check in SumoEnv.__init__ can still catch this.
    out = _compose_connected(tmp_path)
    tree = ET.parse(str(out))
    worldbody = tree.getroot().find("worldbody")
    worldbody.append(_make_shell())
    tree.write(str(out))

    model, data, red, blue = _env_from(out)
    with pytest.raises(ValueError, match="Invalid robot structure"):
        SumoEnv(model, data, str(out), red, blue, max_steps=1)


def test_runtime_rejects_a_prepended_flying_part_without_misattributing_the_root(tmp_path):
    # Same exploit, but the untethered body is inserted BEFORE the real root in
    # document order (so it gets the LOWER body id). A root-selection rule that
    # picks "whichever prefix-matching body has the lowest id" would
    # misattribute 'red_chassis' as the detachable part and let 'red_shell'
    # pose as the legitimate root instead -- the message must name red_shell,
    # not falsely accuse red_chassis of "having a free joint but not being the
    # root".
    out = _compose_connected(tmp_path)
    tree = ET.parse(str(out))
    worldbody = tree.getroot().find("worldbody")
    chassis = worldbody.find("body[@name='red_chassis']")
    assert chassis is not None
    chassis.addprevious(_make_shell())
    tree.write(str(out))

    model, data, red, blue = _env_from(out)
    with pytest.raises(ValueError, match="Invalid robot structure") as excinfo:
        SumoEnv(model, data, str(out), red, blue, max_steps=1)
    message = str(excinfo.value)
    assert "red_shell" in message
    assert "'red_chassis' has a free joint" not in message


def test_runtime_rejects_a_prepended_flying_part_when_root_has_no_free_joint(tmp_path):
    # Same prepended-shell exploit, but red_chassis's own freejoint is also
    # stripped, so the untethered red_shell is the ONLY free joint matching the
    # "red_" prefix. A rule that just counts "exactly one free joint for this
    # prefix" would silently accept this (the count is 1) -- the free joint has
    # to belong to the sole top-level body, not merely be unique. The message
    # must still identify the problem: the root has no free joint, and/or a
    # non-root body (red_shell) has one.
    out = _compose_connected(tmp_path)
    tree = ET.parse(str(out))
    worldbody = tree.getroot().find("worldbody")
    chassis = worldbody.find("body[@name='red_chassis']")
    assert chassis is not None
    chassis.remove(chassis.find("freejoint"))
    chassis.addprevious(_make_shell())
    tree.write(str(out))

    model, data, red, blue = _env_from(out)
    with pytest.raises(ValueError, match="Invalid robot structure") as excinfo:
        SumoEnv(model, data, str(out), red, blue, max_steps=1)
    message = str(excinfo.value)
    assert "red_shell" in message
    assert "no free joint" in message.lower()
