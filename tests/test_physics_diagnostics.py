"""Motor diagnostics must interpret target IDs in the right MuJoCo namespace."""
import mjarena.core.unified_builder  # noqa: F401 (application import order)

import json

import mujoco
import pytest

from mjarena.agents.runtime import BotRuntime
from mjarena.design_shop.tools.physics_diagnostics import get_physics_diagnostics
from mjarena.envs.sumo import SumoEnv
from mjarena.runner.episode import Match


def _model(transmission):
    bodies, tendons, motors = [], [], []
    for color, x in (("red", -1), ("blue", 1)):
        # Extra sites ensure a site ID can be a different robot's joint ID or
        # exceed njnt entirely. Both silent misattribution and crashes matter.
        bodies.append(f'''<body name="{color}_root" pos="{x} 0 .31">
          <freejoint name="{color}_free"/>
          <geom name="{color}_hull" type="box" size=".2 .2 .1" mass="30"/>
          <site name="{color}_anchor"/><site/><site/>
          <body name="{color}_arm" pos="0 0 .3">
            <joint name="{color}_slide" type="slide" axis="0 0 1"/>
            <geom name="{color}_tip" type="sphere" size=".1" mass="10"/>
            <site name="{color}_end"/>
          </body>
        </body>''')
        tendons.append(f'''<fixed name="{color}_cable">
          <joint joint="{color}_slide" coef="1"/></fixed>''')
        target = {
            "joint": f'joint="{color}_slide"',
            "jointinparent": f'jointinparent="{color}_slide"',
            "tendon": f'tendon="{color}_cable"',
            "site": f'site="{color}_end" refsite="{color}_anchor"',
            "slidercrank": (f'cranksite="{color}_anchor" '
                            f'slidersite="{color}_end" cranklength="1"'),
        }[transmission]
        motors.append(f'<motor name="{color}_drive" {target} gear="10"/>')
    return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <geom name="sumo_ring" type="cylinder" size="7.5 .2"/>
      <geom name="outside_floor" type="plane" size="20 20 .1" pos="0 0 -1"/>
      {''.join(bodies)}</worldbody><tendon>{''.join(tendons)}</tendon>
      <actuator>{''.join(motors)}</actuator></mujoco>''')


@pytest.mark.parametrize("transmission", ["joint", "jointinparent", "tendon", "site", "slidercrank"])
@pytest.mark.parametrize("color", ["red", "blue"])
def test_diagnostics_report_the_transmission_mount(transmission, color):
    model = _model(transmission)
    diagnostics = get_physics_diagnostics(model, prefix=f"{color}_")
    assert diagnostics["num_actuators"] == 1
    actuator, = diagnostics["actuators"]
    # Tendon diagnostics use the root assembly, where tendon motor mass is
    # mounted. These static ratios are not effective mechanism inertia.
    mount = "root" if transmission == "tendon" else "arm"
    body = model.body(f"{color}_{mount}")
    assert actuator["name"] == "drive"
    assert actuator["body_name"] == mount
    assert actuator["body_mass_kg"] == (30 if mount == "root" else 10)
    assert actuator["body_inertia_min"] == pytest.approx(min(body.inertia))
    assert actuator["gear_to_inertia_ratio"] == pytest.approx(round(10 / min(body.inertia), 1))
    expected_joint = (model.joint(f"{color}_slide").id
                      if transmission in ("joint", "jointinparent") else -1)
    assert actuator["joint_id"] == expected_joint
    bodies = {body["name"]: body for body in diagnostics["bodies"]}
    assert bodies[mount]["num_actuators"] == 1
    assert bodies[mount]["total_gear"] == 10
    assert sum(body["num_actuators"] for body in bodies.values()) == 1
    json.dumps(diagnostics, allow_nan=False)


@pytest.mark.parametrize("transmission", ["site", "tendon"])
@pytest.mark.parametrize("color", ["red", "blue"])
def test_qacc_with_non_joint_motor_retains_match_diagnostics(transmission, color):
    model = _model(transmission)
    data = mujoco.MjData(model)
    red = BotRuntime(model, data, lambda obs: {}, "red_")
    blue = BotRuntime(model, data, lambda obs: {}, "blue_")
    dof = int(model.jnt_dofadr[model.joint(f"{color}_slide").id])

    class WarningAtFirstStep(SumoEnv):
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            # Deterministic warning injection exercises the real reporting path
            # without relying on a numerically unstable mechanism.
            warning = self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC]
            warning.number = 1
            warning.lastinfo = dof
            return result

    env = WarningAtFirstStep(model, data, "", red, blue, max_steps=5,
                             inactivity_timeout_seconds=None)
    try:
        record = Match(env, red, blue, max_steps=5, headless=True, seed=0).run(
            save_video=False, quiet=True)
        assert record.termination_reason == "qacc"
        assert record.physics_unstable
        assert record.winner == ("blue" if color == "red" else "red")
        assert record.qacc_loser == color
        assert record.qacc_dof_index == dof
        assert record.qacc_body_name == f"{color}_arm"
        diagnostic, = record.qacc_gear_diagnostics
        mount = "root" if transmission == "tendon" else "arm"
        assert diagnostic["color"] == color
        assert diagnostic["actuator"] == "drive"
        assert diagnostic["gear"] == 10
        assert diagnostic["body_inertia_min"] == pytest.approx(
            min(model.body(f"{color}_{mount}").inertia))
        saved = json.loads(json.dumps(record.to_dict(), allow_nan=False))
        assert saved["qacc_gear_diagnostics"] == record.qacc_gear_diagnostics
    finally:
        env.detailed_observations.close()
        env.close()
