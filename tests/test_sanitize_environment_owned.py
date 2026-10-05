import mjarena.core.unified_builder  # noqa: F401
import xml.etree.ElementTree as ET

import mujoco
import pytest

from mjarena.design_shop.rules.hardware_rules import FORBIDDEN_ROOT_TAGS
from mjarena.design_shop.utils import ROBOT_COLLISION_SETTINGS, sanitize_robot_xml_report

ROBOT = """<mujoco>
  <compiler angle="degree"/>
  <asset>
    <mesh name="wedge" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    <material name="steel" rgba="1 0 0 1"/>
  </asset>
  <default><joint damping="30" armature="0.5"/><geom condim="1" friction="9 9 9" solref="0.5 4"/></default>
  <worldbody>
    <body name="chassis" pos="0 0 0.3">
      <freejoint name="root" damping="5" frictionloss="2" stiffness="7" armature="1"/>
      <geom name="box" type="box" size="0.3 0.3 0.1" material="steel" condim="4" priority="0" margin="0.1"/>
      <site name="a" pos="0 0 0"/>
      <body name="arm"><joint name="hinge" damping="120" armature="0.2"/>
        <site name="b" pos="0 0 0"/>
        <geom name="arm_g" type="sdf" mesh="wedge" material="aluminum"/></body>
    </body>
  </worldbody>
  <contact><exclude body1="chassis" body2="arm"/></contact>
  <tendon><spatial name="cable" armature="0.1"><site site="a"/><site site="b"/></spatial></tendon>
  <actuator><motor name="m" joint="hinge" gear="100"/></actuator>
</mujoco>"""


def test_environment_owns_collision_root_armature_units_and_assets():
    xml, removed, ignored = sanitize_robot_xml_report(ROBOT, FORBIDDEN_ROOT_TAGS)
    root = ET.fromstring(xml)
    assert root.find("compiler").get("angle") == "radian"
    assert root.find("asset/mesh").get("inertia") == "exact"
    assert root.find("asset/material") is None            # only <mesh> children survive in <asset>
    assert root.find("contact") is None
    for geom in root.iter("geom"):
        for attr, value in ROBOT_COLLISION_SETTINGS.items():
            assert geom.get(attr) == value, (geom.get("name"), attr)
        assert geom.get("friction") is None or geom.get("friction").split()[1:] == ["0.005", "0.0001"]
    rj = root.find("worldbody/body/freejoint")
    assert all(rj.get(a) is None for a in ("damping", "frictionloss", "stiffness", "armature"))
    hinge = root.find(".//joint[@name='hinge']")
    assert hinge.get("damping") == "120" and hinge.get("armature") == "0"
    assert root.find("tendon/spatial").get("armature") == "0"
    assert any("armature" in note for note in ignored)
    assert any("condim" in note or "collision" in note for note in ignored)
    assert root.find("default/geom").get("condim") == "6"
    assert root.find("default/joint").get("armature") == "0"


def test_default_block_damping_cannot_reach_root_joint():
    xml, _, _ = sanitize_robot_xml_report(ROBOT, FORBIDDEN_ROOT_TAGS)
    model = mujoco.MjModel.from_xml_string(xml.replace('material="steel"', 'density="1000"')
                                           .replace('material="aluminum"', 'density="1000"'))
    root_dofs = range(6)
    assert all(model.dof_damping[d] == 0 and model.dof_frictionloss[d] == 0 for d in root_dofs)
    assert model.dof_damping[6] == 120
    assert (model.dof_armature == 0).all() and (model.tendon_armature == 0).all()
