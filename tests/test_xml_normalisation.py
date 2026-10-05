"""Model XML with a declaration, a namespace or fences must not be turned into ns0 garbage."""
from pathlib import Path

import mjarena.core.unified_builder  # noqa: F401  (import order)
from mjarena.dspy_core import _strip_mjcf_tags
from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig

ROOT = Path(__file__).resolve().parents[1]
BOT = (ROOT / "tests/fixtures/study_bot/robot.xml").read_text()


def test_declaration_and_namespace_are_stripped_before_wrapping():
    raw = '<?xml version="1.0" encoding="UTF-8"?>\n' + BOT.replace("<mujoco", '<mujoco model="q" xmlns="http://www.roboti.us.a"', 1)
    out = _strip_mjcf_tags(raw)
    assert out.startswith("<mujoco ") and 'model="q"' in out.split(">", 1)[0]
    assert "xmlns" not in out and "<?xml" not in out
    assert out.count("<mujoco") == 1          # not wrapped a second time


def test_fenced_output_is_unwrapped():
    out = _strip_mjcf_tags("```xml\n" + BOT + "\n```")
    assert out.startswith("<mujoco") and "```" not in out


def test_namespaced_bot_validates_without_ns0():
    raw = '<?xml version="1.0"?>\n' + BOT.replace("<mujoco", '<mujoco xmlns="http://www.roboti.us.a"', 1)
    r = validate_morphology(_strip_mjcf_tags(raw), ModelValidationConfig(constraints_yaml_path=ROOT / "configs/rules/rules.yaml", physics_mode="3d"))
    assert r.passed, r.feedback
    assert "ns0" not in r.feedback and "ns0" not in r.processed_xml


def test_whitespace_inside_material_quotes_is_forgiven():
    raw = BOT.replace('material="', 'material=" ', 1)          # first geom: material=" steel" (or whatever it is)
    r = validate_morphology(_strip_mjcf_tags(raw), ModelValidationConfig(constraints_yaml_path=ROOT / "configs/rules/rules.yaml", physics_mode="3d"))
    assert r.passed, r.feedback


def test_model_asset_block_is_removed_and_reported():
    from mjarena.design_shop.rules.hardware_rules import sanitize_robot_xml
    import re as _re
    raw = _re.sub(
        r"(<mujoco[^>]*>)",
        r'\1<asset><mesh name="wedge" vertex="0 0 0  1 0 0  0 1 0  0 0 1" face="0 2 1  0 1 3  0 3 2  1 2 3"/>'
        r'<material name="rubber" friction="1.6"/></asset>',
        BOT, count=1)
    out, step = sanitize_robot_xml(raw)
    assert '<mesh name="wedge"' in out and 'inertia="exact"' in out    # <mesh> children survive
    assert "<material" not in out                                     # non-mesh children are removed
    assert step.passed and not step.quiet and "only inline <mesh>" in step.message


def test_child_joint_damping_is_kept():
    from mjarena.design_shop.rules.hardware_rules import sanitize_robot_xml
    raw = BOT.replace('type="hinge"', 'type="hinge" damping="40"', 1)
    out, step = sanitize_robot_xml(raw)
    assert 'damping="40"' in out
    assert "ROOT" not in step.message


def test_root_joint_anchoring_is_removed_and_reported():
    import re as _re
    from mjarena.design_shop.rules.hardware_rules import sanitize_robot_xml
    raw = _re.sub(r"<freejoint[^>]*/>", '<joint type="free" name="root" damping="500" frictionloss="20"/>', BOT, count=1)
    out, step = sanitize_robot_xml(raw)
    root_joint = _re.search(r'<joint [^>]*type="free"[^>]*/>', out).group(0)
    assert 'damping="0"' in root_joint and 'frictionloss="0"' in root_joint  # explicit zero, not anchoring
    assert "ROOT joint" in step.message and "removed automatically" in step.message and not step.quiet


def test_default_block_cannot_anchor_the_root():
    import re as _re
    import mujoco
    from mjarena.design_shop.rules.hardware_rules import sanitize_robot_xml
    raw = _re.sub(r"(<mujoco[^>]*>)", r'\1<default><joint damping="30"/></default>', BOT, count=1)
    out, step = sanitize_robot_xml(raw)
    root_joint = _re.search(r"<freejoint[^>]*/>", out).group(0)
    assert "damping" not in root_joint and "frictionloss" not in root_joint and "stiffness" not in root_joint
    assert 'damping="30"' in out                     # the default still applies to child joints
    # <default><joint damping=...> cannot reach the root freejoint even though it is
    # never spelled out explicitly on the tag: MuJoCo compiles it to 0 regardless.
    compiled_xml = _re.sub(r'material="[^"]*"', 'density="1000"', out)
    model = mujoco.MjModel.from_xml_string(compiled_xml)
    root_dofs = range(6)
    assert all(model.dof_damping[d] == 0 for d in root_dofs)
    assert model.dof_damping[6] == 30                # first wheel joint inherits the default



def test_malformed_xml_is_a_named_failure_that_quotes_the_line():
    from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
    from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
    broken = BOT.replace("</worldbody>", "</body>\n  </worldbody>", 1)     # one closing tag too many
    r = validate_morphology(broken, ModelValidationConfig(constraints_yaml_path=ROOT / "configs/rules/rules.yaml", physics_mode="3d"))
    assert not r.passed
    assert "XML is not well-formed at line" in r.feedback and "mismatched tag" in r.feedback
    assert "Unexpected error" not in r.feedback
    assert "</body>" in r.feedback                                        # the offending line is quoted
