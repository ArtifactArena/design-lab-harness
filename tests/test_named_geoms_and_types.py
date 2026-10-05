"""Two checks the prompt states and the code now runs (Task 15, items J and J2).

`validate_geom_names` is the presence half of "Every geom must have a unique name"
(MuJoCo's compiler owns uniqueness); `validate_material_attributes` rejects
`type="mesh"`, which the supported-types list never included.
"""
import mjarena.core.unified_builder  # noqa: F401  (import order)

import mujoco
import pytest
import yaml
from pathlib import Path

from mjarena.design_shop.rules.mj_validators import validate_geom_names
from mjarena.design_shop.utils import validate_material_attributes

ROOT = Path(__file__).resolve().parents[1]
PALETTE = yaml.safe_load((ROOT / "configs/rules/materials_store.yaml").read_text())["materials"]

BOT = """<mujoco>
  <worldbody>
    <body name="chassis" pos="0 0 0.3">
      <freejoint name="root"/>
      <geom {named} type="box" size="0.3 0.3 0.1" mass="10"/>
    </body>
  </worldbody>
  <actuator/>
</mujoco>"""


def _compile(xml: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(xml)


def test_a_named_geom_passes_and_an_unnamed_one_is_named_in_the_failure():
    assert validate_geom_names(_compile(BOT.format(named='name="deck"'))).passed
    result = validate_geom_names(_compile(BOT.format(named="")))
    assert not result.passed
    assert result.errors == ["Geom #0 in body 'chassis' has no name; every geom needs a unique name="]


def test_many_unnamed_geoms_are_one_message_not_forty():
    """Same shape as validate_actuator_names: one error, every offender listed."""
    extra = '<geom type="box" size="0.1 0.1 0.1" mass="1"/>' * 2
    result = validate_geom_names(_compile(BOT.format(named="").replace("</body>", extra + "</body>")))
    assert not result.passed
    assert len(result.errors) == 1
    assert result.errors[0] == (
        "Geom #0 in body 'chassis', Geom #1 in body 'chassis', Geom #2 in body 'chassis' "
        "have no name; every geom needs a unique name=")
    assert result.details["unnamed_indices"] == [0, 1, 2]


def test_mesh_geoms_are_rejected_with_the_sentence_the_prompt_uses():
    mesh_bot = """<mujoco>
      <asset><mesh name="hull" vertex="0 0 0  0.6 0 0  0 0.4 0  0 0 0.3"
                   face="0 2 1  0 1 3  0 3 2  1 2 3"/></asset>
      <worldbody><body name="chassis"><freejoint/>
        <geom name="hull_geom" type="mesh" mesh="hull" material="steel"/>
      </body></worldbody>
    </mujoco>"""
    errors, _ = validate_material_attributes(mesh_bot, PALETTE, default_material="foam")
    assert errors == ['Geom \'hull_geom\': type="mesh" is not accepted; express the same '
                      'shape as an sdf geom from the same vertices and faces.']
    # the same shape as an sdf geom is accepted
    errors, _ = validate_material_attributes(mesh_bot.replace('type="mesh"', 'type="sdf"'),
                                             PALETTE, default_material="foam")
    assert errors == []


def test_the_prompts_state_both_rules():
    for name in ("autoresearch_prompt.md", "sampling_prompt.md"):
        text = (ROOT / "configs/rules" / name).read_text(encoding="utf-8")
        assert "A geom without a name fails validation." in text, name
        assert '`type="mesh"` is not accepted' in text, name


@pytest.mark.parametrize("group", ["check_actuators"])
def test_geom_names_runs_in_the_same_check_group_as_actuator_names(group):
    from mjarena.design_shop.rules.hardware_rules import validate_mujoco_model
    from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
    cfg = ModelValidationConfig(ROOT / "configs/rules/rules.yaml", physics_mode="3d")
    labels = [r.label for r in validate_mujoco_model(_compile(BOT.format(named="")), cfg)]
    assert labels.index("Geom Names") == labels.index("Actuator Names") + 1
