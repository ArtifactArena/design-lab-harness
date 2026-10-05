"""The qualification opponent is a palette-plastic slab: 1.2 x 1.2 x 0.25 m, 342 kg.

User decision 2026-09-16 (audit flag F-N11): the block must be built from a palette
material and weigh roughly a third to a half of the 800 kg mass cap, so that
qualification is a real push test rather than nudging a 36 kg foam pad off the ring.
Chosen: plastic (950 kg/m3, sliding friction 0.25), half-extents 0.60 0.60 0.125 ->
0.36 m3 -> 342 kg.

Nothing about the block's physics is hand-written: mass and friction are the palette's
(configs/rules/materials_store.yaml) and condim belongs to composition.
"""
import mjarena.core.unified_builder  # noqa: F401  (import order: avoids the dspy_core/design_shop cycle)

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "configs/rules/rules.yaml"
MATERIALS = ROOT / "configs/rules/materials_store.yaml"
ARENA = ROOT / "mjarena/assets/sumo_ring_env_cinematic_3d.xml"
BLOCK = ROOT / "mjarena/core/assets/stationary_block_3d.xml"
FIX = ROOT / "tests/fixtures/study_bot"

HALF_EXTENTS = (0.60, 0.60, 0.125)
EXPECTED_MASS = 342.0
EXPECTED_FRICTION = [0.25, 0.005, 0.0001]

PROMPT_FILES = ("autoresearch_prompt.md", "sampling_prompt.md")
BLOCK_SENTENCE = (
    "The block is a 1.2 x 1.2 x 0.25 m plastic slab of about 342 kg with the palette's "
    "plastic friction."
).replace(" x ", " × ")


def _plastic() -> dict:
    return yaml.safe_load(MATERIALS.read_text(encoding="utf-8"))["materials"]["plastic"]


def _block_geom(xml_text: str) -> ET.Element:
    geom = ET.fromstring(xml_text).find("worldbody/body/geom")
    assert geom is not None, "block asset has no body geom"
    return geom


# ── the asset itself ────────────────────────────────────────────────────────

def test_asset_is_a_plastic_slab_with_no_hand_written_physics():
    root = ET.fromstring(BLOCK.read_text(encoding="utf-8"))
    body = root.find("worldbody/body")
    assert body is not None and body.get("name") == "block"
    assert body.find("freejoint") is not None, "3D block needs its freejoint"
    actuator = root.find("actuator")
    assert actuator is not None and len(actuator) == 0, "the block has no actuators"

    geom = _block_geom(BLOCK.read_text(encoding="utf-8"))
    assert geom.get("material") == "plastic"
    assert tuple(float(v) for v in geom.get("size").split()) == HALF_EXTENTS
    # density / friction / priority are the palette's and composition's business
    for attribute in ("density", "friction", "priority"):
        assert geom.get(attribute) is None, f"{attribute}= is hand-written in the asset"


def test_the_baked_mass_is_exactly_the_palette_product():
    """mass= is baked so a raw compile weighs the qualification block; it may not drift."""
    plastic = _plastic()
    volume = 8.0 * HALF_EXTENTS[0] * HALF_EXTENTS[1] * HALF_EXTENTS[2]
    expected = volume * float(plastic["density_kg_m3"])
    assert expected == pytest.approx(EXPECTED_MASS, abs=1e-6)
    assert float(_block_geom(BLOCK.read_text(encoding="utf-8")).get("mass")) == pytest.approx(
        expected, abs=1e-6)


def test_raw_asset_compiles_to_the_same_block():
    model = mujoco.MjModel.from_xml_path(str(BLOCK))
    block = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    assert float(model.body_mass[block]) == pytest.approx(EXPECTED_MASS, abs=1.0)


# ── the palette resolution the qualification path performs ──────────────────

def test_qualification_block_xml_takes_mass_and_friction_from_the_palette():
    from mjarena.core.qualification_block import qualification_block_xml

    geom = _block_geom(qualification_block_xml(RULES, physics_mode="3d"))
    plastic = _plastic()
    assert float(geom.get("mass")) == pytest.approx(EXPECTED_MASS, abs=1e-6)
    assert [float(v) for v in geom.get("friction").split()] == [
        float(plastic["friction"]), 0.005, 0.0001]
    assert geom.get("priority") == "2"


# ── the block as it is actually played ──────────────────────────────────────

@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    from mjarena.core.qualification_block import write_qualification_block
    from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
    from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
    from mjarena.envs.sumo import compose_sumo_model

    out = tmp_path_factory.mktemp("qualification_block")
    result = validate_morphology((FIX / "robot.xml").read_text(),
                                 ModelValidationConfig(RULES, physics_mode="3d"))
    assert result.passed, result.feedback
    robot = out / "robot.xml"
    robot.write_text(result.processed_xml)
    block = write_qualification_block(out / "block.xml", RULES, physics_mode="3d")
    composed = out / "composed.xml"
    compose_sumo_model(env_xml=str(ARENA), robot_red_xml=str(robot), robot_blue_xml=str(block),
                       out_path=str(composed), randomize_spawn_3d=True, spawn_seed=0)
    return composed


def test_composed_block_weighs_342_kg_with_plastic_contact(composed):
    model = mujoco.MjModel.from_xml_path(str(composed))
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "blue_block")
    assert body != -1, "composed model has no blue_block body"
    assert float(model.body_mass[body]) == pytest.approx(EXPECTED_MASS, abs=1.0)

    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    assert geoms, "block body has no geom"
    for g in geoms:
        np.testing.assert_allclose(model.geom_friction[g], EXPECTED_FRICTION, rtol=0, atol=1e-9)
        assert int(model.geom_condim[g]) == 6
        assert int(model.geom_priority[g]) == 2


def test_a_twenty_second_qualification_match_against_the_block_runs(composed, tmp_path):
    from mjarena.runner.episode import run_match

    namespace: dict = {}
    exec((FIX / "controller.py").read_text(), namespace)
    policy = namespace["policy_step"]

    rec = run_match(
        composed_xml=composed, red_policy_py=policy, blue_policy_py=lambda obs: {},
        out_dir=tmp_path, max_steps=None, use_gui=False, save_video=False,
        camera_mode="tracking", quiet=True, seed=0, match_time=20.0,
        inactivity_timeout_seconds=10.0, inactivity_min_displacement=0.5,
        inactivity_exempt_prefixes=["blue_"],
    )
    assert rec.blue_mass == pytest.approx(EXPECTED_MASS, abs=1.0)
    assert not rec.physics_unstable and rec.controller_errors == []
    assert rec.num_steps > 0
    # the block has no actuators and carries the blue_ exemption: it can never be the
    # one that times out for inactivity
    assert not (rec.termination_reason == "inactivity" and rec.winner == "red"), (
        "the block lost by inactivity — the blue_ exemption is not armed")
    # a study bot that qualified against the old 36 kg pad must not LOSE to the slab
    assert rec.winner != "blue", (rec.winner, rec.termination_reason)


# ── the model is told what it is fighting ───────────────────────────────────

@pytest.mark.parametrize("name", PROMPT_FILES)
def test_the_block_sentence_appears_exactly_once_in_every_prompt(name):
    text = (ROOT / "configs/rules" / name).read_text(encoding="utf-8")
    assert text.count(BLOCK_SENTENCE) == 1, name
