"""A robot bolted to the world (no root freejoint) must fail validation."""
import re
from pathlib import Path

import mjarena.core.unified_builder  # noqa: F401  (import order)
from mjarena.design_shop.pipelines.morphology_pipeline import validate_morphology
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "configs/rules/rules.yaml"
BOT = (ROOT / "tests/fixtures/study_bot/robot.xml").read_text()


def _validate(xml: str):
    return validate_morphology(xml, ModelValidationConfig(constraints_yaml_path=RULES, physics_mode="3d"))


def test_study_bot_has_a_freejoint_and_passes():
    r = _validate(BOT)
    assert r.passed, r.feedback
    assert "── 3D Root Freejoint ── PASSED" in r.feedback
    assert "2D Physics" not in r.feedback


def test_bot_without_root_freejoint_fails_that_check():
    welded = re.sub(r"<freejoint[^>]*/>", "", BOT, count=1)
    assert welded != BOT
    r = _validate(welded)
    assert not r.passed
    assert "── 3D Root Freejoint ── FAILED" in r.feedback
