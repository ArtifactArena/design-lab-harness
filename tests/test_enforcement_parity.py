"""Enforcement parity with the build harness (ArtifactArena/harness): same engine, same
qualification (3 x 20 s vs the block, inactivity armed), same sparring/tournament length
(tournament.match: 300 s, inactivity armed), same pins."""
import shutil
from pathlib import Path

import mjarena.core.unified_builder  # noqa: F401  (import order: avoids the dspy_core/design_shop cycle)
import config
import harness_lib

ROOT = Path(__file__).resolve().parents[1]
KIT = Path(config.__file__).resolve().parent
VALID = ROOT / "tests" / "fixtures" / "valid_bot"

def _zero_controller(robot_xml: str) -> str:
    """A controller that names every actuator explicitly (static validation uses a dummy obs)."""
    M = harness_lib._mj()
    processed = harness_lib._prepare_xml(M, robot_xml, harness_lib.get_match_config())["xml"]
    names = M["get_actuator_names_from_xml_string"](processed)
    body = ", ".join(f"'{n}': 0.0" for n in names)
    return f"def policy_step(obs):\n    return {{{body}}}\n"


def _capturing_M(captured: dict):
    M = dict(harness_lib._mj())
    real = M["create_match_runner"]

    def wrapped(**kw):
        captured.clear()
        captured.update(kw)
        # Assert the real settings, then run only a short integration probe.
        actual = dict(kw)
        if "match_time" in actual:
            actual["match_time"] = 0.1
        return real(**actual)

    M["create_match_runner"] = wrapped
    return M


def test_pins_match_the_build_harness():
    from mjarena.dspy_core import NO_LLM_TIMEOUT_S
    assert NO_LLM_TIMEOUT_S == 6000
    assert "mujoco==3.10.0" in (ROOT / "requirements.txt").read_text()
    assert config.VERIFY_ROLLOUTS == 3 and config.QUALIFY_MATCH_TIME == 20.0
    assert config.MATCH_TIMEOUT >= 3 * 300


def test_qualification_runner_plays_20s_with_inactivity_armed(tmp_path):
    captured = {}
    M = _capturing_M(captured)
    cfg = harness_lib.get_match_config()
    prep = harness_lib._prepare_xml(M, (VALID / "robot.xml").read_text(), cfg)
    assert prep["passed"], prep["feedback"]
    proc = tmp_path / "proc.xml"
    proc.write_text(prep["xml"])
    harness_lib._stationary_match_fn(M, cfg, proc, tmp_path)
    assert captured["match_time"] == 20.0 and "max_steps" not in captured
    assert captured["inactivity_timeout_seconds"] == 10.0
    assert captured["inactivity_min_displacement"] == 0.5
    assert captured["inactivity_exempt_prefixes"] == ["blue_"]
    assert captured["contact_fidelity"] == "high"
    assert captured["size_limits"] == (2.44, 2.44, 3.05)
    harness_lib._stationary_match_fn(M, cfg, proc, tmp_path, max_steps=3)   # the obs probe
    assert captured["max_steps"] == 3 and "match_time" not in captured


def test_run_match_plays_tournament_length_with_inactivity_armed(tmp_path, monkeypatch):
    captured = {}
    M = _capturing_M(captured)
    monkeypatch.setattr(harness_lib, "_mj", lambda: M)
    ws = tmp_path / "ws"
    shutil.copytree(KIT, ws)
    shutil.copy(VALID / "robot.xml", ws / "robot.xml")
    shutil.copy(VALID / "controller.py", ws / "controller.py")
    res = harness_lib.run_match_core(ws, "draft", "stationary", ws / "matches", n_seeds=1)
    assert res.get("ok"), res
    assert captured["match_time"] == 300.0 and "max_steps" not in captured
    assert captured["inactivity_timeout_seconds"] == 10.0
    assert captured["inactivity_min_displacement"] == 0.5
    assert captured["inactivity_exempt_prefixes"] == ["blue_"]       # the zero-policy block
    res = harness_lib.run_match_core(ws, "draft", "draft", ws / "matches", n_seeds=1)
    assert res.get("ok"), res
    assert captured["inactivity_exempt_prefixes"] is None              # an authored opponent must move


def test_motionless_bot_fails_qualification_by_inactivity(tmp_path):
    xml = (VALID / "robot.xml").read_text()
    res = harness_lib.run_qualification_core(xml, _zero_controller(xml), tmp_path / "q")
    assert res["ok"] and res["validation_passed"] is True, res
    assert res["qualification_passed"] is False
    assert "inactiv" in res["feedback"].lower(), res["feedback"][-800:]
