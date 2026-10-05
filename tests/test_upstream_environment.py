"""Pin the environment snapshot and test the tool adapter's current contract."""
import hashlib
import json
from pathlib import Path

import agent
import config
import harness_lib
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_upstream_snapshot_is_exact():
    manifest = json.loads((ROOT / "upstream_environment.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    # Files the tournament batch ran with modified, each declared with its reason.
    for name, patch in manifest["tournament_patches"].items():
        assert name not in manifest["files"] and patch["reason"]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == patch["sha256"], name
    # Files edited for the public release (wording only), each declared with its reason.
    for name, edit in manifest["release_edits"].items():
        assert name not in manifest["files"] and name not in manifest["tournament_patches"] and edit["reason"]
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == edit["sha256"], name


def test_model_visible_engine_matches_runtime():
    manifest = json.loads((ROOT / "upstream_environment.json").read_text())
    unmirrored = manifest["reference_not_mirrored"]
    for path in (ROOT / "mjarena").rglob("*.py"):
        name = str(path.relative_to(ROOT))
        if "__pycache__" not in path.parts and name not in unmirrored:
            assert (ROOT / "arena_kit/reference" / path.relative_to(ROOT / "mjarena")).read_bytes() == path.read_bytes()


def test_prompt_uses_current_sampling_text_and_tool_format():
    import mujoco
    sampling = (ROOT / "configs/rules/sampling_prompt.md").read_text()
    assert "{mujoco_version}" in sampling  # the source keeps the placeholder ...
    prompt = agent.build_system_prompt(ROOT / "arena_kit", step=2, max_steps=10)
    assert prompt.startswith(sampling.replace("{mujoco_version}", mujoco.__version__))
    assert "{mujoco_version}" not in prompt  # ... the model never sees it
    assert f"MuJoCo {mujoco.__version__}" in prompt
    assert "Current model turn: 2 of 10." in prompt
    assert "Model turns remaining after this turn: 8." in prompt
    assert "Always use the tool-call format on every turn, including the final turn." in prompt
    assert "corresponding arguments of `save_bot`" in prompt
    assert "{{" not in prompt


def test_match_config_resolves_from_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = harness_lib.get_match_config()
    assert cfg.size_limits == (2.44, 2.44, 3.05)
    assert cfg.match_time == 300.0
    assert cfg.constraints_path.is_file()
    assert Path.cwd() == tmp_path


def test_verification_exercises_robot_specific_observation():
    xml = (ROOT / "tests/fixtures/valid_bot/robot.xml").read_text()
    controller = '''def policy_step(obs):
    assert obs['my_robot']['bodies']
    assert 'contacts' in obs and 'opponent_proximity' in obs
    return {name: 0.0 for name in obs['my_actuator_velocity']}
'''
    result = harness_lib.run_verification_core(xml, controller)
    assert result["verification_passed"], result
    assert "initial observation of your robot" in result["feedback"]


def test_selection_includes_unqualified_saved_candidates(tmp_path, monkeypatch):
    ws = tmp_path
    for name, qualified in [("a", True), ("b", False)]:
        bot = ws / "bots" / name
        bot.mkdir(parents=True)
        (bot / "robot.xml").write_text(name)
        (bot / "controller.py").write_text(name)
        (bot / "bot_artifact.json").write_text(json.dumps({"name": name, "qualification_passed": qualified}))
    seen = []
    def tournament(workspace, names, n_seeds):
        seen.extend(names)
        return {"best": "b", "ranking": [{"name": "b"}, {"name": "a"}]}
    monkeypatch.setattr(harness_lib, "round_robin_core", tournament)
    assert harness_lib.resolve_submission_core(ws)["name"] == "b"
    assert seen == ["a", "b"]


def test_api_driver_does_not_follow_model_created_symlinks(tmp_path):
    secret = tmp_path / "other_run.txt"
    secret.write_text("OTHER_MODEL_BOT_CANARY")
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "notes.md").symlink_to(secret)
    assert "OTHER_MODEL_BOT_CANARY" not in agent._read_if(ws / "notes.md")
    bot = ws / "bots" / "fake"
    bot.mkdir(parents=True)
    (bot / "bot_artifact.json").symlink_to(secret)
    assert harness_lib.list_bots(ws) == []
    (ws / "_tool_request.json").symlink_to(secret)
    import linux_sandbox
    with pytest.raises(OSError):
        linux_sandbox.execute(tmp_path / "unused-runtime", ws, {"tool": "list_bots"})
    assert secret.read_text() == "OTHER_MODEL_BOT_CANARY"


def test_run_dir_resolves_through_a_symlinked_runs_folder(tmp_path, monkeypatch):
    real = tmp_path / "storage"
    real.mkdir()
    (tmp_path / "runs").symlink_to(real)
    monkeypatch.setattr(agent, "RUNS", tmp_path / "runs")
    run_dir = agent.new_run_dir()
    assert run_dir == run_dir.resolve() and run_dir.parent == real
    (run_dir / "notes.md").write_text("visible")
    assert agent._read_if(run_dir / "notes.md") == "visible"


def test_timeout_result_skips_a_model_made_fifo(tmp_path):
    import os
    import tool_results
    progress = tmp_path / "_tool_progress" / "abc"
    progress.mkdir(parents=True)
    os.mkfifo(progress / "seed-fifo.json")  # reading this would block the host forever
    (progress / "seed-ok.json").write_text(json.dumps({"seed": 0, "winner": "red", "complete": True}))
    result = tool_results.timeout_result(tmp_path, {"tool": "run_match", "_progress_dir": "_tool_progress/abc"}, 60)
    assert result["partial_results"]["completed_seed_count"] == 1
