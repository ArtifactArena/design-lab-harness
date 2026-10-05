"""Inactivity: 3D window diameter, 10 s, 0.5 m; both stalled -> 1 cm COM height tiebreak, else tie."""
import mjarena.core.unified_builder  # noqa: F401
import numpy as np
import pytest

from mjarena.envs.sumo import SumoEnv


def _env_with_history(monkeypatch, dt=0.01, window=10.0):
    env = SumoEnv.__new__(SumoEnv)
    env.control_timestep = dt
    env._inactivity_min_displacement = 0.5
    env._inactivity_window_steps = int(window / dt)
    env._inactivity_exempt = set()
    from collections import deque
    env._red_com_history = deque(maxlen=env._inactivity_window_steps + 1)
    env._blue_com_history = deque(maxlen=env._inactivity_window_steps + 1)
    env._inactivity_timers = {"red": 0.0, "blue": 0.0}
    return env


class _C:
    def __init__(self, pos):
        self.com_position = np.asarray(pos, dtype=float)


def _drive(env, red_positions, blue_positions):
    result = None
    for r, b in zip(red_positions, blue_positions):
        env.red_contender, env.blue_contender = _C(r), _C(b)
        result = env.check_inactivity()
        if result:
            return result
    return result


def test_stationary_bot_loses_at_exactly_ten_seconds(monkeypatch):
    env = _env_with_history(monkeypatch)
    still = [(0, 0, 0.2)] * 1002
    moving = [(0.01 * i, 0, 0.2) for i in range(1002)]
    outcomes = []
    for i, (r, b) in enumerate(zip(still, moving)):
        env.red_contender, env.blue_contender = _C(r), _C(b)
        outcomes.append((i, env.check_inactivity()))
    first = next(i for i, o in outcomes if o)
    assert first == 1000 and outcomes[first][1] == "red"
    assert env._inactivity_timers["red"] == pytest.approx(10.0)


def test_vertical_motion_counts(monkeypatch):
    env = _env_with_history(monkeypatch)
    hop = [(0, 0, 0.2 + (0.6 if i % 200 < 100 else 0.0)) for i in range(1002)]
    assert _drive(env, hop, [(0.01 * i, 0, 0.2) for i in range(1002)]) is None


@pytest.mark.parametrize("dz,expected", [(0.0, "both"), (0.02, "blue"), (-0.02, "red")])
def test_both_stalled_uses_one_centimetre_height_tiebreak(monkeypatch, dz, expected):
    env = _env_with_history(monkeypatch)
    red = [(0, 0, 0.2 + dz)] * 1002
    blue = [(3, 0, 0.2)] * 1002
    assert _drive(env, red, blue) == expected


def test_exempt_side_never_accrues(monkeypatch):
    env = _env_with_history(monkeypatch)
    env._inactivity_exempt = {"blue_"}
    assert _drive(env, [(0.01 * i, 0, 0.2) for i in range(1002)], [(3, 0, 0.2)] * 1002) is None
    assert env._inactivity_timers["blue"] == 0.0


# ── real matches ────────────────────────────────────────────────────────────

from pathlib import Path  # noqa: E402

DT = 0.01            # control timestep at contact_fidelity=high
MIN_DISP = 0.5
WINDOW_STEPS = 1000  # 10 s

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "mjarena/assets/sumo_ring_env_cinematic_3d.xml"
PUSHER = ROOT / "mjarena/core/assets/baseline_bots/baseline-pusher/robot.xml"
BLOCK = ROOT / "mjarena/core/assets/stationary_block_3d.xml"


def _compose(tmp_path: Path, red_xml: Path) -> Path:
    from mjarena.envs.sumo import compose_sumo_model
    out = tmp_path / "composed.xml"
    compose_sumo_model(env_xml=str(ARENA), robot_red_xml=str(red_xml), robot_blue_xml=str(BLOCK),
                       out_path=str(out), randomize_spawn_3d=True, spawn_seed=0)
    return out


def _run(tmp_path: Path, composed: Path, red_policy, match_time: float):
    from mjarena.runner.episode import run_match
    return run_match(
        composed_xml=composed, red_policy_py=red_policy, blue_policy_py=lambda obs: {},
        out_dir=tmp_path, max_steps=None, use_gui=False, save_video=False,
        camera_mode="tracking", quiet=True, seed=0,
        match_time=match_time, inactivity_timeout_seconds=10.0,
        inactivity_min_displacement=MIN_DISP, inactivity_exempt_prefixes=["blue_"],
    )


def test_motionless_bot_loses_by_inactivity_at_ten_seconds(tmp_path):
    rec = _run(tmp_path, _compose(tmp_path, PUSHER), lambda obs: {}, match_time=15.0)
    assert rec.termination_reason == "inactivity"
    assert rec.winner == "blue"
    # the loss lands on the 1000th control step; the record counts completed steps
    assert rec.num_steps in (WINDOW_STEPS - 1, WINDOW_STEPS)
    # the prompt-facing replay carries only informative fields
    d = rec.to_prompt_dict()
    for dead in ("qpos", "video_fps", "control_dt", "blue_actions", "blue_orientations",
                 "blue_velocities", "blue_inactivity_timers", "qacc_gear_diagnostics"):
        assert dead not in d, dead
    assert d["red_inactivity_timers"][-1] >= 9.9 and d["termination_reason"] == "inactivity"
