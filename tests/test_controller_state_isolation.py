"""A reused controller keeps state within a match, never across matches or sides."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from mjarena.eval.runtime import build_policy_callable
from mjarena.policy_spec import PolicySpec
from mjarena.runner.episode import run_match


FIXTURE = Path(__file__).resolve().parent / "fixtures/arena_rules.xml"
STATEFUL_CODE = '''
calls = 0

def make_counter():
    count = 0
    def increment():
        nonlocal count
        count += 1
        return count
    return increment

increment = make_counter()

def policy_step(obs, history=[]):
    global calls
    calls += 1
    history.append(calls)
    policy_step.calls = getattr(policy_step, "calls", 0) + 1
    return {"motor": (calls + len(history) + increment() + policy_step.calls) / 100}
'''


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("from_spec", [False, True])
def test_repeated_matches_start_fresh_on_both_sides(tmp_path, workers, from_spec):
    policy = (
        PolicySpec.controller_code(STATEFUL_CODE, ["motor"]).build_callable()
        if from_spec else build_policy_callable(STATEFUL_CODE, ["motor"])
    )
    # Previous verification calls must not contaminate the first match either.
    assert policy({}) == {"motor": 0.04}
    assert policy({}) == {"motor": 0.08}

    def play(seed):
        return run_match(
            composed_xml=FIXTURE,
            red_policy_py=policy,
            blue_policy_py=policy,
            out_dir=tmp_path / str(seed),
            seed=seed,
            max_steps=3,
            save_video=False,
            quiet=True,
            use_gui=False,
            camera_mode="tracking",
        )

    if workers == 1:
        records = [play(seed) for seed in range(3)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(play, range(3)))

    for record in records:
        assert record.num_steps == 3
        for side, actions in (("red", record.red_actions), ("blue", record.blue_actions)):
            np.testing.assert_allclose([a[f"{side}_motor"] for a in actions], [0.04, 0.08, 0.12])

    # Each match used its own state without resetting/mutating the shared input.
    assert policy({}) == {"motor": 0.12}
