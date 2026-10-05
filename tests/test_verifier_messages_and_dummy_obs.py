"""Verifier feedback states what it actually did, and the fallback observation is real.

Task 15, items K and L: the exercise message claimed branch coverage it did not have
on the normal path, and the dummy observation reported a ring that does not exist.
"""
import mjarena.core.unified_builder  # noqa: F401  (import order)

from mjarena.agents.types import BotObservation
from mjarena.design_shop.policy_base import create_dummy_observation_dict
from mjarena.design_shop.rules.software_rules import exercise_policy
from mjarena.envs.detailed_observations import SURFACE_CUTOFF, SURFACE_LIMIT

POLICY = lambda obs: {"drive": 0.0}          # noqa: E731


def test_the_real_observation_path_says_only_what_it_ran():
    real = create_dummy_observation_dict()   # stands in for the composed robot's observation
    result = exercise_policy(POLICY, ["drive"], base_observation=real)
    assert result.passed
    assert result.message == (
        "policy_step(obs) executed successfully on the initial observation of your robot.")


def test_the_synthetic_suite_names_the_cases_it_ran():
    result = exercise_policy(POLICY, ["drive"])
    assert result.passed
    assert result.message == (
        "policy_step(obs) executed successfully on representative cases: default, "
        "edge_recovery, engage_contact, hold_center_far, anti_inactivity.")


def test_a_failing_case_is_still_named():
    def raises(obs):
        if obs["my_edge_distance"] < 2.0:
            raise ZeroDivisionError("edge branch")
        return {"drive": 0.0}
    result = exercise_policy(raises, ["drive"])
    assert not result.passed and "'edge_recovery'" in result.message


def test_the_fallback_observation_reports_the_numbers_a_match_reports():
    obs = create_dummy_observation_dict()
    assert BotObservation.get_dummy_bot_obs().ring_radius == 7.5
    assert obs["ring_radius"] == obs["platform"]["radius"] == 7.5
    assert obs["control_dt"] == 0.01
    assert obs["proximity_cutoff"] == SURFACE_CUTOFF == 2.0
    assert obs["proximity_limit"] == SURFACE_LIMIT == 32
