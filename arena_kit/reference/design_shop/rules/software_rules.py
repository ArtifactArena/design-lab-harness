"""Software rules — controller static checks (pass/fail gates).

Compile, execute, and validate action output format.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from mjarena.design_shop.types import VerifierResult
from mjarena.design_shop.policy_base import (
    compile_policy_function,
    create_dummy_observation_dict,
)


def _describe_compile_error(exc: Exception, code: str) -> str:
    """'invalid syntax (<unknown>, line 16)' -> the message plus the offending source line."""
    err = exc
    while err is not None and not isinstance(err, SyntaxError):
        err = err.__cause__ or err.__context__
    if isinstance(err, SyntaxError) and err.lineno:
        lines = code.splitlines()
        src = (err.text or (lines[err.lineno - 1] if 0 < err.lineno <= len(lines) else "")).rstrip()
        return f"{err.msg} at line {err.lineno} of controller_code:\n    {src.strip()}"
    return str(exc)


def compile_policy(code: str) -> tuple[Optional[Callable], VerifierResult]:
    """Compile policy code into a callable."""
    t0 = time.time()
    try:
        policy_step = compile_policy_function(code)
        return policy_step, VerifierResult(
            passed=True,
            label="Compile",
            message="Policy compiled successfully.",
            name="verifier:compile",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return None, VerifierResult(
            passed=False,
            label="Compile",
            message=_describe_compile_error(exc, code),
            name="verifier:compile",
            elapsed_sec=time.time() - t0,
        )


def _build_execution_cases(obs_schema: Optional[Mapping] = None, base_observation: Optional[Mapping] = None) -> list[tuple[str, Dict[str, object]]]:
    """Create representative observations to exercise common controller branches.

    The default dummy observation only covers one execution path. Many generated
    policies branch on edge distance, opponent range, or inactivity. Running a
    small fixed suite here catches branch-local runtime bugs before qualification.
    """
    def _base_case() -> Dict[str, object]:
        obs = deepcopy(base_observation) if base_observation is not None else create_dummy_observation_dict()
        if obs_schema:
            for key in obs_schema:
                if key not in obs:
                    obs[key] = None
        return obs

    cases: list[tuple[str, Dict[str, object]]] = []

    default = _base_case()
    cases.append(("default", default))
    if base_observation is not None:
        # Detailed geometry and summaries must describe the same physical state.
        # Real match rollouts exercise subsequent states.
        return cases

    edge_recovery = _base_case()
    edge_recovery["my_pos"] = np.array([4.7, 0.0, 0.0], dtype=np.float32)
    edge_recovery["opponent_pos"] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    edge_recovery["my_velocity"] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    edge_recovery["distance_to_opponent"] = 5.7
    edge_recovery["my_edge_distance"] = 1.0
    edge_recovery["opponent_edge_distance"] = 4.0
    cases.append(("edge_recovery", edge_recovery))

    engage_contact = _base_case()
    engage_contact["my_pos"] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    engage_contact["opponent_pos"] = np.array([0.8, 0.0, 0.0], dtype=np.float32)
    engage_contact["distance_to_opponent"] = 0.8
    engage_contact["opponent_contact"] = True
    cases.append(("engage_contact", engage_contact))

    hold_center_far = _base_case()
    hold_center_far["my_pos"] = np.array([2.5, 0.0, 0.0], dtype=np.float32)
    hold_center_far["opponent_pos"] = np.array([6.0, 0.0, 0.0], dtype=np.float32)
    hold_center_far["distance_to_opponent"] = 3.5
    hold_center_far["my_edge_distance"] = 2.5
    cases.append(("hold_center_far", hold_center_far))

    anti_inactivity = _base_case()
    anti_inactivity["my_inactivity_timer"] = 5.0
    anti_inactivity["t"] = 10.0
    cases.append(("anti_inactivity", anti_inactivity))

    return cases


def execute_policy(
    policy_step: Callable,
    obs_schema: Optional[Mapping] = None,
    base_observation: Optional[Mapping] = None,
) -> tuple[Optional[object], VerifierResult]:
    """Execute policy with dummy observation."""
    t0 = time.time()
    try:
        dummy_obs = deepcopy(base_observation) if base_observation is not None else create_dummy_observation_dict()
        if obs_schema:
            for key in obs_schema:
                if key not in dummy_obs:
                    dummy_obs[key] = None
        raw_action = policy_step(dummy_obs)
        return raw_action, VerifierResult(
            passed=True,
            label="Execute",
            message="policy_step(obs) executed successfully.",
            name="verifier:execute",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return None, VerifierResult(
            passed=False,
            label="Execute",
            message=f"policy_step(obs) raised: {exc}",
            name="verifier:execute",
            elapsed_sec=time.time() - t0,
        )


def _coerce_action_dict(raw_action: object, actuator_names: Sequence[str]) -> Dict[str, float]:
    """Normalize a controller action into a validated actuator dict."""
    n_actuators = len(actuator_names)

    if isinstance(raw_action, Mapping):
        action_dict = dict(raw_action)
    elif hasattr(raw_action, "__len__") or hasattr(raw_action, "__iter__"):
        action_array = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if len(action_array) != n_actuators:
            raise RuntimeError(
                f"Returned array length {len(action_array)}, expected {n_actuators} for: {list(actuator_names)}"
            )
        action_dict = {name: float(action_array[i]) for i, name in enumerate(actuator_names)}
    else:
        raise RuntimeError(
            f"Must return dict or array, got {type(raw_action).__name__}. Expected keys: {list(actuator_names)}"
        )

    missing = [name for name in actuator_names if name not in action_dict]
    if missing:
        raise RuntimeError(
            f"Missing actuator keys: {missing}. Must include all: {list(actuator_names)}"
        )

    return action_dict


def validate_actions(
    raw_action: object,
    actuator_names: Sequence[str],
) -> tuple[Optional[Dict[str, float]], VerifierResult]:
    """Validate action output format and actuator keys."""
    t0 = time.time()
    n_actuators = len(actuator_names)

    try:
        action_dict = _coerce_action_dict(raw_action, actuator_names)

        return action_dict, VerifierResult(
            passed=True,
            label="Validate Actions",
            message=f"All {n_actuators} actuators present and valid.",
            name="verifier:validate_actions",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return None, VerifierResult(
            passed=False,
            label="Validate Actions",
            message=f"Output validation failed: {exc}",
            name="verifier:validate_actions",
            elapsed_sec=time.time() - t0,
        )


def exercise_policy(
    policy_step: Callable,
    actuator_names: Sequence[str],
    obs_schema: Optional[Mapping] = None,
    base_observation: Optional[Mapping] = None,
) -> VerifierResult:
    """Run a small suite of representative observations through the policy."""
    t0 = time.time()
    case_name = "unknown"
    ran: list[str] = []
    try:
        for case_name, obs in _build_execution_cases(obs_schema, base_observation):
            raw_action = policy_step(obs)
            _coerce_action_dict(raw_action, actuator_names)
            ran.append(case_name)
        # Say which cases actually ran: on the real-observation path only the default
        # case exists, and a message claiming branch coverage the model never got is
        # feedback it will reason from.
        if ran == ["default"]:
            message = "policy_step(obs) executed successfully on the initial observation of your robot."
        else:
            message = ("policy_step(obs) executed successfully on representative cases: "
                       + ", ".join(ran) + ".")
        return VerifierResult(
            passed=True,
            label="Exercise Branches",
            message=message,
            name="verifier:exercise",
            elapsed_sec=time.time() - t0,
        )
    except Exception as exc:
        return VerifierResult(
            passed=False,
            label="Exercise Branches",
            message=f"policy_step(obs) raised on representative case '{case_name}': {exc}",
            name="verifier:exercise",
            elapsed_sec=time.time() - t0,
        )
