"""
Runtime policy building for DSPy programs - ACTUATOR format only.

Provides functions to compile controller code into executable policy callables.
"""
from __future__ import annotations

from typing import Sequence

from mjarena.agents.policy_runtime import ActuatorPolicyAdapter
from mjarena.design_shop.policy_base import compile_policy_function


def build_policy_callable(
    controller_code: str,
    actuator_names: Sequence[str],
):
    """
    Compile controller code into a callable matching runtime expectations.

    ACTUATOR format only - the policy returns dict[str, float] mapping
    actuator names to control values in [-1, 1].

    Args:
        controller_code: Python source code containing policy_step function.
        actuator_names: List of actuator names the policy must control.

    Returns:
        Callable that takes an observation and returns dict[str, float].
    """
    policy_step = compile_policy_function(controller_code)
    return ActuatorPolicyAdapter(policy_step, actuator_names, controller_code=controller_code)
