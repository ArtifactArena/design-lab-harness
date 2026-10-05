"""
Controller interfaces for arena agents.

ACTUATOR format only - controllers output dict[str, float] mapping actuator names to values.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def clip_actuator_actions(
    actions: Mapping[str, float], actuator_names: Sequence[str]
) -> dict[str, float]:
    """
    Validate actuator keys match exactly, then clip values to [-1, 1].

    Args:
        actions: Dict mapping actuator names to commanded values.
        actuator_names: Ground-truth actuator names from the hardware spec.

    Returns:
        Dict mapping actuator names to float values clipped to [-1, 1].

    Raises:
        ValueError: If the keys in *actions* don't match *actuator_names* exactly.
    """
    expected = set(actuator_names)
    provided = set(actions.keys())
    missing = expected - provided
    extra = provided - expected
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected: {sorted(extra)}")
        raise ValueError(
            f"Actuator key mismatch — {', '.join(parts)}"
        )
    return {n: float(np.clip(actions[n], -1.0, 1.0)) for n in actuator_names}


class ActuatorPolicyAdapter:
    """
    Wrap a policy_step callable to normalize actions for runtime usage.

    Takes a raw policy_step function and adapts it to return a dict
    mapping actuator names to normalized float values.
    """

    def __init__(self, policy_step, actuator_names: Sequence[str], *, controller_code: str):
        """
        Initialize the adapter.

        Args:
            policy_step: The raw policy function to wrap.
            actuator_names: List of actuator names in order.
            controller_code: Source used to create fresh state for each match.
        """
        self._policy_step = policy_step
        self._actuator_names = list(actuator_names)
        self._controller_code = controller_code

    def new_match(self) -> "ActuatorPolicyAdapter":
        """Re-execute source in a fresh namespace, leaving this policy untouched."""
        from mjarena.design_shop.policy_base import compile_policy_function

        return ActuatorPolicyAdapter(
            compile_policy_function(self._controller_code),
            self._actuator_names,
            controller_code=self._controller_code,
        )

    def __call__(self, obs) -> dict:
        """
        Execute policy and return normalized action dict.

        Args:
            obs: Observation dict or BotObservation.

        Returns:
            Dict mapping actuator names to float values in [-1, 1].
        """
        raw = self._policy_step(obs)
        return clip_actuator_actions(raw, self._actuator_names)

    @property
    def actuator_names(self) -> list:
        """Return the list of actuator names."""
        return list(self._actuator_names)
