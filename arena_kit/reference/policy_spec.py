"""Serializable policy recipes for cross-process simulation workers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from mjarena.agents.policy_runtime import ActuatorPolicyAdapter
from mjarena.design_shop.policy_base import compile_policy_function


@dataclass(frozen=True)
class PolicySpec:
    """Recipe for reconstructing a policy callable in another process."""

    kind: str
    controller_code: str = ""
    actuator_names: tuple[str, ...] = ()
    file_path: str = ""
    function_name: str = "policy_step"
    constant_values: tuple[tuple[str, float], ...] = ()

    @classmethod
    def controller_code(cls, controller_code: str, actuator_names: Sequence[str]) -> "PolicySpec":
        return cls(
            kind="controller_code",
            controller_code=controller_code,
            actuator_names=tuple(actuator_names),
        )

    @classmethod
    def zero(cls, actuator_names: Sequence[str]) -> "PolicySpec":
        return cls(
            kind="zero",
            actuator_names=tuple(actuator_names),
        )

    @classmethod
    def constant(cls, actuator_names: Sequence[str], values: Mapping[str, float]) -> "PolicySpec":
        """Hold a fixed action every step; actuators absent from `values` stay at 0."""
        return cls(
            kind="constant",
            actuator_names=tuple(actuator_names),
            constant_values=tuple((str(k), float(v)) for k, v in values.items()),
        )

    @classmethod
    def python_file(
        cls,
        file_path: str | Path,
        *,
        function_name: str = "policy_step",
    ) -> "PolicySpec":
        return cls(
            kind="python_file",
            file_path=str(file_path),
            function_name=function_name,
        )

    def build_callable(self) -> Callable:
        if self.kind == "controller_code":
            policy_step = compile_policy_function(self.controller_code)
            return ActuatorPolicyAdapter(
                policy_step, list(self.actuator_names), controller_code=self.controller_code
            )
        if self.kind == "zero":
            names = tuple(self.actuator_names)

            def _policy(_obs):
                return {name: 0.0 for name in names}

            return _policy
        if self.kind == "constant":
            action = {name: 0.0 for name in self.actuator_names}
            action.update(dict(self.constant_values))

            def _policy(_obs):
                return dict(action)

            return _policy
        if self.kind == "python_file":
            policy_path = Path(self.file_path)
            ns: dict = {}
            exec(compile(policy_path.read_text(), str(policy_path), "exec"), ns)
            func = ns.get(self.function_name)
            if not callable(func):
                raise ValueError(
                    f"Policy function {self.function_name!r} not found in {policy_path}"
                )
            return func
        raise ValueError(f"Unsupported policy spec kind: {self.kind}")
