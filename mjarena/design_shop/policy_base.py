"""
Base policy compilation and execution utilities.

Moved from mjarena.agents.dspy_programs.verifiers.policy_base.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import collections
import math
from typing import Dict, Iterable, Mapping

import numpy as np

from mjarena.agents.types import BotObservation
from mjarena.envs.history import DEFAULT_HISTORY_LEN
from mjarena.utils.code_blocks import extract_code_block


# =============================================================================
# Safe Code Execution Environment
# =============================================================================

# Builtins the policy namespace does not provide. Calling one raises NameError
# mid-match, which costs the round, so the validator rejects them too — see
# FORBIDDEN_CALLS, which is derived from this one list.
_DENIED_BUILTINS = {
    'open', 'exec', 'eval', 'compile', 'breakpoint', 'exit', 'quit', 'input',
    'globals', 'locals', 'vars',
}
_ALLOWED_IMPORT_ROOTS = {'typing', 'numpy', 'math', 'random', 'collections'}


def _restricted_builtins() -> dict:
    """Create restricted builtins dictionary for safe policy execution.

    Uses a denylist approach: all builtins are allowed EXCEPT dangerous ones.
    This avoids the fragile allowlist pattern where missing a builtin (e.g.
    hasattr, getattr) crashes every policy that uses it.
    """
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Compare the package root, not a prefix: 'numpy_fake' is not numpy.
        # A relative import (level > 0) resolves against the policy's own
        # package and can reach anything, so it is never allowed.
        if level == 0 and name.split('.', 1)[0] in _ALLOWED_IMPORT_ROOTS:
            return _builtins.__import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Blocked import in policy: {name}")

    restricted = {k: v for k, v in vars(_builtins).items() if k not in _DENIED_BUILTINS}
    restricted['__import__'] = _safe_import
    return restricted


FORBIDDEN_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "requests", "urllib", "shutil", "psutil",
    "ctypes", "multiprocessing", "threading", "pexpect", "pty", "pickle", "dill",
    "importlib", "builtins", "signal", "asyncio", "http", "ftplib", "pathlib",
})
# Reject at validation exactly what the runtime namespace denies, plus the
# import hook. Failing with a line number beats a NameError mid-match: the
# model can fix the controller instead of forfeiting a qualification seed.
# getattr() is not here — it is ordinary Python and stays allowed.
FORBIDDEN_CALLS = frozenset(_DENIED_BUILTINS | {"__import__"})


def _validate_policy_code_safety(code: str, blacklist: Iterable[str] = ()) -> None:
    """Static safety check on the parsed code (comments and strings are not code).

    Names are compared as Python identifier paths, never as source substrings:
    'pty' is not 'empty', a local variable called `requests` is not the requests
    module, and a docstring that mentions os.system is prose. Imports are
    inspected wherever they appear — inside a function, in an unreachable branch
    — and import aliases are followed, so `from numpy.lib import os as platform`
    is rejected at the import and again at every use of `platform`.

    Args:
        code: Python source code to validate.
        blacklist: Extra forbidden identifiers, qualified names ("os.system"),
            or names written with a trailing "(" to mean "this name, called".

    Raises:
        RuntimeError: with the construct and its line number.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise RuntimeError(f"Policy code has syntax errors: {exc}") from exc

    blacklist = (tuple(blacklist) + tuple(FORBIDDEN_MODULES)
                 + tuple(f"{name}(" for name in FORBIDDEN_CALLS))
    forbidden = {tuple(bad.removesuffix("(").split(".")) for bad in blacklist}
    # Names that are dangerous even when merely referenced (aliasing `open` and
    # calling the alias is still calling open).
    forbidden_names = {bad[:-1] for bad in blacklist if bad.endswith("(")}
    forbidden_names |= {"__import__", "__builtins__"}
    # Do not let string-key lookup of the builtins table bypass name checks.
    forbidden.add(("__builtins__",))
    # ctypeslib is NumPy's bridge to ctypes, and _ctypes its extension module.
    if ("ctypes",) in forbidden:
        forbidden.update({("ctypeslib",), ("_ctypes",)})

    def reject(what: str, node: ast.AST) -> None:
        raise RuntimeError(
            f"Disallowed construct in policy: {what} (line {node.lineno})"
        )

    def check_path(path: tuple[str, ...], node: ast.AST, what: str = "") -> None:
        for banned in forbidden:
            if any(path[i:i + len(banned)] == banned
                   for i in range(len(path) - len(banned) + 1)):
                reject(what or ".".join(banned), node)

    # Inspect imports even inside functions or unreachable branches. Retain all
    # aliases conservatively so a later import cannot hide an earlier reference.
    aliases: dict[str, set[tuple[str, ...]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        from_import = isinstance(node, ast.ImportFrom)
        module = (node.module or "") if from_import else ""
        for item in node.names:
            imported = f"{module}.{item.name}" if module else item.name
            what = (f"import from '{node.module or '.'}'" if from_import
                    else f"import of '{item.name}'")
            check_path(tuple(imported.split(".")), node, what)
            root = (module or item.name).split(".", 1)[0]
            if (from_import and node.level) or root not in _ALLOWED_IMPORT_ROOTS:
                raise RuntimeError(
                    f"Blocked import in policy: {imported} (line {node.lineno})"
                )
            if isinstance(node, ast.Import) and not item.asname:
                local, path = root, (root,)
            else:
                local, path = item.asname or item.name, tuple(imported.split("."))
            aliases.setdefault(local, set()).add(path)

    def reference_paths(node: ast.AST) -> set[tuple[str, ...]]:
        if isinstance(node, ast.Name):
            return {(node.id,)} | aliases.get(node.id, set())
        if isinstance(node, ast.Attribute):
            bases = reference_paths(node.value)
            return {base + (node.attr,) for base in bases} or {(node.attr,)}
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            key = node.slice.value
            if key in ("__builtins__", "__import__"):
                return {(key,)}
            if (isinstance(key, str) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "__dict__"):
                bases = reference_paths(node.value.value)
                return {base + (key,) for base in bases} or {(key,)}
        # Literal getattr lookups are attribute access too, rather than ordinary
        # string data. This keeps getattr(obj, '__import__') from evading checks.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            attr = node.args[1].value
            bases = reference_paths(node.args[0])
            return {base + (attr,) for base in bases} or {(attr,)}
        return set()

    for node in ast.walk(tree):
        # ast.walk is breadth-first, so a call is seen before the name it calls:
        # the direct case gets the more precise "call to open()" wording, and
        # everything else (aliases, attributes) falls through to the paths below.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALLS):
            reject(f"call to {node.func.id}()", node)
        # A local variable called 'requests' is not a requests import. Dangerous
        # builtin references are still rejected, including assigning an alias.
        if isinstance(node, ast.Name) and node.id not in forbidden_names and node.id not in aliases:
            continue
        if isinstance(node, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
            for path in reference_paths(node):
                check_path(path, node)
        # The escape hatch to __builtins__ / __subclasses__ is a dunder
        # attribute; policies have no legitimate use for one.
        if (isinstance(node, ast.Attribute)
                and node.attr.startswith("__") and node.attr.endswith("__")):
            reject(f"dunder attribute access .{node.attr}", node)
        # NumPy exposes pickle through a keyword rather than a module reference.
        # Explicitly disabling it is safe; enabling it (including dynamically)
        # must not bypass the prohibition on pickle deserialization.
        if isinstance(node, ast.keyword) and node.arg == "allow_pickle" and ("pickle",) in forbidden:
            if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                reject("allow_pickle", node.value)


# =============================================================================
# Policy Compilation
# =============================================================================

def _strip_code_fences(code: str) -> str:
    """Return the one fenced block of controller source, or *code* unchanged.

    Raises:
        ValueError: the reply fences more than one block, or leaves one open.
    """
    return extract_code_block(code)


def compile_policy_function(code: str) -> callable:
    """
    Safely compile policy source code into executable function.

    Args:
        code: Python policy source code containing policy_step function

    Returns:
        Callable policy_step function

    Raises:
        RuntimeError: If code is unsafe or compilation fails
    """
    # Strip markdown code fences if present
    code = _strip_code_fences(code)

    # Security validation (AST-based; see _validate_policy_code_safety)
    _validate_policy_code_safety(code)

    # Create restricted execution environment
    # Pre-inject commonly used libraries so LLM-generated code works
    # even if it forgets to import them.
    namespace: Dict[str, object] = {
        "__name__": "__policy__",
        "__builtins__": _restricted_builtins(),
        "math": math,
        "np": np,
        "numpy": np,
        "collections": collections,
    }

    # Execute policy code
    try:
        exec(compile(code, "<policy>", "exec"), namespace, namespace)  # noqa: S102
    except Exception as exc:
        raise RuntimeError(f"Policy code execution failed: {exc}") from exc

    # Extract policy_step function
    policy_step = namespace.get("policy_step")
    if not callable(policy_step):
        raise RuntimeError("policy_step(obs) function not found in policy code")

    return policy_step


# =============================================================================
# Observation Utilities
# =============================================================================

def _empty_robot_state() -> Dict[str, object]:
    """Named robot record with no parts, for observations built without a robot."""
    return {
        "bodies": {},
        "geoms": {},
        "joints": {},
        "tendons": {},
        "motors": {},
        "sites": {},
        "mass": 0.0,
        "com_position": np.zeros(3),
        "com_velocity": np.zeros(3),
        "root_body": "",
    }


def create_dummy_observation_dict() -> Dict[str, object]:
    """
    Create dummy observation dictionary for policy testing.

    Returns:
        Dictionary observation compatible with policies (full 3D schema).
    """
    from mjarena.envs.detailed_observations import SURFACE_CUTOFF, SURFACE_LIMIT

    dummy_obs = BotObservation.get_dummy_bot_obs()

    # Dummy grids (41x41, cell_size=0.25)
    grid_size = 41
    dummy_int_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    dummy_float_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Convert to dict format matching every field in the prompt's obs_schema block
    return {
        # Position
        "my_pos": dummy_obs.my_pos,
        "opponent_pos": dummy_obs.opponent_pos,
        # Orientation
        "my_yaw": dummy_obs.my_yaw,
        "my_pitch": dummy_obs.my_pitch,
        "my_roll": dummy_obs.my_roll,
        "opponent_yaw": dummy_obs.opponent_yaw,
        "opponent_pitch": dummy_obs.opponent_pitch,
        "opponent_roll": dummy_obs.opponent_roll,
        # Velocity
        "my_velocity": dummy_obs.my_velocity,
        "my_angular_velocity": dummy_obs.my_angular_velocity,
        "opponent_velocity": dummy_obs.opponent_velocity,
        "opponent_angular_velocity": dummy_obs.opponent_angular_velocity,
        "my_actuator_velocity": dummy_obs.my_actuator_velocity,
        "opponent_actuator_velocity": dummy_obs.opponent_actuator_velocity,
        # Distances
        "distance_to_opponent": dummy_obs.distance_to_opponent,
        "my_edge_distance": dummy_obs.my_edge_distance,
        "opponent_edge_distance": dummy_obs.opponent_edge_distance,
        # Contact
        "opponent_contact": dummy_obs.opponent_contact,
        "opponent_contact_force": dummy_obs.opponent_contact_force,
        "ground_contact": dummy_obs.ground_contact,
        # Stability
        "is_tipping": dummy_obs.is_tipping,
        # Time
        "t": dummy_obs.t,
        "max_t": dummy_obs.max_t,
        # Robot properties
        "my_bounding_radius": dummy_obs.my_bounding_radius,
        "opponent_bounding_radius": dummy_obs.opponent_bounding_radius,
        "ring_radius": dummy_obs.ring_radius,
        # Spatial grids
        "arena_grid": dummy_int_grid.copy(),
        "arena_mass_grid": dummy_float_grid.copy(),
        "edge_distance_grid": dummy_float_grid.copy(),
        # Inactivity
        "my_inactivity_timer": dummy_obs.my_inactivity_timer,
        "opponent_inactivity_timer": dummy_obs.opponent_inactivity_timer,
        # Game context
        "game": dummy_obs.game,
        # Legacy aliases
        "my_heading": dummy_obs.my_yaw,
        "my_closest_distance_to_ring": dummy_obs.my_edge_distance,
        # History (empty at t=0)
        "obs_history": [],
        "action_history": [],
        # Timing (simulated seconds)
        "control_dt": 0.01,
        "elapsed_time": 0.0,
        "time_remaining": 0.0,
        # Platform the match is fought on
        "platform": {
            "shape": "cylinder",
            "center": np.zeros(3),
            "radius": 7.5,
            "half_extents": np.array([7.5, 7.5]),
            "top_z": 0.0,
            "floor_z": -2.0,
        },
        # Named physical state (empty without a composed robot)
        "my_robot": _empty_robot_state(),
        "opponent_robot": _empty_robot_state(),
        "my_mass": 0.0,
        "opponent_mass": 0.0,
        "my_com_velocity": np.zeros(3),
        "opponent_com_velocity": np.zeros(3),
        # Exact surface proximity to the opponent
        "opponent_surface_distance": 0.0,
        "opponent_proximity": [],
        "proximity_cutoff": SURFACE_CUTOFF,
        "proximity_limit": SURFACE_LIMIT,
        "proximity_truncated": False,
        # Contacts and their impulses over the last control interval
        "contacts": [],
        "contact_impulses": [],
        "contact_interval": 0.0,
    }


def observation_to_dict(obs) -> Dict[str, object]:
    """
    Convert observation object to dictionary format for policies.

    Args:
        obs: BotObservation object or existing dict

    Returns:
        Dictionary format observation
    """
    if hasattr(obs, "to_dict") and callable(getattr(obs, "to_dict")):
        return obs.to_dict()
    elif isinstance(obs, Mapping):
        return dict(obs)
    else:
        return obs


# =============================================================================
# Policy Wrapper Base Class
# =============================================================================

class BasePolicyWrapper:
    """
    Base class for policy wrappers providing common functionality.

    Handles safe policy execution, observation conversion, and error handling.
    """

    def __init__(self, policy_step_func: callable):
        """
        Initialize policy wrapper.

        Args:
            policy_step_func: Compiled policy_step function.
        """
        self.policy_step_func = policy_step_func

    def _prepare_observation(self, obs) -> Dict[str, object]:
        """
        Prepare observation for policy execution.

        Converts observation to dict format.
        """
        return observation_to_dict(obs)

    def execute_policy(self, obs) -> object:
        """
        Execute policy with observation and return raw result.

        Args:
            obs: Observation data.

        Returns:
            Raw policy output.

        Raises:
            RuntimeError: If policy execution fails.
        """
        try:
            prepared_obs = self._prepare_observation(obs)
            return self.policy_step_func(prepared_obs)
        except Exception as exc:
            raise RuntimeError(f"Policy execution failed: {exc}") from exc


__all__ = [
    "compile_policy_function",
    "create_dummy_observation_dict",
    "observation_to_dict",
    "BasePolicyWrapper",
]
