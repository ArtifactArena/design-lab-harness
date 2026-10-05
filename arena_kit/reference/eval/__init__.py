"""
Evaluation utilities for arena bots.
"""
from __future__ import annotations


def __getattr__(name: str):
    if name == "build_policy_callable":
        from .runtime import build_policy_callable
        return build_policy_callable
    if name in ("make_reward_fn", "arena_metric"):
        from .metrics import make_reward_fn, arena_metric
        return {"make_reward_fn": make_reward_fn, "arena_metric": arena_metric}[name]
    if name in ("run_seeds", "save_matchup_to_disk"):
        from .match_runner import run_seeds, save_matchup_to_disk
        return {"run_seeds": run_seeds, "save_matchup_to_disk": save_matchup_to_disk}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "build_policy_callable",
    "make_reward_fn",
    "run_seeds",
    "save_matchup_to_disk",
    "arena_metric",
]
