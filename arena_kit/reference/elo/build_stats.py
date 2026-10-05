"""
Load generation timing and cost data from bot artifacts.

Walks ``bot_artifact.json`` files in a season directory and aggregates
morphology/controller generation time, LLM call cost, and reasoning token
usage per model.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _extract_stem(bot_name: str) -> str:
    """Extract the generator stem from a full bot name.

    Bot name format: ``season_XX.tournament_name.generator_stem``
    Generator stems may contain dots (e.g. ``gpt-5.2-none``).
    """
    parts = bot_name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[2:])
    return bot_name


def _sum_call_costs(debug_path: Path) -> float:
    """Sum ``calls[].cost`` from a morphology/controller debug JSON."""
    if not debug_path.exists():
        return 0.0
    try:
        with debug_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return sum(
            float(call.get("cost", 0))
            for call in data.get("calls", [])
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("Could not parse costs from %s: %s", debug_path, exc)
        return 0.0


def _extract_num_calls(debug_path: Path) -> int:
    """Read ``num_calls`` from a morphology/controller debug JSON."""
    if not debug_path.exists():
        return 0
    try:
        with debug_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("num_calls", 0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.debug("Could not parse num_calls from %s: %s", debug_path, exc)
        return 0


def _sum_reasoning_tokens(debug_path: Path) -> int:
    """Sum reasoning tokens from a debug JSON.

    Reads ``calls[].usage.completion_tokens_details.reasoning_tokens``
    from each LLM call and returns the total.
    """
    if not debug_path.exists():
        return 0
    try:
        with debug_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        total = 0
        for call in data.get("calls", []):
            usage = call.get("usage", {})
            details = usage.get("completion_tokens_details", {})
            total += int(details.get("reasoning_tokens", 0))
        return total
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.debug("Could not parse reasoning tokens from %s: %s", debug_path, exc)
        return 0


def load_build_stats(season_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Walk bot_artifact.json files and aggregate timing/cost per model.

    Returns::

        {generator_stem: {
            # Per-phase averages (backward compat)
            "morphology_time_avg": float,    # seconds
            "controller_time_avg": float,
            "morphology_cost_avg": float,    # USD
            "controller_cost_avg": float,
            "n_bots": int,
            # Per-tournament lists (morphology + controller summed)
            "costs": [float, ...],           # one total cost per tournament
            "reasoning_tokens": [int, ...],  # one total per tournament
            # Overall averages
            "cost_avg": float,               # mean of costs list
            "reasoning_tokens_avg": float,   # mean of reasoning_tokens list
        }}
    """
    season_dir = Path(season_dir)

    # Accumulators: stem -> list of values (one entry per bot/tournament)
    morph_times: Dict[str, List[float]] = defaultdict(list)
    ctrl_times: Dict[str, List[float]] = defaultdict(list)
    morph_costs: Dict[str, List[float]] = defaultdict(list)
    ctrl_costs: Dict[str, List[float]] = defaultdict(list)
    morph_reasoning: Dict[str, List[int]] = defaultdict(list)
    ctrl_reasoning: Dict[str, List[int]] = defaultdict(list)
    morph_num_calls: Dict[str, List[int]] = defaultdict(list)
    ctrl_num_calls: Dict[str, List[int]] = defaultdict(list)

    for artifact_path in sorted(season_dir.rglob("bot_artifact.json")):
        try:
            with artifact_path.open("r", encoding="utf-8") as f:
                artifact = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping %s: %s", artifact_path, exc)
            continue

        # Skip forfeits — no meaningful timing/cost data
        if artifact.get("forfeit", False):
            continue

        bot_name = artifact.get("name", "")
        stem = artifact.get("generator", "") or _extract_stem(bot_name)
        if not stem:
            continue

        metadata = artifact.get("metadata", {})

        # Timing
        mt = metadata.get("morphology_generation_time_sec")
        if mt is not None:
            morph_times[stem].append(float(mt))

        ct = metadata.get("controller_generation_time_sec")
        if ct is not None:
            ctrl_times[stem].append(float(ct))

        # Cost + reasoning tokens from debug JSONs (sibling debug/ directory)
        bot_dir = artifact_path.parent
        debug_dir = bot_dir / "debug"

        morph_debug = debug_dir / "morphology_debug.json"
        ctrl_debug = debug_dir / "controller_debug.json"

        mc = _sum_call_costs(morph_debug)
        if mc > 0:
            morph_costs[stem].append(mc)

        cc = _sum_call_costs(ctrl_debug)
        if cc > 0:
            ctrl_costs[stem].append(cc)

        mr = _sum_reasoning_tokens(morph_debug)
        morph_reasoning[stem].append(mr)

        cr = _sum_reasoning_tokens(ctrl_debug)
        ctrl_reasoning[stem].append(cr)

        mnc = _extract_num_calls(morph_debug)
        morph_num_calls[stem].append(mnc)

        cnc = _extract_num_calls(ctrl_debug)
        ctrl_num_calls[stem].append(cnc)

    # Aggregate
    all_stems = (
        set(morph_times) | set(ctrl_times)
        | set(morph_costs) | set(ctrl_costs)
        | set(morph_reasoning) | set(ctrl_reasoning)
        | set(morph_num_calls) | set(ctrl_num_calls)
    )
    result: Dict[str, Dict[str, Any]] = {}

    for stem in sorted(all_stems):
        mt_list = morph_times.get(stem, [])
        ct_list = ctrl_times.get(stem, [])
        mc_list = morph_costs.get(stem, [])
        cc_list = ctrl_costs.get(stem, [])
        mr_list = morph_reasoning.get(stem, [])
        cr_list = ctrl_reasoning.get(stem, [])
        mnc_list = morph_num_calls.get(stem, [])
        cnc_list = ctrl_num_calls.get(stem, [])

        # Per-tournament total costs (morph + ctrl summed per tournament)
        # Zip the two lists; if lengths differ, use min length
        n_cost = min(len(mc_list), len(cc_list)) if mc_list and cc_list else max(len(mc_list), len(cc_list))
        costs: List[float] = []
        for j in range(n_cost):
            m = mc_list[j] if j < len(mc_list) else 0.0
            c = cc_list[j] if j < len(cc_list) else 0.0
            costs.append(m + c)

        # Per-tournament total reasoning tokens
        n_reason = max(len(mr_list), len(cr_list))
        reasoning: List[int] = []
        for j in range(n_reason):
            m = mr_list[j] if j < len(mr_list) else 0
            c = cr_list[j] if j < len(cr_list) else 0
            reasoning.append(m + c)

        result[stem] = {
            # Backward-compat per-phase averages
            "morphology_time_avg": sum(mt_list) / len(mt_list) if mt_list else 0.0,
            "controller_time_avg": sum(ct_list) / len(ct_list) if ct_list else 0.0,
            "morphology_cost_avg": sum(mc_list) / len(mc_list) if mc_list else 0.0,
            "controller_cost_avg": sum(cc_list) / len(cc_list) if cc_list else 0.0,
            "n_bots": max(len(mt_list), len(ct_list), 1),
            # Per-tournament lists
            "costs": costs,
            "reasoning_tokens": reasoning,
            # Overall averages
            "cost_avg": sum(costs) / len(costs) if costs else 0.0,
            "reasoning_tokens_avg": sum(reasoning) / len(reasoning) if reasoning else 0.0,
            # LLM call counts
            "morphology_num_calls_avg": sum(mnc_list) / len(mnc_list) if mnc_list else 0.0,
            "controller_num_calls_avg": sum(cnc_list) / len(cnc_list) if cnc_list else 0.0,
        }

    logger.info(
        "Loaded build stats for %d models from %s",
        len(result), season_dir,
    )
    return result


def save_build_costs(season_dir: Path, stats: Dict[str, Dict]) -> Path:
    """Save ``build_costs.json`` with per-tournament cost and reasoning token lists.

    Output structure::

        {
            "costs": {stem: [float, ...]},
            "costs_avg": {stem: float},
            "reasoning_tokens": {stem: [int, ...]},
            "reasoning_tokens_avg": {stem: float},
        }

    Args:
        season_dir: Season output directory.
        stats: Return value of :func:`load_build_stats`.

    Returns:
        Path to the written JSON file.
    """
    out: Dict[str, Dict[str, Any]] = {
        "costs": {},
        "costs_avg": {},
        "reasoning_tokens": {},
        "reasoning_tokens_avg": {},
    }
    for stem, s in stats.items():
        out["costs"][stem] = s.get("costs", [])
        out["costs_avg"][stem] = s.get("cost_avg", 0.0)
        out["reasoning_tokens"][stem] = s.get("reasoning_tokens", [])
        out["reasoning_tokens_avg"][stem] = s.get("reasoning_tokens_avg", 0)

    path = season_dir / "elo" / "build_costs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.info("Saved build costs to %s", path)
    return path
