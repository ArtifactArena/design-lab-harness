"""
Round-robin tournament runner for bot competitions.

Uses existing infrastructure:
- create_match_runner() from dspy_core for match execution
- run_seeds() from eval/match_runner for multi-seed evaluation

Runs matches between all pairs of bots and tracks results.
"""
from __future__ import annotations

import datetime
import json
import logging
import shutil
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import combinations
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from mjarena.agents.types import BotArtifact
from mjarena.dspy_core import create_match_runner
from mjarena.eval.match_runner import run_seeds, save_matchup_to_disk
from mjarena.eval.runtime import build_policy_callable
from mjarena.envs.sumo import compose_sumo_model
from mjarena.runner.recording import VideoOverlayInfo, concatenate_seed_videos
from mjarena.utils.file import ensure_dir
from mjarena.elo.core import MatchOutcome, compute_standings, elo_rank_players
from mjarena.elo.display import _get_display_name
from .types import MatchResult, MatchupResult, TournamentResult

logger = logging.getLogger(__name__)


def _trace(enabled: bool, message: str) -> None:
    """Emit a timestamped progress line immediately."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    thread_name = threading.current_thread().name
    print(f"[{timestamp}] [{thread_name}] {message}", flush=True)


def _format_elapsed(seconds: Optional[float]) -> str:
    """Format seconds as a short human-readable duration."""
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:5.1f}s"
    minutes, secs = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes):02d}:{secs:04.1f}"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:d}:{minutes:02d}:{int(secs):02d}"


def _progress_bar(done: int, total: int, *, width: int = 28) -> str:
    """Render a simple ASCII progress bar."""
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled = int(round(width * done / total))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _truncate(text: str, width: int) -> str:
    """Truncate text to a fixed width for table output."""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[:width - 3] + "..."


def _render_dashboard(
    task_states: Dict[int, Dict[str, Any]],
    *,
    done_count: int,
    total_count: int,
    workers: int,
    tournament_elapsed: float,
) -> str:
    """Build a snapshot table showing queued/running/completed matchups."""
    now = time.monotonic()
    running_count = sum(1 for state in task_states.values() if state["status"] == "running")
    queued_count = sum(1 for state in task_states.values() if state["status"] == "queued")
    completed_count = sum(1 for state in task_states.values() if state["status"] == "completed")
    error_count = sum(1 for state in task_states.values() if state["status"] == "error")
    # "skipped" status appears when run_round_robin's resumability check finds
    # an existing match_result.json on disk (kill+restart resume from previous run).
    skipped_count = sum(1 for state in task_states.values() if state["status"] == "skipped")

    lines = [
        f"[Dashboard] {_progress_bar(done_count, total_count)} {done_count}/{total_count} complete "
        f"| running={running_count} queued={queued_count} completed={completed_count} "
        f"errors={error_count} skipped={skipped_count} "
        f"| workers={workers} | elapsed={_format_elapsed(tournament_elapsed)}"
    ]

    headers = ("Idx", "Matchup", "Seeds", "Status", "Time", "Result")
    rows: List[Tuple[str, str, str, str, str, str]] = []
    # "skipped" sorts last so the table foregrounds active work.
    status_order = {"running": 0, "queued": 1, "completed": 2, "error": 3, "skipped": 4}

    for idx, state in sorted(task_states.items(), key=lambda item: (status_order[item[1]["status"]], item[0])):
        status = state["status"]
        if status == "running":
            time_str = _format_elapsed(now - state["started_at"]) if state["started_at"] else "--"
            result_str = "--"
        elif status == "queued":
            time_str = _format_elapsed(now - state["submitted_at"])
            result_str = "--"
        else:
            time_str = _format_elapsed(state.get("duration"))
            result_str = state.get("result", "ERROR" if status == "error" else "--")

        rows.append((
            str(idx + 1),
            _truncate(state["label"], 42),
            state["seed_range"],
            status,
            time_str,
            _truncate(result_str, 18),
        ))

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))
    lines.append(header_line)
    lines.append(separator)
    lines.extend(
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def run_matchup(
    *,
    bot_a: BotArtifact,
    bot_b: BotArtifact,
    arena_xml: Path,
    out_dir: Path,
    n_seeds: int = 5,
    seeds: Optional[List[int]] = None,
    max_steps: int = 1000,
    save_video_seeds: int = 0,
    video_width: int = 640,
    video_height: int = 480,
    quiet: bool = True,
    use_material_palette: bool = False,
    season_id: str = "",
    tournament_id: str = "",
    camera_mode: str = "tracking",
    score_function: str = "any",
    physics_mode: str = "3d",
    n_parallel_seeds: int = 1,
    spawn_seed: int | None = None,
    gear_clamp_ratio: float = 0.0,
    match_time: Optional[float] = None,
    inactivity_timeout_seconds: Optional[float] = 10.0,
    inactivity_min_displacement: float = 0.5,
    inactivity_exempt_prefixes: Optional[List[str]] = None,
    size_limits: Optional[Tuple[float, float, float]] = None,
    contact_fidelity: str = "high",
    trace_progress: bool = False,
    rendering_flags: Optional[Dict[str, bool]] = None,
) -> MatchupResult:
    """Run multiple matches between two bots.

    Creates ONE composed XML. Position fairness is handled by SumoEnv.reset()
    which randomly assigns starting positions based on seed.

    Per-step obs/action logs are always collected for every match.

    Args:
        bot_a: First bot artifact (always red)
        bot_b: Second bot artifact (always blue)
        arena_xml: Path to arena XML
        out_dir: Output directory for this matchup
        n_seeds: Number of random seeds to run
        seeds: Explicit seed list (overrides n_seeds range).
        max_steps: Maximum simulation steps per match
        save_video_seeds: Number of seeds to save video for (0 = no videos)
        video_width: Video frame width
        video_height: Video frame height
        quiet: Suppress per-match output
        inactivity_exempt_prefixes: Robot prefixes exempt from inactivity timeout.

    Returns:
        MatchupResult with all match results and aggregated stats
    """
    ensure_dir(out_dir)
    all_matches: List[MatchResult] = []
    wins_a = 0
    wins_b = 0
    ties = 0
    matchup_label = f"{bot_a.name} vs {bot_b.name}"

    # Create ONE composed XML (bot_a=red, bot_b=blue)
    composed_xml = out_dir / "composed.xml"
    _trace(trace_progress, f"{matchup_label}: compose_sumo_model start")
    compose_sumo_model(
        env_xml=str(arena_xml),
        robot_red_xml=str(bot_a.morphology_xml),
        robot_blue_xml=str(bot_b.morphology_xml),
        out_path=str(composed_xml),
        use_material_palette=use_material_palette,
        randomize_spawn_3d=(physics_mode == "3d"),
        spawn_seed=spawn_seed,
    )
    _trace(trace_progress, f"{matchup_label}: compose_sumo_model done -> {composed_xml}")

    # Read controller source code (needed for both building and parallel rebuild)
    _trace(trace_progress, f"{matchup_label}: loading controller sources")
    red_code = bot_a.controller_code.read_text()
    blue_code = bot_b.controller_code.read_text()

    # Build policy callables
    _trace(trace_progress, f"{matchup_label}: building policy callables")
    policy_a = build_policy_callable(
        controller_code=red_code,
        actuator_names=bot_a.actuator_names,
    )
    policy_b = build_policy_callable(
        controller_code=blue_code,
        actuator_names=bot_b.actuator_names,
    )

    # Build overlay base for video text overlays
    overlay_base = {
        "red_name": _get_display_name(bot_a.name),
        "blue_name": _get_display_name(bot_b.name),
        "tournament_id": tournament_id,
        "season_id": season_id,
    } if save_video_seeds > 0 else None

    # When running seeds in parallel, rebuild policy callables per call
    # to avoid shared mutable state in controller closures (e.g., prev_pos).
    use_rebuild = n_parallel_seeds > 1

    # Create match runner
    match_runner_fn = create_match_runner(
        composed_xml=composed_xml,
        blue_policy_callable=policy_b,
        out_dir=out_dir,
        max_steps=max_steps,
        save_video_seeds=save_video_seeds,
        quiet=quiet,
        video_width=video_width,
        video_height=video_height,
        overlay_base=overlay_base,
        camera_mode=camera_mode,
        score_function=score_function,
        rebuild_policies=use_rebuild,
        red_controller_code=red_code if use_rebuild else None,
        red_actuator_names=list(bot_a.actuator_names) if use_rebuild else None,
        blue_controller_code=blue_code if use_rebuild else None,
        blue_actuator_names=list(bot_b.actuator_names) if use_rebuild else None,
        gear_clamp_ratio=gear_clamp_ratio,
        match_time=match_time,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        inactivity_min_displacement=inactivity_min_displacement,
        inactivity_exempt_prefixes=inactivity_exempt_prefixes,
        size_limits=size_limits,
        contact_fidelity=contact_fidelity,
        trace_label=matchup_label,
        rendering_flags=rendering_flags,
    )

    # Resolve seeds
    actual_seeds = seeds if seeds is not None else list(range(n_seeds))
    _trace(
        trace_progress,
        f"{matchup_label}: run_seeds start ({len(actual_seeds)} seeds, n_parallel_seeds={n_parallel_seeds}, "
        f"save_video_seeds={save_video_seeds})"
    )

    # Run all seeds via run_seeds() — returns typed MatchupResult from design_shop.types
    from mjarena.design_shop.types import MatchupResult as DSMatchupResult
    seeds_t0 = time.monotonic()
    ds_matchup: DSMatchupResult = run_seeds(
        run_match_fn=match_runner_fn,
        policy_callable=policy_a,
        n_seeds=n_seeds,
        seeds=actual_seeds,
        n_parallel=n_parallel_seeds,
        trace_progress=trace_progress,
        progress_label=matchup_label,
    )
    _trace(
        trace_progress,
        f"{matchup_label}: run_seeds done in {time.monotonic() - seeds_t0:.1f}s"
    )

    # Concatenate per-seed videos into a single match video
    if save_video_seeds > 0:
        _trace(trace_progress, f"{matchup_label}: concatenate_seed_videos start")
        seed_videos = [out_dir / f"seed_{s}.webm" for s in actual_seeds[:save_video_seeds]]
        overlay_infos = [
            VideoOverlayInfo(
                seed=s,
                red_name=_get_display_name(bot_a.name),
                blue_name=_get_display_name(bot_b.name),
                tournament_id=tournament_id,
                season_id=season_id,
            )
            for s in actual_seeds[:save_video_seeds]
        ]
        match_video_path = out_dir / "match.webm"
        concatenate_seed_videos(
            seed_videos=seed_videos,
            output_path=match_video_path,
            overlay_infos=overlay_infos,
            delete_originals=False,
        )
        if match_video_path.exists():
            logger.info("Saved match video: %s", match_video_path)
        _trace(trace_progress, f"{matchup_label}: concatenate_seed_videos done -> {match_video_path}")

    # Convert GameRecords from ds_matchup into tournament MatchResult objects
    for seed_idx, record in enumerate(ds_matchup.game_records):
        winner = record.winner
        match = MatchResult(
            red_bot=bot_a.name,
            blue_bot=bot_b.name,
            winner=winner,
            red_score=1.0 if winner == "red" else (-1.0 if winner == "blue" else 0.0),
            seed=actual_seeds[seed_idx] if seed_idx < len(actual_seeds) else seed_idx,
            steps=record.winner_step or record.num_steps,
            details={"combat_metrics": record.combat_metrics or {}},
        )
        all_matches.append(match)

        # Track wins (red=bot_a, blue=bot_b)
        if winner == "red":
            wins_a += 1
        elif winner == "blue":
            wins_b += 1
        else:
            ties += 1

    return MatchupResult(
        bot_a=bot_a.name,
        bot_b=bot_b.name,
        matches=all_matches,
        _ds_matchup=ds_matchup,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
    )


def _safe_json(val: Any) -> Any:
    """Make a value JSON-serializable (convert numpy arrays, etc.)."""
    if val is None:
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, dict):
        return {str(k): _safe_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_safe_json(v) for v in val]
    # numpy scalar / array
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.tolist()
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
    except ImportError:
        pass
    return str(val)


def _relative_path(path: Path, root: Path) -> Optional[str]:
    """Return path relative to root, or None."""
    try:
        return str(path.relative_to(root))
    except (ValueError, TypeError):
        return None


def _is_stationary_bot(bot: "BotArtifact") -> bool:
    """Check if a bot is stationary (no actuators) and should be exempt from inactivity timeout."""
    # A bot with no actuators cannot move — exempt from inactivity
    if not bot.actuator_names:
        return True
    return False


def _run_single_matchup(
    idx: int,
    name_a: str,
    name_b: str,
    bots: Dict[str, BotArtifact],
    bot_name_to_id: Dict[str, int],
    arena_xml: Path,
    matches_dir: Path,
    n_rollouts: int,
    seed_base: int,
    max_steps: int,
    save_video_seeds: int,
    video_width: int,
    video_height: int,
    use_material_palette: bool,
    season_id: str,
    tournament_id: str,
    camera_mode: str,
    score_function: str,
    physics_mode: str,
    iteration: int,
    season_root: Optional[Path],
    out_dir: Path,
    n_parallel_seeds: int = 1,
    gear_clamp_ratio: float = 0.0,
    match_time: Optional[float] = None,
    inactivity_timeout_seconds: Optional[float] = 10.0,
    inactivity_min_displacement: float = 0.5,
    size_limits: Optional[Tuple[float, float, float]] = None,
    contact_fidelity: str = "high",
    trace_progress: bool = False,
    rendering_flags: Optional[Dict[str, bool]] = None,
) -> Tuple[int, MatchupResult, Dict[str, Any]]:
    """Run a single matchup (bot pair), build enriched JSON, return results.

    Args:
        idx: Matchup index (for deterministic seed computation and ordering).
        name_a, name_b: Bot names.
        bots: Full bots dict (read-only).
        bot_name_to_id: Mapping from bot name to numeric id.
        arena_xml: Path to arena XML.
        matches_dir: Parent directory for matchup subdirectories.
        n_rollouts: Number of seeds per matchup.
        seed_base: Base seed for this tournament.
        max_steps: Max sim steps.
        save_video_seeds: How many seeds get video.
        video_width, video_height: Video resolution.
        use_material_palette: Whether to use material palette.
        season_id, tournament_id: IDs for overlay text.
        camera_mode, score_function: Match settings.
        physics_mode: "2d" or "3d".
        iteration: Current season iteration.
        season_root: Root output dir of the season (for relative paths).
        out_dir: Tournament output directory.
        n_parallel_seeds: Seeds to run in parallel within this matchup.

    Returns:
        Tuple of (idx, matchup_result, match_result_data) for assembly.
    """
    threading.current_thread().name = f"{name_a}_vs_{name_b}"

    # Use generator names for human-readable folder names (lower ID first for determinism)
    id_a, id_b = bot_name_to_id[name_a], bot_name_to_id[name_b]
    gen_a = bots[name_a].generator
    gen_b = bots[name_b].generator
    if id_a > id_b:
        id_a, id_b = id_b, id_a
        gen_a, gen_b = gen_b, gen_a
    matchup_dir = matches_dir / f"{gen_a}_vs_{gen_b}"

    # Compute seeds for this matchup
    matchup_seed_start = seed_base + idx * n_rollouts
    matchup_seeds = [matchup_seed_start + s for s in range(n_rollouts)]

    # Determine inactivity exemptions for stationary bots (e.g. box baseline)
    exempt_prefixes: Optional[List[str]] = None
    bot_a_obj = bots[name_a]
    bot_b_obj = bots[name_b]
    _exempts = []
    if _is_stationary_bot(bot_a_obj):
        _exempts.append("red_")
    if _is_stationary_bot(bot_b_obj):
        _exempts.append("blue_")
    if _exempts:
        exempt_prefixes = _exempts

    _trace(
        trace_progress,
        f"[matchup {idx + 1}] {name_a} vs {name_b}: start "
        f"(seeds={matchup_seeds[0]}..{matchup_seeds[-1]}, out_dir={matchup_dir})"
    )

    matchup_result = run_matchup(
        bot_a=bot_a_obj,
        bot_b=bot_b_obj,
        arena_xml=arena_xml,
        out_dir=matchup_dir,
        n_seeds=n_rollouts,
        seeds=matchup_seeds,
        max_steps=max_steps,
        save_video_seeds=save_video_seeds,
        video_width=video_width,
        video_height=video_height,
        quiet=True,
        use_material_palette=use_material_palette,
        season_id=season_id,
        tournament_id=tournament_id,
        camera_mode=camera_mode,
        score_function=score_function,
        physics_mode=physics_mode,
        n_parallel_seeds=n_parallel_seeds,
        spawn_seed=matchup_seed_start,
        gear_clamp_ratio=gear_clamp_ratio,
        match_time=match_time,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        inactivity_min_displacement=inactivity_min_displacement,
        inactivity_exempt_prefixes=exempt_prefixes,
        size_limits=size_limits,
        contact_fidelity=contact_fidelity,
        trace_progress=trace_progress,
        rendering_flags=rendering_flags,
    )
    _trace(
        trace_progress,
        f"[matchup {idx + 1}] {name_a} vs {name_b}: match execution done "
        f"({matchup_result.wins_a}-{matchup_result.wins_b}-{matchup_result.ties})"
    )

    # Build enriched fields for match_result.json
    path_root = season_root or out_dir.parent

    bot_a_dir = bot_a_obj.morphology_xml.parent
    bot_b_dir = bot_b_obj.morphology_xml.parent

    red_artifact = _relative_path(bot_a_dir / "bot_artifact.json", path_root)
    blue_artifact = _relative_path(bot_b_dir / "bot_artifact.json", path_root)
    red_debug = _relative_path(bot_a_dir / "debug", path_root) if (bot_a_dir / "debug").exists() else None
    blue_debug = _relative_path(bot_b_dir / "debug", path_root) if (bot_b_dir / "debug").exists() else None
    matchup_video_path = matchup_dir / "match.webm"
    if not matchup_video_path.exists():
        matchup_video_path = matchup_dir / "match.mp4"
    matchup_video = _relative_path(matchup_video_path, path_root) if matchup_video_path.exists() else None

    extra_fields = {
        "red_model": bot_a_obj.generator,
        "blue_model": bot_b_obj.generator,
        "red_id": bot_name_to_id[name_a],
        "blue_id": bot_name_to_id[name_b],
        "iteration": iteration,
        "timestamp": datetime.datetime.now().isoformat(),
        "wins_a": matchup_result.wins_a,
        "wins_b": matchup_result.wins_b,
        "ties": matchup_result.ties,
        "red_artifact": red_artifact,
        "blue_artifact": blue_artifact,
        "red_debug_dir": red_debug,
        "blue_debug_dir": blue_debug,
        "matchup_video": matchup_video,
    }

    # Use save_matchup_to_disk for both match_result.json and match_data.json
    ds_matchup = matchup_result._ds_matchup
    if ds_matchup is not None:
        _trace(trace_progress, f"[matchup {idx + 1}] {name_a} vs {name_b}: save_matchup_to_disk start")
        save_matchup_to_disk(
            ds_matchup,
            matchup_dir,
            tool_name="tournament",
            red_bot=name_a,
            blue_bot=name_b,
            extra_match_result_fields=extra_fields,
        )
        _trace(trace_progress, f"[matchup {idx + 1}] {name_a} vs {name_b}: save_matchup_to_disk done")
    else:
        logger.warning("No _ds_matchup on matchup_result for %s vs %s; skipping disk save", name_a, name_b)

    # Memory hygiene: _ds_matchup carries GameRecords with per-step physics
    # (~13MB/match). It's already persisted to disk via save_matchup_to_disk
    # above, and downstream code (run_round_robin's aggregate Elo loop) only
    # uses matchup_result.matches/wins_a/wins_b/ties. Setting to None here
    # means the worker doesn't pickle 13MB back to the driver per match,
    # which previously OOM-killed the driver after ~1500 matches.
    matchup_result._ds_matchup = None

    # Read back the match_result_data for return value (consumed by callers)
    match_result_path = matchup_dir / "match_result.json"
    if match_result_path.exists():
        with match_result_path.open("r", encoding="utf-8") as f:
            match_result_data = json.load(f)
    else:
        match_result_data = {"red_bot": name_a, "blue_bot": name_b, **extra_fields}

    _trace(trace_progress, f"[matchup {idx + 1}] {name_a} vs {name_b}: finished")

    return (idx, matchup_result, match_result_data)


def run_round_robin(
    bots: Dict[str, BotArtifact],
    arena_xml: Path,
    out_dir: Path,
    *,
    n_rollouts: int = 5,
    max_steps: int = 1000,
    record_video: bool = True,
    save_all_videos: bool = True,
    video_width: int = 640,
    video_height: int = 480,
    highres: bool = False,
    verbose: bool = True,
    trace_progress: bool = True,
    use_material_palette: bool = False,
    season_id: str = "",
    tournament_id: str = "",
    camera_mode: str = "tracking",
    score_function: str = "any",
    iteration: int = 0,
    season_root: Optional[Path] = None,
    init_seed: int = 42,
    physics_mode: str = "3d",
    n_parallel_matches: int = 1,
    n_parallel_seeds: int = 1,
    gear_clamp_ratio: float = 0.0,
    match_time: Optional[float] = None,
    inactivity_timeout_seconds: Optional[float] = 10.0,
    inactivity_min_displacement: float = 0.5,
    size_limits: Optional[Tuple[float, float, float]] = None,
    contact_fidelity: str = "high",
    raise_on_error: bool = False,
    rendering_flags: Optional[Dict[str, bool]] = None,
) -> TournamentResult:
    """Run round-robin tournament between all bots.

    Uses existing infrastructure from dspy_core and eval/metrics.
    For N bots: N*(N-1)/2 matchups, each played n_rollouts times.

    Args:
        bots: Dictionary mapping bot name to BotArtifact
        arena_xml: Path to arena XML
        out_dir: Output directory for tournament
        n_rollouts: Number of matches per bot pair
        max_steps: Maximum simulation steps per match
        record_video: Whether to save videos
        save_all_videos: If True, save video for every rollout. If False, save only 1.
        video_width: Video frame width (ignored if highres=True)
        video_height: Video frame height (ignored if highres=True)
        highres: If True, use 1920x1080 resolution for videos
        verbose: Print progress information
        iteration: Current season iteration (for JSON enrichment + seed scheme).
        season_root: Root output directory of the season (for relative paths in JSON).
        init_seed: Base seed for deterministic seed scheme (default 42).
        n_parallel_matches: Number of matchups to run in parallel (1 = sequential).
        n_parallel_seeds: Number of seeds to run in parallel within each matchup (1 = sequential).

    Returns:
        TournamentResult with all matches and standings
    """
    ensure_dir(out_dir)

    # Create matches directory and save env.xml
    matches_dir = out_dir / "matches"
    ensure_dir(matches_dir)
    env_xml_dest = matches_dir / "env.xml"
    shutil.copy(arena_xml, env_xml_dest)

    # Apply highres if requested
    if highres:
        video_width, video_height = 1920, 1080

    # Determine how many videos to save
    # Position swapping is handled by seed (odd seeds swap sides), so we have one direction
    if not record_video:
        save_video_seeds = 0
    elif save_all_videos:
        # Save video for all rollouts
        save_video_seeds = n_rollouts
    else:
        # Save only 1 video
        save_video_seeds = 1

    bot_names = list(bots.keys())
    # Create bot_id mapping: use list index as ID
    bot_name_to_id = {name: idx for idx, name in enumerate(bot_names)}

    # Map bot names to model-level identity (generator) for standings
    _bot_to_model = {name: bot.generator for name, bot in bots.items()}

    def _model_name(bot_name: str) -> str:
        """Resolve bot name to model-level identity for standings."""
        return _bot_to_model.get(bot_name, bot_name)

    all_matchups = list(combinations(bot_names, 2))
    # Exclude baseline-vs-baseline matches (e.g. platonic solids: *hedron vs *hedron)
    matchups = [
        (a, b) for a, b in all_matchups
        if not (a.endswith("dron") and b.endswith("dron"))
    ]
    if len(matchups) < len(all_matchups) and verbose:
        logger.info(
            "Excluded %d baseline-vs-baseline matchups",
            len(all_matchups) - len(matchups),
        )
    num_matchups = len(matchups)
    num_matches_per_tournament = num_matchups * n_rollouts
    seed_base = init_seed + iteration * num_matches_per_tournament

    # Record initial Elo for before/after tracking
    initial_elo = {name: 1000.0 for name in bot_names}

    # Shared keyword args for _run_single_matchup calls
    shared_kwargs = dict(
        bots=bots,
        bot_name_to_id=bot_name_to_id,
        arena_xml=arena_xml,
        matches_dir=matches_dir,
        n_rollouts=n_rollouts,
        seed_base=seed_base,
        max_steps=max_steps,
        save_video_seeds=save_video_seeds,
        video_width=video_width,
        video_height=video_height,
        use_material_palette=use_material_palette,
        season_id=season_id,
        tournament_id=tournament_id,
        camera_mode=camera_mode,
        score_function=score_function,
        physics_mode=physics_mode,
        iteration=iteration,
        season_root=season_root,
        out_dir=out_dir,
        n_parallel_seeds=n_parallel_seeds,
        gear_clamp_ratio=gear_clamp_ratio,
        match_time=match_time,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        inactivity_min_displacement=inactivity_min_displacement,
        size_limits=size_limits,
        contact_fidelity=contact_fidelity,
        trace_progress=trace_progress,
        rendering_flags=rendering_flags,
    )

    use_parallel = n_parallel_matches > 1 and num_matchups > 1

    if use_parallel:
        # ── Parallel path ──────────────────────────────────────────
        workers = min(n_parallel_matches, num_matchups)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round-Robin Tournament: {len(bots)} bots "
                  f"(parallel processes, {workers} concurrent matchups)")
            print(f"Matchups: {num_matchups} | Seeds per matchup: {n_rollouts}")
            if record_video:
                print(f"Videos per matchup: {save_video_seeds} ({'all' if save_all_videos else '1'})")
                print(f"Video resolution: {video_width}x{video_height}")
            print(f"{'='*60}\n")

        completed: Dict[int, Tuple[MatchupResult, Dict[str, Any]]] = {}
        t_start = time.monotonic()
        task_states: Dict[int, Dict[str, Any]] = {}
        queued_indices: Deque[int] = deque()

        # Resumability: skip matchups whose match_result.json already exists.
        # Lets a kill+restart cycle naturally pick up where it left off (e.g.
        # after a MuJoCo deadlock takes down a worker pool). Pair dir naming
        # mirrors the convention in _run_single_matchup line 459.
        preexisting_indices: set = set()
        for idx, (name_a, name_b) in enumerate(matchups):
            pair_dir = matches_dir / f"{name_a}_vs_{name_b}"
            if (pair_dir / "match_result.json").is_file():
                preexisting_indices.add(idx)
        if preexisting_indices and verbose:
            print(
                f"[run_round_robin] resumability: {len(preexisting_indices)}/{num_matchups} "
                f"matchups already have match_result.json — skipping"
            )

        # Index the first `workers` non-skipped matchups as initially "running".
        running_budget = workers
        for idx, (name_a, name_b) in enumerate(matchups):
            submitted_at = time.monotonic()
            matchup_seed_start = seed_base + idx * n_rollouts
            matchup_seed_end = matchup_seed_start + n_rollouts - 1
            if idx in preexisting_indices:
                task_states[idx] = {
                    "label": f"{name_a} vs {name_b}",
                    "seed_range": f"{matchup_seed_start}-{matchup_seed_end}",
                    "status": "skipped",
                    "submitted_at": submitted_at,
                    "started_at": submitted_at,
                    "duration": 0.0,
                    "result": "preexisting",
                }
                continue
            initially_running = running_budget > 0
            if initially_running:
                running_budget -= 1
            state = {
                "label": f"{name_a} vs {name_b}",
                "seed_range": f"{matchup_seed_start}-{matchup_seed_end}",
                "status": "running" if initially_running else "queued",
                "submitted_at": submitted_at,
                "started_at": submitted_at if initially_running else None,
                "duration": None,
                "result": "",
            }
            task_states[idx] = state
            if not initially_running:
                queued_indices.append(idx)

        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for idx, (name_a, name_b) in enumerate(matchups):
                if idx in preexisting_indices:
                    continue
                matchup_seed_start = seed_base + idx * n_rollouts
                matchup_seed_end = matchup_seed_start + n_rollouts - 1
                _trace(
                    trace_progress,
                    f"[queue {idx + 1}/{num_matchups}] {name_a} vs {name_b} "
                    f"(seeds={matchup_seed_start}..{matchup_seed_end})"
                )
                fut = pool.submit(
                    _run_single_matchup,
                    idx=idx,
                    name_a=name_a,
                    name_b=name_b,
                    **shared_kwargs,
                )
                futures[fut] = idx

            done_count = 0
            pending = set(futures)
            heartbeat_seconds = 5.0
            if verbose:
                print(_render_dashboard(
                    task_states,
                    done_count=done_count,
                    total_count=num_matchups,
                    workers=workers,
                    tournament_elapsed=time.monotonic() - t_start,
                ))
            while pending:
                done, pending = wait(
                    pending,
                    timeout=heartbeat_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    if verbose:
                        print(_render_dashboard(
                            task_states,
                            done_count=done_count,
                            total_count=num_matchups,
                            workers=workers,
                            tournament_elapsed=time.monotonic() - t_start,
                        ))
                    continue
                for fut in done:
                    done_count += 1
                    try:
                        m_idx, matchup_result, match_data = fut.result()
                        # Memory hygiene: don't store match_data in `completed`
                        # because it has full per-step traces (~10MB/match)
                        # already saved to disk by _run_single_matchup. The
                        # downstream loop (line ~909) uses only matchup_result.
                        # The local `match_data` ref is still used by the
                        # verbose-print block below (na/nb fields), so we
                        # don't `del` it explicitly — Python's GC reclaims it
                        # when the next loop iteration overwrites the binding.
                        completed[m_idx] = (matchup_result, None)
                        finished_at = time.monotonic()
                        task_states[m_idx]["status"] = "completed"
                        task_states[m_idx]["duration"] = finished_at - (
                            task_states[m_idx]["started_at"] or task_states[m_idx]["submitted_at"]
                        )
                        task_states[m_idx]["result"] = (
                            f"{matchup_result.wins_a}-{matchup_result.wins_b}-{matchup_result.ties}"
                        )
                        if queued_indices:
                            next_idx = queued_indices.popleft()
                            task_states[next_idx]["status"] = "running"
                            task_states[next_idx]["started_at"] = finished_at
                        if verbose:
                            elapsed = time.monotonic() - t_start
                            na = match_data["red_bot"]
                            nb = match_data["blue_bot"]
                            wa = matchup_result.wins_a
                            wb = matchup_result.wins_b
                            ties = matchup_result.ties
                            print(f"  \033[32m[{done_count}/{num_matchups}]\033[0m "
                                  f"{na} vs {nb}: {wa}-{wb}-{ties} "
                                  f"\033[90m[{elapsed:.1f}s]\033[0m")
                            print(_render_dashboard(
                                task_states,
                                done_count=done_count,
                                total_count=num_matchups,
                                workers=workers,
                                tournament_elapsed=time.monotonic() - t_start,
                            ))
                    except Exception as exc:
                        orig_idx = futures[fut]
                        na, nb = matchups[orig_idx]
                        finished_at = time.monotonic()
                        task_states[orig_idx]["status"] = "error"
                        task_states[orig_idx]["duration"] = finished_at - (
                            task_states[orig_idx]["started_at"] or task_states[orig_idx]["submitted_at"]
                        )
                        task_states[orig_idx]["result"] = f"ERROR: {exc}"
                        if queued_indices:
                            next_idx = queued_indices.popleft()
                            task_states[next_idx]["status"] = "running"
                            task_states[next_idx]["started_at"] = finished_at
                        if raise_on_error:
                            _trace(
                                trace_progress,
                                f"[error] {na} vs {nb}: re-raising exception in serial-debug mode"
                            )
                            raise
                        logger.error("Matchup %s vs %s failed: %s", na, nb, exc)
                        if verbose:
                            print(f"  \033[31m[{done_count}/{num_matchups}]\033[0m "
                                  f"{na} vs {nb}: ERROR - {exc}")
                            print(_render_dashboard(
                                task_states,
                                done_count=done_count,
                                total_count=num_matchups,
                                workers=workers,
                                tournament_elapsed=time.monotonic() - t_start,
                            ))

        # Assemble results in original matchup order
        all_matches: List[MatchResult] = []
        for idx in range(num_matchups):
            if idx in completed:
                matchup_result, _ = completed[idx]
                all_matches.extend(matchup_result.matches)

        # Compute Elo once from all matches (batch mode for parallel)
        model_names = [_model_name(n) for n in bot_names]
        elo_history: List[Dict[str, float]] = [{mn: 1000.0 for mn in model_names}]
        all_outcomes = []
        for r in all_matches:
            pa = _model_name(r.red_bot)
            pb = _model_name(r.blue_bot)
            if r.winner == "red":
                winner = pa
            elif r.winner == "blue":
                winner = pb
            else:
                winner = "draw"
            all_outcomes.append(MatchOutcome(
                player_a=pa,
                player_b=pb,
                winner=winner,
            ))

        if all_outcomes:
            final_standings = compute_standings(all_outcomes)
            final_elo = {name: s["elo"] for name, s in final_standings.items()}
            for mn in model_names:
                if mn not in final_elo:
                    final_elo[mn] = 1000.0
            elo_history.append(final_elo)
        else:
            final_standings = {_model_name(name): {"wins": 0, "losses": 0, "draws": 0, "elo": 1000.0}
                               for name in bot_names}

    else:
        # ── Serial path ───────────────────────────────────────────
        # Matchups run one at a time. After each matchup we refit
        # Bradley-Terry over *all matches played so far* (a BT fit at every
        # stage), so the trajectory is consistent with the final standings —
        # no separate per-matchup or sequential-Elo estimator.
        if verbose:
            print(f"\n{'='*60}")
            print(f"Round-Robin Tournament: {len(bots)} bots")
            print(f"Seeds per matchup: {n_rollouts}")
            if record_video:
                print(f"Videos per matchup: {save_video_seeds} ({'all' if save_all_videos else '1'})")
                print(f"Video resolution: {video_width}x{video_height}")
            print(f"{'='*60}\n")

        all_matches = []
        model_names = [_model_name(n) for n in bot_names]
        elo_history = [{mn: 1000.0 for mn in model_names}]
        cumulative_outcomes: List[MatchOutcome] = []

        for idx, (name_a, name_b) in enumerate(matchups):
            if verbose:
                print(f"[{idx+1}/{num_matchups}] {_model_name(name_a)} vs {_model_name(name_b)} ({n_rollouts} seeds)...")

            _, matchup_result, _ = _run_single_matchup(
                idx=idx,
                name_a=name_a,
                name_b=name_b,
                **shared_kwargs,
            )

            all_matches.extend(matchup_result.matches)

            if verbose:
                print(f"    {_model_name(name_a)}: {matchup_result.wins_a} wins | "
                      f"{_model_name(name_b)}: {matchup_result.wins_b} wins | "
                      f"{matchup_result.ties} ties")

            # Refit Bradley-Terry over all matches played so far.
            for r in matchup_result.matches:
                pa = _model_name(r.red_bot)
                pb = _model_name(r.blue_bot)
                if r.winner == "red":
                    winner = pa
                elif r.winner == "blue":
                    winner = pb
                else:
                    winner = "draw"
                cumulative_outcomes.append(MatchOutcome(
                    player_a=pa,
                    player_b=pb,
                    winner=winner,
                ))

            current_standings = compute_standings(cumulative_outcomes)
            current_elo = {name: s["elo"] for name, s in current_standings.items()}
            for mn in model_names:
                if mn not in current_elo:
                    current_elo[mn] = 1000.0
            elo_history.append(current_elo)

        # Final standings = BT over every match (== last trajectory point).
        final_standings = compute_standings(cumulative_outcomes)

    # ── Common: print standings, save results ─────────────────────
    if verbose:
        print(f"\n{'='*60}")
        print("Final Standings:")
        print(f"{'='*60}")
        ranked = elo_rank_players({name: s["elo"] for name, s in final_standings.items()})
        for rank, (name, elo) in enumerate(ranked, 1):
            stats = final_standings[name]
            print(f"  {rank}. {name}: Elo {elo:.0f} | W:{stats['wins']} L:{stats['losses']} D:{stats['draws']}")
        print(f"{'='*60}\n")

    result = TournamentResult(
        matches=all_matches,
        standings=final_standings,
        elo_history=elo_history,
        metadata={
            "num_bots": len(bots),
            "bot_names": bot_names,
            "n_rollouts": n_rollouts,
            "total_matches": len(all_matches),
            "video_resolution": f"{video_width}x{video_height}" if record_video else None,
            "timestamp": datetime.datetime.now().isoformat(),
            "n_parallel_matches": n_parallel_matches,
            "n_parallel_seeds": n_parallel_seeds,
        },
    )

    # Save elo.json with before/after Elo per player using bot directory names
    elo_data = {}
    for name in bot_names:
        bot = bots[name]
        bot_dir_name = bot.morphology_xml.parent.name
        mn = _model_name(name)
        elo_data[bot_dir_name] = {
            "before": initial_elo.get(name, 1000.0),
            "after": final_standings[mn]["elo"] if mn in final_standings else 1000.0,
        }
    elo_path = out_dir / "elo.json"
    with elo_path.open("w", encoding="utf-8") as f:
        json.dump(elo_data, f, indent=2)

    return result


def get_win_rate_matrix(
    tournament: TournamentResult,
) -> Tuple[List[str], List[List[float]]]:
    """Compute win rate matrix from tournament results.

    Args:
        tournament: Tournament results

    Returns:
        Tuple of (bot_names, matrix) where matrix[i][j] is win rate of bot i vs bot j
    """
    bot_names = list(tournament.standings.keys())
    n = len(bot_names)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    # Count wins for each pair
    wins = {(a, b): 0 for a in bot_names for b in bot_names if a != b}
    total = {(a, b): 0 for a in bot_names for b in bot_names if a != b}

    for match in tournament.matches:
        a, b = match.red_bot, match.blue_bot

        total[(a, b)] = total.get((a, b), 0) + 1
        total[(b, a)] = total.get((b, a), 0) + 1

        if match.winner == "red":
            wins[(a, b)] = wins.get((a, b), 0) + 1
        elif match.winner == "blue":
            wins[(b, a)] = wins.get((b, a), 0) + 1
        else:
            # Tie - give 0.5 to each
            wins[(a, b)] = wins.get((a, b), 0) + 0.5
            wins[(b, a)] = wins.get((b, a), 0) + 0.5

    # Compute win rates
    for i, a in enumerate(bot_names):
        for j, b in enumerate(bot_names):
            if a == b:
                matrix[i][j] = 0.5  # Diagonal = 0.5 (self)
            elif total.get((a, b), 0) > 0:
                matrix[i][j] = wins.get((a, b), 0) / total.get((a, b), 1)
            else:
                matrix[i][j] = 0.5  # No matches

    return bot_names, matrix


__all__ = [
    "MatchResult",
    "MatchupResult",
    "TournamentResult",
    "run_matchup",
    "run_round_robin",
    "get_win_rate_matrix",
]
