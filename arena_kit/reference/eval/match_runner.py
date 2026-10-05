"""Unified match runner: run_seeds() and save_matchup_to_disk().

Replaces run_rollouts() from metrics.py. Returns MatchupResult (typed)
instead of RolloutResult (loose dicts).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import time
import datetime
import multiprocessing
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mjarena.design_shop.types import GameRecord, MatchupResult
from mjarena.policy_spec import PolicySpec

logger = logging.getLogger(__name__)


def _trace(enabled: bool, message: str) -> None:
    """Emit a timestamped progress line immediately."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    thread_name = threading.current_thread().name
    print(f"[{timestamp}] [{thread_name}] {message}", flush=True)


# =============================================================================
# run_seeds — run N match seeds, return MatchupResult
# =============================================================================


def run_seeds(
    *,
    run_match_fn: Callable,
    policy_callable: Optional[Callable] = None,
    policy_spec: Optional[PolicySpec] = None,
    n_seeds: int = 3,
    seeds: Optional[List[int]] = None,
    n_parallel: int = 1,
    parallel_backend: str = "thread",
    trace_progress: bool = False,
    progress_label: str = "",
    simulation_progress_listener: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MatchupResult:
    """Run N match seeds and return a typed MatchupResult.

    Args:
        run_match_fn: (policy_callable, seed=int) -> dict OR GameRecord.
        policy_callable: Red bot policy function.
        n_seeds: Number of seeds to run (ignored if seeds is provided).
        seeds: Explicit list of seed ints.
        n_parallel: Max parallel workers (1 = sequential).
        parallel_backend: "thread" or "process" for the multi-seed branch.

    Returns:
        MatchupResult with GameRecords, CombatScores, and seed scores.
    """
    if seeds is None:
        seeds = list(range(n_seeds))
    if not policy_callable and not policy_spec:
        raise ValueError("run_seeds requires either policy_callable or policy_spec")

    backend = (parallel_backend or "thread").strip().lower()
    if backend not in {"thread", "process"}:
        raise ValueError(f"Unsupported seed parallel backend: {parallel_backend}")

    t0 = time.monotonic()
    label = progress_label or "matchup"
    _trace(
        trace_progress,
        f"{label}: run_seeds entered (n_seeds={len(seeds)}, n_parallel={n_parallel}, backend={backend})"
    )

    # Run all seeds
    if n_parallel > 1 and len(seeds) > 1:
        workers = min(n_parallel, len(seeds))
        collected: Dict[int, Any] = {}
        _trace(
            trace_progress,
            f"{label}: running seeds in parallel with {workers} workers via {backend}",
        )
        if backend == "process":
            if policy_spec is None:
                raise ValueError("policy_spec is required when parallel_backend='process'")
            collected = _run_seeds_in_processes(
                run_match_fn=run_match_fn,
                policy_spec=policy_spec,
                seeds=seeds,
                workers=workers,
                trace_progress=trace_progress,
                progress_label=label,
                simulation_progress_listener=simulation_progress_listener,
            )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _run_one_seed,
                        run_match_fn,
                        policy_callable,
                        seed,
                        seed_index=i,
                        trace_progress=trace_progress,
                        progress_label=label,
                        simulation_progress_listener=simulation_progress_listener,
                    ): i
                    for i, seed in enumerate(seeds)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    collected[i] = fut.result()
                    _trace(
                        trace_progress,
                        f"{label}: collected seed {seeds[i]} ({len(collected)}/{len(seeds)})"
                    )
        raw_results = [collected[i] for i in range(len(seeds))]
    else:
        raw_results = [
            _run_one_seed(
                run_match_fn,
                policy_callable,
                seed,
                seed_index=i,
                trace_progress=trace_progress,
                progress_label=label,
                simulation_progress_listener=simulation_progress_listener,
            )
            for i, seed in enumerate(seeds)
        ]

    # Lazy import to avoid circular dependency
    from mjarena.eval.metrics import compute_combat_score, compute_seed_score

    # Convert to GameRecords + score
    game_records: List[GameRecord] = []
    combat_scores = []
    seed_scores: List[float] = []
    video_fps: Optional[int] = None

    for record in raw_results:
        cs = compute_combat_score(record)                       # red / model_a
        record.combat_metrics = {
            "engagement": cs.engagement,
            "dominant_contact": cs.dominant_contact,
            "displacement": cs.displacement,
            "destabilization": cs.destabilization,
            "self_stability": cs.self_stability,
            "composite": cs.composite,
        }
        # Side-neutral: also score the BLUE bot (model_b) from the same match, so
        # behavioural stats aren't biased by which slot a model happened to get.
        cs_b = compute_combat_score(record, flip=True)          # blue / model_b
        record.combat_metrics_b = {
            "engagement": cs_b.engagement,
            "dominant_contact": cs_b.dominant_contact,
            "displacement": cs_b.displacement,
            "destabilization": cs_b.destabilization,
            "self_stability": cs_b.self_stability,
            "composite": cs_b.composite,
        }
        ss = compute_seed_score(record.winner, cs)

        game_records.append(record)
        combat_scores.append(cs)
        seed_scores.append(ss)

    elapsed = time.monotonic() - t0
    _trace(trace_progress, f"{label}: scoring complete in {elapsed:.1f}s")

    return MatchupResult(
        game_records=game_records,
        combat_scores=combat_scores,
        seed_scores=seed_scores,
        elapsed_sec=round(elapsed, 2),
        video_fps=video_fps,
    )


def _run_one_seed(
    run_match_fn: Callable,
    policy_callable: Optional[Callable],
    seed: int,
    *,
    seed_index: Optional[int] = None,
    trace_progress: bool = False,
    progress_label: str = "",
    simulation_progress_listener: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Any:
    """Run a single seed. Returns whatever run_match_fn returns (dict or GameRecord)."""
    label = progress_label or "matchup"
    t0 = time.monotonic()
    _trace(trace_progress, f"{label}: seed {seed} start")
    if simulation_progress_listener:
        simulation_progress_listener({"type": "seed_started", "seed": seed})
    try:
        run_match_params = set()
        try:
            run_match_params = set(inspect.signature(run_match_fn).parameters)
        except (TypeError, ValueError):
            run_match_params = set()

        def _progress_callback(event: Dict[str, Any]) -> None:
            if not simulation_progress_listener:
                return
            payload = dict(event)
            payload["seed"] = seed
            simulation_progress_listener(payload)

        kwargs: Dict[str, Any] = {"seed": seed}
        if "seed_index" in run_match_params:
            kwargs["seed_index"] = seed_index
        if "progress_callback" in run_match_params:
            kwargs["progress_callback"] = _progress_callback
        result = run_match_fn(policy_callable, **kwargs)
        winner = getattr(result, "winner", "?")
        num_steps = getattr(result, "num_steps", "?")
        if simulation_progress_listener:
            simulation_progress_listener({
                "type": "seed_done",
                "seed": seed,
                "step": getattr(result, "num_steps", None),
                "winner": winner,
                "elapsed_sec": time.monotonic() - t0,
            })
        _trace(
            trace_progress,
            f"{label}: seed {seed} done in {time.monotonic() - t0:.1f}s "
            f"(winner={winner}, steps={num_steps})"
        )
        return result
    except Exception as exc:
        logger.warning("Seed %d failed: %s", seed, exc)
        if simulation_progress_listener:
            simulation_progress_listener({
                "type": "seed_failed",
                "seed": seed,
                "error": str(exc),
                "elapsed_sec": time.monotonic() - t0,
            })
        _trace(
            trace_progress,
            f"{label}: seed {seed} failed after {time.monotonic() - t0:.1f}s: {exc}"
        )
        raise


def _run_seeds_in_processes(
    *,
    run_match_fn: Callable,
    policy_spec: PolicySpec,
    seeds: List[int],
    workers: int,
    trace_progress: bool,
    progress_label: str,
    simulation_progress_listener: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[int, Any]:
    """Run seed rollouts in spawned worker processes."""
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager() if simulation_progress_listener else None
    progress_queue = manager.Queue() if manager is not None else None
    listener_thread = None
    stop_event = threading.Event()

    if progress_queue is not None and simulation_progress_listener is not None:
        listener_thread = threading.Thread(
            target=_forward_process_progress,
            args=(progress_queue, stop_event, simulation_progress_listener),
            name="seed-progress-forwarder",
            daemon=True,
        )
        listener_thread.start()

    try:
        collected: Dict[int, Any] = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _run_one_seed_process,
                    run_match_fn,
                    policy_spec,
                    seed,
                    i,
                    trace_progress,
                    progress_label,
                    progress_queue,
                ): i
                for i, seed in enumerate(seeds)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                collected[i] = fut.result()
                _trace(
                    trace_progress,
                    f"{progress_label}: collected seed {seeds[i]} ({len(collected)}/{len(seeds)})",
                )
        return collected
    finally:
        if progress_queue is not None:
            progress_queue.put(None)
        stop_event.set()
        if listener_thread is not None:
            listener_thread.join()
        if manager is not None:
            manager.shutdown()


def _forward_process_progress(
    progress_queue: Any,
    stop_event: threading.Event,
    simulation_progress_listener: Callable[[Dict[str, Any]], None],
) -> None:
    """Relay per-seed progress events from worker processes to the dashboard."""
    while not stop_event.is_set():
        try:
            event = progress_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if event is None:
            return
        simulation_progress_listener(event)


def _run_one_seed_process(
    run_match_fn: Callable,
    policy_spec: PolicySpec,
    seed: int,
    seed_index: int,
    trace_progress: bool,
    progress_label: str,
    progress_queue: Any,
) -> Any:
    """Worker entrypoint for one seed simulation in a separate process."""
    label = progress_label or "matchup"
    t0 = time.monotonic()
    _trace(trace_progress, f"{label}: seed {seed} start")
    if progress_queue is not None:
        progress_queue.put({"type": "seed_started", "seed": seed})

    def _progress_callback(event: Dict[str, Any]) -> None:
        if progress_queue is None:
            return
        payload = dict(event)
        payload["seed"] = seed
        progress_queue.put(payload)

    try:
        if not hasattr(run_match_fn, "run_with_policy_spec"):
            raise TypeError(
                "process-based seed execution requires a picklable match runner with "
                "run_with_policy_spec()"
            )
        result = run_match_fn.run_with_policy_spec(
            policy_spec,
            seed=seed,
            seed_index=seed_index,
            progress_callback=_progress_callback,
        )
        winner = getattr(result, "winner", "?")
        num_steps = getattr(result, "num_steps", "?")
        if progress_queue is not None:
            progress_queue.put({
                "type": "seed_done",
                "seed": seed,
                "step": getattr(result, "num_steps", None),
                "winner": winner,
                "elapsed_sec": time.monotonic() - t0,
            })
        _trace(
            trace_progress,
            f"{label}: seed {seed} done in {time.monotonic() - t0:.1f}s "
            f"(winner={winner}, steps={num_steps})",
        )
        return result
    except Exception as exc:
        logger.warning("Seed %d failed: %s", seed, exc)
        if progress_queue is not None:
            progress_queue.put({
                "type": "seed_failed",
                "seed": seed,
                "error": str(exc),
                "elapsed_sec": time.monotonic() - t0,
            })
        _trace(
            trace_progress,
            f"{label}: seed {seed} failed after {time.monotonic() - t0:.1f}s: {exc}",
        )
        raise


# =============================================================================
# save_matchup_to_disk — write match_result.json + match_data.json
# =============================================================================


def save_matchup_to_disk(
    matchup: MatchupResult,
    output_dir: Path,
    *,
    tool_name: str = "match",
    red_bot: str = "bot",
    blue_bot: str = "",
    extra_match_result_fields: Optional[Dict] = None,
) -> None:
    """Save matchup results to disk.

    Creates:
        output_dir/match_result.json  — lightweight index (for app listings, Elo)
        output_dir/match_data.json    — full replay telemetry (columnar GameRecord)

    Videos (seed_N.mp4) are saved directly by create_match_runner, not here.

    Args:
        matchup: MatchupResult to save.
        output_dir: Directory to write files into (created if needed).
        tool_name: Label for the match type.
        red_bot: Name of the red bot.
        blue_bot: Name of the blue bot.
        extra_match_result_fields: Additional fields merged into match_result.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── match_result.json (lightweight summary) ──
    matches_summary = []
    for i, r in enumerate(matchup.game_records):
        matches_summary.append({
            "seed": r.seed,
            "winner": r.winner,
            "num_steps": r.num_steps,
            "combat_metrics": r.combat_metrics,
            "combat_metrics_b": getattr(r, "combat_metrics_b", None),
            "termination_reason": r.termination_reason,
            "initial_distance": r.initial_distance,
            "initial_red_com": r.initial_red_pos,
            "initial_blue_com": r.initial_blue_pos,
            "physics_unstable": r.physics_unstable,
        })

    match_result = {
        "red_bot": red_bot,
        "blue_bot": blue_bot,
        "tool_name": tool_name,
        "n_seeds": matchup.n_seeds,
        "matches": matches_summary,
    }
    if matchup.video_fps is not None:
        match_result["video_fps"] = matchup.video_fps
    if extra_match_result_fields:
        match_result.update(extra_match_result_fields)

    (output_dir / "match_result.json").write_text(
        json.dumps(match_result, indent=2, default=str)
    )

    # ── match_data.json (full columnar telemetry) ──
    # Skippable: ~7 MB/match replay trace, only needed for video rendering. Set
    # ARENA_SKIP_MATCH_DATA=1 for large headless RR selection passes (no video) to
    # avoid filling disk; match_result.json (used for Elo) is always written above.
    if os.environ.get("ARENA_SKIP_MATCH_DATA") != "1":
        match_data: Dict[str, Any] = {}
        if matchup.video_fps is not None:
            match_data["video_fps"] = matchup.video_fps
        for r in matchup.game_records:
            match_data[f"seed_{r.seed}"] = r.to_dict()

        (output_dir / "match_data.json").write_text(
            json.dumps(match_data, indent=2, default=str)
        )
