"""
Build phase runner for season/tournament iterations.

Runs the build phase where each LLM generates a new bot.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from collections import deque
from concurrent.futures import BrokenExecutor, FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Deque, Dict, Optional, TYPE_CHECKING

from mjarena.agents.types import BotArtifact
from mjarena.core.bot_builder import build_bot
from mjarena.core.build_config import BuildConfig
from mjarena.core.build_progress import BuildProgressReporter, load_progress_snapshot, summarize_progress_snapshot


logger = logging.getLogger(__name__)


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


def _artifact_summary(artifact: BotArtifact) -> str:
    """Render a concise result summary for a completed bot build."""
    if artifact.forfeit:
        return f"FORFEIT ({artifact.forfeit_stage})"
    return f"OK morph={artifact.morphology_score:.2f} ctrl={artifact.controller_score:.2f}"


class _LiveDashboardDisplay:
    """Render a multi-line dashboard in-place on an interactive terminal."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._last_lines = 0

    def render(self, text: str) -> None:
        if not self.enabled:
            return
        writer = sys.stdout
        payload = text.rstrip("\n") + "\n"
        if self._last_lines:
            writer.write_ephemeral(f"\x1b[{self._last_lines}F\x1b[J")
        writer.write_ephemeral(payload)
        self._last_lines = payload.count("\n")


class _BuildDashboardMonitor:
    """Poll build progress files and refresh the build dashboard."""

    def __init__(
        self,
        *,
        task_states: Dict[int, Dict[str, Any]],
        state_lock: threading.Lock,
        progress_paths: Dict[int, Path],
        total_count: int,
        workers: int,
        build_t0: float,
        enabled: bool,
        refresh_seconds: float = 2.0,
    ):
        self._task_states = task_states
        self._state_lock = state_lock
        self._progress_paths = progress_paths
        self._total_count = total_count
        self._workers = workers
        self._build_t0 = build_t0
        self._refresh_seconds = refresh_seconds
        self._display = _LiveDashboardDisplay(enabled)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return self._display.enabled

    def start(self) -> None:
        if not self.enabled:
            return
        self.render()
        self._thread = threading.Thread(
            target=self._run,
            name="build-dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.render()

    def render(self) -> None:
        snapshot_states = self._snapshot_task_states()
        done_count = sum(
            1 for state in snapshot_states.values()
            if state["status"] in {"completed", "error"}
        )
        progress_snapshots = {
            idx: load_progress_snapshot(path)
            for idx, path in self._progress_paths.items()
        }
        self._display.render(_render_build_dashboard(
            snapshot_states,
            progress_snapshots=progress_snapshots,
            done_count=done_count,
            total_count=self._total_count,
            workers=self._workers,
            build_elapsed=time.monotonic() - self._build_t0,
        ))

    def _run(self) -> None:
        while not self._stop_event.wait(self._refresh_seconds):
            self.render()

    def _snapshot_task_states(self) -> Dict[int, Dict[str, Any]]:
        with self._state_lock:
            return {
                idx: dict(state)
                for idx, state in self._task_states.items()
            }


def _render_build_dashboard(
    task_states: Dict[int, Dict[str, Any]],
    *,
    progress_snapshots: Optional[Dict[int, Optional[Dict[str, Any]]]] = None,
    done_count: int,
    total_count: int,
    workers: int,
    build_elapsed: float,
) -> str:
    """Build a snapshot table showing queued/running/completed bot builds."""
    now = time.monotonic()
    detail_width = 78
    if getattr(sys.stdout, "isatty", lambda: False)():
        terminal_cols = shutil.get_terminal_size((200, 40)).columns
        # Fixed widths for all columns except Detail, plus separators.
        fixed_non_detail = 3 + 28 + 8 + 14 + 7 + 28
        separator_width = 3 * 6
        detail_width = max(78, terminal_cols - fixed_non_detail - separator_width)
    running_count = sum(1 for state in task_states.values() if state["status"] == "running")
    queued_count = sum(1 for state in task_states.values() if state["status"] == "queued")
    completed_count = sum(1 for state in task_states.values() if state["status"] == "completed")
    error_count = sum(1 for state in task_states.values() if state["status"] == "error")

    lines = [
        f"[Build Dashboard] {_progress_bar(done_count, total_count)} {done_count}/{total_count} complete "
        f"| running={running_count} queued={queued_count} completed={completed_count} errors={error_count} "
        f"| workers={workers} | elapsed={_format_elapsed(build_elapsed)}"
    ]

    headers = ("Idx", "Builder", "Status", "Phase", "Detail", "Time", "Result")
    rows = []
    status_order = {"running": 0, "queued": 1, "completed": 2, "error": 3}

    for idx, state in sorted(task_states.items(), key=lambda item: (status_order[item[1]["status"]], item[0])):
        status = state["status"]
        snapshot = progress_snapshots.get(idx) if progress_snapshots else None
        phase, detail = summarize_progress_snapshot(snapshot)
        if status == "running":
            time_str = _format_elapsed(now - state["started_at"]) if state["started_at"] else "--"
            result_str = "--"
            if phase == "--":
                phase = "setup"
            if detail == "--":
                detail = "launching worker"
        elif status == "queued":
            time_str = _format_elapsed(now - state["submitted_at"])
            result_str = "--"
            phase = "--"
            detail = "queued"
        else:
            time_str = _format_elapsed(state.get("duration"))
            result_str = state.get("result", "ERROR" if status == "error" else "--")
            if detail == "--":
                detail = "complete" if status == "completed" else "error"

        rows.append((
            str(idx + 1),
            _truncate(state["label"], 28),
            status,
            _truncate(phase, 14),
            _truncate(detail, detail_width),
            time_str,
            _truncate(result_str, 28),
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


def _create_forfeit(
    generator: str,
    bot_dir: Path,
    error_str: str,
    iteration: int,
    env: str,
    season_id: str,
    tournament_id: str,
) -> BotArtifact:
    """Create a forfeit BotArtifact for a failed build.

    Args:
        generator: LLM generator name.
        bot_dir: Directory for this bot's artifacts.
        error_str: Error message from the failed build.
        iteration: Iteration number.
        env: Environment identifier.
        season_id: Season identifier.
        tournament_id: Tournament identifier.

    Returns:
        BotArtifact marked as forfeit.
    """
    if "morphology" in error_str.lower() or "xml" in error_str.lower():
        forfeit_stage = "morphology"
    else:
        forfeit_stage = "controller"

    forfeit_xml = bot_dir / "robot.xml"
    forfeit_code = bot_dir / "controller.py"
    bot_dir.mkdir(parents=True, exist_ok=True)

    forfeit_xml.write_text("<!-- FORFEIT: Bot generation failed -->")
    forfeit_code.write_text("# FORFEIT: Bot generation failed\ndef policy_step(obs): return {}")

    forfeit_artifact = BotArtifact(
        name=f"{generator}_iter{iteration}",
        generator=generator,
        morphology_xml=forfeit_xml,
        controller_code=forfeit_code,
        actuator_names=[],
        morphology_score=-1.0,
        controller_score=-1.0,
        controller_verified=False,
        controller_errors=[error_str],
        forfeit=True,
        forfeit_stage=forfeit_stage,
        forfeit_error=error_str,
        metadata={
            "env": env,
            "season_id": season_id,
            "tournament_id": tournament_id,
            "iteration": iteration,
        },
    )
    forfeit_artifact.save(bot_dir)
    return forfeit_artifact


def _build_single_bot(
    generator: str,
    config_path: Path,
    constraints_path: Path,
    iteration_dir: Path,
    cfg: BuildConfig,
    arena_xml: Path,
    match_feedback: str,
    env: str,
    season_id: str,
    tournament_id: str,
    iteration: int,
    verbose: bool,
    progress_path: Optional[Path] = None,
) -> BotArtifact:
    """Build a single bot, returning a forfeit artifact on failure.

    This is the unit of work for both sequential and parallel paths.
    """
    bot_dir = iteration_dir / generator
    reporter = BuildProgressReporter(progress_path, generator) if progress_path else None

    try:
        artifact = build_bot(
            generator=generator,
            lm_config_path=config_path,
            constraints_path=constraints_path,
            output_dir=bot_dir,
            cfg=cfg,
            eval_dir=bot_dir,
            arena_xml=arena_xml,
            match_feedback=match_feedback,
            env=env,
            season_id=season_id,
            tournament_id=tournament_id,
            iteration=iteration,
            verbose=verbose,
            progress_reporter=reporter,
        )
        if reporter:
            reporter.complete(message=_artifact_summary(artifact))
        return artifact

    except Exception as exc:
        if reporter:
            reporter.fail(str(exc))
        if getattr(cfg, "raise_on_error", False):
            logger.exception("Failed to build bot for %s", generator)
            if verbose:
                print(f"[ERROR] {generator} bot generation failed; re-raising for debug")
            raise
        logger.error(f"Failed to build bot for {generator}: {exc}")
        if verbose:
            print(f"[ERROR] {generator} bot generation failed: {exc}")
        return _create_forfeit(
            generator=generator,
            bot_dir=bot_dir,
            error_str=str(exc),
            iteration=iteration,
            env=env,
            season_id=season_id,
            tournament_id=tournament_id,
        )


def generate_bots(
    llm_configs: Dict[str, Path],
    constraints_path: Path,
    arena_xml: Path,
    iteration_dir: Path,
    cfg: BuildConfig,
    *,
    env: str = "DEBUG",
    season_id: str = "season_00",
    tournament_id: str = "tournament_00",
    iteration: int = 0,
    match_feedback: str = "",
    verbose: bool = True,
    trace_progress: bool = True,
) -> Dict[str, BotArtifact]:
    """Generate a bot for each LLM.

    Args:
        llm_configs: Dictionary mapping LLM name to config path
        constraints_path: Path to BOT_CONSTRAINTS.yaml
        arena_xml: Path to arena XML for controller evaluation
        iteration_dir: Directory for this iteration's outputs
        cfg: BuildConfig with all build-phase tuneable parameters.
        env: Environment identifier ("DEBUG" or "PROD")
        season_id: Season identifier
        tournament_id: Tournament identifier within the season
        iteration: Iteration number within the tournament
        match_feedback: Feedback from previous matches (unused; always empty in this harness)
        verbose: Print progress

    Returns:
        Dictionary mapping LLM name to generated BotArtifact

    Directory structure:
        iteration_dir/                    # This is round_robin_match/bots/
            {generator}/                  # Bot artifacts (robot.xml, controller.py, controller_video.mp4)
    """
    n_parallel = cfg.n_parallel_builds
    use_parallel = n_parallel > 1 and len(llm_configs) > 1
    live_dashboard_enabled = (
        verbose
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
        and hasattr(sys.stdout, "write_ephemeral")
    )
    suppress_thread_logs = live_dashboard_enabled or not trace_progress
    previous_thread_log_setting = os.environ.get("MJARENA_SUPPRESS_THREAD_LOGS")

    if suppress_thread_logs:
        os.environ["MJARENA_SUPPRESS_THREAD_LOGS"] = "1"

    try:
        if use_parallel:
            return _build_bots_parallel(
                llm_configs=llm_configs,
                constraints_path=constraints_path,
                arena_xml=arena_xml,
                iteration_dir=iteration_dir,
                cfg=cfg,
                n_workers=min(n_parallel, len(llm_configs)),
                env=env,
                season_id=season_id,
                tournament_id=tournament_id,
                iteration=iteration,
                match_feedback=match_feedback,
                verbose=verbose,
                trace_progress=trace_progress,
            )

        bots: Dict[str, BotArtifact] = {}
        ordered_builds = list(llm_configs.items())
        progress_dir = iteration_dir / ".build_progress"
        progress_paths = {
            idx: progress_dir / f"{generator}.json"
            for idx, (generator, _) in enumerate(ordered_builds)
        }
        task_states: Dict[int, Dict[str, Any]] = {}
        state_lock = threading.Lock()
        build_t0 = time.monotonic()

        if verbose:
            submitted_at = time.monotonic()
            with state_lock:
                for idx, (generator, _) in enumerate(ordered_builds):
                    task_states[idx] = {
                        "label": generator,
                        "status": "running" if idx == 0 else "queued",
                        "submitted_at": submitted_at,
                        "started_at": submitted_at if idx == 0 else None,
                        "duration": None,
                        "result": "",
                    }

        monitor = _BuildDashboardMonitor(
            task_states=task_states,
            state_lock=state_lock,
            progress_paths=progress_paths,
            total_count=len(ordered_builds),
            workers=1,
            build_t0=build_t0,
            enabled=live_dashboard_enabled,
        )

        if verbose and live_dashboard_enabled:
            monitor.start()
        elif verbose:
            print(_render_build_dashboard(
                task_states,
                progress_snapshots={
                    idx: load_progress_snapshot(path)
                    for idx, path in progress_paths.items()
                },
                done_count=0,
                total_count=len(ordered_builds),
                workers=1,
                build_elapsed=time.monotonic() - build_t0,
            ))

        try:
            for idx, (generator, config_path) in enumerate(ordered_builds):
                if verbose and trace_progress and not live_dashboard_enabled:
                    print(f"\n{'='*60}")
                    print(f"BUILD PHASE: {generator}")
                    print(f"{'='*60}")

                bot_verbose = verbose and trace_progress and not live_dashboard_enabled
                artifact = _build_single_bot(
                    generator=generator,
                    config_path=config_path,
                    constraints_path=constraints_path,
                    iteration_dir=iteration_dir,
                    cfg=cfg,
                    arena_xml=arena_xml,
                    match_feedback=match_feedback,
                    env=env,
                    season_id=season_id,
                    tournament_id=tournament_id,
                    iteration=iteration,
                    verbose=bot_verbose,
                    progress_path=progress_paths[idx],
                )
                bots[artifact.name] = artifact

                if verbose:
                    finished_at = time.monotonic()
                    with state_lock:
                        task_states[idx]["status"] = "completed"
                        task_states[idx]["duration"] = finished_at - (
                            task_states[idx]["started_at"] or task_states[idx]["submitted_at"]
                        )
                        task_states[idx]["result"] = _artifact_summary(artifact)
                        if idx + 1 < len(ordered_builds):
                            task_states[idx + 1]["status"] = "running"
                            task_states[idx + 1]["started_at"] = finished_at

                    if not live_dashboard_enabled:
                        status_text = _artifact_summary(artifact)
                        print(f"  [{idx + 1}/{len(ordered_builds)}] {generator}: {status_text}")
                        print(_render_build_dashboard(
                            task_states,
                            progress_snapshots={
                                task_idx: load_progress_snapshot(path)
                                for task_idx, path in progress_paths.items()
                            },
                            done_count=idx + 1,
                            total_count=len(ordered_builds),
                            workers=1,
                            build_elapsed=time.monotonic() - build_t0,
                        ))
        finally:
            if verbose and live_dashboard_enabled:
                monitor.stop()

        return bots
    finally:
        if suppress_thread_logs:
            if previous_thread_log_setting is None:
                os.environ.pop("MJARENA_SUPPRESS_THREAD_LOGS", None)
            else:
                os.environ["MJARENA_SUPPRESS_THREAD_LOGS"] = previous_thread_log_setting


def _build_bots_parallel(
    llm_configs: Dict[str, Path],
    constraints_path: Path,
    arena_xml: Path,
    iteration_dir: Path,
    cfg: BuildConfig,
    n_workers: int,
    *,
    env: str,
    season_id: str,
    tournament_id: str,
    iteration: int,
    match_feedback: str,
    verbose: bool,
    trace_progress: bool,
) -> Dict[str, BotArtifact]:
    """Build bots in parallel using ProcessPoolExecutor.

    Each bot build runs in its own subprocess, giving it an isolated copy of
    dspy's module-level GLOBAL_HISTORY list. This ensures per-bot debug JSON
    (morphology_debug.json, controller_debug.json) contains ONLY that bot's
    LM calls — no cross-contamination from concurrent builds.

    All arguments are picklable (Path, str, int, bool, dataclasses). The LM
    is created inside each subprocess via configure_lm(), not passed in.

    Per-bot verbose output is suppressed in parallel mode to prevent interleaving.
    A summary line is printed as each build completes.
    """
    print(f"\n{'='*60}")
    print(f"BUILD PHASE: Generating {len(llm_configs)} bots (morphology \u2192 controller \u2192 verification)")
    print(f"{'='*60}")

    bots: Dict[str, BotArtifact] = {}
    futures = {}

    start_time = time.time()
    build_t0 = time.monotonic()
    task_states: Dict[int, Dict[str, Any]] = {}
    state_lock = threading.Lock()
    queued_indices: Deque[int] = deque()
    ordered_builds = list(llm_configs.items())
    progress_dir = iteration_dir / ".build_progress"
    progress_paths = {
        idx: progress_dir / f"{generator}.json"
        for idx, (generator, _) in enumerate(ordered_builds)
    }
    live_dashboard_enabled = (
        verbose
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
        and hasattr(sys.stdout, "write_ephemeral")
    )

    for idx, (generator, _) in enumerate(ordered_builds):
        submitted_at = time.monotonic()
        task_states[idx] = {
            "label": generator,
            "status": "running" if idx < n_workers else "queued",
            "submitted_at": submitted_at,
            "started_at": submitted_at if idx < n_workers else None,
            "duration": None,
            "result": "",
        }
        if idx >= n_workers:
            queued_indices.append(idx)

    monitor = _BuildDashboardMonitor(
        task_states=task_states,
        state_lock=state_lock,
        progress_paths=progress_paths,
        total_count=len(ordered_builds),
        workers=n_workers,
        build_t0=build_t0,
        enabled=live_dashboard_enabled,
    )

    if verbose and live_dashboard_enabled:
        monitor.start()
    elif verbose:
        for gen in llm_configs:
            print(f"  \033[33m\u23f3\033[0m {gen}: generating morphology + controller...")
        print(_render_build_dashboard(
            task_states,
            progress_snapshots={
                idx: load_progress_snapshot(path)
                for idx, path in progress_paths.items()
            },
            done_count=0,
            total_count=len(ordered_builds),
            workers=n_workers,
            build_elapsed=time.monotonic() - build_t0,
        ))

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for idx, (generator, config_path) in enumerate(ordered_builds):
                future = executor.submit(
                    _build_single_bot,
                    generator=generator,
                    config_path=config_path,
                    constraints_path=constraints_path,
                    iteration_dir=iteration_dir,
                    cfg=cfg,
                    arena_xml=arena_xml,
                    match_feedback=match_feedback,
                    env=env,
                    season_id=season_id,
                    tournament_id=tournament_id,
                    iteration=iteration,
                    verbose=False,  # suppress per-bot output to avoid interleaving
                    progress_path=progress_paths[idx],
                )
                futures[future] = idx

            done_count = 0
            pending_futures = set(futures)
            heartbeat_seconds = 2.0

            try:
                while pending_futures:
                    done, pending_futures = wait(
                        pending_futures,
                        timeout=heartbeat_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        if verbose and not live_dashboard_enabled:
                            print(_render_build_dashboard(
                                task_states,
                                progress_snapshots={
                                    task_idx: load_progress_snapshot(path)
                                    for task_idx, path in progress_paths.items()
                                },
                                done_count=done_count,
                                total_count=len(ordered_builds),
                                workers=n_workers,
                                build_elapsed=time.monotonic() - build_t0,
                            ))
                        continue

                    for future in done:
                        idx = futures[future]
                        generator = ordered_builds[idx][0]
                        finished_at = time.monotonic()
                        done_count += 1
                        try:
                            artifact = future.result()
                            bots[artifact.name] = artifact
                            with state_lock:
                                task_states[idx]["status"] = "completed"
                                task_states[idx]["duration"] = finished_at - (
                                    task_states[idx]["started_at"] or task_states[idx]["submitted_at"]
                                )
                                task_states[idx]["result"] = _artifact_summary(artifact)
                                if queued_indices:
                                    next_idx = queued_indices.popleft()
                                    task_states[next_idx]["status"] = "running"
                                    task_states[next_idx]["started_at"] = finished_at

                            if verbose and not live_dashboard_enabled:
                                elapsed = time.time() - start_time
                                if artifact.forfeit:
                                    print(f"  \033[91m\u2717\033[0m {generator}: FORFEIT ({artifact.forfeit_stage}) [{elapsed:.0f}s]")
                                else:
                                    print(
                                        f"  \033[92m\u2713\033[0m {generator}: OK  "
                                        f"morph={artifact.morphology_score:.2f}  "
                                        f"ctrl={artifact.controller_score:.2f}  [{elapsed:.0f}s]"
                                    )
                                print(_render_build_dashboard(
                                    task_states,
                                    progress_snapshots={
                                        task_idx: load_progress_snapshot(path)
                                        for task_idx, path in progress_paths.items()
                                    },
                                    done_count=done_count,
                                    total_count=len(ordered_builds),
                                    workers=n_workers,
                                    build_elapsed=time.monotonic() - build_t0,
                                ))
                        except Exception as exc:
                            with state_lock:
                                task_states[idx]["status"] = "error"
                                task_states[idx]["duration"] = finished_at - (
                                    task_states[idx]["started_at"] or task_states[idx]["submitted_at"]
                                )
                                task_states[idx]["result"] = f"ERROR: {exc}"
                                if queued_indices:
                                    next_idx = queued_indices.popleft()
                                    task_states[next_idx]["status"] = "running"
                                    task_states[next_idx]["started_at"] = finished_at

                            if getattr(cfg, "raise_on_error", False):
                                logger.exception("Build process for %s crashed", generator)
                                raise

                            logger.error("Build process for %s crashed: %s", generator, exc)
                            artifact = _create_forfeit(
                                generator=generator,
                                bot_dir=iteration_dir / generator,
                                error_str=f"Build process crashed: {type(exc).__name__}: {exc}",
                                iteration=iteration,
                                env=env,
                                season_id=season_id,
                                tournament_id=tournament_id,
                            )
                            bots[artifact.name] = artifact

                            if verbose and not live_dashboard_enabled:
                                elapsed = time.time() - start_time
                                print(f"  \033[91m\u2717\033[0m {generator}: CRASHED ({type(exc).__name__}) [{elapsed:.0f}s]")
                                print(_render_build_dashboard(
                                    task_states,
                                    progress_snapshots={
                                        task_idx: load_progress_snapshot(path)
                                        for task_idx, path in progress_paths.items()
                                    },
                                    done_count=done_count,
                                    total_count=len(ordered_builds),
                                    workers=n_workers,
                                    build_elapsed=time.monotonic() - build_t0,
                                ))
            except BrokenExecutor as exc:
                logger.error("Process pool broken: %s", exc)
                if getattr(cfg, "raise_on_error", False):
                    raise

                # Forfeit all bots that haven't been collected yet
                collected_generators = set()
                for bot in bots.values():
                    collected_generators.add(bot.generator)
                for idx, (generator, _) in enumerate(ordered_builds):
                    if generator in collected_generators:
                        continue
                    finished_at = time.monotonic()
                    with state_lock:
                        task_states[idx]["status"] = "error"
                        task_states[idx]["duration"] = finished_at - (
                            task_states[idx]["started_at"] or task_states[idx]["submitted_at"]
                        )
                        task_states[idx]["result"] = f"POOL CRASHED: {exc}"
                    artifact = _create_forfeit(
                        generator=generator,
                        bot_dir=iteration_dir / generator,
                        error_str=f"Process pool broken: {type(exc).__name__}: {exc}",
                        iteration=iteration,
                        env=env,
                        season_id=season_id,
                        tournament_id=tournament_id,
                    )
                    bots[artifact.name] = artifact
                    if verbose and not live_dashboard_enabled:
                        elapsed = time.time() - start_time
                        print(f"  \033[91m\u2717\033[0m {generator}: FORFEIT (pool crashed) [{elapsed:.0f}s]")
    finally:
        if verbose and live_dashboard_enabled:
            monitor.stop()

    if verbose:
        n_ok = sum(1 for a in bots.values() if not a.forfeit)
        n_forfeit = len(bots) - n_ok
        total_elapsed = time.time() - start_time
        print(f"\nBuild complete: {n_ok} succeeded, {n_forfeit} forfeit [{total_elapsed:.0f}s total]")

    return bots


__all__ = ["generate_bots", "BuildConfig"]
