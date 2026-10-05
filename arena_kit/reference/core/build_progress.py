"""Build-phase progress reporting helpers.

Provides a lightweight file-backed reporter so subprocess bot builders can
publish inner build state (create/critique/API/simulation) back to the main
build dashboard.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _now() -> float:
    """Return wall-clock seconds since the epoch."""
    return time.time()


def _humanize(value: str) -> str:
    """Normalize internal identifiers for dashboard display."""
    return value.replace("_", " ").strip() if value else ""


def _truncate_labels(labels: list[str], *, limit: int = 3) -> str:
    """Render a compact list of subtask labels."""
    shown = labels[:limit]
    if len(labels) > limit:
        shown.append(f"+{len(labels) - limit}")
    return ",".join(shown)


def _format_short_duration(seconds: Optional[float]) -> str:
    """Format a short ETA/elapsed duration."""
    if seconds is None:
        return "--"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class BuildProgressReporter:
    """Thread-safe, file-backed progress reporter for one bot build."""

    def __init__(self, path: Path, bot: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        now = _now()
        self._state: Dict[str, Any] = {
            "bot": bot,
            "status": "running",
            "phase": "setup",
            "stage": "launching",
            "wait": "",
            "message": "launching build",
            "started_at": now,
            "updated_at": now,
            "subtasks": {},
        }
        self._write_locked()

    def update(
        self,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        stage: Optional[str] = None,
        wait: Optional[str] = None,
        message: Optional[str] = None,
        clear_subtasks: bool = False,
    ) -> None:
        """Update top-level progress state."""
        with self._lock:
            if status is not None:
                self._state["status"] = status
            if phase is not None:
                self._state["phase"] = phase
            if stage is not None:
                self._state["stage"] = stage
            if wait is not None:
                self._state["wait"] = wait
            if message is not None:
                self._state["message"] = message
            if clear_subtasks:
                self._state["subtasks"] = {}
            self._state["updated_at"] = _now()
            self._write_locked()

    def update_subtask(
        self,
        key: str,
        *,
        label: Optional[str] = None,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        stage: Optional[str] = None,
        wait: Optional[str] = None,
        message: Optional[str] = None,
    ) -> None:
        """Update a named child task, such as a candidate lineage."""
        with self._lock:
            now = _now()
            subtask = self._state["subtasks"].get(key, {
                "label": label or key,
                "status": "running",
                "phase": "",
                "stage": "",
                "wait": "",
                "message": "",
                "started_at": now,
                "updated_at": now,
            })
            if label is not None:
                subtask["label"] = label
            if status is not None:
                subtask["status"] = status
            if phase is not None:
                subtask["phase"] = phase
            if stage is not None:
                subtask["stage"] = stage
            if wait is not None:
                subtask["wait"] = wait
            if message is not None:
                subtask["message"] = message
            subtask["updated_at"] = now
            self._state["subtasks"][key] = subtask
            self._state["updated_at"] = now
            self._write_locked()

    def complete_subtask(self, key: str, *, message: Optional[str] = None) -> None:
        """Mark a child task as completed."""
        self.update_subtask(key, status="completed", wait="", message=message)

    def fail_subtask(self, key: str, *, message: str) -> None:
        """Mark a child task as failed."""
        self.update_subtask(key, status="error", wait="", message=message)

    def complete(self, *, message: str = "") -> None:
        """Mark the overall build as completed."""
        self.update(
            status="completed",
            stage="complete",
            wait="",
            message=message,
            clear_subtasks=True,
        )

    def fail(self, message: str) -> None:
        """Mark the overall build as failed."""
        self.update(
            status="error",
            stage="error",
            wait="",
            message=message,
            clear_subtasks=False,
        )

    def _write_locked(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self._state, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)


def load_progress_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    """Load a progress JSON snapshot if it exists and is valid."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def summarize_progress_snapshot(snapshot: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Summarize a snapshot for table display as (phase, detail)."""
    if not snapshot:
        return "--", "--"

    subtasks = snapshot.get("subtasks", {})
    active_subtasks = [
        subtask for subtask in subtasks.values()
        if subtask.get("status") not in {"completed", "error"}
    ]

    if active_subtasks:
        groups: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
        for subtask in active_subtasks:
            key = (
                subtask.get("phase", ""),
                subtask.get("stage", ""),
                subtask.get("wait", ""),
            )
            groups.setdefault(key, []).append(subtask)

        sorted_groups = sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), item[0][0], item[0][1], item[0][2]),
        )
        lead_phase = _humanize(sorted_groups[0][0][0]) or _humanize(snapshot.get("phase", "")) or "--"
        detail_parts = []
        for (phase, stage, wait), subtasks in sorted_groups[:2]:
            labels = [subtask.get("label", "task") for subtask in subtasks]
            descriptor = []
            if wait:
                descriptor.append(f"waiting {_humanize(wait)}")
            if stage:
                descriptor.append(_humanize(stage))
            base = " | ".join(descriptor) if descriptor else "running"
            part = f"{base} [{_truncate_labels(labels)}]"
            messages = []
            for subtask in sorted(subtasks, key=lambda item: item.get("label", ""))[:2]:
                message = _humanize(str(subtask.get("message", "")))
                if not message:
                    continue
                if len(subtasks) == 1:
                    messages.append(message)
                else:
                    messages.append(f"{subtask.get('label', 'task')}: {message}")
            if messages:
                part = f"{part} | {' ; '.join(messages)}"
            detail_parts.append(part)
        return lead_phase, " ; ".join(detail_parts)

    phase = _humanize(snapshot.get("phase", "")) or "--"
    descriptor = []
    wait = snapshot.get("wait", "")
    stage = snapshot.get("stage", "")
    if wait:
        descriptor.append(f"waiting {_humanize(wait)}")
    if stage:
        descriptor.append(_humanize(stage))
    message = _humanize(str(snapshot.get("message", "")))
    if message:
        descriptor.append(message)
    return phase, " | ".join(part for part in descriptor if part) or "--"


def make_seed_progress_listener(
    progress_reporter: BuildProgressReporter,
    *,
    phase: str,
    stage: str,
    total_seeds: int,
    progress_key: Optional[str] = None,
) -> Callable[[Dict[str, Any]], None]:
    """Create a listener that aggregates per-seed simulation progress."""
    lock = threading.Lock()
    seed_states: Dict[int, Dict[str, Any]] = {}

    def _publish(message: str) -> None:
        if progress_key:
            progress_reporter.update_subtask(
                progress_key,
                phase=phase,
                stage=stage,
                wait="simulation",
                message=message,
            )
        else:
            progress_reporter.update(
                phase=phase,
                stage=stage,
                wait="simulation",
                message=message,
            )

    def _render_message() -> str:
        completed = sum(
            1 for state in seed_states.values()
            if state.get("status") in {"completed", "failed"}
        )
        running = sorted(
            [
                (seed, state)
                for seed, state in seed_states.items()
                if state.get("status") == "running"
            ],
            key=lambda item: item[0],
        )

        parts = [f"{completed}/{total_seeds} done"]
        if running:
            running_parts = []
            for seed, state in running[:2]:
                step = state.get("step")
                max_steps = state.get("max_steps")
                steps_left = state.get("steps_left")
                eta_sec = state.get("eta_sec")
                if step is not None and max_steps:
                    seed_text = f"s{seed} {step}/{max_steps}"
                    if steps_left is not None:
                        seed_text += f" left {steps_left}"
                    if eta_sec is not None:
                        seed_text += f" eta {_format_short_duration(eta_sec)}"
                else:
                    seed_text = f"s{seed} starting"
                running_parts.append(seed_text)
            if len(running) > 2:
                running_parts.append(f"+{len(running) - 2} more")
            parts.append("; ".join(running_parts))
        elif completed < total_seeds:
            parts.append("starting")

        return " | ".join(parts)

    def listener(event: Dict[str, Any]) -> None:
        seed = int(event.get("seed", -1))
        if seed < 0:
            return

        with lock:
            state = seed_states.setdefault(seed, {
                "status": "queued",
                "step": 0,
                "max_steps": None,
                "steps_left": None,
                "eta_sec": None,
            })
            event_type = event.get("type")
            if event_type == "seed_started":
                state["status"] = "running"
                state["step"] = 0
                state["max_steps"] = event.get("max_steps")
                state["steps_left"] = event.get("max_steps")
                state["eta_sec"] = None
            elif event_type in {"progress", "done"}:
                state["status"] = "completed" if event_type == "done" else "running"
                state["step"] = event.get("step", state.get("step"))
                state["max_steps"] = event.get("max_steps", state.get("max_steps"))
                state["steps_left"] = event.get("steps_left", state.get("steps_left"))
                state["eta_sec"] = event.get("eta_sec", state.get("eta_sec"))
            elif event_type == "seed_done":
                state["status"] = "completed"
                state["step"] = event.get("step", state.get("step"))
                state["steps_left"] = 0
                state["eta_sec"] = 0.0
            elif event_type == "seed_failed":
                state["status"] = "failed"
                state["eta_sec"] = None

            _publish(_render_message())

    return listener
