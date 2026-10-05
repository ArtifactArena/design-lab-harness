"""Core harness logic, shared by run_match.py, agent.py and the tests.

Everything here reuses the LIVE mjarena package — no physics/validation is
reimplemented:
  * verify  -> run_unified_env() (morphology -> controller -> qualification)
  * match   -> compose_sumo_model() + create_match_runner() + run_seeds()
               + save_matchup_to_disk()

The agent's bot is always RED; the opponent is always BLUE.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import shutil
import signal
import threading
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import config as _cfg


@contextlib.contextmanager
def _time_limit(seconds: int):
    """Best-effort wall-clock guard (SIGALRM) so a runaway controller/physics can't
    hang the run. Only active on the main thread on Unix; otherwise a no-op."""
    active = (seconds and hasattr(signal, "SIGALRM")
              and threading.current_thread() is threading.main_thread())
    if not active:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(
            f"match exceeded {seconds}s — controller or physics too slow "
            "(e.g. an expensive policy_step). Simplify and retry.")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# --------------------------------------------------------------------------- #
# Lazy mjarena imports (kept inside functions to avoid import-order surprises
# and so that importing this module is cheap for the tests that only need it).
# --------------------------------------------------------------------------- #
def _mj():
    import sys
    if str(_cfg.REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_cfg.REPO_ROOT))
    from mjarena.two_stage.match_config import resolve_match_config
    from mjarena.dspy_core import (
        create_match_runner,
        get_actuator_names,
        get_actuator_names_from_xml_string,
    )
    from mjarena.envs.sumo import compose_sumo_model
    from mjarena.eval.match_runner import run_seeds, save_matchup_to_disk
    from mjarena.policy_spec import PolicySpec
    from mjarena.design_shop import validate_morphology, ModelValidationConfig
    from mjarena.design_shop.pipelines.unified_env import run_unified_env
    return dict(
        resolve_match_config=resolve_match_config,
        create_match_runner=create_match_runner,
        get_actuator_names=get_actuator_names,
        get_actuator_names_from_xml_string=get_actuator_names_from_xml_string,
        compose_sumo_model=compose_sumo_model,
        run_seeds=run_seeds,
        save_matchup_to_disk=save_matchup_to_disk,
        PolicySpec=PolicySpec,
        validate_morphology=validate_morphology,
        ModelValidationConfig=ModelValidationConfig,
        run_unified_env=run_unified_env,
    )


def clean_feedback(feedback: str) -> str:
    """Omit built-in combat scores from the model's feedback."""
    import re
    lines = []
    metric_line = (r"\s*(?:\[Seed \d+\] Combat score:|Qualification score:|"
                   r"(?:engagement|dominant_contact|displacement|destabilization|self_stability):)")
    for line in feedback.splitlines():
        if "Detailed replay attached" in line or re.match(metric_line, line):
            continue
        lines.append(re.sub(r"\s+\(combat=[^)]*\)", "", line).rstrip())
    return "\n".join(lines).strip()


def strip_metrics(value):
    excluded = {"combat_metrics", "combat_metrics_b", "combat_score", "seed_scores",
                "seed_score", "mean_score", "median_score"}
    if isinstance(value, dict):
        return {key: strip_metrics(item) for key, item in value.items() if key not in excluded}
    if isinstance(value, list):
        return [strip_metrics(item) for item in value]
    return value


def get_match_config():
    """Resolve the match config and make its paths absolute (repo-relative in YAML)."""
    from types import SimpleNamespace
    M = _mj()
    # The upstream resolver interprets rule paths relative to cwd. Tool workers
    # run from /workspace, so resolve under the frozen engine root temporarily.
    import os
    previous = Path.cwd()
    try:
        os.chdir(_cfg.REPO_ROOT)
        cfg = M["resolve_match_config"](_cfg.REPO_ROOT, _cfg.TOURNAMENT_CONFIG)
    finally:
        os.chdir(previous)

    def _abs(p: Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (_cfg.REPO_ROOT / p)

    return SimpleNamespace(
        arena_xml=_abs(cfg.arena_xml),
        constraints_path=_abs(cfg.constraints_path),
        physics_mode=cfg.physics_mode,
        score_function=cfg.score_function,
        contact_fidelity=cfg.contact_fidelity,
        size_limits=cfg.size_limits,
        match_time=float(cfg.match_time),
        inactivity_timeout=cfg.inactivity_timeout,
        inactivity_min_displacement=cfg.inactivity_min_displacement,
    )


def _new_match_id() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:6]}"


def _prepare_xml(M, xml_text: str, cfg) -> Dict[str, Any]:
    """Run morphology validation; return processed xml when it passes.

    Authored bots (material=) need the morphology transform; already-processed
    artifacts (e.g. the stationary block) compile as-is, so on failure we fall
    back to the raw text rather than rejecting a valid compiled model.
    """
    mcfg = M["ModelValidationConfig"](
        constraints_yaml_path=cfg.constraints_path,
        physics_mode=cfg.physics_mode,
    )
    res = M["validate_morphology"](xml_text, mcfg)
    if res.passed and res.processed_xml:
        return {"passed": True, "xml": res.processed_xml, "feedback": res.feedback}
    return {"passed": False, "xml": xml_text, "feedback": res.feedback}


# --------------------------------------------------------------------------- #
# Bot refs — "draft" (workspace), "stationary" (the box), "<name>" (saved bot)
# --------------------------------------------------------------------------- #
def is_safe_name(name: str) -> bool:
    """A flat library name/id: no path separators, no parent refs, not absolute.

    Keeps bot names and match ids from escaping the run's bots/ and matches/ dirs
    (the sandbox boundary) — e.g. '../../other-run/bots/x'.
    """
    return bool(name) and isinstance(name, str) and "/" not in name \
        and "\\" not in name and ".." not in name and not name.startswith(".")


def resolve_bot_ref(ref: str, workspace: Path) -> SimpleNamespace:
    """Resolve a bot ref to its files. ctrl_path is None for the stationary box."""
    ws = Path(workspace)
    if ref in ("draft", "current", None):
        return SimpleNamespace(kind="draft", label="draft",
                               xml_path=ws / "robot.xml", ctrl_path=ws / "controller.py")
    if ref in ("stationary", "box"):
        opp = ws / _cfg.DEFAULT_OPPONENT_XML
        if not opp.is_file():
            opp = Path(_cfg.DEFAULT_OPPONENT_XML)
        return SimpleNamespace(kind="stationary", label="stationary",
                               xml_path=opp, ctrl_path=None)
    if not is_safe_name(ref):
        raise ValueError(f"invalid bot ref: {ref!r} (names are flat — no '/', '..')")
    bot = ws / "bots" / ref
    if not (bot / "robot.xml").is_file():
        raise ValueError(f"unknown bot ref: {ref!r} (no bots/{ref}/robot.xml)")
    return SimpleNamespace(kind="saved", label=ref,
                           xml_path=bot / "robot.xml", ctrl_path=bot / "controller.py")


def _stationary_match_fn(M, cfg, processed_xml_path, out_dir, max_steps=None):
    """Build a match runner: red (processed) vs the zero-policy stationary block.

    Same game as the build harness's qualification: QUALIFY_MATCH_TIME seconds per seed
    (20 s) with the inactivity rule armed; the block is exempt. max_steps is only for
    probes that need a few steps of observations.
    """
    from mjarena.core.qualification_block import write_qualification_block
    stat = write_qualification_block(Path(out_dir) / "block.xml", cfg.constraints_path, cfg.physics_mode)
    blue_spec = M["PolicySpec"].zero(M["get_actuator_names"](stat))
    qdir = Path(out_dir) / "qualification"
    qdir.mkdir(parents=True, exist_ok=True)
    composed = qdir / "composed.xml"
    has_palette = (cfg.constraints_path.parent / "materials_store.yaml").exists()
    M["compose_sumo_model"](
        env_xml=str(cfg.arena_xml), robot_red_xml=str(processed_xml_path),
        robot_blue_xml=str(stat), out_path=str(composed),
        randomize_spawn_3d=(cfg.physics_mode == "3d"), use_material_palette=has_palette,
    )
    length = {"max_steps": int(max_steps)} if max_steps is not None else {"match_time": float(_cfg.QUALIFY_MATCH_TIME)}
    return M["create_match_runner"](
        composed_xml=composed, blue_policy_spec=blue_spec, out_dir=qdir,
        quiet=True, save_video_seeds=0, score_function=cfg.score_function,
        contact_fidelity=cfg.contact_fidelity,
        inactivity_timeout_seconds=cfg.inactivity_timeout,
        inactivity_min_displacement=cfg.inactivity_min_displacement,
        size_limits=cfg.size_limits,
        max_obs_lookback=20, max_action_lookback=20,
        inactivity_exempt_prefixes=["blue_"],
        **length,
    )


# --------------------------------------------------------------------------- #
# qualification — validation (morphology + controller static) + qualify-vs-box
# --------------------------------------------------------------------------- #
def run_qualification_core(robot_xml: str, controller_code: str, out_dir: Path,
                           n_rollouts: int = None) -> Dict[str, Any]:
    """Return {ok, validation_passed, qualification_passed, feedback}.

    validation = morphology passed AND all static controller checks passed.
    qualification = lost no seed vs the stationary box. Never raises.
    """
    n_rollouts = _cfg.VERIFY_ROLLOUTS if n_rollouts is None else n_rollouts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        M = _mj()
        cfg = get_match_config()
        from mjarena.design_shop import (  # noqa: E402
            ModelValidationConfig, validate_controller, validate_morphology,
        )
        mcfg = ModelValidationConfig(constraints_yaml_path=cfg.constraints_path,
                                     physics_mode=cfg.physics_mode)
        morph = validate_morphology(robot_xml, mcfg)
        if not morph.passed:
            return {"ok": True, "validation_passed": False, "qualification_passed": False,
                    "feedback": morph.feedback}
        actuators = M["get_actuator_names_from_xml_string"](morph.processed_xml)
        proc_path = out_dir / "qual_robot.xml"  # in the per-run work dir, not system temp
        proc_path.write_text(morph.processed_xml)
        match_fn = _stationary_match_fn(M, cfg, proc_path, out_dir)
        with _time_limit(_cfg.MATCH_TIMEOUT):
            ctrl = validate_controller(controller_code, actuators,
                                       qualification_match_fn=match_fn, n_rollouts=n_rollouts,
                                       processed_robot_xml=morph.processed_xml,
                                       n_parallel_seeds=min(3, n_rollouts), seed_parallel_backend="process",
                                       output_dir=out_dir)
        static_steps = [s for s in ctrl.step_results if s is not ctrl.qualification]
        return {
            "ok": True,
            "validation_passed": bool(morph.passed and all(s.passed for s in static_steps)),
            "qualification_passed": bool(ctrl.qualification and ctrl.qualification.passed),
            "feedback": clean_feedback(morph.feedback + "\n" + ctrl.feedback),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


def run_verification_core(robot_xml: str, controller_code: str) -> Dict[str, Any]:
    """VERIFICATION = validation only: morphology + STATIC controller checks (does the
    MJCF compile within constraints; does policy_step return the right actuator keys in
    range). It does NOT run the box match — that is qualification, a separate step.

    Returns {ok, verification_passed, morphology_passed, controller_passed, feedback}.
    Cheap (no simulation) and never raises.
    """
    try:
        M = _mj()
        cfg = get_match_config()
        from mjarena.design_shop import (  # noqa: E402
            ModelValidationConfig, validate_controller, validate_morphology,
        )
        mcfg = ModelValidationConfig(constraints_yaml_path=cfg.constraints_path,
                                     physics_mode=cfg.physics_mode)
        morph = validate_morphology(robot_xml, mcfg)
        if not morph.passed:
            return {"ok": True, "verification_passed": False, "morphology_passed": False,
                    "controller_passed": None, "feedback": morph.feedback}
        actuators = M["get_actuator_names_from_xml_string"](morph.processed_xml)
        # qualification_match_fn=None -> only the static checks run, no box match. But
        # those checks CALL policy_step, so guard the wall clock here too — otherwise a
        # runaway controller hangs verification (and thus run_match's pre-match gate).
        with _time_limit(_cfg.MATCH_TIMEOUT):
            ctrl = validate_controller(controller_code, actuators, qualification_match_fn=None,
                                       processed_robot_xml=morph.processed_xml)
        ctrl_passed = all(s.passed for s in ctrl.step_results)
        return {
            "ok": True,
            "verification_passed": bool(morph.passed and ctrl_passed),
            "morphology_passed": bool(morph.passed),
            "controller_passed": bool(ctrl_passed),
            "feedback": clean_feedback(morph.feedback + "\n" + ctrl.feedback),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# match — red bot vs blue bot (both bot refs); saves a match-library entry
# --------------------------------------------------------------------------- #
def _g(rec, key, default=None):
    """Field access that works for a GameRecord object OR a match_data.json dict."""
    return rec.get(key, default) if isinstance(rec, dict) else getattr(rec, key, default)


def _seed_summary(rec) -> Dict[str, Any]:
    """Compact per-seed result (no telemetry, no built-in metrics) — the 'full output'."""
    return {
        "seed": _g(rec, "seed"), "winner": _g(rec, "winner"),
        "termination_reason": _g(rec, "termination_reason"), "num_steps": _g(rec, "num_steps"),
    }


def _seed_view(rec, step_interval: int) -> Dict[str, Any]:
    """Downsample one seed (GameRecord object or dict) to a strided telemetry view."""
    n = _g(rec, "num_steps", 0) or 0
    idx = list(range(0, n, max(1, step_interval)))
    if idx and idx[-1] != n - 1:
        idx.append(n - 1)

    def pick(key):
        arr = _g(rec, key) or []
        return [arr[i] for i in idx if i < len(arr)]

    # Raw strided telemetry only (~50x) — NO built-in combat metrics. The model writes
    # its own analytics over this via analyze_match.
    return {
        "seed": _g(rec, "seed"), "winner": _g(rec, "winner"),
        "termination_reason": _g(rec, "termination_reason"), "num_steps": n,
        "sampled_steps": len(idx),
        "red_positions": pick("red_positions"), "blue_positions": pick("blue_positions"),
        "red_orientations": pick("red_orientations"), "blue_orientations": pick("blue_orientations"),
        "red_velocities": pick("red_velocities"), "blue_velocities": pick("blue_velocities"),
        "distances_to_opponent": pick("distances_to_opponent"),
        "red_edge_distances": pick("red_edge_distances"),
        "blue_edge_distances": pick("blue_edge_distances"),
        "contacts": pick("contacts"), "contact_forces": pick("contact_forces"),
        "red_tipping": pick("red_tipping"), "blue_tipping": pick("blue_tipping"),
        "red_actions": pick("red_actions"), "blue_actions": pick("blue_actions"),
        # Seconds since each bot's COM was last 0.5 m away (the inactivity rule's timer).
        "red_inactivity_timers": pick("red_inactivity_timers"),
        "blue_inactivity_timers": pick("blue_inactivity_timers"),
    }


def _match_view(match_id, red, blue, recs, match_dir, match_data) -> Dict[str, Any]:
    """Build the run_match-shaped view from a list of seed records (objects or dicts).

    FULL per-seed results under "seeds" (computed on the full sim, no telemetry) +
    DOWNSAMPLED telemetry for the informative seed(s) under "replay".
    """
    wins = sum(1 for r in recs if _g(r, "winner") == "red")
    losses = sum(1 for r in recs if _g(r, "winner") == "blue")
    draws = len(recs) - wins - losses
    outcome = "win" if wins > losses else ("loss" if losses > wins else "draw")

    losing = [r for r in recs if _g(r, "winner") == "blue"]
    if losing:
        sel, reason = losing, "losses"
    elif len(recs) == 1:
        sel, reason = recs, "single_seed"
    else:
        # no built-in score to rank by — show the longest (most informative) seed
        sel, reason = [max(recs, key=lambda r: _g(r, "num_steps", 0) or 0)], "longest_seed"

    step_interval = max(1, round(100.0 / 2.0))  # ~50x stride (replay_hz=2, control_hz=100)
    return {
        "ok": True, "match_id": match_id, "red": red, "blue": blue,
        "outcome": outcome, "wins": wins, "losses": losses, "draws": draws,
        "n_seeds": len(recs),
        "seeds": [_seed_summary(r) for r in recs],
        "replay": {"selection_reason": reason, "n_selected": len(sel), "replay_hz": 2.0,
                   "seeds": [_seed_view(r, step_interval) for r in sel]},
        "match_dir": str(Path(match_dir).resolve()),
        "match_data": str(Path(match_data).resolve()),
    }


def run_match_core(workspace: Path, red_ref: str = "draft", blue_ref: str = "stationary",
                   out_base: Path = None, n_seeds: int = None) -> Dict[str, Any]:
    """Run red (a bot ref) vs blue (a bot ref); save a match-library entry.

    Writes <out_base>/<match_id>/{metadata/{match_data,match_result}.json,
    composed.xml, red_processed.xml, blue.xml} + bot_artifact/. Returns the compact
    downsampled view + a unique match_id (no full telemetry inline). Never raises.
    Same resolved bot on both sides == self-play.
    """
    n_seeds = _cfg.MATCH_SEEDS if n_seeds is None else n_seeds
    if n_seeds < 1:
        return {"ok": False, "error": f"n_seeds must be >= 1 (got {n_seeds})"}
    workspace = Path(workspace)
    out_base = Path(out_base) if out_base is not None else (workspace / "matches")
    match_id = _new_match_id()
    match_dir = out_base / match_id
    meta_dir = match_dir / "metadata"
    art_dir = match_dir / "bot_artifact"
    try:
        M = _mj()
        cfg = get_match_config()
        has_palette = (cfg.constraints_path.parent / "materials_store.yaml").exists()
        red = resolve_bot_ref(red_ref, workspace)
        blue = resolve_bot_ref(blue_ref, workspace)

        red_xml = red.xml_path.read_text() if red.xml_path.is_file() else ""
        red_ctrl = red.ctrl_path.read_text() if (red.ctrl_path and red.ctrl_path.is_file()) else ""
        if not red_xml.strip() or not red_ctrl.strip():
            raise ValueError(f"red ref {red_ref!r}: robot.xml/controller.py empty — write the draft first")
        blue_xml = blue.xml_path.read_text() if blue.xml_path.is_file() else ""
        blue_ctrl = blue.ctrl_path.read_text() if (blue.ctrl_path and blue.ctrl_path.is_file()) else ""

        # Gate: both authored bots must pass VERIFICATION (validation only — morphology
        # + static controller checks) before a match runs. Qualification (beat-the-box)
        # is NOT required, so two unqualified bots can fight. The stationary box
        # (ctrl_path is None) is pre-built and exempt.
        ver = {}
        rv = run_verification_core(red_xml, red_ctrl)
        if not (rv.get("ok") and rv.get("verification_passed")):
            ver[red.label] = rv
        if blue.ctrl_path is not None:
            bv = run_verification_core(blue_xml, blue_ctrl)
            if not (bv.get("ok") and bv.get("verification_passed")):
                ver[blue.label] = bv
        if ver:
            return {"ok": False, "error": "verification failed before run_match — fix the "
                    "bot(s) below. run_match verifies both bots (validation only); "
                    "qualification is a separate does_bot_qualify call.",
                    "verification": ver}

        meta_dir.mkdir(parents=True, exist_ok=True)
        art_dir.mkdir(parents=True, exist_ok=True)

        # RED: must pass morphology so it composes/qualifies cleanly.
        rp = _prepare_xml(M, red_xml, cfg)
        if not rp["passed"]:
            raise ValueError(f"red ({red_ref}) morphology failed:\n{rp['feedback']}")
        red_actuators = M["get_actuator_names_from_xml_string"](rp["xml"])
        red_xml_file = meta_dir / "red_processed.xml"
        red_xml_file.write_text(rp["xml"])

        # BLUE: authored bots get processed; compiled artifacts (block) fall back to raw.
        bp = _prepare_xml(M, blue.xml_path.read_text(), cfg)
        blue_xml_file = meta_dir / "blue.xml"
        if blue.kind == "stationary":
            from mjarena.core.qualification_block import qualification_block_xml
            bp["xml"] = qualification_block_xml(cfg.constraints_path, cfg.physics_mode)
        blue_xml_file.write_text(bp["xml"])
        blue_actuators = M["get_actuator_names_from_xml_string"](bp["xml"])

        blue_ctrl = blue.ctrl_path.read_text() if (blue.ctrl_path and blue.ctrl_path.is_file()) else ""
        if blue_ctrl.strip() and blue_actuators:
            blue_spec = M["PolicySpec"].controller_code(blue_ctrl, blue_actuators)
            exempt = None  # an authored opponent must keep moving too
        else:
            blue_spec = M["PolicySpec"].zero(blue_actuators)
            exempt = ["blue_"]  # stationary dummy is exempt, mirroring qualification

        composed = meta_dir / "composed.xml"
        M["compose_sumo_model"](
            env_xml=str(cfg.arena_xml), robot_red_xml=str(red_xml_file),
            robot_blue_xml=str(blue_xml_file), out_path=str(composed),
            randomize_spawn_3d=(cfg.physics_mode == "3d"), use_material_palette=has_palette,
        )
        # Same game as the tournament: match_time seconds per seed (300 s) with the
        # inactivity rule armed (tournament.match in the config).
        match_fn = M["create_match_runner"](
            composed_xml=composed, blue_policy_spec=blue_spec, out_dir=meta_dir,
            match_time=float(cfg.match_time), quiet=True, save_video_seeds=0,
            score_function=cfg.score_function, contact_fidelity=cfg.contact_fidelity,
            inactivity_timeout_seconds=cfg.inactivity_timeout,
            inactivity_min_displacement=cfg.inactivity_min_displacement,
            size_limits=cfg.size_limits,
            max_obs_lookback=20, max_action_lookback=20,
            inactivity_exempt_prefixes=exempt,
        )
        red_spec = M["PolicySpec"].controller_code(red_ctrl, red_actuators)
        with _time_limit(_cfg.MATCH_TIMEOUT):
            matchup = M["run_seeds"](run_match_fn=match_fn, policy_spec=red_spec,
                                     policy_callable=red_spec.build_callable(),
                                     n_seeds=n_seeds, n_parallel=min(3, n_seeds),
                                     parallel_backend="process")

        M["save_matchup_to_disk"](matchup, meta_dir, tool_name="harness_match",
                                  red_bot=red.label, blue_bot=blue.label)
        for filename in ("match_data.json", "match_result.json"):
            path = meta_dir / filename
            path.write_text(json.dumps(strip_metrics(json.loads(path.read_text()))))
        shutil.copy(red.xml_path, art_dir / "robot.xml")
        if red.ctrl_path:
            shutil.copy(red.ctrl_path, art_dir / "controller.py")

        return _match_view(match_id, red.label, blue.label, list(matchup.game_records),
                           match_dir, meta_dir / "match_data.json")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# match catalogue + model-written analytics over match_data.json
# --------------------------------------------------------------------------- #
def list_matches_core(matches_dir: Path):
    """Return one compact record per saved match (reads each match_result.json)."""
    matches_dir = Path(matches_dir)
    rows = []
    if not matches_dir.is_dir():
        return rows
    for d in sorted(matches_dir.iterdir()):
        mr = d / "metadata" / "match_result.json"
        if not mr.is_file():
            continue
        data = json.loads(mr.read_text())
        ms = data.get("matches", [])
        wins = sum(1 for m in ms if m.get("winner") == "red")
        losses = sum(1 for m in ms if m.get("winner") == "blue")
        rows.append({
            "match_id": d.name, "red": data.get("red_bot"), "blue": data.get("blue_bot"),
            "outcome": "win" if wins > losses else ("loss" if losses > wins else "draw"),
            "wins": wins, "losses": losses, "draws": len(ms) - wins - losses,
            "match_dir": str(d.resolve()),
        })
    return rows


def get_match_core(matches_dir: Path, match_id: str) -> Dict[str, Any]:
    """Rehydrate a past match's run_match-shaped view (full per-seed results +
    downsampled replay) from disk — no re-run. Never raises."""
    if not is_safe_name(match_id):
        return {"ok": False, "error": f"invalid match_id: {match_id!r}"}
    meta = Path(matches_dir) / match_id / "metadata"
    md = meta / "match_data.json"
    if not md.is_file():
        return {"ok": False, "error": f"no match {match_id!r} (no match_data.json)"}
    try:
        data = json.loads(md.read_text())
        result = json.loads((meta / "match_result.json").read_text()) \
            if (meta / "match_result.json").is_file() else {}
        seed_keys = sorted((k for k in data if k.startswith("seed_")),
                           key=lambda k: int(k.split("_")[1]))
        recs = [data[k] for k in seed_keys]
        if not recs:
            return {"ok": False, "error": f"match {match_id!r} has no seed records"}
        return _match_view(match_id, result.get("red_bot", "?"), result.get("blue_bot", "?"),
                           recs, Path(matches_dir) / match_id, md)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


# Shown to the model when its analyze() throws — the shape of match_data, so it can fix
# its code instead of guessing keys (the #1 cause of analyze_match failures).
_MATCH_DATA_HINT = (
    "match_data = {'seed_0': {...}, 'seed_1': {...}, ...} (one dict per seed). "
    "Each seed dict has: winner ('red'|'blue'|'tie'), termination_reason, num_steps, "
    "control_dt; per-step lists (length num_steps): red_positions/blue_positions "
    "([[x,y,z],...]), red_velocities/blue_velocities, red_orientations/blue_orientations "
    "({'yaw','pitch','roll'}), distances_to_opponent, red_edge_distances/blue_edge_distances "
    "(>0 = inside ring), contacts (bool), contact_forces, red_tipping/blue_tipping (0..1), "
    "red_actions/blue_actions (list of {actuator_name: value}), "
    "red_inactivity_timers/blue_inactivity_timers (seconds since each bot's COM was last "
    "0.5 m away; a bot loses at 10), qpos, and ring_radius. "
    "Iterate seeds with [k for k in match_data if k.startswith('seed_')]."
)


def analyze_match_core(matches_dir: Path, match_id: str, code: str, timeout: int = 120):
    """Run the model's `def analyze(match_data)` over a match's match_data.json.

    Executes in a subprocess (real env, timeout); returns {ok, result} or a
    structured {ok: False, error[, traceback]}. The heavy replay never enters
    the caller's context — only the analytics the model's code returns.
    """
    import os
    import subprocess
    import sys
    mdir = Path(matches_dir)
    # "latest"/"last"/"recent" -> the most recent match (ids are timestamps).
    if match_id in ("latest", "last", "recent"):
        cand = sorted(d.name for d in mdir.iterdir()
                      if (d / "metadata" / "match_data.json").is_file()) if mdir.is_dir() else []
        if not cand:
            return {"ok": False, "error": "no matches yet — run_match first"}
        match_id = cand[-1]
    if not is_safe_name(match_id):
        return {"ok": False, "error": f"invalid match_id: {match_id!r}"}
    md = mdir / match_id / "metadata" / "match_data.json"
    if not md.is_file():
        return {"ok": False, "error": f"no match_data.json for match_id {match_id!r} "
                f"(use list_matches, or match_id='latest')"}
    shim = (
        "import json, sys, types\n"
        f"_data = json.load(open({str(md)!r}))\n"
        # drop the engine's built-in combat metrics: the model derives its own here
        "for _k in [k for k in _data if k.startswith('seed_')]:\n"
        "    _data[_k].pop('combat_metrics', None)\n"
        f"{code}\n"
        # run whatever function the model defined — the NAME doesn't matter: prefer one
        # called 'analyze', else the last top-level def. Feed its return value straight back.
        "_fns = [(_n, _o) for _n, _o in list(globals().items())\n"
        "        if isinstance(_o, types.FunctionType) and getattr(_o, '__module__', None) == '__main__']\n"
        "_fn = next((_o for _n, _o in _fns if _n == 'analyze'), None) or (_fns[-1][1] if _fns else None)\n"
        "if _fn is None:\n"
        "    raise SystemExit('analyze_match: no function defined — write e.g. def analyze(match_data): ...')\n"
        "_out = _fn(_data)\n"
        "try:\n"
        "    _res = json.dumps(_out)\n"
        "except TypeError:\n"
        "    _res = json.dumps(str(_out))\n"  # non-JSON return -> feed back its string form
        "print('__RESULT__' + _res)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_cfg.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run([sys.executable, "-c", shim], env=env, timeout=timeout,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"analyze timed out ({timeout}s)"}
    if proc.returncode != 0:
        tb = proc.stderr.strip()
        # surface the actual exception (last traceback line) so the model sees WHY,
        # not just "it raised" — plus the full traceback and the data schema.
        last = next((ln.strip() for ln in reversed(tb.splitlines()) if ln.strip()),
                    "analyze() raised")
        return {"ok": False, "error": f"analyze() raised — {last}",
                "traceback": tb[-4000:], "match_data_schema": _MATCH_DATA_HINT}
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            try:
                result = json.loads(line[len("__RESULT__"):])
            except json.JSONDecodeError as exc:
                return {"ok": False, "error": f"analyze() returned non-JSON: {exc}"}
            # Persist the analysis alongside the match so it survives the run and can
            # be surfaced later (e.g. by get_bot for bots that played this match).
            try:
                af = Path(matches_dir) / match_id / "analyses.jsonl"
                af.parent.mkdir(parents=True, exist_ok=True)
                with af.open("a") as fh:
                    fh.write(json.dumps({"at": datetime.datetime.now().isoformat(),
                                         "code": code[:1000], "result": result}) + "\n")
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
            return {"ok": True, "result": result, "match_id": match_id}
    return {"ok": False, "error": "analyze() printed no result", "stdout": proc.stdout[-2000:]}


# --------------------------------------------------------------------------- #
# physics diagnostics — static gear/inertia/traction/mass breakdown
# --------------------------------------------------------------------------- #
def diagnose_physics_core(robot_xml: str) -> Dict[str, Any]:
    """Compile the bot and return get_physics_diagnostics(). Never raises.

    A standalone (uncomposed) robot has no `red_`/`blue_` prefix, so we pass
    prefix="" to include every body/actuator.
    """
    try:
        import mujoco
        _mj()  # ensure mjarena is importable
        cfg = get_match_config()
        from mjarena.design_shop import ModelValidationConfig, validate_morphology
        from mjarena.design_shop.tools.physics_diagnostics import get_physics_diagnostics
        mcfg = ModelValidationConfig(constraints_yaml_path=cfg.constraints_path,
                                     physics_mode=cfg.physics_mode)
        morph = validate_morphology(robot_xml, mcfg)
        if not morph.passed:
            return {"ok": False, "error": "morphology failed; fix the body first",
                    "feedback": morph.feedback}
        model = mujoco.MjModel.from_xml_string(morph.processed_xml)
        return {"ok": True, "diagnostics": get_physics_diagnostics(model, prefix="")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# probe_obs — the actual obs dict the controller's policy_step receives
# --------------------------------------------------------------------------- #
def _summarize_obs(obs) -> Dict[str, Any]:
    """JSON-safe, compact view of an obs dict: scalars as-is, arrays as shape+sample,
    nested dicts/lists summarized. Shows the model exactly what policy_step gets."""
    import numpy as np

    def summ(v):
        if isinstance(v, np.ndarray):
            flat = v.flatten()
            return {"ndarray_shape": list(v.shape), "dtype": str(v.dtype),
                    "sample": [round(float(x), 4) for x in flat[:8].tolist()]}
        if isinstance(v, dict):
            return {k: summ(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return {"list_len": len(v), "item0": (summ(v[0]) if v else None)}
        if isinstance(v, (np.floating, np.integer)):
            return round(float(v), 5)
        if isinstance(v, (int, float, bool, str)) or v is None:
            return v
        return str(v)

    return {k: summ(v) for k, v in obs.items()}


def probe_obs_core(workspace: Path, ref: str = "draft") -> Dict[str, Any]:
    """Run the bot one step vs the box and return the actual obs its policy_step
    receives (keys, shapes, sample values) — incl. grids/history not in match_data.
    Never raises."""
    ws = Path(workspace)
    try:
        r = resolve_bot_ref(ref, ws)
        robot_xml = r.xml_path.read_text() if r.xml_path.is_file() else ""
        if not robot_xml.strip():
            return {"ok": False, "error": f"{ref!r} has an empty robot.xml"}
        M = _mj()
        cfg = get_match_config()
        from mjarena.design_shop import ModelValidationConfig, validate_morphology
        mcfg = ModelValidationConfig(constraints_yaml_path=cfg.constraints_path,
                                     physics_mode=cfg.physics_mode)
        morph = validate_morphology(robot_xml, mcfg)
        if not morph.passed:
            return {"ok": False, "error": "morphology failed; fix the body first",
                    "feedback": morph.feedback}
        actuators = M["get_actuator_names_from_xml_string"](morph.processed_xml)
        pdir = ws / "_work" / "probe"
        pdir.mkdir(parents=True, exist_ok=True)
        proc = pdir / "probe_robot.xml"  # in the per-run work dir, not system temp
        proc.write_text(morph.processed_xml)
        match_fn = _stationary_match_fn(M, cfg, proc, pdir, max_steps=3)
        captured = []

        def recorder(obs):
            if not captured:
                captured.append(obs)
            return {a: 0.0 for a in actuators}

        with _time_limit(_cfg.MATCH_TIMEOUT):
            M["run_seeds"](run_match_fn=match_fn, policy_callable=recorder, n_seeds=1)
        if not captured:
            return {"ok": False, "error": "no obs captured"}
        return {"ok": True, "actuators": actuators, "obs": _summarize_obs(captured[0])}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "traceback": traceback.format_exc()}


# --------------------------------------------------------------------------- #
# bot library — per-run named snapshots under <workspace>/bots/<name>/
# --------------------------------------------------------------------------- #
_RESERVED_NAMES = {"draft", "current", "stationary", "box"}


def save_bot(workspace: Path, name: str, source_ref: str = "draft",
             overwrite: bool = False, design: dict = None) -> Dict[str, Any]:
    """Snapshot a bot into bots/<name>/ with a design artifact recording its
    validation/qualification status AND the model's design intent. `design` is a
    free-form dict the model supplies (e.g. summary, design_strategy, hardware_plan,
    combat_plan, reasoning) — mirrors the arena BotArtifact's design fields, so the
    artifact captures WHY this bot is built the way it is. Callable regardless of
    pass/fail."""
    ws = Path(workspace)
    if not is_safe_name(name) or name in _RESERVED_NAMES:
        return {"ok": False, "error": f"invalid bot name: {name!r}"}
    try:
        src = resolve_bot_ref(source_ref, ws)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    dst = ws / "bots" / name
    if dst.exists() and not overwrite:
        return {"ok": False, "error": f"bot {name!r} already exists (pass overwrite=true)"}
    robot_xml = src.xml_path.read_text() if src.xml_path.is_file() else ""
    ctrl = src.ctrl_path.read_text() if (src.ctrl_path and src.ctrl_path.is_file()) else ""
    q = run_qualification_core(robot_xml, ctrl, ws / "_work" / f"save_{name}")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "robot.xml").write_text(robot_xml)
    (dst / "controller.py").write_text(ctrl)
    artifact = {
        "name": name,
        "validation_passed": q.get("validation_passed", False),
        "qualification_passed": q.get("qualification_passed", False),
        "design": dict(design or {}),   # model's design intent (free-form fields)
        "feedback": q.get("feedback", q.get("error", "")),
        "saved_at": datetime.datetime.now().isoformat(),
        "source": source_ref,
    }
    (dst / "bot_artifact.json").write_text(json.dumps(artifact, indent=2))
    return {"ok": True, "name": name, "path": str(dst.resolve()), "design": artifact["design"],
            **{k: artifact[k] for k in ("validation_passed", "qualification_passed")}}


def list_bots(workspace: Path):
    """Return one record per saved bot (reads each bot_artifact.json)."""
    bots = Path(workspace) / "bots"
    rows = []
    if bots.resolve() != bots.absolute() or not bots.is_dir():
        return rows
    for d in sorted(bots.iterdir()):
        a = d / "bot_artifact.json"
        if a.resolve() != a.absolute():
            continue
        if a.is_file():
            try:  # build_state calls this every step — a model-written bad file must not kill the loop
                j = json.loads(a.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            row = {k: j.get(k) for k in
                   ("name", "validation_passed", "qualification_passed", "saved_at")}
            row["summary"] = (j.get("design") or {}).get("summary", "")
            rows.append(row)
    return rows


def _bot_matches_and_analyses(ws: Path, name: str):
    """Match history + stored analyses for matches this bot (by name) played in."""
    matches_dir = ws / "matches"
    history, analyses = [], []
    if not matches_dir.is_dir():
        return history, analyses
    for d in sorted(matches_dir.iterdir()):
        mr = d / "metadata" / "match_result.json"
        if not mr.is_file():
            continue
        try:
            data = json.loads(mr.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        red, blue = data.get("red_bot"), data.get("blue_bot")
        if name not in (red, blue):
            continue
        ms = data.get("matches", [])
        # winner is the SIDE ("red"/"blue"/"tie"), not a bot name — count per side
        # then report from this bot's perspective.
        red_side = sum(1 for m in ms if m.get("winner") == "red")
        blue_side = sum(1 for m in ms if m.get("winner") == "blue")
        role = "red" if red == name else "blue"
        history.append({"match_id": d.name, "role": role,
                        "vs": blue if role == "red" else red,
                        "wins": red_side if role == "red" else blue_side,
                        "losses": blue_side if role == "red" else red_side,
                        "draws": len(ms) - red_side - blue_side})
        af = d / "analyses.jsonl"
        if af.is_file():
            for line in af.read_text().splitlines():
                try:
                    analyses.append({"match_id": d.name, **json.loads(line)})
                except json.JSONDecodeError:
                    continue
    return history, analyses


def get_bot(workspace: Path, name: str) -> Dict[str, Any]:
    """Copy a saved bot's files onto the draft (overwrites workspace robot/controller),
    and surface its design intent, match history, and any stored analyses."""
    ws = Path(workspace)
    if not is_safe_name(name):
        return {"ok": False, "error": f"invalid bot name: {name!r}"}
    src = ws / "bots" / name
    if not (src / "robot.xml").is_file():
        return {"ok": False, "error": f"no saved bot {name!r}"}
    shutil.copy(src / "robot.xml", ws / "robot.xml")
    shutil.copy(src / "controller.py", ws / "controller.py")
    artifact = {}
    af = src / "bot_artifact.json"
    if af.is_file():
        artifact = json.loads(af.read_text())
    history, analyses = _bot_matches_and_analyses(ws, name)
    return {"ok": True, "name": name, "loaded_onto": "draft",
            "design": artifact.get("design", {}),
            "validation_passed": artifact.get("validation_passed"),
            "qualification_passed": artifact.get("qualification_passed"),
            "history": history, "analyses": analyses}


# --------------------------------------------------------------------------- #
# submission — the model's chosen final, in out/, else round-robin the library
# --------------------------------------------------------------------------- #
def submit_core(workspace: Path, ref: str = "draft") -> Dict[str, Any]:
    """Designate the final submission: copy a bot ref's files into out/ (overwrites)."""
    ws = Path(workspace)
    try:
        r = resolve_bot_ref(ref, ws)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    robot = r.xml_path.read_text() if r.xml_path.is_file() else ""
    ctrl = r.ctrl_path.read_text() if (r.ctrl_path and r.ctrl_path.is_file()) else ""
    if not robot.strip() or not ctrl.strip():
        return {"ok": False, "error": f"{ref!r} has an empty robot.xml/controller.py"}
    out = ws / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "robot.xml").write_text(robot)
    (out / "controller.py").write_text(ctrl)
    art = ws / "bots" / ref / "bot_artifact.json"
    if r.kind == "saved" and art.is_file():
        shutil.copy(art, out / "bot_artifact.json")
    return {"ok": True, "submitted": ref, "out": str(out.resolve())}


def _bradley_terry(names, pw, alpha: float = 0.5, iters: int = 1000, tol: float = 1e-10):
    """Bradley-Terry MLE strengths from pairwise win counts pw[a][b] (= times a beat b),
    via the standard MM iteration. A symmetric prior (alpha pseudo-wins each way) keeps
    undefeated/winless players finite and ordered. Normalized to geometric mean 1."""
    import math
    names = list(names)
    if not names:
        return {}
    others = {a: [b for b in names if b != a] for a in names}
    W = {a: sum(pw[a].get(b, 0) for b in others[a]) + alpha * len(others[a]) for a in names}
    N = {a: {b: pw[a].get(b, 0) + pw[b].get(a, 0) + 2 * alpha for b in others[a]} for a in names}
    p = {a: 1.0 for a in names}
    for _ in range(iters):
        nxt = {}
        for a in names:
            denom = sum(N[a][b] / (p[a] + p[b]) for b in others[a])
            nxt[a] = (W[a] / denom) if denom > 0 else p[a]
        gm = math.exp(sum(math.log(v) for v in nxt.values()) / len(nxt))
        nxt = {a: v / gm for a, v in nxt.items()}
        if max(abs(nxt[a] - p[a]) for a in names) < tol:
            p = nxt
            break
        p = nxt
    return p


def round_robin_core(workspace: Path, names, n_seeds: int = 3) -> Dict[str, Any]:
    """Evaluate every saved candidate, both sides; invalid bots forfeit."""
    ws = Path(workspace)
    names = list(names)
    pairwise = {name: {} for name in names}
    wins = {name: 0 for name in names}
    checks = {
        name: run_verification_core(
            (ws / "bots" / name / "robot.xml").read_text(),
            (ws / "bots" / name / "controller.py").read_text(),
        )
        for name in names
    }
    # Do not trust model-created selection checkpoints. This directory is chosen
    # after the model finishes, and no model tool runs during final selection.
    selection_dir = ws / "selection_matches" / uuid.uuid4().hex
    selection_dir.mkdir(parents=True)
    completed = 0
    for red in names:
        for blue in names:
            if red == blue:
                continue
            if not checks[red].get("ok") or not checks[blue].get("ok"):
                raise RuntimeError("Selection verification infrastructure failure")
            red_valid = bool(checks[red].get("verification_passed"))
            blue_valid = bool(checks[blue].get("verification_passed"))
            if not red_valid or not blue_valid:
                result = {
                    "ok": True, "forfeit": True,
                    "wins": n_seeds if red_valid and not blue_valid else 0,
                    "losses": n_seeds if blue_valid and not red_valid else 0,
                    "draws": n_seeds if red_valid == blue_valid else 0,
                }
            else:
                result = run_match_core(ws, red, blue, selection_dir, n_seeds=n_seeds)
            if not result.get("ok"):
                raise RuntimeError("Selection match failed: " + str(result))
            (selection_dir / f"pair_{completed:04d}.json").write_text(
                json.dumps({"red": red, "blue": blue, **result}))
            pairwise[red][blue] = pairwise[red].get(blue, 0) + result["wins"] + 0.5 * result["draws"]
            pairwise[blue][red] = pairwise[blue].get(red, 0) + result["losses"] + 0.5 * result["draws"]
            wins[red] += result["wins"]
            wins[blue] += result["losses"]
            completed += 1
            (ws / "selection_progress.json").write_text(json.dumps({
                "completed_pairings": completed,
                "total_pairings": len(names) * (len(names) - 1),
            }))
    ratings = _bradley_terry(names, pairwise)
    ranking = sorted(names, key=lambda name: (-ratings[name], -wins[name], name))
    return {
        "method": "bradley_terry_draws_half_win", "seeds_per_pairing": n_seeds,
        "both_sides": True,
        "ranking": [{"name": name, "bt_rating": ratings[name], "seed_wins": wins[name]}
                    for name in ranking],
        "best": ranking[0] if ranking else None,
    }


def resolve_submission_core(workspace: Path, rr_seeds: int = 3) -> Dict[str, Any]:
    """Select from all saved candidates; never fall back to a draft or manual out/."""
    ws = Path(workspace)
    names = sorted(bot["name"] for bot in list_bots(ws))
    if not names:
        raise RuntimeError("No complete bot recorded with save_bot")
    tournament = (round_robin_core(ws, names, rr_seeds) if len(names) > 1 else
                  {"best": names[0], "ranking": [{"name": names[0]}]})
    bot = ws / "bots" / tournament["best"]
    return {
        "how": "round_robin" if len(names) > 1 else "single_candidate",
        "name": tournament["best"],
        "robot_xml": (bot / "robot.xml").read_text(),
        "controller_py": (bot / "controller.py").read_text(),
        "artifact": json.loads((bot / "bot_artifact.json").read_text()),
        "ranking": tournament["ranking"],
    }
