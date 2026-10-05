"""
Core DSPy pipeline for arena bots - ACTUATOR format only.

This module provides:
- Config loaders: constraints, MJCF syntax, design reasoning, obs schema
- Task spec generators for 2D/3D morphology
- Match runner factory and zero-policy helper

Unified generation orchestration lives in:
- mjarena.core.unified_builder (unified morphology + controller generation)
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass
import json
import logging
import datetime
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib import request

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY
import litellm
import mujoco
import yaml

# Register Gemini model variants that litellm doesn't know about yet.
# Always force supports_reasoning=True so litellm allows reasoning_effort param.
# Patch BOTH "gemini/<model>" and bare "<model>" keys.
_gemini_base = litellm.get_model_cost_map("").get("gemini/gemini-3-pro-preview", {})
for _variant in (
    "gemini/gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview",
    "gemini/gemini-3-flash-preview",
    "gemini-3-flash-preview",
):
    if _variant not in litellm.model_cost:
        litellm.model_cost[_variant] = {**_gemini_base}
    litellm.model_cost[_variant]["supports_reasoning"] = True

# LiteLLM's bundled model_cost JSON is older than the live one and is missing
# recent models (anthropic/claude-opus-4-{6,7}, openai/gpt-5.{3,4,5}, etc.).
# Subprocesses that fail the model_cost network fetch (5s timeout) fall back to
# the bundle and then `litellm.get_model_info()` raises "model isn't mapped yet"
# — which `configure_lm` surfaces as the "max_tokens=null requires LiteLLM to
# know..." setup error. Pre-populate the entries we care about with the values
# from the live JSON so this works deterministically offline.
_MAX_OUTPUT_TOKEN_FALLBACKS = {
    "anthropic/claude-opus-4-7":   128000,
    "anthropic/claude-opus-4-6":   128000,
    "anthropic/claude-sonnet-4-6":  64000,
    "openai/gpt-5.5":              128000,
    "openai/gpt-5.4":              128000,
    "openai/gpt-5.3-codex":        128000,
    "openai/gpt-5.2-codex":        128000,
    "openai/gpt-5.1-codex":        128000,
    "openai/gpt-5.1-codex-max":    128000,
    "openai/gpt-5.2-pro":          128000,
    "openai/gpt-5.5-pro":          128000,
    "gemini/gemini-3.1-pro-preview": 65536,
    "gemini/gemini-3-flash-preview": 65535,
}
for _m, _native in _MAX_OUTPUT_TOKEN_FALLBACKS.items():
    _entry = litellm.model_cost.setdefault(_m, {})
    _entry.setdefault("max_output_tokens", _native)
    _entry.setdefault("max_tokens", _native)
    _entry.setdefault("litellm_provider", _m.split("/", 1)[0])
    # Every model in this dict is reasoning-capable (all our -high configs pass
    # reasoning_effort). Force the flag so the bundle-fallback path can't strip
    # the param. setdefault would lose to a stale bundle entry of False — assign.
    _entry["supports_reasoning"] = True

# Monkey-patch litellm.utils.supports_reasoning. Root cause: LiteLLM's
# GoogleAIStudioGeminiConfig.get_supported_openai_params calls
# `supports_reasoning(model)` with the *bare* model name (no `gemini/` prefix).
# That bare form is unknown to `get_llm_provider` (it only pattern-matches a
# hardcoded list), so `_supports_factory` raises BadRequestError and returns
# False — making LiteLLM reject `reasoning_effort` even though model_cost says
# the model supports it. The Gemini → bundle path always fails this way in
# subprocesses where the model_cost network fetch falls back to the bundled
# JSON. We hardcode the allowlist for models we know support reasoning.
#
# This is NOT Gemini-specific. The 2026-05-29 zeroshot run forfeited every
# Anthropic bot with "anthropic does not support parameters: ['reasoning_effort']"
# because the build subprocesses lost the model_cost network fetch and fell back
# to litellm's stale bundle (which lists none of these models), so AnthropicConfig
# called supports_reasoning("claude-opus-4-7") → False → stripped reasoning_effort.
# The allowlist must therefore cover EVERY reasoning model our configs send
# reasoning_effort to (Anthropic, OpenAI, Gemini), in both bare and prefixed form.
import litellm.utils as _llm_utils  # noqa: E402

_REASONING_MODELS_OVERRIDE = {
    "gemini-3.1-pro-preview", "gemini/gemini-3.1-pro-preview",
    "gemini-3-pro-preview",   "gemini/gemini-3-pro-preview",
    "gemini-3-flash-preview", "gemini/gemini-3-flash-preview",
    "claude-opus-4-7",        "anthropic/claude-opus-4-7",
    "claude-opus-4-6",        "anthropic/claude-opus-4-6",
    "claude-sonnet-4-6",      "anthropic/claude-sonnet-4-6",
    "gpt-5.5",                "openai/gpt-5.5",
    "gpt-5.4",                "openai/gpt-5.4",
    "gpt-5.3-codex",          "openai/gpt-5.3-codex",
}
_orig_supports_reasoning = _llm_utils.supports_reasoning


def _patched_supports_reasoning(model, custom_llm_provider=None):
    if model in _REASONING_MODELS_OVERRIDE:
        return True
    return _orig_supports_reasoning(model, custom_llm_provider)


_llm_utils.supports_reasoning = _patched_supports_reasoning
litellm.supports_reasoning = _patched_supports_reasoning
# Also patch the import site Gemini's transformation reads from.
import litellm.llms.gemini.chat.transformation as _gemini_transform  # noqa: E402
_gemini_transform.supports_reasoning = _patched_supports_reasoning

from mjarena.agents.types import VerificationResult
from mjarena.design_shop.rules.mj_validators import ModelValidationConfig
from mjarena.policy_spec import PolicySpec
from mjarena.runner.episode import run_match
from mjarena.runner.recording import VideoOverlayInfo

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROXY_SERVICE_NAME = "mjarena-gemma31b-load-balancer"


def _normalize_api_base(api_base: str) -> str:
    """Accept either an API base or a full chat completions endpoint."""
    normalized = api_base.rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized


def _proxy_base_url(url: str) -> str:
    normalized = url.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized.rstrip("/")


def _proxy_health_payload(proxy_base: str, *, timeout: float = 0.5) -> Optional[dict[str, Any]]:
    try:
        with request.urlopen(f"{proxy_base}/health", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if resp.status == 200 and payload.get("service") == _PROXY_SERVICE_NAME:
            return payload
    except Exception:
        pass
    return None


def _proxy_health(proxy_base: str, *, timeout: float = 0.5) -> bool:
    return _proxy_health_payload(proxy_base, timeout=timeout) is not None


def _ensure_load_balancer_proxy(lb_config: dict[str, Any]) -> None:
    if not lb_config.get("enabled", False):
        return

    endpoints = lb_config.get("endpoints") or []
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("load_balancer.enabled=true requires a non-empty endpoints list")
    expected_backends = {_proxy_base_url(str(endpoint)) for endpoint in endpoints}

    proxy_url = _proxy_base_url(str(lb_config.get("proxy_url", "http://127.0.0.1:8010")))
    lock_path = Path(str(lb_config.get("lock_path", "/tmp/mjarena_gemma31b_lb_proxy.lock")))
    pid_path = Path(str(lb_config.get("pid_path", "/tmp/mjarena_gemma31b_lb_proxy.pid")))
    log_path = Path(str(lb_config.get("log_path", "/tmp/mjarena_gemma31b_lb_proxy.log")))
    startup_timeout_s = float(lb_config.get("startup_timeout_s", 5.0))
    listen_host = lb_config.get("listen_host")
    listen_port = lb_config.get("listen_port")
    if listen_host is None or listen_port is None:
        from urllib.parse import urlparse

        parsed = urlparse(proxy_url)
        listen_host = parsed.hostname or "127.0.0.1"
        listen_port = parsed.port or 8010

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = _REPO_ROOT / "scripts" / "gemma31b_lb_proxy.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Gemma load-balancer proxy script not found: {script_path}")

    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX fallback
        fcntl = None

    with lock_path.open("w", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = _proxy_health_payload(proxy_url)
            if payload is not None:
                running_backends = {
                    _proxy_base_url(str(backend.get("base_url", "")))
                    for backend in payload.get("backends", [])
                    if isinstance(backend, dict)
                }
                if running_backends == expected_backends:
                    return
                raise RuntimeError(
                    f"Gemma load-balancer proxy is already running at {proxy_url}, "
                    f"but with backends {sorted(running_backends)} instead of "
                    f"{sorted(expected_backends)}. Stop the old proxy or use a different proxy_url."
                )

            command = [
                sys.executable,
                str(script_path),
                "--listen-host",
                str(listen_host),
                "--listen-port",
                str(listen_port),
                "--metrics-timeout",
                str(lb_config.get("metrics_timeout_s", 0.4)),
                "--metrics-ttl",
                str(lb_config.get("metrics_ttl_s", 0.5)),
                "--forward-timeout",
                str(lb_config.get("forward_timeout_s", 720.0)),
            ]
            for endpoint in endpoints:
                command.extend(["--backend", str(endpoint)])

            log_file = log_path.open("ab")
            try:
                proc = subprocess.Popen(
                    command,
                    cwd=str(_REPO_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                log_file.close()

            pid_path.write_text(str(proc.pid), encoding="utf-8")
            deadline = time.time() + startup_timeout_s
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                if _proxy_health(proxy_url, timeout=0.25):
                    logger.info("[configure_lm] started Gemma 31B proxy at %s (pid=%s)", proxy_url, proc.pid)
                    return
                time.sleep(0.1)

            if _proxy_health(proxy_url, timeout=0.25):
                return
            raise RuntimeError(
                f"Gemma load-balancer proxy failed to start at {proxy_url}; "
                f"pid={proc.pid}, log={log_path}"
            )
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _lazy_build_policy_callable(*args, **kwargs):
    from mjarena.eval.runtime import build_policy_callable
    return build_policy_callable(*args, **kwargs)


def _trace(message: str, *, enabled: bool) -> None:
    """Emit a timestamped progress line immediately."""
    if not enabled:
        return
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    thread_name = threading.current_thread().name
    print(f"[{timestamp}] [{thread_name}] {message}", flush=True)


def _log_litellm_params(kwargs, response, start_time, end_time):
    """Debug: log reasoning/thinking params sent to provider."""
    model = kwargs.get("model", "")
    extra = kwargs.get("extra_body") or kwargs.get("optional_params", {}).get("extra_body")
    reasoning = kwargs.get("reasoning_effort")
    if extra or reasoning:
        logger.info(f"[litellm] {model} | reasoning_effort={reasoning} extra_body={extra}")


def _patch_reasoning_content(kwargs, response, start_time, end_time):
    """Promote reasoning_content → content for models that return thinking output only.

    Some models (e.g. Kimi K2.5 on Together AI) put all output in the
    reasoning_content field and leave content empty. Some OpenAI-compatible
    vLLM endpoints expose the same payload under a reasoning field. DSPy only
    reads content, so we copy it over in-place before DSPy processes the
    response.
    """
    try:
        for choice in response.choices:
            msg = choice.message
            if msg.content:
                continue
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
            )
            if reasoning:
                msg.content = reasoning
    except Exception:
        pass


# Per-call context: the refinement loop sets this before each LLM call so the
# success callback knows which bot / commit the usage record belongs to.
# Workers are one-bot-per-process, so there is no cross-bot contention.
USAGE_CTX: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "USAGE_CTX", default=None
)


def _wall_ms(start_time, end_time) -> Optional[int]:
    """Convert litellm's (start_time, end_time) pair to integer milliseconds.
    litellm sometimes passes datetime objects (→ timedelta) and sometimes floats
    (seconds-since-epoch); handle both."""
    if not start_time or not end_time:
        return None
    try:
        delta = end_time - start_time
        seconds = delta.total_seconds() if hasattr(delta, "total_seconds") else float(delta)
        return int(seconds * 1000)
    except Exception:
        return None


def _count_reasoning_content_tokens(response) -> Optional[int]:
    """Fallback reasoning-token counter for providers (Together AI) that do not
    report reasoning_tokens in usage but do return reasoning text."""
    try:
        msg = response.choices[0].message
        rc = getattr(msg, "reasoning_content", None) or ""
        if not rc:
            return None
        import tiktoken  # lazy — keeps import off the hot path
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(rc))
    except Exception:
        return None


def _persist_usage(kwargs, response, start_time, end_time):
    """Append one JSON line to {bot_dir}/refinement/usage.jsonl per LLM call.

    Fields: ts, bot, commit, role, model, prompt_tokens, completion_tokens,
    reasoning_tokens, reasoning_source, cache_read_tokens, total_tokens, wall_ms.
    Reasoning_source is "native" when the provider reports reasoning_tokens in
    usage, "tiktoken-cl100k" when we tokenize reasoning_content as a fallback.
    Safe to fail — wrapped in try/except so a write error never breaks a build.
    """
    try:
        ctx = USAGE_CTX.get()
        if not ctx:
            return  # not in a tracked scope (ad-hoc calls / tests w/o context)

        usage_obj = getattr(response, "usage", None)
        if usage_obj is None:
            return
        if hasattr(usage_obj, "model_dump"):
            u = usage_obj.model_dump(exclude_none=True)
        else:
            try:
                u = dict(usage_obj)
            except Exception:
                u = {}

        # Use explicit None checks — 0 is a valid (if rare) token count.
        prompt = u.get("prompt_tokens")
        if prompt is None:
            prompt = u.get("input_tokens")
        completion = u.get("completion_tokens")
        if completion is None:
            completion = u.get("output_tokens")
        total = u.get("total_tokens")

        details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
        reasoning_native = details.get("reasoning_tokens") if isinstance(details, dict) else None

        reasoning_estimated = None
        reasoning_source = None
        if reasoning_native is not None:
            reasoning_source = "native"
        else:
            reasoning_estimated = _count_reasoning_content_tokens(response)
            if reasoning_estimated is not None:
                reasoning_source = "tiktoken-cl100k"

        prompt_details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
        cache_read = u.get("cache_read_input_tokens")
        if cache_read is None and isinstance(prompt_details, dict):
            cache_read = prompt_details.get("cached_tokens")

        record = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "bot": ctx.get("bot"),
            "commit": ctx.get("commit"),
            "role": ctx.get("role"),
            "model": kwargs.get("model", ""),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning_native if reasoning_native is not None else reasoning_estimated,
            "reasoning_source": reasoning_source,
            "cache_read_tokens": cache_read,
            "total_tokens": total,
            "wall_ms": _wall_ms(start_time, end_time),
        }

        out_dir = Path(ctx["output_dir"]) / "refinement"
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "usage.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"_persist_usage failed: {exc}")


litellm.success_callback.append(_log_litellm_params)
litellm.success_callback.append(_patch_reasoning_content)
litellm.success_callback.append(_persist_usage)


# =============================================================================
# LM Configuration
# =============================================================================


NO_LLM_TIMEOUT_S = 6000  # 100 min per call, the same default in every arena harness. Longer
# than any legitimate call seen (~80 min for kimi-k3 / codex), short enough that a hung
# connection is aborted and retried (num_retries).


def configure_lm(lm_config_path: str, use_cache: bool = True) -> dspy.LM:
    """
    Configure DSPy LM from config file.

    Args:
        lm_config_path: Path to YAML config with model settings.
        use_cache: If True, enable DSPy caching. If False, disable caching
            to force fresh LLM calls each time (useful when same model is
            used for multiple bots with identical prompts).

    Returns:
        Configured dspy.LM instance.
    """
    cfg_path = Path(lm_config_path)
    if not cfg_path.is_file():
        raise ValueError(f"LM config not found: {lm_config_path}")

    params: dict = yaml.safe_load(cfg_path.read_text()) or {}
    model = params.pop("model", "gpt-4o")
    provider = params.pop("provider", None)
    load_balancer = params.pop("load_balancer", None)
    if isinstance(load_balancer, dict) and load_balancer.get("enabled", False):
        _ensure_load_balancer_proxy(load_balancer)
        if "api_base" not in params:
            proxy_url = _proxy_base_url(str(load_balancer.get("proxy_url", "http://127.0.0.1:8010")))
            params["api_base"] = f"{proxy_url}/v1/chat/completions"

    api_base = params.get("api_base")
    if isinstance(api_base, str):
        params["api_base"] = _normalize_api_base(api_base)

    # max_tokens: null in YAML → resolve to the model's LiteLLM-documented
    # native max_output_tokens. Used so cross-model comparisons run each model at
    # its full output budget instead of LiteLLM's DEFAULT_MAX_TOKENS=4096 fallback
    # (which silently truncates Anthropic). Fail loudly if LiteLLM doesn't map the
    # model — caller must then pin an explicit int or use a known alias.
    if "max_tokens" in params and params["max_tokens"] is None:
        try:
            info = litellm.get_model_info(model)
        except Exception as exc:
            raise ValueError(
                f"{cfg_path}: max_tokens=null requires LiteLLM to know {model!r}, "
                f"but get_model_info failed: {exc}. Use an alias LiteLLM maps or "
                f"pin max_tokens to an int."
            ) from exc
        native_max = info.get("max_output_tokens")
        if not native_max:
            raise ValueError(
                f"{cfg_path}: max_tokens=null requested but LiteLLM has no "
                f"max_output_tokens entry for {model!r}. Pin an explicit int."
            )
        params["max_tokens"] = native_max
        logger.info(
            f"[configure_lm] {model}: max_tokens=null → native {native_max}"
        )

    # Add cache control
    if not use_cache:
        params["cache"] = False

    # Per-call timeout: 100 min (NO_LLM_TIMEOUT_S). litellm coerces timeout=None back
    # to its own 600 s default, so this has to be an explicit number. Model yamls
    # must not set a smaller one (tests/test_llm_timeout.py enforces both).
    params.setdefault("timeout", NO_LLM_TIMEOUT_S)

    if provider:
        lm = dspy.LM(model=model, provider=provider, **params)
    else:
        lm = dspy.LM(model=model, **params)

    return lm


def _extract_actuator_names(model: mujoco.MjModel) -> List[str]:
    """Extract actuator names from a compiled MuJoCo model.

    Raises ValueError if any actuator is unnamed, since our pipeline
    requires named actuators for controller dict keys.
    """
    names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if not name:
            raise ValueError(
                f"Actuator at index {i} has no name. "
                f"All actuators must have a name= attribute so the controller can reference them."
            )
        names.append(name)
    return names


def get_actuator_names(xml_path: Path) -> List[str]:
    """Extract actuator names from robot XML."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return _extract_actuator_names(model)


def get_actuator_names_from_xml_string(xml_string: str) -> List[str]:
    """Extract actuator names from XML string content."""
    model = mujoco.MjModel.from_xml_string(xml_string)
    return _extract_actuator_names(model)


# =============================================================================
# Morphology Generation
# =============================================================================


def assemble_bot_design(
    overview: str = "",
    movement: str = "",
    attack: str = "",
    defense: str = "",
    robot_xml: str = "",
) -> str:
    """Assemble bot_design string from morphology output fields.

    Morphology outputs separate fields describing hardware capabilities.
    This function assembles them into a single bot_design string for the controller,
    including the actual hardware XML so the controller LLM sees exact actuator
    structure, gear ratios, and joint types.

    Args:
        overview: Shape, structure, dimensions, mass distribution, strategy.
        movement: What actuators ENABLE movement.
        attack: What hardware ENABLES attack.
        defense: What hardware ENABLES defense.
        robot_xml: Actual MJCF XML of the robot hardware.

    Returns:
        Assembled bot_design string for controller signature.
    """
    sections = []

    if overview:
        sections.append(f"OVERVIEW: {overview}")
    if movement:
        sections.append(f"MOVEMENT HARDWARE: {movement}")
    if attack:
        sections.append(f"ATTACK HARDWARE: {attack}")
    if defense:
        sections.append(f"DEFENSE HARDWARE: {defense}")
    if robot_xml:
        sections.append(f"HARDWARE XML:\n{robot_xml}")

    return "\n\n".join(sections)


def _strip_markdown_code_block(text: str) -> str:
    """Return the one fenced block in *text*, or *text* unchanged.

    Extraction happens early, before any validator has run, so an ambiguous
    reply is handed on untouched: the validators report what is wrong with it
    rather than the loop dying here with the diagnostic destroyed.
    """
    from mjarena.utils.code_blocks import extract_code_block
    try:
        return extract_code_block(text)
    except ValueError:
        return text


def _strip_mjcf_tags(text: str) -> str:
    """Normalise model XML output: drop code fences, the XML declaration, <mjcf> wrappers
    and xmlns declarations (ElementTree would otherwise re-serialise a namespaced document
    with ns0: prefixes that MuJoCo rejects), then wrap in <mujoco> if the root is missing."""
    import re
    from mjarena.utils.code_blocks import extract_code_block
    try:
        text = extract_code_block(text)
    except ValueError:
        return text  # ambiguous fencing: strip_wrappers turns this into feedback
    text = re.sub(r'<\?xml[^>]*\?>\s*', '', text)
    text = re.sub(r'<mjcf[^>]*>\s*', '', text)
    text = re.sub(r'\s*</mjcf>', '', text)
    text = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', '', text)
    text = text.strip()

    if text and not text.lstrip().startswith('<mujoco'):
        text = f'<mujoco>\n{text}\n</mujoco>'

    return text


# =============================================================================
# Zero Policy (Fixed Blue Opponent)
# =============================================================================


def create_zero_policy(actuator_names: Sequence[str]) -> Callable:
    """
    Create a zero-action policy (immobile box).

    This is the fixed blue opponent for Task I.
    """
    names = list(actuator_names)

    def _policy(_obs):
        return {name: 0.0 for name in names}

    return _policy


# =============================================================================
# Match Runner Factory
# =============================================================================


@dataclass
class MatchRunner:
    """Picklable callable that runs one match seed."""

    composed_xml: Path
    out_dir: Path
    blue_policy_callable: Optional[Callable] = None
    blue_policy_spec: Optional[PolicySpec] = None
    max_steps: int = 1000
    action_history_len: int = 10
    save_video_seeds: int = 0
    camera_mode: str = "tracking"
    quiet: bool = False
    video_width: int = 640
    video_height: int = 480
    overlay_base: Optional[Dict[str, str]] = None
    score_function: str = "any"
    rebuild_policies: bool = False
    red_controller_code: Optional[str] = None
    red_actuator_names: Optional[List[str]] = None
    blue_controller_code: Optional[str] = None
    blue_actuator_names: Optional[List[str]] = None
    max_obs_lookback: int = 20
    max_action_lookback: int = 20
    gear_clamp_ratio: float = 0.0
    observation_config: Optional["ObservationConfig"] = None
    match_time: Optional[float] = None
    inactivity_timeout_seconds: Optional[float] = 10.0
    inactivity_min_displacement: float = 0.5
    inactivity_exempt_prefixes: Optional[List[str]] = None
    size_limits: Optional[Tuple[float, float, float]] = None
    contact_fidelity: str = "high"
    trace_label: Optional[str] = None
    rendering_flags: Optional[Dict[str, bool]] = None
    game_context: Optional[Dict[str, Any]] = None
    initial_state: Optional[Dict[str, Any]] = None

    def __call__(
        self,
        red_policy_callable: Callable,
        seed: int = 0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        seed_index: Optional[int] = None,
    ):
        """Run a seed using the supplied red policy callable."""
        if red_policy_callable is None:
            raise ValueError("red_policy_callable is required for in-process match execution")

        if self.rebuild_policies and self.red_controller_code and self.blue_controller_code:
            red_policy = _lazy_build_policy_callable(self.red_controller_code, self.red_actuator_names)
            blue_policy = _lazy_build_policy_callable(self.blue_controller_code, self.blue_actuator_names)
        else:
            red_policy = red_policy_callable
            blue_policy = self._build_blue_policy()

        return self._run_seed(
            red_policy=red_policy,
            blue_policy=blue_policy,
            seed=seed,
            seed_index=seed_index,
            progress_callback=progress_callback,
        )

    def run_with_policy_spec(
        self,
        red_policy_spec: PolicySpec,
        *,
        seed: int = 0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        seed_index: Optional[int] = None,
    ):
        """Run a seed by rebuilding the red policy inside the worker process."""
        red_policy = red_policy_spec.build_callable()
        blue_policy = self._build_blue_policy()
        return self._run_seed(
            red_policy=red_policy,
            blue_policy=blue_policy,
            seed=seed,
            seed_index=seed_index,
            progress_callback=progress_callback,
        )

    def _build_blue_policy(self) -> Callable:
        if self.rebuild_policies and self.blue_controller_code:
            return _lazy_build_policy_callable(self.blue_controller_code, self.blue_actuator_names)
        if self.blue_policy_spec is not None:
            return self.blue_policy_spec.build_callable()
        if self.blue_policy_callable is not None:
            return self.blue_policy_callable
        raise ValueError("No blue policy source was provided for match execution")

    def _run_seed(
        self,
        *,
        red_policy: Callable,
        blue_policy: Callable,
        seed: int,
        seed_index: Optional[int],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    ):
        trace_enabled = bool(self.trace_label)
        should_save_video = False
        if self.save_video_seeds > 0:
            ordinal = seed if seed_index is None else seed_index
            should_save_video = ordinal < self.save_video_seeds
        video_path = self.out_dir / f"seed_{seed}.webm" if should_save_video else None
        seed_label = self.trace_label or "match"
        t0 = time.monotonic()
        _trace(
            f"{seed_label}: create_match_runner start seed={seed} "
            f"(save_video={should_save_video})",
            enabled=trace_enabled,
        )

        ov_info = None
        if self.overlay_base and should_save_video:
            ov_info = VideoOverlayInfo(seed=seed, **self.overlay_base)

        _trace(f"{seed_label}: run_match start seed={seed}", enabled=trace_enabled)
        result = run_match(
            composed_xml=self.composed_xml,
            red_policy_py=red_policy,
            blue_policy_py=blue_policy,
            out_dir=self.out_dir,
            max_steps=self.max_steps,
            use_gui=False,
            save_video=should_save_video,
            camera_mode=self.camera_mode,
            action_history_len=self.action_history_len,
            video_path=str(video_path) if video_path else None,
            seed=seed,
            quiet=self.quiet,
            video_width=self.video_width,
            video_height=self.video_height,
            overlay_info=ov_info,
            score_function=self.score_function,
            max_obs_lookback=self.max_obs_lookback,
            max_action_lookback=self.max_action_lookback,
            gear_clamp_ratio=self.gear_clamp_ratio,
            observation_config=self.observation_config,
            match_time=self.match_time,
            inactivity_timeout_seconds=self.inactivity_timeout_seconds,
            inactivity_min_displacement=self.inactivity_min_displacement,
            inactivity_exempt_prefixes=self.inactivity_exempt_prefixes,
            size_limits=self.size_limits,
            env_kwargs={"contact_fidelity": self.contact_fidelity},
            trace_label=f"{seed_label} | seed {seed}" if trace_enabled else None,
            progress_callback=progress_callback,
            rendering_flags=self.rendering_flags,
            game_context=self.game_context,
            initial_state=self.initial_state,
        )
        _trace(
            f"{seed_label}: run_match done seed={seed} in {time.monotonic() - t0:.1f}s",
            enabled=trace_enabled,
        )
        return result


def create_match_runner(
    *,
    composed_xml: Path,
    blue_policy_callable: Optional[Callable] = None,
    blue_policy_spec: Optional[PolicySpec] = None,
    out_dir: Path,
    max_steps: int = 1000,
    action_history_len: int = 10,
    save_video_seeds: int = 0,
    camera_mode: str = "tracking",
    quiet: bool = False,
    video_width: int = 640,
    video_height: int = 480,
    overlay_base: Optional[Dict[str, str]] = None,
    score_function: str = "any",
    rebuild_policies: bool = False,
    red_controller_code: Optional[str] = None,
    red_actuator_names: Optional[List[str]] = None,
    blue_controller_code: Optional[str] = None,
    blue_actuator_names: Optional[List[str]] = None,
    max_obs_lookback: int = 20,
    max_action_lookback: int = 20,
    gear_clamp_ratio: float = 0.0,
    observation_config: Optional["ObservationConfig"] = None,
    match_time: Optional[float] = None,
    inactivity_timeout_seconds: Optional[float] = 10.0,
    inactivity_min_displacement: float = 0.5,
    inactivity_exempt_prefixes: Optional[List[str]] = None,
    size_limits: Optional[Tuple[float, float, float]] = None,
    contact_fidelity: str = "high",
    trace_label: Optional[str] = None,
    rendering_flags: Optional[Dict[str, bool]] = None,
    game_context: Optional[Dict[str, Any]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Callable:
    """
    Create a match runner function for rollouts.

    Args:
        composed_xml: Path to composed arena XML.
        blue_policy_callable: Blue robot policy function (fixed opponent).
        out_dir: Output directory for rollouts.
        max_steps: Maximum steps per match.
        action_history_len: Length of action history.
        save_video_seeds: Number of seeds to save video for (0 = none).
        camera_mode: Camera mode ("side", "tracking", or "topdown").
        score_function: Score function ("any" or "thres-50").
        rebuild_policies: If True, rebuild policy callables from source code
            on every call (thread-safe — each call gets fresh mutable state).
            Requires red/blue_controller_code and red/blue_actuator_names.
        red_controller_code: Source code for red policy (used when rebuild_policies=True).
        red_actuator_names: Actuator names for red policy (used when rebuild_policies=True).
        blue_controller_code: Source code for blue policy (used when rebuild_policies=True).
        blue_actuator_names: Actuator names for blue policy (used when rebuild_policies=True).
        observation_config: ObservationConfig controlling spatial grid features.

    Returns a function that takes (red_policy_callable, seed) and returns match result.

    Note: The environment swaps starting positions on odd seeds to prevent
    policies from overfitting to one starting side.
    """

    return MatchRunner(
        composed_xml=composed_xml,
        out_dir=out_dir,
        blue_policy_callable=blue_policy_callable,
        blue_policy_spec=blue_policy_spec,
        max_steps=max_steps,
        action_history_len=action_history_len,
        save_video_seeds=save_video_seeds,
        camera_mode=camera_mode,
        quiet=quiet,
        video_width=video_width,
        video_height=video_height,
        overlay_base=overlay_base,
        score_function=score_function,
        rebuild_policies=rebuild_policies,
        red_controller_code=red_controller_code,
        red_actuator_names=red_actuator_names,
        blue_controller_code=blue_controller_code,
        blue_actuator_names=blue_actuator_names,
        max_obs_lookback=max_obs_lookback,
        max_action_lookback=max_action_lookback,
        gear_clamp_ratio=gear_clamp_ratio,
        observation_config=observation_config,
        match_time=match_time,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        inactivity_min_displacement=inactivity_min_displacement,
        inactivity_exempt_prefixes=inactivity_exempt_prefixes,
        size_limits=size_limits,
        contact_fidelity=contact_fidelity,
        trace_label=trace_label,
        rendering_flags=rendering_flags,
        game_context=game_context,
        initial_state=initial_state,
    )
