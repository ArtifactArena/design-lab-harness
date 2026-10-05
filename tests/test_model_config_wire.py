"""Every model yaml under configs/models/ must put its reasoning + output-budget settings ON
THE WIRE (regression test, 2026-09-17).

``model_type: responses`` yamls used the chat-completions keys ``reasoning_effort`` /
``max_tokens``; DSPy hands the yaml to ``litellm.responses(**request)``, which only knows
``reasoning={"effort": ...}`` / ``max_output_tokens`` and silently drops the rest, so every
GPT run went out at OpenAI's default effort with no output cap. ``claude-opus-4-7/4-8``
additionally need ``thinking: {type: adaptive}``: effort alone leaves thinking off.

Drives each yaml through the real path (configure_lm -> dspy.LM -> litellm -> provider SDK)
with httpx stubbed offline and asserts on the JSON body the provider would receive.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest
import yaml

import mjarena.core  # noqa: F401  (import order: avoids the dspy_core circular import)
from mjarena.dspy_core import configure_lm  # imports litellm BEFORE httpx is stubbed below

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(p for p in (ROOT / "configs/models").rglob("*.yaml") if "model:" in p.read_text())
CHAT_ONLY_KEYS = ("reasoning_effort", "max_tokens", "max_completion_tokens")
# Anthropic models that run WITHOUT thinking unless `thinking` is sent explicitly.
THINKING_OFF_BY_DEFAULT = ("claude-opus-4-7", "claude-opus-4-8")


def _fake_response(req: httpx.Request) -> httpx.Response:
    host, path = req.url.host, req.url.path
    if host == "api.openai.com" and path.endswith("/responses"):
        payload = {
            "id": "resp_test", "object": "response", "created_at": 0, "status": "completed",
            "model": "test", "error": None, "incomplete_details": None, "instructions": None,
            "metadata": {}, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "text": {"format": {"type": "text"}}, "reasoning": {"effort": None, "summary": None},
            "temperature": 1.0, "top_p": 1.0, "truncation": "disabled", "max_output_tokens": None,
            "previous_response_id": None, "user": None,
            "output": [{"type": "message", "id": "msg_test", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }
    elif host == "api.anthropic.com":
        payload = {"id": "msg_test", "type": "message", "role": "assistant", "model": "test",
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}}
    elif host == "generativelanguage.googleapis.com":
        payload = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                   "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}
    else:  # OpenAI-compatible chat completions (xAI, Together, ...)
        payload = {"id": "chatcmpl_test", "object": "chat.completion", "created": 0, "model": "test",
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
    return httpx.Response(200, json=payload, headers={"content-type": "application/json"}, request=req)


@pytest.fixture
def wire(monkeypatch):
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "TOGETHER_API_KEY"):
        monkeypatch.setenv(var, "test-key")
    bodies: list[dict] = []
    lock = threading.Lock()

    def send(self, request, *args, **kwargs):
        # LiteLLM can also fetch metadata; capture inference POSTs only.
        if request.method == "POST":
            with lock:
                bodies.append(json.loads(request.content or b"{}"))
        return _fake_response(request)

    async def asend(self, request, *args, **kwargs):
        return send(self, request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    return bodies


@pytest.mark.parametrize("cfg", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_settings_reach_the_wire(cfg, wire):
    spec = yaml.safe_load(cfg.read_text())
    if spec.get("load_balancer") or spec.get("api_base"):
        pytest.skip("proxy / custom endpoint config")
    lm = configure_lm(str(cfg), use_cache=False)
    lm(messages=[{"role": "user", "content": "Say OK."}])
    assert len(wire) == 1, f"expected one HTTP request, saw {len(wire)}"
    body = wire[0]
    model = spec["model"]

    if spec.get("model_type") == "responses":
        for key in CHAT_ONLY_KEYS:
            assert key not in spec, f"{cfg.name}: `{key}` is a chat-completions key; the Responses API drops it"
        assert body.get("reasoning", {}).get("effort") == spec["reasoning"]["effort"], body
        assert body.get("max_output_tokens") == spec["max_output_tokens"], body
    elif model.startswith("anthropic/"):
        assert body.get("output_config", {}).get("effort") == spec["output_config"]["effort"], body
        assert body.get("max_tokens") == spec["max_tokens"], body
        if any(m in model for m in THINKING_OFF_BY_DEFAULT):
            assert body.get("thinking", {}).get("type") == "adaptive", (
                f"{cfg.name}: {model} runs without extended thinking unless thinking.type=adaptive is sent; body={body}")
    elif model.startswith("gemini/"):
        gc = body.get("generationConfig", {})
        assert gc.get("max_output_tokens") == spec["max_tokens"], body
        if "temperature" in spec:
            assert gc.get("temperature") == spec["temperature"], body
        if "reasoning_effort" in spec:
            tc = gc.get("thinkingConfig", {})
            assert tc.get("thinkingLevel") == spec["reasoning_effort"] or tc.get("thinkingBudget"), (
                f"{cfg.name}: reasoning_effort={spec['reasoning_effort']} not on the wire: {body}")
    else:
        # Chat-completions providers: every non-transport yaml key must appear verbatim
        # (litellm renames max_tokens -> max_completion_tokens for OpenAI chat models).
        for key, val in spec.items():
            if key in ("model", "model_type", "num_retries", "timeout", "cache", "provider"):
                continue
            if key == "extra_body":
                for k2, v2 in val.items():
                    assert body.get(k2) == v2, f"{cfg.name}: extra_body.{k2} not on the wire: {body}"
            elif key == "max_tokens":
                assert val in (body.get("max_tokens"), body.get("max_completion_tokens")), (
                    f"{cfg.name}: max_tokens={val} not on the wire: {body}")
            else:
                assert body.get(key) == val, f"{cfg.name}: {key}={val!r} not on the wire: {body}"
