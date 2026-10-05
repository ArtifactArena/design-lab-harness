#!/usr/bin/env python3
"""Cross-model smoke: run the agent loop a few steps against each model and report.

Exercises the text tool-protocol (model-calling + reply extraction + JSON parsing)
across providers, where format quirks surface. Prints a pass/fail table and, for each
model, how many steps dispatched a real tool vs failed to parse — plus a snippet of
the first reply so format bugs are easy to see.

    MUJOCO_GL= python scripts/smoke_models.py                 # default cheap set
    MUJOCO_GL= python scripts/smoke_models.py xai/grok-3-mini google/gemini-2-5-flash
    STEPS=3 python scripts/smoke_models.py

Each model is skipped (not failed) if its API key is absent.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # design-lab-harness/
sys.path.insert(0, str(HERE / "arena_kit"))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import agent  # noqa: E402

# Validated cheap set (all dispatch cleanly here). Pass other model names as args to
# try them — e.g. together_ai deepseek/kimi need a dedicated endpoint on your account,
# and anthropic needs a valid ANTHROPIC_API_KEY.
DEFAULT_MODELS = [
    "google/gemini-2-5-flash",
    "google/gemini-2-5-flash-lite",
    "xai/grok-3-mini",
    "openai/gpt-5-nano",
    "qwen/qwen3.5-397b-nothinking",
]

# Which env var each provider needs (Together hosts deepseek/kimi/qwen).
_KEY = {"google": "GEMINI_API_KEY", "xai": "XAI_API_KEY", "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", "deepseek": "TOGETHER_API_KEY",
        "kimi": "TOGETHER_API_KEY", "qwen": "TOGETHER_API_KEY"}


def smoke(model: str, steps: int) -> dict:
    provider = model.split("/", 1)[0]
    key = _KEY.get(provider)
    if key and not os.environ.get(key):
        return {"model": model, "status": "SKIP", "reason": f"no {key}"}
    cfg = config.REPO_ROOT / "configs" / "models" / f"{model}.yaml"
    if not cfg.is_file():
        return {"model": model, "status": "SKIP", "reason": "no config in bundle"}
    config.MODEL_CONFIG = cfg
    with tempfile.TemporaryDirectory() as td:
        agent.RUNS = Path(td) / "runs"
        try:
            run_dir = agent.run(max_steps=steps)
        except Exception as exc:  # noqa: BLE001
            return {"model": model, "status": "FAIL", "reason": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()}
        jl = [json.loads(x) for x in (run_dir / "journal.jsonl").read_text().splitlines()]
        tr = [json.loads(x) for x in (run_dir / "transcript.jsonl").read_text().splitlines()]
        dispatched = sum(1 for j in jl if j["tool"] not in (None, "parse_error", "finish"))
        parse_errs = sum(1 for j in jl if j["tool"] == "parse_error")
        first_reply = next((t["content"] for t in tr if t.get("role") == "assistant"), "")
        return {"model": model, "status": "OK" if dispatched else "NO-DISPATCH",
                "dispatched": dispatched, "parse_errors": parse_errs,
                "steps": len(jl), "first_reply": first_reply[:240]}


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    steps = int(os.environ.get("STEPS", "3"))
    rows = []
    for m in models:
        print(f">>> smoking {m} ({steps} steps) ...", flush=True)
        r = smoke(m, steps)
        rows.append(r)
        detail = r.get("reason") or f"dispatched={r.get('dispatched')} parse_errors={r.get('parse_errors')}"
        print(f"    {r['status']}: {detail}", flush=True)
    print("\n================ SUMMARY ================")
    for r in rows:
        line = f"{r['status']:11} {r['model']}"
        if r["status"] in ("OK", "NO-DISPATCH"):
            line += f"  dispatched={r['dispatched']} parse_errors={r['parse_errors']} steps={r['steps']}"
        elif r.get("reason"):
            line += f"  ({r['reason']})"
        print(line)
    # show details for anything that didn't cleanly dispatch
    for r in rows:
        if r["status"] in ("FAIL", "NO-DISPATCH"):
            print(f"\n----- {r['model']} ({r['status']}) -----")
            if r.get("traceback"):
                print(r["traceback"][-1500:])
            if r.get("first_reply"):
                print("first_reply:", repr(r["first_reply"]))
    fails = [r for r in rows if r["status"] in ("FAIL", "NO-DISPATCH")]
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
