"""Run model tools in a fresh Linux namespace; the API client stays outside."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import uuid
from tool_results import timeout_result

ROOT = Path(__file__).resolve().parent


def prepare(workspace: Path) -> Path:
    if platform.system() != "Linux":
        raise RuntimeError("Model tool runs require Linux namespaces. Launch on a Linux host.")
    for command in ("unshare", "mount", "pivot_root", "setpriv"):
        if not shutil.which(command):
            raise RuntimeError(f"Tool isolation requires {command}; refusing an unisolated run")
    runtime = workspace.parent / (workspace.name + "-runtime")
    runtime.mkdir(exist_ok=False)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for name in ("mjarena", "configs"):
        shutil.copytree(ROOT / name, runtime / "engine" / name, ignore=ignore)
    code = runtime / "code"
    (code / "arena_kit").mkdir(parents=True)
    for name in ("agent.py", "tool_worker.py", "tool_results.py"):
        shutil.copy2(ROOT / name, code / name)
    for name in ("config.py", "harness_lib.py"):
        shutil.copy2(ROOT / "arena_kit" / name, code / "arena_kit" / name)
    shutil.copy2(ROOT / "sandbox.sh", runtime / "sandbox.sh")
    (runtime / "prompt/docs").mkdir(parents=True)
    shutil.copy2(ROOT / "arena_kit/agent.md", runtime / "prompt/agent.md")
    shutil.copy2(ROOT / "configs/rules/sampling_prompt.md", runtime / "prompt/docs/sampling_prompt.md")
    # Resolve symlinked environments before mounting so /venv is self-contained.
    (runtime / "runtime.json").write_text(json.dumps({
        "python_env": str(Path(sys.prefix).resolve()),
        "source": str(ROOT),
        "upstream": json.loads((ROOT / "upstream_environment.json").read_text())["commit"],
    }, indent=2))
    execute(runtime, workspace, {"operation": "probe"}, timeout=60)
    return runtime


def execute(runtime: Path, workspace: Path, request: dict, timeout: int = 3700):
    """One sandbox per tool call. Never fall back to host execution."""
    request = dict(request)
    if request.get("tool"):
        request["_progress_dir"] = "_tool_progress/" + uuid.uuid4().hex
    request_path = workspace / "_tool_request.json"
    fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(json.dumps(request))
    result_path = workspace / "_tool_result.json"
    result_path.unlink(missing_ok=True)
    settings = json.loads((runtime / "runtime.json").read_text())
    with (runtime / "tools.log").open("a") as log:
        log_start = log.tell()
        process = subprocess.Popen([
            "bash", str(runtime / "sandbox.sh"), str(runtime), str(workspace),
            settings["python_env"],
        ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if not request.get("tool"):
                raise TimeoutError(f"Tool exceeded {timeout} seconds")
            with (runtime / "tools.log").open("rb") as output:
                output.seek(max(log_start, (runtime / "tools.log").stat().st_size - 8000))
                partial_output = output.read().decode(errors="replace")
            return json.dumps(timeout_result(workspace, request, timeout, log_tail=partial_output))
    if code or not result_path.is_file():
        raise RuntimeError(f"Sandbox worker failed (exit {code}); see {runtime / 'tools.log'}")
    if result_path.resolve() != result_path.absolute():
        raise RuntimeError("Sandbox result must be a regular file in the run workspace")
    result = json.loads(result_path.read_text())
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["result"]
