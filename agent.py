#!/usr/bin/env python3
"""State-refresh agentic bot-building loop.

Each step the prompt is rebuilt from durable state — agent.md + the static docs +
the current draft + a one-line-per-step journal + the previous tool result — so the
context never grows unbounded. The model emits one or more fenced ```json tool calls
(carrying a short "note"); the harness dispatches it, appends a journal line, and
loops. Complete candidates are recorded with save_bot and selected by round robin.

Usage:
    ./run_agent.sh --max-steps 10                 # via the env wrapper (recommended)
    python agent.py --max-steps 10                # if mjarena + an API key are set
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "arena_kit"
RUNS = HERE / "runs"

# import config + harness_lib from the template
sys.path.insert(0, str(TEMPLATE))
import config  # noqa: E402
import harness_lib  # noqa: E402


# Destructive-operation guard for run_bash / run_python. A heuristic denylist — it
# stops the obviously catastrophic stuff (data loss, format, elevation, shutdown,
# remote code exec) but is NOT a security boundary (real isolation needs an OS
# sandbox). Blocked attempts are still logged like any other action.
_DESTRUCTIVE = re.compile(
    r"(?i)("
    r"\brm\b|\brmdir\b|\bshred\b|\bunlink\b"   # any rm/rmdir/shred/unlink (delete)
    r"|\bfind\b[^\n]*-delete\b"         # find ... -delete
    r"|\bdd\b.*\bof=|\bof=/dev/|>\s*/dev/"  # dd / writing to a device
    r"|\bmkfs\w*|\bfdisk\b|\bparted\b"
    r"|\bmv\b[^\n]*/dev/null"           # mv ... /dev/null  (silent delete)
    r"|\btruncate\s+-s"                 # truncate -s 0 (zero a file)
    r"|:\s*\(\s*\)\s*\{"                # :(){ ... }  fork bomb
    r"|\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b|\binit\s+[06]\b"
    r"|\bsudo\b|\bdoas\b|\bsu\s+-"
    r"|\bchmod\s+-R|\bchown\s+-R"
    r"|\bgit\s+reset\s+--hard|\bgit\s+clean\s+-[a-z]*f"  # destroy tracked/untracked files
    r"|\bkillall\b|\bpkill\b|\bkill\s+-9"
    r"|\bshutil\.rmtree\b|\bos\.removedirs\b|\bos\.(remove|unlink|rmdir)\b"
    r"|\.unlink\(|\.rmdir\("            # pathlib Path.unlink()/.rmdir()
    r"|\|\s*(sh|bash|zsh)\b"            # curl ... | sh  (pipe-to-shell / RCE)
    r")"
)


def destructive_guard(text: str):
    """Return a refusal string if `text` contains a blocked destructive op, else None."""
    m = _DESTRUCTIVE.search(text or "")
    if not m:
        return None
    return (f"ERROR: refused — destructive operation blocked (matched {m.group(0)!r}). "
            "This sandbox forbids deletion/format/elevation/shutdown/remote-exec. "
            "Work only inside your own workspace.")


# Workspace-escape guard for run_bash / run_python. The structured tools
# (read_file/grep/list_dir/write_file) are already path-confined to the run
# workspace, but run_bash/run_python execute arbitrary shell/python and could reach
# repo-root files OUTSIDE the workspace (agent.py, README, the eval framing) via `..`
# or $ARENA_REPO_ROOT. This blocks the obvious traversal; like the destructive guard
# it deters a wandering model but is NOT an airtight boundary — a determined process
# can still escape, and true isolation needs an OS sandbox (chroot / bwrap / container).
_ESCAPE = re.compile(
    r"\.\.[\\/]"                          # ../ or ..\  (parent-dir traversal)
    r"|(?<![.\w])\.\.(?![.\w])"          # a lone `..` token (e.g. cd ..)
    r"|\bARENA_REPO_ROOT\b"             # env var pointing at the repo root
)


def escape_guard(text: str):
    """Refuse run_bash/run_python that reaches outside the workspace via `..` etc."""
    if not _ESCAPE.search(text or ""):
        return None
    return ("ERROR: refused — this is a sandbox: run_bash / run_python run INSIDE your "
            "run workspace; reaching outside it (`..`, the repo root) is blocked. "
            "Everything you need is in the workspace — your draft, docs/, the engine "
            "source in reference/, assets/, and the harness libs. Use a "
            "workspace-relative path.")


# --------------------------------------------------------------------------- #
# Action parsing
# --------------------------------------------------------------------------- #
def reply_text(raw) -> str:
    """Normalize a dspy.LM completion to a string.

    dspy.LM returns a list of completions; with reasoning models each completion can
    be a dict like {'reasoning_content': ..., 'text': '...'} rather than a bare
    string, so pull out the visible text.
    """
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    if isinstance(raw, dict):
        return raw.get("text") or raw.get("content") or ""
    return str(raw)


def _json_objects(text: str):
    """Yield every balanced top-level {...} substring (string/escape aware).

    Fence-agnostic on purpose: models vary wildly — some fence the call, some emit
    bare JSON, some get cut off before the closing ``` — so we match braces, not
    fences. Truncated (unbalanced) objects are simply skipped.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        j = i
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    break
            j += 1
        i = (j + 1) if depth == 0 else (i + 1)


def _kimi_native_actions(text: str):
    """Parse Kimi/Moonshot NATIVE tool-call format into [(tool, args, note), ...].

    Tool-trained models (Kimi K2.x) emit their own special-token tool-call syntax
    instead of the ```json block the harness asks for, e.g.:
        <|tool_calls_section_begin|>
          <|tool_call_begin|>functions.read_file:0<|tool_call_argument_begin|>
            {"path": "docs/rules.yaml"}
          <|tool_call_end|>
          ...
        <|tool_calls_section_end|>
    The {...} after argument_begin holds ONLY the args (no "tool" key), so the
    generic JSON path skips it -> ~100% parse-error. Here we recover each call:
    tool = the name after 'functions.' (':<idx>' stripped), args = the JSON blob.
    """
    if "<|tool_call_begin|>" not in text:
        return []
    out = []
    for seg in text.split("<|tool_call_begin|>")[1:]:
        if "<|tool_call_argument_begin|>" not in seg:
            continue
        head, rest = seg.split("<|tool_call_argument_begin|>", 1)
        m = re.match(r"\s*(?:functions\.)?([\w.\-]+?)(?::\d+)?\s*$", head)
        if not m:
            continue
        tool = m.group(1)
        argstr = rest.split("<|tool_call_end|>", 1)[0].strip()
        # take the first balanced {...} so trailing markers/whitespace don't break json
        objs = list(_json_objects(argstr))
        try:
            args = json.loads(objs[0]) if objs else (json.loads(argstr) if argstr else {})
        except json.JSONDecodeError:
            continue
        out.append((tool, args if isinstance(args, dict) else {}, ""))
    return out


def parse_actions(text: str):
    """Return EVERY JSON tool call in the reply, in order, as [(tool, args, note), ...].

    Fence-agnostic and lenient: handles ```json blocks, bare JSON, prose-wrapped
    JSON, and calls cut off before the closing fence (as long as the JSON object
    itself is complete). A reply may contain several calls — they all run, in order.
    Also recovers Kimi/Moonshot native <|tool_call_begin|> tool-call syntax.
    """
    text = text or ""
    native = _kimi_native_actions(text)
    if native:
        return native
    actions = []
    for obj_str in _json_objects(text):
        try:
            obj = json.loads(obj_str)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "tool" in obj:
            actions.append((obj.get("tool"), (obj.get("args", {}) or {}), (obj.get("note", "") or "")))
    return actions


_TRUNCATED_MSG = (
    "Your previous API call failed because there was not enough output budget to "
    "finish the response. That call counts toward your turn budget. No tools from "
    "that response executed; your draft and saved bots are unchanged. Retry the "
    "intended work on this next turn. Reserve enough output tokens for complete "
    "tool calls; keep explanations concise and split large edits across turns. "
    "The configured output limit is unchanged."
)

_FORMAT_MSG = (
    "Your previous response had an invalid tool-call format. That API call counts "
    "toward your turn budget. No tools from that response executed; your draft and "
    "saved bots are unchanged. Reissue the intended actions on this next turn. "
    "Use plain-text fenced JSON blocks interpreted by the harness, not native API "
    "function calls. Each block must contain a JSON object with tool, args, and note. "
    "For example:\n```json\n"
    '{"tool":"does_bot_verify","args":{"ref":"draft"},"note":"Validate the draft"}'
    "\n```"
)


def response_actions(response):
    """Validate a whole response before allowing any of its tools to execute.

    Return (actions, recovery) for a successful or recoverable model response.
    Provider/billing/transport failures remain errors. Recovery consumes the call;
    the driver checkpoints it and sends the notice on the NEXT budgeted turn.
    """
    details = response.get("incomplete_details") or {}
    reason = response.get("stop_reason") or response.get("finish_reason")
    if not reason and isinstance(details, dict):
        reason = details.get("reason")
    reason = str(reason or "").upper()
    if reason in {"LENGTH", "MAX_TOKENS", "MAX_OUTPUT_TOKENS"}:
        return [], {"kind": "output_truncation", "reason": reason, "notice": _TRUNCATED_MSG}
    if reason in {"MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"}:
        return [], {"kind": "format_error", "reason": reason, "notice": _FORMAT_MSG}
    if response.get("status", "completed") != "completed":
        raise RuntimeError(f"Model status {response.get('status')}: {reason or response.get('error')}")
    text = response.get("output_text") or ""
    if _looks_truncated(text):
        return [], {"kind": "output_truncation", "reason": "INCOMPLETE_TOOL_JSON", "notice": _TRUNCATED_MSG}
    # Reject mixed valid/invalid JSON atomically, rather than applying half an edit.
    malformed = False
    for raw in _json_objects(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            if '"tool"' in raw:
                malformed = True
            continue
        if isinstance(obj, dict) and "tool" in obj:
            if not isinstance(obj["tool"], str) or not isinstance(obj.get("args", {}), dict):
                malformed = True
    actions = parse_actions(text) if not malformed else []
    if not actions:
        return [], {"kind": "format_error", "reason": "INVALID_TOOL_JSON", "notice": _FORMAT_MSG}
    return actions, None


def lm_response_metadata(lm, text):
    """Read completion status from DSPy's latest response when available."""
    history = getattr(lm, "history", None) or []
    raw = history[-1].get("response", {}) if history else {}
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    raw = raw if isinstance(raw, dict) else {}
    choices = raw.get("choices") or []
    reason = choices[0].get("finish_reason") if choices else raw.get("stop_reason")
    return {"output_text": text, "status": raw.get("status") or "completed",
            "stop_reason": reason, "incomplete_details": raw.get("incomplete_details")}


def _looks_truncated(text: str) -> bool:
    """True if the reply ends with an unbalanced {...} that names a tool — i.e. a tool
    call cut off by the model's output cap (it parsed to nothing and got dropped)."""
    objs = list(_json_objects(text or ""))
    end = (text.rfind(objs[-1]) + len(objs[-1])) if objs else 0
    tail = text[end:]
    brace = tail.find("{")
    if brace == -1:
        return False
    rest = tail[brace:]
    return '"tool"' in rest and rest.count("{") > rest.count("}")


def _truncate(s, limit: int) -> str:
    s = s if isinstance(s, str) else str(s)
    if len(s) <= limit:
        return s
    return (s[:limit] + f"\n... [truncated to fit the model's context: {len(s) - limit} "
            "more chars — re-read with offset/limit, or use analyze_match for full match data]")


def _context_chars(model_config_path) -> int:
    """Best-effort INPUT context budget (in chars) for the driving model.

    Uses litellm's per-model max_input_tokens (the real context window — NOT the
    config's max_tokens, which is the output cap). Falls back to 128k tokens if the
    model is unknown to litellm. ~4 chars/token. This is what makes the cap fair:
    each model is bounded by its own context, not a magic number.
    """
    tokens = None
    try:
        import yaml
        cfg = yaml.safe_load(Path(model_config_path).read_text()) or {}
        model = cfg.get("model", "")
        import litellm
        info = litellm.get_model_info(model) or {}
        tokens = info.get("max_input_tokens") or info.get("max_tokens")
    except Exception:  # noqa: BLE001
        tokens = None
    return int((tokens or 128000) * 4)


# --------------------------------------------------------------------------- #
# Tools — bound to a single run workspace
# --------------------------------------------------------------------------- #
class Tools:
    def __init__(self, workspace: Path):
        self.ws = Path(workspace).resolve()

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.ws / p)

    def _in_workspace(self, p: Path) -> bool:
        try:
            p.resolve().relative_to(self.ws)
            return True
        except ValueError:
            return False

    # -- read/inspect: confined to YOUR run workspace (your own creations only) --
    _OUTSIDE = ("ERROR: refused — this is a sandbox. read/list/grep are restricted to "
                "YOUR run workspace (your draft, docs/, reference/, and the bots/matches "
                "you create). You cannot access other runs' or models' creations: {path}")

    def list_dir(self, path: str = ".") -> str:
        d = self._resolve(path)
        if not self._in_workspace(d):
            return self._OUTSIDE.format(path=path)
        if not d.is_dir():
            return f"ERROR: not a directory: {path}"
        return "\n".join(sorted(
            (f"{p.name}/" if p.is_dir() else p.name) for p in d.iterdir())) or "(empty)"

    def read_file(self, path: str, offset: int = 0, limit: int = None) -> str:
        f = self._resolve(path)
        if not self._in_workspace(f):
            return self._OUTSIDE.format(path=path)
        if not f.is_file():
            return f"ERROR: not a file: {path}"
        text = f.read_text(errors="replace")
        if offset or limit is not None:
            lines = text.splitlines()
            end = (offset + limit) if limit is not None else len(lines)
            text = "\n".join(lines[offset:end])
        if len(text) > config.MAX_READ:
            text = text[:config.MAX_READ] + (
                f"\n... [truncated at {config.MAX_READ} chars — re-read with "
                '"offset"/"limit" (lines) to page through the rest]')
        return text

    def grep(self, pattern: str, path: str = ".") -> str:
        root = self._resolve(path)
        if not self._in_workspace(root):
            return self._OUTSIDE.format(path=path)
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            return f"ERROR: bad pattern: {exc}"
        hits = []
        files = [root] if root.is_file() else root.rglob("*")
        for f in files:
            if not f.is_file():
                continue
            try:
                for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                    if rx.search(line):
                        rel = f.relative_to(self.ws) if self._in_workspace(f) else f
                        hits.append(f"{rel}:{i}: {line.strip()}")
                        if len(hits) >= 200:
                            return "\n".join(hits) + "\n... [more matches omitted]"
            except (OSError, UnicodeError):
                continue
        return "\n".join(hits) or "(no matches)"

    # -- write: workspace only --
    def write_file(self, path: str, content: str) -> str:
        f = self._resolve(path)
        if not self._in_workspace(f):
            return f"ERROR: refused — writes are restricted to the workspace ({path})"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return f"wrote {len(content)} chars to {path}"

    # -- arbitrary python (inline snippet or a .py file in the workspace) --
    def run_python(self, code: str = None, path: str = None) -> str:
        if path:
            target = self._resolve(path)
            if not self._in_workspace(target):
                return self._OUTSIDE.format(path=path)
            src = target.read_text(errors="replace") if target.is_file() else ""
            blocked = destructive_guard(src) or escape_guard(src)
            if blocked:
                return blocked
            cmd = [sys.executable, str(target)]
        elif code is not None:
            blocked = destructive_guard(code) or escape_guard(code)
            if blocked:
                return blocked
            cmd = [sys.executable, "-c", code]
        else:
            return "ERROR: run_python needs 'code' or 'path'"
        return self._subprocess(cmd)

    # -- shell: arbitrary commands, confined to the workspace --
    def run_bash(self, cmd: str) -> str:
        if not cmd:
            return "ERROR: run_bash needs 'cmd'"
        blocked = destructive_guard(cmd) or escape_guard(cmd)
        if blocked:
            return blocked
        return self._subprocess(["bash", "-lc", cmd])

    def _subprocess(self, cmd) -> str:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(config.REPO_ROOT) + (os.pathsep + existing if existing else "")
        try:
            proc = subprocess.run(cmd, cwd=str(self.ws), env=env, timeout=600,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
            return ("ERROR: command took too long and hit its timeout (600s). "
                    "The run will continue; edits already made may persist.\n"
                    "[partial stdout]\n" + decoded(exc.stdout) +
                    "\n[partial stderr]\n" + decoded(exc.stderr))
        out = proc.stdout
        if proc.stderr:
            out += "\n[stderr]\n" + proc.stderr
        return out or "(no output)"

    # -- verification: validation only (no box match) --
    def does_bot_verify(self, ref: str = "draft") -> str:
        r = harness_lib.resolve_bot_ref(ref, self.ws)
        robot = r.xml_path.read_text() if r.xml_path.is_file() else ""
        ctrl = r.ctrl_path.read_text() if (r.ctrl_path and r.ctrl_path.is_file()) else ""
        return json.dumps(harness_lib.run_verification_core(robot, ctrl), indent=2)

    # -- validation + qualification (two bools) --
    def does_bot_qualify(self, ref: str = "draft") -> str:
        r = harness_lib.resolve_bot_ref(ref, self.ws)
        robot = r.xml_path.read_text() if r.xml_path.is_file() else ""
        ctrl = r.ctrl_path.read_text() if (r.ctrl_path and r.ctrl_path.is_file()) else ""
        return json.dumps(harness_lib.run_qualification_core(
            robot, ctrl, self.ws / "_work" / "qualify"), indent=2)

    # -- match: red ref vs blue ref --
    def run_match(self, red: str = "draft", blue: str = "stationary", n_seeds: int = None) -> str:
        return json.dumps(harness_lib.run_match_core(
            self.ws, red, blue, self.ws / "matches", n_seeds), indent=2)

    def list_matches(self) -> str:
        return json.dumps(harness_lib.list_matches_core(self.ws / "matches"), indent=2)

    def get_match(self, match_id: str) -> str:
        return json.dumps(harness_lib.get_match_core(self.ws / "matches", match_id), indent=2)

    def analyze_match(self, match_id: str, code: str) -> str:
        return json.dumps(harness_lib.analyze_match_core(
            self.ws / "matches", match_id, code), indent=2)

    def diagnose_physics(self, ref: str = "draft") -> str:
        r = harness_lib.resolve_bot_ref(ref, self.ws)
        robot = r.xml_path.read_text() if r.xml_path.is_file() else ""
        return json.dumps(harness_lib.diagnose_physics_core(robot), indent=2)

    def probe_obs(self, ref: str = "draft") -> str:
        return json.dumps(harness_lib.probe_obs_core(self.ws, ref), indent=2)

    # -- bot library --
    def save_bot(self, name: str, source: str = "draft", overwrite: bool = False, **design) -> str:
        # Any extra args (summary, design_strategy, hardware_plan, combat_plan,
        # reasoning, …) are captured as the model's design intent in the artifact.
        return json.dumps(harness_lib.save_bot(
            self.ws, name, source_ref=source, overwrite=overwrite, design=design or None), indent=2)

    def list_bots(self) -> str:
        return json.dumps(harness_lib.list_bots(self.ws), indent=2)

    def get_bot(self, name: str) -> str:
        return json.dumps(harness_lib.get_bot(self.ws, name), indent=2)

    # -- submission: designate the final bot (copies it into out/) --
    def submit(self, ref: str = "draft") -> str:
        return json.dumps(harness_lib.submit_core(self.ws, ref), indent=2)


_FNS = ("list_dir", "read_file", "grep", "write_file", "run_python", "run_bash",
        "does_bot_verify", "does_bot_qualify", "run_match", "list_matches", "get_match",
        "analyze_match", "diagnose_physics", "probe_obs", "save_bot", "list_bots",
        "get_bot")


# Forgive cross-model arg drift: wrong key -> canonical key (applied only when the
# canonical key is absent). Models guess synonyms constantly.
_PATH = {"filename": "path", "file": "path", "filepath": "path", "file_path": "path",
         "dir": "path", "directory": "path"}
_ALIASES = {
    "read_file": _PATH, "write_file": _PATH, "list_dir": _PATH,
    "grep": {**_PATH, "query": "pattern", "regex": "pattern"},
    "run_match": {"opponent": "blue", "against": "blue", "enemy": "blue",
                  "num_seeds": "n_seeds", "seeds": "n_seeds"},
    "does_bot_verify": {"name": "ref", "bot": "ref"},
    "does_bot_qualify": {"name": "ref", "bot": "ref"},
    "diagnose_physics": {"name": "ref", "bot": "ref"},
    "probe_obs": {"name": "ref", "bot": "ref"},
    "analyze_match": {"id": "match_id", "match": "match_id"},
    "get_match": {"id": "match_id", "match": "match_id"},
    "save_bot": {"bot_name": "name", "from": "source", "from_ref": "source"},
    "get_bot": {"bot": "name", "bot_name": "name"},
    "submit": {"name": "ref", "bot": "ref"},
}

# Coerce values models send as the wrong type (numbers/bools as strings, etc.).
def _to_int(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        try:
            return int(float(str(v).strip()))
        except (ValueError, TypeError):
            return v

def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "on")
    return bool(v)

def _to_str(v):
    return v if isinstance(v, str) else ("" if v is None else str(v))

_COERCE = {
    "run_match": {"n_seeds": _to_int},
    "read_file": {"offset": _to_int, "limit": _to_int},
    "save_bot": {"overwrite": _to_bool},
    "write_file": {"content": _to_str},
}


def _normalize_args(tool: str, args: dict) -> dict:
    """Forgive arg-name drift (aliases) and arg-type drift (string numbers/bools)."""
    if not isinstance(args, dict):
        # The model emitted args that aren't a JSON object (e.g. a bare list/string).
        # Don't let one malformed call crash the whole run — return empty so dispatch
        # surfaces a clear bad-args error the model can correct on the next step.
        return {}
    args = dict(args)
    for wrong, right in _ALIASES.get(tool, {}).items():
        if wrong in args and right not in args:
            args[right] = args.pop(wrong)
    for key, cast in _COERCE.get(tool, {}).items():
        if key in args:
            args[key] = cast(args[key])
    return args


def dispatch(tools: Tools, tool: str, args: dict) -> str:
    if tool not in _FNS:
        return f"ERROR: unknown tool '{tool}'"
    try:
        # No cap here — the full result is returned (and logged). It's truncated only
        # to fit the model's context when assembled into the next prompt (build_state).
        return getattr(tools, tool)(**_normalize_args(tool, dict(args)))
    except TypeError as exc:
        return f"ERROR: bad args for {tool}: {exc}"
    except Exception:  # noqa: BLE001
        return "ERROR running tool:\n" + traceback.format_exc()


# --------------------------------------------------------------------------- #
# Journal / state assembly
# --------------------------------------------------------------------------- #
def _arg_hint(tool: str, args: dict) -> str:
    if tool in ("write_file", "read_file", "list_dir", "grep"):
        return " " + str(args.get("path", args.get("pattern", "")))
    if tool == "run_match":
        return f" {args.get('red', 'draft')} vs {args.get('blue', 'stationary')}"
    if tool in ("save_bot", "get_bot"):
        return " " + str(args.get("name", ""))
    if tool in ("analyze_match", "get_match"):
        return " " + str(args.get("match_id", ""))
    if tool in ("does_bot_verify", "does_bot_qualify", "diagnose_physics", "probe_obs", "submit"):
        return " " + str(args.get("ref", "draft"))
    return ""


def _summarize(tool: str, result: str) -> str:
    """One-line journal outcome from a tool result."""
    try:
        obj = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        first = result.strip().splitlines()[0] if result.strip() else "ok"
        return first[:140]
    if isinstance(obj, list):
        return f"{len(obj)} item(s)"
    if not isinstance(obj, dict):
        return "ok"
    if obj.get("ok") is False:
        return "error: " + str(obj.get("error", ""))[:120]
    if tool == "does_bot_verify":
        return f"verification{'✓' if obj.get('verification_passed') else '✗'}"
    if tool == "does_bot_qualify":
        v = "✓" if obj.get("validation_passed") else "✗"
        q = "✓" if obj.get("qualification_passed") else "✗"
        return f"validation{v} qualification{q}"
    if tool == "run_match":
        return (f"{obj.get('outcome')} {obj.get('wins')}-{obj.get('losses')}-{obj.get('draws')} "
                f"match_id={obj.get('match_id')}")
    if tool == "save_bot":
        return f"saved {obj.get('name')} (qual={'✓' if obj.get('qualification_passed') else '✗'})"
    return "ok"


def _qual_status(result: str):
    """Parse a does_bot_qualify result into {validation, qualification} bools, or None."""
    try:
        obj = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(obj, dict) and obj.get("ok") is not False:
        return {"validation": bool(obj.get("validation_passed")),
                "qualification": bool(obj.get("qualification_passed"))}
    return None


def _read_if(p: Path) -> str:
    p = Path(p).absolute()
    if p.resolve() != p:
        return "(unavailable: symbolic links are not read by the API driver)"
    return p.read_text(errors="replace") if p.is_file() else ""


def _draft_fp(run_dir: Path) -> str:
    """Short fingerprint of the draft (robot.xml + controller.py) to detect edits."""
    import hashlib
    r = _read_if(Path(run_dir) / "robot.xml")
    c = _read_if(Path(run_dir) / "controller.py")
    return hashlib.sha1((r + "\x00" + c).encode("utf-8", "replace")).hexdigest()[:12]


def build_state(run_dir: Path, journal, last_result: str, latest_qual=None,
                latest_qual_fp=None, result_budget: int = 200000,
                step: int | None = None, max_steps: int | None = None) -> str:
    run_dir = Path(run_dir)
    # Cap model-grown content so it can't push the prompt past the model's context.
    # These are far above any real bot/notes; full files remain on disk (read_file).
    def _cap(text, limit, what):
        if len(text) <= limit:
            return text
        return (text[:limit] + f"\n... [{what} capped at {limit} chars in the prompt — "
                f"{len(text) - limit} more on disk; read_file it for the rest]")

    parts = []
    # Budget banner FIRST — anchor the model in the loop so it builds incrementally
    # instead of (as some models do on a blank workspace) narrating a finished bot on
    # call 1. Each reply == one LLM call; at the cap the run finalizes automatically.
    if step is not None and max_steps is not None:
        remaining = max(0, max_steps - step)
        parts += [
            f"== LLM CALL {step}/{max_steps} ==",
            f"You have {remaining} more LLM call(s) remaining before `finish` is "
            f"automatically called and the run ends (at the cap the harness finalizes "
            f"and selects a saved bot through the round robin). Each reply you send is "
            f"one LLM call — build incrementally; do NOT claim a bot is done until you "
            f"have actually written robot.xml + controller.py and saved/verified it.",
            "",
        ]

    robot = _cap(_read_if(run_dir / "robot.xml") or "(empty — write your robot here)", 60000, "robot.xml")
    ctrl = _cap(_read_if(run_dir / "controller.py") or "(empty — write your controller here)", 60000, "controller.py")
    parts += [
        "== YOUR CURRENT DRAFT (use save_bot to enter it in selection) ==",
        "--- robot.xml ---", robot,
        "--- controller.py ---", ctrl,
    ]
    notes = _read_if(run_dir / "notes.md")
    if notes.strip():
        parts += ["", "== notes.md (your scratchpad) ==", _cap(notes, 24000, "notes.md")]
    if latest_qual:
        stale = latest_qual_fp is not None and latest_qual_fp != _draft_fp(run_dir)
        if stale:
            line = "(draft edited since last check — run does_bot_qualify to refresh)"
        else:
            v = "✓" if latest_qual.get("validation") else "✗"
            q = "✓" if latest_qual.get("qualification") else "✗"
            line = f"validation {v}   qualification {q}"
        parts += ["", "== CURRENT DRAFT STATUS ==", line]
    bots = harness_lib.list_bots(run_dir)
    parts += ["", "== SAVED BOTS =="]
    if bots:
        parts += [f"  {b['name']}: validation={b.get('validation_passed')} "
                  f"qualification={b.get('qualification_passed')}"
                  + (f"  — {b['summary']}" if b.get("summary") else "")
                  for b in bots]
    else:
        parts.append("  (none yet — save_bot a good draft)")
    parts += ["", "== JOURNAL (your past steps) =="]
    window = journal[-config.JOURNAL_WINDOW:]
    if len(journal) > len(window):
        parts.append(f"  ... ({len(journal) - len(window)} earlier steps elided)")
    for e in window:
        line = f"  {e['step']:>3}  {e['tool']}{_arg_hint(e['tool'], e.get('args', {}))} → {e['outcome']}"
        if e.get("note"):
            note = e["note"] if len(e["note"]) <= 200 else e["note"][:200] + "…"
            line += f"   [{note}]"
        parts.append(line)
    if not window:
        parts.append("  (no steps yet)")
    parts += ["", "== RESULT(S) OF YOUR LAST STEP ==", _truncate(last_result, result_budget), "",
              'Decide your next action. Think briefly, then end with one or more fenced '
              '```json tool calls (each with a short "note"). Multiple calls run in '
              'order and you get every result back — batch independent calls; use '
              'separate steps when a call needs a previous result (e.g. a match_id).']
    return "\n".join(parts)


def _engine_placeholders() -> dict:
    """Single-brace placeholders the harness fills at prompt-render time (the
    sampling harness does this in unified_builder; the model must never see the
    literal `{mujoco_version}`)."""
    import mujoco
    return {"mujoco_version": mujoco.__version__}


def fill_engine_placeholders(text: str) -> str:
    for key, value in _engine_placeholders().items():
        text = text.replace("{" + key + "}", value)
    return text


# Workspace docs that carry engine placeholders; filled once when the run is created.
_PLACEHOLDER_DOCS = ("docs/sampling_prompt.md", "docs/mjcf_syntax.yaml")


def build_system_prompt(run_dir: Path, step: int = 1, max_steps: int = 10) -> str:
    """Render the sampling prompt (engine placeholders filled) followed by the
    tool-use instructions."""
    run_dir = Path(run_dir)
    template = _read_if(run_dir / "agent.md")
    for key, value in {
        "model_turn_number": str(step),
        "model_turns_remaining_after_this": str(max(0, max_steps - step)),
        "model_turn_budget": str(max_steps),
    }.items():
        template = template.replace("{{" + key + "}}", value)
    sampling = fill_engine_placeholders(_read_if(run_dir / "docs/sampling_prompt.md"))
    return template.replace("{{sampling_prompt}}", sampling)


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #
def new_run_dir() -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Resolved: the API driver's symlink guards compare resolve() with absolute(), so
    # an unresolved path under a symlinked runs/ would reject every tool result.
    d = RUNS.resolve() / f"{ts}-{uuid.uuid4().hex[:6]}"  # suffix avoids same-second collisions
    shutil.copytree(TEMPLATE, d)
    for rel in _PLACEHOLDER_DOCS:  # the model reads these with read_file
        doc = d / rel
        if doc.is_file():
            doc.write_text(fill_engine_placeholders(doc.read_text()))
    return d


def run(max_steps: int) -> Path:
    run_dir = new_run_dir()
    import linux_sandbox
    runtime = linux_sandbox.prepare(run_dir)
    transcript = run_dir / "transcript.jsonl"
    journal_path = run_dir / "journal.jsonl"

    harness_lib._mj()  # initialize mjarena in an order that avoids a circular import
    from mjarena.dspy_core import configure_lm
    # use_cache=False: every run must make fresh calls so repeated runs of the same
    # model explore independently (identical step-1 prompts would otherwise cache-hit).
    lm = configure_lm(str(config.MODEL_CONFIG), use_cache=False)
    system = build_system_prompt(runtime / "prompt", max_steps=max_steps)
    # Fair, model-specific cap: the whole prompt must fit the driving model's context
    # window. We leave a margin, then give the last tool result whatever context is
    # left after the (fixed) system prompt + the rest of the state.
    prompt_budget = max(40000, _context_chars(config.MODEL_CONFIG) - 12000)

    def log(fp: Path, obj):
        fd = os.open(fp, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    log(transcript, {"role": "system", "content": system})

    journal = []
    last_result = "(no actions yet — inspect the workspace and start building your bot)"
    latest_qual = None
    latest_qual_fp = None
    finish_summary = None
    step = 0
    for step in range(1, max_steps + 1):
        system = build_system_prompt(runtime / "prompt", step=step, max_steps=max_steps)
        shell = build_state(run_dir, journal, "", latest_qual, latest_qual_fp, result_budget=0,
                            step=step, max_steps=max_steps)
        result_budget = max(8000, prompt_budget - len(system) - len(shell))
        state = build_state(run_dir, journal, last_result, latest_qual, latest_qual_fp,
                            result_budget=result_budget, step=step, max_steps=max_steps)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": state}]
        reply = reply_text(lm(messages=messages))
        log(transcript, {"step": step, "role": "assistant", "content": reply})

        actions = parse_actions(reply)
        truncated = _looks_truncated(reply)
        if not actions:
            msg = (_TRUNCATED_MSG if truncated else
                   ("ERROR: no valid json tool call found. End your turn with at least one "
                    'fenced ```json {"tool":...,"args":...,"note":...} block (you may emit '
                    "several to run in order)."))
            entry = {"step": step, "tool": "parse_error", "args": {}, "note": "",
                     "outcome": "truncated tool call" if truncated else "parse error: no tool call"}
            journal.append(entry)
            log(journal_path, entry)
            log(transcript, {"step": step, "role": "tool", "content": msg})
            last_result = msg
            continue

        # Run every tool call in the reply, in order; collect all results.
        blocks = []
        finished = False
        for tool, args, note in actions:
            if tool == "finish":
                # Only saved candidates can enter the final selection tournament.
                has_saved = bool(harness_lib.list_bots(run_dir))
                if not has_saved:
                    msg = ("⚠ finish REFUSED — you have not built a bot yet: robot.xml / "
                           "controller.py have not been recorded with save_bot. Write the "
                           "draft (write_file robot.xml + controller.py), check it with "
                           "does_bot_verify / does_bot_qualify, save_bot, THEN finish. Do "
                           "not claim a bot is done before you have actually built it.")
                    entry = {"step": step, "tool": "finish", "args": {}, "note": note,
                             "outcome": "finish refused: nothing built"}
                    journal.append(entry)
                    log(journal_path, entry)
                    log(transcript, {"step": step, "role": "tool", "content": msg})
                    blocks.append(f"[finish] → finish refused: nothing built\n{msg}")
                    continue
                finish_summary = args.get("summary", "")
                entry = {"step": step, "tool": "finish", "args": {}, "note": note,
                         "outcome": "finished"}
                journal.append(entry)
                log(journal_path, entry)
                log(transcript, {"step": step, "role": "finish", "content": finish_summary})
                finished = True
                break
            args = _normalize_args(tool, args)  # canonical keys for dispatch, journal, and stamping
            try:
                result = linux_sandbox.execute(runtime, run_dir, {"tool": tool, "args": args})
            except Exception as exc:
                result = "ERROR: " + str(exc)
            outcome = _summarize(tool, result)
            if tool == "does_bot_qualify":
                qs = _qual_status(result)
                # Only the DRAFT's status is shown as CURRENT DRAFT STATUS; stamp the
                # draft fingerprint so we can flag it stale once the draft is edited.
                if qs and args.get("ref", "draft") in ("draft", "current"):
                    latest_qual = qs
                    latest_qual_fp = _draft_fp(run_dir)
            entry = {"step": step, "tool": tool, "args": args, "note": note, "outcome": outcome}
            journal.append(entry)
            log(journal_path, entry)
            log(transcript, {"step": step, "role": "tool", "tool": tool, "content": result})
            blocks.append(f"[{tool}{_arg_hint(tool, args)}] → {outcome}\n{result}")

        if blocks:
            last_result = ("\n\n".join(blocks) if len(blocks) > 1
                           else blocks[0])
            if len(actions) > 1:
                last_result = (f"(ran {len(blocks)} tool calls this step)\n\n" + last_result)
            if truncated:  # some calls ran, but a trailing one was cut off — say so
                last_result += "\n\n" + _TRUNCATED_MSG
        if finished:
            break

    linux_sandbox.execute(runtime, run_dir, {"operation": "finalize",
        "summary": finish_summary, "steps": step}, timeout=86400)
    return run_dir


def _finalize(run_dir: Path, summary, steps: int):
    """Write the selection-tournament winner, or report a selection failure."""
    final = run_dir / "final"
    final.mkdir(exist_ok=True)
    try:
        sub = harness_lib.resolve_submission_core(run_dir)
    except Exception as exc:
        sub = {"how": "selection_failed", "name": None, "error": str(exc),
               "robot_xml": "", "controller_py": "", "artifact": {}}
    (final / "robot.xml").write_text(sub.get("robot_xml") or "")
    (final / "controller.py").write_text(sub.get("controller_py") or "")
    (final / "bot_artifact.json").write_text(json.dumps(sub.get("artifact") or {}, indent=2))
    (run_dir / "summary.json").write_text(json.dumps({
        "steps": steps,
        "finish_summary": summary,
        "finished": summary is not None,
        "deliverables_present": {
            "robot.xml": bool((sub.get("robot_xml") or "").strip()),
            "controller.py": bool((sub.get("controller_py") or "").strip()),
        },
        "submission": {"how": sub.get("how"), "name": sub.get("name"),
                       "ranking": sub.get("ranking"), "error": sub.get("error")},
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description="State-refresh bot-building agent loop")
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    args = ap.parse_args()
    run_dir = run(args.max_steps)
    print(f"Run complete -> {run_dir}")
    print(f"Final artifact -> {run_dir / 'final'}")


if __name__ == "__main__":
    main()
