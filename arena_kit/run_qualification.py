#!/usr/bin/env python3
"""Standalone validation + qualification for a bot.

Runs the real pipeline (morphology + controller static checks = validation; then a
qualifying round vs the stationary box = qualification) and prints:

    {"ok": true, "validation_passed": bool, "qualification_passed": bool,
     "feedback": "..."}

By default it qualifies the DRAFT (robot.xml + controller.py at --workspace). Point
--ref at a saved bot name to qualify that one instead. stdout is pure JSON.

    python run_qualification.py --workspace .
    python run_qualification.py --workspace . --ref wedge_v2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import harness_lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate + qualify a bot (prints two bools).")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--ref", default="draft", help="Bot ref to qualify (draft|<name>)")
    ap.add_argument("--out", default="_work/qualify")  # out/ is reserved for the submission
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    try:
        ref = harness_lib.resolve_bot_ref(args.ref, ws)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    robot = ref.xml_path.read_text() if ref.xml_path.is_file() else ""
    ctrl = ref.ctrl_path.read_text() if (ref.ctrl_path and ref.ctrl_path.is_file()) else ""

    with contextlib.redirect_stdout(sys.stderr):
        res = harness_lib.run_qualification_core(robot, ctrl, ws / args.out)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
