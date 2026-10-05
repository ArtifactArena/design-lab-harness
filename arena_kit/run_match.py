#!/usr/bin/env python3
"""Standalone match runner: red bot vs blue bot, both given as BOT REFS.

A bot ref is one of:
    draft         the robot.xml + controller.py at the workspace root
    stationary    the box (a zero-policy dummy)
    <name>        a saved bot under <workspace>/bots/<name>/

    python run_match.py --workspace . --red draft --blue stationary --n-seeds 3
    python run_match.py --workspace . --red wedge_v2 --blue wedge_v1   # saved vs saved
    python run_match.py --workspace . --red draft --blue draft         # self-play

On success it writes a match folder under <workspace>/matches/<match_id>/ and prints
the compact downsampled result (per-seed winner/termination/composite + a strided
replay view) plus the match_id and the path to the full match_data.json. On any
failure it prints a JSON error+traceback instead of crashing. stdout is pure JSON.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

# arena_kit/ (this dir) holds config.py + harness_lib.py — import as siblings.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import harness_lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run red bot vs blue bot (both bot refs).")
    ap.add_argument("--workspace", default=".", help="Workspace root (holds the draft + bots/ + matches/)")
    ap.add_argument("--red", default="draft", help="Red bot ref (draft|stationary|<name>)")
    ap.add_argument("--blue", default="stationary", help="Blue bot ref (draft|stationary|<name>)")
    ap.add_argument("--n-seeds", type=int, default=config.MATCH_SEEDS)
    ap.add_argument("--out", default="matches", help="Match-library dir (default: matches)")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    # Keep stdout pure JSON: route the engine's INFO prints to stderr.
    with contextlib.redirect_stdout(sys.stderr):
        result = harness_lib.run_match_core(
            workspace=ws, red_ref=args.red, blue_ref=args.blue,
            out_base=(ws / args.out), n_seeds=args.n_seeds,
        )
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
