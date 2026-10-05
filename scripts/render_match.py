#!/usr/bin/env python3
"""Render a match_data.json.

Default (no deps but matplotlib): a PNG with top-down red/blue trajectories +
edge-distance + tipping, one row of panels per seed.

    python scripts/render_match.py <match_data.json> [out.png]

WEBM (a real 3D playback): replays the saved per-frame qpos into the match's
composed.xml and encodes a video. Needs a GL backend (MUJOCO_GL=glfw on macOS,
egl on a Linux box) + ffmpeg.

    MUJOCO_GL=glfw python scripts/render_match.py --webm <match_data.json> [out.webm] [seed]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402


def render(md_path, out_path=None):
    md_path = Path(md_path)
    data = json.loads(md_path.read_text())
    seeds = sorted((k for k in data if k.startswith("seed_")), key=lambda k: int(k.split("_")[1]))
    if not seeds:
        raise SystemExit(f"no seeds in {md_path}")
    out_path = Path(out_path) if out_path else md_path.with_suffix(".png")

    fig, axes = plt.subplots(len(seeds), 3, figsize=(13, 4.2 * len(seeds)), squeeze=False)
    for r, k in enumerate(seeds):
        s = data[k]
        rp = s.get("red_positions") or []
        bp = s.get("blue_positions") or []
        ring = float(s.get("ring_radius") or 7.5)
        winner = s.get("winner"); term = s.get("termination_reason"); n = s.get("num_steps")

        # top-down trajectories
        ax = axes[r][0]
        ax.add_patch(Circle((0, 0), ring, fill=False, color="#9B958A", lw=1.5))
        if rp:
            ax.plot([p[0] for p in rp], [p[1] for p in rp], color="#D9443A", lw=1.4, label="red")
            ax.plot(rp[0][0], rp[0][1], "o", color="#D9443A"); ax.plot(rp[-1][0], rp[-1][1], "X", color="#7a1f1a")
        if bp:
            ax.plot([p[0] for p in bp], [p[1] for p in bp], color="#2C6FB0", lw=1.4, label="blue")
            ax.plot(bp[0][0], bp[0][1], "o", color="#2C6FB0"); ax.plot(bp[-1][0], bp[-1][1], "X", color="#163e63")
        ax.set_aspect("equal"); ax.set_xlim(-ring * 1.15, ring * 1.15); ax.set_ylim(-ring * 1.15, ring * 1.15)
        ax.set_title(f"{k}: winner={winner}  {term}  ({n} steps)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)

        # edge distance over time (positive = inside ring)
        ax = axes[r][1]
        red_e = s.get("red_edge_distances") or []
        blue_e = s.get("blue_edge_distances") or []
        if red_e: ax.plot(red_e, color="#D9443A", label="red edge")
        if blue_e: ax.plot(blue_e, color="#2C6FB0", label="blue edge")
        ax.axhline(0, color="#9B958A", lw=0.8, ls="--")
        ax.set_title("edge distance (>0 = inside)", fontsize=10); ax.legend(fontsize=8)

        # tipping over time (0 upright, 1 fallen)
        ax = axes[r][2]
        rt = s.get("red_tipping") or []
        bt = s.get("blue_tipping") or []
        if rt: ax.plot(rt, color="#D9443A", label="red tip")
        if bt: ax.plot(bt, color="#2C6FB0", label="blue tip")
        ax.set_ylim(-0.05, 1.05); ax.set_title("tipping (0 upright, 1 fallen)", fontsize=10); ax.legend(fontsize=8)

    fig.suptitle(f"{md_path.parent.parent.name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(out_path)
    return out_path


def render_webm(md_path, out_path=None, seed=0, fps=30, size=(640, 480)):
    """Replay the seed's qpos into the match's composed.xml and encode a webm."""
    import os
    if not os.environ.get("MUJOCO_GL"):  # setdefault would leave an explicit empty value (physics-only)
        os.environ["MUJOCO_GL"] = "glfw"
    import mujoco
    import numpy as np
    import imageio.v2 as iio

    md_path = Path(md_path)
    data = json.loads(md_path.read_text())
    avail = sorted(k for k in data if k.startswith("seed_"))
    key = f"seed_{seed}"
    if key not in data:
        raise SystemExit(f"no {key} in {md_path} (have: {avail or 'none'})")
    s = data[key]
    qpos = s.get("qpos")
    if not qpos:
        raise SystemExit("no qpos in match_data.json (need the full replay)")
    composed = md_path.parent / "composed.xml"
    if not composed.is_file():
        raise SystemExit(f"missing composed.xml next to {md_path}")
    out_path = Path(out_path) if out_path else md_path.with_suffix(".webm")

    model = mujoco.MjModel.from_xml_path(str(composed))
    mjdata = mujoco.MjData(model)
    w, h = size
    renderer = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ring = float(s.get("ring_radius") or 7.5)
    cam.lookat[:] = [0, 0, 0.5]
    cam.distance = ring * 2.6
    cam.elevation = -28
    cam.azimuth = 90

    control_hz = round(1.0 / float(s.get("control_dt") or 0.01))
    stride = max(1, round(control_hz / fps))
    nq = model.nq
    frames = []
    for i in range(0, len(qpos), stride):
        q = np.asarray(qpos[i], dtype=float)
        mjdata.qpos[:] = q[:nq] if q.shape[0] >= nq else np.pad(q, (0, nq - q.shape[0]))
        mujoco.mj_forward(model, mjdata)
        renderer.update_scene(mjdata, camera=cam)
        frames.append(renderer.render())
    iio.mimwrite(str(out_path), frames, fps=fps, codec="libvpx-vp9",
                 output_params=["-b:v", "2M"])
    print(out_path)
    return out_path


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--webm":
        args = args[1:]
        if not args:
            raise SystemExit("usage: render_match.py --webm <match_data.json> [out.webm] [seed]")
        out = args[1] if len(args) > 1 else None
        seed = int(args[2]) if len(args) > 2 else 0
        render_webm(args[0], out, seed=seed)
    elif args:
        render(args[0], args[1] if len(args) > 1 else None)
    else:
        raise SystemExit("usage: render_match.py [--webm] <match_data.json> [out] [seed]")
