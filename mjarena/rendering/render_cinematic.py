#!/usr/bin/env python3
"""Orchestrate cinematic rendering of a match via Blender Cycles.

Combines export_trajectory (MuJoCo → npz) and render_blender (npz → Blender → frames),
then stitches frames into a video.

Usage:
    python -m mjarena.rendering.render_cinematic \
        --match-dir path/to/match_results \
        --seed 0 \
        --output cinematic.mp4

Requires:
    - Blender installed and on PATH (or --blender-path)
    - ffmpeg for frame stitching (or use --no-stitch to keep frames)
    - Optional: ``opencv-python-headless`` only if you pass ``--post-process-cv`` (off by default; OpenCV re-encode can color-shift on some systems)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_MAC_APP_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/blender")


def verify_blender_bpy(blender_exe: str) -> None:
    """Fail fast if ``blender_exe`` is missing bpy (broken symlink / incomplete .app)."""
    cmd = [
        blender_exe,
        "--background",
        "--python-expr",
        "import bpy; print('MJARENA_BPY_OK', bpy.app.version_string)",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.error("Blender preflight timed out after 120s.")
        sys.exit(1)

    blob = (result.stdout or "") + (result.stderr or "")
    ok = result.returncode == 0 and "MJARENA_BPY_OK" in (result.stdout or "")

    if ok:
        tail = [ln for ln in (result.stdout or "").splitlines() if ln.startswith("MJARENA_BPY_OK")]
        if tail:
            logger.info("Blender preflight OK (%s).", tail[-1].replace("MJARENA_BPY_OK", "").strip())
        return

    hint = (
        "This Blender build cannot load bpy (install is incomplete or broken).\n"
        "  Common cause: /usr/local/bin/blender is a symlink to the binary only, without the\n"
        "  rest of Blender.app (datafiles, scripts, bundled Python).\n\n"
        "  Fix: install a full Blender from https://www.blender.org/download/ and run e.g.:\n"
        f"    --blender-path {_MAC_APP_BLENDER}\n\n"
        "  Then remove or replace any bad symlink: ls -l /usr/local/bin/blender\n"
    )
    logger.error(
        "%s\nBlender preflight command: %s\n\n--- output (tail) ---\n%s",
        hint,
        " ".join(cmd),
        blob[-6000:] if len(blob) > 6000 else blob,
    )
    sys.exit(1)


def find_match_files(match_dir: Path) -> tuple[Path, Path]:
    """Find composed.xml and match_data.json in a match directory."""
    composed = match_dir / "composed.xml"
    match_data = match_dir / "match_data.json"

    if not composed.exists():
        raise FileNotFoundError(f"composed.xml not found in {match_dir}")
    if not match_data.exists():
        raise FileNotFoundError(f"match_data.json not found in {match_dir}")

    return composed, match_data


def _infer_metadata(match_dir: Path) -> dict[str, str]:
    """Best-effort extraction of red_name / blue_name / tournament from the match dir layout.

    Directory name (pattern ``<red>_vs_<blue>``) is preferred — bot names like
    ``gpt-5.2-medium`` contain dots that break naive splitting of the
    dotted-namespace values in ``match_result.json``. Falls back to
    ``match_result.json`` only when the dir name lacks the ``_vs_`` marker.
    Tournament label comes from any ancestor chunk ``tournament_NN``.
    """
    meta: dict[str, str] = {}
    name = match_dir.name
    if "_vs_" in name:
        a, b = name.split("_vs_", 1)
        meta["red_name"] = a
        meta["blue_name"] = b
    else:
        result_json = match_dir / "match_result.json"
        if result_json.exists():
            try:
                data = json.loads(result_json.read_text())
                if data.get("red_bot"):
                    meta["red_name"] = str(data["red_bot"])
                if data.get("blue_bot"):
                    meta["blue_name"] = str(data["blue_bot"])
            except (json.JSONDecodeError, OSError):
                pass
    for part in match_dir.resolve().parts:
        if part.startswith("tournament_"):
            meta["tournament"] = f"T{part.removeprefix('tournament_')}"
            break
    return meta


def run_export(composed_xml: Path, match_data_json: Path, seed: int, output_npz: Path,
               red_name: str = "", blue_name: str = "",
               season: str = "", tournament: str = ""):
    """Run the trajectory export step."""
    from mjarena.rendering.export_trajectory import export_trajectory
    export_trajectory(composed_xml, match_data_json, seed, output_npz,
                      red_name=red_name, blue_name=blue_name,
                      season=season, tournament=tournament)


def run_blender(
    blender_path: str,
    trajectory_npz: Path,
    output_dir: Path,
    width: int,
    height: int,
    samples: int,
    engine: str = "eevee",
    frame_step: int = 3,
):
    """Run Blender in background mode to render frames."""
    render_script = Path(__file__).parent / "render_blender.py"

    cmd = [
        blender_path,
        "--background",
        "--python", str(render_script),
        "--",
        str(trajectory_npz),
        str(output_dir),
        str(width),
        str(height),
        str(samples),
        engine,
        str(frame_step),
    ]

    # Do not capture stdout/stderr — Blender may run many minutes with no interim
    # feedback otherwise (looks hung). Output goes straight to this terminal.
    logger.info(f"Running Blender (live logs below — not silent):\n  {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logger.error(f"Blender exited with code {result.returncode}")
        raise RuntimeError(f"Blender render failed with code {result.returncode}")

    logger.info("Blender render complete")


def stitch_frames(frames_dir: Path, output_video: Path, fps: int = 100) -> bool:
    """Stitch PNG frames into an MP4 video using ffmpeg. Returns False if ffmpeg is missing."""
    fps = max(1, int(round(fps)))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found — frames saved but video not stitched")
        logger.info(f"Frames at: {frames_dir}")
        logger.info(
            f"Manual stitch: ffmpeg -framerate {fps} -i {frames_dir}/frame_%04d.png "
            f"-c:v libx264 -pix_fmt yuv420p {output_video}"
        )
        return False

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_video),
    ]

    logger.info(f"Stitching frames: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg failed:\n{result.stderr}")
        raise RuntimeError(f"ffmpeg failed with code {result.returncode}")

    logger.info(f"Video saved: {output_video.resolve()}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Render a cinematic match video using Blender Cycles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--match-dir", type=Path, required=True, help="Directory containing composed.xml and match_data.json")
    parser.add_argument("--seed", type=int, default=0, help="Which seed to render (default: 0)")
    parser.add_argument("--output", type=Path, default=Path("cinematic.mp4"), help="Output video path")
    parser.add_argument("--blender-path", type=str, default="blender", help="Path to Blender executable")
    parser.add_argument("--width", type=int, default=1920, help="Render width")
    parser.add_argument("--height", type=int, default=1080, help="Render height")
    parser.add_argument("--samples", type=int, default=16, help="Render samples (default: 16)")
    parser.add_argument("--engine", choices=["eevee", "cycles"], default="eevee", help="Render engine (default: eevee, fast; cycles for ray-traced)")
    parser.add_argument("--frame-step", type=int, default=3, help="Render every Nth frame (default: 3, gives 30fps from 100Hz data)")
    parser.add_argument("--keep-frames", action="store_true", help="Keep individual frame PNGs")
    parser.add_argument("--no-stitch", action="store_true", help="Don't stitch frames into video (keeps PNGs)")
    parser.add_argument("--fps", type=int, default=None, help="Video FPS (default: computed from data + frame-step)")
    parser.add_argument("--red-name", default=None, help="Red bot display name (default: inferred from match dir)")
    parser.add_argument("--blue-name", default=None, help="Blue bot display name (default: inferred from match dir)")
    parser.add_argument("--season", default=None, help="Season label (default: empty)")
    parser.add_argument("--tournament", default=None, help="Tournament label (default: inferred from match dir)")
    parser.add_argument("--single-frame", type=int, default=None,
                        help="Render only frame N (1-indexed). Outputs a PNG at --output with .png suffix and skips ffmpeg.")
    parser.add_argument("--keep-mj-ring", action="store_true",
                        help="Render MuJoCo's sumo_ring cylinder (recolored) instead of Blender's tapered octagon. "
                             "Physics-accurate: matches the straight cylinder that actually constrained bots during the match, "
                             "so bots falling past the edge don't clip into a flared Blender wall.")
    parser.add_argument(
        "--post-process-cv",
        dest="post_process_cv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After ffmpeg stitch, boost saturation/contrast on the MP4 with OpenCV (default: off — "
             "keeps the ffmpeg output; enable only if you want this extra pass).",
    )
    parser.add_argument(
        "--opencv-saturation",
        type=float,
        default=1.35,
        help="L*a*b* chroma multiplier when post-process-cv is on (default 1.35; neutrals stay neutral)",
    )
    parser.add_argument(
        "--opencv-contrast-alpha",
        type=float,
        default=1.12,
        help="Linear contrast scale when post-process-cv is on (default 1.12)",
    )
    parser.add_argument(
        "--opencv-contrast-beta",
        type=float,
        default=0.0,
        help="Brightness offset after contrast when post-process-cv is on (default 0)",
    )
    parser.add_argument(
        "--opencv-fourcc",
        type=str,
        default="mp4v",
        help="OpenCV VideoWriter fourcc when post-process-cv is on (default mp4v)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.post_process_cv:
        try:
            import cv2  # noqa: F401
        except ImportError:
            logger.error(
                "OpenCV is required when --post-process-cv is set. "
                "Install with: pip install opencv-python-headless\n"
                "Or omit --post-process-cv (default)."
            )
            sys.exit(1)

    # Verify Blender is available
    blender = shutil.which(args.blender_path) or args.blender_path
    if not Path(blender).exists() and not shutil.which(blender):
        logger.error(f"Blender not found at '{args.blender_path}'. Install with: brew install --cask blender")
        sys.exit(1)

    verify_blender_bpy(blender)

    # Find match files
    composed, match_data = find_match_files(args.match_dir)
    logger.info(f"Match dir: {args.match_dir}")
    logger.info(f"Composed XML: {composed}")
    logger.info(f"Match data: {match_data}")

    inferred = _infer_metadata(args.match_dir)
    meta_kwargs = {
        "red_name":   args.red_name   if args.red_name   is not None else inferred.get("red_name", ""),
        "blue_name":  args.blue_name  if args.blue_name  is not None else inferred.get("blue_name", ""),
        "season":     args.season     if args.season     is not None else inferred.get("season", ""),
        "tournament": args.tournament if args.tournament is not None else inferred.get("tournament", ""),
    }
    logger.info(f"Scoreboard metadata: {meta_kwargs}")

    if args.single_frame is not None:
        os.environ["RENDER_ONLY_FRAME"] = str(args.single_frame)
    if args.keep_mj_ring:
        os.environ["KEEP_MJ_RING"] = "1"

    with tempfile.TemporaryDirectory(prefix="cinematic_") as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Export trajectory
        trajectory_npz = tmpdir / "trajectory.npz"
        logger.info("Step 1/3: Exporting trajectory from MuJoCo...")
        run_export(composed, match_data, args.seed, trajectory_npz, **meta_kwargs)

        traj_preview = np.load(trajectory_npz, allow_pickle=True)
        n_phys = int(traj_preview["n_frames"])
        n_render = (n_phys + args.frame_step - 1) // args.frame_step
        logger.info(
            f"Expect ~{n_render} Blender frame(s) ({n_phys} physics ticks, "
            f"frame-step={args.frame_step}). Step 2 often takes minutes; "
            "Blender prints progress below (use Activity Monitor → blender if idle)."
        )

        # Step 2: Render in Blender
        frames_dir = tmpdir / "frames"
        frames_dir.mkdir()
        logger.info(f"Step 2/3: Rendering in Blender ({args.engine}, {args.width}x{args.height}, {args.samples} samples, step={args.frame_step})...")
        run_blender(blender, trajectory_npz, frames_dir, args.width, args.height, args.samples, args.engine, args.frame_step)

        # Step 3: Add scoreboard overlay
        traj_data = __import__("numpy").load(trajectory_npz, allow_pickle=True)
        # npz may omit fps or use 0; ffmpeg requires framerate >= 1.
        raw = traj_data.get("fps", np.array(100.0))
        try:
            phys_fps = float(np.asarray(raw).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            phys_fps = 100.0
        if not np.isfinite(phys_fps) or phys_fps <= 0.0:
            phys_fps = 100.0
        step = max(1, int(args.frame_step))
        computed_fps = max(1, int(round(phys_fps / float(step))))
        if args.fps is not None:
            fps = max(1, int(round(float(args.fps))))
        else:
            fps = computed_fps

        metadata = {}
        for key in ("red_name", "blue_name", "season", "tournament", "seed"):
            if key in traj_data:
                metadata[key] = str(traj_data[key])

        png_frames = sorted(frames_dir.glob("*.png"))
        if not png_frames:
            logger.error("No rendered frames found!")
            return

        _total_steps = 5 if (args.single_frame is None and args.post_process_cv) else 4
        if metadata:
            logger.info(f"Step 3/{_total_steps}: Adding scoreboard overlay to {len(png_frames)} frame(s)...")
            try:
                from mjarena.rendering.render_blender import _add_scoreboard_overlay
                _add_scoreboard_overlay(frames_dir, metadata, args.width, args.height)
            except Exception as e:
                logger.warning(f"Scoreboard overlay failed: {e}")

        if args.single_frame is not None:
            # One frame in, one PNG out — no ffmpeg, no video.
            png_out = args.output.with_suffix(".png") if args.output.suffix != ".png" else args.output
            png_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(png_frames[0], png_out)
            logger.info(f"Single frame saved: {png_out}")
        else:
            # Step 4: Stitch frames into video
            out_mp4 = args.output.resolve()
            logger.info(f"Step 4/{_total_steps}: Stitching {len(png_frames)} frames at {fps} fps → {out_mp4} ...")
            if not stitch_frames(frames_dir, args.output, fps):
                dest = out_mp4.parent / f"{out_mp4.stem}_frames"
                dest.mkdir(parents=True, exist_ok=True)
                for p in sorted(frames_dir.glob("frame_*.png")):
                    shutil.copy2(p, dest / p.name)
                logger.error(
                    "ffmpeg is not on your PATH — the MP4 was NOT written to %s.\n"
                    "The PNG sequence (with scoreboard) was copied to: %s\n"
                    "Fix: install ffmpeg (e.g. `brew install ffmpeg`), open a new terminal, "
                    "run `which ffmpeg`, then either re-run this command or stitch manually:\n"
                    "  ffmpeg -y -framerate %d -i \"%s/frame_%%04d.png\" -c:v libx264 -pix_fmt yuv420p \"%s\"",
                    out_mp4,
                    dest.resolve(),
                    fps,
                    dest,
                    out_mp4,
                )
                sys.exit(1)

            if args.post_process_cv:
                from mjarena.rendering.postprocess_video_cv import process_video

                tmp_cv = tmpdir / "_opencv_enhanced.mp4"
                logger.info(
                    "Step 5/%d: OpenCV post-process (saturation=%s contrast_alpha=%s) → %s",
                    _total_steps,
                    args.opencv_saturation,
                    args.opencv_contrast_alpha,
                    out_mp4,
                )
                try:
                    process_video(
                        out_mp4,
                        tmp_cv,
                        saturation_mult=args.opencv_saturation,
                        contrast_alpha=args.opencv_contrast_alpha,
                        contrast_beta=args.opencv_contrast_beta,
                        fourcc_str=args.opencv_fourcc,
                    )
                    tmp_cv.replace(out_mp4)
                    logger.info("OpenCV post-process complete.")
                except ImportError as e:
                    logger.error("%s Install with: pip install opencv-python-headless", e)
                    sys.exit(1)
                except Exception as e:
                    logger.error("OpenCV post-process failed: %s", e)
                    sys.exit(1)

    logger.info("Done!")


if __name__ == "__main__":
    main()
