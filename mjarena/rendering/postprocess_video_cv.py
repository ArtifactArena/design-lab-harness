"""Increase saturation and contrast of a video using OpenCV (no Blender).

Saturation is done in **L*a*b*** (scale ``a``/``b`` about neutral), not HSV. Boosting HSV ``S``
on near-black or gray pixels exposes garbage **H** values and can turn the whole frame green.
Contrast uses ``convertScaleAbs`` on BGR.

Usage::

    python -m mjarena.rendering.postprocess_video_cv input.mp4 output.mp4 \\
        --saturation 1.35 --contrast-alpha 1.12

Requires: ``pip install opencv-python-headless`` (or ``opencv-python``) and **ffmpeg** on PATH.

Output is encoded with **ffmpeg** ``libx264`` + ``yuv420p`` (raw BGR pipe). Using
``cv2.VideoWriter`` for MP4 often yields **green or wrong colors** on macOS; we avoid that when ffmpeg is available.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def enhance_frame_bgr(
    bgr: np.ndarray,
    *,
    saturation_mult: float = 1.35,
    contrast_alpha: float = 1.12,
    contrast_beta: float = 0.0,
) -> np.ndarray:
    """Return a new BGR uint8 frame with boosted saturation and contrast."""
    import cv2

    if saturation_mult != 1.0:
        m = float(saturation_mult)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        Lc, ac, bc = cv2.split(lab)
        a_f = (ac.astype(np.float32) - 128.0) * m + 128.0
        b_f = (bc.astype(np.float32) - 128.0) * m + 128.0
        ac = np.clip(a_f, 0.0, 255.0).astype(np.uint8)
        bc = np.clip(b_f, 0.0, 255.0).astype(np.uint8)
        lab = cv2.merge([Lc, ac, bc])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if contrast_alpha != 1.0 or contrast_beta != 0.0:
        bgr = cv2.convertScaleAbs(bgr, alpha=float(contrast_alpha), beta=float(contrast_beta))

    return bgr


def _write_video_ffmpeg_pipe(
    ffmpeg_exe: str,
    output_path: Path,
    *,
    w: int,
    h: int,
    fps: float,
    frame_iter,
    n_frames_hint: int,
) -> int:
    """Stream BGR uint8 frames (h, w, 3) to ffmpeg rawvideo → H.264. Returns frame count."""
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        f"{fps:.6g}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    n = 0
    try:
        for bgr in frame_iter:
            if not bgr.flags["C_CONTIGUOUS"]:
                bgr = np.ascontiguousarray(bgr)
            if bgr.shape != (h, w, 3):
                raise ValueError(f"Expected frame shape ({h}, {w}, 3), got {bgr.shape}")
            proc.stdin.write(bgr.tobytes())
            n += 1
            if n % 200 == 0 and n_frames_hint > 0:
                print(f"  frames {n}/{n_frames_hint}", flush=True)
    finally:
        proc.stdin.close()
    err_b = proc.stderr.read() if proc.stderr else b""
    code = proc.wait()
    if code != 0:
        msg = err_b.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg exited {code}: {msg or '(no stderr)'}")
    return n


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    saturation_mult: float = 1.35,
    contrast_alpha: float = 1.12,
    contrast_beta: float = 0.0,
    fourcc_str: str = "mp4v",
) -> None:
    """Read ``input_path``, enhance every frame, write ``output_path``."""
    try:
        import cv2
    except ImportError as e:
        raise ImportError(
            "OpenCV is required. Install with: pip install opencv-python-headless"
        ) from e

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps or fps < 1e-3:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ffmpeg = shutil.which("ffmpeg")
    idx = 0

    def _frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield enhance_frame_bgr(
                frame,
                saturation_mult=saturation_mult,
                contrast_alpha=contrast_alpha,
                contrast_beta=contrast_beta,
            )

    try:
        if ffmpeg:
            print("Encoding with ffmpeg (libx264) — avoids OpenCV VideoWriter color bugs.", flush=True)
            gen = _frames()
            first = next(gen, None)
            if first is None:
                raise RuntimeError(f"No frames read from {input_path}")

            def _all_frames():
                yield first
                yield from gen

            idx = _write_video_ffmpeg_pipe(
                ffmpeg,
                output_path,
                w=w,
                h=h,
                fps=fps,
                frame_iter=_all_frames(),
                n_frames_hint=n_frames,
            )
        else:
            print(
                "WARNING: ffmpeg not on PATH — using cv2.VideoWriter (may show green/wrong colors on some systems).",
                flush=True,
            )
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str[:4])
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not open VideoWriter for {output_path} (codec {fourcc_str!r}). "
                    "Install ffmpeg and re-run."
                )
            for frame in _frames():
                writer.write(frame)
                idx += 1
                if idx % 200 == 0 and n_frames > 0:
                    print(f"  frames {idx}/{n_frames}", flush=True)
            writer.release()
    finally:
        cap.release()

    if idx == 0:
        raise RuntimeError(f"No frames read from {input_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Boost saturation/contrast of a video with OpenCV.")
    p.add_argument("input", type=Path, help="Input video (e.g. MP4)")
    p.add_argument("output", type=Path, help="Output video path")
    p.add_argument(
        "--saturation",
        type=float,
        default=1.35,
        help="Chroma boost in L*a*b* (1=no change; >1 more vivid; neutrals stay neutral) (default 1.35)",
    )
    p.add_argument(
        "--contrast-alpha",
        type=float,
        default=1.12,
        help="Linear contrast scale passed to convertScaleAbs (default 1.12)",
    )
    p.add_argument(
        "--contrast-beta",
        type=float,
        default=0.0,
        help="Brightness offset after contrast (default 0)",
    )
    p.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="Only if ffmpeg is missing: OpenCV VideoWriter fourcc (default mp4v; often causes green MP4 on macOS)",
    )
    args = p.parse_args()

    print(
        f"OpenCV post: {args.input} → {args.output} "
        f"(sat={args.saturation}, contrast_alpha={args.contrast_alpha}, beta={args.contrast_beta})",
        flush=True,
    )
    process_video(
        args.input,
        args.output,
        saturation_mult=args.saturation,
        contrast_alpha=args.contrast_alpha,
        contrast_beta=args.contrast_beta,
        fourcc_str=args.fourcc,
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
