# video & replay I/O helpers
import imageio
import logging
from dataclasses import dataclass
from mujoco import Renderer, MjData
import numpy as np
import warnings
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_FPS = 60


# =============================================================================
# Video Overlay
# =============================================================================


@dataclass
class VideoOverlayInfo:
    """Metadata drawn on each video frame."""
    seed: int = 0
    red_name: str = ""
    blue_name: str = ""
    tournament_id: str = ""
    season_id: str = ""
    winner: Optional[str] = None  # "red", "blue", or "tie" — set during match

    @property
    def red_side(self) -> str:
        return "Left" if self.seed % 2 == 0 else "Right"

    @property
    def blue_side(self) -> str:
        return "Right" if self.seed % 2 == 0 else "Left"


def _load_overlay_font(font_size: int):
    """Load a monospace font, falling back gracefully."""
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
        except (OSError, IOError):
            return ImageFont.load_default()


def overlay_frame(frame: np.ndarray, info: VideoOverlayInfo) -> np.ndarray:
    """Draw text overlay on a video frame (RGB uint8 numpy array).

    Single compact top-left box with bot names + metadata.
    Semi-transparent background via alpha compositing.

    Returns a new frame with overlay text burned in.
    """
    from PIL import Image, ImageDraw

    h, w = frame.shape[:2]
    img = Image.fromarray(frame).convert("RGBA")

    font_size = max(10, int(14 * h / 480))
    font = _load_overlay_font(font_size)

    # Smaller font for metadata line
    meta_font_size = max(9, font_size - 4)
    meta_font = _load_overlay_font(meta_font_size)

    pad = max(4, int(6 * h / 480))

    # --- Build text lines ---
    red_suffix = " (wins)" if info.winner == "red" else ""
    blue_suffix = " (wins)" if info.winner == "blue" else ""
    red_text = f"Red: {info.red_name}{red_suffix}"
    blue_text = f"Blue: {info.blue_name}{blue_suffix}"
    seed_text = f"Seed: {info.seed}"

    meta_parts = []
    if info.season_id:
        meta_parts.append(f"Season: {info.season_id}")
    if info.tournament_id:
        meta_parts.append(f"Tournament: {info.tournament_id}")
    meta_text = "  |  ".join(meta_parts) if meta_parts else ""

    # --- Measure text to compute box size ---
    scratch = Image.new("RGBA", (1, 1))
    sdraw = ImageDraw.Draw(scratch)

    lines = [(red_text, font), (blue_text, font), (seed_text, font)]
    if meta_text:
        lines.append((meta_text, meta_font))

    line_sizes = []
    for text, f in lines:
        bb = sdraw.textbbox((0, 0), text, font=f)
        line_sizes.append((bb[2] - bb[0], bb[3] - bb[1]))

    content_w = max(s[0] for s in line_sizes)
    max_box_w = int(w * 0.6)
    bg_w = min(content_w + pad * 2, max_box_w)
    # Extra pad at top for breathing room above red text
    top_margin = pad
    bg_h = top_margin + sum(s[1] for s in line_sizes) + pad * (len(lines) + 1)

    # --- Draw opaque background on RGBA overlay ---
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        [pad, pad - 2, pad + bg_w, pad + bg_h],
        fill=(0, 0, 0, 210),
    )

    # --- Draw text on the overlay ---
    y = pad + top_margin
    draw.text((pad * 2, y), red_text, fill=(255, 80, 80, 255), font=font)
    y += line_sizes[0][1] + pad
    draw.text((pad * 2, y), blue_text, fill=(80, 130, 255, 255), font=font)
    y += line_sizes[1][1] + pad
    draw.text((pad * 2, y), seed_text, fill=(255, 255, 255, 255), font=font)
    y += line_sizes[2][1] + pad
    if meta_text:
        draw.text((pad * 2, y), meta_text, fill=(180, 180, 180, 255), font=meta_font)

    # --- Alpha composite and convert back to RGB ---
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def make_title_frames(
    width: int,
    height: int,
    info: VideoOverlayInfo,
    fps: int,
    duration_s: float = 1.0,
) -> List[np.ndarray]:
    """Generate black title card frames with bot names and seed number."""
    from PIL import Image, ImageDraw

    n_frames = max(1, int(fps * duration_s))

    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Title line: "Red: X  vs  Blue: Y"
    title_font_size = max(14, int(24 * height / 480))
    title_font = _load_overlay_font(title_font_size)

    # Seed line
    seed_font_size = max(12, int(18 * height / 480))
    seed_font = _load_overlay_font(seed_font_size)

    title_text = f"Red: {info.red_name}  vs  Blue: {info.blue_name}"
    seed_text = f"Seed: {info.seed}"

    # Measure
    tb = draw.textbbox((0, 0), title_text, font=title_font)
    sb = draw.textbbox((0, 0), seed_text, font=seed_font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    sw, sh = sb[2] - sb[0], sb[3] - sb[1]

    gap = max(6, int(10 * height / 480))
    total_h = th + gap + sh

    # Center vertically and horizontally
    y_start = (height - total_h) // 2
    draw.text(((width - tw) // 2, y_start), title_text, fill=(255, 255, 255), font=title_font)
    draw.text(((width - sw) // 2, y_start + th + gap), seed_text, fill=(180, 180, 180), font=seed_font)

    frame = np.array(img)
    return [frame] * n_frames


# =============================================================================
# Video Concatenation
# =============================================================================


def concatenate_seed_videos(
    seed_videos: List[Path],
    output_path: Path,
    overlay_infos: List[VideoOverlayInfo],
    fps: int = DEFAULT_FPS,
    delete_originals: bool = True,
) -> Optional[Path]:
    """Concatenate per-seed video files into a single video with title cards.

    Args:
        seed_videos: List of seed video file paths (e.g. seed_0.mp4, seed_1.mp4).
        output_path: Path for the concatenated output video.
        overlay_infos: One VideoOverlayInfo per seed (for title cards).
        fps: Frames per second for the output video.
        delete_originals: If True, delete individual seed mp4s after concat.

    Returns:
        output_path if successful, None if no videos found.
    """
    # Collect existing video files
    video_files = []
    for mp4 in seed_videos:
        if mp4.exists():
            video_files.append(mp4)
        else:
            video_files.append(None)

    existing = [v for v in video_files if v is not None]
    if not existing:
        return None

    # Read first video to get dimensions
    reader = imageio.get_reader(str(existing[0]))
    meta = reader.get_meta_data()
    first_frame = reader.get_data(0)
    h, w = first_frame.shape[:2]
    vid_fps = meta.get("fps", fps)
    reader.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = get_offscreen_writer(output_path, video_fps=int(vid_fps))

    try:
        for i, (vf, info) in enumerate(zip(video_files, overlay_infos)):
            if vf is None:
                continue

            # Insert title card before each seed segment
            title_frames = make_title_frames(w, h, info, int(vid_fps), duration_s=1.0)
            for tf in title_frames:
                writer.append_data(tf)

            # Copy frames from seed video
            reader = imageio.get_reader(str(vf))
            for frame in reader:
                if frame.shape[-1] == 4:
                    frame = frame[..., :3]
                writer.append_data(frame)
            reader.close()
    finally:
        writer.close()

    logger.info(f"Concatenated {len(existing)} seed videos -> {output_path}")

    # Clean up individual seed mp4s
    if delete_originals:
        for vf in existing:
            try:
                vf.unlink()
            except OSError:
                pass

    return output_path


# =============================================================================
# Writer / Frame Helpers
# =============================================================================


def get_offscreen_writer(video_path: Path | str, video_fps: int = None, control_timestep: float = None) -> imageio.get_writer:
    """Get an offscreen writer for a video path.

    `.webm` paths get VP9 (CRF 32, realtime). MuJoCo flat-shaded frames compress
    ~10% smaller than the default H.264 mp4; the bigger lever is just not
    writing tournament videos at all (see tournament.save_video in YAML).
    """
    video_path = Path(video_path)  # Ensure it's a Path object
    default_fps = int(round(1.0 / float(control_timestep))) if control_timestep is not None else DEFAULT_FPS
    fps = int(video_fps if video_fps is not None else default_fps)

    # Ensure parent directory exists
    video_path.parent.mkdir(parents=True, exist_ok=True)

    if video_path.parent.exists() and not video_path.parent.is_dir():
        raise ValueError(f"Video path parent {video_path.parent} exists but is not a directory")

    if video_path.suffix.lower() == ".webm":
        writer = imageio.get_writer(
            str(video_path),
            fps=fps,
            codec="libvpx-vp9",
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=[
                "-b:v", "0",
                "-crf", "32",
                "-deadline", "realtime",
                "-cpu-used", "4",
                "-row-mt", "1",
            ],
        )
    else:
        writer = imageio.get_writer(str(video_path), fps=fps, macro_block_size=1)
    return writer

def capture_frame(writer: imageio.get_writer, rgb: np.ndarray,
                  overlay_info: Optional[VideoOverlayInfo] = None) -> None:
    """Capture a frame from a renderer and write it to a writer."""
    if writer is None:
        warnings.warn("writer is None")
        return

    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        frame = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        frame = rgb

    if overlay_info is not None:
        frame = overlay_frame(frame, overlay_info)

    writer.append_data(frame)

ansi_reset = "\033[0m"
ansi_green = "\033[92m"
ansi_red = "\033[91m"
ansi_orange = "\033[38;5;208m"
ansi_cyan = "\033[36m"
ansi_magenta = "\033[95m"

def format_tqdm_bar(current_step: int, distance: Optional[float], current_winner: Optional[str]) -> str:
    """Build a colorful progress bar string for tqdm."""
    distance_text = "--"
    distance_color = ansi_orange
    if distance is not None:
        distance_text = f"{distance:.2f}"
        if distance < 1.0:
            distance_color = ansi_red
    winner_text = "None"
    winner_color = ansi_magenta
    if current_winner:
        winner_text = current_winner
        winner_color = ansi_green
    return (
        f"{ansi_cyan}Match step{ansi_reset}: {current_step} | "
        f"{distance_color}Distance between coms{ansi_reset}: {distance_text} | "
        f"Winner: {winner_color}{winner_text}{ansi_reset}"
    )

__all__ = [
    "VideoOverlayInfo",
    "overlay_frame",
    "make_title_frames",
    "concatenate_seed_videos",
    "capture_frame",
    "format_tqdm_bar",
    "get_offscreen_writer",
]
