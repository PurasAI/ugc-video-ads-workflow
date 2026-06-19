"""Shared helpers for the UGC video-ads post-processing tools.

Pure stdlib + ffmpeg/ffprobe on PATH. Every tool imports this for: probing a
clip, finding a usable font, building the canonical scale/pad/crop filters, and
running ffmpeg with consistent encode settings so clips compose cleanly.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# ── canonical encode settings ──────────────────────────────────────────────
# Everything is re-encoded to this so heterogeneous sources (a Seedance render,
# a user's outro, a logo PNG) concat and overlay without codec/sar/fps clashes.
V_CODEC = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
           "-preset", "medium", "-crf", "20", "-movflags", "+faststart"]
A_CODEC = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
DEFAULT_FPS = 30
DEFAULT_SR = 48000

# Aspect-ratio name → (w, h) at 1080-class resolution. The UGC default is 9:16.
RATIO_CANVAS = {
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "16:9": (1920, 1080),
    "4:5": (1080, 1350),
}


def die(msg: str, code: int = 1) -> "None":
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def need_ffmpeg() -> None:
    for b in ("ffmpeg", "ffprobe"):
        if not shutil.which(b):
            die(f"`{b}` not found on PATH (brew install ffmpeg)")


def run(cmd: list[str], *, quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising with ffmpeg's stderr tail on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        die(f"command failed ({proc.returncode}):\n  {shlex.join(cmd)}\n{tail}")
    if not quiet and proc.stderr:
        print(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "")
    return proc


@dataclass
class Probe:
    width: int
    height: int
    fps: float
    duration: float
    has_audio: bool

    @property
    def ratio_name(self) -> str:
        if not self.height:
            return "9:16"
        r = self.width / self.height
        best, bn = 99.0, "9:16"
        for name, (w, h) in RATIO_CANVAS.items():
            d = abs(r - w / h)
            if d < best:
                best, bn = d, name
        return bn


def probe(path: str) -> Probe:
    if not os.path.exists(path):
        die(f"file not found: {path}")
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]).stdout
    data = json.loads(out)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v:
        die(f"no video stream in {path}")
    # avg_frame_rate is "num/den"
    fps = DEFAULT_FPS
    for key in ("avg_frame_rate", "r_frame_rate"):
        val = v.get(key) or ""
        if "/" in val:
            n, d = val.split("/")
            if float(d or 0):
                fps = round(float(n) / float(d), 3)
                if fps > 0:
                    break
    dur = 0.0
    for src in (v.get("duration"), data.get("format", {}).get("duration")):
        try:
            dur = float(src)
            if dur > 0:
                break
        except (TypeError, ValueError):
            continue
    return Probe(
        width=int(v["width"]), height=int(v["height"]),
        fps=fps or DEFAULT_FPS, duration=dur, has_audio=a is not None,
    )


# ── font discovery (for drawtext labels) ─────────────────────────────────────
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/SFNSDisplay.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(override: str | None = None) -> str:
    if override:
        if not os.path.exists(override):
            die(f"font not found: {override}")
        return override
    for f in _FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    die("no usable .ttf font found — pass --font /path/to/font.ttf")


# ── filter builders ──────────────────────────────────────────────────────────
def fit_filter(w: int, h: int, *, mode: str = "cover", pad_color: str = "black") -> str:
    """Scale a source onto a w×h canvas.

    cover   = fill the frame, center-crop the overflow (no bars; UGC default).
    contain = fit whole frame inside, pad the rest (keeps a designed outro intact).
    """
    if mode == "contain":
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={pad_color},setsar=1")
    return (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},setsar=1")


def corner_xy(pos: str, margin: int) -> str:
    """overlay x:y expression for a named corner (logo placement)."""
    m = margin
    return {
        "tl": f"{m}:{m}",
        "tr": f"W-w-{m}:{m}",
        "bl": f"{m}:H-h-{m}",
        "br": f"W-w-{m}:H-h-{m}",
        "tc": f"(W-w)/2:{m}",
        "bc": f"(W-w)/2:H-h-{m}",
        "c": "(W-w)/2:(H-h)/2",
    }.get(pos, f"W-w-{m}:{m}")


def band_y(pos: str, frame_h: int, png_h: int) -> int:
    """Top-left y to place a full-width text-band PNG of height png_h."""
    if pos == "top":
        return int(frame_h * 0.06)
    if pos == "center":
        return max(0, (frame_h - png_h) // 2)
    return max(0, frame_h - png_h - int(frame_h * 0.07))


def parse_color(c: str, *, default_alpha: int = 255) -> tuple:
    """'white', '#C96442', or 'black@0.45' → an RGBA tuple."""
    from PIL import ImageColor
    alpha = default_alpha
    if "@" in c:
        c, a = c.split("@", 1)
        alpha = max(0, min(255, int(round(float(a) * 255))))
    rgb = ImageColor.getrgb(c)
    return (rgb[0], rgb[1], rgb[2], alpha)


def esc_filter_path(p: str) -> str:
    """Escape a path for use inside an ffmpeg filtergraph value."""
    return p.replace("\\", "\\\\").replace(":", "\\:")


def render_text_png(text: str, *, width: int, font_path: str, font_size: int,
                    color: str = "white", box: bool = False,
                    box_color: str = "black@0.45") -> tuple:
    """Render a centered, word-wrapped text band to a transparent full-width PNG.

    Returns (png_path, png_height). Used instead of ffmpeg `drawtext` so labels
    render regardless of how ffmpeg was built — the Homebrew build here ships
    without libfreetype, so the drawtext filter is unavailable. Pillow gives us
    word-wrap, a shadow, and an optional contrast box for free.
    """
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, font_size)
    margin = int(width * 0.06)
    max_w = width - 2 * margin

    def tw(s: str) -> int:
        b = font.getbbox(s)
        return b[2] - b[0]

    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if not cur or tw(trial) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = [" "]

    asc, desc = font.getmetrics()
    line_h = int((asc + desc) * 1.18)
    pad = int(font_size * 0.45)
    img_h = line_h * len(lines) + 2 * pad
    img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if box:
        bw = min(width, max(tw(s) for s in lines) + 2 * pad)
        bx = (width - bw) // 2
        d.rounded_rectangle([bx, 0, bx + bw, img_h], radius=pad,
                            fill=parse_color(box_color, default_alpha=115))
    fill = parse_color(color)
    cx, y = width // 2, pad
    for s in lines:
        d.text((cx + 2, y + 2), s, font=font, fill=(0, 0, 0, 160), anchor="ma")
        d.text((cx, y), s, font=font, fill=fill, anchor="ma")
        y += line_h
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img.save(path)
    return path, img_h


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def ok(out_path: str, label: str = "wrote") -> None:
    p = probe(out_path)
    size_kb = os.path.getsize(out_path) // 1024
    print(json.dumps({
        "ok": True, "out": os.path.abspath(out_path), "label": label,
        "width": p.width, "height": p.height, "ratio": p.ratio_name,
        "fps": p.fps, "duration": round(p.duration, 2),
        "has_audio": p.has_audio, "size_kb": size_kb,
    }, indent=2))
