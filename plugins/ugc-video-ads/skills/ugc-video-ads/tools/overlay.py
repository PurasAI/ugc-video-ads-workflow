#!/usr/bin/env python3
"""Burn a logo and/or a text label onto a video (parameterized; no UI yet).

Both the logo and the label are composited as image overlays — the label text
is rendered to a transparent PNG with Pillow (word-wrapped, shadowed, optional
contrast box), so it works even on an ffmpeg built without drawtext/libfreetype.
Logo placement is a named corner; the label is a centered band. Audio passes
through untouched.

Examples
--------
  python3 overlay.py --video in.mp4 --out out.mp4 --logo logo.png --logo-pos tr
  python3 overlay.py --video in.mp4 --out out.mp4 --label "50% OFF TODAY" --label-box
  python3 overlay.py --video in.mp4 --out out.mp4 \
      --logo logo.png --logo-pos tl --label "@brand" --label-pos top
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay a logo and/or text label.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    # logo
    ap.add_argument("--logo", help="PNG (ideally transparent) to overlay")
    ap.add_argument("--logo-pos", default="tr",
                    choices=["tl", "tr", "bl", "br", "tc", "bc", "c"])
    ap.add_argument("--logo-scale", type=float, default=0.16,
                    help="logo width as a fraction of video width (default 0.16)")
    ap.add_argument("--logo-margin", type=int, default=40)
    ap.add_argument("--logo-opacity", type=float, default=1.0, help="0..1")
    # label
    ap.add_argument("--label", help="text label to draw")
    ap.add_argument("--label-pos", default="bottom",
                    choices=["top", "center", "bottom"])
    ap.add_argument("--label-size", type=float, default=0.05,
                    help="font size as a fraction of video height (default 0.05)")
    ap.add_argument("--label-color", default="white")
    ap.add_argument("--label-box", action="store_true")
    ap.add_argument("--label-box-color", default="black@0.45")
    ap.add_argument("--font", help="path to a .ttf (default: auto-detect)")
    args = ap.parse_args()

    C.need_ffmpeg()
    if not args.logo and not args.label:
        C.die("nothing to do — pass --logo and/or --label")
    if not os.path.exists(args.video):
        C.die(f"video not found: {args.video}")
    if args.logo and not os.path.exists(args.logo):
        C.die(f"logo not found: {args.logo}")

    p = C.probe(args.video)
    inputs = ["-i", args.video]
    parts = [f"[0:v]fps={int(round(p.fps)) or C.DEFAULT_FPS},format=yuv420p[base]"]
    cur = "[base]"
    idx = 1
    label_png = None

    if args.logo:
        inputs += ["-i", args.logo]
        logo_w = max(2, int(round(p.width * args.logo_scale)))
        lf = f"[{idx}:v]scale={logo_w}:-1"
        if args.logo_opacity < 1.0:
            lf += f",format=rgba,colorchannelmixer=aa={args.logo_opacity:.3f}"
        lf += "[logo]"
        parts.append(lf)
        parts.append(f"{cur}[logo]overlay={C.corner_xy(args.logo_pos, args.logo_margin)}[w1]")
        cur = "[w1]"
        idx += 1

    if args.label:
        font = C.find_font(args.font)
        size = max(10, int(round(p.height * args.label_size)))
        label_png, png_h = C.render_text_png(
            args.label, width=p.width, font_path=font, font_size=size,
            color=args.label_color, box=args.label_box,
            box_color=args.label_box_color)
        inputs += ["-i", label_png]
        y = C.band_y(args.label_pos, p.height, png_h)
        parts.append(f"{cur}[{idx}:v]overlay=0:{y}[w2]")
        cur = "[w2]"
        idx += 1

    C.ensure_parent(args.out)
    cmd = ["ffmpeg", "-y", *inputs,
           "-filter_complex", ";".join(parts),
           "-map", cur, "-map", "0:a?", *C.V_CODEC, "-c:a", "copy", args.out]
    try:
        C.run(cmd)
    finally:
        if label_png and os.path.exists(label_png):
            os.unlink(label_png)
    bits = []
    if args.logo:
        bits.append(f"logo@{args.logo_pos}")
    if args.label:
        bits.append(f"label@{args.label_pos}")
    C.ok(args.out, label="overlay " + "+".join(bits))


if __name__ == "__main__":
    main()
