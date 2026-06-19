#!/usr/bin/env python3
"""Generate ad variants from one base clip — the cheap, local A/B axes.

Two axes (cartesian product): aspect-ratio reframes and on-screen label/hook
swaps. Each variant is one ffmpeg pass (reframe + optional label). Writes the
files plus a `variants.json` manifest describing them.

Examples
--------
  # three aspect ratios of the same clip
  python3 make_variants.py --video base.mp4 --outdir variants \
      --ratios 9:16,1:1,16:9

  # one ratio, three hook labels (A/B/C testing the opening text)
  python3 make_variants.py --video base.mp4 --outdir variants \
      --labels "Wait for it...|You won't believe this|POV: you found it"

  # full grid: 2 ratios x 2 labels = 4 variants
  python3 make_variants.py --video base.mp4 --outdir variants \
      --ratios 9:16,1:1 --labels "Hook A|Hook B" --label-pos top
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

MAX_VARIANTS = 24


def slug(s: str, n: int = 24) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower()).strip("-")
    return (s[:n] or "x").strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description="Make aspect/label variants of a clip.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ratios", help="comma list, e.g. 9:16,1:1,16:9 "
                                     "(default: keep source ratio)")
    ap.add_argument("--labels", help="pipe-separated label texts, one variant each")
    ap.add_argument("--label-pos", default="top", choices=["top", "center", "bottom"])
    ap.add_argument("--label-size", type=float, default=0.05)
    ap.add_argument("--label-color", default="white")
    ap.add_argument("--label-box", action="store_true")
    ap.add_argument("--fit", choices=["cover", "contain"], default="cover",
                    help="reframe mode for ratio changes (default cover = "
                         "fill+crop, the UGC norm)")
    ap.add_argument("--font")
    args = ap.parse_args()

    C.need_ffmpeg()
    if not os.path.exists(args.video):
        C.die(f"video not found: {args.video}")
    p = C.probe(args.video)

    ratios = [r.strip() for r in args.ratios.split(",")] if args.ratios else [p.ratio_name]
    for r in ratios:
        if r not in C.RATIO_CANVAS:
            C.die(f"unknown ratio {r}; choose from {sorted(C.RATIO_CANVAS)}")
    labels = [s for s in args.labels.split("|")] if args.labels else [None]

    combos = [(r, l) for r in ratios for l in labels]
    if len(combos) > MAX_VARIANTS:
        C.die(f"{len(combos)} variants exceeds cap {MAX_VARIANTS} — narrow the axes")

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    font = C.find_font(args.font) if args.labels else None
    fps = int(round(p.fps)) or C.DEFAULT_FPS
    manifest = []

    for ratio, label in combos:
        w, h = C.RATIO_CANVAS[ratio]
        name = f"{base}__{ratio.replace(':', 'x')}"
        if label is not None:
            name += f"_{slug(label)}"
        out = os.path.join(args.outdir, name + ".mp4")

        png = None
        inputs = ["-i", args.video]
        parts = [f"[0:v]{C.fit_filter(w, h, mode=args.fit)},fps={fps},format=yuv420p[v]"]
        cur = "[v]"
        if label is not None:
            size = max(10, int(round(h * args.label_size)))
            png, png_h = C.render_text_png(
                label, width=w, font_path=font, font_size=size,
                color=args.label_color, box=args.label_box)
            inputs += ["-i", png]
            y = C.band_y(args.label_pos, h, png_h)
            parts.append(f"{cur}[1:v]overlay=0:{y}[vt]")
            cur = "[vt]"

        cmd = ["ffmpeg", "-y", *inputs,
               "-filter_complex", ";".join(parts),
               "-map", cur, "-map", "0:a?", *C.V_CODEC, "-c:a", "copy",
               "-r", str(fps), out]
        try:
            C.run(cmd)
        finally:
            if png and os.path.exists(png):
                os.unlink(png)
        manifest.append({"file": os.path.abspath(out), "ratio": ratio,
                         "label": label})

    man_path = os.path.join(args.outdir, "variants.json")
    with open(man_path, "w") as f:
        json.dump({"source": os.path.abspath(args.video), "count": len(manifest),
                   "variants": manifest}, f, indent=2)
    print(json.dumps({"ok": True, "count": len(manifest), "outdir":
                      os.path.abspath(args.outdir), "manifest": man_path,
                      "variants": manifest}, indent=2))


if __name__ == "__main__":
    main()
