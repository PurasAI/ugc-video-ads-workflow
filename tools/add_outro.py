#!/usr/bin/env python3
"""Stitch clips in order (e.g. hook → main → outro), normalized to one canvas.

Heterogeneous sources are scaled to a common w×h + fps + 48k stereo AAC, then
joined with the concat filter (hard cut, robust) or an optional crossfade.
Clips with no audio get a silent track so the join never desyncs.

Examples
--------
  # add an outro to a UGC clip (canvas inherited from the main clip)
  python3 add_outro.py --main render.mp4 --outro outro.mp4 --out final.mp4

  # full assembly with a hook, forced to 9:16, half-second crossfades
  python3 add_outro.py --intro hook.mp4 --main render.mp4 --outro outro.mp4 \
      --ratio 9:16 --xfade 0.5 --out final.mp4

  # arbitrary ordered list
  python3 add_outro.py --clips a.mp4 b.mp4 c.mp4 --out joined.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def build_clip_list(args) -> list[str]:
    if args.clips:
        clips = list(args.clips)
    else:
        clips = [c for c in (args.intro, args.main, args.outro) if c]
    if len(clips) < 2:
        C.die("need at least 2 clips (use --clips, or --main with --outro/--intro)")
    for c in clips:
        if not os.path.exists(c):
            C.die(f"clip not found: {c}")
    return clips


def main() -> None:
    ap = argparse.ArgumentParser(description="Stitch clips into one video.")
    ap.add_argument("--clips", nargs="+", help="ordered list of clips to join")
    ap.add_argument("--intro", help="clip to prepend (hook)")
    ap.add_argument("--main", help="the main clip")
    ap.add_argument("--outro", help="clip to append (outro)")
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--ratio", choices=sorted(C.RATIO_CANVAS),
                    help="force a canvas ratio (default: inherit from first clip)")
    ap.add_argument("--fit", choices=["cover", "contain"], default="contain",
                    help="how to fit each clip onto the canvas (default contain "
                         "= letterbox, keeps designed outros intact)")
    ap.add_argument("--fps", type=int, help="output fps (default: first clip's)")
    ap.add_argument("--xfade", type=float, default=0.0,
                    help="crossfade seconds between clips (default 0 = hard cut)")
    args = ap.parse_args()

    C.need_ffmpeg()
    clips = build_clip_list(args)
    probes = [C.probe(c) for c in clips]

    if args.ratio:
        w, h = C.RATIO_CANVAS[args.ratio]
    else:
        w, h = probes[0].width, probes[0].height
    fps = args.fps or int(round(probes[0].fps)) or C.DEFAULT_FPS
    x = max(0.0, args.xfade)
    # a crossfade can't exceed the shortest clip
    if x > 0:
        x = min(x, min(p.duration for p in probes) * 0.5)

    # ── inputs: each clip, plus a synthetic silent track for any clip w/o audio
    cmd: list[str] = ["ffmpeg", "-y"]
    for c in clips:
        cmd += ["-i", c]
    silent_idx: dict[int, int] = {}
    next_input = len(clips)
    for i, p in enumerate(probes):
        if not p.has_audio:
            cmd += ["-f", "lavfi", "-t", f"{max(p.duration, 0.1):.3f}",
                    "-i", f"anullsrc=channel_layout=stereo:sample_rate={C.DEFAULT_SR}"]
            silent_idx[i] = next_input
            next_input += 1

    fit = C.fit_filter(w, h, mode=args.fit)
    parts: list[str] = []
    v_labels, a_labels = [], []
    for i, p in enumerate(probes):
        parts.append(f"[{i}:v]{fit},fps={fps},format=yuv420p,setpts=PTS-STARTPTS[v{i}]")
        a_src = f"[{silent_idx[i]}:a]" if i in silent_idx else f"[{i}:a]"
        parts.append(f"{a_src}aresample={C.DEFAULT_SR},asetpts=PTS-STARTPTS[a{i}]")
        v_labels.append(f"[v{i}]")
        a_labels.append(f"[a{i}]")

    if x > 0:
        # chained crossfades with a running offset
        running = probes[0].duration
        cur_v, cur_a = "[v0]", "[a0]"
        for i in range(1, len(clips)):
            offset = running - x
            ov, oa = f"[vx{i}]", f"[ax{i}]"
            parts.append(f"{cur_v}[v{i}]xfade=transition=fade:duration={x:.3f}:offset={offset:.3f}{ov}")
            parts.append(f"{cur_a}[a{i}]acrossfade=d={x:.3f}{oa}")
            cur_v, cur_a = ov, oa
            running = running + probes[i].duration - x
        out_v, out_a = cur_v, cur_a
    else:
        parts.append("".join(v_labels) + f"concat=n={len(clips)}:v=1:a=0[v]")
        parts.append("".join(a_labels) + f"concat=n={len(clips)}:v=0:a=1[a]")
        out_v, out_a = "[v]", "[a]"

    C.ensure_parent(args.out)
    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", out_v, "-map", out_a,
        *C.V_CODEC, *C.A_CODEC, "-r", str(fps), args.out,
    ]
    C.run(cmd)
    C.ok(args.out, label=f"stitched {len(clips)} clips ({'xfade' if x>0 else 'cut'})")


if __name__ == "__main__":
    main()
