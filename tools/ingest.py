#!/usr/bin/env python3
"""Collect a user's existing assets into a project working directory.

Creates `<projects-dir>/<slug>/{assets,generated,work,out}`, copies the given
files in (classified into assets/{video,image,logo,audio,other}), can download
remote URLs, and writes a `project.json` inventory (with probed dims/duration
for media). Re-runnable: it merges new assets into an existing project.

Examples
--------
  python3 ingest.py --project acme \
      ~/Downloads/outro.mp4 ~/brand/logo.png ~/screens/*.png

  # remote assets + a brand URL recorded on the project
  python3 ingest.py --project acme --url https://acme.com \
      https://cdn.acme.com/hero.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_EXT = {".mp4", ".mov", ".webm", ".m4v", ".avi", ".mkv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def classify(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    name = os.path.basename(path).lower()
    if ext in IMAGE_EXT and ("logo" in name or "wordmark" in name or "icon" in name):
        return "logo"
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    if ext in AUDIO_EXT:
        return "audio"
    return "other"


def download(url: str, dest_dir: str) -> str:
    import httpx
    name = os.path.basename(urlparse(url).path) or "asset"
    dest = os.path.join(dest_dir, name)
    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 16):
                f.write(chunk)
    return dest


def describe(path: str, kind: str) -> dict:
    info = {"file": os.path.abspath(path), "type": kind,
            "size_kb": os.path.getsize(path) // 1024}
    if kind in ("video", "image"):
        try:
            if kind == "video":
                p = C.probe(path)
                info.update(width=p.width, height=p.height, ratio=p.ratio_name,
                            fps=p.fps, duration=round(p.duration, 2),
                            has_audio=p.has_audio)
            else:
                from PIL import Image
                with Image.open(path) as im:
                    info.update(width=im.width, height=im.height,
                                ratio=_ratio_name(im.width, im.height))
        except Exception:  # noqa: BLE001 — inventory is best-effort
            pass
    return info


def _ratio_name(w: int, h: int) -> str:
    if not h:
        return "?"
    r = w / h
    best, bn = 9.9, "?"
    for name, (cw, ch) in C.RATIO_CANVAS.items():
        d = abs(r - cw / ch)
        if d < best:
            best, bn = d, name
    return bn if best < 0.06 else f"{w}x{h}"


def main():
    ap = argparse.ArgumentParser(description="Ingest assets into a project dir.")
    ap.add_argument("--project", required=True, help="project slug")
    ap.add_argument("--projects-dir", default=os.path.join(SKILL_ROOT, "projects"))
    ap.add_argument("--url", action="append", help="brand/product URL to record")
    ap.add_argument("sources", nargs="*", help="local files and/or http(s) URLs")
    args = ap.parse_args()

    proj = os.path.join(args.projects_dir, args.project)
    dirs = {d: os.path.join(proj, d) for d in ("assets", "generated", "work", "out")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    meta_path = os.path.join(proj, "project.json")
    meta = {"slug": args.project, "created": int(time.time()), "urls": [],
            "assets": []}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
        except (OSError, ValueError):
            pass
    meta.setdefault("urls", [])
    meta.setdefault("assets", [])

    for u in args.url or []:
        if u not in meta["urls"]:
            meta["urls"].append(u)

    known = {a["file"] for a in meta["assets"]}
    added = []
    for src in args.sources:
        if src.startswith(("http://", "https://")):
            tmp = download(src, dirs["assets"])
            kind = classify(tmp)
            final = os.path.join(dirs["assets"], kind, os.path.basename(tmp))
            os.makedirs(os.path.dirname(final), exist_ok=True)
            shutil.move(tmp, final)
        else:
            if not os.path.exists(src):
                C.die(f"source not found: {src}")
            kind = classify(src)
            final = os.path.join(dirs["assets"], kind, os.path.basename(src))
            os.makedirs(os.path.dirname(final), exist_ok=True)
            shutil.copy2(src, final)
        rec = describe(final, kind)
        if rec["file"] not in known:
            meta["assets"].append(rec)
            known.add(rec["file"])
            added.append(rec)

    meta["paths"] = {k: os.path.abspath(v) for k, v in dirs.items()}
    meta["root"] = os.path.abspath(proj)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    by_type: dict[str, int] = {}
    for a in meta["assets"]:
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    print(json.dumps({"ok": True, "project": args.project,
                      "root": meta["root"], "paths": meta["paths"],
                      "added": len(added), "total_assets": len(meta["assets"]),
                      "by_type": by_type, "urls": meta["urls"],
                      "project_json": os.path.abspath(meta_path)}, indent=2))


if __name__ == "__main__":
    main()
