#!/usr/bin/env python3
"""Run a hosted Puras skill and pull its media outputs into a local folder.

Thin wrapper over the `puras` Python client. It submits a job to a deployed
skill (e.g. `product-ad-studio/ugc-video`), client-side-polls until it finishes
(media jobs run for minutes), saves the raw result, and downloads every media
URL in the output to `--download-dir`.

Auth: PURAS_API_KEY / PURAS_API_BASE env, else ~/.puras/config.json (written by
`puras login`). Local files passed with `--image KEY=PATH` are inlined as
base64 data URLs, which hosted skills accept for `image`/`video` inputs.

A markdown brief (`--brief`) is the readable alternative to `--json`: YAML
frontmatter becomes the structured inputs, the body becomes the `brief` field,
and a `skill:` key in the frontmatter makes the positional skill arg optional.

Examples
--------
  # from a human-readable markdown brief (frontmatter → inputs, body → brief)
  python3 puras_skill.py --brief projects/acme/briefs/hook.md \
      --download-dir projects/acme/generated

  # UGC clip from a brief (captions burned in by the skill), download the mp4
  python3 puras_skill.py product-ad-studio/ugc-video \
      --json '{"brief":"https://apps.apple.com/...","aspect_ratios":["9:16"],"duration_seconds":8}' \
      --download-dir projects/acme/generated

  # talking-avatar with a local presenter photo
  python3 puras_skill.py avatar-studio/talking-avatar \
      --json '{"script":"Big news — dark mode is here."}' \
      --image avatar_image=assets/presenter.jpg \
      --download-dir projects/acme/generated

  # caption an already-rendered clip with the hosted karaoke captioner
  python3 puras_skill.py game-ad-studio/auto-caption \
      --image video=generated/clip.mp4 --json '{"brand_terms":["Acme"]}' \
      --download-dir projects/acme/generated
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from urllib.parse import urlparse

TERMINAL = ("succeeded", "failed", "cancelled")
MEDIA_KEYS = {"video_url", "video", "image", "image_url", "url", "audio",
              "audio_url", "drive_path", "poster", "thumbnail", "logo",
              "screenshot", "preview"}
MEDIA_EXT = (".mp4", ".mov", ".webm", ".m4v", ".png", ".jpg", ".jpeg",
             ".webp", ".gif", ".mp3", ".wav", ".m4a")


def die(msg: str, code: int = 1):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def load_auth() -> tuple[str, str | None]:
    key = os.environ.get("PURAS_API_KEY")
    base = os.environ.get("PURAS_API_BASE")
    cfg = os.path.expanduser("~/.puras/config.json")
    if (not key or not base) and os.path.exists(cfg):
        try:
            with open(cfg) as f:
                d = json.load(f)
            key = key or d.get("api_key")
            base = base or d.get("api_base")
        except (OSError, ValueError):
            pass
    if not key:
        die("no Puras API key — set PURAS_API_KEY or run `puras login`")
    return key, base


def data_url(path: str) -> str:
    if not os.path.exists(path):
        die(f"input file not found: {path}")
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{b64}"


_BRIEF_LOADER = None


def _brief_loader():
    """A YAML SafeLoader that does NOT treat `9:16` as base-60 (=556).

    YAML 1.1's implicit int/float resolvers parse `N:M[:…]` as a sexagesimal
    number, silently turning aspect ratios like `9:16` into `556`. We drop those
    two resolvers and re-add decimal/hex/octal-only ones, so `9:16` stays the
    string we want while `duration_seconds: 10` still parses as an int. Cached.
    """
    global _BRIEF_LOADER
    if _BRIEF_LOADER is not None:
        return _BRIEF_LOADER
    try:
        import yaml
    except ImportError:
        die("`--brief` needs PyYAML — pip install pyyaml")

    class L(yaml.SafeLoader):
        pass

    L.yaml_implicit_resolvers = {
        ch: [(tag, rx) for (tag, rx) in mapping
             if tag not in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float")]
        for ch, mapping in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    L.add_implicit_resolver(
        "tag:yaml.org,2002:int",
        re.compile(r"^[-+]?(?:0b[0-1_]+|0o?[0-7_]+|0x[0-9a-fA-F_]+|0|[1-9][0-9_]*)$"),
        list("-+0123456789"))
    L.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9_]+|[0-9][0-9_]*[eE][-+]?[0-9]+)$"),
        list("-+0123456789."))
    _BRIEF_LOADER = L
    return L


def parse_brief_md(path: str) -> tuple[dict, str]:
    """Split a markdown brief into (frontmatter dict, body string).

    Frontmatter is a YAML mapping fenced by `---` lines at the top of the file;
    everything after it is the body (sent as the `brief` field). A file with no
    frontmatter is treated as all-body.
    """
    if not os.path.exists(path):
        die(f"brief file not found: {path}")
    text = open(path).read()
    meta: dict = {}
    body = text
    stripped = text.lstrip()
    if stripped.startswith("---"):
        rest = stripped[3:]
        end = rest.find("\n---")
        if end == -1:
            die(f"brief {path}: opening `---` has no closing `---`")
        loader = _brief_loader()  # friendly error if PyYAML is missing
        import yaml
        try:
            meta = yaml.load(rest[:end], Loader=loader) or {}
        except yaml.YAMLError as e:
            die(f"brief frontmatter is not valid YAML: {e}")
        if not isinstance(meta, dict):
            die("brief frontmatter must be a YAML mapping (key: value)")
        # body starts after the closing `---` line
        body = rest[end + 4:]
    return meta, body.strip()


def build_inputs(args) -> tuple[dict, str | None]:
    """Return (inputs, skill_from_brief). Precedence: brief < --json < -i < --image."""
    inputs: dict = {}
    brief_skill: str | None = None
    if args.brief:
        meta, body = parse_brief_md(args.brief)
        brief_skill = meta.pop("skill", None)
        inputs.update(meta)
        if body and "brief" not in inputs:
            inputs["brief"] = body
    if args.json:
        try:
            parsed = json.loads(args.json)
        except json.JSONDecodeError as e:
            die(f"--json is not valid JSON: {e}")
        if not isinstance(parsed, dict):
            die("--json must be a JSON object")
        inputs.update(parsed)
    for kv in args.input or []:
        k, _, v = kv.partition("=")
        inputs[k.strip()] = v
    # --image KEY=PATH (repeatable): repeats on the same key become a list
    grouped: dict[str, list[str]] = {}
    for kv in args.image or []:
        k, _, p = kv.partition("=")
        grouped.setdefault(k.strip(), []).append(data_url(p.strip()))
    for k, vals in grouped.items():
        existing = inputs.get(k)
        if isinstance(existing, list):
            existing.extend(vals)
        elif len(vals) > 1:
            inputs[k] = vals
        else:
            inputs[k] = vals[0]
    return inputs, brief_skill


def iter_strings(obj, key=None):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_strings(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_strings(v, key)
    elif isinstance(obj, str):
        yield key, obj


def is_media_url(key, s: str) -> bool:
    if not re.match(r"^https?://", s):
        return False
    path = urlparse(s).path.lower()
    return key in MEDIA_KEYS or path.endswith(MEDIA_EXT)


def download(url: str, dest: str) -> str:
    import httpx
    with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1 << 16):
                f.write(chunk)
    return ctype


def main():
    ap = argparse.ArgumentParser(description="Run a hosted Puras skill, pull outputs.")
    ap.add_argument("skill", nargs="?",
                    help="skill path, e.g. product-ad-studio/ugc-video or "
                         "puras/product-ad-studio/ugc-video (optional when the "
                         "--brief frontmatter sets `skill:`)")
    ap.add_argument("--brief", help="markdown brief file: YAML frontmatter → "
                                    "inputs, body → the `brief` field")
    ap.add_argument("--json", help="inputs as a JSON object (overrides --brief)")
    ap.add_argument("-i", "--input", action="append", metavar="KEY=VALUE",
                    help="scalar string input (repeatable)")
    ap.add_argument("--image", action="append", metavar="KEY=PATH",
                    help="local file inlined as a base64 data URL (repeatable; "
                         "repeat a key for an array field like product_images)")
    ap.add_argument("--download-dir", help="where to save downloaded media")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--version", type=int, help="pin to a deployment version")
    ap.add_argument("--timeout", type=int, default=900, help="max seconds to wait")
    ap.add_argument("--poll", type=float, default=3.0, help="poll interval seconds")
    ap.add_argument("--prefix", default="", help="filename prefix for downloads")
    args = ap.parse_args()

    api_key, api_base = load_auth()
    try:
        import puras
    except ImportError:
        die("`puras` package not importable — pip install puras")

    inputs, brief_skill = build_inputs(args)
    skill = args.skill or brief_skill
    if not skill:
        die("no skill given — pass it positionally or set `skill:` in the "
            "--brief frontmatter")
    client = puras.Client(api_key=api_key, api_base=api_base)

    log(f"→ submitting {skill}  (inputs: {', '.join(sorted(inputs)) or 'none'})")
    try:
        job = client.submit(skill, inputs, version=args.version)
    except Exception as e:  # noqa: BLE001 — surface any submit failure cleanly
        die(f"submit failed: {e}")

    jid = job["id"]
    log(f"  job {jid} · {job.get('status')}")
    start = time.monotonic()
    last = job.get("status")
    while job.get("status") not in TERMINAL:
        if time.monotonic() - start > args.timeout:
            die(f"timeout after {args.timeout}s — job {jid} still "
                f"{job.get('status')}; `puras logs {jid}` to keep watching")
        time.sleep(args.poll)
        try:
            job = client.get(jid)
        except Exception as e:  # noqa: BLE001
            log(f"  (poll error, retrying: {e})")
            continue
        if job.get("status") != last:
            last = job.get("status")
            log(f"  … {last}  (+{int(time.monotonic() - start)}s)")

    status = job.get("status")
    result = job.get("result") or {}
    output = result.get("output")

    out_dir = args.download_dir
    saved = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"result-{jid[:8]}.json"), "w") as f:
            json.dump(job, f, indent=2, default=str)

    if status != "succeeded":
        print(json.dumps({"ok": False, "job_id": jid, "status": status,
                          "error": job.get("error"), "result": result},
                         indent=2, default=str))
        die(f"job {status}", code=2)

    downloads = []
    if output is not None and not args.no_download and out_dir:
        seen, counters, unresolved = set(), {}, []
        for key, s in iter_strings(output):
            if is_media_url(key, s):
                if s in seen:
                    continue
                seen.add(s)
                kslug = re.sub(r"[^a-z0-9]+", "-", (key or "media").lower()).strip("-")
                n = counters.get(kslug, 0)
                counters[kslug] = n + 1
                ext = os.path.splitext(urlparse(s).path)[1].lower()
                dest = os.path.join(out_dir, f"{args.prefix}{kslug}-{n}{ext or ''}")
                try:
                    ctype = download(s, dest)
                except Exception as e:  # noqa: BLE001
                    log(f"  ! download failed for {key}: {e}")
                    continue
                if not ext:
                    guessed = mimetypes.guess_extension((ctype or "").split(";")[0].strip())
                    if guessed and not dest.endswith(guessed):
                        os.rename(dest, dest + guessed)
                        dest += guessed
                downloads.append({"key": key, "url": s, "file": os.path.abspath(dest)})
                log(f"  ↓ {key} → {os.path.basename(dest)}")
            elif key in MEDIA_KEYS and "/" in s and not s.startswith("data:"):
                unresolved.append({"key": key, "value": s})
        if unresolved:
            log(f"  ! {len(unresolved)} media field(s) came back as bare drive "
                f"paths, not URLs — see result-{jid[:8]}.json")

    summary = {"ok": True, "job_id": jid, "status": status,
               "steps": result.get("steps"), "skill": skill,
               "downloads": downloads, "output": output}
    if out_dir:
        summary["result_file"] = os.path.abspath(
            os.path.join(out_dir, f"result-{jid[:8]}.json"))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
