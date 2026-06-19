---
name: ugc-video-ads
description: >
  Create UGC video ads end-to-end using Puras hosted studios (Avatar Studio,
  Product Ad Studio, Game Ad Studio). Independent, à-la-carte modules: ingest a
  user's assets (videos, outros, logo, screenshots) into a working dir; scan
  TikTok/Instagram/competitor trends and suggest concepts; generate clips, hooks
  and outros via Puras skills (ugc-video, talking-avatar, promo, auto-caption,
  motion-ad-video, …); post-process locally with ffmpeg (stitch outro, burn
  captions, overlay logo + label) into a finished ad; and spin off A/B variants
  (aspect ratios, hooks). Use when the user wants to make, edit, caption,
  brand, or vary short-form video ads.
---

# UGC Video Ads — orchestrator

Make short-form UGC video ads by combining **Puras hosted studios** (the
generation muscle) with **local ffmpeg tools** (the finishing/editing muscle).

**These modules are independent — run whichever the user asks for, in any
order.** A user might only want a trend scan, or only to add an outro + logo to
a clip they already have, or the whole pipeline. Don't force the full sequence.

- **Skill root:** `${CLAUDE_SKILL_DIR}` — the folder this `SKILL.md` lives in
  (Claude Code sets this variable to the skill's install path); tools are in
  `${CLAUDE_SKILL_DIR}/tools/`.
- **Run tools with the skill's venv python** (Setup creates it):
  `${CLAUDE_SKILL_DIR}/.venv/bin/python3 ${CLAUDE_SKILL_DIR}/tools/<tool>.py …`
  The bash examples below write plain `python3 tools/…` for brevity — in practice
  use the venv python and absolute `${CLAUDE_SKILL_DIR}` paths (cwd is the user's
  project, not the skill).
- **Projects live in:** `${CLAUDE_SKILL_DIR}/projects/<slug>/` with
  `assets/ generated/ work/ out/` and a `project.json` inventory. Always work
  inside a project dir; create one with `ingest.py` (or just `mkdir`). Pass
  absolute paths to every tool.

## Setup — the skill installs its own dependencies
Before the first tool run, bootstrap the environment (idempotent — a fast no-op
once it's set up, so it's safe to run at the start of any session):
```bash
bash ${CLAUDE_SKILL_DIR}/tools/setup.sh
```
It creates a venv at `${CLAUDE_SKILL_DIR}/.venv`, installs the Python deps from
`requirements.txt`, and installs `ffmpeg` (via Homebrew/apt) if missing — **the
user never runs `pip` or `brew` by hand.** If it reports Puras is not logged in,
ask the user to run `${CLAUDE_SKILL_DIR}/.venv/bin/puras login` once (this funds
renders from the workspace balance).

> **⚠ Cost.** Generation (the `generate` module) runs on **puras.co and spends
> the workspace balance** per render (video models: Seedance/Kling, TTS,
> transcription). Post-processing & variants are **local ffmpeg = free**. Before
> kicking off paid renders (especially multiple ratios or variants), tell the
> user roughly what will run and confirm. Check balance with `puras whoami`.

---

## Module: ingest — pull the user's assets into a project
Collect existing assets (videos, outros, logo, screenshots, audio) and a brand
URL into a project working dir; classifies + probes them into `project.json`.

```bash
python3 tools/ingest.py --project <slug> --url <brand-url> <file|glob|http-url> …
```
Logos are auto-detected by filename (`logo`/`icon`/`wordmark`). Re-run to add
more. Read `projects/<slug>/project.json` to see what's available before
generating or editing.

---

## Module: trends — scan social + competitor ads, suggest concepts
Goal: a short `projects/<slug>/trends.md` with trending formats/hooks/sounds and
**3–5 concrete ad concepts** ready to feed `generate`. Two sources:

1. **WebSearch** (built-in). Run several focused queries, e.g.:
   - `"<category> TikTok ads trends <month year>"`, `"UGC hook formats trending <year>"`
   - `"<competitor/brand> Reels ads"`, `"trending TikTok sounds <category>"`
   - `"<product type> ad creative angles that convert"`
   Read the strongest results; extract recurring **formats** (unboxing, POV,
   problem→solution, tutorial, listicle…), **hook patterns** (first 3 seconds),
   and **sound/caption** conventions.
2. **Meta Ads Library** (MCP — competitor ads actually running now). Load it
   first: `ToolSearch("select:mcp__claude_ai_meta_ads__ads_library_search")`,
   then search by brand/keyword + country to see live creative angles, formats,
   and longevity (long-running ads = winners worth echoing).

Synthesize into `trends.md`: *what's trending → why → 3–5 concept briefs*
(format + hook + one message + suggested Puras skill). Don't copy competitors;
extract the pattern and adapt to the user's product.

---

## Module: generate — make clips/hooks/outros with Puras skills
Run any hosted skill and pull its media into the project's `generated/`. Two ways
to pass inputs:

**Preferred — a human-readable markdown brief** (`--brief`). YAML frontmatter
holds the structured params; the markdown body becomes the `brief` field. Easy
for the user to read and edit:
```bash
python3 tools/puras_skill.py --brief projects/<slug>/briefs/<name>.md \
    [--image KEY=PATH …] --download-dir projects/<slug>/generated
```
```markdown
---
skill: product-ad-studio/ugc-video
ugc_template: tutorial
aspect_ratios: ["9:16"]       # quote ratios — bare 9:16 is YAML base-60 = 556
duration_seconds: 10
captions: true
# music: upbeat acoustic     # optional, a STRING mood/genre — omit to auto-pick
---
Chordie AI is an AI guitar tutor… (the full ad brief, in plain prose the
user can read and tweak. This whole body is sent as the `brief` input.)
```
The positional skill path is optional when frontmatter sets `skill:`. `--json`
and `-i KEY=VALUE` still work and **override** frontmatter, so you can tweak one
field without rewriting the file. Write briefs to `projects/<slug>/briefs/`.

**Or — inline JSON** (quick one-offs):
```bash
python3 tools/puras_skill.py <studio>/<skill> \
    --json '<inputs JSON>' [--image KEY=PATH …] \
    --download-dir projects/<slug>/generated
```
It submits, polls (video jobs take minutes), saves `result-*.json`, and
downloads every output video/image. Local files (product photos, presenter
portrait, a clip to caption) go via `--image KEY=PATH` (inlined as base64 — keep
images small; repeat the flag for array fields like `product_images`).

> **Input types matter — a wrong type is a 400 error.** Match each field's type
> exactly. For `ugc-video`: `captions` is a **boolean**, but `music` is a
> **string** (a mood/genre like `"upbeat acoustic"`), not a boolean — omit it to
> let the skill auto-pick. `aspect_ratios` is a **list**. When unsure of a
> field's type, check the skill's page on https://puras.co/skills or its
> `skill.yaml` rather than guessing.

### Hosted skill catalog (paths = `<studio>/<skill>`)
**product-ad-studio**
- `ugc-video` — **the core UGC ad.** `brief` (text or product/App-Store URL, req),
  `product_images[]`, `ugc_template` (auto|unboxing|tryout|interview|tutorial|
  product_review|problem_solution|grwm|asmr_demo|pov|before_after|listicle|
  reaction|storytime), `creator_persona`, `creator_image`, `first_frame`/
  `last_frame` (stitch seams), `aspect_ratios[]` (`9:16`/`1:1`/`16:9`),
  `duration_seconds` (4–15, def 8), `music` (**string** mood/genre, optional —
  omit to auto-pick), `captions` (**boolean**, def true — burns karaoke captions
  in-skill). → `videos[].video_url`. **Planning a screen replacement?** Ground the
  phone screen with the bundled tracker image (`assets/tracker-green-3x3.png`) — see
  the **screen-replace** module, Step 1.
- `motion-ad-video` — storyboarded motion ad. `product-reveal-video` — cinematic
  reveal. `social-photo` / `static-image-ad` — product stills/image ads.
  `promo` — GSAP motion-graphics promo (`brief`, `product_name`, `accent_color`,
  `duration_sec`). `landing-page-designer` — landing page.

**avatar-studio**
- `talking-avatar` — lip-synced talking head. `script` (verbatim, req),
  `avatar_image` (presenter photo), `look`, `voice` (auto|warm_*|energetic_*|
  calm_narrator|authoritative_male), `language`, `aspect_ratios[]`. Great for
  hooks/outros with a spoken line. → `videos[].video_url`.

**game-ad-studio**
- `auto-caption` — burn word-synced karaoke captions onto ANY video. `video`
  (req — pass local clip via `--image video=PATH`), `brand_terms[]`,
  `language_code`, `caption_position`, colors. → `video_url` + `transcript`.
  (Use for clips that don't already have captions; `ugc-video` captions itself.)
- `aspect-ratio-converter` — reframe to other ratios. `game-ad-generator`,
  `2d-playable-ad`, `end-card-generator` — game ad formats / end cards.

**content-studio**: `content-repurposer` — repurpose one asset into many formats.

Discover exact inputs/examples for any skill via its page on
https://puras.co/skills or its `skill.yaml`.

---

## Module: post-process — finish a clip locally (free, ffmpeg)
Take a generated clip (or a user's own) and brand/assemble it. Compose these in
any order; each writes a new file (work in `projects/<slug>/work/`, final in
`out/`).

**Stitch hook → main → outro** (normalizes mismatched size/fps/audio):
```bash
python3 tools/add_outro.py --main main.mp4 --outro outro.mp4 --out work/stitched.mp4
# also: --intro hook.mp4, --clips a.mp4 b.mp4 c.mp4, --ratio 9:16, --xfade 0.5
```
**Overlay logo + label** (text is rendered with Pillow → works on any ffmpeg):
```bash
python3 tools/overlay.py --video in.mp4 --out out/final.mp4 \
    --logo logo.png --logo-pos tr --logo-scale 0.16 \
    --label "50% OFF TODAY" --label-pos bottom --label-box
```
**Captions:** `ugc-video` already burns them. For a clip without captions, either
run the hosted `game-ad-studio/auto-caption` (best — real transcription +
karaoke) via `generate`, or overlay a fixed `--label` for a simple on-screen line.

> **UI:** the logo/label overlay is parameterized by flags for now (no GUI). A
> visual "show the video, let the user place text/logo" editor is a planned
> follow-up (local web page → JSON back); until then, take placement/text from
> the user in chat and pass as flags, then iterate.

---

## Module: screen-replace — drop real UI into a green phone screen (free, OpenCV)
A **two-step pipeline**: **(1)** generate (or shoot) a clip whose phone holds the bundled
**green tracker screen**, then **(2)** replace that screen with your UI. The replace step
is **marker-assisted planar tracking** — markers detected every frame (drift-free,
sub-pixel), a homography warps the insert onto the screen, and the real green is keyed out
*on top* so the bezel and any finger/hand in front stay untouched. Deterministic classic
CV (no ML); rock-solid, jitter-free.

### Step 1 — generate the clip with the bundled tracker screen
The skill ships a ready tracker image tuned for this tool:
**`${CLAUDE_SKILL_DIR}/assets/tracker-green-3x3.png`** — a vivid chroma-green screen with
a **3×3 grid of black "+" markers** (auto-detected, reaches the corners, sub-pixel). When
generating, hand it to `ugc-video` as the **phone-screen reference** and describe it in
the brief, so the rendered phone holds exactly that screen:
```bash
python3 tools/puras_skill.py --brief projects/<slug>/briefs/<name>.md \
    --image first_frame=${CLAUDE_SKILL_DIR}/assets/tracker-green-3x3.png \
    --download-dir projects/<slug>/generated
```
In the brief body, say the creator **holds a phone whose screen is a solid green screen
showing a 3×3 grid of black "+" crosshair tracking markers**, held flat and front-facing.
(Pass the image to whichever `ugc-video` input grounds on-screen content — its
screen-grounding/first-frame reference; confirm the field on the skill's page.)

### Step 2 — replace the screen
```bash
# still UI image:
python3 tools/green_screen.py --video generated/clip.mp4 --image app-ui.png --out out/final.mp4
# or a looping video, + a tracking-overlay diagnostic:
python3 tools/green_screen.py --video generated/clip.mp4 --screen app-demo.mp4 \
    --out out/final.mp4 --debug debug.mp4
```

### ⚠ For the track to lock — keep these in the brief / when shooting
The bundled tracker image already satisfies "green + markers"; the rest is framing:
- **Flat & front-facing.** Phone as perpendicular to the camera as possible; **avoid
  large tilt/rotation** and fast whip motion (heavy perspective + motion blur is the hard
  case).
- **Whole screen in frame.** It must **stay fully visible** — don't let it clip at the
  frame edges or move partly in/out; markers leaving the frame break the lock.
- **No other big green areas** in the shot (a green wall/object gets keyed too), and
  nothing should fully cover the markers for long.
- **Custom screen?** Any regular marker grid works — the tool auto-detects **2×3 / 3×2 /
  2×2 …** and orientation; **6+ markers reaching the corners** track best.

### Low-confidence → black (graceful fallback)
Wherever those conditions break (too tilted, blurred, markers occluded or off-screen),
tracking confidence drops and the tool **keys the screen to solid black** instead of
showing a misaligned insert — crossfading in/out over a few frames so failures look
**intentional, never broken**. `--low-conf ui` forces the insert everywhere instead
(useful to inspect the raw track); `--debug` writes an overlay clip (green quad =
full-marker track, yellow = partial, orange = green-outline fallback, red = hold).

**Tuning:** narrow `--hue` / `--sat-min` / `--val-min` if the key grabs scene green;
`--rot-window` / `--scale-window` raise smoothing if a residual wobble remains. Marker
**shape-validation** (rejects non-cross blobs → kills false positives) is on by default;
`--no-shape-validate` disables it.

---

## Module: variants — A/B spin-offs (free, ffmpeg)
From one finished clip, fan out aspect ratios and/or hook-text variants:
```bash
python3 tools/make_variants.py --video out/final.mp4 --outdir out/variants \
    --ratios 9:16,1:1,16:9 --labels "Wait for it…|POV: you found it" --label-pos top
```
Writes the files + `variants.json`. Cap is 24 (ratios × labels). Reframe is
`cover` (fill+crop, the UGC norm); `--fit contain` to letterbox instead.

---

## A typical end-to-end flow (when the user wants the whole thing)
1. `ingest` the user's assets → project.  2. (optional) `trends` → concepts.
3. `generate` `ugc-video` from a brief/URL (+ product images) → clip in `generated/`.
4. `post-process`: `add_outro` (their outro) → `overlay` (logo + label) → `out/final.mp4`.
5. `variants` for ratios/hooks. Show the user the files in `out/`.

## Conventions & gotchas
- Always pass **absolute paths**; read `project.json` to find assets/outputs.
- `generate` is the only module that costs money — confirm before big/multi runs.
- This Homebrew ffmpeg lacks `drawtext` (no libfreetype) — that's why text is
  drawn via Pillow PNG overlays. Don't reintroduce drawtext.
- Long renders: `puras_skill.py` polls up to `--timeout` (def 900s). If it times
  out the job keeps running server-side and the tool prints its id — re-attach and
  download (no resubmit, no extra cost) with
  `puras_skill.py --resume <job_id> --download-dir projects/<slug>/generated`.
