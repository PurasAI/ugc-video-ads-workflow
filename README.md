# UGC Video Ads Workflow — a Claude Code skill

A [Claude Code](https://claude.com/claude-code) skill that creates short-form
**UGC video ads end-to-end**. It orchestrates **Puras hosted studios**
(Avatar / Product Ad / Game Ad Studio) for the AI generation and **local ffmpeg
tools** for the finishing. You describe the ad you want in chat — Claude picks
the steps, writes the briefs, runs the tools, and hands you the finished files.

The modules are independent, so you can ask for the whole pipeline or just one
step (e.g. *"add my outro + logo to this clip"*).

| Module | What it does | Cost |
|---|---|---|
| **ingest** | Collect assets (video / outro / logo / screenshots) into a project dir | free |
| **trends** | WebSearch + Meta Ads Library → trend scan + ready-to-use ad concepts | free |
| **generate** | Run Puras skills (`ugc-video`, `talking-avatar`, `promo`, `auto-caption`, …) and download the output | **$ (Puras balance)** |
| **post-process** | Stitch hook/outro, overlay logo + label | free |
| **screen-replace** | Replace a green phone/laptop screen with real UI (chroma-key + homography) | free |
| **variants** | A/B spin-offs (aspect ratios, hooks) | free |

## Install

Install it as a Claude Code **plugin** — two slash commands, typed **inside
Claude Code**. No terminal, nothing to move:

```text
/plugin marketplace add PurasAI/ugc-video-ads-workflow
/plugin install ugc-video-ads@purasai
```

That's the whole install. The first time you ask for an ad, the skill
**bootstraps its own dependencies** (a local Python venv + `ffmpeg`) — you never
run `pip` or `brew` yourself. The only thing it can't do for you is log in to
Puras (that funds the AI renders): if you're not logged in, Claude tells you to
run `puras login` once.

<details>
<summary>Manual / developer install (no plugin manager)</summary>

Clone the repo and copy the skill into your Claude Code skills folder — Claude
Code auto-discovers it (no restart; run `/skills` to confirm):

```bash
git clone https://github.com/PurasAI/ugc-video-ads-workflow.git /tmp/uva
cp -R /tmp/uva/plugins/ugc-video-ads/skills/ugc-video-ads ~/.claude/skills/ugc-video-ads
```

Dependencies still auto-install on first use — or prime them now with
`bash ~/.claude/skills/ugc-video-ads/tools/setup.sh`.
</details>

## Getting started

You don't run the tools yourself — you ask Claude in plain language and it drives
the skill. Some things to try:

**Make a full ad from a URL**
> "Make a 9:16 UGC ad for https://chordie.ai — tutorial style, 10s. Then stitch
> my outro (`~/brand/outro.mp4`) and put my logo (`~/brand/logo.png`) top-right
> with a 'Try it free' label."

**Just finish a clip you already have**
> "Add my outro and logo to `~/Downloads/clip.mp4` and put '50% OFF TODAY' across
> the bottom."

**Scout what's working before you create**
> "Scan TikTok + competitor ad trends for AI guitar apps and give me 5 ad concepts."

**Spin off A/B variants**
> "Take `out/final.mp4` and give me 9:16, 1:1 and 16:9 versions with two different
> hook texts."

**Replace a green screen with real UI**
> "This clip has a phone with a green screen — drop my app demo (`app-demo.mp4`)
> onto it."

Claude organizes everything under `projects/<slug>/` (`assets/ generated/ work/
out/`) and shows you the finished files in `out/`. For the full operating manual
(skill catalog, input types, the trends playbook, conventions), see
**[SKILL.md](plugins/ugc-video-ads/skills/ugc-video-ads/SKILL.md)**.

## FAQ

**What can I make with this skill?**
Short-form video ads for TikTok / Reels / Shorts: UGC-style spots (unboxing,
tutorial, POV, problem→solution, before/after, listicle, …), talking-head hooks
& outros, motion-graphics promos, cinematic product reveals, and static image
ads — plus all the finishing work (captions, logo/label overlays, outro
stitching, green-screen UI replacement) and A/B variants in any aspect ratio.

**What is Puras?**
[Puras](https://puras.co) is a hosted AI creative platform — a set of "studios"
(Avatar, Product Ad, Game Ad, Content) exposing skills like `ugc-video`,
`talking-avatar`, and `auto-caption` that generate video, audio and images on the
server. This Claude Code skill calls those Puras skills for the heavy AI
generation and assembles the results locally. Browse the catalog at
[puras.co/skills](https://puras.co/skills).

**Does it cost money?**
Only the **generate** module does. AI renders run on puras.co and spend your
**Puras workspace balance** (video models, TTS, transcription). Everything else —
ingest, trends, post-processing, screen-replace, variants — is **free local
ffmpeg / OpenCV**. Claude tells you roughly what will run before kicking off paid
renders; check your balance any time with `puras whoami`.

**Do I need to know ffmpeg or write code?**
No. You describe the ad in chat; Claude picks the modules, writes the briefs, and
runs the tools. The Python scripts in `tools/` are the skill's machinery, not
something you call by hand.

**Where do my files go?**
Everything lives under the skill's `projects/<slug>/` folder — `assets/` (your
inputs), `generated/` (AI output), `work/` (intermediate), `out/` (finished ads +
variants). These are gitignored, so your projects stay local.

**Can I run just one piece?**
Yes — the modules are à-la-carte. *"Just add captions to this clip"* or *"just
give me a trend report"* both work without the rest of the pipeline.

**What do I need installed?**
Just [Claude Code](https://claude.com/claude-code) and a [Puras](https://puras.co)
account. The skill installs its own Python deps and `ffmpeg` on first run (a local
venv + Homebrew/apt) — you only run `puras login` once to fund renders. See
[Install](#install) above.
