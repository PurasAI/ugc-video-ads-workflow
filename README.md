# ugc-video-ads — a Claude Code skill

Create UGC video ads end-to-end by orchestrating **Puras hosted studios**
(Avatar / Product Ad / Game Ad Studio) for generation and **local ffmpeg tools**
for finishing. The modules are independent — run whichever you need.

| Module | Tool | Cost | What it does |
|---|---|---|---|
| ingest | `tools/ingest.py` | free | Collect assets (video/outro/logo/screenshots) into a project dir |
| trends | *(SKILL.md playbook)* | free | WebSearch + Meta Ads Library → trend scan + concepts |
| generate | `tools/puras_skill.py` | **$ (workspace balance)** | Run `ugc-video`, `talking-avatar`, `promo`, `auto-caption`, … and download outputs |
| post-process | `tools/add_outro.py`, `tools/overlay.py` | free | Stitch hook/outro, overlay logo + label |
| screen-replace | `tools/green_screen.py` | free | Replace a green phone/laptop screen with real UI (chroma-key + homography) |
| variants | `tools/make_variants.py` | free | A/B spin-offs (aspect ratios, hooks) |

## Setup
```bash
pip install -r requirements.txt   # puras, Pillow, httpx
brew install ffmpeg               # ffmpeg + ffprobe
puras login                       # workspace auth (funds renders)
```

## Quick start
```bash
ROOT=/Users/mehmetecevit/work/purasroot/ugc-video-ads
python3 $ROOT/tools/ingest.py --project acme --url https://acme.com ~/brand/logo.png ~/outro.mp4
python3 $ROOT/tools/puras_skill.py product-ad-studio/ugc-video \
    --json '{"brief":"https://acme.com","aspect_ratios":["9:16"],"duration_seconds":8}' \
    --download-dir $ROOT/projects/acme/generated
python3 $ROOT/tools/add_outro.py --main <clip> --outro <outro> --out $ROOT/projects/acme/work/stitched.mp4
python3 $ROOT/tools/overlay.py --video <stitched> --logo <logo> --logo-pos tr \
    --label "Try it free" --label-pos bottom --label-box --out $ROOT/projects/acme/out/final.mp4
python3 $ROOT/tools/make_variants.py --video <final> --outdir $ROOT/projects/acme/out/variants --ratios 9:16,1:1
```

See **[SKILL.md](SKILL.md)** for the full operating manual (skill catalog, inputs,
trends playbook, conventions). Projects and outputs live under `projects/<slug>/`.

> Generation spends the Puras workspace balance per render; post-processing and
> variants are free local ffmpeg. Confirm before large/multi-render runs.

UI note: logo/label placement is flag-driven for now; a visual overlay editor
(local web page) is a planned follow-up.
