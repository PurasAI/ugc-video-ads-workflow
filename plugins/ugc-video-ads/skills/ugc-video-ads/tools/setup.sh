#!/usr/bin/env bash
# Bootstrap the ugc-video-ads skill so the user never touches pip/brew by hand.
# Creates a self-contained venv, installs the Python deps + ffmpeg, checks auth.
# Idempotent: fast no-op once set up, so SKILL.md can run it every session.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
MARKER="$VENV/.deps-ok"
PY="$VENV/bin/python3"

log() { printf '[ugc-setup] %s\n' "$*" >&2; }

# --- Python 3 (macOS ships only a stub; install via brew if it can't run) ----
ensure_python() {
  if python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
    return 0
  fi
  log "python3 (>=3.9) not available — installing…"
  if command -v brew >/dev/null 2>&1; then
    brew install python
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y python3 python3-venv
  else
    log "ERROR: install Python 3.9+ manually (https://www.python.org/downloads/) and re-run."
    exit 1
  fi
}

# --- ffmpeg + ffprobe --------------------------------------------------------
ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
    return 0
  fi
  log "ffmpeg not found — installing…"
  if command -v brew >/dev/null 2>&1; then
    brew install ffmpeg
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y ffmpeg
  else
    log "ERROR: install ffmpeg manually (https://ffmpeg.org/download.html) and re-run."
    exit 1
  fi
}

# --- venv + Python deps (reinstall only when requirements.txt changes) --------
ensure_venv() {
  if [ ! -x "$PY" ]; then
    log "creating venv at $VENV"
    python3 -m venv "$VENV"
  fi
  if [ -f "$MARKER" ] && [ ! "$ROOT/requirements.txt" -nt "$MARKER" ]; then
    return 0
  fi
  log "installing Python deps (first run pulls opencv/scipy — may take a minute)…"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$ROOT/requirements.txt"
  touch "$MARKER"
}

ensure_python
ensure_ffmpeg
ensure_venv

# --- Puras auth (interactive — can't be automated, just report) --------------
if ! "$VENV/bin/puras" whoami >/dev/null 2>&1; then
  log "Puras not logged in. Run once to fund AI renders:  $VENV/bin/puras login"
fi

log "ready ✔  run tools with:  $PY $ROOT/tools/<tool>.py"
