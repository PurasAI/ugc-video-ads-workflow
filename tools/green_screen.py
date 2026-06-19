#!/usr/bin/env python3
"""Flicker-free green-screen replacement, built like a VFX screen-insert.

The amateur way — key the green every frame and let that noisy per-frame matte
define the visible edge — flickers, because compression/noise/motion-blur jiggle
the matte shape and the tracked corners each frame. This tool follows the
professional pipeline instead (Mocha planar track -> corner-pin -> locked screen
matte -> holdout for fingers), done deterministically and OFFLINE in two passes:

  PASS 1 (analyse). Detect the 4 screen corners on every frame, order them
  consistently, and accumulate each frame's green key warped into a canonical
  "plane space". Because the screen is rigid, averaging over time yields ONE
  rock-stable screen matte (the moving fingers average away). The corner tracks
  are then smoothed with a ZERO-PHASE filter (Savitzky-Golay) — offline, so it
  removes jitter with no lag.

  PASS 2 (render). Per frame, corner-pin the replacement onto the SMOOTH track
  (so its content sticks to the screen and never wobbles), warp the stable plane
  matte back to the frame to define the edge (flicker-free), subtract a per-frame
  finger holdout, despill, then composite as LAYERS:

      out = card * alpha + original * (1 - alpha)

  i.e. the replacement sits UNDERNEATH and shows only through the keyed screen;
  the bezel, hands and reflections remain because they're the original on top.

ffmpeg muxes the original audio back and re-encodes to the project's settings.

Examples
--------
  python3 green_screen.py --video ugc.mp4 --screen card.mp4 --out out.mp4
  python3 green_screen.py --video ugc.mp4 --screen card.mp4 --out out.mp4 \
      --smooth-window 13 --hue 35,85 --debug dbg.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


# ── geometry ─────────────────────────────────────────────────────────────────
def order_xsplit(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL via an x-split (robust near 45°)."""
    pts = pts.reshape(4, 2).astype(np.float32)
    xs = pts[np.argsort(pts[:, 0])]
    left = xs[:2][np.argsort(xs[:2, 1])]
    right = xs[2:]
    tl, bl = left[0], left[1]
    if np.linalg.norm(right[0] - tl) >= np.linalg.norm(right[1] - tl):
        br, tr = right[0], right[1]
    else:
        br, tr = right[1], right[0]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def temporal_align(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """Pick the cyclic rotation of `cur` closest to `prev` so corner identity is
    stable across frames (needed for trajectory smoothing)."""
    best, best_d = cur, float("inf")
    for r in range(4):
        rot = np.roll(cur, -r, axis=0)
        d = float(np.linalg.norm(rot - prev))
        if d < best_d:
            best, best_d = rot, d
    return best


def detect_quad(mask: np.ndarray, min_area: float, frame_area: int) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area * frame_area:
        return None
    peri = cv2.arcLength(c, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    return order_xsplit(quad)


def quad_aspect(quad: np.ndarray) -> float:
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    return float(w / h) if h > 1 else 0.5


def green_mask(frame: np.ndarray, hue: tuple[int, int], smin: int, vmin: int) -> np.ndarray:
    """Per-frame green key. Chroma channels are median-denoised first so block
    noise doesn't punch holes; CLOSE seals glare specks inside the screen."""
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 1] = cv2.medianBlur(ycc[:, :, 1], 3)
    ycc[:, :, 2] = cv2.medianBlur(ycc[:, :, 2], 3)
    hsv = cv2.cvtColor(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (hue[0], smin, vmin), (hue[1], 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    return m


def cover_crop(src: np.ndarray, aspect: float) -> np.ndarray:
    h, w = src.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return src[:, x0:x0 + nw]
    nh = int(round(w / aspect))
    y0 = (h - nh) // 2
    return src[y0:y0 + nh, :]


def despill(bgr: np.ndarray, region: np.ndarray, bias: float) -> np.ndarray:
    """Clamp G to bias*R+(1-bias)*B inside `region` (no luma add-back, which
    would tint the rim magenta). Neutralises green spill on finger/edge."""
    out = bgr.astype(np.float32)
    b, g, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    ref = bias * r + (1.0 - bias) * b
    sel = region.astype(bool)
    g[sel] = np.minimum(g, ref)[sel]
    return out.astype(np.uint8)


def _smooth(arr: np.ndarray, good: np.ndarray, window: int) -> np.ndarray:
    """Zero-phase smooth an (N,D) signal: interpolate gaps, then Savitzky-Golay
    (scipy) or a centered Gaussian fallback — both symmetric in time, no lag."""
    N, D = arr.shape
    out = arr.astype(np.float64).copy()
    idx = np.arange(N)
    g = good.astype(bool)
    if g.sum() < 2:
        return out
    for c in range(D):
        out[~g, c] = np.interp(idx[~g], idx[g], out[g, c])
    win = min(window if window % 2 else window + 1, N if N % 2 else N - 1)
    win = max(win, 3)
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(out, win, 2, axis=0, mode="interp")
    except Exception:
        r = win // 2
        x = np.arange(-r, r + 1)
        k = np.exp(-(x ** 2) / (2 * (r / 2.0 + 1e-6) ** 2)); k /= k.sum()
        sm = np.empty_like(out)
        for c in range(D):
            sm[:, c] = np.convolve(np.pad(out[:, c], r, mode="edge"), k, "valid")
        return sm


def track_pose(corners: np.ndarray, detected: np.ndarray, W: int, H: int,
               aspect: float, pos_window: int, rot_window: int) -> np.ndarray:
    """Recover the screen's rigid 3D pose each frame with solvePnP (IPPE, planar),
    smooth the physical pose — rotation HARD (it's nearly constant; this kills the
    spurious Z-roll), translation lighter (it must follow the hand) — then reproject
    the rectangle corners. Treating the screen as a fixed-aspect rigid rectangle is
    what stops 4 independently-detected corners from reading as rotation/shear."""
    N = len(corners)
    f = 1.2 * max(W, H)                       # focal guess; consistent K in & out
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], np.float64)
    objp = np.array([[0, 0, 0], [aspect, 0, 0], [aspect, 1, 0], [0, 1, 0]], np.float64)
    rv = np.zeros((N, 3)); tv = np.zeros((N, 3)); ok = np.zeros(N, bool)
    prevr = None
    for i in range(N):
        if not detected[i]:
            continue
        try:
            ret, rs, ts, _ = cv2.solvePnPGeneric(objp, corners[i].astype(np.float64),
                                                 K, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            ret = 0
        if not ret:
            continue
        j = 0
        if prevr is not None and len(rs) > 1:  # resolve planar ambiguity by continuity
            j = int(np.argmin([np.linalg.norm(r.reshape(3) - prevr) for r in rs]))
        rv[i] = rs[j].reshape(3); tv[i] = ts[j].reshape(3); ok[i] = True; prevr = rv[i]
    if not ok.any():
        return corners.astype(np.float32)
    rv = _smooth(rv, ok, rot_window)
    tv = _smooth(tv, ok, pos_window)
    out = np.zeros((N, 4, 2), np.float32)
    for i in range(N):
        pj, _ = cv2.projectPoints(objp, rv[i], tv[i], K, None)
        out[i] = pj.reshape(4, 2)
    return out


def keep_large(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop connected components smaller than min_area (kills glare specks so
    only real occluders — fingers — survive in the holdout)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 255
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Flicker-free green-screen replacement.")
    ap.add_argument("--video", required=True, help="base UGC clip (green screen)")
    ap.add_argument("--screen", required=True, help="video to play inside the screen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hue", default="35,85", help="green hue lo,hi (OpenCV 0-179)")
    ap.add_argument("--sat-min", type=int, default=35)
    ap.add_argument("--val-min", type=int, default=40)
    ap.add_argument("--pos-window", type=int, default=11,
                    help="zero-phase smoothing window for translation/position (frames; default 11)")
    ap.add_argument("--rot-window", type=int, default=25,
                    help="zero-phase smoothing window for rotation — large, since a handheld "
                         "phone's tilt is nearly constant; kills the spurious Z-roll (default 25)")
    ap.add_argument("--matte-thresh", type=float, default=0.35,
                    help="plane-space occupancy above which a pixel is 'screen' (default 0.35)")
    ap.add_argument("--grow", type=int, default=10,
                    help="px to grow the stable matte outward so it always covers the "
                         "per-frame green even when the smooth track lags it (default 10)")
    ap.add_argument("--feather", type=float, default=1.5, help="screen-edge feather sigma (px)")
    ap.add_argument("--holdout-feather", type=float, default=2.0, help="finger-edge feather (px)")
    ap.add_argument("--despill-bias", type=float, default=0.5)
    ap.add_argument("--min-area", type=float, default=0.01,
                    help="min green-blob area fraction to count as a detection")
    ap.add_argument("--debug", help="optional path: matte/track diagnostic clip")
    args = ap.parse_args()

    C.need_ffmpeg()
    for p in (args.video, args.screen):
        if not os.path.exists(p):
            C.die(f"file not found: {p}")
    try:
        h_lo, h_hi = (int(x) for x in args.hue.split(","))
    except ValueError:
        C.die("--hue must be 'lo,hi', e.g. 35,85")
    hue = (h_lo, h_hi)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        C.die(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or C.DEFAULT_FPS
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = W * H

    # ── PASS 1: detect corners + accumulate the stable plane matte ───────────
    corners_list: list[np.ndarray] = []
    detected_list: list[bool] = []
    aspects: list[float] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        m = green_mask(frame, hue, args.sat_min, args.val_min)
        q = detect_quad(m, args.min_area, frame_area)
        if q is not None:
            q = temporal_align(q, prev) if prev is not None else q
            prev = q
            corners_list.append(q)
            detected_list.append(True)
            aspects.append(quad_aspect(q))
        else:
            corners_list.append(prev if prev is not None else np.zeros((4, 2), np.float32))
            detected_list.append(False)
    N = len(corners_list)
    if N == 0:
        C.die("no frames read from --video")
    if not any(detected_list):
        C.die("no green screen detected — widen --hue/--sat-min/--val-min, --debug")

    aspect = float(np.median(aspects))
    PH = 1080
    PW = max(2, int(round(PH * aspect)))
    plane_corners = np.array([[0, 0], [PW, 0], [PW, PH], [0, PH]], dtype=np.float32)

    # accumulate each frame's green key into plane space; average -> stable matte
    corners = np.array(corners_list, dtype=np.float32)
    detected = np.array(detected_list, dtype=bool)
    acc = np.zeros((PH, PW), np.float32)
    cnt = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if detected[fi]:
            m = green_mask(frame, hue, args.sat_min, args.val_min)
            Hi2p = cv2.getPerspectiveTransform(corners[fi], plane_corners)
            warp = cv2.warpPerspective(m, Hi2p, (PW, PH), flags=cv2.INTER_LINEAR)
            acc += warp.astype(np.float32) / 255.0
            cnt += 1
        fi += 1
    cap.release()
    occ = acc / max(1, cnt)                                   # occupancy 0..1
    lo, hi = max(0.05, args.matte_thresh - 0.2), min(0.99, args.matte_thresh + 0.2)
    alpha_plane = np.clip((occ - lo) / (hi - lo), 0.0, 1.0)   # smoothstep-ish edge
    alpha_plane = cv2.GaussianBlur(alpha_plane, (0, 0), 2.0)

    # recover + smooth the rigid 3D pose, then reproject (no spurious Z-roll, no lag)
    corners_s = track_pose(corners, detected, W, H, aspect, args.pos_window, args.rot_window)

    # ── PASS 2: corner-pin the card + composite layers ───────────────────────
    rep = cv2.VideoCapture(args.screen)
    card_frames: list[np.ndarray] = []
    while True:
        ok, rf = rep.read()
        if not ok:
            break
        card_frames.append(cv2.resize(cover_crop(rf, aspect), (PW, PH)))
    rep.release()
    if not card_frames:
        C.die(f"no frames decoded from {args.screen}")

    tmp_fd, tmp_video = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_video, fourcc, fps, (W, H))
    if not writer.isOpened():
        C.die("cv2.VideoWriter failed to open")
    dbg = None
    if args.debug:
        C.ensure_parent(args.debug)
        dbg = cv2.VideoWriter(args.debug + ".tmp.mp4", fourcc, fps, (W, H))

    cap = cv2.VideoCapture(args.video)
    hold_buf: deque[np.ndarray] = deque(maxlen=3)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        S = corners_s[fi]
        Hp2i = cv2.getPerspectiveTransform(plane_corners, S)
        card = cv2.warpPerspective(card_frames[fi % len(card_frames)], Hp2i, (W, H),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        scr = cv2.warpPerspective(alpha_plane, Hp2i, (W, H), flags=cv2.INTER_LINEAR)
        scr = np.clip(scr, 0.0, 1.0)
        # grow the matte outward so it always covers the per-frame green even when
        # the (smooth) track lags the actual green during fast motion -> edge lands
        # on the black bezel, stays stable (no flicker) and leaks no green
        if args.grow > 0:
            gk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.grow * 2 + 1,) * 2)
            scr = cv2.dilate(scr, gk)
        screen_bin = (scr > 0.5).astype(np.uint8) * 255

        # finger holdout: non-green inside the screen, cleaned + temporally median'd.
        # erode past the grow margin so the bezel ring we just covered isn't mistaken
        # for an occluder
        gm = green_mask(frame, hue, args.sat_min, args.val_min)
        er = args.grow + 5
        inside = cv2.erode(screen_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er * 2 + 1,) * 2))
        fg = cv2.bitwise_and(inside, cv2.bitwise_not(gm))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        fg = keep_large(fg, max(50, int(0.01 * np.count_nonzero(screen_bin))))
        hold_buf.append(fg)
        fg = np.median(np.stack(hold_buf), axis=0).astype(np.uint8)
        holdout = cv2.GaussianBlur(fg.astype(np.float32) / 255.0, (0, 0), args.holdout_feather)

        scr = cv2.GaussianBlur(scr, (0, 0), args.feather)
        alpha = np.clip(scr * (1.0 - holdout), 0.0, 1.0)[:, :, None]

        region = cv2.dilate(screen_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        plate = despill(frame, region, args.despill_bias)
        out = (card.astype(np.float32) * alpha
               + plate.astype(np.float32) * (1.0 - alpha))
        # final safety despill: clamp any bright green that survived, keyed off the
        # green mask itself (dilated) so it works even where a leak fell outside the
        # matte. The clamp is a no-op on warm card/skin tones (already G<=(R+B)/2).
        sel = cv2.dilate(gm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
        out[:, :, 1][sel] = np.minimum(out[:, :, 1], (out[:, :, 2] + out[:, :, 0]) / 2.0)[sel]
        out = np.clip(out, 0, 255).astype(np.uint8)
        writer.write(out)

        if dbg is not None:
            d = frame.copy()
            d[gm > 0] = (0, 0, 255)
            cv2.polylines(d, [S.astype(np.int32)], True, (255, 255, 0), 2)
            cv2.polylines(d, [corners[fi].astype(np.int32)], True, (0, 255, 0), 1)
            dbg.write(d)
        fi += 1

    cap.release()
    writer.release()
    if dbg is not None:
        dbg.release()

    C.ensure_parent(args.out)
    cmd = ["ffmpeg", "-y", "-i", tmp_video, "-i", args.video,
           "-map", "0:v", "-map", "1:a?", "-shortest",
           *C.V_CODEC, *C.A_CODEC, args.out]
    try:
        C.run(cmd)
    finally:
        if os.path.exists(tmp_video):
            os.unlink(tmp_video)
    if dbg is not None:
        C.run(["ffmpeg", "-y", "-i", args.debug + ".tmp.mp4", *C.V_CODEC, args.debug])
        os.unlink(args.debug + ".tmp.mp4")

    pct = round(100 * int(detected.sum()) / max(1, N))
    C.ok(args.out, label=f"green-screen replaced, flicker-free ({pct}% of {N} frames tracked)")


if __name__ == "__main__":
    main()
