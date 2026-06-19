#!/usr/bin/env python3
"""Flicker-free green-screen replacement, built like a VFX screen-insert.

Compositing model (what the user asked for): the replacement is the BOTTOM layer
and the original clip is the TOP layer with the green keyed out — so the insert
shows ONLY through the green screen, while the bezel, hands and reflections stay
because they're the original on top. Mathematically, per frame:

    out = card · greenAlpha + original · (1 - greenAlpha)

Stability is the hard part. Three things would otherwise wobble:

  • CONTENT JITTER / Z-ROLL — fixed by not trusting 4 independently-detected
    corners. We track the screen as a rigid, fixed-aspect rectangle and decompose
    its motion into position / rotation / uniform-scale / perspective-residual,
    then smooth EACH channel with a zero-phase filter (offline ⇒ no lag): rotation
    and scale hard (a handheld phone's tilt and distance barely change — this is
    what kills the spurious roll and the size pulsing), position lighter (it must
    follow the hand). We deliberately avoid solvePnP: its depth is weakly
    observable for a near-frontal planar target and injects scale pulsing.

  • EDGE FLICKER — the visible edge is the real screen edge (the green key), so it
    sits exactly on the bezel. The card is over-filled under it and clipped by the
    key, so no green leaks and nothing spills onto the bezel.

Readability: the card is down-sampled with INTER_AREA (true anti-aliasing) and
warped with INTER_LANCZOS4, so the text stays crisp at screen size.

ffmpeg muxes the original audio back and re-encodes to the project's settings.

Examples
--------
  python3 green_screen.py --video ugc.mp4 --screen card.mp4 --out out.mp4
  python3 green_screen.py --video ugc.mp4 --screen card.mp4 --out out.mp4 \
      --rot-window 31 --scale-window 31 --overfill 1.06 --debug dbg.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

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


def expand(quad: np.ndarray, factor: float) -> np.ndarray:
    ctr = quad.mean(axis=0)
    return (ctr + (quad - ctr) * factor).astype(np.float32)


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
    """Clamp G to bias*R+(1-bias)*B inside `region` (no luma add-back, which would
    tint the rim magenta). Neutralises green spill on the key edge / fingers."""
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


def track_similarity(corners: np.ndarray, detected: np.ndarray, aspect: float,
                     pos_w: int, rot_w: int, scale_w: int, persp_w: int) -> np.ndarray:
    """Decompose each frame's quad into position / rotation / uniform-scale (via a
    Procrustes fit to a canonical fixed-aspect rectangle) plus a perspective
    residual, smooth each channel with its own zero-phase window, and recompose.

    This is the crux of the stability: scale and rotation are smoothed HARD (a
    handheld phone's distance and tilt are nearly constant — high-frequency
    variation there is detector noise, not real motion), so the insert neither
    pulses in size nor rolls in Z, while position tracks the hand. Unlike
    solvePnP, scale is an explicit, directly-smoothed channel — no depth pulsing."""
    N = len(corners)
    Q0 = np.array([[-aspect / 2, -0.5], [aspect / 2, -0.5],
                   [aspect / 2, 0.5], [-aspect / 2, 0.5]], np.float64)
    q0_ss = (Q0 ** 2).sum() / 4.0
    s = np.zeros(N); th = np.zeros(N); t = np.zeros((N, 2)); res = np.zeros((N, 8))
    for i in range(N):
        q = corners[i].astype(np.float64)
        mu = q.mean(0)
        Sig = ((q - mu).T @ Q0) / 4.0
        U, D, Vt = np.linalg.svd(Sig)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        s[i] = max(1e-6, D.sum() / q0_ss)
        th[i] = np.arctan2(R[1, 0], R[0, 0])
        t[i] = mu
        fit = (s[i] * (R @ Q0.T)).T + mu
        res[i] = (q - fit).reshape(8)
    th = np.unwrap(th)
    g = detected.astype(bool)
    s_s = np.exp(_smooth(np.log(s).reshape(-1, 1), g, scale_w)).ravel()
    th_s = _smooth(th.reshape(-1, 1), g, rot_w).ravel()
    t_s = _smooth(t, g, pos_w)
    res_s = _smooth(res, g, persp_w).reshape(N, 4, 2)
    out = np.zeros((N, 4, 2), np.float32)
    for i in range(N):
        c, si = np.cos(th_s[i]), np.sin(th_s[i])
        R = np.array([[c, -si], [si, c]])
        out[i] = (s_s[i] * (R @ Q0.T)).T + t_s[i] + res_s[i]
    return out


def track_klt(grays: list[np.ndarray], ref: int, ref_quad: np.ndarray,
              ref_mask: np.ndarray, n_pts: int, win: int,
              fb_thresh: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Track the screen by following rigid feature points on the BEZEL/body just
    OUTSIDE the green (the green interior is featureless). Seed ~n_pts points in a
    ring around the screen on a reference frame, track them bidirectionally with
    pyramidal Lucas-Kanade + a forward-backward consistency check (drop any point
    whose round-trip error exceeds fb_thresh), then per frame fit a RANSAC
    homography from the surviving points and map the reference screen quad through
    it. Because the SAME physical points are tracked continuously, the resulting
    quad is far more temporally consistent than re-detecting green corners each
    frame (whose landing on the rounded corners jitters → apparent Z-roll).

    Returns (quad[N,4,2], good[N]) or None if too few points could be seeded."""
    N = len(grays)
    H, W = grays[ref].shape[:2]
    ring = cv2.subtract(cv2.dilate(ref_mask, np.ones((75, 75), np.uint8)),
                        cv2.dilate(ref_mask, np.ones((7, 7), np.uint8)))
    p0 = cv2.goodFeaturesToTrack(grays[ref], n_pts, 0.01, 10, blockSize=9, mask=ring)
    if p0 is None or len(p0) < 4:
        return None
    M = len(p0)
    lk = dict(winSize=(win, win), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    pos = np.full((N, M, 2), np.nan, np.float32)
    alive = np.zeros((N, M), bool)
    pos[ref] = p0.reshape(M, 2); alive[ref] = True

    def sweep(rng: range, step: int) -> None:
        cur = p0.copy(); al = np.ones(M, bool)
        for f in rng:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(grays[f - step], grays[f], cur, None, **lk)
            bak, st2, _ = cv2.calcOpticalFlowPyrLK(grays[f], grays[f - step], nxt, None, **lk)
            fb = np.linalg.norm((cur - bak).reshape(M, 2), axis=1)
            al &= (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_thresh)
            pos[f] = nxt.reshape(M, 2); alive[f] = al; cur = nxt

    sweep(range(ref + 1, N), 1)
    sweep(range(ref - 1, -1, -1), -1)

    p0f = p0.reshape(M, 2)
    rq = ref_quad.reshape(4, 1, 2).astype(np.float32)
    quad = np.zeros((N, 4, 2), np.float32)
    good = np.zeros(N, bool)
    last = ref_quad.astype(np.float32)
    for f in range(N):
        a = alive[f]
        if a.sum() >= 4:
            Hh, _ = cv2.findHomography(p0f[a], pos[f][a], cv2.RANSAC, 3.0)
            if Hh is not None:
                quad[f] = cv2.perspectiveTransform(rq, Hh).reshape(4, 2)
                good[f] = True; last = quad[f]; continue
        quad[f] = last                    # hold; smoothing will interpolate
    return quad, good


def main() -> None:
    ap = argparse.ArgumentParser(description="Flicker-free green-screen replacement.")
    ap.add_argument("--video", required=True, help="base UGC clip (green screen)")
    ap.add_argument("--screen", required=True, help="video to play inside the screen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hue", default="35,85", help="green hue lo,hi (OpenCV 0-179)")
    ap.add_argument("--sat-min", type=int, default=35)
    ap.add_argument("--val-min", type=int, default=40)
    ap.add_argument("--pos-window", type=int, default=11,
                    help="zero-phase smoothing window for position (frames; tracks the hand)")
    ap.add_argument("--rot-window", type=int, default=41,
                    help="smoothing window for rotation — large, kills the spurious Z-roll")
    ap.add_argument("--scale-window", type=int, default=41,
                    help="smoothing window for uniform scale — large, kills the size pulsing")
    ap.add_argument("--persp-window", type=int, default=25,
                    help="smoothing window for the perspective/keystone residual")
    ap.add_argument("--track-points", type=int, default=40,
                    help="bezel feature points to seed for KLT tracking (default 40)")
    ap.add_argument("--klt-win", type=int, default=21, help="KLT optical-flow window (px)")
    ap.add_argument("--no-track", action="store_true",
                    help="disable KLT bezel tracking; use per-frame green-corner detection only")
    ap.add_argument("--overfill", type=float, default=1.05,
                    help="warp the card onto a quad expanded by this much so it always "
                         "covers the green; the key clips it back to the screen (default 1.05)")
    ap.add_argument("--feather", type=float, default=1.2, help="key edge feather sigma (px)")
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

    # ── PASS 1: detect the screen quad on every frame; cache grays for KLT ────
    corners_list: list[np.ndarray] = []
    detected_list: list[bool] = []
    aspects: list[float] = []
    grays: list[np.ndarray] = []
    areas: list[int] = []
    prev = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        m = green_mask(frame, hue, args.sat_min, args.val_min)
        areas.append(int(np.count_nonzero(m)))
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
    cap.release()
    N = len(corners_list)
    if N == 0:
        C.die("no frames read from --video")
    if not any(detected_list):
        C.die("no green screen detected — widen --hue/--sat-min/--val-min, --debug")

    corners = np.array(corners_list, dtype=np.float32)
    detected = np.array(detected_list, dtype=bool)
    aspect = float(np.median(aspects))

    # Track the screen via rigid bezel features (KLT) for a temporally-consistent
    # quad, then smooth its decomposed motion. Falls back to green-corner detection
    # if tracking can't be seeded (e.g. featureless bezel).
    quad_in, good_in = corners, detected
    if not args.no_track:
        ref = int(np.argmax(areas))
        cap = cv2.VideoCapture(args.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ref)
        ok, ref_frame = cap.read()
        cap.release()
        if ok:
            ref_mask = green_mask(ref_frame, hue, args.sat_min, args.val_min)
            tracked = track_klt(grays, ref, corners[ref], ref_mask,
                                args.track_points, args.klt_win, 1.0)
            if tracked is not None:
                quad_in, good_in = tracked
    grays.clear()  # free ~hundreds of MB before the render pass

    # rigid-rectangle motion model: stable scale + rotation, no lag
    corners_s = track_similarity(quad_in, good_in, aspect,
                                 args.pos_window, args.rot_window,
                                 args.scale_window, args.persp_window)

    # ── prep the card: anti-aliased down-sample once with INTER_AREA ──────────
    rep = cv2.VideoCapture(args.screen)
    PH = 1280
    PW = max(2, int(round(PH * aspect)))
    card_frames: list[np.ndarray] = []
    while True:
        ok, rf = rep.read()
        if not ok:
            break
        card_frames.append(cv2.resize(cover_crop(rf, aspect), (PW, PH),
                                      interpolation=cv2.INTER_AREA))
    rep.release()
    if not card_frames:
        C.die(f"no frames decoded from {args.screen}")
    card_src = np.array([[0, 0], [PW, 0], [PW, PH], [0, PH]], dtype=np.float32)

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

    # ── PASS 2: corner-pin the card, key the green, composite as layers ───────
    cap = cv2.VideoCapture(args.video)
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        S = corners_s[fi]

        # card warped (anti-aliased) onto a slightly over-filled quad so it always
        # covers the green; the green key below clips it back to the screen
        Hc = cv2.getPerspectiveTransform(card_src, expand(S, args.overfill))
        card = cv2.warpPerspective(card_frames[fi % len(card_frames)], Hc, (W, H),
                                   flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)

        # alpha = per-frame green key, gated to the screen so stray scene green is
        # ignored. Edge sits on the real bezel; fingers (non-green) stay on top.
        gm = green_mask(frame, hue, args.sat_min, args.val_min)
        gate = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(gate, expand(S, 1.2).astype(np.int32), 255)
        gm = cv2.bitwise_and(gm, gate)
        alpha = cv2.GaussianBlur(gm.astype(np.float32) / 255.0, (0, 0), args.feather)
        alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

        region = cv2.dilate(gm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        plate = despill(frame, region, args.despill_bias)
        out = card.astype(np.float32) * alpha + plate.astype(np.float32) * (1.0 - alpha)
        # final safety despill keyed off the green mask: clamp any bright green that
        # survived the feather band (warm card/skin tones are untouched)
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
    C.ok(args.out, label=f"green-screen replaced, stable+flicker-free ({pct}% of {N} tracked)")


if __name__ == "__main__":
    main()
