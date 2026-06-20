#!/usr/bin/env python3
"""Marker-assisted planar tracking + green-screen screen replacement (deterministic CV).

Drop a replacement image/video into a phone/device GREEN SCREEN, tracked per frame
and warped in with a homography, so it sticks to a handheld device and anything in
front of the screen (a finger, a hand) stays untouched.

Best results when the green screen carries small black "+" CROSSHAIR tracking markers
in a regular 2x3 grid (the way VFX artists mark a screen): they are detected every
frame and give a drift-free, sub-pixel, rock-stable track. With no markers it still
works, falling back to the green-screen quad.

Pipeline
--------
1. Per frame: green key -> screen quad; detect the black "+" markers (sub-pixel).
2. Calibrate the screen rectangle in a canonical 2x3 marker-lattice space (median over
   many frames -> immune to single-frame noise; the rectangle is forced axis-aligned to
   the lattice so notch/rounded-corner quad noise can't inject a constant tilt).
3. Per frame: match markers to the lattice, fit a homography lattice->image and map the
   screen rect through it. When the matched markers span the screen they define it alone;
   a partial set is FUSED with the green quad (quad pins the extent, markers refine the
   interior -> no extrapolation shear). Fall back to the green quad when <4 markers match.
4. Stabilise: decompose the 4 corners into position / rotation / uniform-scale (Procrustes)
   + perspective residual and zero-phase smooth each channel (scale & rotation hardest).
5. Composite: warp the replacement onto the screen (Lanczos) and key the REAL green out on
   top (tracking-independent matte) so the bezel and fingers occlude the insert. Where the
   marker track is low-confidence, the screen is keyed to BLACK (no risk of a misaligned
   insert), crossfading in/out. Despill + feather the edge. Mux the original audio.

Usage
-----
  python3 green_screen.py --video clip.mp4 --image app-ui.png --out out.mp4
  python3 green_screen.py --video clip.mp4 --screen app-demo.mp4 --out out.mp4
  # inspect tracking (green=full-marker, yellow=partial, orange=quad-fallback, red=hold):
  python3 green_screen.py --video clip.mp4 --image app.png --out out.mp4 --debug dbg.mp4
  # show the (possibly misaligned) insert everywhere instead of blacking low-conf frames:
  python3 green_screen.py --video clip.mp4 --image app.png --out out.mp4 --low-conf ui
"""
import argparse
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


# ============================ detection ============================
def green_mask(bgr, hue=(35, 90), smin=40, vmin=40):
    """Denoised green key (median on chroma so block noise doesn't punch holes)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 1] = cv2.medianBlur(ycc[:, :, 1], 3)
    ycc[:, :, 2] = cv2.medianBlur(ycc[:, :, 2], 3)
    hsv = cv2.cvtColor(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (hue[0], smin, vmin), (hue[1], 255, 255))


def fill_small_holes(mask, max_area):
    """Fill interior holes smaller than max_area (marker crosses, speckle) while
    KEEPING large interior holes (fingers crossing the screen) open for occlusion."""
    inv = cv2.bitwise_not(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 8)
    out = mask.copy(); H, W = mask.shape
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if not (x == 0 or y == 0 or x + w == W or y + h == H) and a < max_area:
            out[lab == i] = 255
    return out


def screen_matte(bgr, hue, smin, vmin):
    """COLOR-DIFFERENCE key (the core of Keylight): the key signal is
    d = G - max(R, B) — large on green, ~0 on the neutral/dark bezel — and the alpha is a
    clipped linear ramp of d. Because d ~ 0 on the bezel, the soft edge sits EXACTLY on the
    green->bezel transition (no creep into the black frame, no ragged binary stair-step).
    Returns a continuous [0,1] matte. Interior dark marks (the "+" markers, the notch) are
    filled to solid; large foreground occluders (fingers) stay transparent for occlusion.
    A garbage matte (the largest green blob) ignores stray scene green."""
    b, g, r = cv2.split(bgr.astype(np.float32))
    d = g - np.maximum(r, b)                              # color difference (greenness)
    H, W = bgr.shape[:2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats((d > 20).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((H, W), np.float32)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 0.0015 * W * H:
        return np.zeros((H, W), np.float32)
    screen = (lab == idx).astype(np.uint8) * 255
    area = float(stats[idx, cv2.CC_STAT_AREA])
    inner = cv2.erode(screen, np.ones((11, 11), np.uint8)) > 0
    # Clip-Black / Clip-White as a fraction of the solid-green level: the band [cb,cw] sits
    # in the green->bezel transition (NOT inside the green distribution, which would map the
    # green's own noise to a speckled alpha). hi = solid-green key level.
    hi = float(np.percentile(d[inner], 50)) if int(np.count_nonzero(inner)) > 50 else float(d.max())
    cb, cw = 0.10 * hi, 0.55 * hi
    alpha = np.clip((d - cb) / max(1.0, cw - cb), 0.0, 1.0)
    alpha[~(cv2.dilate(screen, np.ones((7, 7), np.uint8)) > 0)] = 0.0     # garbage matte
    # fill interior dark marks (markers + notch) incl. their anti-aliased halo: small enclosed
    # holes in the solid-green core -> set to 1. Large holes (fingers) stay open -> occlusion.
    solid = (alpha > 0.85).astype(np.uint8) * 255
    filled = fill_small_holes(solid, max_area=max(3000.0, 0.06 * area))
    fillmask = cv2.dilate(((filled > 0) & (solid == 0)).astype(np.uint8), np.ones((3, 3), np.uint8))
    alpha[fillmask > 0] = 1.0
    # sub-pixel CHOKE: raise the ramp floor to bury the outermost green-contaminated ring,
    # sub-pixel and soft (a morphological erode would move the edge by whole px and re-ragged it).
    cval = 0.08
    alpha = np.clip((alpha - cval) / (1.0 - cval), 0.0, 1.0)
    return np.clip(cv2.GaussianBlur(alpha, (0, 0), 0.6), 0.0, 1.0)


def order_quad(pts):
    # Corners in a consistent CLOCKWISE cyclic order, started at the image-top-left-most vertex.
    # A pure angular sort about the centroid is robust to ANY in-plane rotation, unlike the
    # classic sum/diff corner assignment, which collapses two corners onto one role once the
    # quad rolls past ~45deg (duplicate corners -> degenerate homography -> garbage track ->
    # frame keyed black). PASS 1 then LOCKS corner identity across frames temporally, so a
    # heavily tilted / rotating phone keeps a stable TL,TR,BR,BL labelling.
    pts = pts.reshape(4, 2).astype(np.float32)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]            # cyclic order
    area = float(sum(pts[k, 0] * pts[(k + 1) % 4, 1] - pts[(k + 1) % 4, 0] * pts[k, 1] for k in range(4)))
    if area < 0:                                          # enforce CW winding (image coords, y down)
        pts = pts[::-1]
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0).astype(np.float32)      # start near TL


def screen_quad_and_filled(mask):
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    filled = np.zeros_like(m); cv2.drawContours(filled, [c], -1, 255, -1)
    peri = cv2.arcLength(c, True); quad = None
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            quad = ap.reshape(4, 2).astype(np.float32); break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    return order_quad(quad), filled


def detect_markers(gray, filled, shape_validate=True, subpix="cornersubpix"):
    ys, xs = np.where(filled > 0)
    if len(xs) < 50:
        return np.zeros((0, 2), np.float32)
    sw, sh = xs.max() - xs.min(), ys.max() - ys.min()
    inner = cv2.erode(filled, np.ones((15, 15), np.uint8)) > 0
    if int(np.count_nonzero(inner)) < 200:
        return np.zeros((0, 2), np.float32)
    # contrast-INVARIANT dark-spot detection: black-hat finds dark marks smaller than the
    # kernel regardless of absolute darkness — generated markers are often low-contrast grey,
    # which an absolute threshold misses (markers detected late / not at all).
    k = int(np.clip(round(0.13 * min(sw, sh)), 9, 61)); k += 1 - k % 2
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    vals = bh[inner]
    if vals.max() < 10:
        return np.zeros((0, 2), np.float32)
    t = max(10.0, 0.35 * float(vals.max()))
    dark = ((bh >= t) & inner).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    area = int(np.count_nonzero(inner)); pts = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]; w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 6 or a > area * 0.02:
            continue
        if max(w, h) > 0.2 * gray.shape[0]:
            continue
        if shape_validate:
            # a "+" is sparse in its bbox; reject solid blobs (fingertips/notch -> high
            # extent) and streaks (extreme aspect). Rotation-invariant.
            extent = a / float(max(1, w * h)); ar = w / float(max(1, h))
            if extent > 0.85 or extent < 0.06 or ar < 0.30 or ar > 3.3:
                continue
        pts.append([float(cent[i][0]), float(cent[i][1])])
    if not pts:
        return np.zeros((0, 2), np.float32)
    pts = np.array(pts, np.float32)
    if subpix != "none":
        # Lock onto the "+" CENTRE, not an arm corner. cornerSubPix latches onto one of the
        # plus's ~12 corners (which flips frame-to-frame -> the marker jumps side to side).
        # Instead use a black-hat-weighted centroid in a window: the cross is symmetric, so its
        # intensity-weighted centre of mass IS the centre, regardless of rotation, and it uses
        # ALL the cross pixels (not one corner) -> stable, sub-pixel.
        H0, W0 = gray.shape; win = max(5, int(round(0.5 * 0.13 * min(sw, sh)))) | 1
        gx, gy = np.meshgrid(np.arange(-win, win + 1), np.arange(-win, win + 1))
        out = pts.copy()
        for j in range(len(pts)):
            for _ in range(2):                       # 2 mean-shift iterations to the centre
                xi, yi = int(round(out[j, 0])), int(round(out[j, 1]))
                if xi - win < 0 or yi - win < 0 or xi + win >= W0 or yi + win >= H0:
                    break
                wpatch = bh[yi - win:yi + win + 1, xi - win:xi + win + 1].astype(np.float32)
                wpatch = np.maximum(wpatch - 0.25 * wpatch.max(), 0.0)   # keep only the dark cross
                s = wpatch.sum()
                if s < 1:
                    break
                out[j] = [xi + (gx * wpatch).sum() / s, yi + (gy * wpatch).sum() / s]
        pts = out
    return pts.reshape(-1, 2)


# ============================ marker grid ============================
# The markers form a regular nx-by-ny grid (e.g. 2x3 portrait, 3x2 landscape, 2x2).
# Nothing is hardcoded — the grid dims AND the canonical marker positions (in the
# screen's unit square) are auto-detected per clip in main(), so any orientation /
# inset / marker count works. The screen rectangle IS the unit square [0,1]x[0,1].
UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)   # screen corners TL,TR,BR,BL


def match_lattice(det, predH, diag, canon, thr_frac=0.12):
    """Greedy-match canonical markers (projected by predH) to detections.
    Returns (indices into canon, matched canon pts, matched detection pts)."""
    proj = cv2.perspectiveTransform(canon.reshape(-1, 1, 2), predH).reshape(-1, 2)
    thr = thr_frac * diag
    idxs, dst, used = [], [], set()
    for gi in np.argsort([np.min(np.linalg.norm(det - p, axis=1)) for p in proj]):
        dd = np.linalg.norm(det - proj[gi], axis=1); j = int(np.argmin(dd))
        if dd[j] < thr and j not in used:
            used.add(j); idxs.append(int(gi)); dst.append(det[j])
    idxs = np.array(idxs, int)
    return (idxs,
            canon[idxs] if len(idxs) else np.zeros((0, 2), np.float32),
            np.array(dst, np.float32) if dst else np.zeros((0, 2), np.float32))


# ============================ stabilisation ============================
def _smooth(arr, good, window):
    """Zero-phase smooth an (N,D) signal: interpolate gaps, then Savitzky-Golay (scipy)
    or centred-Gaussian fallback — both symmetric in time, no lag."""
    N, D = arr.shape
    out = arr.astype(np.float64).copy(); idx = np.arange(N); g = good.astype(bool)
    if g.sum() < 2:
        return out
    for c in range(D):
        out[~g, c] = np.interp(idx[~g], idx[g], out[g, c])
    win = min(window if window % 2 else window + 1, N if N % 2 else N - 1); win = max(win, 3)
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(out, win, 2, axis=0, mode="interp")
    except Exception:
        r = win // 2; x = np.arange(-r, r + 1)
        k = np.exp(-(x ** 2) / (2 * (r / 2.0 + 1e-6) ** 2)); k /= k.sum()
        sm = np.empty_like(out)
        for c in range(D):
            sm[:, c] = np.convolve(np.pad(out[:, c], r, mode="edge"), k, "valid")
        return sm


def stabilise(corners, good, aspect, pos_w, rot_w, scale_w, persp_w):
    """Decompose each frame's quad into pos / rot / uniform-scale (Procrustes) + a
    perspective residual, smooth each channel on its own window, recompose. Scale &
    rotation smoothed hardest -> kills size-pulsing and Z-roll."""
    N = len(corners)
    Q0 = np.array([[-aspect / 2, -0.5], [aspect / 2, -0.5],
                   [aspect / 2, 0.5], [-aspect / 2, 0.5]], np.float64)
    q0ss = (Q0 ** 2).sum() / 4.0
    s = np.zeros(N); th = np.zeros(N); t = np.zeros((N, 2)); res = np.zeros((N, 8))
    for i in range(N):
        q = corners[i].astype(np.float64); mu = q.mean(0)
        U, Dg, Vt = np.linalg.svd(((q - mu).T @ Q0) / 4.0); R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        s[i] = max(1e-6, Dg.sum() / q0ss); th[i] = np.arctan2(R[1, 0], R[0, 0]); t[i] = mu
        res[i] = (q - ((s[i] * (R @ Q0.T)).T + mu)).reshape(8)
    th = np.unwrap(th); g = good.astype(bool)
    s_s = np.exp(_smooth(np.log(s).reshape(-1, 1), g, scale_w)).ravel()
    th_s = _smooth(th.reshape(-1, 1), g, rot_w).ravel()
    t_s = _smooth(t, g, pos_w)
    res_s = _smooth(res, g, persp_w).reshape(N, 4, 2)
    out = np.zeros((N, 4, 2), np.float32)
    for i in range(N):
        c, si = np.cos(th_s[i]), np.sin(th_s[i]); R = np.array([[c, -si], [si, c]])
        out[i] = (s_s[i] * (R @ Q0.T)).T + t_s[i] + res_s[i]
    return out


# ============================ compositing helpers ============================
def expand(quad, f):
    ctr = quad.mean(0); return (ctr + (quad - ctr) * f).astype(np.float32)


def cover_crop(src, aspect):
    h, w = src.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2; return src[:, x0:x0 + nw]
    nh = int(round(w / aspect)); y0 = (h - nh) // 2; return src[y0:y0 + nh, :]


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser(description="Marker-tracked green-screen replacement.")
    ap.add_argument("--video", required=True, help="base clip with a green device screen")
    ap.add_argument("--image", help="replacement still image")
    ap.add_argument("--screen", help="replacement video (looped) — alternative to --image")
    ap.add_argument("--screen-volume", type=float, default=0.2,
                    help="volume of the --screen video's own audio mixed under the original "
                         "(0 = mute; default 0.2). Original clip audio stays at full volume.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hue", default="35,90", help="green hue lo,hi (OpenCV 0-179)")
    ap.add_argument("--sat-min", type=int, default=40)
    ap.add_argument("--val-min", type=int, default=40)
    # zero-phase (offline, symmetric) smoothing -> no lag. Scale/rotation/keystone are
    # near-constant for a handheld phone, so high-freq variation there is detector noise:
    # smooth them HARD. Position tracks the hand, so keep it lighter.
    ap.add_argument("--pos-window", type=int, default=9, help="zero-phase smoothing window: position")
    ap.add_argument("--rot-window", type=int, default=31, help="smoothing window: rotation (smooth hard)")
    ap.add_argument("--scale-window", type=int, default=41, help="smoothing window: uniform scale (smooth hard)")
    ap.add_argument("--persp-window", type=int, default=25, help="smoothing window: perspective/keystone (smooth hard)")
    ap.add_argument("--overfill", type=float, default=1.04, help="warp the insert onto a quad this much "
                    "bigger so it always covers the green; the key clips it back")
    ap.add_argument("--feather", type=float, default=0.6, help="extra key-edge feather sigma "
                    "(px; the matte is already anti-aliased — raise only if you want softer)")
    ap.add_argument("--despill-bias", type=float, default=0.5)
    ap.add_argument("--low-conf", choices=["black", "ui"], default="black",
                    help="where the marker track is low-confidence: 'black' keys the green to black "
                         "(default — never shows a misaligned insert); 'ui' composites anyway")
    ap.add_argument("--low-conf-fade", type=float, default=5.0,
                    help="frames to crossfade insert<->black at confidence boundaries")
    ap.add_argument("--no-shape-validate", dest="shape_validate", action="store_false",
                    help="keep all dark blobs as markers (don't reject by cross shape)")
    ap.add_argument("--subpix", choices=["center", "none"], default="center",
                    help="sub-pixel marker centre refinement (black-hat-weighted centroid)")
    ap.add_argument("--debug", help="optional overlay diagnostic clip (tracked quad + markers)")
    ap.set_defaults(shape_validate=True)
    args = ap.parse_args()

    C.need_ffmpeg()
    if not args.image and not args.screen:
        C.die("need --image (still) or --screen (video)")
    for p in (args.video, args.image, args.screen):
        if p and not os.path.exists(p):
            C.die(f"file not found: {p}")
    try:
        hue = tuple(int(x) for x in args.hue.split(","))
        assert len(hue) == 2
    except (ValueError, AssertionError):
        C.die("--hue must be 'lo,hi', e.g. 35,90")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        C.die(f"cannot open {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or C.DEFAULT_FPS

    # ---- PASS 1: detect green + markers per frame ----
    grays, quads, marks, ok_s = [], [], [], []
    prev_oq = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); grays.append(g)
        m = green_mask(fr, hue, args.sat_min, args.val_min)
        if int(np.count_nonzero(m)) < 4000:
            quads.append(None); marks.append(np.zeros((0, 2), np.float32)); ok_s.append(False); continue
        q, filled = screen_quad_and_filled(m)
        if q is not None and prev_oq is not None:
            # LOCK corner identity frame-to-frame: pick the cyclic roll of this (consistently
            # wound) quad that best matches the previous frame's corners. Tracks TL,TR,BR,BL
            # smoothly through large rotation, where a purely per-frame geometric label jumps
            # between physical corners and breaks the lattice match -> quad-fallback -> black.
            k = min(range(4), key=lambda r: float(np.sum(np.linalg.norm(np.roll(q, -r, 0) - prev_oq, axis=1))))
            q = np.roll(q, -k, 0)
        if q is not None:
            prev_oq = q
        quads.append(q); ok_s.append(q is not None)
        marks.append(detect_markers(g, filled, args.shape_validate, args.subpix)
                     if filled is not None else np.zeros((0, 2), np.float32))
    cap.release(); N = len(grays)
    if N == 0:
        C.die("no frames read from --video")
    if not any(ok_s):
        C.die("no green screen detected — widen --hue/--sat-min/--val-min, use --debug")

    # ---- auto-detect the marker grid (nx x ny), any orientation / inset / count ----
    cand = [i for i in range(N) if ok_s[i] and quads[i] is not None and len(marks[i]) >= 4]
    if not cand:
        C.die("no tracking markers found in the green screen (need black '+' marks); "
              "this tool tracks a marker-grid screen — check --hue or add markers.")

    def perspectiveness(q):
        # 0 = a perfect rectangle (frontal); higher = more perspective distortion.
        d1 = np.linalg.norm(q[0] - q[2]); d2 = np.linalg.norm(q[1] - q[3])
        top = np.linalg.norm(q[1] - q[0]); bot = np.linalg.norm(q[2] - q[3])
        lft = np.linalg.norm(q[3] - q[0]); rgt = np.linalg.norm(q[2] - q[1])
        return (abs(d1 - d2) / (d1 + d2 + 1e-6) + abs(top - bot) / (top + bot + 1e-6)
                + abs(lft - rgt) / (lft + rgt + 1e-6))

    # Screen orientation (median over green frames) + the full marker count M, taken as the
    # most common count with real support so false-positive frames (extra blobs) don't win.
    asp = []
    for i in cand:
        q = quads[i]
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if h > 1:
            asp.append(w / h)
    landscape = bool(asp) and float(np.median(asp)) > 1.0
    from collections import Counter
    cnt = Counter(len(marks[i]) for i in cand)
    # full grid size = the MOST COMMON marker count (mode), robust to occasional
    # false-positive frames with extra blobs. ties -> larger count.
    counts = [(v, k) for k, v in cnt.items() if 4 <= k <= 12]
    M = max(counts)[1] if counts else max(len(marks[i]) for i in cand)
    # grid dims from M + orientation (markers run 2 along the short side, 3 along the long)
    DIMS = {4: (2, 2), 6: (3, 2) if landscape else (2, 3), 8: (4, 2) if landscape else (2, 4),
            9: (3, 3)}
    # GENERATED footage routinely yields a spurious/merged blob or two (the notch, glare, a
    # caption edge), so the MODAL detected count is often 1-2 above the true grid — e.g. a real
    # 3x3=9 screen reads as a stable 10. Falling through to the wrong default (2x3) then forces a
    # 3x3 lattice into a 2x3 model, breaking full-span on most frames -> quad-fallback -> black.
    # Snap M to the NEAREST known grid size (ties -> larger) instead.
    if M not in DIMS:
        M = min(DIMS, key=lambda k: (abs(k - M), -k))
    nx, ny = DIMS[M]
    G_IDEAL = np.array([[c, r] for r in range(ny) for c in range(nx)], np.float32)

    # Calibration prior bootstraps the lattice->image match. Two sources, and we keep
    # whichever calibrates the MOST frames: (a) a generic inset (works when markers sit
    # near the screen edges — c1g/c3g/g2), (b) priors derived from frontal full-grid
    # frames by SORTING markers into the grid (needed when markers are inset oddly, e.g.
    # g1's central cluster). Trying several frames + validating beats trusting one.
    sx = 0.76 / max(1, nx - 1) if nx > 1 else 1.0
    sy = 0.76 / max(1, ny - 1) if ny > 1 else 1.0
    generic = np.array([[-0.12 / sx, -0.12 / sy], [(nx - 1) + 0.12 / sx, -0.12 / sy],
                        [(nx - 1) + 0.12 / sx, (ny - 1) + 0.12 / sy], [-0.12 / sx, (ny - 1) + 0.12 / sy]], np.float32)

    def prior_from_ref(i):
        qr = quads[i]; mr = marks[i]
        e = qr[1] - qr[0]; ang = np.arctan2(e[1], e[0]); co, si = np.cos(-ang), np.sin(-ang)
        mrr = (mr - mr.mean(0)) @ np.array([[co, -si], [si, co]]).T   # de-rotate to screen axes
        order = np.argsort(mrr[:, 1]); lab = np.zeros((len(mr), 2), np.float32)
        for r in range(ny):
            ri = order[r * nx:(r + 1) * nx]
            for c, idx in enumerate(ri[np.argsort(mrr[ri, 0])]):
                lab[idx] = [c, r]
        Hl, _ = cv2.findHomography(lab, mr, 0)
        if Hl is None or not np.all(np.isfinite(Hl)):
            return None
        pr = cv2.perspectiveTransform(qr.reshape(-1, 1, 2), np.linalg.inv(Hl)).reshape(-1, 2)
        if not np.all(np.isfinite(pr)) or np.any(np.abs(pr) > nx + ny + 6):
            return None
        pr = order_quad(pr)
        return pr if cv2.contourArea(pr) > 0.2 else None

    def collect(prior):
        out = []
        for i in cand:
            Hq, _ = cv2.findHomography(prior, quads[i], 0)
            if Hq is None:
                continue
            diag = np.linalg.norm(quads[i][0] - quads[i][2])
            _, src, dst = match_lattice(marks[i], Hq, diag, G_IDEAL)
            if len(src) < 4:
                continue
            Hg, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
            if Hg is None:
                continue
            rg = cv2.perspectiveTransform(quads[i].reshape(-1, 1, 2), np.linalg.inv(Hg)).reshape(-1, 2)
            if not np.any(np.abs(rg) > max(nx, ny) + 4):
                out.append(rg)
        return out

    full = [i for i in cand if len(marks[i]) == nx * ny]
    candidates = [generic]
    for i in sorted(full, key=lambda i: perspectiveness(quads[i]))[:6]:
        p = prior_from_ref(i)
        if p is not None:
            candidates.append(p)
    rects = []
    for p in candidates:
        r = collect(p)
        if len(r) > len(rects):
            rects = r
    if not rects:
        C.die(f"detected a {nx}x{ny} marker grid but could not calibrate it — check --debug.")
    rect_g = np.median(np.array(rects), axis=0).astype(np.float32)
    r2 = collect(rect_g)
    if len(r2) >= len(rects):
        rect_g = np.median(np.array(r2), axis=0).astype(np.float32)
    # markers are parallel to the screen edges -> axis-align the rect (drop quad skew that
    # would inject a constant tilt).
    xL = float((rect_g[0, 0] + rect_g[3, 0]) / 2); xR = float((rect_g[1, 0] + rect_g[2, 0]) / 2)
    yT = float((rect_g[0, 1] + rect_g[1, 1]) / 2); yB = float((rect_g[2, 1] + rect_g[3, 1]) / 2)
    rect_g = np.array([[xL, yT], [xR, yT], [xR, yB], [xL, yB]], np.float32)

    # ---- PASS 2: per-frame homography (markers fused with the green quad) ----
    # markers map the lattice -> image; the quad pins the extent when the matched markers
    # don't span the whole grid (avoids extrapolation shear).
    MK_W = 3
    corners = np.zeros((N, 4, 2), np.float32)
    good = np.zeros(N, bool); source = ["none"] * N
    prevH = None; last_corn = None
    for i in range(N):
        Hq = None
        if ok_s[i] and quads[i] is not None:
            Hq, _ = cv2.findHomography(rect_g, quads[i], 0)
        diag = np.linalg.norm(quads[i][0] - quads[i][2]) if (quads[i] is not None) else 200.0
        msrc = mdst = None; full_span = False
        if ok_s[i] and len(marks[i]) >= 4:
            predH = Hq if Hq is not None else prevH
            if predH is not None:
                idxs, s0, d0 = match_lattice(marks[i], predH, diag, G_IDEAL, thr_frac=0.10)
                if len(s0) >= 4:
                    _, mask = cv2.findHomography(s0, d0, cv2.RANSAC, 0.03 * diag)
                    if mask is not None and mask.ravel().sum() >= 4:
                        keep = mask.ravel().astype(bool)
                        msrc, mdst = s0[keep], d0[keep]
                        cset = {int(round(c)) for c in msrc[:, 0]}; rset = {int(round(r)) for r in msrc[:, 1]}
                        full_span = (0 in cset and nx - 1 in cset and 0 in rset and ny - 1 in rset)
        # do the matched markers SPAN the screen? a sparse/clustered grid (e.g. 4 markers
        # bunched in the centre) must NOT define the extent on its own — extrapolating the
        # screen rect from a small cluster amplifies sub-pixel noise into huge jitter.
        span_ok = False
        if msrc is not None and full_span:
            span_ok = float(np.linalg.norm(mdst.max(0) - mdst.min(0))) > 0.55 * diag
        Hcur = None
        if msrc is not None and full_span and span_ok:
            Hf, _ = cv2.findHomography(msrc, mdst, 0)
            if Hf is not None and Hq is not None:
                cn = cv2.perspectiveTransform(rect_g.reshape(-1, 1, 2), Hf).reshape(4, 2)
                if np.max(np.linalg.norm(cn - quads[i], axis=1)) > 0.45 * diag:
                    Hf = None
            if Hf is not None:
                Hcur = Hf; source[i] = "marker"; good[i] = True
        if Hcur is None and msrc is not None and Hq is not None:
            # fuse: the quad pins the extent (no extrapolation), markers refine the interior.
            # markers weighted less when clustered (they're a noisy local patch then).
            mw = MK_W if span_ok else 1
            Hf, _ = cv2.findHomography(np.vstack([np.repeat(msrc, mw, 0), rect_g]),
                                       np.vstack([np.repeat(mdst, mw, 0), quads[i]]), 0)
            if Hf is not None:
                cn = cv2.perspectiveTransform(rect_g.reshape(-1, 1, 2), Hf).reshape(4, 2)
                if np.max(np.linalg.norm(cn - quads[i], axis=1)) < 0.30 * diag:
                    Hcur = Hf; source[i] = "marker"; good[i] = True   # confident quad+marker fusion
                else:
                    Hcur = Hq; source[i] = "quad"
            else:
                Hcur = Hq; source[i] = "quad"
        if Hcur is None and msrc is not None and Hq is None:
            Hf, _ = cv2.findHomography(msrc, mdst, 0)
            if Hf is not None:
                Hcur = Hf; source[i] = "marker"; good[i] = full_span
        if Hcur is None and Hq is not None:
            Hcur = Hq; source[i] = "quad"
        if Hcur is None:
            corners[i] = last_corn if last_corn is not None else 0; source[i] = "hold"; continue
        prevH = Hcur
        cn = cv2.perspectiveTransform(rect_g.reshape(-1, 1, 2), Hcur).reshape(4, 2)
        corners[i] = cn; last_corn = cn
    first = next((i for i in range(N) if source[i] != "hold"), None)
    if first:
        for i in range(first):
            corners[i] = corners[first]

    def quad_aspect(q):
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        return float(w / h) if h > 1 else 0.46
    gidx = [i for i in range(N) if good[i]]
    aspect = float(np.median([quad_aspect(corners[i]) for i in gidx])) if gidx else 0.46

    # ---- stabilise ----
    corners_s = stabilise(corners, good, aspect,
                          args.pos_window, args.rot_window, args.scale_window, args.persp_window)
    if os.environ.get("DUMP_CORNERS"):
        np.savez(args.out + ".corners.npz", corners=corners, corners_s=corners_s,
                 good=good, source=np.array(source), nx=nx, ny=ny)

    # ---- low-confidence -> black (time-eased confidence weight) ----
    def gauss(x, sigma):
        r = int(max(1, round(sigma * 3))); k = np.exp(-(np.arange(-r, r + 1) ** 2) / (2 * sigma ** 2)); k /= k.sum()
        return np.convolve(np.pad(x, r, mode="edge"), k, "valid")
    conf = good.astype(np.float64)
    if args.low_conf == "black" and args.low_conf_fade > 0:
        w = np.clip(gauss(conf, max(0.5, args.low_conf_fade / 2.0)), 0.0, 1.0)
    elif args.low_conf == "black":
        w = conf
    else:
        w = np.ones(N)
    n_black = int(np.sum(w < 0.5))

    # ---- prep replacement (anti-aliased downsample once) ----
    PH = 1280; PW = max(2, int(round(PH * aspect)))
    cards = []
    if args.image:
        im = cv2.imread(args.image)
        if im is None:
            C.die(f"cannot read {args.image}")
        cards = [cv2.resize(cover_crop(im, aspect), (PW, PH), interpolation=cv2.INTER_AREA)]
    else:
        rc = cv2.VideoCapture(args.screen)
        while True:
            ok, rf = rc.read()
            if not ok:
                break
            cards.append(cv2.resize(cover_crop(rf, aspect), (PW, PH), interpolation=cv2.INTER_AREA))
        rc.release()
        if not cards:
            C.die(f"no frames decoded from {args.screen}")
    card_src = np.array([[0, 0], [PW, 0], [PW, PH], [0, PH]], np.float32)

    # ---- PASS 3: composite ----
    tmp_fd, tmp = tempfile.mkstemp(suffix=".mp4"); os.close(tmp_fd)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not vw.isOpened():
        C.die("cv2.VideoWriter failed to open")
    dbg = None
    if args.debug:
        C.ensure_parent(args.debug)
        dbg = cv2.VideoWriter(args.debug + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap = cv2.VideoCapture(args.video); fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        S = corners_s[fi]
        Hc = cv2.getPerspectiveTransform(card_src, expand(S, args.overfill))
        card = cv2.warpPerspective(cards[fi % len(cards)], Hc, (W, H),
                                   flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        a = screen_matte(frame, hue, args.sat_min, args.val_min)   # continuous, anti-aliased
        if args.feather > 0:
            a = np.clip(cv2.GaussianBlur(a, (0, 0), args.feather), 0, 1)
        gm = (a > 0.5).astype(np.uint8) * 255
        # DESPILL the PLATE (not the output): matte-controlled, strongest on the soft edge,
        # with luminance restore -> kills the green fringe without desaturating the inserted UI.
        fp = frame.astype(np.float32); pb, pg, pr = fp[:, :, 0], fp[:, :, 1], fp[:, :, 2]
        ref = args.despill_bias * pr + (1 - args.despill_bias) * pb
        edge_w = 4.0 * a * (1.0 - a)                                 # peaks on the soft edge band
        gnew = pg - edge_w * np.maximum(0.0, pg - ref)
        loss = pg - gnew
        region = (cv2.dilate(gm, np.ones((5, 5), np.uint8)) > 0)[:, :, None]
        plate = np.where(region, np.stack([pb + 0.5 * loss, gnew, pr + 0.5 * loss], -1), fp)
        content = card.astype(np.float32) * float(w[fi])     # fade to black where low-confidence
        alpha = a[:, :, None]
        out = content * alpha + plate * (1 - alpha)
        vw.write(np.clip(out, 0, 255).astype(np.uint8))
        if dbg is not None:
            d = frame.copy()
            col = {"marker": (255, 0, 255), "partial": (0, 255, 255), "quad": (0, 165, 255),
                   "hold": (0, 0, 255), "none": (128, 128, 128)}[source[fi]]   # magenta=marker (visible on green)
            cv2.polylines(d, [S.astype(np.int32)], True, col, 3)
            for p in marks[fi]:
                cv2.drawMarker(d, (int(p[0]), int(p[1])), (255, 0, 0), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(d, f"f{fi} {source[fi]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            dbg.write(d)
        fi += 1
    cap.release(); vw.release()
    if dbg is not None:
        dbg.release()

    C.ensure_parent(args.out)
    # audio: original clip at full volume; if the inserted --screen video has audio, mix it
    # in at --screen-volume (default low, so it sits under the creator's voice).
    v_audio = C.probe(args.video).has_audio
    s_audio = bool(args.screen) and args.screen_volume > 0 and C.probe(args.screen).has_audio
    if v_audio and s_audio:
        C.run(["ffmpeg", "-y", "-i", tmp, "-i", args.video, "-i", args.screen, "-filter_complex",
               f"[2:a]volume={args.screen_volume}[sa];[1:a][sa]amix=inputs=2:duration=first:normalize=0[aout]",
               "-map", "0:v", "-map", "[aout]", "-shortest", *C.V_CODEC, *C.A_CODEC, args.out])
    elif s_audio:                                     # original has no audio -> just the screen audio
        C.run(["ffmpeg", "-y", "-i", tmp, "-i", args.screen, "-filter_complex",
               f"[1:a]volume={args.screen_volume}[aout]", "-map", "0:v", "-map", "[aout]",
               "-shortest", *C.V_CODEC, *C.A_CODEC, args.out])
    else:
        C.run(["ffmpeg", "-y", "-i", tmp, "-i", args.video, "-map", "0:v", "-map", "1:a?",
               "-shortest", *C.V_CODEC, *C.A_CODEC, args.out])
    os.unlink(tmp)
    if dbg is not None:
        C.run(["ffmpeg", "-y", "-i", args.debug + ".tmp.mp4", *C.V_CODEC, args.debug])
        os.unlink(args.debug + ".tmp.mp4")

    mk = sum(1 for s in source if s == "marker")
    C.ok(args.out, label=f"screen replaced — {mk}/{N} marker-tracked, "
         f"{n_black} low-confidence frames keyed to black")


if __name__ == "__main__":
    main()
