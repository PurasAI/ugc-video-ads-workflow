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
   + perspective residual and zero-phase smooth each channel; optional One-Euro pre-filter.
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


def key_green_mask(bgr, hue, smin, vmin):
    """Inclusive green key for COMPOSITING (wider than the tracking mask so dark/
    desaturated screen-green at the edges is still caught and keyed out)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 1] = cv2.medianBlur(ycc[:, :, 1], 3)
    ycc[:, :, 2] = cv2.medianBlur(ycc[:, :, 2], 3)
    den = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    hsv = cv2.cvtColor(den, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (max(0, hue[0] - 3), max(0, smin - 15), max(0, vmin - 15)),
                    (min(179, hue[1] + 5), 255, 255))
    lab = cv2.cvtColor(den, cv2.COLOR_BGR2LAB)
    b, g, r = cv2.split(den.astype(np.int16))
    greenish = ((g - np.maximum(r, b)) > 12).astype(np.uint8) * 255
    la = cv2.inRange(lab[:, :, 1], 0, 125)     # LAB a* low = green, robust to brightness
    return cv2.bitwise_or(m, cv2.bitwise_and(la, greenish))


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
    """Tracking-INDEPENDENT screen alpha: the largest green blob (the device screen),
    with marker holes filled and finger occlusions preserved. Returns a uint8 mask."""
    gm = key_green_mask(bgr, hue, smin, vmin)
    gm = cv2.morphologyEx(gm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(gm, 8)
    H, W = bgr.shape[:2]
    if n <= 1:
        return np.zeros((H, W), np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 0.0015 * W * H:
        return np.zeros((H, W), np.uint8)
    screen = (lab == idx).astype(np.uint8) * 255
    area = float(stats[idx, cv2.CC_STAT_AREA])
    screen = cv2.morphologyEx(screen, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return fill_small_holes(screen, max_area=max(1500.0, 0.03 * area))


def greenness(bgr):
    b, g, r = cv2.split(bgr.astype(np.int16))
    return (g - np.maximum(r, b)).astype(np.float32)   # large+ on pure green


def screen_matte_global(bgr, cb, cw):
    """Continuous chroma matte with GLOBALLY-FIXED clip levels (computed once over the
    whole clip) -> identical mapping every frame. Experimental (off by default)."""
    d = greenness(bgr)
    soft = np.clip((d - cb) / max(1.0, (cw - cb)), 0.0, 1.0)
    binary = cv2.morphologyEx((d > cb).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    H, W = bgr.shape[:2]
    if n <= 1:
        return np.zeros((H, W), np.float32)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < 0.0015 * W * H:
        return np.zeros((H, W), np.float32)
    screen = (lab == idx).astype(np.uint8) * 255
    area = float(stats[idx, cv2.CC_STAT_AREA])
    screen = cv2.morphologyEx(screen, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    screen = fill_small_holes(screen, max_area=max(1500.0, 0.03 * area))
    a = soft * (screen.astype(np.float32) / 255.0)
    a[(screen > 0) & (soft < 1.0)] = 1.0
    return a


def order_quad(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1); d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], np.float32)  # TL TR BR BL


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


def saddle_refine(gray, seeds, win=4):
    """Sub-pixel via analytic saddle fit: a "+" centre is an intensity saddle. Fit
    z = ax^2+bxy+cy^2+dx+ey+f on a window and solve for the critical point — more
    robust to motion blur than cornerSubPix."""
    g = gray.astype(np.float64); H, W = gray.shape
    xs, ys = np.meshgrid(np.arange(-win, win + 1), np.arange(-win, win + 1))
    xr, yr = xs.ravel().astype(float), ys.ravel().astype(float)
    Ap = np.linalg.pinv(np.column_stack([xr * xr, xr * yr, yr * yr, xr, yr, np.ones_like(xr)]))
    out = seeds.copy()
    for k, (x, y) in enumerate(seeds):
        xi, yi = int(round(x)), int(round(y))
        if xi - win < 0 or yi - win < 0 or xi + win >= W or yi + win >= H:
            continue
        a, b, c, d, e, _ = Ap @ g[yi - win:yi + win + 1, xi - win:xi + win + 1].ravel()
        try:
            dxy = np.linalg.solve(np.array([[2 * a, b], [b, 2 * c]]), [-d, -e])
        except np.linalg.LinAlgError:
            continue
        if abs(dxy[0]) <= win and abs(dxy[1]) <= win:
            out[k] = [xi + dxy[0], yi + dxy[1]]
    return out.astype(np.float32)


def detect_markers(gray, filled, shape_validate=True, subpix="saddle"):
    inner = cv2.erode(filled, np.ones((15, 15), np.uint8)) > 0
    if inner.sum() < 200:
        return np.zeros((0, 2), np.float32)
    thr = float(np.clip(np.median(gray[inner]) * 0.62, 55, 135))   # markers far darker than green
    dark = (inner & (gray < thr)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    area = inner.sum(); pts = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]; w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        if a < 8 or a > area * 0.01:
            continue
        if max(w, h) > 0.15 * gray.shape[0]:
            continue
        if shape_validate:
            # a "+" is sparse in its bbox; reject solid blobs (fingertips/notch -> high
            # extent) and streaks (extreme aspect). Rotation-invariant.
            extent = a / float(max(1, w * h)); ar = w / float(max(1, h))
            if extent > 0.62 or extent < 0.10 or ar < 0.35 or ar > 2.9:
                continue
        pts.append([float(cent[i][0]), float(cent[i][1])])
    if not pts:
        return np.zeros((0, 2), np.float32)
    pts = np.array(pts, np.float32)
    if subpix == "saddle":
        pts = saddle_refine(gray, pts)
    elif subpix == "cornersubpix":
        p = pts.reshape(-1, 1, 2).copy()
        cv2.cornerSubPix(gray, p, (5, 5), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        pts = p.reshape(-1, 2)
    return pts.reshape(-1, 2)


# ============================ lattice model ============================
G_IDEAL = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], np.float32)  # (col,row)
PRIOR_RECT_G = np.array([[-0.125, -0.25], [1.125, -0.25],
                         [1.125, 2.25], [-0.125, 2.25]], np.float32)


def match_lattice(det, predH, diag, thr_frac=0.12):
    """Greedy match the 6 lattice nodes (projected by predH) to detections."""
    proj = cv2.perspectiveTransform(G_IDEAL.reshape(-1, 1, 2), predH).reshape(-1, 2)
    thr = thr_frac * diag
    src, dst, used = [], [], set()
    for gi in np.argsort([np.min(np.linalg.norm(det - p, axis=1)) for p in proj]):
        dd = np.linalg.norm(det - proj[gi], axis=1); j = int(np.argmin(dd))
        if dd[j] < thr and j not in used:
            used.add(j); src.append(G_IDEAL[gi]); dst.append(det[j])
    return np.array(src, np.float32), np.array(dst, np.float32)


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


def _oneeuro_causal(x, mincut, beta, fs):
    out = np.empty_like(x); out[0] = x[0]; xp = x[0]; dxf = 0.0
    a_d = 1.0 / (1.0 + (fs / (2 * np.pi * 1.0)))
    for i in range(1, len(x)):
        dxf = a_d * ((x[i] - xp) * fs) + (1 - a_d) * dxf
        a = 1.0 / (1.0 + (fs / (2 * np.pi * (mincut + beta * abs(dxf)))))
        xf = a * x[i] + (1 - a) * xp
        out[i] = xf; xp = xf
    return out


def oneeuro_fb(arr, good, fs, mincut=1.2, beta=0.4):
    """Zero-phase One-Euro (forward+backward): speed-adaptive -> heavy smoothing at rest
    (kills shimmer), low lag during motion. Gaps interpolated over `good`."""
    N, D = arr.shape; out = arr.astype(np.float64).copy(); idx = np.arange(N); g = good.astype(bool)
    if g.sum() >= 2:
        for c in range(D):
            out[~g, c] = np.interp(idx[~g], idx[g], out[g, c])
    for c in range(D):
        f = _oneeuro_causal(out[:, c], mincut, beta, fs)
        b = _oneeuro_causal(out[::-1, c], mincut, beta, fs)[::-1]
        out[:, c] = 0.5 * (f + b)
    return out


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


def despill(bgr, region, bias=0.5):
    out = bgr.astype(np.float32); b, gch, r = out[:, :, 0], out[:, :, 1], out[:, :, 2]
    ref = bias * r + (1 - bias) * b; sel = region.astype(bool)
    gch[sel] = np.minimum(gch, ref)[sel]; return out.astype(np.uint8)


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser(description="Marker-tracked green-screen replacement.")
    ap.add_argument("--video", required=True, help="base clip with a green device screen")
    ap.add_argument("--image", help="replacement still image")
    ap.add_argument("--screen", help="replacement video (looped) — alternative to --image")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hue", default="35,90", help="green hue lo,hi (OpenCV 0-179)")
    ap.add_argument("--sat-min", type=int, default=40)
    ap.add_argument("--val-min", type=int, default=40)
    ap.add_argument("--pos-window", type=int, default=5, help="zero-phase smoothing window: position")
    ap.add_argument("--rot-window", type=int, default=9, help="smoothing window: rotation")
    ap.add_argument("--scale-window", type=int, default=11, help="smoothing window: uniform scale")
    ap.add_argument("--persp-window", type=int, default=7, help="smoothing window: perspective residual")
    ap.add_argument("--overfill", type=float, default=1.04, help="warp the insert onto a quad this much "
                    "bigger so it always covers the green; the key clips it back")
    ap.add_argument("--feather", type=float, default=1.2, help="key edge feather sigma (px)")
    ap.add_argument("--despill-bias", type=float, default=0.5)
    ap.add_argument("--low-conf", choices=["black", "ui"], default="black",
                    help="where the marker track is low-confidence: 'black' keys the green to black "
                         "(default — never shows a misaligned insert); 'ui' composites anyway")
    ap.add_argument("--low-conf-fade", type=float, default=5.0,
                    help="frames to crossfade insert<->black at confidence boundaries")
    ap.add_argument("--no-shape-validate", dest="shape_validate", action="store_false",
                    help="keep all dark blobs as markers (don't reject by cross shape)")
    ap.add_argument("--subpix", choices=["saddle", "cornersubpix", "none"], default="saddle",
                    help="marker sub-pixel method (default saddle: best on motion blur)")
    ap.add_argument("--no-oneeuro", dest="oneeuro", action="store_false",
                    help="disable the One-Euro speed-adaptive pre-filter")
    ap.add_argument("--global-key", action="store_true",
                    help="experimental globally-fixed continuous chroma matte (off by default)")
    ap.add_argument("--debug", help="optional overlay diagnostic clip (tracked quad + markers)")
    ap.set_defaults(shape_validate=True, oneeuro=True)
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
    green_d = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); grays.append(g)
        m = green_mask(fr, hue, args.sat_min, args.val_min)
        if int(np.count_nonzero(m)) < 4000:
            quads.append(None); marks.append(np.zeros((0, 2), np.float32)); ok_s.append(False); continue
        q, filled = screen_quad_and_filled(m)
        quads.append(q); ok_s.append(q is not None)
        marks.append(detect_markers(g, filled, args.shape_validate, args.subpix)
                     if filled is not None else np.zeros((0, 2), np.float32))
        if args.global_key and filled is not None:
            green_d.append(np.percentile(greenness(fr)[filled > 0], [8, 60]))
    cap.release(); N = len(grays)
    if N == 0:
        C.die("no frames read from --video")
    if not any(ok_s):
        C.die("no green screen detected — widen --hue/--sat-min/--val-min, use --debug")
    gk_cb = gk_cw = None
    if args.global_key and green_d:
        gd = np.array(green_d); gk_cb = float(np.median(gd[:, 0])); gk_cw = float(np.median(gd[:, 1]))
        if gk_cw - gk_cb < 8:
            gk_cw = gk_cb + 8

    # ---- calibrate the screen rectangle in lattice space ----
    def collect(prior):
        out = []
        for i in range(N):
            if not ok_s[i] or quads[i] is None or len(marks[i]) < 4:
                continue
            Hq, _ = cv2.findHomography(prior, quads[i], 0)
            if Hq is None:
                continue
            diag = np.linalg.norm(quads[i][0] - quads[i][2])
            src, dst = match_lattice(marks[i], Hq, diag)
            if len(src) < 4:
                continue
            Hg, _ = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
            if Hg is None:
                continue
            rg = cv2.perspectiveTransform(quads[i].reshape(-1, 1, 2), np.linalg.inv(Hg)).reshape(-1, 2)
            if not np.any(np.abs(rg) > 6):
                out.append(rg)
        return out

    rects = collect(PRIOR_RECT_G)
    if not rects:
        C.die("no tracking markers found in the green screen (need black '+' marks); "
              "this tool tracks a marked screen. Add markers or check --hue.")
    rect_g = np.median(np.array(rects), axis=0).astype(np.float32)
    r2 = collect(rect_g)
    if len(r2) >= len(rects):
        rect_g = np.median(np.array(r2), axis=0).astype(np.float32)
    # markers are drawn parallel to the screen edges -> the rectangle is axis-aligned in
    # lattice space; drop the green-quad skew (notch/rounded corners) that would inject tilt.
    xL = float((rect_g[0, 0] + rect_g[3, 0]) / 2); xR = float((rect_g[1, 0] + rect_g[2, 0]) / 2)
    yT = float((rect_g[0, 1] + rect_g[1, 1]) / 2); yB = float((rect_g[2, 1] + rect_g[3, 1]) / 2)
    rect_g = np.array([[xL, yT], [xR, yT], [xR, yB], [xL, yB]], np.float32)

    # ---- PASS 2: per-frame homography (markers fused with the green quad) ----
    MK_W = 3
    corners = np.zeros((N, 4, 2), np.float32)
    good = np.zeros(N, bool); source = ["none"] * N
    prevH = None; last_corn = None
    for i in range(N):
        Hq = None
        if ok_s[i] and quads[i] is not None:
            Hq, _ = cv2.findHomography(rect_g, quads[i], 0)
        diag = np.linalg.norm(quads[i][0] - quads[i][2]) if (quads[i] is not None) else 200.0
        msrc = mdst = None
        if ok_s[i] and len(marks[i]) >= 4:
            predH = Hq if Hq is not None else prevH
            if predH is not None:
                s0, d0 = match_lattice(marks[i], predH, diag, thr_frac=0.10)
                if len(s0) >= 4:
                    _, mask = cv2.findHomography(s0, d0, cv2.RANSAC, 0.03 * diag)
                    if mask is not None and mask.ravel().sum() >= 4:
                        keep = mask.ravel().astype(bool); msrc, mdst = s0[keep], d0[keep]
        full_span = False
        if msrc is not None:
            rows = {int(round(v)) for v in msrc[:, 1]}; cols = {int(round(v)) for v in msrc[:, 0]}
            full_span = (0 in rows and 2 in rows and 0 in cols and 1 in cols)
        Hcur = None
        if msrc is not None and full_span:
            Hf, _ = cv2.findHomography(msrc, mdst, 0)
            if Hf is not None and Hq is not None:
                cn = cv2.perspectiveTransform(rect_g.reshape(-1, 1, 2), Hf).reshape(4, 2)
                if np.max(np.linalg.norm(cn - quads[i], axis=1)) > 0.45 * diag:
                    Hf = None
            if Hf is not None:
                Hcur = Hf; source[i] = "marker"; good[i] = True
        if Hcur is None and msrc is not None and Hq is not None:
            Hf, _ = cv2.findHomography(np.vstack([np.repeat(msrc, MK_W, 0), rect_g]),
                                       np.vstack([np.repeat(mdst, MK_W, 0), quads[i]]), 0)
            if Hf is not None:
                cn = cv2.perspectiveTransform(rect_g.reshape(-1, 1, 2), Hf).reshape(4, 2)
                if np.max(np.linalg.norm(cn - quads[i], axis=1)) < 0.30 * diag:
                    Hcur = Hf; source[i] = "partial"
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
    corners_in = corners
    if args.oneeuro:
        corners_in = oneeuro_fb(corners.reshape(N, 8), good, fps).reshape(N, 4, 2).astype(np.float32)
    corners_s = stabilise(corners_in, good, aspect,
                          args.pos_window, args.rot_window, args.scale_window, args.persp_window)

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
        if args.global_key and gk_cb is not None:
            a2 = screen_matte_global(frame, gk_cb, gk_cw)
            gm = (a2 > 0.5).astype(np.uint8) * 255
            alpha = np.clip(cv2.GaussianBlur(a2, (0, 0), args.feather), 0, 1)[:, :, None]
        else:
            gm = screen_matte(frame, hue, args.sat_min, args.val_min)
            alpha = np.clip(cv2.GaussianBlur(gm.astype(np.float32) / 255.0, (0, 0), args.feather), 0, 1)[:, :, None]
        region = cv2.dilate(gm, np.ones((5, 5), np.uint8))
        plate = despill(frame, region, args.despill_bias)
        content = card.astype(np.float32) * float(w[fi])     # fade to black where low-confidence
        out = content * alpha + plate.astype(np.float32) * (1 - alpha)
        sel = cv2.dilate(gm, np.ones((7, 7), np.uint8)).astype(bool)
        out[:, :, 1][sel] = np.minimum(out[:, :, 1], (out[:, :, 2] + out[:, :, 0]) / 2)[sel]
        vw.write(np.clip(out, 0, 255).astype(np.uint8))
        if dbg is not None:
            d = frame.copy()
            col = {"marker": (0, 255, 0), "partial": (0, 255, 255), "quad": (0, 165, 255),
                   "hold": (0, 0, 255), "none": (128, 128, 128)}[source[fi]]
            cv2.polylines(d, [S.astype(np.int32)], True, col, 2)
            for p in marks[fi]:
                cv2.circle(d, (int(p[0]), int(p[1])), 6, (255, 0, 0), 1)
            cv2.putText(d, f"f{fi} {source[fi]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            dbg.write(d)
        fi += 1
    cap.release(); vw.release()
    if dbg is not None:
        dbg.release()

    C.ensure_parent(args.out)
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
