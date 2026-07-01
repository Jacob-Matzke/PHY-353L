"""
Speed-of-Light (rotating-mirror) batch pipeline  (PHY-353L, Heinzen manual).

Each .tif in data/ is a stack from the CCD viewing the stage micrometer.  Every
frame has a permanent saturated vertical line (the fixed reference beam) and, in
a subset of frames, a transient compact *flash* -- the swept return beam.  We
locate each flash by temporal-median subtraction + compact-blob detection and
record its horizontal offset dx = cx - line_x (signed: + = right, - = left).

The rotation rate (signed RPS) is encoded in the filename ("D2 323 RPS",
"D2 -257 RPS"): sign = direction, so + flashes land right of the line, - left.

Physics (Heinzen Eq. 3):  x = 4 a (f+b) w / c ,  w = 2*pi*nu.  With L1 one focal
length from R the return retraces, so displacement is LINEAR in path length:
    c = 4 a (f+b) w / x = 8 pi a (f+b) nu / x
    a   = lever arm R -> SM         (BS->R + BS->SM)
    f+b = optical path R -> mirror M (R->Lens + folded Lens->M)
    x   = beam displacement at SM vs the zero-rotation position

Two estimates of c are produced:
  * per-trial  -- treats the stable line as the zero position (single-frequency),
  * slope fit  -- dx vs nu across ALL trials; the SLOPE gives c and the fitted
    intercept absorbs any offset of the stable line from true zero.  This is the
    manual's recommended method and the headline result.

Caching: heavy per-frame detection is written to a master JSON (outputs/
master.json).  Re-runs only analyse NEW files; `--recompute` re-derives x and c
from the cache with no video decoding.  No per-trial image/CSV files are written
(use `--inspect FILE` to spot-check one trial's flash grid on demand).

Usage
-----
    python analyze_flash.py                 # analyse new tifs, update master.json
    python analyze_flash.py --reanalyze     # re-detect every tif from scratch
    python analyze_flash.py --recompute      # redo x/c/fit from cache only (fast)
    python analyze_flash.py --mpp 0.0201     # override calibration mpp
    python analyze_flash.py --inspect "data/D2 323 RPS.tif"   # one grid image
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import tifffile
from scipy import ndimage as ndi

C_DEF = 2.99792458e8  # defined speed of light (m/s)

# --- detection defaults (tuned on data/"D2 323 RPS.tif") --------------------
THR = 110          # residual intensity (frame - median) to count as "bright"
MIN_AREA = 80      # min connected-component area (px) for a real flash blob
MIN_PEAK = 150     # min residual peak inside the blob
GUARD = 35         # ignore pixels within +/-GUARD px of the line (band edges)
# ROI-constrained detection (used when a box exists for the trial).  We look
# only in a small window around the box CENTRE (where the flash was marked),
# build the per-frame peak-residual signal there, flag frames where it spikes
# above an adaptive baseline (the flash "lighting up"), and centroid those.
# The centre window rejects the near-line and far-side contaminants that sit at
# the box edges; the adaptive spike test rejects flash-off frames.
THR_ROI = 30
MINA_ROI = 12
CUSHION = 15       # px added around the annotated box when searching
WIN_HALF = 16      # centre-window half-size around the box centre
MAD_K = 4.0        # spike threshold: baseline + K * MAD of the box signal
SIG_FLOOR = 15     # absolute floor so pure-noise files stay empty

MASTER = "outputs/master.json"

# --- apparatus geometry (the standard PHY-353L setup) -----------------------
# Measured component separations (cm).  Per-leg uncertainty +/-DIST_UNC_CM.
DISTANCES_CM = {
    ("Laser", "Rotating Mirror"): 489.85,
    ("Rotating Mirror", "Lens"): 491.0,
    ("Lens", "Mirror A"): 11.0,
    ("Mirror A", "Mirror B"): 15.5,
    ("Mirror B", "Mirror C"): 488.4,
    ("Mirror C", "Mirror D"): 466.3,
    ("Beam Splitter", "SM"): 18.6,
    ("Beam Splitter", "Rotating Mirror"): 475.5,
}
DIST_UNC_CM = 0.05


def _geometry_legs():
    d = DISTANCES_CM
    a_legs = [d[("Beam Splitter", "Rotating Mirror")], d[("Beam Splitter", "SM")]]
    L_legs = [d[("Rotating Mirror", "Lens")], d[("Lens", "Mirror A")],
              d[("Mirror A", "Mirror B")], d[("Mirror B", "Mirror C")],
              d[("Mirror C", "Mirror D")]]
    return a_legs, L_legs


def geometry(dist_unc_cm=DIST_UNC_CM):
    """Return (a, L, a_unc, L_unc) in metres for the manual's formula."""
    a_legs, L_legs = _geometry_legs()
    a = sum(a_legs) / 100.0
    L = sum(L_legs) / 100.0
    a_unc = dist_unc_cm * math.sqrt(len(a_legs)) / 100.0
    L_unc = dist_unc_cm * math.sqrt(len(L_legs)) / 100.0
    return a, L, a_unc, L_unc


def parse_rps(stem: str) -> float:
    """Signed RPS from the filename stem (first signed-numeric token)."""
    for tok in stem.split():
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", tok):
            return float(tok)
    raise ValueError(f"no RPS value in filename stem: {stem!r}")


def find_line_x(median: np.ndarray) -> float:
    """Centre of the saturated core of the vertical reference band."""
    profile = median.mean(axis=0)
    core = np.where(profile >= 0.99 * profile.max())[0]
    return float(core.mean())


def detect_flashes_roi(stack, median, line_x, box, win_half=WIN_HALF, k=MAD_K,
                       floor=SIG_FLOOR, want_bbox=False):
    """Centre-window peak-detection flash finder (uses the annotated box).

    Builds the per-frame peak-residual signal in a small window around the box
    centre, flags frames where it exceeds median + k*MAD (and an absolute
    floor) -- these are the frames where the flash lights up -- and returns the
    50%-of-peak intensity centroid for each.
    """
    n_frames = stack.shape[0]
    cx0 = (box[0] + box[2]) // 2; cy0 = (box[1] + box[3]) // 2
    x0 = max(box[0], cx0 - win_half); x1 = min(box[2], cx0 + win_half)
    y0 = max(box[1], cy0 - win_half); y1 = min(box[3], cy0 + win_half)

    win = stack[:, y0:y1, x0:x1].astype(np.int16) - median[y0:y1, x0:x1]
    sig = win.reshape(n_frames, -1).max(axis=1)
    base = float(np.median(sig))
    mad = float(np.median(np.abs(sig - base))) * 1.4826 + 1e-6
    thr = max(base + k * mad, floor)

    flashes = []
    for i in np.where(sig > thr)[0]:
        w_img = stack[i].astype(np.int16) - median
        sub = w_img[y0:y1, x0:x1]
        pk = sub.max()
        mask = ndi.binary_opening(sub > 0.5 * pk, structure=np.ones((2, 2)))
        if not mask.any():
            continue
        lab, n = ndi.label(mask)
        areas = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        c = int(np.argmax(areas)) + 1
        ys, xs = np.where(lab == c)
        weights = sub[ys, xs].astype(float)
        cx = float((xs * weights).sum() / weights.sum()) + x0
        cy = float((ys * weights).sum() / weights.sum()) + y0
        rec = dict(frame=int(i), cx=cx, cy=cy, dx=cx - line_x)
        if want_bbox:
            rec.update(area=int(len(xs)), peak=float(weights.max()),
                       bx0=int(xs.min()) + x0, by0=int(ys.min()) + y0,
                       bx1=int(xs.max()) + x0, by1=int(ys.max()) + y0)
        flashes.append(rec)
    return flashes


def detect_flashes(stack, median, line_x, side="right", thr=THR, min_area=MIN_AREA,
                   min_peak=MIN_PEAK, guard=GUARD, roi=None, cushion=15,
                   want_bbox=False):
    """Detect one flash centroid per frame.

    With an annotated `roi` box, delegates to the centre-window peak detector.
    Otherwise falls back to the side/guard column heuristic (low precision).

    Returns list of dicts with frame, cx, cy, dx (= cx - line_x, signed), and
    (if want_bbox) the blob bounding box + area/peak for grid rendering.
    """
    if roi is not None:
        return detect_flashes_roi(stack, median, line_x, roi, want_bbox=want_bbox)

    n_frames, h, w = stack.shape
    # region-of-interest mask: user box (+cushion) if given, else side/guard
    region = np.zeros((h, w), dtype=bool)
    if roi is not None:
        x0 = max(0, int(roi[0]) - cushion); y0 = max(0, int(roi[1]) - cushion)
        x1 = min(w, int(roi[2]) + cushion); y1 = min(h, int(roi[3]) + cushion)
        region[y0:y1, x0:x1] = True
    else:
        xs_grid = np.arange(w)
        col_ok = xs_grid > line_x + guard if side == "right" else xs_grid < line_x - guard
        region[:, col_ok] = True

    flashes = []
    for i in range(n_frames):
        resid = stack[i].astype(np.int16) - median
        mask = (resid > thr) & region
        if not mask.any():
            continue
        mask = ndi.binary_opening(mask, structure=np.ones((2, 2)))  # kill hot px
        if not mask.any():
            continue
        lab, n = ndi.label(mask)
        areas = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        c = int(np.argmax(areas)) + 1
        area = int(areas[c - 1])
        if area < min_area:
            continue
        ys, xs = np.where(lab == c)
        weights = resid[ys, xs].astype(float)
        if weights.max() < min_peak:
            continue
        cx = float((xs * weights).sum() / weights.sum())
        cy = float((ys * weights).sum() / weights.sum())
        rec = dict(frame=i, cx=cx, cy=cy, dx=cx - line_x)
        if want_bbox:
            rec.update(area=area, peak=float(weights.max()),
                       bx0=int(xs.min()), by0=int(ys.min()),
                       bx1=int(xs.max()), by1=int(ys.max()))
        flashes.append(rec)
    return flashes


def load_roi(path="outputs/roi.json"):
    """Return {stem: (x0,y0,x1,y1)} from the ROI tool, or {} if none."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        boxes = json.load(fh).get("boxes", {})
    return {s: (b["x0"], b["y0"], b["x1"], b["y1"]) for s, b in boxes.items()}


def analyze_one(tif, thr, min_area, min_peak, guard, roi=None, cushion=15):
    """Heavy step: decode a tif and return its cached detection record.

    With an ROI (from the annotation tool) detection is confined to the box and
    a low threshold is used; without one it falls back to the side heuristic.
    """
    stem = os.path.splitext(os.path.basename(tif))[0]
    rps = parse_rps(stem)
    side = "right" if rps >= 0 else "left"
    stack = tifffile.imread(tif)
    if stack.ndim != 3:
        raise ValueError(f"{tif}: expected 3-D stack, got {stack.shape}")
    median = np.median(stack, axis=0).astype(np.int16)
    line_x = find_line_x(median)
    flashes = detect_flashes(stack, median, line_x, side=side, thr=thr,
                             min_area=min_area, min_peak=min_peak, guard=guard,
                             roi=roi, cushion=cushion)
    dx = np.array([f["dx"] for f in flashes], dtype=float)
    n = len(dx)
    rec = {
        "file": tif.replace("\\", "/"),
        "rps": rps, "side": side, "n_frames": int(stack.shape[0]),
        "line_x": round(line_x, 3), "n_flashes": n,
        "roi": list(roi) if roi else None,
        "mean_dx_px": round(float(dx.mean()), 4) if n else None,
        "std_dx_px": round(float(dx.std(ddof=1)), 4) if n > 1 else 0.0,
        "sem_dx_px": round(float(dx.std(ddof=1) / math.sqrt(n)), 4) if n > 1 else 0.0,
        "flashes": [[f["frame"], round(f["cx"], 3), round(f["cy"], 3)] for f in flashes],
    }
    return stem, rec


def load_mpp(calib_path, mpp_override, mpp_unc_override):
    if mpp_override is not None:
        return mpp_override, (mpp_unc_override or 0.0), "manual --mpp"
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as fh:
            s = json.load(fh).get("summary", {})
        mpp = s.get("mpp_weighted", s.get("mpp_mean"))
        unc = s.get("mpp_weighted_uncertainty", s.get("mpp_sem", 0.0))
        if mpp:
            return float(mpp), float(unc), f"calib ({os.path.basename(calib_path)})"
    return None, None, None


def derive_trial_c(rec, mpp, mpp_unc, a, L, a_unc, L_unc):
    """Per-trial c treating the stable line as zero (single-frequency estimate)."""
    if not rec.get("n_flashes"):
        return
    mean_dx = abs(rec["mean_dx_px"])
    sem = rec["sem_dx_px"] or 0.0
    x_mm = mean_dx * mpp
    rel_x = math.sqrt((sem / mean_dx) ** 2 + (mpp_unc / mpp) ** 2) if mean_dx else 0.0
    x_unc_mm = x_mm * rel_x
    rec["x_mm"] = round(x_mm, 4)
    rec["x_unc_mm"] = round(x_unc_mm, 4)
    x_m = x_mm / 1000.0
    w = 2.0 * math.pi * abs(rec["rps"])
    c = 4.0 * a * L * w / x_m
    rel_c = math.sqrt((a_unc / a) ** 2 + (L_unc / L) ** 2 + rel_x ** 2)
    rec["c_m_s"] = c
    rec["c_unc_m_s"] = c * rel_c
    rec["c_over_cdef"] = round(c / C_DEF, 4)


def weighted_linfit(x, y, yerr):
    """Weighted least squares y = m x + b.  Returns (m, b, m_unc, r2)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = 1.0 / np.maximum(np.asarray(yerr, float), 0.5) ** 2   # floor sem at 0.5 px
    S = w.sum(); Sx = (w * x).sum(); Sy = (w * y).sum()
    Sxx = (w * x * x).sum(); Sxy = (w * x * y).sum()
    delta = S * Sxx - Sx * Sx
    m = (S * Sxy - Sx * Sy) / delta
    b = (Sxx * Sy - Sx * Sxy) / delta
    m_unc = math.sqrt(S / delta)
    # weighted R^2
    ybar = Sy / S
    ss_res = (w * (y - (m * x + b)) ** 2).sum()
    ss_tot = (w * (y - ybar) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return m, b, m_unc, r2


def fit_c(trials, mpp, mpp_unc, a, L, a_unc, L_unc):
    """Slope-based c: fit dx(px) vs nu(Hz) over all trials.

    dx = (8 pi a L / (c * mpp_m)) nu  ->  c = 8 pi a L / (slope * mpp_m).
    """
    pts = [(t["rps"], t["mean_dx_px"], t["sem_dx_px"])
           for t in trials.values() if t.get("n_flashes")]
    if len(pts) < 2:
        return None
    nu, dx, sem = zip(*pts)
    slope, intercept, slope_unc, r2 = weighted_linfit(nu, dx, sem)
    mpp_m = mpp / 1000.0
    c = 8.0 * math.pi * a * L / (slope * mpp_m)
    rel = math.sqrt((slope_unc / slope) ** 2 + (a_unc / a) ** 2
                    + (L_unc / L) ** 2 + (mpp_unc / mpp) ** 2)
    return {
        "n_trials": len(pts),
        "slope_px_per_hz": slope, "slope_unc": slope_unc,
        "intercept_px": intercept, "r2": r2,
        "c_m_s": c, "c_unc_m_s": c * rel, "c_over_cdef": c / C_DEF,
    }


def save_flash_grid(tif, thr, min_area, min_peak, guard, path,
                    roi=None, cushion=CUSHION, pad=28, pad_y=34, cell_w=260):
    """Spot-check montage for one trial: each flash boxed red + green line."""
    from PIL import Image, ImageDraw
    stem = os.path.splitext(os.path.basename(tif))[0]
    rps = parse_rps(stem)
    side = "right" if rps >= 0 else "left"
    stack = tifffile.imread(tif)
    median = np.median(stack, axis=0).astype(np.int16)
    line_x = find_line_x(median)
    flashes = detect_flashes(stack, median, line_x, side=side, thr=thr,
                             min_area=min_area, min_peak=min_peak, guard=guard,
                             roi=roi, cushion=cushion, want_bbox=True)
    if not flashes:
        print("no flashes to inspect"); return
    h, w = stack.shape[1:]
    cxs = [f["cx"] for f in flashes]; cys = [f["cy"] for f in flashes]
    x0 = max(0, int(min(min(cxs), line_x) - pad)); x1 = min(w, int(max(max(cxs), line_x) + pad))
    y0 = max(0, int(min(cys) - pad_y)); y1 = min(h, int(max(cys) + pad_y))
    cw, ch = x1 - x0, y1 - y0; scale = max(1, round(cell_w / cw))
    cells = []
    for f in flashes:
        img = Image.fromarray(stack[f["frame"], y0:y1, x0:x1].astype("uint8")).convert("RGB")
        img = img.resize((cw * scale, ch * scale), Image.NEAREST)
        d = ImageDraw.Draw(img)
        lx = (line_x - x0) * scale
        d.line([(lx, 0), (lx, img.height)], fill=(0, 255, 0), width=2)
        d.rectangle([(f["bx0"] - x0 - 2) * scale, (f["by0"] - y0 - 2) * scale,
                     (f["bx1"] - x0 + 2) * scale, (f["by1"] - y0 + 2) * scale],
                    outline=(255, 50, 50), width=2)
        d.text((4, 3), f"f{f['frame']} dx={f['dx']:+.1f}", fill=(255, 230, 0))
        cells.append(img)
    cols = math.ceil(math.sqrt(len(cells))); rows = math.ceil(len(cells) / cols)
    gap = 6; cw_s, ch_s = cells[0].width, cells[0].height
    grid = Image.new("RGB", (cols * cw_s + (cols + 1) * gap, rows * ch_s + (rows + 1) * gap), (20, 20, 20))
    for i, cimg in enumerate(cells):
        r, q = divmod(i, cols)
        grid.paste(cimg, (gap + q * (cw_s + gap), gap + r * (ch_s + gap)))
    grid.save(path)
    print(f"inspect grid -> {path}  ({len(flashes)} flashes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--thr", type=int, default=THR)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--min-peak", type=int, default=MIN_PEAK)
    ap.add_argument("--guard", type=int, default=GUARD)
    ap.add_argument("--reanalyze", action="store_true", help="re-detect every tif")
    ap.add_argument("--recompute", action="store_true",
                    help="redo x/c/fit from master.json only (no video decoding)")
    ap.add_argument("--positive-only", action="store_true",
                    help="use only right-side (RPS >= 0) trials for the fit/report")
    ap.add_argument("--inspect", metavar="TIF", help="write one flash-grid image and exit")
    ap.add_argument("--roi", default="outputs/roi.json", help="ROI boxes from flash_roi_tool.py")
    ap.add_argument("--thr-roi", type=int, default=THR_ROI, help="threshold inside an ROI")
    ap.add_argument("--cushion", type=int, default=CUSHION, help="px around the ROI box")
    ap.add_argument("--no-roi", action="store_true", help="ignore roi.json (side heuristic)")
    # calibration / physics
    ap.add_argument("--calib", default=None)
    ap.add_argument("--mpp", type=float, default=None, help="override mm/px")
    ap.add_argument("--mpp-unc", type=float, default=None)
    ap.add_argument("--dist-unc", type=float, default=DIST_UNC_CM,
                    help=f"per-leg distance uncertainty (cm), default {DIST_UNC_CM}")
    args = ap.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("outputs", exist_ok=True)

    if args.inspect:
        stem = os.path.splitext(os.path.basename(args.inspect))[0]
        roi = None if args.no_roi else load_roi(args.roi).get(stem)
        thr = args.thr_roi if roi else args.thr
        min_area = MINA_ROI if roi else args.min_area
        min_peak = 0 if roi else args.min_peak
        save_flash_grid(args.inspect, thr, min_area, min_peak, args.guard,
                        f"outputs/inspect_{stem}.png", roi=roi, cushion=args.cushion)
        return

    # load cache
    master = {}
    if os.path.exists(MASTER):
        with open(MASTER) as fh:
            master = json.load(fh)
    trials = master.get("trials", {})

    # calibration
    calib = args.calib
    if calib is None:
        m = sorted(glob.glob("outputs/calibration/*_calibration.json"))
        calib = m[0] if m else None
    mpp, mpp_unc, mpp_src = load_mpp(calib, args.mpp, args.mpp_unc)
    if mpp is None:
        ap.error("no mpp: provide --mpp or a calibration JSON in outputs/calibration/")
    a, L, a_unc, L_unc = geometry(args.dist_unc)

    roi_map = {} if args.no_roi else load_roi(args.roi)

    # --- detection (skip when --recompute) ----------------------------------
    if not args.recompute:
        tifs = sorted(glob.glob(os.path.join(args.datadir, "*.tif")))
        n_boxed = 0
        for tif in tifs:
            stem = os.path.splitext(os.path.basename(tif))[0]
            try:
                parse_rps(stem)
            except ValueError:
                continue  # not a trial (e.g. the calibration stack)
            if args.positive_only and parse_rps(stem) < 0:
                continue  # skip left-side trials
            if stem in trials and not args.reanalyze:
                continue  # cached
            roi = roi_map.get(stem)
            if roi:
                n_boxed += 1
            print(f"analysing {stem} {'[ROI]' if roi else '[side]'} ...", flush=True)
            # ROI path uses the low threshold + tiny area floor; else the defaults
            thr = args.thr_roi if roi else args.thr
            min_area = MINA_ROI if roi else args.min_area
            min_peak = 0 if roi else args.min_peak
            stem, rec = analyze_one(tif, thr, min_area, min_peak, args.guard,
                                    roi=roi, cushion=args.cushion)
            trials[stem] = rec
            print(f"  {rec['n_flashes']} flashes, mean dx = "
                  f"{rec['mean_dx_px']} px" if rec['n_flashes'] else "  no flashes")
        if roi_map:
            print(f"({n_boxed} trials used an annotated ROI from {args.roi})")

    if not trials:
        ap.error("no trials in cache; run without --recompute first")

    # --- derive physics for every trial + slope fit -------------------------
    for rec in trials.values():
        for k in ("x_mm", "x_unc_mm", "c_m_s", "c_unc_m_s", "c_over_cdef"):
            rec.pop(k, None)
        derive_trial_c(rec, mpp, mpp_unc, a, L, a_unc, L_unc)
    active = {s: r for s, r in trials.items()
              if not args.positive_only or r["rps"] >= 0}
    fit = fit_c(active, mpp, mpp_unc, a, L, a_unc, L_unc)

    master.update({
        "detection": {"thr": args.thr, "min_area": args.min_area,
                      "min_peak": args.min_peak, "guard": args.guard},
        "calibration": {"mpp_mm_per_px": mpp, "mpp_unc": mpp_unc, "source": mpp_src},
        "geometry": {"a_m": a, "L_m": L, "a_unc_m": a_unc, "L_unc_m": L_unc,
                     "dist_unc_cm": args.dist_unc},
        "trials": trials,
        "fit": fit,
        "scope": "positive-only" if args.positive_only else "all",
    })
    with open(MASTER, "w") as fh:
        json.dump(master, fh, indent=2)

    # --- report -------------------------------------------------------------
    scope = "positive-only" if args.positive_only else "all trials"
    ok = [t for t in active.values() if t.get("n_flashes")]
    print(f"\nscope: {scope}  --  {len(active)} trials ({len(ok)} with flashes), "
          f"cache -> {MASTER}")
    print(f"mpp={mpp:.5g} mm/px   a={a:.3f} m   f+b={L:.3f} m\n")
    print(f"{'trial':<16}{'nu':>7}{'flash':>6}{'dx px':>9}{'x mm':>9}{'c (e8)':>9}")
    for stem in sorted(active, key=lambda s: active[s]["rps"]):
        t = active[stem]
        if not t.get("n_flashes"):
            print(f"{stem:<16}{t['rps']:>7.0f}{0:>6}   (no flash detected)")
            continue
        print(f"{stem:<16}{t['rps']:>7.0f}{t['n_flashes']:>6}"
              f"{t['mean_dx_px']:>9.1f}{t['x_mm']:>9.3f}{t['c_m_s']/1e8:>9.3f}")

    if fit:
        print(f"\n--- slope fit (dx vs nu, {fit['n_trials']} trials) ---")
        print(f"slope   : {fit['slope_px_per_hz']:.4f} +/- {fit['slope_unc']:.4f} px/Hz"
              f"   (R^2 = {fit['r2']:.4f})")
        print(f"intercept: {fit['intercept_px']:.2f} px  (stable-line offset from true zero)")
        print(f"c (slope): {fit['c_m_s']/1e8:.4f}e8 +/- {fit['c_unc_m_s']/1e8:.4f}e8 m/s"
              f"   ({(fit['c_over_cdef']-1)*100:+.1f}% vs c_def)")


if __name__ == "__main__":
    main()
