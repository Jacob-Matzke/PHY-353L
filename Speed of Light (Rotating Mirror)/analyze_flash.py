"""
Speed-of-Light (rotating-mirror) flash-displacement pipeline.

The camera records a .tif stack of a rotating-mirror speed-of-light setup.  Each
frame contains:

  * a permanent, saturated *vertical bright line* -- the stable reference beam
    (full image height, x ~= 554 in the D2 323 RPS sample), and
  * a dynamic interference-fringe disc that shifts every frame, plus
  * transient *flashes*: a compact bright spot that pulses on/off in a subset of
    frames as the mirror face sweeps past the return angle.  The flash sits at a
    fixed horizontal offset from the reference line (the deflection we measure).

What we want is the horizontal pixel distance dx = |flash_x - line_x|, averaged
over every frame in which a flash is detected.

Why not raw frame differencing?  The moving fringe disc dominates a naive
frame-to-frame difference and swamps the flash.  Instead we subtract a *temporal
median* background (which captures the stable line and the static part of the
fringe pattern) and detect compact, bright, transient blobs in the residual.

Sign / side convention
----------------------
The rotation rate is encoded in the filename as a signed number of revolutions
per second (RPS), e.g. "323 RPS", "D2 323 RPS", "D2 -256 RPS".  The sign gives
the rotation direction and therefore which side of the reference line the flash
appears on:

    RPS > 0  ->  flash on the RIGHT of the line   (clockwise)
    RPS < 0  ->  flash on the LEFT of the line     (counter-clockwise)

The RPS magnitude is read from the first signed-numeric whitespace-separated
token in the filename (so a leading "D2" prefix is ignored).

Usage
-----
    python analyze_flash.py                       # auto: first data/*.tif
    python analyze_flash.py "data/D2 323 RPS.tif"
    python analyze_flash.py "data/D2 323 RPS.tif" --thr 110 --min-area 80
    python analyze_flash.py "data/D2 323 RPS.tif" --no-overlay

Outputs (under outputs/<stem>/):
    <stem>_flashes.csv   one row per detected flash frame
    <stem>_summary.txt   reference line, mean/median/std dx, frame count
    <stem>_overlay.png   max-projection with the line + every flash centroid
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


# --- detection defaults (tuned on data/"D2 323 RPS.tif") --------------------
THR = 110          # residual intensity (frame - median) to count as "bright"
MIN_AREA = 80      # min connected-component area (px) for a real flash blob
MIN_PEAK = 150     # min residual peak inside the blob
GUARD = 35         # ignore pixels within +/-GUARD px of the line (band edges)

# --- apparatus geometry (the standard PHY-353L setup) -----------------------
# Measured component separations (cm).  Uncertainty +/-0.05 cm each.
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

# Foucault formula  c = 4 A D^2 w / ((D + B) ds),  w = 2*pi*nu (rad/s), where
#   D = optical path rotating-mirror -> distant fixed mirror (time of flight),
#   B = lens -> rotating mirror,
#   A = lens -> image/source spot (where the microscope sees the spot).
# Defaults map the distance sheet onto these (folded path through mirrors A-D):
def _geometry_defaults():
    d = DISTANCES_CM
    # D: rotating mirror -> Mirror D, folded through the lens and fold mirrors
    D = (d[("Rotating Mirror", "Lens")] + d[("Lens", "Mirror A")]
         + d[("Mirror A", "Mirror B")] + d[("Mirror B", "Mirror C")]
         + d[("Mirror C", "Mirror D")])
    # B: lens -> rotating mirror
    B = d[("Rotating Mirror", "Lens")]
    # A: lens -> spot, unfolded back through MR and the beam splitter to SM
    A = (d[("Rotating Mirror", "Lens")] + d[("Beam Splitter", "Rotating Mirror")]
         + d[("Beam Splitter", "SM")])
    return A / 100.0, B / 100.0, D / 100.0   # metres


def parse_rps(stem: str) -> float:
    """Return the signed RPS encoded in the filename stem.

    The first whitespace-separated token that parses as a (optionally signed)
    number is taken as the RPS, so both "323 RPS" and "D2 323 RPS" work, and a
    leading sign ("-256 RPS") selects the rotation direction.
    """
    for tok in stem.split():
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", tok):
            return float(tok)
    raise ValueError(f"Could not find an RPS value in filename stem: {stem!r}")


def find_line_x(median: np.ndarray) -> float:
    """Horizontal position of the saturated stable reference line.

    Taken as the centre of the columns whose median-image column-mean profile is
    within 1% of its peak (the saturated core of the vertical band).
    """
    profile = median.mean(axis=0)
    peak = profile.max()
    core = np.where(profile >= 0.99 * peak)[0]
    return float(core.mean())


def detect_flashes(stack: np.ndarray, median: np.ndarray, line_x: float,
                   side: str, thr=THR, min_area=MIN_AREA, min_peak=MIN_PEAK,
                   guard=GUARD):
    """Detect one flash centroid per frame (where present).

    Returns a list of dicts: frame, cx, cy, area, peak, dx (= |cx - line_x|).
    For each frame we keep the largest connected residual blob on the expected
    side of the line that clears the area/peak gates; its intensity-weighted
    centroid is the flash position.
    """
    n_frames, h, w = stack.shape
    xs_grid = np.arange(w)
    # column mask: keep only the side of the line where the flash appears
    if side == "right":
        col_ok = xs_grid > line_x + guard
    else:
        col_ok = xs_grid < line_x - guard

    flashes = []
    for i in range(n_frames):
        resid = stack[i].astype(np.int16) - median
        mask = resid > thr
        mask[:, ~col_ok] = False
        if not mask.any():
            continue
        lab, n = ndi.label(mask)
        if n == 0:
            continue
        # largest component
        areas = ndi.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        c = int(np.argmax(areas)) + 1
        area = int(areas[c - 1])
        if area < min_area:
            continue
        ys, xs = np.where(lab == c)
        weights = resid[ys, xs].astype(float)
        peak = float(weights.max())
        if peak < min_peak:
            continue
        cx = float((xs * weights).sum() / weights.sum())
        cy = float((ys * weights).sum() / weights.sum())
        flashes.append(dict(frame=i, cx=cx, cy=cy, area=area,
                            peak=peak, dx=abs(cx - line_x),
                            bx0=int(xs.min()), by0=int(ys.min()),
                            bx1=int(xs.max()), by1=int(ys.max())))
    return flashes


def save_overlay(stack, median, line_x, flashes, path):
    from PIL import Image, ImageDraw
    base = stack.max(axis=0).astype("uint8")
    img = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.line([(line_x, 0), (line_x, img.height)], fill=(0, 255, 0), width=2)
    draw.text((line_x + 4, 6), f"line x={line_x:.0f}", fill=(0, 255, 0))
    for f in flashes:
        cx, cy = f["cx"], f["cy"]
        draw.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 70, 70), width=2)
    draw.text((6, 6), f"{len(flashes)} flashes", fill=(255, 120, 120))
    img.save(path)


def load_mpp(calib_path, mpp_override, mpp_unc_override):
    """Return (mpp_mm_per_px, mpp_uncertainty, source) for the calibration.

    --mpp takes precedence; otherwise read the weighted mpp from the calibration
    JSON written by calibration_tool.py.
    """
    if mpp_override is not None:
        return mpp_override, (mpp_unc_override or 0.0), "manual --mpp"
    if calib_path and os.path.exists(calib_path):
        with open(calib_path) as fh:
            data = json.load(fh)
        s = data.get("summary", {})
        mpp = s.get("mpp_weighted", s.get("mpp_mean"))
        unc = s.get("mpp_weighted_uncertainty", s.get("mpp_sem", 0.0))
        if mpp:
            return float(mpp), float(unc), f"calib JSON ({os.path.basename(calib_path)})"
    return None, None, None


def speed_of_light(rps, ds_m, ds_unc_m, A, B, D, dist_unc_m=DIST_UNC_CM / 100.0):
    """Foucault rotating-mirror c = 4 A D^2 w / ((D+B) ds), with uncertainty.

    rps is signed; |rps| sets the magnitude (w = 2*pi*|rps|).  Relative errors in
    A, D (twice), D+B and ds add in quadrature.
    """
    w = 2.0 * math.pi * abs(rps)
    c = 4.0 * A * D * D * w / ((D + B) * ds_m)
    rel = math.sqrt(
        (dist_unc_m / A) ** 2
        + (2.0 * dist_unc_m / D) ** 2
        + (math.sqrt(2.0) * dist_unc_m / (D + B)) ** 2
        + (ds_unc_m / ds_m) ** 2
    )
    return c, c * rel


def save_flash_grid(stack, line_x, flashes, path, pad=28, pad_y=34, cell_w=260):
    """Montage of every flash: each frame cropped to a common window showing the
    green reference line and the flash boxed in red, tiled into a grid."""
    import math

    from PIL import Image, ImageDraw

    h, w = stack.shape[1:]
    cxs = [f["cx"] for f in flashes]
    cys = [f["cy"] for f in flashes]
    # common crop window: contains the line and every flash, with padding
    x0 = max(0, int(min(min(cxs), line_x) - pad))
    x1 = min(w, int(max(max(cxs), line_x) + pad))
    y0 = max(0, int(min(cys) - pad_y))
    y1 = min(h, int(max(cys) + pad_y))
    cw, ch = x1 - x0, y1 - y0
    scale = max(1, round(cell_w / cw))

    cells = []
    for f in flashes:
        crop = stack[f["frame"], y0:y1, x0:x1].astype("uint8")
        img = Image.fromarray(crop).convert("RGB")
        img = img.resize((cw * scale, ch * scale), Image.NEAREST)
        d = ImageDraw.Draw(img)
        lx = (line_x - x0) * scale
        d.line([(lx, 0), (lx, img.height)], fill=(0, 255, 0), width=2)
        rb = [(f["bx0"] - x0 - 2) * scale, (f["by0"] - y0 - 2) * scale,
              (f["bx1"] - x0 + 2) * scale, (f["by1"] - y0 + 2) * scale]
        d.rectangle(rb, outline=(255, 50, 50), width=2)
        d.text((4, 3), f"f{f['frame']}  dx={f['dx']:.1f}", fill=(255, 230, 0))
        cells.append(img)

    n = len(cells)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    gap = 6
    cw_s, ch_s = cells[0].width, cells[0].height
    grid = Image.new("RGB", (cols * cw_s + (cols + 1) * gap,
                             rows * ch_s + (rows + 1) * gap), (20, 20, 20))
    for i, c in enumerate(cells):
        r, q = divmod(i, cols)
        grid.paste(c, (gap + q * (cw_s + gap), gap + r * (ch_s + gap)))
    grid.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif", nargs="?", help="input .tif (default: first data/*.tif)")
    ap.add_argument("--thr", type=int, default=THR)
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--min-peak", type=int, default=MIN_PEAK)
    ap.add_argument("--guard", type=int, default=GUARD)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--no-overlay", action="store_true")
    # calibration / physics
    ap.add_argument("--calib", default=None,
                    help="calibration JSON (default: outputs/calibration/<stem>_calibration.json)")
    ap.add_argument("--mpp", type=float, default=None,
                    help="manual mm/px override (skips the calibration JSON)")
    ap.add_argument("--mpp-unc", type=float, default=None,
                    help="uncertainty on --mpp (mm/px)")
    _A, _B, _D = _geometry_defaults()
    ap.add_argument("--A", type=float, default=_A, help=f"lens->spot (m), default {_A:.3f}")
    ap.add_argument("--B", type=float, default=_B, help=f"lens->rot.mirror (m), default {_B:.3f}")
    ap.add_argument("--D", type=float, default=_D, help=f"rot.mirror->far mirror (m), default {_D:.3f}")
    ap.add_argument("--no-c", action="store_true", help="skip the speed-of-light calc")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    tif = args.tif
    if tif is None:
        cands = sorted(glob.glob("data/*.tif"))
        if not cands:
            ap.error("no .tif found in data/")
        tif = cands[0]
    stem = os.path.splitext(os.path.basename(tif))[0]

    rps = parse_rps(stem)
    side = "right" if rps >= 0 else "left"

    print(f"file      : {tif}")
    print(f"RPS       : {rps:+g}  ->  flash on the {side.upper()}")

    stack = tifffile.imread(tif)
    if stack.ndim != 3:
        raise ValueError(f"expected a 3-D stack, got shape {stack.shape}")
    print(f"stack     : {stack.shape[0]} frames, {stack.shape[1]}x{stack.shape[2]}, {stack.dtype}")

    median = np.median(stack, axis=0).astype(np.int16)
    line_x = find_line_x(median)
    print(f"line_x    : {line_x:.1f} px (stable reference beam)")

    flashes = detect_flashes(stack, median, line_x, side,
                             thr=args.thr, min_area=args.min_area,
                             min_peak=args.min_peak, guard=args.guard)

    outdir = os.path.join(args.outdir, stem)
    os.makedirs(outdir, exist_ok=True)

    if not flashes:
        print("\nNo flashes detected -- try lowering --thr / --min-area.")
        return

    dx = np.array([f["dx"] for f in flashes])
    mean_dx = float(dx.mean())
    sem = float(dx.std(ddof=1) / np.sqrt(len(dx))) if len(dx) > 1 else 0.0
    print(f"\nflashes   : {len(flashes)} frames")
    print(f"mean dx   : {mean_dx:.2f} px")
    print(f"median dx : {np.median(dx):.2f} px")
    print(f"std dx    : {dx.std(ddof=1) if len(dx) > 1 else 0:.2f} px")
    print(f"SEM dx    : {sem:.2f} px")

    # --- calibration -> physical displacement -> speed of light -------------
    calib = args.calib
    if calib is None:
        m = sorted(glob.glob("outputs/calibration/*_calibration.json"))
        calib = m[0] if m else None
    mpp, mpp_unc, mpp_src = load_mpp(calib, args.mpp, args.mpp_unc)

    ds_mm = ds_unc_mm = c = c_unc = None
    if mpp is not None:
        ds_mm = mean_dx * mpp
        rel_ds = math.sqrt((sem / mean_dx) ** 2 + (mpp_unc / mpp) ** 2)
        ds_unc_mm = ds_mm * rel_ds
        print(f"\nmpp       : {mpp:.5g} mm/px  ({mpp_src})")
        print(f"Delta s   : {ds_mm:.4f} +/- {ds_unc_mm:.4f} mm")
        if not args.no_c:
            c, c_unc = speed_of_light(rps, ds_mm / 1000.0, ds_unc_mm / 1000.0,
                                      args.A, args.B, args.D)
            print(f"\ngeometry  : A={args.A:.3f} m  B={args.B:.3f} m  D={args.D:.3f} m  "
                  f"(w=2pi*{abs(rps):g})")
            print(f"c (meas)  : {c/1e8:.4f}e8 +/- {c_unc/1e8:.4f}e8 m/s")
            print(f"c (def)   : 2.9979e8 m/s   "
                  f"(ratio {c/2.99792458e8:.3f}, {(c/2.99792458e8 - 1)*100:+.1f}%)")
    else:
        print("\nNo mpp: provide --mpp or a calibration JSON to get Delta s / c.")

    # CSV
    csv_path = os.path.join(outdir, f"{stem}_flashes.csv")
    with open(csv_path, "w") as fh:
        fh.write("frame,cx,cy,area,peak,dx\n")
        for f in flashes:
            fh.write(f"{f['frame']},{f['cx']:.3f},{f['cy']:.3f},"
                     f"{f['area']},{f['peak']:.0f},{f['dx']:.3f}\n")

    # summary
    txt_path = os.path.join(outdir, f"{stem}_summary.txt")
    with open(txt_path, "w") as fh:
        fh.write(f"file        : {tif}\n")
        fh.write(f"RPS         : {rps:+g} ({side})\n")
        fh.write(f"frames      : {stack.shape[0]}\n")
        fh.write(f"line_x      : {line_x:.2f} px\n")
        fh.write(f"flashes     : {len(flashes)}\n")
        fh.write(f"mean dx     : {mean_dx:.3f} px\n")
        fh.write(f"median dx   : {np.median(dx):.3f} px\n")
        fh.write(f"std dx      : {dx.std(ddof=1) if len(dx) > 1 else 0:.3f} px\n")
        fh.write(f"SEM dx      : {sem:.3f} px\n")
        if mpp is not None:
            fh.write(f"mpp         : {mpp:.6g} mm/px ({mpp_src})\n")
            fh.write(f"Delta s     : {ds_mm:.4f} +/- {ds_unc_mm:.4f} mm\n")
        if c is not None:
            fh.write(f"A,B,D (m)   : {args.A:.4f}, {args.B:.4f}, {args.D:.4f}\n")
            fh.write(f"c (meas)    : {c:.4e} +/- {c_unc:.2e} m/s\n")
            fh.write(f"c/c_def     : {c/2.99792458e8:.4f}\n")

    if not args.no_overlay:
        overlay_path = os.path.join(outdir, f"{stem}_overlay.png")
        save_overlay(stack, median, line_x, flashes, overlay_path)
        print(f"overlay   : {overlay_path}")
        grid_path = os.path.join(outdir, f"{stem}_flash_grid.png")
        save_flash_grid(stack, line_x, flashes, grid_path)
        print(f"grid      : {grid_path}")

    print(f"csv       : {csv_path}")
    print(f"summary   : {txt_path}")


if __name__ == "__main__":
    main()
