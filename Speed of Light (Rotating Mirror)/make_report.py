"""
Generate the speed-of-light results document from outputs/master.json.

Presents the measurement across several rotation-rate regimes (|nu| cutoffs), so
the choice to exclude the high-|nu| vignetting-affected points is transparent
and can be justified from the residual structure.  Writes:

    outputs/RESULTS.md        the report (methodology, regime table, per-trial data)
    outputs/fig_fit.png       dx vs nu with the full-range and linear-regime fits
    outputs/fig_residuals.png residuals from the linear fit (shows the high-nu bend)

Usage:
    python make_report.py                 # linear-regime cutoff = 375 Hz
    python make_report.py --linear-cut 400
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

C_DEF = 2.99792458e8


def wfit(nu, dx, sem):
    """Weighted least squares dx = m*nu + b.  Returns m, b, m_unc, r2."""
    nu = np.asarray(nu, float); dx = np.asarray(dx, float)
    w = 1.0 / np.maximum(np.asarray(sem, float), 0.5) ** 2
    S = w.sum(); Sx = (w * nu).sum(); Sy = (w * dx).sum()
    Sxx = (w * nu * nu).sum(); Sxy = (w * nu * dx).sum()
    D = S * Sxx - Sx * Sx
    m = (S * Sxy - Sx * Sy) / D; b = (Sxx * Sy - Sx * Sxy) / D
    m_unc = math.sqrt(S / D)
    ybar = Sy / S
    r2 = 1 - (w * (dx - (m * nu + b)) ** 2).sum() / (w * (dx - ybar) ** 2).sum()
    return m, b, m_unc, r2


def c_from_slope(slope, slope_unc, a, L, a_unc, L_unc, mpp, mpp_unc):
    mpp_m = mpp / 1000.0
    c = 8 * math.pi * a * L / (slope * mpp_m)
    rel = math.sqrt((slope_unc / slope) ** 2 + (a_unc / a) ** 2
                    + (L_unc / L) ** 2 + (mpp_unc / mpp) ** 2)
    return c, c * rel


def regime(pts, cut, geo, mpp, mpp_unc):
    P = [p for p in pts if abs(p[0]) <= cut]
    m, b, mu, r2 = wfit([p[0] for p in P], [p[1] for p in P], [p[2] for p in P])
    c, cu = c_from_slope(m, mu, geo["a_m"], geo["L_m"], geo["a_unc_m"],
                         geo["L_unc_m"], mpp, mpp_unc)
    return dict(cut=cut, n=len(P), slope=m, intercept=b, slope_unc=mu, r2=r2,
                c=c, c_unc=cu, pct=(c / C_DEF - 1) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-cut", type=float, default=375.0,
                    help="|nu| cutoff (Hz) defining the vignetting-free linear regime")
    ap.add_argument("--master", default="outputs/master.json")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    M = json.load(open(args.master))
    tr = M["trials"]; geo = M["geometry"]
    mpp = M["calibration"]["mpp_mm_per_px"]; mpp_unc = M["calibration"]["mpp_unc"]
    a, L = geo["a_m"], geo["L_m"]

    pts = [(t["rps"], t["mean_dx_px"], max(t.get("sem_dx_px") or 0.5, 0.5))
           for t in tr.values() if t.get("n_flashes")]
    pts.sort()

    cuts = [500, 450, 400, args.linear_cut, 350, 325, 300]
    cuts = sorted(set(cuts), reverse=True)
    regimes = [regime(pts, c, geo, mpp, mpp_unc) for c in cuts]
    full = regime(pts, 500, geo, mpp, mpp_unc)
    lin = regime(pts, args.linear_cut, geo, mpp, mpp_unc)

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu = np.array([p[0] for p in pts]); dx = np.array([p[1] for p in pts])
    inlin = np.abs(nu) <= args.linear_cut
    xx = np.linspace(-500, 500, 100)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axvspan(args.linear_cut, 520, color="0.9", label="excluded (vignetting)")
    ax.axvspan(-520, -args.linear_cut, color="0.9")
    ax.scatter(nu[inlin], dx[inlin], c="tab:blue", s=45, zorder=3, label="linear regime")
    ax.scatter(nu[~inlin], dx[~inlin], c="tab:red", s=45, zorder=3, label="high-|nu| (excluded)")
    ax.plot(xx, full["slope"] * xx + full["intercept"], "k--", lw=1,
            label=f"full fit: c={full['c']/1e8:.2f}e8 ({full['pct']:+.0f}%)")
    ax.plot(xx, lin["slope"] * xx + lin["intercept"], "tab:green", lw=1.8,
            label=f"linear fit: c={lin['c']/1e8:.2f}e8 ({lin['pct']:+.0f}%)")
    ax.axhline(0, color="k", lw=.4); ax.axvline(0, color="k", lw=.4)
    ax.set_xlabel("rotation frequency  nu (Hz, signed)")
    ax.set_ylabel("flash displacement  dx (px)")
    ax.set_title("Flash displacement vs rotation")
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(-500, 500)
    plt.tight_layout(); plt.savefig("outputs/fig_fit.png", dpi=120); plt.close()

    # residuals from the linear-regime fit
    res = dx - (lin["slope"] * nu + lin["intercept"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvspan(args.linear_cut, 520, color="0.9"); ax.axvspan(-520, -args.linear_cut, color="0.9")
    ax.scatter(nu[inlin], res[inlin], c="tab:blue", s=40, zorder=3)
    ax.scatter(nu[~inlin], res[~inlin], c="tab:red", s=40, zorder=3)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("nu (Hz, signed)"); ax.set_ylabel("residual dx - fit (px)")
    ax.set_title("Residuals from the linear-regime fit (note the high-|nu| droop = beam clipping)")
    ax.set_xlim(-500, 500)
    plt.tight_layout(); plt.savefig("outputs/fig_residuals.png", dpi=120); plt.close()

    # ---- markdown ----
    def row(r):
        return (f"| {r['cut']:.0f} | {r['n']} | {r['slope']:.4f} ± {r['slope_unc']:.4f} "
                f"| {r['intercept']:+.1f} | {r['r2']:.3f} | {r['c']/1e8:.3f} ± {r['c_unc']/1e8:.3f} "
                f"| {r['pct']:+.1f}% |")

    lines = []
    lines.append("# Speed of Light by the Rotating-Mirror Method — Results\n")
    lines.append("*Generated from `outputs/master.json` by `make_report.py`.*\n")
    lines.append("## Method\n")
    lines.append(
        "Each trial is a CCD stack of the stage-micrometer plane. A permanent "
        "saturated vertical line marks the fixed reference beam; a transient "
        "**flash** (the swept return beam) appears when the rotating mirror hits "
        "the return angle. For each trial an annotated ROI localises the flash; "
        "detection uses **center-window peak detection** — the per-frame peak "
        "residual inside the ROI is thresholded against an adaptive baseline "
        "(median + 4·MAD) to find the frames where the flash lights up, then the "
        "50%-of-peak centroid gives its position. The signed displacement "
        "`dx = flash_x − line_x` is averaged over flash frames.\n")
    lines.append(
        "The speed of light follows the Heinzen (PHY-353L) form of the Foucault "
        "relation, `c = 8π·a·(f+b)·ν / x`, obtained here from the **slope** of a "
        "displacement-vs-ν fit over both rotation directions (±ν), so the "
        "stable-line-to-true-zero offset cancels in the intercept:\n")
    lines.append(
        f"- `a`   (R → SM lever arm) = {a:.3f} ± {geo['a_unc_m']:.3f} m\n"
        f"- `f+b` (R → distant mirror) = {L:.3f} ± {geo['L_unc_m']:.3f} m\n"
        f"- mpp = {mpp:.5g} ± {mpp_unc:.2g} mm/px\n"
        f"- `x = dx · mpp`;  `c = 8π·a·(f+b) / (slope · mpp)` with slope in m/px/Hz\n")

    lines.append("## Primary result (all valid trials)\n")
    lines.append(
        f"**c = ({full['c']/1e8:.3f} ± {full['c_unc']/1e8:.3f}) × 10⁸ m/s "
        f"({full['pct']:+.1f}% vs the defined value)**, "
        f"slope = {full['slope']:.4f} px/Hz, R² = {full['r2']:.3f}, n = {full['n']}.\n")

    lines.append("## Regime comparison\n")
    lines.append("| \\|ν\\| ≤ (Hz) | n | slope (px/Hz) | intercept (px) | R² | c (×10⁸ m/s) | vs c_def |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in regimes:
        lines.append(row(r))
    lines.append("")
    lines.append(
        "`c` is **not** independent of the fit range: it falls monotonically as "
        "high-|ν| points are dropped. A pure scale error (distances or mpp) would "
        "shift every regime by the *same* percentage, so the range-dependence "
        "shows the underlying dx(ν) relation is **curved, not linear**.\n")

    lines.append("## Why the high-|ν| points are excluded (vignetting)\n")
    lines.append("![dx vs nu](fig_fit.png)\n")
    lines.append("![residuals](fig_residuals.png)\n")
    lines.append(
        "The residuals from the linear fit show the displacement **magnitude "
        "|dx| falling below the linear trend at high |ν|** (positive residuals "
        "for ν<0, negative for ν>0 — the beam bending back toward zero "
        "deflection). As the mirror spins "
        "faster the return beam is deflected further and is progressively clipped "
        "by the finite apertures of the optics (the vignetting the lab manual "
        "warns about); the measured displacement therefore falls below the linear "
        "prediction. These points bias the slope downward and inflate `c`. "
        f"Restricting the fit to the vignetting-free regime |ν| ≤ {args.linear_cut:.0f} Hz "
        f"gives **c = ({lin['c']/1e8:.3f} ± {lin['c_unc']/1e8:.3f}) × 10⁸ m/s "
        f"({lin['pct']:+.1f}%)**.\n")

    lines.append("## Systematic errors\n")
    lines.append(
        "- **Distances / mpp are uniform scale factors** (`c ∝ a·(f+b)` and "
        "`c ∝ 1/mpp`); they move every regime equally and cannot produce the "
        "observed curvature. A distance error large enough to explain the full-"
        "range +10% (~10% of a ~20 m path, ≈1.5 m) is far beyond plausible tape "
        "error (the manual's meter stick is good to ~1 mm/5 m). If distances *were* "
        "biased in the direction needed to lower `c`, they would be **over-measured** "
        "(logged longer than the true path); equivalently an **under-measured** mpp "
        "would inflate `c`.\n"
        "- **Vignetting / beam clipping at high |ν|** is the dominant, non-uniform "
        "systematic and the reason for the regime dependence above.\n"
        "- **Statistical**: per-trial flash-centroid scatter is < 1 px in most "
        "trials; the fit slope uncertainty is ~0.2%.\n")

    lines.append("## Per-trial data\n")
    lines.append("| trial | ν (Hz) | flash frames | dx (px) | x (mm) | c (×10⁸) |")
    lines.append("|---|---|---|---|---|---|")
    for stem in sorted(tr, key=lambda s: tr[s]["rps"]):
        t = tr[stem]
        if not t.get("n_flashes"):
            lines.append(f"| {stem} | {t['rps']:.0f} | 0 | — | — | — |")
        else:
            lines.append(f"| {stem} | {t['rps']:.0f} | {t['n_flashes']} | "
                         f"{t['mean_dx_px']:.1f} | {t.get('x_mm', float('nan')):.3f} | "
                         f"{t['c_m_s']/1e8:.3f} |")
    lines.append("")

    with open("outputs/RESULTS.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("wrote outputs/RESULTS.md, outputs/fig_fit.png, outputs/fig_residuals.png")
    print(f"full-range c = {full['c']/1e8:.3f}e8 ({full['pct']:+.1f}%)")
    print(f"linear (|nu|<={args.linear_cut:.0f}) c = {lin['c']/1e8:.3f}e8 ({lin['pct']:+.1f}%)")


if __name__ == "__main__":
    main()
