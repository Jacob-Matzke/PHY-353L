"""
Neon line-shape check (standalone).

Applies the SAME wavelength offset that was calibrated from the helium He I line
(pipeline.compute_helium_offset) to the neon trials, averages the trials, and:
  1. plots the averaged neon spectrum on the He-corrected wavelength axis, and
  2. overlays the strongest neon lines (centered on their peak, peak-normalized)
     and reports a skew metric, to see whether the rightward (red) tail seen on
     the helium line persists on neon.

If the same red skew appears on neon, it points to an INSTRUMENTAL origin
(detector charge-transfer trailing / scattered light), common to every bright
line, rather than something specific to the helium source.

Run:  python neon_skew.py
"""

from __future__ import annotations

import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import pipeline
from pipeline import CONFIG, load_spectrum, compute_helium_offset

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
NEON_ROOT = "Neon Data"
SAVE_DIR = "Neon Processed Outputs"
N_LINES = 5            # number of strongest lines to examine for skew
HALF_WIN = 0.30        # nm half-window around each line for the skew overlay
SHOW = False           # plt.show() in addition to saving


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def average_neon(cfg: dict):
    """Mean intensity over all neon trials, on the He-offset-corrected axis."""
    files = sorted(glob.glob(os.path.join(NEON_ROOT, "*.txt")))
    if not files:
        raise FileNotFoundError(f"no neon spectra in {NEON_ROOT!r}")
    wl0, stack = None, []
    for f in files:
        df = load_spectrum(f, cfg, calibrate=True)   # offset already applied
        if wl0 is None:
            wl0 = df["wavelength"].to_numpy(dtype=float)
        stack.append(df["intensity"].to_numpy(dtype=float))
    inten = np.mean(np.vstack(stack), axis=0)
    return wl0, inten, [os.path.basename(f) for f in files]


def strong_lines(wl: np.ndarray, it: np.ndarray, n: int = N_LINES) -> np.ndarray:
    """Indices of the n most prominent emission lines."""
    rng = float(it.max() - it.min()) or 1.0
    idx, props = find_peaks(it, prominence=0.05 * rng, distance=8)
    order = np.argsort(props["prominences"])[::-1][:n]
    return np.sort(idx[order])


def line_skew(wl: np.ndarray, it: np.ndarray, ipk: int, half: float = HALF_WIN):
    """Return (x, y_baselined, peak_wl, centroid) for the line near index ipk.

    The centroid is taken over points above 10% of the peak; centroid - peak is
    a simple skew metric (positive => red / rightward tail).
    """
    lo, hi = wl[ipk] - half, wl[ipk] + half
    m = (wl >= lo) & (wl <= hi)
    x, y = wl[m], it[m].copy()
    base = float(np.median(np.concatenate([y[:3], y[-3:]])))
    y = y - base
    peak_wl = float(x[np.argmax(y)])
    sel = y > 0.10 * y.max()
    centroid = float(np.sum(x[sel] * y[sel]) / np.sum(y[sel]))
    return x, y, peak_wl, centroid


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_average(wl, it, lines, off, names):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(wl, it, lw=0.8, color="0.3")
    for i in lines:
        ax.annotate(f"{wl[i]:.2f}", (wl[i], it[i]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7, color="C3")
        ax.plot(wl[i], it[i], "v", color="C3", ms=5)
    ax.set_xlabel("Wavelength (nm, He-offset corrected)")
    ax.set_ylabel("Intensity (counts, mean of trials)")
    ax.set_title(f"Neon averaged spectrum  ({len(names)} trials, "
                 f"He offset {off.offset:+.4f} nm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "neon_average_spectrum.png")
    fig.savefig(out, dpi=130)
    if SHOW:
        plt.show()
    plt.close(fig)
    return out


def plot_skew(wl, it, lines):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    print(f"\n  {'line (nm)':>10s} {'peak':>9s} {'centroid':>9s} "
          f"{'skew = centroid-peak (nm)':>26s}")
    rows = []
    for k, i in enumerate(lines):
        x, y, peak_wl, centroid = line_skew(wl, it, i)
        ax.plot(x - peak_wl, y / y.max(), lw=1.4, color=cmap(k % 10),
                label=f"{wl[i]:.2f} nm")
        skew = centroid - peak_wl
        rows.append(skew)
        print(f"  {wl[i]:10.3f} {peak_wl:9.3f} {centroid:9.3f} {skew:+26.4f}")
    ax.axvline(0, color="0.5", lw=0.8, ls="--", label="peak")
    ax.set_xlabel("Wavelength offset from peak (nm)   [+ = red / rightward]")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Neon strong lines, centered & normalized -- skew check")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(SAVE_DIR, "neon_line_skew.png")
    fig.savefig(out, dpi=130)
    if SHOW:
        plt.show()
    plt.close(fig)
    print(f"\n  mean skew over {len(rows)} lines: {np.mean(rows):+.4f} nm "
          f"(positive => same red/rightward tail as helium)")
    return out


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    # 1. reproduce the helium offset (same calibration as the H2D2 analysis)
    off = compute_helium_offset(CONFIG, apply=True)
    print(off.summary())

    # 2. average the neon trials (now offset-corrected)
    wl, it, names = average_neon(CONFIG)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"\naveraged {len(names)} neon trials: {', '.join(names)}")
    print(f"applied He offset: {off.offset:+.4f} +/- {off.offset_err:.4f} nm")

    lines = strong_lines(wl, it)
    p1 = plot_average(wl, it, lines, off, names)
    p2 = plot_skew(wl, it, lines)
    print(f"\nplots written:\n  {p1}\n  {p2}")


if __name__ == "__main__":
    main()
