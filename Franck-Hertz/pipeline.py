"""
Franck-Hertz (Mercury) data-analysis pipeline.

Pipeline stages (each toggleable via CONFIG):
    1. load        - read the two scope channels for a run and combine to (Second, V, I)
    2. denoise     - running-average based noise rejection (default: remove narrow spikes only)
    3. downsample  - thin the point count (default: random sampling)
    4. plot_iv     - I-vs-V graph to sanity-check the curve shape
    5. find_peaks  - locate the Franck-Hertz local maxima in I
    6. plot_peaks  - I-vs-V graph with the maxima marked
    7. spacings    - distance (in V) between consecutive maxima

Conventions (matching the existing notebook):
    C1  -> V  (accelerating voltage ramp, recorded volts)
    C2  -> I  (collector current, recorded as a voltage)

The recorded V is left in *recorded volts*; set CONFIG["v_calibration"] to the
divider/scale factor once it is known to convert spacings to true volts
(mercury should give ~4.9 V).

Peak finding works in time/sample order (V rises ~monotonically with time), so
the natural sample order preserves the curve shape even though V itself is noisy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import median_filter


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

CONFIG = {
    # --- loading ---------------------------------------------------------- #
    "data_root": "Neon Data",
    "skip_rows": 18,                 # scope CSV header rows before "Second,Value"
    "channel_v": "F0001CH1.csv",   # -> V (accelerating voltage)
    "channel_i": "F0001CH2.csv",   # -> I (collector current)

    # --- 2. noise removal ------------------------------------------------- #
    "denoise": {
        "enabled": True,
        # window (in samples) of the centered running average that defines the
        # "expected" local value. Should be wide vs. noise, narrow vs. FH peaks.
        "window": 75,
        # a point is FLAGGED when it deviates from the running average by more
        # than this percent of the running-average value's full I-range.
        "threshold_pct": 3.0,
        # mode:
        #   "narrow_spikes" - only remove flagged points that occur in short
        #                     runs (<= max_spike_width); protects broad FH peaks
        #   "above"         - remove any point above avg + threshold (literal)
        #   "symmetric"     - remove any point deviating +/- threshold
        "mode": "narrow_spikes",
        "max_spike_width": 40,       # only used by "narrow_spikes"; wide enough to kill bursts
    },

    # --- 3. downsampling -------------------------------------------------- #
    "downsample": {
        "enabled": True,
        "method": "random",          # "random" | "stride"
        "fraction": 0.10,            # keep this fraction (random)
        "stride": 10,                # keep every Nth point (stride)
        "seed": 42,
    },

    # --- 5. peak finding -------------------------------------------------- #
    "peaks": {
        # median pre-filter (samples) applied to I before smoothing; rejects
        # noise bursts/spikes that the running average would turn into false
        # peaks. Set to 0 or 1 to disable.
        "median_window": 51,
        # smoothing applied to I (samples) before peak detection AND for the
        # trend curve drawn on the plots
        "smooth_window": 401,
        # minimum separation between peaks, in samples (post-downsample scale)
        "min_distance": 30,
        # minimum prominence of a peak, in I units (volts). None -> auto (10% of range)
        "prominence": None,
    },

    # --- calibration ------------------------------------------------------ #
    # recorded V * factor = true accelerating volts. Run 1 was recorded at a
    # ~2x different scope scale than Runs 2-6, so calibration is PER RUN.
    # Fill in real factors once known (derive from scope gain/divider settings).
    # "default" applies to any run not listed explicitly.
    "v_calibration": {
        "default": 10.0,
        # "Run 2": ...,
    },

    # 1-sigma RELATIVE uncertainty of the V calibration factor (set by the
    # multimeter used to fix the scope gain). Propagates as a SYSTEMATIC on DV:
    #   sigma_cal = v_calibration_rel_err * DV
    # Unlike the fit error it does NOT shrink with more data points -- it is the
    # accuracy floor. e.g. 0.005 = 0.5% DMM accuracy. 0.0 disables it.
    "v_calibration_rel_err": 0.0,

    # --- expected physics (combined-fit annotation only) ------------------ #
    # element symbol and its literature Franck-Hertz peak spacing (V), shown as
    # the reference line in the combined-fit plot. Hg ~ 4.9 V, Ne ~ 18.7 V.
    "element": "Hg",
    "literature_dv": 4.9,

    # --- uncertainty propagation ------------------------------------------ #
    # Per-peak position uncertainty is estimated by BOOTSTRAP: the denoise +
    # random-sampling stage is repeated n_bootstrap times (resampling the cleaned
    # points with replacement), and the spread of each peak's position is its
    # 1-sigma. That sigma is then carried into the shared-slope fit via WEIGHTED
    # least squares, so the cleaning + sampling uncertainty propagates into DV.
    # 0 disables (fit falls back to unweighted residual-based errors).
    "uncertainty": {
        "n_bootstrap": 200,
        "seed": 12345,
    },

    # --- peak indexing ---------------------------------------------------- #
    # n assigned to each run's FIRST detected peak in the V = n*DV + V0 fit.
    # Run 1's sweep started above the first maximum, so its first detected peak
    # is really the 2nd peak -> index it from n=2.
    "peak_index": {
        "default": 1,
    },

    # --- output ----------------------------------------------------------- #
    "save_dir": "outputs",
    "show": False,                   # plt.show() in addition to saving
}


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class PipelineResult:
    run: str
    raw: pd.DataFrame
    processed: pd.DataFrame
    peak_idx: np.ndarray
    peaks_v: np.ndarray
    peaks_i: np.ndarray
    spacings_recorded: np.ndarray
    spacings_calibrated: np.ndarray
    v_factor: float = 1.0
    peaks_v_err: np.ndarray = None        # per-peak 1-sigma (recorded V), bootstrap
    spacings_err: np.ndarray = None       # per-spacing 1-sigma (recorded V), bootstrap
    config: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Run: {self.run}",
            f"  raw points       : {len(self.raw)}",
            f"  after processing : {len(self.processed)} "
            f"({100*len(self.processed)/max(1,len(self.raw)):.1f}%)",
            f"  peaks found      : {len(self.peaks_v)}",
        ]
        if len(self.peaks_v):
            if self.peaks_v_err is not None and len(self.peaks_v_err):
                vs = ", ".join(f"{v:.3f}+/-{e:.3f}"
                               for v, e in zip(self.peaks_v, self.peaks_v_err))
            else:
                vs = ", ".join(f"{v:.3f}" for v in self.peaks_v)
            lines.append(f"  peak V (recorded): {vs}")
        if len(self.spacings_recorded):
            if self.spacings_err is not None and len(self.spacings_err):
                sp = ", ".join(f"{s:.3f}+/-{e:.3f}"
                               for s, e in zip(self.spacings_recorded, self.spacings_err))
            else:
                sp = ", ".join(f"{s:.3f}" for s in self.spacings_recorded)
            lines.append(f"  spacings (rec. V): {sp}")
            # mean spacing uncertainty: bootstrap (per-spacing) if available, else
            # the run-to-run scatter of the spacings
            if self.spacings_err is not None and len(self.spacings_err):
                mean_err = float(np.sqrt(np.sum(self.spacings_err ** 2)) / len(self.spacings_err))
                src = "bootstrap"
            else:
                mean_err = (self.spacings_recorded.std(ddof=1)
                            if len(self.spacings_recorded) > 1 else 0.0)
                src = "scatter"
            lines.append(
                f"  mean spacing     : {self.spacings_recorded.mean():.3f} "
                f"+/- {mean_err:.3f} V (recorded, {src})"
            )
            if self.v_factor != 1.0:
                lines.append(
                    f"  mean spacing     : {self.spacings_calibrated.mean():.3f} "
                    f"+/- {mean_err*self.v_factor:.3f} V (calibrated, x{self.v_factor})"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Loading                                                                  #
# --------------------------------------------------------------------------- #

def _find_header_row(path: str) -> int:
    """Locate the data header line (the one with both 'Second' and 'Value').

    Works across scope export formats (Siglent: line 12; Tektronix: line 19),
    where metadata rows precede the real columns.
    """
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            fields = [c.strip() for c in line.split(",")]
            if "Second" in fields and "Value" in fields:
                return i
            if i > 100:
                break
    return 0  # fall back to top if not found


def _load_channel(path: str, skip_rows: int = None) -> pd.DataFrame:
    """Read one channel file -> DataFrame with numeric Second, Value columns.

    Auto-detects the header row, then selects Second/Value by COLUMN POSITION
    (read header=None). This handles both the Siglent layout (data in cols 0-1)
    and the Tektronix layout (data in cols 3-4 with metadata interleaved and
    trailing commas that would otherwise misalign a name-based selection).
    """
    hdr = _find_header_row(path)
    with open(path, "r", errors="ignore") as f:
        fields = None
        for i, line in enumerate(f):
            if i == hdr:
                fields = [c.strip() for c in line.split(",")]
                break
    sec_i, val_i = fields.index("Second"), fields.index("Value")
    raw = pd.read_csv(path, skiprows=hdr + 1, header=None, skipinitialspace=True)
    sec = pd.to_numeric(raw.iloc[:, sec_i], errors="coerce")
    val = pd.to_numeric(raw.iloc[:, val_i], errors="coerce")
    return pd.DataFrame({"Second": sec, "Value": val}).dropna().reset_index(drop=True)


def _find_channel_file(run_dir: str, configured: str, which: int) -> str:
    """Resolve a channel file: use the configured name if present, else search
    for a CH1/CH2 (or C1/C2) file so different scopes' naming both work."""
    p = os.path.join(run_dir, configured)
    if os.path.isfile(p):
        return p
    import glob as _glob
    patterns = [f"*CH{which}*", f"*C{which}_*", f"*_C{which}.*", f"*CH{which}.*"]
    for pat in patterns:
        hits = sorted(h for h in _glob.glob(os.path.join(run_dir, pat))
                      if h.lower().endswith(".csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no channel-{which} CSV found in {run_dir!r} "
        f"(looked for {configured!r} and *CH{which}* patterns)")


def load_run(run: str, cfg: dict = CONFIG) -> pd.DataFrame:
    """Load one run directory into a combined DataFrame with columns Second, V, I."""
    run_dir = run if os.path.isdir(run) else os.path.join(cfg["data_root"], run)
    vpath = _find_channel_file(run_dir, cfg["channel_v"], 1)
    ipath = _find_channel_file(run_dir, cfg["channel_i"], 2)
    v = _load_channel(vpath)
    i = _load_channel(ipath)
    n = min(len(v), len(i))  # guard against unequal lengths between channels
    df = pd.DataFrame({"Second": v["Second"].to_numpy()[:n],
                       "V": v["Value"].to_numpy()[:n],
                       "I": i["Value"].to_numpy()[:n]})
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Noise removal                                                            #
# --------------------------------------------------------------------------- #

def _running_average(y: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window < 2 or len(y) == 0:
        return y.copy()
    window = min(window, len(y))   # clamp so it works on small (e.g. Neon) datasets
    kernel = np.ones(window) / window
    # 'same' length, edge-padded so endpoints are not pulled toward zero
    pad = window // 2
    ypad = np.pad(y, pad, mode="edge")
    return np.convolve(ypad, kernel, mode="valid")[: len(y)]


def denoise(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Reject noise on I using a running-average deviation rule.

    See CONFIG["denoise"] for modes. Returns a filtered copy (time order kept).
    """
    c = cfg["denoise"]
    if not c.get("enabled", True):
        return df.copy()

    y = df["I"].to_numpy(dtype=float)
    avg = _running_average(y, c["window"])
    dev = y - avg

    # threshold is a percent of the full I range -> absolute I units
    irange = float(np.nanmax(y) - np.nanmin(y)) or 1.0
    thresh = c["threshold_pct"] / 100.0 * irange

    mode = c.get("mode", "narrow_spikes")
    if mode == "above":
        flagged = dev > thresh
    elif mode == "symmetric":
        flagged = np.abs(dev) > thresh
    elif mode == "narrow_spikes":
        flagged_any = np.abs(dev) > thresh
        flagged = _narrow_runs_only(flagged_any, c["max_spike_width"])
    else:
        raise ValueError(f"unknown denoise mode: {mode!r}")

    keep = ~flagged
    return df.loc[keep].reset_index(drop=True)


def _narrow_runs_only(flagged: np.ndarray, max_width: int) -> np.ndarray:
    """Keep a flag only if it belongs to a contiguous run no longer than max_width.

    Broad excursions (real FH peaks) span many samples and are de-flagged so they
    survive; isolated narrow spikes stay flagged for removal.
    """
    out = np.zeros_like(flagged, dtype=bool)
    n = len(flagged)
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        j = i
        while j < n and flagged[j]:
            j += 1
        if (j - i) <= max_width:
            out[i:j] = True
        i = j
    return out


# --------------------------------------------------------------------------- #
# 3. Downsampling                                                             #
# --------------------------------------------------------------------------- #

def downsample(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    c = cfg["downsample"]
    if not c.get("enabled", True):
        return df.copy()

    if c["method"] == "random":
        n_keep = max(1, int(round(len(df) * c["fraction"])))
        rng = np.random.default_rng(c["seed"])
        idx = np.sort(rng.choice(len(df), size=n_keep, replace=False))
        out = df.iloc[idx]
    elif c["method"] == "stride":
        out = df.iloc[:: c["stride"]]
    else:
        raise ValueError(f"unknown downsample method: {c['method']!r}")

    # keep ordered by time so the curve shape / peak finding stay valid
    return out.sort_values("Second").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4 & 6. Plotting                                                             #
# --------------------------------------------------------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def smooth_curve(df: pd.DataFrame, cfg: dict = CONFIG):
    """Clean trend curve for plotting: rolling-average BOTH V and I in time order.

    Smoothing V (which is itself noisy and never denoised) removes the horizontal
    zig-zag that a raw line plot shows; smoothing I (median pre-filter + running
    average, same as peak detection) removes the vertical scatter.
    """
    c = cfg["peaks"]
    w = c["smooth_window"]
    v = df["V"].to_numpy(dtype=float)
    i = df["I"].to_numpy(dtype=float)
    mw = c.get("median_window", 0)
    iclean = median_filter(i, size=mw) if mw and mw > 1 else i
    return _running_average(v, w), _running_average(iclean, w)


def _vlabel(factor: float) -> str:
    return ("Accelerating voltage U (V, calibrated)" if factor != 1.0
            else "Accelerating voltage U (recorded V)")


def _label_vlines(ax, positions: dict, fmt, line_label: str,
                  headroom: float = 0.18) -> None:
    """Draw labeled vertical dashed lines without colliding with the title.

    Adds a headroom band at the top of the axes and places each label INSIDE
    that band (just under the top spine) with a small white backing box.
    `positions` maps key -> x; `fmt(key, x)` returns the label text.
    """
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + headroom * (ymax - ymin))
    for i, key in enumerate(sorted(positions)):
        x = positions[key]
        ax.axvline(x, color="0.25", ls="--", lw=1.0, alpha=0.7,
                   label=line_label if i == 0 else None)
        ax.text(x, 0.99, fmt(key, x), transform=ax.get_xaxis_transform(),
                va="top", ha="center", fontsize=8, color="0.2",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))


def plot_iv(df: pd.DataFrame, run: str, cfg: dict = CONFIG,
            tag: str = "iv", title: Optional[str] = None,
            trend: bool = True, save_path: Optional[str] = None) -> str:
    factor = calibration_factor(run, cfg)
    out = save_path or os.path.join(cfg["save_dir"], f"{_slug(run)}_{tag}.png")
    _ensure_dir(os.path.dirname(out) or ".")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["V"] * factor, df["I"], s=1.0, alpha=0.25, color="0.6", label="points")
    if trend:
        vs, ism = smooth_curve(df, cfg)
        ax.plot(vs * factor, ism, lw=1.6, color="C0", label="smoothed trend")
        ax.legend()
    ax.set_xlabel(_vlabel(factor))
    ax.set_ylabel("Collector current I (recorded V)")
    ax.set_title(title or f"{run}: I vs V ({tag})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def plot_peaks(df: pd.DataFrame, result: "PipelineResult", cfg: dict = CONFIG,
               save_path: Optional[str] = None) -> str:
    factor = result.v_factor
    out = save_path or os.path.join(cfg["save_dir"], f"{_slug(result.run)}_peaks.png")
    _ensure_dir(os.path.dirname(out) or ".")
    fig, ax = plt.subplots(figsize=(9, 5))

    # light scatter of actual processed points, clean smoothed trend on top
    ax.scatter(df["V"] * factor, df["I"], s=1.0, alpha=0.2, color="0.6",
               label="processed points")
    vs, ism = smooth_curve(df, cfg)
    vs = vs * factor
    ax.plot(vs, ism, lw=1.8, color="C0", label="smoothed I(V)")

    # place markers on the smoothed curve at the detected peak indices
    idx = result.peak_idx
    mx, my = (vs[idx], ism[idx]) if len(idx) else (result.peaks_v * factor, result.peaks_i)
    ax.scatter(mx, my, color="red", zorder=5, s=50, marker="x",
               label=f"maxima (n={len(result.peaks_v)})")
    for v, x, y in zip(result.peaks_v * factor, mx, my):
        ax.annotate(f"{v:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color="red")

    ax.set_xlabel(_vlabel(factor))
    ax.set_ylabel("Collector current I (recorded V)")
    ax.set_title(f"{result.run}: Franck-Hertz maxima")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def plot_overlay(results: dict, cfg: dict = CONFIG, normalize: bool = True,
                 mark_peaks: bool = True, tag: str = "overlay") -> str:
    """Overlay every run's calibrated smoothed I(V) curve on one axis.

    results : {run_name: PipelineResult}
    normalize : scale each curve's I to [0, 1] so shapes overlay despite the
                differing current scales between runs (recommended for comparing
                peak positions). Set False to keep raw recorded-V current.
    """
    _ensure_dir(cfg["save_dir"])
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")

    for k, (name, r) in enumerate(results.items()):
        vs, ism = smooth_curve(r.processed, cfg)
        vs = vs * r.v_factor
        if normalize:
            rng = float(ism.max() - ism.min()) or 1.0
            ism = (ism - ism.min()) / rng
        color = cmap(k % 10)
        ax.plot(vs, ism, lw=1.6, color=color,
                label=f"{name} (x{r.v_factor:g})")
        if mark_peaks and len(r.peak_idx):
            py = ism[r.peak_idx]
            ax.scatter(vs[r.peak_idx], py, color=color, s=30, marker="o",
                       edgecolor="k", linewidth=0.4, zorder=5)

    # vertical line at each peak's mean voltage across runs, grouped by TRUE peak
    # index n (respects per-run peak_index, e.g. Run 1 starting at n=2) so peaks
    # of the same order are averaged together, not by find-order.
    n_to_vs = {}
    for r in results.values():
        if not len(r.peaks_v):
            continue
        pv = np.sort(r.peaks_v * r.v_factor)
        start = peak_n_start(r.run, cfg)
        for j, v in enumerate(pv):
            n_to_vs.setdefault(start + j, []).append(float(v))
    means = {n: float(np.mean(v)) for n, v in n_to_vs.items()}
    _label_vlines(ax, means, fmt=lambda n, mv: f"n{n}\n{mv:.2f} V",
                  line_label="mean peak V (by n)")

    ax.set_xlabel("Accelerating voltage U (V, calibrated)")
    ax.set_ylabel("Collector current I (normalized)" if normalize
                  else "Collector current I (recorded V)")
    ax.set_title("Franck-Hertz curves - all runs (calibrated)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    out = os.path.join(cfg["save_dir"], f"all_runs_{tag}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def combined_fit(results: dict, cfg: dict = CONFIG, n_start: int = 1) -> dict:
    """Shared-slope fit: V_peak(run, n) = slope*n + offset_run.

    One common slope (= mercury excitation energy DV, with 1-sigma uncertainty)
    plus a separate intercept per run (= that run's contact-potential offset).
    This is the right model because runs differ in offset but should share DV.
    Also returns each run's independent fit for reference.
    """
    run_names = [name for name, r in results.items() if len(r.peaks_v) >= 2]
    if not run_names:
        counts = {name: len(r.peaks_v) for name, r in results.items()}
        raise ValueError(
            "combined_fit: no run has >=2 detected peaks "
            f"(peaks per run: {counts}). The peak-finding config is likely wrong "
            "for this dataset -- e.g. smooth_window/downsample too large for a "
            "small (Neon) record, or prominence too high. Retune CONFIG['peaks'].")
    idx_map = {name: j for j, name in enumerate(run_names)}
    R = len(run_names)

    rows, V, sig, n_per_run, sig_per_run = [], [], [], {}, {}
    have_sigma = True
    for name in run_names:
        r = results[name]
        order = np.argsort(r.peaks_v)
        pv = (r.peaks_v * r.v_factor)[order]
        # per-peak 1-sigma (calibrated); fall back if bootstrap unavailable
        if r.peaks_v_err is not None and len(r.peaks_v_err) == len(pv):
            sv = (np.asarray(r.peaks_v_err) * r.v_factor)[order]
        else:
            sv = np.full(len(pv), np.nan); have_sigma = False
        n = np.arange(len(pv)) + peak_n_start(r.run, cfg, n_start)
        n_per_run[name] = (n, pv); sig_per_run[name] = sv
        for ni, vi, si in zip(n, pv, sv):
            row = [float(ni)] + [0.0] * R   # [shared slope | per-run intercepts]
            row[1 + idx_map[name]] = 1.0
            rows.append(row); V.append(vi); sig.append(si)
    A = np.array(rows); V = np.array(V); sig = np.array(sig)

    # guard against zero/NaN sigma -> use weighting only if all are finite & >0
    weighted = have_sigma and np.all(np.isfinite(sig)) and np.all(sig > 0)
    dof = max(1, len(V) - A.shape[1])
    if weighted:
        W = 1.0 / sig ** 2
        AtWA = A.T @ (A * W[:, None])
        coef = np.linalg.solve(AtWA, A.T @ (W * V))
        cov = np.linalg.inv(AtWA)                      # measurement-error propagated
        resid = V - A @ coef
        chi2 = float(np.sum((resid / sig) ** 2))
        chi2_dof = chi2 / dof
        slope_err_meas = float(np.sqrt(cov[0, 0]))     # if bootstrap sigmas are the whole story
        # Birge-ratio inflation: when chi2/dof > 1 the runs scatter more than the
        # bootstrap sigmas predict (run-to-run systematics), so scale up the error
        # so it reflects the ACTUAL spread, not just the sampling noise.
        birge = float(np.sqrt(max(1.0, chi2_dof)))
        slope_err = slope_err_meas * birge
        fit_mode = "weighted"
    else:
        coef, *_ = np.linalg.lstsq(A, V, rcond=None)
        resid = V - A @ coef
        cov = (float(resid @ resid) / dof) * np.linalg.inv(A.T @ A)
        slope_err = float(np.sqrt(cov[0, 0]))
        slope_err_meas = slope_err
        chi2_dof = float("nan"); birge = float("nan")
        fit_mode = "unweighted"
    slope = coef[0]
    offsets = {name: float(coef[1 + idx_map[name]]) for name in run_names}

    per_run = {}
    for name in run_names:
        n, pv = n_per_run[name]; sv = sig_per_run[name]
        w = 1.0 / sv if (weighted and np.all(np.isfinite(sv)) and np.all(sv > 0)) else None
        (s_i, b_i), cov_i = np.polyfit(n, pv, 1, w=w, cov=True)
        per_run[name] = {
            "slope": float(s_i), "slope_err": float(np.sqrt(cov_i[0, 0])),
            "intercept": offsets[name],                # shared-slope offset (for correction)
            "indep_intercept": float(b_i),
            "n": n, "v": pv, "v_err": sv,
        }

    # calibration systematic: DV scales linearly with the gain factor, so a
    # relative factor uncertainty maps 1:1 onto DV (common-mode, does not shrink
    # with N). Combined with the fit (statistical) error in quadrature.
    rel = abs(float(cfg.get("v_calibration_rel_err", 0.0)))
    slope_cal_err = rel * slope
    slope_total_err = float(np.hypot(slope_err, slope_cal_err))

    return {
        "slope": float(slope),
        "slope_err": slope_err,                 # statistical, scatter-inflated (reported)
        "slope_err_meas": float(slope_err_meas),  # pure bootstrap propagation (no inflation)
        "slope_cal_err": float(slope_cal_err),  # calibration systematic
        "slope_total_err": slope_total_err,     # stat (+) cal in quadrature
        "rel_cal_err": rel,
        "fit_mode": fit_mode, "chi2_dof": chi2_dof, "birge": birge,
        "offsets": offsets, "intercept": float(np.mean(list(offsets.values()))),
        "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
        "n_all": np.concatenate([n_per_run[k][0] for k in run_names]),
        "v_all": np.concatenate([n_per_run[k][1] for k in run_names]),
        "per_run": per_run, "n_start": n_start,
    }


def plot_combined_fit(results: dict, cfg: dict = CONFIG, n_start: int = 1,
                      hg_lit: float = None, tag: str = "") -> str:
    """Scatter peak voltage vs peak number for all runs + the pooled linear fit."""
    _ensure_dir(cfg["save_dir"])
    element = cfg.get("element", "")
    lit = hg_lit if hg_lit is not None else cfg.get("literature_dv", 4.9)
    fit = combined_fit(results, cfg, n_start)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    nn = np.array([fit["n_all"].min() - 0.3, fit["n_all"].max() + 0.3])
    for k, (name, pr) in enumerate(fit["per_run"].items()):
        color = cmap(k % 10)
        yerr = pr.get("v_err")
        yerr = yerr if (yerr is not None and np.all(np.isfinite(yerr))) else None
        ax.errorbar(pr["n"], pr["v"], yerr=yerr, fmt="o", color=color, ms=6,
                    capsize=3, zorder=4, label=f"{name} (V$_0$={pr['intercept']:.3f})")
        # shared slope, this run's offset -> parallel lines
        ax.plot(nn, fit["slope"] * nn + pr["intercept"], color=color,
                lw=1.2, alpha=0.7, zorder=3)
    if fit["slope_cal_err"] > 0:
        dv_line = (f"shared $\\Delta V$ = {fit['slope']:.3f} V\n"
                   f"   $\\pm$ {fit['slope_err']:.3f} (stat) "
                   f"$\\pm$ {fit['slope_cal_err']:.3f} (cal)\n"
                   f"   = {fit['slope']:.3f} $\\pm$ {fit['slope_total_err']:.3f} V (total)")
    else:
        dv_line = f"shared $\\Delta V$ = {fit['slope']:.3f} $\\pm$ {fit['slope_err']:.3f} V"
    lit_label = f"{element} literature".strip() or "literature"
    chi = (f"  ({fit['fit_mode']}, $\\chi^2$/dof={fit['chi2_dof']:.2f})"
           if fit.get("fit_mode") == "weighted" else f"  ({fit.get('fit_mode','')})")
    txt = (f"{dv_line}\n"
           f"RMS resid = {fit['rms_resid']:.3f} V{chi}\n"
           f"({lit_label}: {lit} V)")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xlabel("Peak number n")
    ax.set_ylabel("Peak voltage (V, calibrated)")
    ax.set_title("Combined fit: peak voltage vs peak number (all runs)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    out = os.path.join(cfg["save_dir"], f"combined_fit{tag}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def plot_contact_corrected(results: dict, cfg: dict = CONFIG, normalize: bool = True,
                           per_run_offset: bool = True, n_start: int = 1,
                           tag: str = "") -> str:
    """Overlay curves after subtracting the contact-potential offset.

    Each run's voltage axis is shifted by its fitted intercept (per_run_offset=True)
    or the pooled intercept, so peak n lands on n*DV. Vertical lines mark n*DV.
    """
    _ensure_dir(cfg["save_dir"])
    fit = combined_fit(results, cfg, n_start)
    slope = fit["slope"]
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for k, (name, r) in enumerate(results.items()):
        vs, ism = smooth_curve(r.processed, cfg)
        vs = vs * r.v_factor
        off = (fit["per_run"][name]["intercept"]
               if per_run_offset and name in fit["per_run"] else fit["intercept"])
        vs = vs - off
        if normalize:
            rng = float(ism.max() - ism.min()) or 1.0
            ism = (ism - ism.min()) / rng
        ax.plot(vs, ism, lw=1.6, color=cmap(k % 10), label=name)

    # gridlines at every peak index present across runs (Run 1 may reach higher n)
    all_n = sorted({int(v) for pr in fit["per_run"].values() for v in pr["n"]})
    positions = {n: slope * n for n in all_n}
    _label_vlines(ax, positions, fmt=lambda n, x: f"n{n}",
                  line_label=f"n$\\cdot\\Delta V$ ({slope:.2f} V)")

    ax.set_xlabel("Contact-potential-corrected voltage  U - V$_0$  (V)")
    ax.set_ylabel("Collector current I (normalized)" if normalize
                  else "Collector current I (recorded V)")
    ax.set_title("Contact-potential-corrected overlay (peaks fall on n$\\cdot\\Delta V$)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    out = os.path.join(cfg["save_dir"], f"contact_corrected_overlay{tag}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def _run_name(run: str) -> str:
    return os.path.basename(run.rstrip("/\\")) or run


def _slug(run: str) -> str:
    return run.strip().replace(os.sep, "_").replace(" ", "_")


# --------------------------------------------------------------------------- #
# 5 & 7. Peak finding and spacing                                             #
# --------------------------------------------------------------------------- #

def _parabolic_vertex(y: np.ndarray, i: int) -> float:
    """Sub-sample peak location via a 3-point parabola around index i."""
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        denom = a - 2.0 * b + c
        if denom != 0:
            return i + 0.5 * (a - c) / denom
    return float(i)


def find_fh_peaks(df: pd.DataFrame, cfg: dict = CONFIG):
    """Find Franck-Hertz maxima in I (time/sample order). Returns (idx, V, I).

    Peak VOLTAGES are read off the SMOOTHED V curve at the sub-sample (parabolic)
    peak location, NOT the raw V sample. The raw V is quantized by the scope
    (e.g. 0.2 V on the Neon scope), which would snap every peak onto that grid and
    make spacings artificially round; smoothing+interpolation removes that.
    """
    c = cfg["peaks"]
    y = df["I"].to_numpy(dtype=float)
    v = df["V"].to_numpy(dtype=float)
    mw = c.get("median_window", 0)
    yclean = median_filter(y, size=mw) if mw and mw > 1 else y
    ysm = _running_average(yclean, c["smooth_window"])
    vsm = _running_average(v, c["smooth_window"])   # de-quantised voltage axis

    prominence = c["prominence"]
    if prominence is None:
        prominence = 0.10 * (float(np.nanmax(ysm) - np.nanmin(ysm)) or 1.0)

    idx, _ = find_peaks(ysm, distance=max(1, c["min_distance"]),
                        prominence=prominence)
    grid = np.arange(len(vsm))
    sub = np.array([_parabolic_vertex(ysm, i) for i in idx]) if len(idx) else np.array([])
    peaks_v = np.interp(sub, grid, vsm) if len(sub) else np.array([])
    peaks_i = y[idx]
    return idx, peaks_v, peaks_i


def peak_spacings(peaks_v: np.ndarray) -> np.ndarray:
    if len(peaks_v) < 2:
        return np.array([])
    return np.diff(np.sort(peaks_v))


def bootstrap_peak_uncertainty(den: pd.DataFrame, cfg: dict, n_nominal: int):
    """Bootstrap the sampling stage to get per-peak and per-spacing 1-sigma.

    Resamples the denoised points WITH REPLACEMENT (size = the downsample target,
    mimicking the random-sampling step), re-finds the peaks, and takes the spread
    across replicates. Captures the uncertainty injected by cleaning + sampling +
    peak localisation. Returns (peaks_err, spacings_err) in RECORDED volts, or
    (None, None) if disabled / too few valid replicates.
    """
    u = cfg.get("uncertainty", {})
    n_boot = int(u.get("n_bootstrap", 0))
    if n_boot <= 0 or n_nominal < 1 or len(den) < 3:
        return None, None

    ds = cfg["downsample"]
    size = (max(1, int(round(len(den) * ds["fraction"])))
            if ds.get("enabled", True) else len(den))
    base = int(u.get("seed", 12345))
    cols = den[["Second", "V", "I"]].to_numpy(dtype=float)

    peak_sets, spac_sets = [], []
    for b in range(n_boot):
        rng = np.random.default_rng(base + b)
        sel = rng.integers(0, len(cols), size=size)
        sub = cols[sel]
        sub = sub[np.argsort(sub[:, 0])]                 # keep time order
        d2 = pd.DataFrame({"Second": sub[:, 0], "V": sub[:, 1], "I": sub[:, 2]})
        _, pv, _ = find_fh_peaks(d2, cfg)
        if len(pv) == n_nominal:
            pv = np.sort(pv)
            peak_sets.append(pv)
            spac_sets.append(np.diff(pv))
    if len(peak_sets) < max(5, n_boot // 10):
        return None, None
    peaks_err = np.std(np.array(peak_sets), axis=0, ddof=1)
    spac_err = (np.std(np.array(spac_sets), axis=0, ddof=1)
                if n_nominal >= 2 else np.array([]))
    return peaks_err, spac_err


def calibration_factor(run: str, cfg: dict = CONFIG) -> float:
    """Resolve the recorded-V -> true-V factor for a run (per-run, with default)."""
    cal = cfg.get("v_calibration", 1.0)
    if isinstance(cal, dict):
        return float(cal.get(os.path.basename(run.rstrip(os.sep)),
                             cal.get("default", 1.0)))
    return float(cal)


def peak_n_start(run: str, cfg: dict = CONFIG, default: int = 1) -> int:
    """Peak index assigned to a run's first detected peak (per-run, with default)."""
    pi = cfg.get("peak_index", {})
    if isinstance(pi, dict):
        return int(pi.get(os.path.basename(run.rstrip(os.sep)),
                          pi.get("default", default)))
    return int(default)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run_pipeline(run: str, cfg: dict = CONFIG, plots: bool = True) -> PipelineResult:
    raw = load_run(run, cfg)

    den = denoise(raw, cfg)                 # denoised only (pre-downsample)
    proc = downsample(den, cfg)             # denoised + downsampled

    idx, pv, pi = find_fh_peaks(proc, cfg)
    sp = peak_spacings(pv)
    cal = calibration_factor(run, cfg)

    # bootstrap the cleaning+sampling to get per-peak / per-spacing 1-sigma
    peaks_err, spac_err = bootstrap_peak_uncertainty(den, cfg, len(pv))

    result = PipelineResult(
        run=run, raw=raw, processed=proc,
        peak_idx=idx, peaks_v=pv, peaks_i=pi,
        spacings_recorded=sp, spacings_calibrated=sp * cal,
        v_factor=cal, peaks_v_err=peaks_err, spacings_err=spac_err,
        config=dict(cfg),
    )

    if plots:
        # all four graphs into outputs/<run name>/
        name = _run_name(run)
        rdir = os.path.join(cfg["save_dir"], name)
        plot_iv(raw,  run, cfg, tag="raw",       title=f"{name}: raw I vs V",
                save_path=os.path.join(rdir, "1_raw.png"))
        plot_iv(den,  run, cfg, tag="denoised",  title=f"{name}: denoised I vs V",
                save_path=os.path.join(rdir, "2_denoised.png"))
        plot_iv(proc, run, cfg, tag="processed", title=f"{name}: processed I vs V",
                save_path=os.path.join(rdir, "3_processed.png"))
        plot_peaks(proc, result, cfg, save_path=os.path.join(rdir, "4_peaks.png"))

    return result


if __name__ == "__main__":
    import sys
    run = sys.argv[1] if len(sys.argv) > 1 else "Run 1"
    res = run_pipeline(run)
    print(res.summary())
    print(f"\nPlots written to: {CONFIG['save_dir']}/")
