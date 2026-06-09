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
    "data_root": "Mercury Data",
    "skip_rows": 11,                 # scope CSV header rows before "Second,Value"
    "channel_v": "SDS824X_HD_CSV_C1_1.csv",   # -> V (accelerating voltage)
    "channel_i": "SDS824X_HD_CSV_C2_1.csv",   # -> I (collector current)

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
        "Run 1"  : 5.0,
        # "Run 2": ...,
    },

    # --- peak indexing ---------------------------------------------------- #
    # n assigned to each run's FIRST detected peak in the V = n*DV + V0 fit.
    # Run 1's sweep started above the first maximum, so its first detected peak
    # is really the 2nd peak -> index it from n=2.
    "peak_index": {
        "default": 1,
        "Run 1"  : 2,
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
            vs = ", ".join(f"{v:.3f}" for v in self.peaks_v)
            lines.append(f"  peak V (recorded): {vs}")
        if len(self.spacings_recorded):
            sp = ", ".join(f"{s:.3f}" for s in self.spacings_recorded)
            lines.append(f"  spacings (rec. V): {sp}")
            lines.append(
                f"  mean spacing     : {self.spacings_recorded.mean():.3f} "
                f"+/- {self.spacings_recorded.std(ddof=1) if len(self.spacings_recorded)>1 else 0:.3f} V (recorded)"
            )
            if self.v_factor != 1.0:
                lines.append(
                    f"  mean spacing     : {self.spacings_calibrated.mean():.3f} V "
                    f"(calibrated, x{self.v_factor})"
                )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Loading                                                                  #
# --------------------------------------------------------------------------- #

def _load_channel(path: str, skip_rows: int) -> pd.DataFrame:
    # skip_rows metadata lines, then the next line ("Second,Value") is the header
    return pd.read_csv(path, skiprows=skip_rows)


def load_run(run: str, cfg: dict = CONFIG) -> pd.DataFrame:
    """Load one run directory into a combined DataFrame with columns Second, V, I."""
    run_dir = run if os.path.isdir(run) else os.path.join(cfg["data_root"], run)
    v = _load_channel(os.path.join(run_dir, cfg["channel_v"]), cfg["skip_rows"])
    i = _load_channel(os.path.join(run_dir, cfg["channel_i"]), cfg["skip_rows"])
    df = pd.DataFrame({"Second": v["Second"], "V": v["Value"], "I": i["Value"]})
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Noise removal                                                            #
# --------------------------------------------------------------------------- #

def _running_average(y: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return y.copy()
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
        ax.annotate(f"{v:.2f}", (x, y), textcoords="offset points",
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
    for i, n in enumerate(sorted(n_to_vs)):
        mv = float(np.mean(n_to_vs[n]))
        ax.axvline(mv, color="0.25", ls="--", lw=1.0, alpha=0.7,
                   label="mean peak V (by n)" if i == 0 else None)
        ax.text(mv, 1.01, f"n{n}\n{mv:.2f}", va="bottom", ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8, color="0.25")

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
    idx_map = {name: j for j, name in enumerate(run_names)}
    R = len(run_names)

    rows, V, n_per_run = [], [], {}
    for name in run_names:
        pv = np.sort(results[name].peaks_v * results[name].v_factor)
        n = np.arange(len(pv)) + peak_n_start(results[name].run, cfg, n_start)
        n_per_run[name] = (n, pv)
        for ni, vi in zip(n, pv):
            row = [float(ni)] + [0.0] * R   # [shared slope | per-run intercepts]
            row[1 + idx_map[name]] = 1.0
            rows.append(row); V.append(vi)
    A = np.array(rows); V = np.array(V)

    coef, *_ = np.linalg.lstsq(A, V, rcond=None)
    slope = coef[0]
    offsets = {name: float(coef[1 + idx_map[name]]) for name in run_names}

    resid = V - A @ coef
    dof = max(1, len(V) - A.shape[1])
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    slope_err = float(np.sqrt(cov[0, 0]))

    per_run = {}
    for name in run_names:
        n, pv = n_per_run[name]
        (s_i, b_i), cov_i = np.polyfit(n, pv, 1, cov=True)
        per_run[name] = {
            "slope": float(s_i), "slope_err": float(np.sqrt(cov_i[0, 0])),
            "intercept": offsets[name],                # shared-slope offset (for correction)
            "indep_intercept": float(b_i),
            "n": n, "v": pv,
        }

    return {
        "slope": float(slope), "slope_err": slope_err,
        "offsets": offsets, "intercept": float(np.mean(list(offsets.values()))),
        "rms_resid": float(np.sqrt(np.mean(resid ** 2))),
        "n_all": np.concatenate([n_per_run[k][0] for k in run_names]),
        "v_all": np.concatenate([n_per_run[k][1] for k in run_names]),
        "per_run": per_run, "n_start": n_start,
    }


def plot_combined_fit(results: dict, cfg: dict = CONFIG, n_start: int = 1,
                      hg_lit: float = 4.9, tag: str = "") -> str:
    """Scatter peak voltage vs peak number for all runs + the pooled linear fit."""
    _ensure_dir(cfg["save_dir"])
    fit = combined_fit(results, cfg, n_start)
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    nn = np.array([fit["n_all"].min() - 0.3, fit["n_all"].max() + 0.3])
    for k, (name, pr) in enumerate(fit["per_run"].items()):
        color = cmap(k % 10)
        ax.scatter(pr["n"], pr["v"], color=color, s=40, zorder=4,
                   label=f"{name} (V$_0$={pr['intercept']:.2f})")
        # shared slope, this run's offset -> parallel lines
        ax.plot(nn, fit["slope"] * nn + pr["intercept"], color=color,
                lw=1.2, alpha=0.7, zorder=3)
    txt = (f"shared $\\Delta V$ = {fit['slope']:.3f} $\\pm$ {fit['slope_err']:.3f} V\n"
           f"RMS resid = {fit['rms_resid']:.3f} V\n"
           f"(Hg literature: {hg_lit} V)")
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
    for j, n in enumerate(all_n):
        ax.axvline(slope * n, color="0.25", ls="--", lw=1.0, alpha=0.7,
                   label=f"n$\\cdot\\Delta V$ ({slope:.2f} V)" if j == 0 else None)
        ax.text(slope * n, 1.01, f"{n}", va="bottom", ha="center",
                transform=ax.get_xaxis_transform(), fontsize=8, color="0.25")

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

def find_fh_peaks(df: pd.DataFrame, cfg: dict = CONFIG):
    """Find Franck-Hertz maxima in I (time/sample order). Returns (idx, V, I)."""
    c = cfg["peaks"]
    y = df["I"].to_numpy(dtype=float)
    mw = c.get("median_window", 0)
    yclean = median_filter(y, size=mw) if mw and mw > 1 else y
    ysm = _running_average(yclean, c["smooth_window"])

    prominence = c["prominence"]
    if prominence is None:
        prominence = 0.10 * (float(np.nanmax(ysm) - np.nanmin(ysm)) or 1.0)

    idx, _ = find_peaks(ysm, distance=max(1, c["min_distance"]),
                        prominence=prominence)
    peaks_v = df["V"].to_numpy()[idx]
    peaks_i = y[idx]
    return idx, peaks_v, peaks_i


def peak_spacings(peaks_v: np.ndarray) -> np.ndarray:
    if len(peaks_v) < 2:
        return np.array([])
    return np.diff(np.sort(peaks_v))


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

    result = PipelineResult(
        run=run, raw=raw, processed=proc,
        peak_idx=idx, peaks_v=pv, peaks_i=pi,
        spacings_recorded=sp, spacings_calibrated=sp * cal,
        v_factor=cal, config=dict(cfg),
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
