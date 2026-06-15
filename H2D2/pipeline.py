"""
H2D2 (hydrogen/deuterium Balmer-alpha) spectroscopy pipeline.

The HR4C2500 spectrometer records the H-alpha / D-alpha doublet near 656 nm as a
pair of barely-resolved lines (isotope shift ~0.18 nm). The single-channel peak
maximum is a poor line-center estimate at this resolution, so we instead FIT the
doublet and read the centers off the fit. The handout's emphasis is on the
SEPARATION between the two lines, which is far more robust than either absolute
wavelength (a constant calibration offset cancels in the difference).

Pipeline stages (each toggleable via CONFIG):
    1. load        - read an Ocean Optics text file -> (wavelength, intensity)
    2. calibrate   - optional linear wavelength correction (offset + scale)
    3. crop        - restrict to the doublet window before fitting
    4. fit         - least-squares fit of the two-Gaussian + linear-background model
    5. plot_fit    - data + fitted curve + the two components + residuals
    6. extract     - lambda_H, lambda_D, and Delta lambda = lambda_H - lambda_D

Model (CONFIG-locked physics):
    I(lambda) = B + m*lambda
              + A_H * exp(-(lambda - lambda_H)^2 / (2 sigma_H^2))
              + A_D * exp(-(lambda - lambda_D)^2 / (2 sigma_D^2))

i.e. a sloped background plus two Gaussian peaks. By convention the longer-
wavelength line is hydrogen (H-alpha, ~656.28 nm) and the shorter is deuterium
(D-alpha, ~656.10 nm), so Delta lambda = lambda_H - lambda_D > 0.

Conventions match the Franck-Hertz pipeline in ../Franck-Hertz/pipeline.py:
CONFIG holds every knob, run_pipeline() drives one file, and a combine step
pools the per-run results into a single reported value with uncertainty.
"""

from __future__ import annotations

import os
import glob
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #

CONFIG = {
    # --- 1. loading ------------------------------------------------------- #
    "data_root": "H2D2 Data",
    # the Ocean Optics export prints metadata until this marker line; the
    # numeric "wavelength<TAB>intensity" rows follow it.
    "header_marker": "Begin Spectral Data",

    # --- 2. calibration --------------------------------------------------- #
    # wavelength_true = offset + scale * wavelength_recorded.
    # Default is the identity (no correction). The Neon calibration spectra in
    # "Neon Calibration Data" can be used to derive (offset, scale); note that a
    # pure offset cancels in Delta lambda, so the separation is largely immune to
    # calibration error -- only the absolute lambda_H / lambda_D depend on it.
    "calibration": {
        "enabled": False,
        "offset": 0.0,     # nm
        "scale": 1.0,      # dimensionless
        # OceanView pixel -> wavelength calibration of the HR4000 (used for the
        # wavelength-calibration UNCERTAINTY, not to recompute the axis -- the
        # files already store wavelength = poly(pixel) to within the readout):
        #     lambda(p) = c0 + c1*p + c2*p^2 + c3*p^3 ,  p = pixel index (0..N-1)
        "poly": {
            "c0": 629.6664627,      # intercept (nm)
            "c1": 0.018337985,      # nm / pixel
            "c2": -1.22763e-06,     # nm / pixel^2
            "c3": 0.0,              # nm / pixel^3
            "r_squared": 0.99999993,
            "n_pixels": 3648,       # Toshiba TCD1304 linear CCD (HR4000)
        },
    },

    # --- readout / quantization uncertainty ------------------------------- #
    # 1-sigma taken as +/- 1/2 of the last recorded decimal place:
    #   wavelengths printed to 1e-5 nm  -> 0.5e-5 = 5e-6 nm
    #   intensities printed to 1e-2 cts -> 0.5e-2 = 5e-3 counts
    # The wavelength readout is the SAME pixel grid every trial (common-mode);
    # the intensity readout is independent per sample and enters the fit.
    "readout": {
        "wavelength_nm": 5e-6,
        "intensity": 5e-3,
    },

    # --- 3. crop ---------------------------------------------------------- #
    # Fit window (nm) around the doublet. Wide enough to pin the background on
    # both sides, tight enough to exclude unrelated lines / cosmic-ray spikes.
    "fit_window": (654.5, 656.4),

    # --- 4. fit ----------------------------------------------------------- #
    "fit": {
        # initial-guess controls. Peak centers are auto-detected from the two
        # most prominent maxima in the window (see initial_guess); these are the
        # fallbacks if detection fails, plus the shared starting width.
        "sigma_guess": 0.07,            # nm, ~FWHM/2.355 for this instrument
        "lambda_H_guess": 656.28,       # nm, H-alpha (air); fallback only
        "lambda_D_guess": 656.10,       # nm, D-alpha (air); fallback only
        # bounds keep the fit physical: positive amplitudes/widths, centers
        # inside the window, sigma below sigma_max so the two peaks can't merge
        # into one broad blob.
        "sigma_min": 0.01,              # nm
        "sigma_max": 0.30,              # nm
        "maxfev": 20000,
    },

    # --- expected physics (annotation only) ------------------------------- #
    # literature H-alpha / D-alpha and their separation (nm), shown on plots.
    "lambda_H_lit": 656.279,
    "lambda_D_lit": 656.100,
    "dlambda_lit": 0.179,

    # --- output ----------------------------------------------------------- #
    "save_dir": "H2D2 Processed Outputs",
    "show": False,                      # plt.show() in addition to saving
}


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #

@dataclass
class FitResult:
    run: str
    wavelength: np.ndarray            # full (calibrated) spectrum
    intensity: np.ndarray
    fit_wl: np.ndarray                # cropped window actually fitted
    fit_i: np.ndarray
    popt: np.ndarray                  # [B, m, A_H, lamH, sigH, A_D, lamD, sigD]
    pcov: np.ndarray
    lambda_H: float
    lambda_D: float
    dlambda: float                    # lambda_H - lambda_D
    # reported 1-sigma = quadrature of the fit, calibration and readout terms
    lambda_H_err: float
    lambda_D_err: float
    dlambda_err: float
    chi2_dof: float
    # --- uncertainty breakdown (so the budget is transparent) ------------- #
    # statistical, from the weighted fit covariance (independent per trial):
    lambda_H_err_fit: float = float("nan")
    lambda_D_err_fit: float = float("nan")
    dlambda_err_fit: float = float("nan")
    # wavelength-calibration systematic (common-mode across trials):
    lambda_H_err_cal: float = float("nan")
    lambda_D_err_cal: float = float("nan")
    dlambda_err_cal: float = float("nan")
    rel_cal: float = 0.0              # fractional calibration uncertainty
    success: bool = True
    config: dict = field(default_factory=dict)

    @property
    def params(self) -> dict:
        B, m, A_H, lamH, sigH, A_D, lamD, sigD = self.popt
        return {"B": B, "m": m,
                "A_H": A_H, "lambda_H": lamH, "sigma_H": sigH,
                "A_D": A_D, "lambda_D": lamD, "sigma_D": sigD}

    def summary(self) -> str:
        p = self.params
        lines = [
            f"Run: {self.run}",
            f"  fit window       : {self.fit_wl.min():.3f} - {self.fit_wl.max():.3f} nm "
            f"({len(self.fit_wl)} pts)",
            f"  converged        : {self.success}   chi2/dof = {self.chi2_dof:.2f}",
            f"  lambda_H (H-alpha): {self.lambda_H:.4f} +/- {self.lambda_H_err:.4f} nm "
            f"(fit {self.lambda_H_err_fit:.4f}, cal {self.lambda_H_err_cal:.4f})",
            f"  lambda_D (D-alpha): {self.lambda_D:.4f} +/- {self.lambda_D_err:.4f} nm "
            f"(fit {self.lambda_D_err_fit:.4f}, cal {self.lambda_D_err_cal:.4f})",
            f"  Delta lambda      : {self.dlambda:.4f} +/- {self.dlambda_err:.4f} nm "
            f"(fit {self.dlambda_err_fit:.4f}, cal {self.dlambda_err_cal:.1e})",
            f"  peak widths sigma : H {p['sigma_H']:.4f} nm,  D {p['sigma_D']:.4f} nm",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 1. Loading                                                                  #
# --------------------------------------------------------------------------- #

def _data_start(path: str, marker: str) -> int:
    """Return the line number AFTER the spectral-data marker (0-based count of
    lines to skip). Falls back to 0 if the marker is absent."""
    with open(path, "r", errors="ignore") as f:
        for i, line in enumerate(f):
            if marker in line:
                return i + 1
            if i > 200:
                break
    return 0


def load_spectrum(run: str, cfg: dict = CONFIG) -> pd.DataFrame:
    """Load one spectrum file into a DataFrame with columns wavelength, intensity.

    `run` may be a full path to a .txt file, or a bare name resolved under
    cfg["data_root"] (with or without the .txt suffix).
    """
    path = _resolve_path(run, cfg)
    skip = _data_start(path, cfg["header_marker"])
    raw = pd.read_csv(path, skiprows=skip, header=None, sep=r"\s+",
                      engine="python", names=["wavelength", "intensity"])
    wl = pd.to_numeric(raw["wavelength"], errors="coerce")
    it = pd.to_numeric(raw["intensity"], errors="coerce")
    df = pd.DataFrame({"wavelength": wl, "intensity": it}).dropna()
    df = df.sort_values("wavelength").reset_index(drop=True)
    return apply_calibration(df, cfg)


def _resolve_path(run: str, cfg: dict) -> str:
    """Find the spectrum file for `run` (path, name, or name w/o .txt)."""
    if os.path.isfile(run):
        return run
    for cand in (run, run + ".txt",
                 os.path.join(cfg["data_root"], run),
                 os.path.join(cfg["data_root"], run + ".txt")):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"no spectrum file for run={run!r} (looked in "
                            f"{cfg['data_root']!r})")


def list_runs(cfg: dict = CONFIG) -> list[str]:
    """All .txt spectra under the data root, sorted (acquisition order)."""
    return sorted(glob.glob(os.path.join(cfg["data_root"], "*.txt")))


# --------------------------------------------------------------------------- #
# 2. Calibration                                                              #
# --------------------------------------------------------------------------- #

def apply_calibration(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    """Apply the linear wavelength correction wl -> offset + scale*wl (in place
    on a copy). No-op unless CONFIG["calibration"]["enabled"]."""
    c = cfg.get("calibration", {})
    if not c.get("enabled", False):
        return df
    out = df.copy()
    out["wavelength"] = c.get("offset", 0.0) + c.get("scale", 1.0) * out["wavelength"]
    return out


def calibration_rel_error(wl_full: np.ndarray, cfg: dict = CONFIG) -> float:
    """Fractional 1-sigma of the wavelength calibration, from the OceanView fit.

    The pixel->wavelength polynomial was fit to reference lines with goodness
    R^2 (CONFIG["calibration"]["poly"]["r_squared"]). The unexplained fraction
    (1 - R^2) of the wavelength variance is the calibration's residual scatter,
    so the absolute 1-sigma accuracy of any wavelength is
        s_cal = sqrt(1 - R^2) * std(lambda)
    and the FRACTIONAL accuracy is rel = s_cal / mean(lambda). Treating the
    calibration error as a multiplicative scale (the standard model) means:
        * an absolute line center carries  sigma = rel * lambda  (~few mAA), and
        * the SEPARATION carries  sigma = rel * Delta_lambda,
    because a common additive offset cancels in lambda_H - lambda_D and only the
    scale survives -- which is why Delta lambda is essentially calibration-free.
    Returns 0.0 if R^2 is missing or >= 1.
    """
    poly = cfg.get("calibration", {}).get("poly", {})
    r2 = poly.get("r_squared")
    if r2 is None or r2 >= 1.0:
        return 0.0
    wl = np.asarray(wl_full, dtype=float)
    mean = float(np.mean(wl)) or 1.0
    return float(np.sqrt(max(0.0, 1.0 - r2)) * np.std(wl) / mean)


# --------------------------------------------------------------------------- #
# 3. Crop                                                                      #
# --------------------------------------------------------------------------- #

def crop(df: pd.DataFrame, cfg: dict = CONFIG) -> pd.DataFrame:
    lo, hi = cfg["fit_window"]
    m = (df["wavelength"] >= lo) & (df["wavelength"] <= hi)
    return df.loc[m].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Model and fit                                                             #
# --------------------------------------------------------------------------- #

def two_gaussian(lmbda, B, m, A_H, lam_H, sig_H, A_D, lam_D, sig_D):
    """Sloped background plus two Gaussian peaks (the H2D2 line-shape model)."""
    bg = B + m * lmbda
    gH = A_H * np.exp(-((lmbda - lam_H) ** 2) / (2.0 * sig_H ** 2))
    gD = A_D * np.exp(-((lmbda - lam_D) ** 2) / (2.0 * sig_D ** 2))
    return bg + gH + gD


def initial_guess(wl: np.ndarray, it: np.ndarray, cfg: dict = CONFIG):
    """Build (p0, lower, upper) for curve_fit from the cropped data.

    Peak centers are taken from the two most prominent maxima in the window
    (so the fit starts near the real doublet regardless of the nominal
    calibration); amplitudes/background from the data; widths from CONFIG.
    Peak 1 is initialized at the LOWER wavelength, peak 2 at the higher, but the
    final H/D assignment is made after fitting by wavelength order.
    """
    f = cfg["fit"]
    lo, hi = float(wl.min()), float(wl.max())
    B0 = float(np.median(np.concatenate([it[:max(3, len(it)//10)],
                                          it[-max(3, len(it)//10):]])))

    # detect the two strongest peaks; prominence scaled to the data range
    rng = float(it.max() - it.min()) or 1.0
    idx, props = find_peaks(it, prominence=0.05 * rng,
                            distance=max(1, len(it) // 50))
    if len(idx) >= 2:
        top = idx[np.argsort(props["prominences"])[-2:]]
        c1, c2 = sorted(wl[top])
        a1 = float(it[top[wl[top].argmin()]] - B0)
        a2 = float(it[top[wl[top].argmax()]] - B0)
    else:  # fallback to literature-ish guesses
        c1, c2 = f["lambda_D_guess"], f["lambda_H_guess"]
        a1 = a2 = float(it.max() - B0)
    a1 = max(a1, 0.05 * rng)
    a2 = max(a2, 0.05 * rng)
    sig = f["sigma_guess"]

    #     [ B,      m,    A1,      lam1, sig1, A2,      lam2, sig2 ]
    p0 = [B0, 0.0, a1, c1, sig, a2, c2, sig]
    lower = [-np.inf, -np.inf, 0.0, lo, f["sigma_min"], 0.0, lo, f["sigma_min"]]
    upper = [np.inf, np.inf, np.inf, hi, f["sigma_max"], np.inf, hi, f["sigma_max"]]
    return p0, (lower, upper)


def readout_center_floor(wl: np.ndarray, it: np.ndarray, cfg: dict) -> float:
    """The +/- 1/2-last-decimal INTENSITY readout's direct contribution to a
    fitted line center, by linear error propagation.

    If the intensity quantization were the ONLY noise, a Gaussian peak of
    amplitude A and width sigma sampled at spacing d would locate its center to
    roughly  sigma_I * sqrt(d / (sqrt(pi) * sigma)) / A  (the Cramer-Rao-style
    floor). This is reported so the readout term is explicit; it is utterly
    dominated by the real scatter (captured in the fit error), as the budget
    shows. Returned per center (nm).
    """
    r = cfg.get("readout", {})
    sig_I = float(r.get("intensity", 0.0))
    if sig_I <= 0 or len(wl) < 2:
        return 0.0
    d = float(np.median(np.diff(np.sort(wl))))
    rng = float(it.max() - it.min()) or 1.0
    sig = float(cfg["fit"].get("sigma_guess", 0.07))
    A = 0.5 * rng                                    # rough per-peak amplitude
    return sig_I * np.sqrt(d / (np.sqrt(np.pi) * sig)) / max(A, 1.0)


def fit_spectrum(df: pd.DataFrame, run: str, cfg: dict = CONFIG) -> FitResult:
    """Crop to the doublet window and fit the two-Gaussian model.

    Uncertainty propagation (each term reported separately, summed in quadrature):
      - FIT (statistical): the model is fit with UNIFORM weighting -- correct
        here because the per-point noise is ~homoscedastic, so this is the
        unbiased (max-likelihood) estimator; weighting by the tiny readout floor
        instead would pathologically up-weight the flat baseline and bias the
        centers. The center covariance is read off pcov with absolute_sigma=False,
        so the error SCALE is set by the actual residuals (real shot noise + the
        slight non-Gaussian line wings), which is the honest statistical error.
        Delta lambda uses the full covariance var(lamH)+var(lamD)-2cov(lamH,lamD).
      - READOUT: +/- 1/2 last decimal. The wavelength readout is a per-center
        floor (sigma_wl); the intensity readout's center contribution
        (readout_center_floor) is tracked but dominated -- both fold into the
        total in quadrature.
      - CALIBRATION (systematic): the OceanView pixel->wavelength fit accuracy
        as a fractional (scale) error rel_cal; common-mode across trials, and it
        nearly cancels in Delta lambda (offset cancels, only the scale remains).
    """
    win = crop(df, cfg)
    wl = win["wavelength"].to_numpy(dtype=float)
    it = win["intensity"].to_numpy(dtype=float)
    if len(wl) < 8:
        raise ValueError(f"only {len(wl)} points in fit window {cfg['fit_window']} "
                         f"for run {run!r}; widen CONFIG['fit_window'].")

    p0, bounds = initial_guess(wl, it, cfg)
    success = True
    try:
        popt, pcov = curve_fit(two_gaussian, wl, it, p0=p0, bounds=bounds,
                               absolute_sigma=False, maxfev=cfg["fit"]["maxfev"])
    except Exception as exc:  # keep the run; flag it as not converged
        print(f"  [warn] fit failed for {run!r}: {exc}")
        popt = np.array(p0, dtype=float)
        pcov = np.full((len(p0), len(p0)), np.nan)
        success = False

    perr = np.sqrt(np.diag(pcov))

    # assign H (longer wavelength) vs D (shorter) from the fitted centers
    lam1, lam2 = popt[3], popt[6]
    i_hi, i_lo = (3, 6) if lam1 >= lam2 else (6, 3)
    lambda_H, lambda_D = popt[i_hi], popt[i_lo]
    lamH_fit, lamD_fit = perr[i_hi], perr[i_lo]
    dlambda = lambda_H - lambda_D
    # Delta lambda statistical error from the covariance of the two centers:
    #   var(lamH - lamD) = var(lamH) + var(lamD) - 2 cov(lamH, lamD)
    var = pcov[i_hi, i_hi] + pcov[i_lo, i_lo] - 2.0 * pcov[i_hi, i_lo]
    dl_fit = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else float("nan")

    # reorder popt so params property always reports H then D consistently
    B, m = popt[0], popt[1]
    A_H, sig_H = popt[i_hi - 1], popt[i_hi + 1]
    A_D, sig_D = popt[i_lo - 1], popt[i_lo + 1]
    popt_ordered = np.array([B, m, A_H, lambda_H, sig_H, A_D, lambda_D, sig_D])

    # calibration systematic (fractional scale error, common across trials)
    rel_cal = calibration_rel_error(df["wavelength"].to_numpy(dtype=float), cfg)
    lamH_cal = rel_cal * lambda_H
    lamD_cal = rel_cal * lambda_D
    dl_cal = rel_cal * abs(dlambda)        # offset cancels; only the scale survives

    # readout floors on a center: wavelength quantization (sigma_wl) and the
    # intensity quantization propagated to the center, in quadrature.
    sig_wl = float(cfg.get("readout", {}).get("wavelength_nm", 0.0))
    sig_read_I = readout_center_floor(wl, it, cfg)
    read_center = float(np.hypot(sig_wl, sig_read_I))

    def _tot(fit, cal, n_read):
        return float(np.sqrt(np.nansum([fit ** 2, cal ** 2, (n_read * read_center) ** 2])))

    lambda_H_err = _tot(lamH_fit, lamH_cal, 1.0)
    lambda_D_err = _tot(lamD_fit, lamD_cal, 1.0)
    dlambda_err = _tot(dl_fit, dl_cal, np.sqrt(2.0))

    resid = it - two_gaussian(wl, *popt)
    dof = max(1, len(wl) - len(popt))
    # noise estimate from the flat background tails (robust to the peaks)
    edge = max(3, len(it) // 10)
    noise = float(np.std(np.concatenate([resid[:edge], resid[-edge:]]))) or 1.0
    chi2_dof = float(np.sum((resid / noise) ** 2) / dof)

    return FitResult(
        run=_run_name(run),
        wavelength=df["wavelength"].to_numpy(dtype=float),
        intensity=df["intensity"].to_numpy(dtype=float),
        fit_wl=wl, fit_i=it,
        popt=popt_ordered, pcov=pcov,
        lambda_H=float(lambda_H), lambda_D=float(lambda_D), dlambda=float(dlambda),
        lambda_H_err=lambda_H_err, lambda_D_err=lambda_D_err, dlambda_err=dlambda_err,
        lambda_H_err_fit=float(lamH_fit), lambda_D_err_fit=float(lamD_fit),
        dlambda_err_fit=dl_fit,
        lambda_H_err_cal=float(lamH_cal), lambda_D_err_cal=float(lamD_cal),
        dlambda_err_cal=float(dl_cal), rel_cal=float(rel_cal),
        chi2_dof=chi2_dof, success=success, config=dict(cfg),
    )


# --------------------------------------------------------------------------- #
# 5. Plotting                                                                  #
# --------------------------------------------------------------------------- #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _run_name(run: str) -> str:
    base = os.path.basename(run.rstrip("/\\")) or run
    return base[:-4] if base.lower().endswith(".txt") else base


def _slug(run: str) -> str:
    return _run_name(run).strip().replace(os.sep, "_").replace(" ", "_")


def plot_spectrum(df: pd.DataFrame, run: str, cfg: dict = CONFIG,
                  save_path: Optional[str] = None) -> str:
    """Full recorded spectrum with the fit window shaded -- a sanity check that
    the doublet sits inside the crop."""
    out = save_path or os.path.join(cfg["save_dir"], f"{_slug(run)}_spectrum.png")
    _ensure_dir(os.path.dirname(out) or ".")
    lo, hi = cfg["fit_window"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["wavelength"], df["intensity"], lw=0.8, color="0.4")
    ax.axvspan(lo, hi, color="C1", alpha=0.15, label="fit window")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"{_run_name(run)}: recorded spectrum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def plot_fit(result: FitResult, cfg: dict = CONFIG,
             save_path: Optional[str] = None) -> str:
    """Cropped data + fitted curve + the two component Gaussians, with a
    residual panel underneath."""
    out = save_path or os.path.join(cfg["save_dir"], f"{_slug(result.run)}_fit.png")
    _ensure_dir(os.path.dirname(out) or ".")
    p = result.params
    wl, it = result.fit_wl, result.fit_i
    grid = np.linspace(wl.min(), wl.max(), 1000)
    model = two_gaussian(grid, *result.popt)
    bg = p["B"] + p["m"] * grid
    gH = bg + p["A_H"] * np.exp(-((grid - p["lambda_H"]) ** 2) / (2 * p["sigma_H"] ** 2))
    gD = bg + p["A_D"] * np.exp(-((grid - p["lambda_D"]) ** 2) / (2 * p["sigma_D"] ** 2))

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    ax.scatter(wl, it, s=10, color="0.35", alpha=0.7, label="data", zorder=2)
    ax.plot(grid, model, color="C3", lw=2.0, label="two-Gaussian fit", zorder=4)
    ax.plot(grid, gH, color="C0", lw=1.2, ls="--", label="H-alpha component")
    ax.plot(grid, gD, color="C2", lw=1.2, ls="--", label="D-alpha component")
    ax.plot(grid, bg, color="0.6", lw=1.0, ls=":", label="background")
    for lam, lab, col in ((result.lambda_H, "H", "C0"), (result.lambda_D, "D", "C2")):
        ax.axvline(lam, color=col, lw=0.8, alpha=0.6)
        ax.annotate(f"$\\lambda_{lab}$={lam:.3f}", (lam, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(3, -12),
                    fontsize=8, color=col)
    txt = (f"$\\lambda_H$ = {result.lambda_H:.4f} $\\pm$ {result.lambda_H_err:.4f} nm\n"
           f"$\\lambda_D$ = {result.lambda_D:.4f} $\\pm$ {result.lambda_D_err:.4f} nm\n"
           f"$\\Delta\\lambda$ = {result.dlambda:.4f} $\\pm$ {result.dlambda_err:.4f} nm\n"
           f"   (fit {result.dlambda_err_fit:.4f}, cal {result.dlambda_err_cal:.0e})\n"
           f"(lit. {cfg.get('dlambda_lit', float('nan')):.3f} nm)\n"
           f"$\\chi^2$/dof = {result.chi2_dof:.2f}")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_ylabel("Intensity (counts)")
    ax.set_title(f"{result.run}: H-alpha / D-alpha doublet fit")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    resid = it - two_gaussian(wl, *result.popt)
    axr.axhline(0, color="0.6", lw=0.8)
    axr.scatter(wl, resid, s=8, color="C3", alpha=0.7)
    axr.set_xlabel("Wavelength (nm)")
    axr.set_ylabel("residual")
    axr.grid(True, alpha=0.3)

    fig.savefig(out, dpi=130, bbox_inches="tight")
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def plot_combined(results: dict, cfg: dict = CONFIG, tag: str = "combined") -> str:
    """Delta lambda per run with error bars + the weighted-mean band."""
    _ensure_dir(cfg["save_dir"])
    comb = combine_runs(results, cfg)
    names = list(results.keys())
    x = np.arange(len(names))
    dl = np.array([results[n].dlambda for n in names])
    err = np.array([results[n].dlambda_err for n in names])
    err_plot = np.where(np.isfinite(err), err, 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, dl, yerr=err_plot, fmt="o", color="C3", capsize=4, zorder=4,
                label="per-run fit")
    ax.axhline(comb["dlambda"], color="C0", lw=1.6,
               label=f"weighted mean = {comb['dlambda']:.4f} $\\pm$ {comb['dlambda_err']:.4f} nm")
    ax.axhspan(comb["dlambda"] - comb["dlambda_err"],
               comb["dlambda"] + comb["dlambda_err"], color="C0", alpha=0.15)
    lit = cfg.get("dlambda_lit")
    if lit is not None:
        ax.axhline(lit, color="0.4", ls="--", lw=1.2, label=f"literature = {lit:.3f} nm")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("$\\Delta\\lambda = \\lambda_H - \\lambda_D$  (nm)")
    ax.set_title("H-D Balmer-alpha isotope shift across runs")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    out = os.path.join(cfg["save_dir"], f"{tag}_dlambda.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 6. Combine runs                                                              #
# --------------------------------------------------------------------------- #

def combine_runs(results: dict, cfg: dict = CONFIG) -> dict:
    """Pool per-trial fits into one reported value + uncertainty per quantity.

    The error sources are combined in the physically correct order:
      - STATISTICAL (the per-trial fit error) is INDEPENDENT between trials, so
        it is pooled by inverse-variance weighting, with the Birge ratio applied
        when the trials scatter more than their fit errors predict. This term
        shrinks ~1/sqrt(N) with more trials.
      - CALIBRATION is COMMON-MODE (same spectrometer / OceanView fit every
        trial), so it does NOT average down -- it is added ONCE, in quadrature,
        evaluated at the pooled value (rel_cal * value).
      - WAVELENGTH READOUT is the same pixel grid every trial (common-mode) and
        is likewise added once.
    The reported `_err` is the quadrature total; `_stat`, `_cal`, `_read` expose
    the breakdown, and the plain mean / SEM / scatter are kept for reference.
    """
    good = {k: r for k, r in results.items() if r.success}
    if not good:
        raise ValueError("combine_runs: no converged fits to combine.")

    rel_cal = float(np.mean([r.rel_cal for r in good.values()]))
    sig_wl = float(cfg.get("readout", {}).get("wavelength_nm", 0.0))

    out = {"n_runs": len(good), "rel_cal": rel_cal}
    for key, fitkey, n_read in (("lambda_H", "lambda_H_err_fit", 1.0),
                                ("lambda_D", "lambda_D_err_fit", 1.0),
                                ("dlambda", "dlambda_err_fit", np.sqrt(2.0))):
        vals = np.array([getattr(r, key) for r in good.values()], dtype=float)
        errs = np.array([getattr(r, fitkey) for r in good.values()], dtype=float)
        c = _weighted_combine(vals, errs)             # pools the statistical term
        value = c["value"]
        stat = c["error"]
        cal = rel_cal * abs(value)                    # common-mode, added once
        read = n_read * sig_wl                        # common-mode, added once
        total = float(np.sqrt(stat ** 2 + cal ** 2 + read ** 2))
        out[key] = value
        out[key + "_err"] = total        # reported 1-sigma (stat (+) cal (+) read)
        out[key + "_stat"] = stat        # pooled statistical (Birge-inflated)
        out[key + "_cal"] = cal          # calibration systematic (common-mode)
        out[key + "_read"] = read        # wavelength readout floor
        out[key + "_mean"] = c["mean"]   # plain mean, for reference
        out[key + "_sem"] = c["sem"]     # standard error of the plain mean
        out[key + "_std"] = c["std"]     # run-to-run scatter
    return out


def _weighted_combine(vals: np.ndarray, errs: np.ndarray) -> dict:
    """Inverse-variance weighted mean with Birge-ratio inflation, plus the
    plain mean / standard error for reference."""
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0

    finite = np.isfinite(errs) & (errs > 0)
    if finite.all() and n >= 1:
        w = 1.0 / errs ** 2
        wmean = float(np.sum(w * vals) / np.sum(w))
        werr = float(np.sqrt(1.0 / np.sum(w)))
        if n > 1:
            chi2 = float(np.sum(((vals - wmean) / errs) ** 2))
            birge = np.sqrt(max(1.0, chi2 / (n - 1)))  # inflate if over-scattered
            werr *= birge
        return {"value": wmean, "error": werr, "mean": mean, "sem": sem, "std": std}
    # no usable per-fit errors -> fall back to the standard error of the mean
    return {"value": mean, "error": sem, "mean": mean, "sem": sem, "std": std}


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #

def run_pipeline(run: str, cfg: dict = CONFIG, plots: bool = True) -> FitResult:
    """Load one spectrum, fit the doublet, and (optionally) write its plots."""
    df = load_spectrum(run, cfg)
    result = fit_spectrum(df, run, cfg)
    if plots:
        name = _run_name(run)
        rdir = os.path.join(cfg["save_dir"], name)
        plot_spectrum(df, run, cfg, save_path=os.path.join(rdir, "1_spectrum.png"))
        plot_fit(result, cfg, save_path=os.path.join(rdir, "2_fit.png"))
    return result


if __name__ == "__main__":
    import sys
    runs = sys.argv[1:] or list_runs()
    results = {}
    for r in runs:
        res = run_pipeline(r, plots=True)
        results[res.run] = res
        print(res.summary())
        print()
    if len(results) > 1:
        comb = combine_runs(results)
        plot_combined(results)
        print("=" * 60)
        print(f"Combined over {comb['n_runs']} runs "
              f"(rel. calibration error {comb['rel_cal']:.2e}):")
        for k, lab in (("lambda_H", "lambda_H    "), ("lambda_D", "lambda_D    "),
                       ("dlambda", "Delta lambda")):
            print(f"  {lab} = {comb[k]:.4f} +/- {comb[k+'_err']:.4f} nm "
                  f"(stat {comb[k+'_stat']:.4f}, cal {comb[k+'_cal']:.1e}, "
                  f"read {comb[k+'_read']:.1e})")
        print(f"  (plain mean Delta lambda = {comb['dlambda_mean']:.4f} "
              f"+/- {comb['dlambda_sem']:.4f} nm SEM; scatter {comb['dlambda_std']:.4f})")
    print(f"\nPlots written to: {CONFIG['save_dir']}/")
