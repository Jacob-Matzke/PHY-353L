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
from scipy.special import voigt_profile

# physical constants (SI) for the Doppler-temperature readout
_C_LIGHT = 2.99792458e8        # m/s
_K_BOLTZ = 1.380649e-23        # J/K
_AMU = 1.66053906660e-27       # kg
_M_H = 1.007825 * _AMU         # hydrogen atom mass
_M_HE = 4.002602 * _AMU        # helium atom mass


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
        "offset": 0.0,     # nm  (set by the helium offset step; ADDED to lambda)
        "offset_err": 0.0, # nm  1-sigma of the measured offset (common-mode)
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

    # --- helium wavelength-offset calibration ----------------------------- #
    # A strong He I line at a known NIST wavelength is measured in each helium
    # trial to determine the spectrometer's absolute horizontal (wavelength)
    # shift. The mean offset is then ADDED to every spectrum (via the
    # "calibration" block above), turning the H2D2 line centers into MEASURED
    # absolute wavelengths instead of quoted ones -- needed for particle-mass
    # calculations. A constant offset cancels in Delta lambda, so this does not
    # change the separation; it fixes the absolute scale and its uncertainty.
    "helium_cal": {
        "enabled": True,
        "data_root": "Helium Calibration Data",
        "nist_line": 667.81517,         # NIST He I reference wavelength (nm)
        "nist_line_err": 1e-4,          # reference uncertainty (nm); 0 to ignore
        "fit_window": (667.00, 667.55),  # window around the He line to fit
        "sigma_guess": 0.02,            # nm; the He line is near instrument-limited
    },

    # --- 3. crop ---------------------------------------------------------- #
    # Fit window (nm) around the doublet. Wide enough to pin the background on
    # both sides, tight enough to exclude unrelated lines / cosmic-ray spikes.
    "fit_window": (655.0, 657.0),

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
        # Voigt-only: Lorentzian (lifetime + pressure) HWHM start and bound.
        # The natural width is ~1e-4 nm (unresolvable here), so a free gamma
        # mostly captures pressure/instrumental wings -- see notes in fit_spectrum.
        "gamma_guess": 0.01,            # nm
        "gamma_max": 0.30,              # nm
        "maxfev": 20000,
    },

    # --- line shape ------------------------------------------------------- #
    # "voigt"   : background + two Voigt peaks (Gaussian Doppler/instrument
    #             convolved with Lorentzian lifetime/pressure); shared sigma and
    #             gamma across the H/D doublet. The physically-motivated choice.
    # "gaussian": background + two independent Gaussians (the simpler model).
    # The helium calibration line uses the SAME shape for consistency.
    "lineshape": "voigt",

    # --- instrumental skew (red-tail) correction -------------------------- #
    # Every bright line on this spectrometer shows the same one-sided RED tail
    # (detector charge-transfer trailing / scattered light). We model it as the
    # symmetric line convolved with a one-sided exponential of decay tau (nm,
    # toward longer wavelength). The neon spectrum has many clean, isolated,
    # near-instrument-limited lines, so tau is MEASURED there (measure_skew) and
    # then FIXED when fitting the helium offset line and the H2D2 doublet -- so
    # the fitted centers are the de-skewed (true) line positions.
    #   enabled=False -> tau=0 -> the plain symmetric Voigt (toggle to compare).
    "skew": {
        "enabled": True,
        "tau": None,                 # nm; filled in by measure_skew() (or set by hand)
        "neon_root": "Neon Data",
        "n_lines": 5,                # strongest neon lines used to measure tau
        "half_window": 0.30,         # nm half-window around each neon line
        "tau_guess": 0.02,           # nm
        "tau_min": 1e-3,             # nm (lower bound for the free-tau neon fit)
        "tau_max": 0.20,             # nm
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
    popt: np.ndarray                  # raw fitted parameter vector (model-dependent)
    pcov: np.ndarray
    lambda_H: float
    lambda_D: float
    dlambda: float                    # lambda_H - lambda_D
    # reported 1-sigma = quadrature of the fit, calibration and readout terms
    lambda_H_err: float
    lambda_D_err: float
    dlambda_err: float
    chi2_dof: float
    # canonical unpacked parameters (always B,m,A_H,lambda_H,sigma_H,gamma_H,...):
    pdict: dict = field(default_factory=dict)
    lineshape: str = "gaussian"
    skew_tau: float = 0.0              # applied instrumental red-tail decay (nm; 0=off)
    rms_resid: float = float("nan")    # RMS fit residual (counts; comparable across toggle)
    sigma_gauss: float = float("nan")  # shared/representative Gaussian width (nm)
    gamma_lor: float = 0.0             # shared Lorentzian HWHM (nm; 0 for gaussian)
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
        return self.pdict

    def model(self, x):
        """Evaluate the fitted curve (background + both peaks) at x."""
        p = self.pdict
        tau = p.get("tau", self.skew_tau)
        return (p["B"] + p["m"] * x
                + peak_profile(self.lineshape, x, p["A_H"], p["lambda_H"],
                               p["sigma_H"], p["gamma_H"], tau)
                + peak_profile(self.lineshape, x, p["A_D"], p["lambda_D"],
                               p["sigma_D"], p["gamma_D"], tau))

    def summary(self) -> str:
        if self.lineshape == "voigt":
            wline = (f"  widths (shared)  : Gaussian sigma {self.sigma_gauss:.4f} nm "
                     f"(FWHM {fwhm_gauss(self.sigma_gauss):.4f}), "
                     f"Lorentzian gamma {self.gamma_lor:.4f} nm "
                     f"(FWHM_V {fwhm_voigt(self.sigma_gauss, self.gamma_lor):.4f})")
        else:
            p = self.pdict
            wline = (f"  peak widths sigma : H {p['sigma_H']:.4f} nm,  "
                     f"D {p['sigma_D']:.4f} nm")
        lines = [
            f"Run: {self.run}  [{self.lineshape}]",
            f"  fit window       : {self.fit_wl.min():.3f} - {self.fit_wl.max():.3f} nm "
            f"({len(self.fit_wl)} pts)",
            f"  converged        : {self.success}   chi2/dof = {self.chi2_dof:.2f}"
            f"   RMS resid = {self.rms_resid:.1f} cts",
            f"  lambda_H (H-alpha): {self.lambda_H:.4f} +/- {self.lambda_H_err:.4f} nm "
            f"(fit {self.lambda_H_err_fit:.4f}, cal {self.lambda_H_err_cal:.4f})",
            f"  lambda_D (D-alpha): {self.lambda_D:.4f} +/- {self.lambda_D_err:.4f} nm "
            f"(fit {self.lambda_D_err_fit:.4f}, cal {self.lambda_D_err_cal:.4f})",
            f"  Delta lambda      : {self.dlambda:.4f} +/- {self.dlambda_err:.4f} nm "
            f"(fit {self.dlambda_err_fit:.4f}, cal {self.dlambda_err_cal:.1e})",
            wline,
        ]
        return "\n".join(lines)


@dataclass
class OffsetResult:
    """Result of the helium wavelength-offset calibration."""
    nist_line: float                  # reference He I wavelength (nm)
    trial_names: list                 # per-trial file names
    centers: np.ndarray               # fitted He centers, raw scale (nm)
    centers_err: np.ndarray           # per-trial fit 1-sigma (nm)
    chi2_dof: np.ndarray              # per-trial reduced chi^2
    center_combined: float            # inverse-variance mean He center (nm)
    center_combined_err: float        # pooled (Birge-inflated) 1-sigma (nm)
    offset: float                     # nist_line - center_combined  (nm, ADD to lambda)
    offset_err: float                 # total 1-sigma of the offset (nm)
    offset_scatter: float             # std of per-trial offsets (nm)
    lineshape: str = "gaussian"       # shape used for the He center fit
    sigma_he: float = float("nan")    # mean He Gaussian width sigma (nm)
    gamma_he: float = 0.0             # mean He Lorentzian HWHM gamma (nm)
    skew_tau: float = 0.0             # applied instrumental skew tau (nm; 0=off)
    fits: list = field(default_factory=list)   # (wl, it, popt) per trial, for plots

    def summary(self) -> str:
        lines = [
            f"Helium wavelength-offset calibration  [{self.lineshape}]",
            f"  NIST He I line   : {self.nist_line:.5f} nm",
            f"  trials           : {len(self.centers)}",
        ]
        for n, c, e, x in zip(self.trial_names, self.centers,
                              self.centers_err, self.chi2_dof):
            lines.append(f"    {n:28s} center {c:.4f} +/- {e:.4f} nm  "
                         f"(chi2/dof {x:.1f})  offset {self.nist_line - c:+.4f} nm")
        lines += [
            f"  combined center  : {self.center_combined:.4f} +/- "
            f"{self.center_combined_err:.4f} nm",
            f"  trial scatter    : {self.offset_scatter:.4f} nm",
            f"  OFFSET (added)   : {self.offset:+.4f} +/- {self.offset_err:.4f} nm",
        ]
        return "\n".join(lines)


@dataclass
class SkewResult:
    """Result of measuring the instrumental red-tail skew from the neon lines."""
    lineshape: str
    line_wl: np.ndarray               # fitted neon line centers (nm)
    taus: np.ndarray                  # per-line exponential tail decay tau (nm)
    sigmas: np.ndarray                # per-line Gaussian sigma (nm)
    chi2_dof: np.ndarray
    tau: float                        # adopted (median) skew tau (nm)
    tau_scatter: float                # spread of per-line tau (nm)
    n_trials: int = 0
    fits: list = field(default_factory=list)   # (x, y, popt) per line, for plots

    def summary(self) -> str:
        lines = [
            f"Instrumental skew (red-tail) measured from neon  [{self.lineshape}]",
            f"  neon trials      : {self.n_trials}",
            f"  lines used       : {len(self.taus)}",
        ]
        for w, t, x in zip(self.line_wl, self.taus, self.chi2_dof):
            lines.append(f"    {w:9.3f} nm   tau {t:.4f} nm   (chi2/dof {x:.1f})")
        lines.append(f"  ADOPTED tau      : {self.tau:.4f} +/- {self.tau_scatter:.4f} nm "
                     f"(median over lines)")
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


def load_spectrum(run: str, cfg: dict = CONFIG, calibrate: bool = True,
                  data_root: Optional[str] = None) -> pd.DataFrame:
    """Load one spectrum file into a DataFrame with columns wavelength, intensity.

    `run` may be a full path to a .txt file, or a bare name resolved under
    `data_root` (default cfg["data_root"]), with or without the .txt suffix.
    Set calibrate=False to get the RAW recorded wavelengths (used when measuring
    the helium offset, so the offset is not applied on top of itself).
    """
    path = _resolve_path(run, cfg, data_root)
    skip = _data_start(path, cfg["header_marker"])
    raw = pd.read_csv(path, skiprows=skip, header=None, sep=r"\s+",
                      engine="python", names=["wavelength", "intensity"])
    wl = pd.to_numeric(raw["wavelength"], errors="coerce")
    it = pd.to_numeric(raw["intensity"], errors="coerce")
    df = pd.DataFrame({"wavelength": wl, "intensity": it}).dropna()
    df = df.sort_values("wavelength").reset_index(drop=True)
    return apply_calibration(df, cfg) if calibrate else df


def _resolve_path(run: str, cfg: dict, data_root: Optional[str] = None) -> str:
    """Find the spectrum file for `run` (path, name, or name w/o .txt)."""
    if os.path.isfile(run):
        return run
    root = data_root or cfg["data_root"]
    for cand in (run, run + ".txt",
                 os.path.join(root, run),
                 os.path.join(root, run + ".txt")):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"no spectrum file for run={run!r} (looked in {root!r})")


def list_runs(cfg: dict = CONFIG, data_root: Optional[str] = None) -> list[str]:
    """All .txt spectra under a data root, sorted (acquisition order)."""
    return sorted(glob.glob(os.path.join(data_root or cfg["data_root"], "*.txt")))


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


def fit_helium_line(df: pd.DataFrame, cfg: dict = CONFIG):
    """Fit a single line (+ linear background) to the He I calibration line, using
    the configured shape and the APPLIED skew tau (_skew_tau).

    Returns (center, center_err, chi2_dof, sigma, gamma, popt, wl, it) on the
    wavelength scale of the supplied df (use a RAW-loaded df so the offset is
    measured against the uncorrected axis). The center is the de-skewed line
    position used for the offset.
    """
    hc = cfg["helium_cal"]
    lo, hi = hc["fit_window"]
    m = (df["wavelength"] >= lo) & (df["wavelength"] <= hi)
    wl = df.loc[m, "wavelength"].to_numpy(dtype=float)
    it = df.loc[m, "intensity"].to_numpy(dtype=float)
    if len(wl) < 6:
        raise ValueError(f"only {len(wl)} points in helium fit window "
                         f"{hc['fit_window']}; adjust CONFIG['helium_cal']['fit_window'].")

    edge = max(3, len(it) // 10)
    B0 = float(np.median(np.concatenate([it[:edge], it[-edge:]])))
    lam0 = float(wl[np.argmax(it)])
    A0 = max(float(it.max() - B0), 1.0)
    sig0 = float(hc.get("sigma_guess", 0.02))
    shape = _lineshape(cfg)
    f = cfg["fit"]
    tau = skew_tau_at(cfg, float(hc["nist_line"]))   # red-tail decay local to He line
    func = single_model(cfg, tau)
    if shape == "voigt":
        p0 = [B0, 0.0, A0, lam0, sig0, f.get("gamma_guess", 0.01)]
        bounds = ([-np.inf, -np.inf, 0.0, lo, 1e-3, 0.0],
                  [np.inf, np.inf, np.inf, hi, f["sigma_max"], f["gamma_max"]])
    else:
        p0 = [B0, 0.0, A0, lam0, sig0]
        bounds = ([-np.inf, -np.inf, 0.0, lo, 1e-3],
                  [np.inf, np.inf, np.inf, hi, f["sigma_max"]])
    popt, pcov = curve_fit(func, wl, it, p0=p0, bounds=bounds,
                           absolute_sigma=False, maxfev=f["maxfev"])
    perr = np.sqrt(np.diag(pcov))
    center, center_err = float(popt[3]), float(perr[3])
    sigma = float(popt[4])
    gamma = float(popt[5]) if shape == "voigt" else 0.0

    resid = it - func(wl, *popt)
    dof = max(1, len(wl) - len(popt))
    noise = float(np.std(np.concatenate([resid[:edge], resid[-edge:]]))) or 1.0
    chi2_dof = float(np.sum((resid / noise) ** 2) / dof)
    return center, center_err, chi2_dof, sigma, gamma, popt, wl, it


def _average_spectrum(files: list, cfg: dict, calibrate: bool = False):
    """Mean intensity over trial files sharing a wavelength grid."""
    wl0, stack = None, []
    for fp in files:
        df = load_spectrum(fp, cfg, calibrate=calibrate)
        if wl0 is None:
            wl0 = df["wavelength"].to_numpy(dtype=float)
        stack.append(df["intensity"].to_numpy(dtype=float))
    return wl0, np.mean(np.vstack(stack), axis=0)


def measure_skew(cfg: dict = CONFIG, apply: bool = True) -> SkewResult:
    """Measure the instrumental red-tail decay tau from the neon lines.

    Averages the neon trials, finds the strongest isolated lines, and fits each
    with the configured symmetric shape convolved with a one-sided exponential
    (tau FREE). The adopted skew is the median tau across the lines -- robust to
    the odd blended line. If apply=True, tau is written into cfg["skew"]["tau"]
    so the helium and H2D2 fits pick it up.
    """
    sc = cfg["skew"]
    files = list_runs(cfg, data_root=sc["neon_root"])
    if not files:
        raise FileNotFoundError(f"no neon spectra in {sc['neon_root']!r}")
    wl, it = _average_spectrum(files, cfg, calibrate=False)   # raw; tail is shift-free

    rng = float(it.max() - it.min()) or 1.0
    idx, props = find_peaks(it, prominence=0.05 * rng, distance=8)
    order = np.argsort(props["prominences"])[::-1][:int(sc.get("n_lines", 5))]
    lines = np.sort(idx[order])

    ls = _lineshape(cfg)
    f = cfg["fit"]
    half = float(sc.get("half_window", 0.30))
    # free-tau single-line model (adds tau as the last fit parameter)
    if ls == "voigt":
        def fline(l, B, m, A, lam0, sigma, gamma, tau):
            return B + m * l + A * skewed_peak_values(l, lam0, sigma, gamma, tau, ls)
    else:
        def fline(l, B, m, A, lam0, sigma, tau):
            return B + m * l + A * skewed_peak_values(l, lam0, sigma, 0.0, tau, ls)

    wls, taus, sigs, chis, fits = [], [], [], [], []
    for i in lines:
        lo, hi = wl[i] - half, wl[i] + half
        mm = (wl >= lo) & (wl <= hi)
        x, y = wl[mm], it[mm]
        if len(x) < 8:
            continue
        edge = max(3, len(y) // 10)
        B0 = float(np.median(np.concatenate([y[:edge], y[-edge:]])))
        A0 = max(float(y.max() - B0), 1.0)
        lam0 = float(x[np.argmax(y)])
        if ls == "voigt":
            p0 = [B0, 0.0, A0, lam0, f["sigma_guess"], f.get("gamma_guess", 0.01),
                  sc.get("tau_guess", 0.02)]
            lb = [-np.inf, -np.inf, 0.0, lo, 1e-3, 0.0, sc["tau_min"]]
            ub = [np.inf, np.inf, np.inf, hi, f["sigma_max"], f["gamma_max"], sc["tau_max"]]
        else:
            p0 = [B0, 0.0, A0, lam0, f["sigma_guess"], sc.get("tau_guess", 0.02)]
            lb = [-np.inf, -np.inf, 0.0, lo, 1e-3, sc["tau_min"]]
            ub = [np.inf, np.inf, np.inf, hi, f["sigma_max"], sc["tau_max"]]
        try:
            popt, _ = curve_fit(fline, x, y, p0=p0, bounds=(lb, ub),
                                absolute_sigma=False, maxfev=f["maxfev"])
        except Exception as exc:
            print(f"  [warn] skew fit failed at {wl[i]:.2f} nm: {exc}")
            continue
        resid = y - fline(x, *popt)
        edge = max(3, len(y) // 10)
        noise = float(np.std(np.concatenate([resid[:edge], resid[-edge:]]))) or 1.0
        chi = float(np.sum((resid / noise) ** 2) / max(1, len(x) - len(popt)))
        wls.append(float(wl[i])); sigs.append(float(popt[4]))
        taus.append(float(popt[-1])); chis.append(chi)
        fits.append((x, y, popt))

    taus = np.array(taus)
    wls = np.array(wls)
    tau_med = float(np.median(taus)) if len(taus) else 0.0
    tau_scatter = float(np.std(taus, ddof=1)) if len(taus) > 1 else 0.0
    res = SkewResult(lineshape=ls, line_wl=wls, taus=taus,
                     sigmas=np.array(sigs), chi2_dof=np.array(chis),
                     tau=tau_med, tau_scatter=tau_scatter,
                     n_trials=len(files), fits=fits)
    if apply:
        cfg["skew"]["tau"] = tau_med
        # tau(lambda) table (sorted) so the applied skew tracks the red growth
        if len(taus):
            order = np.argsort(wls)
            cfg["skew"]["tau_table"] = (wls[order], taus[order])
    return res


def compute_helium_offset(cfg: dict = CONFIG, apply: bool = True) -> OffsetResult:
    """Measure the spectrometer's wavelength offset from the helium trials.

    Each helium spectrum is loaded RAW (uncalibrated), the He I line is fit, and
    the per-trial offset is nist_line - center. The trial centers are pooled by
    inverse-variance weighting (Birge-inflated for run-to-run scatter); the
    reported offset is nist_line - pooled_center and is ADDED to every spectrum.

    If apply=True the offset (and its 1-sigma) is written into
    cfg["calibration"], turning on the additive correction for all later loads.
    """
    hc = cfg["helium_cal"]
    files = list_runs(cfg, data_root=hc["data_root"])
    if not files:
        raise FileNotFoundError(f"no helium spectra in {hc['data_root']!r}")

    names, centers, cerr, chi, sigmas, gammas, fits = [], [], [], [], [], [], []
    for f in files:
        df = load_spectrum(f, cfg, calibrate=False)     # raw recorded wavelengths
        c, e, x, sg, gm, popt, wl, it = fit_helium_line(df, cfg)
        names.append(_run_name(f)); centers.append(c); cerr.append(e); chi.append(x)
        sigmas.append(sg); gammas.append(gm)
        fits.append((wl, it, popt))
    centers = np.array(centers); cerr = np.array(cerr); chi = np.array(chi)

    comb = _weighted_combine(centers, cerr)             # pooled He center
    center_comb, center_comb_err = comb["value"], comb["error"]
    nist = float(hc["nist_line"]); nist_err = float(hc.get("nist_line_err", 0.0))
    offset = nist - center_comb
    offset_err = float(np.hypot(center_comb_err, nist_err))

    res = OffsetResult(
        nist_line=nist, trial_names=names, centers=centers, centers_err=cerr,
        chi2_dof=chi, center_combined=center_comb, center_combined_err=center_comb_err,
        offset=offset, offset_err=offset_err, offset_scatter=comb["std"],
        lineshape=_lineshape(cfg), sigma_he=float(np.mean(sigmas)),
        gamma_he=float(np.mean(gammas)),
        skew_tau=skew_tau_at(cfg, float(hc["nist_line"])), fits=fits,
    )
    if apply:
        cal = cfg["calibration"]
        cal["enabled"] = True
        cal["offset"] = offset
        cal["offset_err"] = offset_err
        cal["scale"] = cal.get("scale", 1.0) or 1.0
    return res


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


def one_gaussian(lmbda, B, m, A, lam0, sig):
    """Sloped background plus a single Gaussian peak (the helium-line model)."""
    return B + m * lmbda + A * np.exp(-((lmbda - lam0) ** 2) / (2.0 * sig ** 2))


def _voigt_peak(dl, sigma, gamma):
    """Height-normalized Voigt (value 1 at dl=0): the area-normalized
    scipy Voigt divided by its peak, so the amplitude A is the PEAK HEIGHT
    (counts), directly comparable to the Gaussian model's amplitude."""
    return voigt_profile(dl, sigma, gamma) / voigt_profile(0.0, sigma, gamma)


def two_voigt(lmbda, B, m, A_H, lam_H, A_D, lam_D, sigma, gamma):
    """Sloped background plus two Voigt peaks sharing the Gaussian width sigma
    (Doppler + instrument) and the Lorentzian HWHM gamma (lifetime + pressure)."""
    return (B + m * lmbda
            + A_H * _voigt_peak(lmbda - lam_H, sigma, gamma)
            + A_D * _voigt_peak(lmbda - lam_D, sigma, gamma))


def one_voigt(lmbda, B, m, A, lam0, sigma, gamma):
    """Sloped background plus a single Voigt peak (the helium-line model)."""
    return B + m * lmbda + A * _voigt_peak(lmbda - lam0, sigma, gamma)


def _lineshape(cfg: dict) -> str:
    return str(cfg.get("lineshape", "gaussian")).lower()


def _symmetric_peak(dl, sigma, gamma, lineshape):
    """Height-normalized SYMMETRIC line (value 1 at dl=0)."""
    if lineshape == "voigt":
        return _voigt_peak(dl, sigma, gamma)
    return np.exp(-(dl ** 2) / (2.0 * sigma ** 2))


def skewed_peak_values(lmbda, lam0, sigma, gamma, tau, lineshape):
    """Height-normalized line at center lam0, optionally skewed by an instrumental
    one-sided exponential of decay tau (nm) toward the RED (longer wavelength).

    tau <= 0 returns the plain symmetric line. Otherwise the symmetric profile is
    built on a fine uniform grid and convolved with a normalized causal kernel
    exp(-t/tau) (t >= 0), then renormalized to unit peak height and interpolated
    back to lmbda. Modeling the tail this way means the fitted lam0 is the TRUE
    (pre-skew) center -- the asymmetry is absorbed by the kernel, not the center.
    """
    lmbda = np.asarray(lmbda, dtype=float)
    if tau is None or tau <= 0:
        return _symmetric_peak(lmbda - lam0, sigma, gamma, lineshape)
    lo, hi = float(lmbda.min()), float(lmbda.max())
    diffs = np.diff(np.sort(lmbda))
    dx = float(np.median(diffs)) / 4.0 if len(diffs) else (hi - lo) / 400.0
    dx = max(dx, 1e-5)
    grid = np.arange(lo - 5 * dx, hi + 8.0 * tau + dx, dx)
    sym = _symmetric_peak(grid - lam0, sigma, gamma, lineshape)
    n = int(np.ceil(8.0 * tau / dx))
    t = np.arange(n + 1) * dx
    kern = np.exp(-t / tau)
    kern /= kern.sum()
    conv = np.convolve(sym, kern)[:len(grid)]    # causal -> tail toward +lambda
    peak = conv.max() or 1.0
    return np.interp(lmbda, grid, conv / peak)


def peak_profile(lineshape, lmbda, A, lam0, sigma, gamma, tau=0.0):
    """Evaluate a single line's contribution (peak height A), with optional skew."""
    return A * skewed_peak_values(lmbda, lam0, sigma, gamma, tau, lineshape)


def _skew_tau(cfg: dict) -> float:
    """Representative (median) skew tau to APPLY (nm), or 0 if skew is off.
    Used for display; the fits use the wavelength-local value (skew_tau_at)."""
    sc = cfg.get("skew", {})
    if not sc.get("enabled", False):
        return 0.0
    tau = sc.get("tau")
    return float(tau) if tau else 0.0


def skew_tau_at(cfg: dict, lam: float) -> float:
    """Wavelength-local skew tau (nm) to apply at wavelength `lam`.

    The neon tail grows toward the red (a charge-transfer-inefficiency signature),
    so we interpolate tau(lambda) from the per-line measurements (clamped at the
    ends) rather than forcing a single constant. Falls back to the scalar tau (or
    0) when no table / skew disabled."""
    sc = cfg.get("skew", {})
    if not sc.get("enabled", False):
        return 0.0
    tbl = sc.get("tau_table")
    if tbl is not None and len(tbl[0]) >= 2:
        lams, taus = tbl
        return float(np.interp(lam, lams, taus))
    tau = sc.get("tau")
    return float(tau) if tau else 0.0


def doublet_model(cfg: dict, tau: float):
    """curve_fit model for the H/D doublet at fixed skew tau (same parameter
    layout as the symmetric model, so toggling skew never changes the DOF)."""
    ls = _lineshape(cfg)
    if ls == "voigt":
        def f(l, B, m, A_H, lam_H, A_D, lam_D, sigma, gamma):
            return (B + m * l
                    + A_H * skewed_peak_values(l, lam_H, sigma, gamma, tau, ls)
                    + A_D * skewed_peak_values(l, lam_D, sigma, gamma, tau, ls))
    else:
        def f(l, B, m, A_H, lam_H, sig_H, A_D, lam_D, sig_D):
            return (B + m * l
                    + A_H * skewed_peak_values(l, lam_H, sig_H, 0.0, tau, ls)
                    + A_D * skewed_peak_values(l, lam_D, sig_D, 0.0, tau, ls))
    return f


def single_model(cfg: dict, tau: float):
    """curve_fit model for a single line (helium / neon) at fixed skew tau."""
    ls = _lineshape(cfg)
    if ls == "voigt":
        def f(l, B, m, A, lam0, sigma, gamma):
            return B + m * l + A * skewed_peak_values(l, lam0, sigma, gamma, tau, ls)
    else:
        def f(l, B, m, A, lam0, sigma):
            return B + m * l + A * skewed_peak_values(l, lam0, sigma, 0.0, tau, ls)
    return f


def fwhm_gauss(sigma: float) -> float:
    return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma)


def fwhm_voigt(sigma: float, gamma: float) -> float:
    """Olivero-Longbothum approximation for the Voigt FWHM (nm)."""
    fG = fwhm_gauss(sigma)
    fL = 2.0 * gamma
    return float(0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2))


def doppler_temperature(sigma_doppler_nm: float, lambda0_nm: float,
                        mass_kg: float = _M_H) -> float:
    """Gas temperature (K) implied by a Gaussian Doppler width sigma (nm):
        sigma/lambda = sqrt(kT / (m c^2))  ->  T = (sigma c / lambda)^2 m / k."""
    if sigma_doppler_nm <= 0:
        return float("nan")
    frac = sigma_doppler_nm / lambda0_nm
    return float((frac * _C_LIGHT) ** 2 * mass_kg / _K_BOLTZ)


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

    if _lineshape(cfg) == "voigt":
        gam = f.get("gamma_guess", 0.01)
        #     [ B,    m,   A1, lam1, A2, lam2, sigma,         gamma ]
        p0 = [B0, 0.0, a1, c1, a2, c2, sig, gam]
        lower = [-np.inf, -np.inf, 0.0, lo, 0.0, lo, f["sigma_min"], 0.0]
        upper = [np.inf, np.inf, np.inf, hi, np.inf, hi,
                 f["sigma_max"], f["gamma_max"]]
    else:
        #     [ B,    m,   A1, lam1, sig1, A2, lam2, sig2 ]
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
    """Crop to the doublet window and fit the configured line shape.

    Line shape (CONFIG["lineshape"]):
      - "voigt"    : two Voigt peaks sharing the Gaussian width sigma (Doppler +
                     instrument) and Lorentzian HWHM gamma (lifetime + pressure).
      - "gaussian" : two independent Gaussians.

    Uncertainty propagation (each term reported separately, summed in quadrature):
      - FIT (statistical): the model is fit with UNIFORM weighting -- correct
        here because the per-point noise is ~homoscedastic, so this is the
        unbiased (max-likelihood) estimator; weighting by the tiny readout floor
        instead would pathologically up-weight the flat baseline and bias the
        centers. The center covariance is read off pcov with absolute_sigma=False,
        so the error SCALE is set by the actual residuals (real shot noise + any
        residual line-shape mismatch), which is the honest statistical error.
        Delta lambda uses the full covariance var(lamH)+var(lamD)-2cov(lamH,lamD).
      - READOUT: +/- 1/2 last decimal. The wavelength readout is a per-center
        floor (sigma_wl); the intensity readout's center contribution
        (readout_center_floor) is tracked but dominated -- both fold into the
        total in quadrature.
      - CALIBRATION (systematic): the OceanView pixel->wavelength fit accuracy
        as a fractional (scale) error rel_cal plus the helium-offset error;
        common-mode across trials, and nearly cancels in Delta lambda.
    """
    shape = _lineshape(cfg)
    lo_w, hi_w = cfg["fit_window"]
    tau = skew_tau_at(cfg, 0.5 * (lo_w + hi_w))   # red-tail decay local to ~656 nm
    model = doublet_model(cfg, tau)
    win = crop(df, cfg)
    wl = win["wavelength"].to_numpy(dtype=float)
    it = win["intensity"].to_numpy(dtype=float)
    if len(wl) < 8:
        raise ValueError(f"only {len(wl)} points in fit window {cfg['fit_window']} "
                         f"for run {run!r}; widen CONFIG['fit_window'].")

    p0, bounds = initial_guess(wl, it, cfg)
    success = True
    try:
        popt, pcov = curve_fit(model, wl, it, p0=p0, bounds=bounds,
                               absolute_sigma=False, maxfev=cfg["fit"]["maxfev"])
    except Exception as exc:  # keep the run; flag it as not converged
        print(f"  [warn] fit failed for {run!r}: {exc}")
        popt = np.array(p0, dtype=float)
        pcov = np.full((len(p0), len(p0)), np.nan)
        success = False

    perr = np.sqrt(np.diag(pcov))

    # center parameter positions in popt: voigt [.,.,A1,lam1,A2,lam2,sig,gam],
    # gaussian [.,.,A1,lam1,sig1,A2,lam2,sig2]
    c1, c2 = (3, 5) if shape == "voigt" else (3, 6)
    # assign H (longer wavelength) vs D (shorter) from the fitted centers
    i_hi, i_lo = (c1, c2) if popt[c1] >= popt[c2] else (c2, c1)
    lambda_H, lambda_D = popt[i_hi], popt[i_lo]
    lamH_fit, lamD_fit = perr[i_hi], perr[i_lo]
    dlambda = lambda_H - lambda_D
    # Delta lambda statistical error from the covariance of the two centers:
    #   var(lamH - lamD) = var(lamH) + var(lamD) - 2 cov(lamH, lamD)
    var = pcov[i_hi, i_hi] + pcov[i_lo, i_lo] - 2.0 * pcov[i_hi, i_lo]
    dl_fit = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else float("nan")

    # unpack widths/amplitudes (amplitude index = center-1 for both shapes)
    B, m = float(popt[0]), float(popt[1])
    A_H, A_D = float(popt[i_hi - 1]), float(popt[i_lo - 1])
    if shape == "voigt":
        sig_H = sig_D = float(popt[6])       # shared Gaussian width
        gam_H = gam_D = float(popt[7])       # shared Lorentzian HWHM
    else:
        sig_H, sig_D = float(popt[i_hi + 1]), float(popt[i_lo + 1])
        gam_H = gam_D = 0.0
    pdict = {"B": B, "m": m,
             "A_H": A_H, "lambda_H": float(lambda_H), "sigma_H": sig_H, "gamma_H": gam_H,
             "A_D": A_D, "lambda_D": float(lambda_D), "sigma_D": sig_D, "gamma_D": gam_D,
             "tau": float(tau)}

    # calibration systematic (common-mode across trials). Two pieces hit the
    # ABSOLUTE centers: the fractional dispersion/scale error (from R^2) and the
    # measured helium-offset uncertainty. Only the scale survives in Delta lambda
    # -- the additive offset (and its error) cancel in lambda_H - lambda_D.
    rel_cal = calibration_rel_error(df["wavelength"].to_numpy(dtype=float), cfg)
    off_err = float(cfg.get("calibration", {}).get("offset_err", 0.0))
    lamH_cal = float(np.hypot(rel_cal * lambda_H, off_err))
    lamD_cal = float(np.hypot(rel_cal * lambda_D, off_err))
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

    resid = it - model(wl, *popt)
    dof = max(1, len(wl) - len(popt))
    # noise estimate from the flat background tails (robust to the peaks)
    edge = max(3, len(it) // 10)
    noise = float(np.std(np.concatenate([resid[:edge], resid[-edge:]]))) or 1.0
    chi2_dof = float(np.sum((resid / noise) ** 2) / dof)
    # RMS residual (counts): a fit-quality metric that is COMPARABLE across the
    # skew toggle (unlike chi2/dof, whose baseline-noise denominator shifts when
    # the wings start fitting better).
    rms_resid = float(np.sqrt(np.mean(resid ** 2)))

    return FitResult(
        run=_run_name(run),
        wavelength=df["wavelength"].to_numpy(dtype=float),
        intensity=df["intensity"].to_numpy(dtype=float),
        fit_wl=wl, fit_i=it,
        popt=np.asarray(popt, dtype=float), pcov=pcov,
        lambda_H=float(lambda_H), lambda_D=float(lambda_D), dlambda=float(dlambda),
        lambda_H_err=lambda_H_err, lambda_D_err=lambda_D_err, dlambda_err=dlambda_err,
        pdict=pdict, lineshape=shape, skew_tau=float(tau), rms_resid=rms_resid,
        sigma_gauss=0.5 * (sig_H + sig_D), gamma_lor=gam_H,
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
    ls = result.lineshape
    wl, it = result.fit_wl, result.fit_i
    grid = np.linspace(wl.min(), wl.max(), 1000)
    tau = p.get("tau", result.skew_tau)
    model = result.model(grid)
    bg = p["B"] + p["m"] * grid
    gH = bg + peak_profile(ls, grid, p["A_H"], p["lambda_H"], p["sigma_H"], p["gamma_H"], tau)
    gD = bg + peak_profile(ls, grid, p["A_D"], p["lambda_D"], p["sigma_D"], p["gamma_D"], tau)

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

    ax.scatter(wl, it, s=10, color="0.35", alpha=0.7, label="data", zorder=2)
    ax.plot(grid, model, color="C3", lw=2.0, label=f"two-{ls} fit", zorder=4)
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
    if result.lineshape == "voigt":
        txt += (f"\n$\\sigma_G$={result.sigma_gauss:.4f}, "
                f"$\\gamma_L$={result.gamma_lor:.4f} nm")
    if result.skew_tau > 0:
        txt += f"\nskew $\\tau$={result.skew_tau:.4f} nm"
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_ylabel("Intensity (counts)")
    skewlab = f", skew on" if result.skew_tau > 0 else ", skew off"
    ax.set_title(f"{result.run}: H-alpha / D-alpha doublet fit "
                 f"({result.lineshape}{skewlab})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    resid = it - result.model(wl)
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


def plot_helium(offset: OffsetResult, cfg: dict = CONFIG,
                save_path: Optional[str] = None) -> str:
    """Overlay the helium-line fits (raw scale) with the NIST line and the
    fitted centers, annotated with the measured offset."""
    out = save_path or os.path.join(cfg["save_dir"], "helium_offset.png")
    _ensure_dir(os.path.dirname(out) or ".")
    he_model = single_model(cfg, offset.skew_tau)   # matches the (skewed) fit
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for k, ((wl, it, popt), name, c) in enumerate(
            zip(offset.fits, offset.trial_names, offset.centers)):
        color = cmap(k % 10)
        ax.scatter(wl, it, s=8, alpha=0.4, color=color)
        grid = np.linspace(wl.min(), wl.max(), 800)
        ax.plot(grid, he_model(grid, *popt), lw=1.4, color=color,
                label=f"{name}: center {c:.4f} nm")
        ax.axvline(c, color=color, ls="--", lw=0.8, alpha=0.6)
    ax.axvline(offset.nist_line, color="k", lw=1.6, ls="-",
               label=f"NIST He I {offset.nist_line:.5f} nm")
    ax.set_xlabel("Wavelength (nm, raw / uncorrected)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_title("Helium offset calibration "
                 f"(offset = {offset.offset:+.4f} $\\pm$ {offset.offset_err:.4f} nm)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    if cfg.get("show"):
        plt.show()
    plt.close(fig)
    return out


def plot_skew(skew: SkewResult, cfg: dict = CONFIG,
              save_path: Optional[str] = None) -> str:
    """Show the neon skew fits: each strong line centered on its fitted peak,
    peak-normalized, with the asymmetric (skewed) model overlaid."""
    out = save_path or os.path.join(cfg["save_dir"], "neon_skew_fit.png")
    _ensure_dir(os.path.dirname(out) or ".")
    ls = skew.lineshape
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    for k, ((x, y, popt), w, t) in enumerate(zip(skew.fits, skew.line_wl, skew.taus)):
        color = cmap(k % 10)
        lam0 = popt[3]
        ax.scatter(x - lam0, y / y.max(), s=8, alpha=0.4, color=color)
        grid = np.linspace(x.min(), x.max(), 800)
        # rebuild the model curve for this line (free-tau popt)
        if ls == "voigt":
            B, m, A, l0, sig, gam, tau = popt
            curve = B + m * grid + A * skewed_peak_values(grid, l0, sig, gam, tau, ls)
        else:
            B, m, A, l0, sig, tau = popt
            curve = B + m * grid + A * skewed_peak_values(grid, l0, sig, 0.0, tau, ls)
        ax.plot(grid - lam0, (curve - curve.min()) / (curve.max() - curve.min()),
                lw=1.4, color=color, label=f"{w:.2f} nm  ($\\tau$={t:.4f})")
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_xlabel("Wavelength offset from peak (nm)   [+ = red / rightward]")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(f"Neon skew fit  (adopted $\\tau$ = {skew.tau:.4f} "
                 f"$\\pm$ {skew.tau_scatter:.4f} nm)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
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
    off_err = float(cfg.get("calibration", {}).get("offset_err", 0.0))

    out = {"n_runs": len(good), "rel_cal": rel_cal, "offset_err": off_err}
    for key, fitkey, n_read in (("lambda_H", "lambda_H_err_fit", 1.0),
                                ("lambda_D", "lambda_D_err_fit", 1.0),
                                ("dlambda", "dlambda_err_fit", np.sqrt(2.0))):
        vals = np.array([getattr(r, key) for r in good.values()], dtype=float)
        errs = np.array([getattr(r, fitkey) for r in good.values()], dtype=float)
        c = _weighted_combine(vals, errs)             # pools the statistical term
        value = c["value"]
        stat = c["error"]
        # calibration systematic (common-mode, added once): scale error always,
        # plus the helium-offset error on ABSOLUTE centers (cancels in dlambda).
        cal = rel_cal * abs(value)
        if key != "dlambda":
            cal = float(np.hypot(cal, off_err))
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


def physical_widths(results: dict, offset: Optional[OffsetResult] = None,
                    cfg: dict = CONFIG) -> dict:
    """Decompose the (Voigt) line widths into their physical contributions.

    The shared Gaussian width sigma_G combines instrument + Doppler in quadrature;
    using the helium line as the instrumental width proxy (its own Doppler is only
    ~half of hydrogen's and is neglected here), the Doppler width and the implied
    gas temperature follow from sigma_Doppler^2 = sigma_G^2 - sigma_inst^2 and
    T = (sigma_Doppler c / lambda)^2 m_H / k. The Lorentzian width gamma is the
    lifetime + pressure contribution (natural part ~1e-4 nm is unresolvable here).
    Returns NaNs gracefully if widths are unavailable or the subtraction is
    negative (instrument already broader than the H line).
    """
    good = [r for r in results.values() if r.success]
    if not good:
        return {}
    sig_G = float(np.mean([r.sigma_gauss for r in good]))
    gam_L = float(np.mean([r.gamma_lor for r in good]))
    lam = float(np.mean([r.lambda_H for r in good]))
    out = {
        "lineshape": _lineshape(cfg),
        "sigma_gauss": sig_G, "fwhm_gauss": fwhm_gauss(sig_G),
        "gamma_lor": gam_L, "fwhm_lor": 2.0 * gam_L,
        "fwhm_voigt": fwhm_voigt(sig_G, gam_L),
        "sigma_inst": float("nan"), "sigma_doppler": float("nan"),
        "T_doppler": float("nan"),
    }
    if offset is not None and np.isfinite(offset.sigma_he) and offset.sigma_he > 0:
        sig_inst = float(offset.sigma_he)        # He line ~ instrumental width
        out["sigma_inst"] = sig_inst
        d2 = sig_G ** 2 - sig_inst ** 2
        if d2 > 0:
            sig_dopp = float(np.sqrt(d2))
            out["sigma_doppler"] = sig_dopp
            out["T_doppler"] = doppler_temperature(sig_dopp, lam, _M_H)
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
# Text report                                                                 #
# --------------------------------------------------------------------------- #

def write_report(results: dict, cfg: dict = CONFIG,
                 offset: Optional[OffsetResult] = None,
                 skew: Optional[SkewResult] = None,
                 save_path: Optional[str] = None) -> str:
    """Write a human-readable .txt summary: the helium offset calibration, every
    per-trial fit (parameters, peaks, uncertainty breakdown), and the final
    combined result with its full uncertainty budget."""
    from datetime import datetime
    out = save_path or os.path.join(cfg["save_dir"], "H2D2_summary.txt")
    _ensure_dir(os.path.dirname(out) or ".")
    L = []
    bar = "=" * 78
    sub = "-" * 78
    L.append(bar)
    L.append(" H2D2  -  Hydrogen / Deuterium Balmer-alpha analysis summary")
    L.append(f" generated {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(bar)
    L.append("")
    shape = _lineshape(cfg)
    if shape == "voigt":
        L.append(" MODEL   I(lambda) = B + m*lambda + A_H*V(lambda-lambda_H; sigma,gamma)")
        L.append("                                  + A_D*V(lambda-lambda_D; sigma,gamma)")
        L.append("         V = Voigt (Gaussian Doppler/instrument (x) Lorentzian")
        L.append("         lifetime/pressure); sigma and gamma shared across H,D.")
    else:
        L.append(" MODEL   I(lambda) = B + m*lambda")
        L.append("                   + A_H*exp(-(lambda-lambda_H)^2 / (2 sigma_H^2))")
        L.append("                   + A_D*exp(-(lambda-lambda_D)^2 / (2 sigma_D^2))")
    lo, hi = cfg["fit_window"]
    skew_on = bool(cfg.get("skew", {}).get("enabled", False)) and bool(cfg["skew"].get("tau"))
    L.append(f"         line shape: {shape};  fit window {lo}-{hi} nm;  "
             f"H = longer-wavelength line.")
    L.append(f"         skew correction: {'ON' if skew_on else 'OFF'}"
             + (f"  (instrumental red-tail tau = {cfg['skew']['tau']:.4f} nm)"
                if skew_on else ""))
    L.append("")

    # ---- instrumental skew ------------------------------------------------ #
    if skew is not None:
        L.append(sub)
        L.append(" INSTRUMENTAL SKEW (red-tail) FROM NEON")
        L.append(sub)
        L.append(f"  model: symmetric {skew.lineshape} (x) one-sided exponential "
                 f"exp(-t/tau), t>=0 (red)")
        L.append(f"  {'neon line (nm)':>15s} {'tau (nm)':>10s} {'chi2/dof':>10s}")
        for w, t, x in zip(skew.line_wl, skew.taus, skew.chi2_dof):
            L.append(f"  {w:15.3f} {t:10.4f} {x:10.1f}")
        L.append(f"  median tau          : {skew.tau:.4f} +/- {skew.tau_scatter:.4f} nm "
                 f"({skew.n_trials} trials, {len(skew.taus)} lines)")
        L.append("  tau GROWS toward the red (charge-transfer-inefficiency signature),")
        L.append("  so tau is interpolated in wavelength and applied LOCALLY:")
        L.append(f"     tau(He 667.8 nm)  = {skew_tau_at(cfg, cfg['helium_cal']['nist_line']):.4f} nm")
        L.append(f"     tau(H2D2 ~656 nm) = {skew_tau_at(cfg, 0.5*sum(cfg['fit_window'])):.4f} nm")
        L.append("  -> fixed when fitting the helium and H2D2 lines (de-skews centers).")
        L.append("")

    # ---- helium offset ---------------------------------------------------- #
    L.append(sub)
    L.append(" HELIUM WAVELENGTH-OFFSET CALIBRATION")
    L.append(sub)
    if offset is not None:
        hlo, hhi = cfg["helium_cal"]["fit_window"]
        L.append(f"  NIST He I reference : {offset.nist_line:.5f} nm "
                 f"(+/- {cfg['helium_cal'].get('nist_line_err', 0.0):.0e})")
        L.append(f"  fit window          : {hlo}-{hhi} nm  "
                 f"(single {offset.lineshape} + linear background)")
        L.append("")
        L.append(f"  {'trial':28s} {'center (raw, nm)':>20s} {'chi2/dof':>9s} "
                 f"{'offset (nm)':>12s}")
        for n, c, e, x in zip(offset.trial_names, offset.centers,
                              offset.centers_err, offset.chi2_dof):
            L.append(f"  {n:28s} {c:10.4f} +/- {e:.4f} {x:9.1f} "
                     f"{offset.nist_line - c:+12.4f}")
        L.append("")
        L.append(f"  He line widths      : Gaussian sigma {offset.sigma_he:.4f} nm"
                 + (f",  Lorentzian gamma {offset.gamma_he:.4f} nm"
                    if offset.lineshape == "voigt" else ""))
        L.append(f"  combined He center  : {offset.center_combined:.4f} +/- "
                 f"{offset.center_combined_err:.4f} nm  "
                 f"(trial scatter {offset.offset_scatter:.4f} nm)")
        L.append(f"  APPLIED OFFSET      : {offset.offset:+.4f} +/- "
                 f"{offset.offset_err:.4f} nm   (added to every wavelength)")
        L.append("  note: a constant offset cancels in Delta lambda; it fixes the")
        L.append("        ABSOLUTE centers and contributes the offset_err to them.")
    else:
        L.append("  (helium offset not applied)")
    L.append("")

    # ---- per-trial fits --------------------------------------------------- #
    L.append(sub)
    L.append(" PER-TRIAL H2D2 FITS   (wavelengths are offset-corrected / measured)")
    L.append(sub)
    for name, r in results.items():
        p = r.params
        L.append(f" {name}  [{r.lineshape}]")
        L.append(f"   background : B = {p['B']:.3f} counts,  m = {p['m']:.4f} counts/nm")
        if r.lineshape == "voigt":
            L.append(f"   H-alpha    : A = {p['A_H']:.1f} counts")
            L.append(f"   D-alpha    : A = {p['A_D']:.1f} counts")
            L.append(f"   widths     : Gaussian sigma = {r.sigma_gauss:.4f} nm "
                     f"(FWHM {fwhm_gauss(r.sigma_gauss):.4f}), "
                     f"Lorentzian gamma = {r.gamma_lor:.4f} nm  [shared H,D]")
        else:
            L.append(f"   H-alpha    : A = {p['A_H']:.1f} counts,  sigma = {p['sigma_H']:.4f} nm")
            L.append(f"   D-alpha    : A = {p['A_D']:.1f} counts,  sigma = {p['sigma_D']:.4f} nm")
        L.append(f"   lambda_H   : {r.lambda_H:.4f} +/- {r.lambda_H_err:.4f} nm "
                 f"(fit {r.lambda_H_err_fit:.4f}, cal {r.lambda_H_err_cal:.4f})")
        L.append(f"   lambda_D   : {r.lambda_D:.4f} +/- {r.lambda_D_err:.4f} nm "
                 f"(fit {r.lambda_D_err_fit:.4f}, cal {r.lambda_D_err_cal:.4f})")
        L.append(f"   Delta lam  : {r.dlambda:.4f} +/- {r.dlambda_err:.4f} nm "
                 f"(fit {r.dlambda_err_fit:.4f}, cal {r.dlambda_err_cal:.1e})")
        L.append(f"   chi2/dof   : {r.chi2_dof:.2f}   RMS resid {r.rms_resid:.1f} cts   "
                 f"(points {len(r.fit_wl)}, converged {r.success})")
        L.append("")

    # ---- combined --------------------------------------------------------- #
    comb = combine_runs(results, cfg)
    L.append(sub)
    L.append(f" COMBINED RESULT   ({comb['n_runs']} trials)")
    L.append(sub)
    L.append(f"  rel. calibration (scale) error : {comb['rel_cal']:.2e}")
    L.append(f"  helium offset error            : {comb['offset_err']:.4f} nm")
    for k, lab in (("lambda_H", "lambda_H    "), ("lambda_D", "lambda_D    "),
                   ("dlambda", "Delta lambda")):
        L.append(f"  {lab} = {comb[k]:.4f} +/- {comb[k+'_err']:.4f} nm "
                 f"(stat {comb[k+'_stat']:.4f}, cal {comb[k+'_cal']:.1e}, "
                 f"read {comb[k+'_read']:.1e})")
    L.append(f"  plain mean Delta lambda = {comb['dlambda_mean']:.4f} +/- "
             f"{comb['dlambda_sem']:.4f} nm (SEM);  scatter {comb['dlambda_std']:.4f} nm")
    L.append("")
    L.append(f"  literature : lambda_H {cfg.get('lambda_H_lit', float('nan')):.4f}, "
             f"lambda_D {cfg.get('lambda_D_lit', float('nan')):.4f}, "
             f"Delta lambda {cfg.get('dlambda_lit', float('nan')):.4f} nm")
    L.append("")

    # ---- line widths & Doppler temperature -------------------------------- #
    if shape == "voigt":
        pw = physical_widths(results, offset, cfg)
        L.append(sub)
        L.append(" LINE-SHAPE WIDTHS & DOPPLER TEMPERATURE  (shared Voigt, mean over trials)")
        L.append(sub)
        L.append(f"  Gaussian sigma (Doppler+instrument) : {pw['sigma_gauss']:.4f} nm "
                 f"(FWHM {pw['fwhm_gauss']:.4f} nm)")
        L.append(f"  Lorentzian gamma (lifetime+pressure): {pw['gamma_lor']:.4f} nm "
                 f"(FWHM {pw['fwhm_lor']:.4f} nm)")
        L.append(f"  Voigt FWHM                          : {pw['fwhm_voigt']:.4f} nm")
        if np.isfinite(pw.get("sigma_inst", float("nan"))):
            L.append(f"  instrumental sigma (from He line)   : {pw['sigma_inst']:.4f} nm")
            if np.isfinite(pw.get("T_doppler", float("nan"))):
                L.append(f"  Doppler sigma (H, deconvolved)      : "
                         f"{pw['sigma_doppler']:.4f} nm  ->  T ~ {pw['T_doppler']:.0f} K "
                         f"(ROUGH UPPER ESTIMATE -- see note)")
            else:
                L.append("  Doppler width: instrument >= observed Gaussian width here,")
                L.append("                 so the thermal part is not resolvable.")
        L.append("  notes:")
        L.append("   - the natural (lifetime) Lorentzian for H-alpha is ~1e-4 nm, far")
        L.append("     below resolution: the fitted gamma is pressure/instrument, not")
        L.append("     the lifetime.")
        L.append("   - the He line is asymmetric (red tail) and the symmetric Voigt")
        L.append("     puts almost all its width into gamma, leaving its Gaussian sigma")
        L.append("     ~0. So sigma_inst is poorly determined and the Doppler T is a")
        L.append("     model-dependent UPPER estimate (the H Gaussian width also")
        L.append("     contains unresolved Stark/instrument structure, not pure thermal).")
        L.append("     A clean T needs a symmetric, well-resolved calibration line.")
        L.append("")
    L.append(bar)

    text = "\n".join(L) + "\n"
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


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

    # 0a. instrumental skew (red-tail tau) measured from neon, applied to all fits
    skew = None
    if CONFIG.get("skew", {}).get("enabled", False):
        skew = measure_skew(CONFIG, apply=True)
        plot_skew(skew, CONFIG)
        print(skew.summary())
        print()

    # 0b. helium wavelength-offset calibration (applied to all later loads)
    offset = None
    if CONFIG.get("helium_cal", {}).get("enabled", False):
        offset = compute_helium_offset(CONFIG, apply=True)
        plot_helium(offset, CONFIG)
        print(offset.summary())
        print()

    runs = sys.argv[1:] or list_runs()
    results = {}
    for r in runs:
        res = run_pipeline(r, plots=True)
        results[res.run] = res
        print(res.summary())
        print()

    if len(results) >= 1:
        report = write_report(results, CONFIG, offset, skew)
        if len(results) > 1:
            comb = combine_runs(results)
            plot_combined(results)
            print("=" * 60)
            print(f"Combined over {comb['n_runs']} runs "
                  f"(rel. cal {comb['rel_cal']:.2e}, offset err {comb['offset_err']:.4f} nm):")
            for k, lab in (("lambda_H", "lambda_H    "), ("lambda_D", "lambda_D    "),
                           ("dlambda", "Delta lambda")):
                print(f"  {lab} = {comb[k]:.4f} +/- {comb[k+'_err']:.4f} nm "
                      f"(stat {comb[k+'_stat']:.4f}, cal {comb[k+'_cal']:.1e}, "
                      f"read {comb[k+'_read']:.1e})")
            print(f"  (plain mean Delta lambda = {comb['dlambda_mean']:.4f} "
                  f"+/- {comb['dlambda_sem']:.4f} nm SEM; scatter {comb['dlambda_std']:.4f})")
            if _lineshape(CONFIG) == "voigt":
                pw = physical_widths(results, offset, CONFIG)
                print(f"  Voigt widths: Gaussian sigma {pw['sigma_gauss']:.4f} nm, "
                      f"Lorentzian gamma {pw['gamma_lor']:.4f} nm")
                if np.isfinite(pw.get("T_doppler", float("nan"))):
                    print(f"  -> instrumental sigma {pw['sigma_inst']:.4f} nm (He), "
                          f"Doppler T ~ {pw['T_doppler']:.0f} K")
        print(f"\nReport written to: {report}")
    print(f"Plots written to: {CONFIG['save_dir']}/")
