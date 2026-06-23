"""
Brownian-motion bead tracker (trackpy core + OpenCV viewer).

This uses trackpy (the standard Crocker-Grier particle-tracking library) for
the heavy lifting and keeps a custom OpenCV window for visualisation:

    detect   ->  trackpy.batch        (subpixel feature finding, bandpass)
    link     ->  trackpy.link         (combinatorial linking + memory)
    stubs    ->  trackpy.filter_stubs (drop short spurious tracks)
    clean    ->  physics-aware cascade (auto mass cut, jump + alpha gates) that
                 strips the noise lobe from the real-bead lobe -- see --no-clean
    analyse  ->  trackpy.emsd         (ensemble MSD -> diffusion coefficient)
    show     ->  OpenCV               (bounding boxes + trajectory traces)

Because the beads here are faint, low-contrast, phase-contrast objects on a
*bright* background, trackpy is run with invert=True.  The defaults below were
tuned on input/t1.tif (300 frames, 768x1024, uint8):

    diameter=11  minmass=100  search-range=18  memory=12  min-track=25

The expensive detect+link step is cached to a CSV next to the .tif, so the
viewer reopens instantly.  Pass --recompute after changing any detection or
linking parameter.

Usage
-----
    python track_blobs.py input/t1.tif
    python track_blobs.py input/t1.tif --fps 30 --mpp 0.5      # physical units
    python track_blobs.py input/t1.tif --minmass 120 --recompute
    python track_blobs.py input/t1.tif --no-show --save out.mp4

    # Full Stokes-Einstein analysis with uncertainty propagation:
    python track_blobs.py input/t1.tif --fps 30 --mpp 50/360 \
                          --bead-diameter 1.0 --bead-diameter-err 0.05 \
                          --water-temp 21.5 --water-temp-err 0.5 --mpp-err 0.002

Giving --bead-diameter (um), --water-temp (degC) and --mpp turns the measured
diffusion coefficient into the Boltzmann constant and N_A via
D = kB*T / (6*pi*eta*r).  The *-err options feed a full quadrature error budget
(statistics + calibration + bead size + temperature) through to kB and N_A.

All artefacts (trajectory CSV, four-panel summary, trajectory map, displacement
histogram, *_analysis.txt) are written to --outdir (default outputs/), named by
the input stem -- never into the input folder.

Controls (in the window)
------------------------
    space   play / pause
    n / d   next frame
    p / a   previous frame
    [ / ]   slower / faster
    t       toggle trajectory trails
    b       toggle bounding boxes
    r       restart from frame 0
    q / esc quit
"""

import argparse
import os
import warnings

import cv2
import numpy as np
import tifffile as tiff

warnings.filterwarnings("ignore")          # silence trackpy/pandas chatter
import trackpy as tp                        # noqa: E402

tp.quiet()


# --------------------------------------------------------------------------- #
#  Physical constants (CODATA / SI)
# --------------------------------------------------------------------------- #
KB_LIT = 1.380649e-23        # Boltzmann constant      [J/K]   (exact, SI 2019)
NA_LIT = 6.02214076e23       # Avogadro constant       [1/mol] (exact, SI 2019)
R_GAS = 8.314462618          # molar gas constant      [J/(mol*K)]


# Vogel-type empirical fit for water viscosity:  eta = A * 10**(B / (T - C)).
# Accurate to ~1% over 0-100 degC (gives 1.002 mPa*s at 20 degC, tabulated).
VISC_A = 2.414e-5            # Pa*s
VISC_B = 247.8              # K
VISC_C = 140.0             # K


def water_viscosity(temp_c):
    """Dynamic viscosity of liquid water [Pa*s] at temperature [degC]."""
    T = temp_c + 273.15
    return VISC_A * 10.0 ** (VISC_B / (T - VISC_C))


def dln_viscosity_dT(temp_c):
    """d(ln eta)/dT  [1/K] for the Vogel fit, used in error propagation."""
    T = temp_c + 273.15
    return -np.log(10.0) * VISC_B / (T - VISC_C) ** 2


def stokes_einstein(D_um2_s, bead_diameter_um, temp_c,
                    D_rel_err_stat=0.0, mpp_rel_err=0.0,
                    bead_diameter_rel_err=0.0, temp_err_c=0.0):
    """
    Invert the Stokes-Einstein relation to recover fundamental constants, with
    full first-order (quadrature) uncertainty propagation.

        D = kB * T / (6 * pi * eta(T) * r)     (translational diffusion, 3D drag)
        =>  kB = D * 6 * pi * eta(T) * r / T

    Inputs (all relative errors are dimensionless fractions, sigma/value):
        D_um2_s              measured diffusion coefficient        [um^2/s]
        bead_diameter_um     bead diameter                         [um]
        temp_c               water temperature                     [degC]
        D_rel_err_stat       statistical rel. error on D (e.g. bead-to-bead SEM)
        mpp_rel_err          rel. error on the microns/pixel calibration
        bead_diameter_rel_err rel. error on the bead diameter
        temp_err_c           absolute temperature uncertainty      [degC == K]

    Error budget (each term is a relative contribution to kB, added in
    quadrature):
        D statistical :  sigma_D_stat / D
        D calibration :  2 * sigma_mpp / mpp        (since D ~ mpp**2)
        bead radius   :  sigma_r / r  =  sigma_d / d
        temperature   :  |d ln(eta/T)/dT| * sigma_T  (eta AND 1/T both move
                         with T and are fully correlated, so combine first)

    Returns a dict of physical quantities plus 'kB_err'/'NA_err' (absolute) and
    a 'budget' sub-dict of the relative contributions.
    """
    import math

    T = temp_c + 273.15                          # K
    eta = water_viscosity(temp_c)                # Pa*s
    r = (bead_diameter_um / 2.0) * 1e-6          # m
    D_si = D_um2_s * 1e-12                        # m^2/s

    drag = 6.0 * math.pi * eta * r               # Stokes drag coefficient gamma
    kB = D_si * drag / T                          # measured Boltzmann constant
    NA = R_GAS / kB if kB > 0 else float("nan")   # measured Avogadro number

    D_theory_si = KB_LIT * T / drag               # expected D from literature kB
    D_theory = D_theory_si * 1e12                 # back to um^2/s

    # --- error budget (relative contributions to kB) --------------------- #
    rel_D_cal = 2.0 * mpp_rel_err                 # D proportional to mpp^2
    rel_D_total = math.hypot(D_rel_err_stat, rel_D_cal)
    rel_r = bead_diameter_rel_err                 # radius = diameter / 2
    # Temperature: kB ~ eta(T)/T, so combine the two correlated T-dependencies.
    dlnk_dT = dln_viscosity_dT(temp_c) - 1.0 / T
    rel_T = abs(dlnk_dT) * temp_err_c

    budget = {
        "D_stat": D_rel_err_stat,
        "D_calibration": rel_D_cal,
        "bead_radius": rel_r,
        "temperature": rel_T,
    }
    rel_kB = math.sqrt(sum(v ** 2 for v in budget.values()))

    out = {
        "T_K": T,
        "eta_Pa_s": eta,
        "radius_m": r,
        "D_um2_s": D_um2_s,
        "D_si": D_si,
        "drag_gamma": drag,
        "kB": kB,
        "kB_pct_err": 100.0 * (kB - KB_LIT) / KB_LIT,
        "NA": NA,
        "NA_pct_err": 100.0 * (NA - NA_LIT) / NA_LIT,
        "D_theory_um2_s": D_theory,
        "D_pct_err": 100.0 * (D_um2_s - D_theory) / D_theory,
        "rel_kB": rel_kB,                          # = rel_NA (R exact)
        "kB_err": abs(kB) * rel_kB,
        "NA_err": abs(NA) * rel_kB,
        "D_rel_err_total": rel_D_total,
        "budget": budget,                          # relative contributions
    }
    return out


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_tif_stack(path):
    """Load a TIFF / TIFF stack as a uint array of shape (time, y, x)."""
    with tiff.TiffFile(path) as tf:
        arr = tf.series[0].asarray()
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)[None, :, :]
    elif arr.ndim == 4:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., 0]
        arr = arr[:, 0, :, :]
    elif arr.ndim > 4:
        arr = arr.reshape((-1,) + arr.shape[-2:])

    print(f"Loaded {path}  ->  {arr.shape[0]} frames of "
          f"{arr.shape[2]}x{arr.shape[1]} ({arr.dtype})")
    return arr


def to_uint8(frame, low_pct=0.5, high_pct=99.5):
    """Contrast-stretch any dtype into uint8 for display."""
    if frame.dtype == np.uint8:
        return frame
    f = frame.astype(np.float32)
    lo, hi = np.percentile(f, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1
    f = np.clip((f - lo) / (hi - lo), 0, 1)
    return (255 * f).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Detection + linking  (trackpy)
# --------------------------------------------------------------------------- #
def analyse(frames, cfg):
    """
    Run trackpy detection + linking + stub filtering.

    Returns a trajectory DataFrame with columns including
    x, y, mass, size, frame, particle.
    """
    # numba JIT-compiles trackpy's subpixel-refinement hot loop, the dominant
    # cost of locate(). With it ~0.09 s/frame; without it ~0.7 s/frame (~8x).
    # trackpy uses it automatically (engine='auto'); this just reports status.
    try:
        from trackpy import try_numba
        accel = "ON (numba)" if try_numba.NUMBA_AVAILABLE else \
            "OFF - pip install numba for ~8x speedup"
    except Exception:
        accel = "unknown"
    print(f"Detecting beads with trackpy "
          f"(diameter={cfg.diameter}, minmass={cfg.minmass}, invert=True) "
          f"| acceleration: {accel}")
    n = len(frames)
    if cfg.processes == 1:
        # Locate frame-by-frame so we can show progress.
        import pandas as pd
        parts = []
        for i in range(n):
            f = tp.locate(frames[i], cfg.diameter, invert=True,
                          minmass=cfg.minmass)
            f["frame"] = i
            parts.append(f)
            print(f"\r  detecting frame {i + 1}/{n} "
                  f"({100 * (i + 1) // n}%)", end="", flush=True)
        print()
        features = pd.concat(parts).reset_index(drop=True)
    else:
        features = tp.batch(frames, cfg.diameter, invert=True,
                            minmass=cfg.minmass, processes=cfg.processes)
    print(f"  found {len(features)} raw features "
          f"({len(features) / n:.1f}/frame)")

    print(f"Linking (search_range={cfg.search_range}, memory={cfg.memory})...")
    # No motion predictor: the motion is Brownian (random), so velocity
    # prediction actually *hurts* the linking here.
    traj = tp.link(features, search_range=cfg.search_range, memory=cfg.memory)

    traj = tp.filter_stubs(traj, cfg.min_track).reset_index(drop=True)
    n = traj["particle"].nunique()
    print(f"  {n} trajectories survive filter_stubs(>= {cfg.min_track} frames)")
    return traj


# --------------------------------------------------------------------------- #
#  Noise rejection  (physics-aware trajectory cleaning)
# --------------------------------------------------------------------------- #
def otsu_threshold(values, nbins=128):
    """
    Otsu's method: the threshold that best splits a bimodal 1-D distribution
    into two classes by maximising between-class variance.  We run it on
    log10(mass), where faint noise detections (a tight peak just above minmass)
    and real beads (a heavier, separate peak) form two clear lobes -- so the
    cut lands automatically in the valley between them, no hand-tuning.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or v.min() == v.max():
        return float("nan")
    hist, edges = np.histogram(v, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w = hist.astype(float)
    wB = np.cumsum(w)
    wF = w.sum() - wB
    mB = np.cumsum(w * centers)
    mT = mB[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        muB = mB / wB
        muF = (mT - mB) / wF
        between = wB * wF * (muB - muF) ** 2
    return float(centers[np.nanargmax(between)])


def _track_features(traj):
    """Per-track summary used by the cleaner: length, mass level/stability,
    largest single-frame jump, spatial spread, and the MSD power-law exponent
    alpha (1 = ordinary Brownian, ->0 = stuck, ->2 = drift/ballistic)."""
    recs = {}
    for pid, grp in traj.sort_values("frame").groupby("particle"):
        g = grp.sort_values("frame")
        f = g["frame"].values
        x = g["x"].values
        y = g["y"].values
        step = np.hypot(np.diff(x), np.diff(y))[np.diff(f) == 1]
        mass = g["mass"].values

        # Per-track anomalous exponent from its own short-lag MSD.
        fidx = {int(fr): i for i, fr in enumerate(f)}
        L, M = [], []
        for lag in range(1, min(20, len(f))):
            sq = [(x[j] - x[i]) ** 2 + (y[j] - y[i]) ** 2
                  for i, fr in enumerate(f)
                  if (j := fidx.get(int(fr) + lag)) is not None]
            if len(sq) >= 3:
                L.append(lag)
                M.append(np.mean(sq))
        alpha = (np.polyfit(np.log(L), np.log(M), 1)[0]
                 if len(L) >= 4 else float("nan"))

        recs[int(pid)] = dict(
            n=len(g),
            med_mass=float(np.median(mass)),
            mass_cv=float(np.std(mass) / np.mean(mass)) if np.mean(mass) else 0.0,
            med_step=float(np.median(step)) if step.size else 0.0,
            max_jump=float(step.max()) if step.size else 0.0,
            rg=float(np.sqrt(((x - x.mean()) ** 2 +
                              (y - y.mean()) ** 2).mean())),
            alpha=float(alpha),
        )
    import pandas as pd
    # orient='index' -> rows = particle ids, columns = features, numeric dtype
    # (a plain DataFrame(recs).T would come back as object dtype).
    return pd.DataFrame.from_dict(recs, orient="index")


def subtract_drift(traj):
    """
    Remove collective stage / fluid drift: at each frame, the ensemble-mean
    displacement is the common motion every bead shares.  Subtracting its
    cumulative sum leaves only each bead's own random walk, so a slow pan no
    longer masquerades as ballistic (alpha->2) motion or inflates D.
    """
    import pandas as pd
    t = traj.sort_values(["particle", "frame"]).copy()
    t["dx"] = t.groupby("particle")["x"].diff()
    t["dy"] = t.groupby("particle")["y"].diff()
    drift = t.groupby("frame")[["dx", "dy"]].mean().fillna(0.0)
    drift = drift.cumsum()                       # cumulative common trajectory
    t = t.merge(drift.rename(columns={"dx": "dxc", "dy": "dyc"}),
                left_on="frame", right_index=True, how="left")
    t["x"] = t["x"] - t["dxc"].fillna(0.0)
    t["y"] = t["y"] - t["dyc"].fillna(0.0)
    return t.drop(columns=["dx", "dy", "dxc", "dyc"]).reset_index(drop=True)


def clean_trajectories(traj, cfg):
    """
    Cascade of physics-aware filters that drop whole spurious tracks while
    leaving genuine Brownian beads untouched.  Every stage is configurable and
    each one's toll is reported, so the funnel is transparent rather than a
    black box.  Operates purely on the linked table (no re-linking needed), so
    it also cleans cached results.

    Stages (each removes entire tracks):
      1. mass       median per-track mass must clear a threshold.  Default is
                    auto (Otsu on log-mass) which finds the noise/bead valley.
      2. mass_cv    reject flickering detections whose mass is wildly unstable.
      3. jump       reject tracks containing an implausible single-frame jump
                    (a hallmark of the linker bridging two unrelated features).
      4. alpha      reject tracks whose motion isn't diffusive (stuck or drift).
      5. rg         optionally reject near-immobile tracks (stuck dirt).
    """
    tf = _track_features(traj)
    n0 = len(tf)
    report = [("linked + stubs", n0, 0)]
    diag = {"tf": tf}

    # --- resolve thresholds (auto where requested) ----------------------- #
    if cfg.mass_threshold is not None:
        mass_thr = cfg.mass_threshold
        diag["mass_auto"] = False
    else:
        log_thr = otsu_threshold(np.log10(tf["med_mass"].values))
        mass_thr = 10.0 ** log_thr if np.isfinite(log_thr) else 0.0
        diag["mass_auto"] = True
    diag["mass_thr"] = mass_thr

    if cfg.max_jump is not None:
        jump_thr = cfg.max_jump
    else:
        # jump_factor x the typical (per-track median) consecutive-frame step:
        # well beyond real Brownian steps, flagging linker bridges between
        # unrelated features.
        typ_step = float(np.median(tf["med_step"])) if len(tf) else 0.0
        jump_thr = cfg.jump_factor * max(typ_step, 1.0)
    diag["jump_thr"] = jump_thr

    import pandas as pd
    keep = pd.Series(True, index=tf.index)

    def stage(label, cond):
        nonlocal keep
        before = int(keep.sum())
        keep = keep & cond.reindex(keep.index).fillna(False)
        report.append((label, int(keep.sum()), before - int(keep.sum())))

    stage(f"mass >= {mass_thr:.0f}", tf["med_mass"] >= mass_thr)
    stage(f"mass_cv <= {cfg.mass_cv_max:g}", tf["mass_cv"] <= cfg.mass_cv_max)
    stage(f"max_jump <= {jump_thr:.1f}px", tf["max_jump"] <= jump_thr)
    if cfg.alpha_min > 0 or np.isfinite(cfg.alpha_max):
        # NaN alpha (too few points to estimate) is kept -- don't punish short
        # but otherwise clean tracks for an unmeasurable exponent.
        a = tf["alpha"]
        ok = ((a >= cfg.alpha_min) & (a <= cfg.alpha_max)) | a.isna()
        stage(f"alpha in [{cfg.alpha_min:g},{cfg.alpha_max:g}]", ok)
    if cfg.min_rg > 0:
        stage(f"rg >= {cfg.min_rg:g}px", tf["rg"] >= cfg.min_rg)

    kept_ids = set(keep.index[keep].astype(int))
    clean = traj[traj["particle"].isin(kept_ids)].reset_index(drop=True)
    diag["keep"] = keep
    return clean, report, diag


def print_filter_report(report):
    print("\nNoise rejection funnel (tracks):")
    for label, kept, removed in report:
        bar = "-" * min(40, removed // 10)
        if removed == 0 and label == report[0][0]:
            print(f"  {label:28s} {kept:4d}")
        else:
            print(f"  {label:28s} {kept:4d}   (-{removed}) {bar}")
    start, end = report[0][1], report[-1][1]
    pct = 100 * end / start if start else 0
    print(f"  => kept {end}/{start} tracks ({pct:.0f}%)")


def save_filter_diagnostics(diag, cfg, out_png):
    """Show *why* tracks were cut: the bimodal mass split, the jump-size cut,
    and the per-track alpha gate, with each threshold drawn in."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tf = diag["tf"]
    keep = diag["keep"]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))

    # (1) log mass with threshold
    a = axs[0]
    lm = np.log10(tf["med_mass"].values)
    a.hist(lm, bins=40, color="#bbbbbb", label="all tracks")
    a.hist(lm[keep.values], bins=40, color="#2ca02c", alpha=0.8, label="kept")
    a.axvline(np.log10(diag["mass_thr"]), color="#d62728", lw=2,
              label=f"cut = {diag['mass_thr']:.0f}"
                    f"{' (auto)' if diag['mass_auto'] else ''}")
    a.set_xlabel("log10(median per-track mass)")
    a.set_ylabel("tracks")
    a.set_title("(1) mass: noise vs bead lobes")
    a.legend(fontsize=8)

    # (2) max jump per track
    b = axs[1]
    b.hist(tf["max_jump"].values, bins=40, color="#bbbbbb", label="all")
    b.hist(tf["max_jump"][keep].values, bins=40, color="#2ca02c", alpha=0.8,
           label="kept")
    b.axvline(diag["jump_thr"], color="#d62728", lw=2,
              label=f"cut = {diag['jump_thr']:.1f}px")
    b.axvline(cfg.search_range, color="black", ls="--", lw=1,
              label=f"search_range = {cfg.search_range:g}")
    b.set_xlabel("largest single-frame jump (px)")
    b.set_title("(2) over-linking jumps")
    b.legend(fontsize=8)

    # (3) alpha gate
    c = axs[2]
    al = tf["alpha"].values
    al = al[np.isfinite(al)]
    c.hist(al, bins=40, range=(-0.2, 2.2), color="#bbbbbb", label="all")
    ak = tf["alpha"][keep].values
    ak = ak[np.isfinite(ak)]
    c.hist(ak, bins=40, range=(-0.2, 2.2), color="#2ca02c", alpha=0.8,
           label="kept")
    c.axvline(cfg.alpha_min, color="#d62728", lw=2)
    if np.isfinite(cfg.alpha_max):
        c.axvline(cfg.alpha_max, color="#d62728", lw=2,
                  label=f"gate [{cfg.alpha_min:g},{cfg.alpha_max:g}]")
    c.axvline(1.0, color="black", ls=":", lw=1, label="alpha = 1 (ideal)")
    c.set_xlabel("per-track MSD exponent alpha")
    c.set_title("(3) motion-type gate")
    c.legend(fontsize=8)

    fig.suptitle(f"Noise-rejection diagnostics  -  "
                 f"{os.path.basename(cfg.path)}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved filter diagnostics -> {out_png}")


def compute_msd(traj, cfg):
    """
    Ensemble mean-squared displacement and a diffusion coefficient from the
    short-lag slope (MSD = 4*D*t in 2D).

    Returns a dict with the ensemble MSD series plus the fitted quantities:
        em        ensemble MSD (pandas Series indexed by lag time in s)
        D         diffusion coefficient = slope/4   [um^2/s or px^2/s]
        slope     linear-fit slope                  [units^2/s]
        intercept localization-error floor          [units^2]
        n_fit     number of short-lag points used in the linear fit
        r2        coefficient of determination of that linear fit
        alpha     anomalous-diffusion exponent (MSD ~ t**alpha, ~1 = normal)
        loglog_A  prefactor of the power-law fit MSD = A * t**alpha
    Units are px^2/s if mpp==1, else um^2/s.
    """
    em = tp.emsd(traj, mpp=cfg.mpp, fps=cfg.fps,
                 max_lagtime=cfg.msd_max_lag)
    nan = float("nan")
    if len(em) < 2:
        return {"em": em, "D": nan, "slope": nan, "intercept": nan,
                "n_fit": 0, "r2": nan, "alpha": nan, "loglog_A": nan}
    lags, msd = em.index.values, em.values

    # Linear fit over the first msd_fit_frac of the curve.  MSD(t) = 4*D*t +
    # offset, where offset is the static localization-error floor in 2D.  Only
    # the short-lag, well-sampled part of the curve is statistically reliable
    # (fewer overlapping intervals contribute at long lags), so we fit a
    # leading fraction rather than the whole thing.
    n_fit = max(2, int(round(len(lags) * cfg.msd_fit_frac)))
    slope, intercept = np.polyfit(lags[:n_fit], msd[:n_fit], 1)

    # R^2 of the linear fit (how straight / diffusive the fitted window is).
    yfit = slope * lags[:n_fit] + intercept
    ss_res = float(np.sum((msd[:n_fit] - yfit) ** 2))
    ss_tot = float(np.sum((msd[:n_fit] - msd[:n_fit].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else nan

    # Power-law fit over the whole positive curve: MSD = A * t**alpha.  alpha
    # near 1 confirms ordinary (non-anomalous) Brownian diffusion.
    pos = (lags > 0) & (msd > 0)
    if pos.sum() >= 2:
        alpha, logA = np.polyfit(np.log(lags[pos]), np.log(msd[pos]), 1)
        loglog_A = float(np.exp(logA))
    else:
        alpha, loglog_A = nan, nan

    return {"em": em, "D": slope / 4.0, "slope": slope, "intercept": intercept,
            "n_fit": n_fit, "r2": r2, "alpha": float(alpha),
            "loglog_A": loglog_A}


def per_particle_diffusion(traj, cfg, n_fit):
    """
    Diffusion coefficient of every individual trajectory, from the same
    short-lag MSD slope used for the ensemble.  The spread of these per-bead
    values gives an empirical uncertainty on D (and hence on kB / N_A) without
    assuming anything about the noise model.

    Returns a 1-D numpy array of D values [um^2/s or px^2/s], one per particle
    that yields a positive slope.
    """
    im = tp.imsd(traj, mpp=cfg.mpp, fps=cfg.fps, max_lagtime=cfg.msd_max_lag)
    lags = im.index.values
    k = min(max(2, n_fit), len(lags))
    Ds = []
    for pid in im.columns:
        msd = im[pid].values[:k]
        good = np.isfinite(msd)
        if good.sum() >= 2:
            slope = np.polyfit(lags[:k][good], msd[good], 1)[0]
            if slope > 0:
                Ds.append(slope / 4.0)
    return np.asarray(Ds, dtype=float)


def save_msd_plot(msd_res, cfg, out_png):
    """Standalone ensemble-MSD figure: data, the linear D fit, and the
    extrapolated fit line across the whole curve so the fit window is obvious."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    em = msd_res["em"]
    D, intercept, n_fit = msd_res["D"], msd_res["intercept"], msd_res["n_fit"]
    unit = "um" if cfg.mpp != 1.0 else "px"
    lags, msd = em.index.values, em.values

    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(lags, msd, "o", ms=4, color="#1f77b4", label="ensemble MSD")
    # Fit line drawn across the full range (dashed past the fitted window).
    ax.plot(lags[:n_fit], 4 * D * lags[:n_fit] + intercept, "-", lw=2,
            color="#d62728",
            label=f"fit (first {n_fit} lags): D = {D:.3g} {unit}$^2$/s")
    ax.plot(lags, 4 * D * lags + intercept, "--", lw=1, color="#d62728",
            alpha=0.5)
    ax.axvspan(lags[0], lags[n_fit - 1], color="#d62728", alpha=0.07,
               label="fit window")
    txt = f"$R^2$ = {msd_res['r2']:.4f}\n" + r"$\alpha$ = " + \
          f"{msd_res['alpha']:.3f}"
    ax.text(0.97, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("lag time t (s)")
    ax.set_ylabel(f"MSD ({unit}$^2$)")
    ax.set_title("Brownian motion: ensemble MSD")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved MSD plot -> {out_png}")


def save_summary_figure(msd_res, Ds, phys, cfg, out_png):
    """
    Four-panel back-end summary:
        (A) ensemble MSD + linear D fit
        (B) log-log MSD + power-law (anomalous-exponent) fit
        (C) per-particle D histogram (empirical spread -> uncertainty)
        (D) text table of the physical results (only when --bead-diameter and
            --water-temp were supplied; otherwise a short note).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    em = msd_res["em"]
    D, intercept, n_fit = msd_res["D"], msd_res["intercept"], msd_res["n_fit"]
    unit = "um" if cfg.mpp != 1.0 else "px"
    lags, msd = em.index.values, em.values

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"Brownian motion analysis  -  {os.path.basename(cfg.path)}",
                 fontsize=13, fontweight="bold")

    # (A) ensemble MSD + fit
    a = axs[0, 0]
    a.plot(lags, msd, "o", ms=4, color="#1f77b4", label="ensemble MSD")
    a.plot(lags[:n_fit], 4 * D * lags[:n_fit] + intercept, "-", lw=2,
           color="#d62728", label=f"D = {D:.3g} {unit}$^2$/s")
    a.axvspan(lags[0], lags[n_fit - 1], color="#d62728", alpha=0.07)
    a.set_xlabel("lag time t (s)")
    a.set_ylabel(f"MSD ({unit}$^2$)")
    a.set_title(f"(A) ensemble MSD  ($R^2$={msd_res['r2']:.4f})")
    a.legend(fontsize=8)

    # (B) log-log MSD + power law
    b = axs[0, 1]
    pos = (lags > 0) & (msd > 0)
    b.loglog(lags[pos], msd[pos], "o", ms=4, color="#1f77b4", label="MSD")
    if np.isfinite(msd_res["alpha"]):
        b.loglog(lags[pos], msd_res["loglog_A"] * lags[pos] ** msd_res["alpha"],
                 "-", lw=2, color="#2ca02c",
                 label=r"$\alpha$ = " + f"{msd_res['alpha']:.3f}")
    b.set_xlabel("lag time t (s)")
    b.set_ylabel(f"MSD ({unit}$^2$)")
    b.set_title(r"(B) log-log MSD  (slope $\alpha$, 1 = normal)")
    b.legend(fontsize=8)

    # (C) per-particle D histogram
    c = axs[1, 0]
    if Ds.size:
        c.hist(Ds, bins=min(40, max(8, Ds.size // 8)), color="#9467bd",
               alpha=0.8, edgecolor="white")
        c.axvline(D, color="#d62728", lw=2,
                  label=f"ensemble D = {D:.3g}")
        c.axvline(float(np.median(Ds)), color="black", lw=1.5, ls="--",
                  label=f"median = {np.median(Ds):.3g}")
        c.legend(fontsize=8)
    else:
        c.text(0.5, 0.5, "no per-particle D", ha="center", va="center",
               transform=c.transAxes)
    c.set_xlabel(f"per-particle D ({unit}$^2$/s)")
    c.set_ylabel("count")
    c.set_title(f"(C) per-particle D  (n = {Ds.size})")

    # (D) physical results table
    d = axs[1, 1]
    d.axis("off")
    lines = ["(D) Diffusion / physical results", ""]
    Dmean = float(np.mean(Ds)) if Ds.size else float("nan")
    Dsem = float(np.std(Ds, ddof=1) / np.sqrt(Ds.size)) if Ds.size > 1 \
        else float("nan")
    lines.append(f"ensemble D      = {D:.4g} {unit}^2/s")
    if Ds.size:
        lines.append(f"per-bead D      = {Dmean:.4g} +/- {Dsem:.2g} "
                     f"(SEM, n={Ds.size})")
    lines.append(f"anomalous exp.  alpha = {msd_res['alpha']:.3f}")
    lines.append(f"linear-fit R^2  = {msd_res['r2']:.4f}")
    if phys is not None:
        rel = phys["rel_kB"]
        big = max(phys["budget"], key=phys["budget"].get) if rel > 0 else "-"
        lines += [
            "",
            f"T               = {phys['T_K']:.2f} K "
            f"({cfg.water_temp:.1f} degC)",
            f"water viscosity = {phys['eta_Pa_s'] * 1e3:.4f} mPa*s",
            f"bead radius     = {phys['radius_m'] * 1e9:.1f} nm",
            "",
            f"kB  = {phys['kB']:.3e} +/- {phys['kB_err']:.1e}",
            f"      ({phys['kB_pct_err']:+.1f} % vs literature)",
            f"N_A = {phys['NA']:.3e} +/- {phys['NA_err']:.1e}",
            f"      ({phys['NA_pct_err']:+.1f} % vs literature)",
            f"rel. uncertainty = {100 * rel:.1f} %  (dominant: {big})",
            "",
            f"D (theory)      = {phys['D_theory_um2_s']:.4g} um^2/s"
            f"   (meas {phys['D_pct_err']:+.1f} %)",
        ]
    else:
        lines += ["", "Pass --bead-diameter and --water-temp",
                  "(and --mpp) to recover kB and N_A."]
    d.text(0.0, 1.0, "\n".join(lines), transform=d.transAxes, va="top",
           ha="left", family="monospace", fontsize=9.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved summary figure -> {out_png}")


def save_tracks_plot(traj, cfg, out_png):
    """Trajectory map: every linked bead path, colored per particle.  A quick
    visual sanity check that the linker followed real beads (smooth random
    walks) and not noise (jagged jumps across the field)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = "um" if cfg.mpp != 1.0 else "px"
    s = cfg.mpp if cfg.mpp != 1.0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 7 * cfg._aspect))
    n = 0
    for _, grp in traj.sort_values("frame").groupby("particle"):
        g = grp.sort_values("frame")
        ax.plot(g["x"].values * s, g["y"].values * s, "-", lw=0.6, alpha=0.7)
        n += 1
    ax.set_aspect("equal")
    ax.invert_yaxis()                       # image coordinates: y grows downward
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"Tracked trajectories  (n = {n})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved trajectory map -> {out_png}")


def save_displacement_hist(traj, cfg, out_png):
    """
    Histogram of single-frame step displacements with a zero-mean Gaussian
    overlay.  For genuine Brownian motion each coordinate's step distribution
    is Gaussian with variance 2*D*dt; a good Gaussian fit is strong evidence
    the motion is diffusive rather than drift- or vibration-dominated.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = "um" if cfg.mpp != 1.0 else "px"
    s = cfg.mpp if cfg.mpp != 1.0 else 1.0

    dx, dy = [], []
    for _, grp in traj.sort_values("frame").groupby("particle"):
        g = grp.sort_values("frame")
        # Only consecutive frames count as a single-step displacement.
        f = g["frame"].values
        x = g["x"].values * s
        y = g["y"].values * s
        step = np.diff(f) == 1
        dx.append(np.diff(x)[step])
        dy.append(np.diff(y)[step])
    dx = np.concatenate(dx) if dx else np.array([])
    dy = np.concatenate(dy) if dy else np.array([])
    steps = np.concatenate([dx, dy])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if steps.size:
        sigma = float(np.std(steps))
        counts, bins, _ = ax.hist(steps, bins=60, density=True, alpha=0.75,
                                  color="#17becf", edgecolor="white",
                                  label="x & y steps")
        xs = np.linspace(bins[0], bins[-1], 300)
        gauss = np.exp(-xs ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(xs, gauss, "-", lw=2, color="#d62728",
                label=f"Gaussian  $\\sigma$={sigma:.3g} {unit}")
        ax.set_title("Single-step displacement distribution")
    else:
        ax.text(0.5, 0.5, "no consecutive-frame steps", ha="center",
                va="center", transform=ax.transAxes)
    ax.set_xlabel(f"displacement per frame ({unit})")
    ax.set_ylabel("probability density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved displacement histogram -> {out_png}")


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def color_for_id(pid):
    """Deterministic bright BGR colour per particle ID."""
    h = int((pid * 2654435761) & 0xFFFFFFFF) % 180
    b, g, r = cv2.cvtColor(np.uint8([[[h, 200, 255]]]),
                           cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def build_lookup(traj):
    """
    Pre-index the trajectory table for fast per-frame drawing.

    by_frame[f] -> list of (pid, x, y, size) detected in frame f
    paths[pid]  -> Nx3 float array of (frame, x, y) sorted by frame
    """
    by_frame = {}
    for row in traj.itertuples(index=False):
        by_frame.setdefault(int(row.frame), []).append(
            (int(row.particle), float(row.x), float(row.y),
             float(getattr(row, "size", 3.0))))

    paths = {}
    for pid, grp in traj.sort_values("frame").groupby("particle"):
        paths[int(pid)] = grp[["frame", "x", "y"]].values.astype(np.float32)
    return by_frame, paths


def render(gray, by_frame, paths, idx, total, playing, fps,
           show_boxes, show_trails, trail_length, n_tracks):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Bounding boxes for beads detected in this exact frame.
    if show_boxes:
        for pid, x, y, size in by_frame.get(idx, []):
            half = int(np.clip(round(2.0 * size), 4, 40))
            col = color_for_id(pid)
            cv2.rectangle(vis, (int(x) - half, int(y) - half),
                          (int(x) + half, int(y) + half), col, 1)
            cv2.putText(vis, str(pid), (int(x) - half, int(y) - half - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    # Trajectory traces: the recent path of every particle active near now.
    if show_trails:
        lo = idx - trail_length
        for pid, path in paths.items():
            m = (path[:, 0] <= idx) & (path[:, 0] > lo)
            if m.sum() < 2:
                continue
            pts = path[m, 1:].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, color_for_id(pid), 1, cv2.LINE_AA)

    status = "playing" if playing else "paused"
    live = len(by_frame.get(idx, []))
    msg = (f"Frame {idx + 1}/{total} | {status} | fps={fps:.1f} | "
           f"beads now={live} | tracks total={n_tracks}")
    cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def view(frames, traj, cfg):
    by_frame, paths = build_lookup(traj)
    n_tracks = traj["particle"].nunique()
    total = len(frames)

    writer = None
    if cfg.save:
        h, w = frames.shape[1], frames.shape[2]
        writer = cv2.VideoWriter(cfg.save,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 cfg.fps_play, (w, h))

    window = "Brownian Bead Tracker (trackpy)"
    if cfg.show:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, frames.shape[2], frames.shape[1])
        print("\nControls: space=play/pause  n/p=step  [ ]=speed  "
              "t=trails  b=boxes  r=restart  q=quit\n")

    i = 0
    playing = True
    fps = cfg.fps_play
    show_boxes = True
    show_trails = True

    while True:
        gray = to_uint8(frames[i])
        display = render(gray, by_frame, paths, i, total, playing, fps,
                         show_boxes, show_trails, cfg.trail_length, n_tracks)

        if writer is not None:
            writer.write(display)
            i += 1
            if i >= total:
                break
            continue

        cv2.imshow(window, display)
        key = cv2.waitKeyEx(max(1, int(1000 / fps)) if playing else 0)
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key in (ord("n"), ord("d")):
            playing = False
            i = (i + 1) % total
        elif key in (ord("p"), ord("a")):
            playing = False
            i = (i - 1) % total
        elif key == ord("]"):
            fps = min(120, fps * 1.25)
        elif key == ord("["):
            fps = max(0.5, fps / 1.25)
        elif key == ord("t"):
            show_trails = not show_trails
        elif key == ord("b"):
            show_boxes = not show_boxes
        elif key == ord("r"):
            i = 0
        elif playing and key == -1:
            i += 1
            if i >= total:
                i = total - 1
                playing = False

    if writer is not None:
        writer.release()
        print(f"Saved annotated video -> {cfg.save}")
    if cfg.show:
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def ratio(s):
    """
    argparse type that accepts a plain float ("0.139") or a simple ratio
    ("50/360") and returns the float.  Lets you pass a calibration directly
    as known_microns / measured_pixels without pre-dividing.
    """
    s = s.strip()
    if "/" in s:
        num, den = s.split("/", 1)
        return float(num) / float(den)
    return float(s)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="Path to the .tif / .tiff stack")

    d = p.add_argument_group("detection (trackpy.locate/batch)")
    d.add_argument("--diameter", type=int, default=11,
                   help="Expected bead diameter in px (odd integer)")
    d.add_argument("--minmass", type=float, default=100.0,
                   help="Minimum integrated brightness. Higher = fewer, "
                        "stronger detections (less sensitive to noise)")
    d.add_argument("--processes", default=1,
                   help="trackpy worker processes (int or 'auto'). Keep 1 on "
                        "Windows unless run as a guarded script")

    l = p.add_argument_group("linking (trackpy.link/filter_stubs)")
    l.add_argument("--search-range", type=float, default=18.0,
                   help="Max px a bead may move between frames")
    l.add_argument("--memory", type=int, default=12,
                   help="Frames a bead may vanish and still be re-linked")
    l.add_argument("--min-track", type=int, default=25,
                   help="Drop trajectories shorter than this many frames")

    c = p.add_argument_group("noise rejection (post-link trajectory cleaning)")
    c.add_argument("--no-clean", dest="clean", action="store_false",
                   help="Disable the physics-aware noise-rejection cascade")
    c.add_argument("--mass-threshold", type=float, default=None,
                   help="Min median per-track mass. Default: AUTO (Otsu on the "
                        "bimodal log-mass distribution finds the noise/bead "
                        "valley). Set a number to override")
    c.add_argument("--mass-cv-max", type=float, default=0.6,
                   help="Reject tracks whose per-track mass scatter (std/mean) "
                        "exceeds this -- flickering noise (default 0.6)")
    c.add_argument("--max-jump", type=float, default=None,
                   help="Reject tracks with any single-frame jump larger than "
                        "this many px (over-linking artefact). Default: AUTO "
                        "(--jump-factor x median jump)")
    c.add_argument("--jump-factor", type=float, default=6.0,
                   help="Auto max-jump = this multiple of the median per-track "
                        "jump (default 6; real Brownian steps are far smaller)")
    c.add_argument("--alpha-min", type=float, default=0.4,
                   help="Reject tracks whose MSD exponent alpha is below this "
                        "(stuck dirt -> alpha 0). Default 0.4; set 0 to disable")
    c.add_argument("--alpha-max", type=float, default=1.8,
                   help="Reject tracks whose MSD exponent alpha exceeds this "
                        "(residual drift/ballistic -> alpha 2). Default 1.8")
    c.add_argument("--min-rg", type=float, default=0.0,
                   help="Reject tracks with radius of gyration below this many "
                        "px (immobile dirt). Default 0 = off (avoids biasing D)")
    c.add_argument("--subtract-drift", action="store_true",
                   help="Remove the ensemble-mean (common) motion before MSD, "
                        "so stage/fluid drift doesn't inflate D or alpha")

    m = p.add_argument_group("analysis (MSD / diffusion)")
    m.add_argument("--fps", type=float, default=30.0,
                   help="ACQUISITION frame rate (for MSD time axis)")
    m.add_argument("--mpp", type=ratio, default=1.0,
                   help="Microns per pixel. Accepts a plain number (0.139) or "
                        "a ratio (50/360 = known_um/measured_px). Leave 1 to "
                        "work in pixel units. Required for physical results")
    m.add_argument("--msd-max-lag", type=int, default=40,
                   help="Max lag time (frames) for the MSD curve")
    m.add_argument("--msd-fit-frac", type=float, default=0.5,
                   help="Fraction of the MSD curve (short lags) used for the "
                        "linear D fit (default 0.5; the reported R^2 tells you "
                        "if the window stayed in the straight diffusive regime)")

    s = p.add_argument_group("physics (Stokes-Einstein: kB, N_A)")
    s.add_argument("--bead-diameter", "--bead_diameter", dest="bead_diameter",
                   type=ratio, default=None,
                   help="Bead diameter in MICRONS (plain number or ratio). "
                        "With --water-temp and --mpp this recovers the "
                        "Boltzmann constant and N_A")
    s.add_argument("--water-temp", "--water_temp", dest="water_temp",
                   type=float, default=None,
                   help="Water temperature in DEGREES CELSIUS. Sets the "
                        "viscosity used in the Stokes-Einstein inversion")
    s.add_argument("--bead-diameter-err", "--bead_diameter_err",
                   dest="bead_diameter_err", type=ratio, default=0.0,
                   help="1-sigma uncertainty on the bead diameter [um] "
                        "(e.g. manufacturer tolerance). Propagated to kB / N_A")
    s.add_argument("--water-temp-err", "--water_temp_err",
                   dest="water_temp_err", type=float, default=0.0,
                   help="1-sigma uncertainty on the water temperature [degC]. "
                        "Propagated through viscosity AND the explicit T")
    s.add_argument("--mpp-err", "--mpp_err", dest="mpp_err", type=ratio,
                   default=0.0,
                   help="1-sigma uncertainty on the microns/pixel calibration "
                        "[same units as --mpp]. Contributes 2x to D (D ~ mpp^2)")

    v = p.add_argument_group("viewer / output")
    v.add_argument("--fps-play", type=float, default=15.0,
                   help="Playback fps for the window / saved video")
    v.add_argument("--trail-length", type=int, default=80,
                   help="How many past frames of trajectory to draw")
    v.add_argument("--save", metavar="OUT.mp4",
                   help="Render the whole annotated stack to a video file")
    v.add_argument("--no-show", dest="show", action="store_false",
                   help="Skip the interactive window (e.g. with --save)")
    v.add_argument("--recompute", action="store_true",
                   help="Force re-running detect+link instead of using cache")
    v.add_argument("--outdir", default="outputs",
                   help="Directory for the trajectory CSV, figures and analysis "
                        "text (created if needed; default: outputs/)")
    return p.parse_args()


def main():
    cfg = parse_args()
    if isinstance(cfg.processes, str) and cfg.processes.isdigit():
        cfg.processes = int(cfg.processes)

    frames = load_tif_stack(cfg.path)
    cfg._aspect = frames.shape[1] / frames.shape[2]   # y/x, for the tracks plot

    # All generated artefacts (CSV cache, figures, analysis text) live in
    # --outdir, named by the input stem, instead of polluting the input folder.
    os.makedirs(cfg.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(cfg.path))[0]
    base = os.path.join(cfg.outdir, stem)

    cache = base + "_trajectories.csv"
    legacy_cache = os.path.splitext(cfg.path)[0] + "_trajectories.csv"
    if not os.path.exists(cache) and os.path.exists(legacy_cache):
        cache = legacy_cache                         # reuse an old in-place cache

    if os.path.exists(cache) and not cfg.recompute:
        import pandas as pd
        traj = pd.read_csv(cache)
        print(f"Loaded cached trajectories <- {cache} "
              f"({traj['particle'].nunique()} tracks). "
              f"Use --recompute to redo detection.")
    else:
        traj = analyse(frames, cfg)
        cache = base + "_trajectories.csv"           # always (re)write into outdir
        traj.to_csv(cache, index=False)
        print(f"Saved trajectories -> {cache}")

    # --- Noise rejection ---------------------------------------------------- #
    # clean_traj keeps image-aligned coordinates (for the viewer / tracks map);
    # analysis_traj may additionally have drift removed (for MSD / physics).
    clean_traj = traj
    if cfg.clean and len(traj):
        clean_traj, report, diag = clean_trajectories(traj, cfg)
        print_filter_report(report)
        save_filter_diagnostics(diag, cfg, base + "_filter_diagnostics.png")
        clean_cache = base + "_trajectories_clean.csv"
        clean_traj.to_csv(clean_cache, index=False)
        print(f"Saved cleaned trajectories -> {clean_cache}")
        if clean_traj["particle"].nunique() < 2:
            print("\n[warn] Cleaning left <2 tracks; relax the filters "
                  "(e.g. --mass-threshold / --alpha-min / --no-clean).")

    analysis_traj = clean_traj
    if cfg.subtract_drift and len(analysis_traj):
        analysis_traj = subtract_drift(analysis_traj)
        print("Subtracted ensemble drift before MSD analysis.")

    # --- Diffusion analysis ------------------------------------------------- #
    msd_res = compute_msd(analysis_traj, cfg)
    D, n_fit = msd_res["D"], msd_res["n_fit"]
    unit = "um^2/s" if cfg.mpp != 1.0 else "px^2/s"

    if not np.isfinite(D):
        print("\nNot enough trajectory overlap to estimate MSD/D.")
        view(frames, clean_traj, cfg)
        return

    print(f"\nDiffusion coefficient  D = {D:.4g} {unit}  "
          f"(MSD slope {msd_res['slope']:.4g}, fit over first {n_fit} lags, "
          f"R^2 = {msd_res['r2']:.4f}, 2D: MSD = 4*D*t + offset)")
    print(f"Anomalous-diffusion exponent  alpha = {msd_res['alpha']:.3f}  "
          f"(1.0 = ordinary Brownian diffusion)")

    # Per-particle spread -> statistical (bead-to-bead) uncertainty on D.
    Ds = per_particle_diffusion(analysis_traj, cfg, n_fit)
    D_rel_err_stat = 0.0
    if Ds.size > 1 and np.mean(Ds):
        D_sem = float(np.std(Ds, ddof=1) / np.sqrt(Ds.size))
        D_rel_err_stat = D_sem / abs(float(np.mean(Ds)))
        print(f"Per-particle D = {np.mean(Ds):.4g} +/- {D_sem:.2g} {unit} "
              f"(SEM over n = {Ds.size} beads, {100 * D_rel_err_stat:.1f}%)")

    # --- Stokes-Einstein physics (only if the bead/temperature are given) --- #
    phys = None
    if cfg.bead_diameter is not None and cfg.water_temp is not None:
        if cfg.mpp == 1.0:
            print("\n[warn] --mpp is 1.0 (pixel units); kB / N_A would be "
                  "meaningless. Re-run with the real microns-per-pixel.")
        else:
            mpp_rel = cfg.mpp_err / cfg.mpp if cfg.mpp else 0.0
            dia_rel = (cfg.bead_diameter_err / cfg.bead_diameter
                       if cfg.bead_diameter else 0.0)
            phys = stokes_einstein(
                D, cfg.bead_diameter, cfg.water_temp,
                D_rel_err_stat=D_rel_err_stat, mpp_rel_err=mpp_rel,
                bead_diameter_rel_err=dia_rel, temp_err_c=cfg.water_temp_err)
            print("\n--- Stokes-Einstein  (D = kB*T / (6*pi*eta*r)) ---")
            print(f"  T               = {phys['T_K']:.2f} K "
                  f"({cfg.water_temp:.1f} degC)")
            print(f"  water viscosity = {phys['eta_Pa_s'] * 1e3:.4f} mPa*s")
            print(f"  bead radius     = {phys['radius_m'] * 1e9:.1f} nm "
                  f"(diameter {cfg.bead_diameter:.3g} um)")
            print(f"  Stokes drag     = {phys['drag_gamma']:.3e} kg/s")
            print(f"  kB (measured)   = {phys['kB']:.4e} +/- "
                  f"{phys['kB_err']:.2e} J/K  "
                  f"(lit {KB_LIT:.4e}, {phys['kB_pct_err']:+.1f} %)")
            print(f"  N_A (measured)  = {phys['NA']:.4e} +/- "
                  f"{phys['NA_err']:.2e} /mol "
                  f"(lit {NA_LIT:.4e}, {phys['NA_pct_err']:+.1f} %)")
            print(f"  D (theory)      = {phys['D_theory_um2_s']:.4g} um^2/s "
                  f"(measured is {phys['D_pct_err']:+.1f} %)")
            _print_error_budget(phys)
            _write_analysis_txt(base + "_analysis.txt", msd_res, Ds, phys, cfg)

    # --- Graphics ----------------------------------------------------------- #
    save_msd_plot(msd_res, cfg, base + "_msd.png")
    save_summary_figure(msd_res, Ds, phys, cfg, base + "_summary.png")
    save_tracks_plot(clean_traj, cfg, base + "_tracks.png")       # image-aligned
    save_displacement_hist(analysis_traj, cfg, base + "_displacements.png")

    view(frames, clean_traj, cfg)


def _budget_lines(phys, indent=""):
    """Human-readable error budget: each source's relative contribution to kB
    (and N_A), in quadrature, sorted largest-first."""
    rel = phys["rel_kB"]
    items = sorted(phys["budget"].items(), key=lambda kv: -kv[1])
    out = [f"{indent}error budget (relative, added in quadrature):"]
    for name, val in items:
        frac = (val / rel) ** 2 * 100 if rel > 0 else 0.0   # % of variance
        out.append(f"{indent}  {name:<14}= {100 * val:6.2f} %"
                   f"   ({frac:4.0f} % of total variance)")
    out.append(f"{indent}  {'TOTAL kB/N_A':<14}= {100 * rel:6.2f} %")
    return out


def _print_error_budget(phys):
    for line in _budget_lines(phys, indent="  "):
        print(line)


def _write_analysis_txt(path, msd_res, Ds, phys, cfg):
    """Dump the numeric results next to the figures for the lab write-up."""
    unit = "um^2/s" if cfg.mpp != 1.0 else "px^2/s"
    Dmean = float(np.mean(Ds)) if Ds.size else float("nan")
    Dsem = float(np.std(Ds, ddof=1) / np.sqrt(Ds.size)) if Ds.size > 1 \
        else float("nan")
    lines = [
        f"Brownian motion analysis  -  {os.path.basename(cfg.path)}",
        "=" * 56,
        "",
        f"ensemble D        = {msd_res['D']:.6g} {unit}",
        f"per-particle D    = {Dmean:.6g} +/- {Dsem:.3g} {unit} "
        f"(SEM, n={Ds.size})",
        f"MSD linear-fit R2 = {msd_res['r2']:.5f} "
        f"(first {msd_res['n_fit']} lags)",
        f"anomalous alpha   = {msd_res['alpha']:.4f}",
        "",
        "inputs (value +/- 1 sigma):",
        f"  acquisition fps   = {cfg.fps}",
        f"  microns/pixel     = {cfg.mpp} +/- {cfg.mpp_err}",
        f"  bead diameter     = {cfg.bead_diameter} +/- "
        f"{cfg.bead_diameter_err} um",
        f"  water temperature = {cfg.water_temp} +/- {cfg.water_temp_err} degC",
        "",
        f"T                 = {phys['T_K']:.3f} K",
        f"water viscosity   = {phys['eta_Pa_s']:.6e} Pa*s",
        f"bead radius       = {phys['radius_m']:.6e} m",
        f"Stokes drag gamma = {phys['drag_gamma']:.6e} kg/s",
        "",
        f"kB  (measured)    = {phys['kB']:.6e} +/- {phys['kB_err']:.3e} J/K",
        f"kB  (literature)  = {KB_LIT:.6e} J/K   "
        f"({phys['kB_pct_err']:+.2f} %)",
        f"N_A (measured)    = {phys['NA']:.6e} +/- {phys['NA_err']:.3e} /mol",
        f"N_A (literature)  = {NA_LIT:.6e} /mol   "
        f"({phys['NA_pct_err']:+.2f} %)",
        f"D   (theory)      = {phys['D_theory_um2_s']:.6g} um^2/s   "
        f"(measured {phys['D_pct_err']:+.2f} %)",
        "",
    ] + _budget_lines(phys)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved analysis text -> {path}")


if __name__ == "__main__":
    main()
