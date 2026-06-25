# References for the Brownian-motion diffusion / Boltzmann-constant analysis

Citations underpinning `track_blobs.py`, with where each is used in the code.
Diffusion coefficient `D` here is the isotropic 2-D coefficient (MSD = 4·D·t).

---

## Physical theory

**Einstein, A. (1905).** *Über die von der molekularkinetischen Theorie der
Wärme geforderte Bewegung von in ruhenden Flüssigkeiten suspendierten
Teilchen.* Annalen der Physik **322**(8), 549–560.
doi:10.1002/andp.19053220806
- The Stokes–Einstein relation `D = kB·T / (6π·η·r)` and `⟨r²⟩ = 2d·D·t`.
- **Used in:** `stokes_einstein()` — inverting `D` to recover `kB` and `N_A`.

**Perrin, J. (1909).** *Mouvement brownien et réalité moléculaire.* Annales de
Chimie et de Physique **18**, 5–114.
- First determination of Avogadro's number from Brownian motion (1926 Nobel);
  the historical precedent for this whole experiment.
- **Used in:** report framing / motivation (no code).

---

## Detection and tracking

**Crocker, J. C. & Grier, D. G. (1996).** *Methods of Digital Video Microscopy
for Colloidal Studies.* Journal of Colloid and Interface Science **179**(1),
298–310. doi:10.1006/jcis.1996.0217
- The centroid feature-finding (subpixel) and nearest-neighbour linking
  algorithm used to detect and track the beads.
- **Used in:** `analyse()` via `trackpy.batch`/`locate` (detection) and
  `trackpy.link` + `filter_stubs` (linking).

**Allan, D. B., Caswell, T., Keim, N. C., van der Wel, C. M. & Verweij, R. W.
(2024).** *soft-matter/trackpy: v0.7.* Zenodo. doi:10.5281/zenodo.16089574
- The software implementation of Crocker–Grier that we actually run (v0.7).
- **Used in:** detection (`tp.batch`/`tp.locate`), linking (`tp.link`,
  `tp.filter_stubs`), MSD (`tp.emsd`, `tp.imsd`), drift (`tp.compute_drift`,
  `tp.subtract_drift`).

---

## Diffusion-coefficient estimation and its uncertainty

**Qian, H., Sheetz, M. P. & Elson, E. L. (1991).** *Single particle tracking.
Analysis of diffusion and flow in two-dimensional systems.* Biophysical Journal
**60**(4), 910–921. doi:10.1016/S0006-3495(91)82125-7
- Foundational treatment of the statistical accuracy of `D` extracted from an
  MSD curve (why short lags dominate; finite-trajectory scatter).
- **Used in:** the uncertainty analysis — per-particle `D` spread
  (`per_particle_diffusion()`) and the bead-level bootstrap
  (`bootstrap_ensemble_D()`).

**Michalet, X. (2010).** *Mean square displacement analysis of single-particle
trajectories with localization error: Brownian motion in an isotropic medium.*
Physical Review E **82**(4), 041914. doi:10.1103/PhysRevE.82.041914
- Reduced localization error `x = σ²/(D·Δt)` (Eq. 20) and the variance-optimal
  number of MSD points to fit, `p_min = ⌊2 + 2.7·√x⌋` (Eq. 30).
- **Used in:** `compute_msd()` (`x_red`), `michalet_pmin()`, and the fit-window
  selection + fit-window systematic in `main()`.

**Vestergaard, C. L., Blainey, P. C. & Flyvbjerg, H. (2014).** *Optimal
estimation of diffusion coefficients from single-particle trajectories.*
Physical Review E **89**(2), 022726. doi:10.1103/PhysRevE.89.022726
- The covariance-based estimator (CVE): an MSD-free, regression-free, unbiased,
  near-optimal `D` estimator that self-corrects for localization error.
- **Used in:** `cve_diffusion()` — the independent cross-check on the MSD `D`.

**Michalet, X. & Berglund, A. J. (2012).** *Optimal diffusion coefficient
estimation in single-particle tracking.* Physical Review E **85**(6), 061916.
doi:10.1103/PhysRevE.85.061916
- Unifies the optimal-MSD and CVE/MLE estimators and shows both approach the
  Cramér–Rao lower bound on the `D` uncertainty.
- **Used in:** justification for combining the Michalet fit-window choice with
  the CVE cross-check (report discussion).

**Berglund, A. J. (2010).** *Statistics of camera-based single-particle
tracking.* Physical Review E **82**(1), 011917.
doi:10.1103/PhysRevE.82.011917
- Motion-blur coefficient `R` (= 1/6 for uniform full-frame exposure) and the
  dynamic-localization-error model behind the CVE; also the basis for why
  *correlated* (non-white) localization error biases the CVE.
- **Used in:** `cve_diffusion()` `blur_R` (the `--blur-coeff` flag) for the
  localization-error estimate `σ_loc`; and the justification for flagging the
  CVE as an upper bound when pixel-locking is detected.

---

## Localization-quality diagnostic

**Pixel-locking / sub-pixel bias** — the sub-pixel-remainder histogram test
(positions piling up at integer pixels) is the standard check for biased,
*correlated* localization error; it is built into trackpy as `tp.subpx_bias`
(see the trackpy citation above) and discussed in the camera-tracking-error
literature (Berglund 2010; Savin & Doyle 2005, Biophys. J. **88**, 623–638,
doi:10.1529/biophysj.104.042457).
- **Used in:** `subpixel_bias()` — produces `*_subpixel_bias.png`; when the
  modulation exceeds `--pixel-lock-thresh`, the CVE is excluded from the
  `D_method` systematic and reported as an upper bound.
