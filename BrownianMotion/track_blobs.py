"""
Brownian-motion bead tracker (trackpy core + OpenCV viewer).

This uses trackpy (the standard Crocker-Grier particle-tracking library) for
the heavy lifting and keeps a custom OpenCV window for visualisation:

    detect   ->  trackpy.batch        (subpixel feature finding, bandpass)
    link     ->  trackpy.link         (combinatorial linking + memory)
    clean    ->  trackpy.filter_stubs (drop short spurious tracks)
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
    print(f"Detecting beads with trackpy "
          f"(diameter={cfg.diameter}, minmass={cfg.minmass}, invert=True)...")
    features = tp.batch(frames, cfg.diameter, invert=True,
                        minmass=cfg.minmass, processes=cfg.processes)
    print(f"  found {len(features)} raw features "
          f"({len(features) / len(frames):.1f}/frame)")

    print(f"Linking (search_range={cfg.search_range}, memory={cfg.memory})...")
    # No motion predictor: the motion is Brownian (random), so velocity
    # prediction actually *hurts* the linking here.
    traj = tp.link(features, search_range=cfg.search_range, memory=cfg.memory)

    traj = tp.filter_stubs(traj, cfg.min_track).reset_index(drop=True)
    n = traj["particle"].nunique()
    print(f"  {n} trajectories survive filter_stubs(>= {cfg.min_track} frames)")
    return traj


def compute_msd(traj, cfg):
    """
    Ensemble mean-squared displacement and a diffusion coefficient from the
    short-lag slope (MSD = 4*D*t in 2D).

    Returns (emsd_series, D, slope, n_fit) in the chosen units
    (px^2/s if mpp==1, else um^2/s).
    """
    em = tp.emsd(traj, mpp=cfg.mpp, fps=cfg.fps,
                 max_lagtime=cfg.msd_max_lag)
    if len(em) < 2:
        return em, float("nan"), float("nan"), float("nan"), 0
    lags, msd = em.index.values, em.values
    n_fit = max(2, int(len(lags) * cfg.msd_fit_frac))
    # MSD(t) = 4*D*t + offset (offset = localization-error floor in 2D).
    slope, intercept = np.polyfit(lags[:n_fit], msd[:n_fit], 1)
    return em, slope / 4.0, slope, intercept, n_fit


def save_msd_plot(em, D, intercept, n_fit, cfg, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unit = "um" if cfg.mpp != 1.0 else "px"
    lags, msd = em.index.values, em.values
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(lags, msd, "o", ms=4, label="ensemble MSD")
    fit_x = lags[:n_fit]
    ax.plot(fit_x, 4 * D * fit_x + intercept, "-", lw=2,
            label=f"fit: D = {D:.3g} {unit}$^2$/s")
    ax.set_xlabel("lag time t (s)")
    ax.set_ylabel(f"MSD ({unit}$^2$)")
    ax.set_title("Brownian motion: ensemble MSD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved MSD plot -> {out_png}")


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

    m = p.add_argument_group("analysis (MSD / diffusion)")
    m.add_argument("--fps", type=float, default=30.0,
                   help="ACQUISITION frame rate (for MSD time axis)")
    m.add_argument("--mpp", type=float, default=1.0,
                   help="Microns per pixel. Leave 1 to work in pixel units")
    m.add_argument("--msd-max-lag", type=int, default=40,
                   help="Max lag time (frames) for the MSD curve")
    m.add_argument("--msd-fit-frac", type=float, default=0.25,
                   help="Fraction of the MSD curve (short lags) used for the "
                        "linear D fit")

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
    return p.parse_args()


def main():
    cfg = parse_args()
    if isinstance(cfg.processes, str) and cfg.processes.isdigit():
        cfg.processes = int(cfg.processes)

    frames = load_tif_stack(cfg.path)

    cache = os.path.splitext(cfg.path)[0] + "_trajectories.csv"
    if os.path.exists(cache) and not cfg.recompute:
        import pandas as pd
        traj = pd.read_csv(cache)
        print(f"Loaded cached trajectories <- {cache} "
              f"({traj['particle'].nunique()} tracks). "
              f"Use --recompute to redo detection.")
    else:
        traj = analyse(frames, cfg)
        traj.to_csv(cache, index=False)
        print(f"Saved trajectories -> {cache}")

    # Diffusion analysis.
    em, D, slope, intercept, n_fit = compute_msd(traj, cfg)
    unit = "um^2/s" if cfg.mpp != 1.0 else "px^2/s"
    if np.isfinite(D):
        print(f"\nDiffusion coefficient  D = {D:.4g} {unit}  "
              f"(MSD slope {slope:.4g}, fit over first {n_fit} lags, "
              f"2D: MSD = 4*D*t + offset)")
        save_msd_plot(em, D, intercept, n_fit, cfg,
                      os.path.splitext(cfg.path)[0] + "_msd.png")
    else:
        print("\nNot enough trajectory overlap to estimate MSD/D.")

    view(frames, traj, cfg)


if __name__ == "__main__":
    main()
