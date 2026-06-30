"""
Calibration measuring tool (one-time, NOT part of the analysis pipeline).

Loads the central frame of a calibration .tif (a ruler/reticle imaged at the
data magnification), shows it in a zoomable, brightness-adjustable window, and
lets you click pairs of points to measure the HORIZONTAL (x-only) pixel distance
between ruler features.  For each measurement you enter the known physical
distance and its uncertainty; the tool derives metres-per-pixel (here in your
chosen unit per pixel) and propagates the uncertainty, then writes everything to
a JSON log so the calibration (and its error) flows into later calculations.

Only x is used: each click drops a vertical guide line and the distance is
|x2 - x1| in pixels, matching the horizontal-only flash displacement we measure.

Run (on your machine, needs a display):
    python calibration_tool.py
    python calibration_tool.py "data/D2 Calibration.tif" --units mm

Workflow in the window:
    * Zoom +/- (1x..8x) and drag the brightness slider until the ticks are clear.
    * Click the first ruler feature, then the second -> live dx (px) is shown.
    * Type the known distance, its uncertainty and a label, click "Add".
    * Repeat for as many gradations as you like, then "Save JSON".

JSON (outputs/calibration/<stem>_calibration.json) holds every measurement plus
a summary mpp with mean / std / SEM and an error-propagated estimate.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import tifffile


# --------------------------------------------------------------------------
# Pure helpers (no Tk) -- importable + unit-testable headless.
# --------------------------------------------------------------------------
def load_central_frame(path: str):
    """Return (frame_uint8, frame_index) for the central frame of a tif stack."""
    arr = tifffile.imread(path)
    if arr.ndim == 2:
        return arr.astype("uint8"), 0
    if arr.ndim == 3:
        idx = arr.shape[0] // 2
        return arr[idx].astype("uint8"), idx
    raise ValueError(f"unexpected calibration shape {arr.shape}")


def stretch(frame: np.ndarray, gain: float = 1.0, p_lo=1.0, p_hi=99.5):
    """Percentile contrast-stretch then apply a linear gain, for display only."""
    lo, hi = np.percentile(frame, [p_lo, p_hi])
    hi = max(hi, lo + 1)
    out = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo))
    out *= gain
    return np.clip(out, 0, 255).astype("uint8")


def measurement_mpp(dx_px: float, known: float, known_unc: float,
                    px_unc: float = 1.0):
    """metres(unit)-per-pixel for one measurement, with propagated uncertainty.

    dx uncertainty = sqrt(2) * px_unc (independent error on each clicked point).
    Relative errors add in quadrature: (s_mpp/mpp)^2 = (s_known/known)^2 + (s_dx/dx)^2.
    """
    dx_unc = math.sqrt(2.0) * px_unc
    mpp = known / dx_px
    rel = math.sqrt((known_unc / known) ** 2 + (dx_unc / dx_px) ** 2)
    return mpp, mpp * rel


def summarize(measurements, px_unc: float = 1.0):
    """Combine per-measurement mpp values into a summary with uncertainties."""
    if not measurements:
        return {}
    mpps, uncs = [], []
    for m in measurements:
        mpp, u = measurement_mpp(m["dx_px"], m["known_distance"],
                                 m["known_uncertainty"], px_unc)
        m["mpp"] = mpp
        m["mpp_uncertainty"] = u
        mpps.append(mpp)
        uncs.append(u)
    mpps = np.array(mpps)
    uncs = np.array(uncs)
    n = len(mpps)
    # inverse-variance (error-propagated) weighted mean
    w = 1.0 / uncs ** 2
    wmean = float((w * mpps).sum() / w.sum())
    wmean_unc = float(math.sqrt(1.0 / w.sum()))
    return {
        "n": n,
        "mpp_mean": float(mpps.mean()),
        "mpp_std": float(mpps.std(ddof=1)) if n > 1 else 0.0,
        "mpp_sem": float(mpps.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "mpp_weighted": wmean,
        "mpp_weighted_uncertainty": wmean_unc,
    }


def write_json(path, source, frame_idx, units, px_unc, measurements):
    summary = summarize(measurements, px_unc)
    payload = {
        "source": source,
        "frame": frame_idx,
        "units": units,
        "pixel_click_uncertainty_px": px_unc,
        "mpp_units": f"{units}/px",
        "measurements": measurements,
        "summary": summary,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return summary


# --------------------------------------------------------------------------
# Tk GUI
# --------------------------------------------------------------------------
def run_gui(path: str, units: str, px_unc: float):
    import tkinter as tk
    from tkinter import messagebox, ttk

    from PIL import Image, ImageTk

    frame, frame_idx = load_central_frame(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_json = os.path.join("outputs", "calibration", f"{stem}_calibration.json")

    root = tk.Tk()
    root.title(f"Calibration  -  {os.path.basename(path)}  (frame {frame_idx})")

    state = {"zoom": 1, "gain": 3.0, "points": [], "measurements": [],
             "photo": None, "disp": None}

    # ---- layout: controls on top, scrollable canvas below ----
    top = ttk.Frame(root, padding=6)
    top.pack(side="top", fill="x")

    cframe = ttk.Frame(root)
    cframe.pack(side="top", fill="both", expand=True)
    canvas = tk.Canvas(cframe, bg="black", cursor="crosshair")
    hbar = ttk.Scrollbar(cframe, orient="horizontal", command=canvas.xview)
    vbar = ttk.Scrollbar(cframe, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    hbar.grid(row=1, column=0, sticky="ew")
    cframe.rowconfigure(0, weight=1)
    cframe.columnconfigure(0, weight=1)

    status = tk.StringVar(value="Click two ruler features to measure x-distance.")
    ttk.Label(root, textvariable=status, relief="sunken", anchor="w").pack(
        side="bottom", fill="x")

    def redraw_image():
        disp = stretch(frame, state["gain"])
        img = Image.fromarray(disp)
        z = state["zoom"]
        if z != 1:
            img = img.resize((img.width * z, img.height * z), Image.NEAREST)
        state["disp"] = img
        state["photo"] = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        canvas.configure(scrollregion=(0, 0, img.width, img.height))
        draw_markers()

    def draw_markers():
        canvas.delete("marker")
        z = state["zoom"]
        h = frame.shape[0] * z
        for i, x in enumerate(state["points"]):
            cx = x * z
            canvas.create_line(cx, 0, cx, h, fill="#ff3030", width=1, tags="marker")
            canvas.create_text(cx + 3, 12 + 14 * i, anchor="w",
                               text=f"x{i+1}={x:.1f}", fill="#ffd000", tags="marker")
        if len(state["points"]) == 2:
            dx = abs(state["points"][1] - state["points"][0])
            status.set(f"dx = {dx:.1f} px   (enter known distance, then Add)")

    def on_click(event):
        x_img = canvas.canvasx(event.x) / state["zoom"]
        if x_img < 0 or x_img > frame.shape[1]:
            return
        pts = state["points"]
        if len(pts) >= 2:
            pts.clear()
        pts.append(x_img)
        draw_markers()

    canvas.bind("<Button-1>", on_click)

    def set_zoom(z):
        state["zoom"] = z
        redraw_image()

    def set_gain(v):
        state["gain"] = float(v)
        redraw_image()

    # ---- controls ----
    ttk.Label(top, text="Zoom:").pack(side="left")
    for z in (1, 2, 4, 8):
        ttk.Button(top, text=f"{z}x", width=3,
                   command=lambda z=z: set_zoom(z)).pack(side="left", padx=1)

    ttk.Label(top, text="  Brightness:").pack(side="left")
    gain_scale = ttk.Scale(top, from_=0.5, to=10.0, value=state["gain"],
                           length=140, command=set_gain)
    gain_scale.pack(side="left", padx=4)

    ttk.Label(top, text=f"  known dist ({units}):").pack(side="left")
    e_dist = ttk.Entry(top, width=8)
    e_dist.pack(side="left")
    ttk.Label(top, text="+/-").pack(side="left")
    e_unc = ttk.Entry(top, width=6)
    e_unc.pack(side="left")
    ttk.Label(top, text="label:").pack(side="left")
    e_lbl = ttk.Entry(top, width=10)
    e_lbl.pack(side="left", padx=(0, 4))

    def clear_points():
        state["points"].clear()
        draw_markers()
        status.set("Points cleared.")

    def add_measurement():
        if len(state["points"]) != 2:
            messagebox.showwarning("Need 2 points", "Click two features first.")
            return
        try:
            known = float(e_dist.get())
            unc = float(e_unc.get()) if e_unc.get().strip() else 0.0
        except ValueError:
            messagebox.showwarning("Bad input", "Distance/uncertainty must be numbers.")
            return
        if known <= 0:
            messagebox.showwarning("Bad input", "Known distance must be > 0.")
            return
        x1, x2 = state["points"]
        dx = abs(x2 - x1)
        m = {
            "label": e_lbl.get().strip() or f"m{len(state['measurements'])+1}",
            "x1": round(x1, 3), "x2": round(x2, 3), "dx_px": round(dx, 3),
            "known_distance": known, "known_uncertainty": unc,
        }
        mpp, u = measurement_mpp(dx, known, unc, px_unc)
        state["measurements"].append(m)
        tree.insert("", "end", values=(
            m["label"], f"{dx:.2f}", f"{known:g}", f"{unc:g}",
            f"{mpp:.5g}", f"{u:.2g}"))
        clear_points()
        status.set(f"Added '{m['label']}': dx={dx:.1f}px -> mpp={mpp:.5g} {units}/px")

    def delete_selected():
        sel = tree.selection()
        if not sel:
            return
        for item in sel:
            idx = tree.index(item)
            tree.delete(item)
            del state["measurements"][idx]

    def save_json():
        if not state["measurements"]:
            messagebox.showwarning("Nothing to save", "Add at least one measurement.")
            return
        summary = write_json(out_json, path, frame_idx, units, px_unc,
                             state["measurements"])
        msg = (f"Saved {len(state['measurements'])} measurements to\n{out_json}\n\n"
               f"mpp (weighted) = {summary['mpp_weighted']:.5g} "
               f"+/- {summary['mpp_weighted_uncertainty']:.2g} {units}/px\n"
               f"mpp (mean)     = {summary['mpp_mean']:.5g}  "
               f"(std {summary['mpp_std']:.2g})")
        messagebox.showinfo("Saved", msg)
        status.set(f"Saved -> {out_json}")

    ttk.Button(top, text="Add", command=add_measurement).pack(side="left", padx=2)
    ttk.Button(top, text="Clear pts", command=clear_points).pack(side="left")
    ttk.Button(top, text="Save JSON", command=save_json).pack(side="left", padx=4)

    # ---- measurements table ----
    tframe = ttk.Frame(root, padding=(6, 0, 6, 6))
    tframe.pack(side="bottom", fill="x")
    cols = ("label", "dx_px", f"known", "unc", f"mpp", "mpp_unc")
    tree = ttk.Treeview(tframe, columns=cols, show="headings", height=5)
    for c, w in zip(cols, (90, 70, 70, 60, 100, 80)):
        tree.heading(c, text=c)
        tree.column(c, width=w, anchor="center")
    tree.pack(side="left", fill="x", expand=True)
    ttk.Button(tframe, text="Delete row", command=delete_selected).pack(
        side="left", padx=4)

    redraw_image()
    set_zoom(2)
    root.geometry("1200x900")
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tif", nargs="?", default="data/D2 Calibration.tif")
    ap.add_argument("--units", default="mm", help="physical units (default: mm)")
    ap.add_argument("--px-unc", type=float, default=1.0,
                    help="per-click pixel uncertainty (default: 1.0 px)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    run_gui(args.tif, args.units, args.px_unc)


if __name__ == "__main__":
    main()
