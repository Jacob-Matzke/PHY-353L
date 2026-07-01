"""
Flash ROI viewer (one-time annotation helper, separate from the pipeline).

Step frame-by-frame through every data/*.tif, find a frame where the flash is
clearly visible, and drag a bounding box around it.  The box (one per file) is
saved to outputs/roi.json and later used by analyze_flash.py to restrict the
flash search to that region (+ a cushion), which rejects the fixed-reflection /
fringe contaminants and speeds detection up.

Why this works: the flash sits at a smooth, nu-dependent location, but the
contaminants (a bright spot near the line, a far-right spot, the left fringe
arcs) sit elsewhere.  Telling the detector where you actually see the flash
turns an unreliable blind search into a reliable local one.

Run (on your machine, needs a display):
    python flash_roi_tool.py

Controls
--------
  File dropdown / Prev-file / Next-file  : choose the .tif  (checkmark = boxed)
  Left / Right arrows, slider            : step frames  (PgUp/PgDn = +/-10)
  View: Raw / Residual / Max             : Residual (frame - median) makes the
                                           flash pop; Max is a max-projection
  Brightness slider                      : boost dim flashes
  "Brightest" button                     : jump to the highest-signal frame
  Drag on the image                      : draw the flash box
  Save box / Clear box                   : store / discard the box for this file

Boxes are stored in image pixel coordinates; the pipeline adds its own cushion.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import tifffile

ROI_PATH = "outputs/roi.json"
DISPLAY_W = 960          # on-screen width; image coords are recovered via scale


def parse_rps(stem: str):
    for tok in stem.split():
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", tok):
            return float(tok)
    return None


def list_trials(datadir="data"):
    """All data/*.tif with a parseable RPS (skips the calibration stack)."""
    out = []
    for f in sorted(glob.glob(os.path.join(datadir, "*.tif"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        if parse_rps(stem) is not None:
            out.append((stem, f.replace("\\", "/")))
    return out


def load_roi(path=ROI_PATH):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {"boxes": {}}


def save_roi(data, path=ROI_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def stretch(frame, gain, p_lo=1.0, p_hi=99.7):
    lo, hi = np.percentile(frame, [p_lo, p_hi])
    hi = max(hi, lo + 1)
    out = (frame.astype(np.float32) - lo) * (255.0 / (hi - lo)) * gain
    return np.clip(out, 0, 255).astype("uint8")


# --------------------------------------------------------------------------
def run_gui(datadir, roi_path):
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    trials = list_trials(datadir)
    if not trials:
        raise SystemExit(f"no trial .tif found in {datadir}/")
    roi = load_roi(roi_path)
    roi.setdefault("boxes", {})

    root = tk.Tk()
    root.title("Flash ROI viewer")

    S = {"idx": 0, "frame": 0, "stack": None, "median": None, "peak": None,
         "scale": 1.0, "gain": 3.0, "view": "Residual", "photo": None,
         "box": None,           # (x0,y0,x1,y1) in image coords, saved for file
         "drag": None}          # (x0,y0) live drag start in canvas coords

    # ---- top controls ----
    top = ttk.Frame(root, padding=6); top.pack(side="top", fill="x")
    file_var = tk.StringVar()
    combo = ttk.Combobox(top, textvariable=file_var, width=26, state="readonly")
    combo.pack(side="left")
    ttk.Button(top, text="< file", width=6,
               command=lambda: switch(S["idx"] - 1)).pack(side="left", padx=2)
    ttk.Button(top, text="file >", width=6,
               command=lambda: switch(S["idx"] + 1)).pack(side="left", padx=2)
    view_var = tk.StringVar(value="Residual")
    for v in ("Raw", "Residual", "Max"):
        ttk.Radiobutton(top, text=v, value=v, variable=view_var,
                        command=lambda: set_view(view_var.get())).pack(side="left")
    ttk.Label(top, text=" bright").pack(side="left")
    gain = ttk.Scale(top, from_=0.5, to=12.0, value=3.0, length=110,
                     command=lambda v: set_gain(float(v)))
    gain.pack(side="left", padx=3)
    ttk.Button(top, text="Brightest", command=lambda: goto_brightest()).pack(side="left", padx=4)

    # ---- canvas ----
    canvas = tk.Canvas(root, bg="black", cursor="crosshair"); canvas.pack()
    canvas.bind("<ButtonPress-1>", lambda e: on_press(e))
    canvas.bind("<B1-Motion>", lambda e: on_drag(e))
    canvas.bind("<ButtonRelease-1>", lambda e: on_release(e))

    # ---- frame nav ----
    nav = ttk.Frame(root, padding=6); nav.pack(side="top", fill="x")
    ttk.Button(nav, text="< prev", command=lambda: step(-1)).pack(side="left")
    frame_scale = ttk.Scale(nav, from_=0, to=1, length=520,
                            command=lambda v: set_frame(int(float(v))))
    frame_scale.pack(side="left", padx=6)
    ttk.Button(nav, text="next >", command=lambda: step(1)).pack(side="left")
    ttk.Button(nav, text="Save box", command=lambda: save_box()).pack(side="right")
    ttk.Button(nav, text="Clear box", command=lambda: clear_box()).pack(side="right", padx=4)

    status = tk.StringVar()
    ttk.Label(root, textvariable=status, relief="sunken", anchor="w").pack(
        side="bottom", fill="x")

    # ---- helpers ----
    def combo_labels():
        return [("✓ " if t[0] in roi["boxes"] else "   ") + t[0] for t in trials]

    def load_stack(i):
        stem, path = trials[i]
        status.set(f"loading {stem} ..."); root.update_idletasks()
        stack = tifffile.imread(path)
        S["stack"] = stack
        S["median"] = None; S["peak"] = None
        S["scale"] = DISPLAY_W / stack.shape[2]
        S["idx"] = i; S["frame"] = 0
        b = roi["boxes"].get(stem)
        S["box"] = (b["x0"], b["y0"], b["x1"], b["y1"]) if b else None
        frame_scale.configure(to=stack.shape[0] - 1)
        canvas.configure(width=int(stack.shape[2] * S["scale"]),
                         height=int(stack.shape[1] * S["scale"]))
        combo["values"] = combo_labels()
        combo.current(i)

    def ensure_median():
        if S["median"] is None:
            status.set("computing median ..."); root.update_idletasks()
            S["median"] = np.median(S["stack"], axis=0).astype(np.int16)
            resid = S["stack"].astype(np.int16) - S["median"]
            S["peak"] = resid.reshape(resid.shape[0], -1).max(axis=1)

    def disp_frame():
        st = S["stack"]; f = S["frame"]
        if S["view"] == "Raw":
            img = st[f].astype("uint8")
        elif S["view"] == "Max":
            img = st.max(axis=0).astype("uint8")
        else:
            ensure_median()
            img = np.clip(st[f].astype(np.int16) - S["median"], 0, 255).astype("uint8")
        return stretch(img, S["gain"])

    def redraw():
        img = Image.fromarray(disp_frame())
        sc = S["scale"]
        img = img.resize((int(img.width * sc), int(img.height * sc)), Image.BILINEAR)
        S["photo"] = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=S["photo"])
        if S["box"]:
            x0, y0, x1, y1 = [v * sc for v in S["box"]]
            canvas.create_rectangle(x0, y0, x1, y1, outline="#00ff00", width=2, tags="box")
        stem = trials[S["idx"]][0]
        rps = parse_rps(stem)
        bx = "box set" if S["box"] else "no box"
        status.set(f"{stem}  (nu={rps:+g})   frame {S['frame']}/{S['stack'].shape[0]-1}"
                   f"   view={S['view']}   {bx}   [drag to box, Save box to store]")

    # ---- events ----
    def switch(i):
        save_box(silent=True)
        i = max(0, min(len(trials) - 1, i))
        load_stack(i); redraw()

    def set_view(v):
        S["view"] = v; redraw()

    def set_gain(g):
        S["gain"] = g; redraw()

    def set_frame(f):
        S["frame"] = max(0, min(S["stack"].shape[0] - 1, f)); redraw()

    def step(d):
        set_frame(S["frame"] + d)

    def goto_brightest():
        ensure_median()
        set_frame(int(np.argmax(S["peak"])))

    def on_press(e):
        S["drag"] = (e.x, e.y)

    def on_drag(e):
        if not S["drag"]:
            return
        canvas.delete("live")
        canvas.create_rectangle(S["drag"][0], S["drag"][1], e.x, e.y,
                                outline="#ffcc00", width=2, tags="live")

    def on_release(e):
        if not S["drag"]:
            return
        sc = S["scale"]
        x0, y0 = S["drag"]; x1, y1 = e.x, e.y
        S["drag"] = None
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return
        ix0, ix1 = sorted((x0 / sc, x1 / sc)); iy0, iy1 = sorted((y0 / sc, y1 / sc))
        S["box"] = (round(ix0), round(iy0), round(ix1), round(iy1))
        redraw()

    def save_box(silent=False):
        stem = trials[S["idx"]][0]
        if S["box"]:
            x0, y0, x1, y1 = S["box"]
            roi["boxes"][stem] = {"x0": int(x0), "y0": int(y0), "x1": int(x1),
                                  "y1": int(y1), "frame": int(S["frame"]),
                                  "n_frames": int(S["stack"].shape[0])}
            save_roi(roi, roi_path)
            combo["values"] = combo_labels(); combo.current(S["idx"])
            if not silent:
                status.set(f"saved box for {stem} -> {roi_path}")

    def clear_box():
        stem = trials[S["idx"]][0]
        S["box"] = None
        roi["boxes"].pop(stem, None)
        save_roi(roi, roi_path)
        combo["values"] = combo_labels(); combo.current(S["idx"])
        redraw()

    combo.bind("<<ComboboxSelected>>", lambda e: switch(combo.current()))
    root.bind("<Left>", lambda e: step(-1))
    root.bind("<Right>", lambda e: step(1))
    root.bind("<Prior>", lambda e: step(10))
    root.bind("<Next>", lambda e: step(-10))
    root.protocol("WM_DELETE_WINDOW", lambda: (save_box(silent=True), root.destroy()))

    load_stack(0); redraw()
    root.geometry(f"{DISPLAY_W + 40}x{int(768 * DISPLAY_W / 1024) + 150}")
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datadir", default="data")
    ap.add_argument("--roi", default=ROI_PATH)
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_gui(args.datadir, args.roi)


if __name__ == "__main__":
    main()
