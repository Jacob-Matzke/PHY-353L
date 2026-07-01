import argparse
import numpy as np
import tifffile as tiff
import cv2


def load_tif_stack(path):
    """
    Loads a TIFF or TIFF stack and returns frames as:
        frames[time, y, x]          for grayscale
        frames[time, y, x, color]   for RGB/RGBA
    """
    with tiff.TiffFile(path) as tf:
        series = tf.series[0]
        arr = series.asarray()
        axes = getattr(series, "axes", "")

    arr = np.squeeze(arr)

    print(f"Loaded: {path}")
    print(f"Raw shape: {arr.shape}")
    print(f"Axes from tifffile: {axes}")

    # Simple common cases
    if arr.ndim == 2:
        # Single grayscale image
        arr = arr[None, :, :]

    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            # Single RGB/RGBA image
            arr = arr[None, :, :, :]
        else:
            # Assume time, y, x
            pass

    elif arr.ndim == 4:
        if arr.shape[-1] in (3, 4):
            # Assume time, y, x, color
            pass
        else:
            # Common microscopy case: T, Z, Y, X or T, C, Y, X
            # Take the first Z/C plane.
            arr = arr[:, 0, :, :]

    elif arr.ndim > 4:
        # Flatten all leading dimensions except Y, X.
        # This is crude but useful for quick inspection.
        arr = arr.reshape((-1,) + arr.shape[-2:])

    else:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}")

    print(f"Viewer shape: {arr.shape}")
    return arr


def normalize_for_display(frame, low_pct=0.5, high_pct=99.5):
    """
    Converts frame to uint8 for display.
    Good for 12-bit/16-bit microscope images.
    """
    if frame.dtype == np.uint8:
        return frame

    f = frame.astype(np.float32)

    lo, hi = np.percentile(f, [low_pct, high_pct])
    if hi <= lo:
        hi = lo + 1

    f = (f - lo) / (hi - lo)
    f = np.clip(f, 0, 1)

    return (255 * f).astype(np.uint8)


def make_display_frame(frame, index, total, playing, fps):
    img = normalize_for_display(frame)

    # Convert grayscale/RGB to OpenCV-friendly BGR display
    if img.ndim == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[-1] == 3:
        display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.ndim == 3 and img.shape[-1] == 4:
        display = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    else:
        raise ValueError(f"Cannot display frame shape: {img.shape}")

    status = "playing" if playing else "paused"
    text = f"Frame {index + 1}/{total} | {status} | fps={fps:.1f}"

    cv2.putText(
        display,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to .tif or .tiff stack")
    parser.add_argument("--fps", type=float, default=15, help="Playback fps")
    args = parser.parse_args()

    frames = load_tif_stack(args.path)
    n_frames = frames.shape[0]

    i = 0
    playing = False
    fps = args.fps

    window_name = "TIFF Stack Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print()
    print("Controls:")
    print("  space   play/pause")
    print("  n / d   next frame")
    print("  p / a   previous frame")
    print("  f       faster")
    print("  s       slower")
    print("  q / esc quit")
    print()

    while True:
        display = make_display_frame(frames[i], i, n_frames, playing, fps)
        cv2.imshow(window_name, display)

        delay_ms = max(1, int(1000 / fps)) if playing else 0
        key = cv2.waitKeyEx(delay_ms)

        if key in (ord("q"), 27):  # q or esc
            break

        elif key == ord(" "):
            playing = not playing

        elif key in (ord("n"), ord("d")):
            playing = False
            i = (i + 1) % n_frames

        elif key in (ord("p"), ord("a")):
            playing = False
            i = (i - 1) % n_frames

        elif key == ord("f"):
            fps *= 1.25
            print(f"FPS: {fps:.2f}")

        elif key == ord("s"):
            fps /= 1.25
            print(f"FPS: {fps:.2f}")

        elif playing and key == -1:
            i = (i + 1) % n_frames

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()