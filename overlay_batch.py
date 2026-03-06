from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIGURATION
# =========================

# Input directories
IMAGES_DIR = Path("test_frames/images")
DISP_DIR = Path("test_frames/images")

# Output directories
OUTPUT_ROOT = Path("outputs")
DEPTH_DIR = OUTPUT_ROOT / "depth_arrays"
OVERLAY_DIR = OUTPUT_ROOT / "sparse_overlays"

USE_INVERSE = True
EPS = 1e-6

MIN_DEPTH = 0.1
MAX_DEPTH = 100.0

STRIDE = 1
ROI_Y_MIN_FRAC = 0.4
ROI_Y_MAX_FRAC = 1.0

EXCLUDE_BOTTOM_FRAC = 0.00

DEPTH_MAX_M = 2.0

DEPTH_MIN_M = None

POINT_SIZE = 1.0
ALPHA = 0.9
CMAP = "jet"

VMIN_PCTL = 1
VMAX_PCTL = 99

FIGSIZE = (10, 6)
DPI = 150


# =========================
# LOAD DATA
# =========================

def load_rgb(path: Path):
    """Load RGB image."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_disp(path: Path):
    """Load disparity numpy array."""
    disp = np.load(path).astype(np.float32)
    disp = np.squeeze(disp)
    return disp


# =========================
# DISP → DEPTH
# =========================

def disp_to_depth(disp):
    """
    Convert disparity to depth-like map.
    Do NOT clip disp to [0, 1] unless you are sure it is raw sigmoid output.
    """
    disp = np.asarray(disp, dtype=np.float32)

    min_disp = 1.0 / MAX_DEPTH
    max_disp = 1.0 / MIN_DEPTH

    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / np.clip(scaled_disp, EPS, None)

    return depth


# =========================
# VISUALIZATION
# =========================

def plot_sparse_overlay(rgb, depth, out_path):

    H_img, W_img = rgb.shape[:2]
    H_d, W_d = depth.shape

    valid = np.isfinite(depth) & (depth > 0)

    y0 = int(ROI_Y_MIN_FRAC * H_d)
    y1 = int(ROI_Y_MAX_FRAC * H_d)

    valid[:y0, :] = False
    valid[y1:, :] = False

    if EXCLUDE_BOTTOM_FRAC > 0:
        cut = int((1 - EXCLUDE_BOTTOM_FRAC) * H_d)
        valid[cut:, :] = False

    if DEPTH_MIN_M is not None:
        valid &= depth >= DEPTH_MIN_M

    if DEPTH_MAX_M is not None:
        valid &= depth <= DEPTH_MAX_M

    if STRIDE > 1:
        stride_mask = np.zeros_like(valid)
        stride_mask[::STRIDE, ::STRIDE] = True
        valid &= stride_mask

    ys, xs = np.where(valid)
    vals = depth[ys, xs]

    scale_y = H_img / H_d
    scale_x = W_img / W_d

    ys = ys * scale_y
    xs = xs * scale_x

    vmin = np.percentile(vals, VMIN_PCTL)
    vmax = np.percentile(vals, VMAX_PCTL)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.imshow(rgb)

    sc = ax.scatter(
        xs,
        ys,
        c=vals,
        s=POINT_SIZE,
        cmap=CMAP,
        alpha=ALPHA,
        vmin=vmin,
        vmax=vmax,
        linewidths=0
    )

    ax.axis("off")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Depth")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================

def main():

    DEPTH_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    rgb_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        rgb_files += sorted(IMAGES_DIR.glob(ext))

    rgb_files = [f for f in rgb_files if not f.stem.endswith("_disp")]

    for rgb_path in rgb_files:

        stem = rgb_path.stem
        disp_path = DISP_DIR / f"{stem}_disp.npy"

        if not disp_path.exists():
            print(f"[SKIP] {disp_path}")
            continue

        rgb = load_rgb(rgb_path)
        disp = load_disp(disp_path)

        depth = disp_to_depth(disp)

        # Save depth numpy array
        depth_path = DEPTH_DIR / f"{stem}_depth.npy"
        np.save(depth_path, depth.astype(np.float32))

        # Save overlay image
        overlay_path = OVERLAY_DIR / f"{stem}_sparse_overlay.png"
        plot_sparse_overlay(rgb, depth, overlay_path)

        print(f"Saved: {stem}")

    print("\nFinished processing.")
    print("Depth arrays:", DEPTH_DIR.resolve())
    print("Overlay images:", OVERLAY_DIR.resolve())


if __name__ == "__main__":
    main()