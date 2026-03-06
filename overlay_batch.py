from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIGURATION
# =========================

# Input directories
IMAGES_DIR = Path("/Users/devrathod/Documents/classes_2026/Senior Design/Eco-Cars-Depth-Estimation-2026/data/output/images")
DISP_DIR = Path("/Users/devrathod/Documents/classes_2026/Senior Design/Eco-Cars-Depth-Estimation-2026/data/output/images")
all_depths = np.load("/Users/devrathod/Documents/classes_2026/Senior Design/Eco-Cars-Depth-Estimation-2026/data/output/all_depths.npy")

# Output directories
OUTPUT_ROOT = Path("/Users/devrathod/Documents/classes_2026/Senior Design/Eco-Cars-Depth-Estimation-2026/data/monodepth_v2")
DEPTH_DIR = OUTPUT_ROOT / "depth_arrays"
OVERLAY_DIR = OUTPUT_ROOT / "sparse_overlays"

USE_INVERSE = True

STRIDE = 1
ROI_Y_MIN_FRAC = 0.4
ROI_Y_MAX_FRAC = 1.0

EXCLUDE_BOTTOM_FRAC = 0.00

DEPTH_MAX_M = None

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
    disp = np.asarray(disp, dtype=np.float32)
    depth = 1.0 / (disp + 1e-6)
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
        frame_idx = int(stem.split("_")[1])

        rgb = load_rgb(rgb_path)
        disp = all_depths[frame_idx]

        # depth = disp_to_depth(disp)
        depth = disp

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