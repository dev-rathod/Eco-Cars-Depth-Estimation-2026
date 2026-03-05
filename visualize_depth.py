"""
Visualize predicted depth overlay on RGB image.
Matches the GT scatter-overlay style.
Usage:
    python visualize_depth.py                          # frame_00000
    python visualize_depth.py --frame 5
    python visualize_depth.py --frame 5 --gt output/depth/frame_00005.npy
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def load_pred_depth(disp_path: Path, use_inverse=True):
    d = np.load(str(disp_path)).astype(np.float32)
    d = np.squeeze(d)  # (1,1,192,640) -> (192,640)
    if use_inverse:
        d = 1.0 / (d + 1e-6)
    return d


def overlay_depth_on_rgb(ax, rgb, depth_h, depth_w, depth_vals,
                          xs, ys, title, vmin=None, vmax=None):
    """Plot RGB with depth scatter overlay."""
    ax.imshow(rgb)
    sc = ax.scatter(xs, ys, c=depth_vals, cmap='jet', s=2, alpha=0.8,
                    vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="Depth (meters)")
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--pred_dir", default="test_frames/images")
    parser.add_argument("--gt", default=None, help="path to GT .npy (optional)")
    parser.add_argument("--no_inverse", action="store_true",
                        help="disable 1/disp conversion")
    args = parser.parse_args()

    stem = f"frame_{args.frame:05d}"
    pred_dir = Path(args.pred_dir)
    img_path = pred_dir / f"{stem}.png"
    disp_path = pred_dir / f"{stem}_disp.npy"

    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not disp_path.exists():
        raise FileNotFoundError(disp_path)

    rgb = np.array(Image.open(img_path))
    H_img, W_img = rgb.shape[:2]

    pred = load_pred_depth(disp_path, use_inverse=not args.no_inverse)
    H_pred, W_pred = pred.shape

    # --- map every pred pixel -> image pixel coords ---
    gy, gx = np.meshgrid(np.arange(H_pred), np.arange(W_pred), indexing='ij')
    xs = (gx.ravel() + 0.5) * (W_img / W_pred)
    ys = (gy.ravel() + 0.5) * (H_img / H_pred)
    vals = pred.ravel()

    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals, 95))

    has_gt = args.gt is not None and Path(args.gt).exists()

    if has_gt:
        gt = np.load(args.gt).astype(np.float32)
        gt = np.squeeze(gt)
        valid = np.isfinite(gt) & (gt > 0)
        ys_gt, xs_gt = np.where(valid)
        gt_vals = gt[ys_gt, xs_gt]
        gt_vmin = float(np.percentile(gt_vals, 5))
        gt_vmax = float(np.percentile(gt_vals, 95))

        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        overlay_depth_on_rgb(axes[0], rgb, H_pred, W_pred, vals, xs, ys,
                             f"Pred Depth Overlay — {stem}", vmin, vmax)
        overlay_depth_on_rgb(axes[1], rgb, *gt.shape, gt_vals, xs_gt, ys_gt,
                             f"GT Depth Overlay — {stem}", gt_vmin, gt_vmax)
        print(f"GT  — valid pixels: {int(valid.sum())}, "
              f"min: {gt_vals.min():.2f}, max: {gt_vals.max():.2f}, "
              f"mean: {gt_vals.mean():.2f}")
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        overlay_depth_on_rgb(ax, rgb, H_pred, W_pred, vals, xs, ys,
                             f"Pred Depth Overlay — {stem}", vmin, vmax)

    print(f"RGB shape: {rgb.shape}")
    print(f"Pred shape: {pred.shape} (mapped to {H_img}x{W_img} for display)")
    print(f"Depth range (5–95 pct): {vmin:.2f} – {vmax:.2f} m")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
