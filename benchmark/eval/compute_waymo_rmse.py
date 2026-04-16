import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare depth prediction stacks against Waymo ground truth using RMSE."
    )
    parser.add_argument("--groundtruth", required=True, help="Path to Waymo groundtruth.npy.")
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="One or more prediction .npy stacks to evaluate.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional display names matching --predictions.",
    )
    parser.add_argument(
        "--target-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(1280, 1920),
        help="Comparison resolution. Default matches Depth Anything V2 output size.",
    )
    parser.add_argument(
        "--no-positive-pred-mask",
        action="store_true",
        help="Do not require prediction values to be greater than zero in the valid mask.",
    )
    return parser.parse_args()


def resize_to_target(frame, target_h, target_w):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (target_h, target_w):
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return frame.astype(np.float32, copy=False)


def compute_metrics(groundtruth_path, prediction_path, name, target_shape, require_pred_positive):
    gt = np.load(groundtruth_path, mmap_mode="r")
    pred = np.load(prediction_path, mmap_mode="r")

    target_h, target_w = target_shape
    frame_count = min(gt.shape[0], pred.shape[0])

    valid_count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    raw_sse = 0.0

    for index in range(frame_count):
        gt_frame = resize_to_target(gt[index], target_h, target_w)
        pred_frame = resize_to_target(pred[index], target_h, target_w)

        mask = np.isfinite(gt_frame) & (gt_frame > 0) & np.isfinite(pred_frame)
        if require_pred_positive:
            mask &= pred_frame > 0

        x = pred_frame[mask].astype(np.float64)
        y = gt_frame[mask].astype(np.float64)
        if x.size == 0:
            continue

        valid_count += x.size
        sum_x += x.sum()
        sum_y += y.sum()
        sum_xx += (x * x).sum()
        sum_xy += (x * y).sum()
        raw_sse += ((x - y) ** 2).sum()

    if valid_count == 0:
        raise ValueError(f"No valid pixels found for {prediction_path}")

    denom = valid_count * sum_xx - sum_x * sum_x
    if denom == 0:
        raise ValueError(f"Cannot compute linear alignment for {prediction_path}")

    slope = (valid_count * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / valid_count

    aligned_sse = 0.0
    for index in range(frame_count):
        gt_frame = resize_to_target(gt[index], target_h, target_w)
        pred_frame = resize_to_target(pred[index], target_h, target_w)

        mask = np.isfinite(gt_frame) & (gt_frame > 0) & np.isfinite(pred_frame)
        if require_pred_positive:
            mask &= pred_frame > 0

        x = pred_frame[mask].astype(np.float64)
        y = gt_frame[mask].astype(np.float64)
        aligned_sse += ((slope * x + intercept - y) ** 2).sum()

    return {
        "name": name,
        "prediction_path": str(prediction_path),
        "target_shape": (target_h, target_w),
        "frames": frame_count,
        "valid_pixels": valid_count,
        "linear_fit_slope": slope,
        "linear_fit_intercept": intercept,
        "rmse_before_alignment": (raw_sse / valid_count) ** 0.5,
        "rmse_after_alignment": (aligned_sse / valid_count) ** 0.5,
    }


def print_metrics(metrics):
    print(metrics["name"])
    print(f"  prediction_path: {metrics['prediction_path']}")
    print(f"  target_shape: {metrics['target_shape']}")
    print(f"  frames: {metrics['frames']}")
    print(f"  valid_pixels: {metrics['valid_pixels']}")
    print(f"  linear_fit_slope: {metrics['linear_fit_slope']:.12g}")
    print(f"  linear_fit_intercept: {metrics['linear_fit_intercept']:.12g}")
    print(f"  rmse_before_alignment: {metrics['rmse_before_alignment']:.6f}")
    print(f"  rmse_after_alignment: {metrics['rmse_after_alignment']:.6f}")
    print()


def main():
    args = parse_args()
    prediction_paths = [Path(path) for path in args.predictions]
    names = args.names or [path.stem for path in prediction_paths]

    if len(names) != len(prediction_paths):
        raise ValueError("--names must have the same count as --predictions")

    for name, prediction_path in zip(names, prediction_paths):
        metrics = compute_metrics(
            groundtruth_path=Path(args.groundtruth),
            prediction_path=prediction_path,
            name=name,
            target_shape=tuple(args.target_shape),
            require_pred_positive=not args.no_positive_pred_mask,
        )
        print_metrics(metrics)


if __name__ == "__main__":
    main()
