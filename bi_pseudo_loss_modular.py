from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


# IO utilities

def load_npy(path: str | Path) -> np.ndarray:
   
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    arr = np.load(path, allow_pickle=True)

    if arr.dtype == object:
        raise ValueError(f"Expected numeric depth array, got object dtype: {path}")

    return arr.astype(np.float32)


def canonical_depth_shape(arr: np.ndarray, name: str = "depth") -> np.ndarray:
  
    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        return arr

    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(f"{name} has 4 dims but channel dim is not 1: {arr.shape}")
        return arr[:, 0, :, :]

    raise ValueError(f"{name} must have shape [H,W], [N,H,W], or [N,1,H,W], got {arr.shape}")


def load_params(params_json: Optional[str], params_file: Optional[str]) -> Dict[str, Any]:
    
    if params_json and params_file:
        raise ValueError("Use either --params-json or --params-file, not both.")

    if params_json:
        return json.loads(params_json)

    if params_file:
        with open(params_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


# Depth transformation utilities

def transform_depth_np(
    depth: np.ndarray,
    mode: str = "identity",
    params: Optional[Dict[str, Any]] = None,
    gt_depth: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> np.ndarray:

    if params is None:
        params = {}

    mode = mode.lower()

    if mode == "identity":
        return depth.astype(np.float32)

    if mode == "affine":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        return (a * depth + b).astype(np.float32)

    if mode == "power":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        c = float(params.get("c", 0.0))
        x = np.clip(depth, eps, None)
        return (a * np.power(x, b) + c).astype(np.float32)

    if mode == "exponential":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        c = float(params.get("c", 0.0))
        return (a * np.exp(b * depth) + c).astype(np.float32)

    if mode == "log":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 1.0))
        c = float(params.get("c", 0.0))
        x = np.clip(depth + b, eps, None)
        return (a * np.log(x) + c).astype(np.float32)

    if mode == "inverse":
        a = float(params.get("a", 1.0))
        b = float(params.get("b", eps))
        c = float(params.get("c", 0.0))
        return (a / (depth + b + eps) + c).astype(np.float32)

    if mode == "minmax":
        out_min = float(params.get("out_min", 0.0))
        out_max = float(params.get("out_max", 80.0))
        transformed = np.empty_like(depth, dtype=np.float32)

        for i in range(depth.shape[0]):
            d = depth[i]
            valid = np.isfinite(d)
            if valid.sum() == 0:
                transformed[i] = np.nan
                continue

            d_min = np.nanmin(d[valid])
            d_max = np.nanmax(d[valid])
            transformed[i] = (d - d_min) / (d_max - d_min + eps)
            transformed[i] = transformed[i] * (out_max - out_min) + out_min

        return transformed.astype(np.float32)

    if mode == "median_scale_to_gt":
        if gt_depth is None:
            raise ValueError("median_scale_to_gt requires --gt.")

        transformed = np.empty_like(depth, dtype=np.float32)

        for i in range(depth.shape[0]):
            pred = depth[i]
            gt = gt_depth[i]
            valid = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0)

            if valid.sum() == 0:
                transformed[i] = pred
                continue

            scale = np.median(gt[valid]) / (np.median(pred[valid]) + eps)
            transformed[i] = pred * scale

        return transformed.astype(np.float32)

    raise ValueError(f"Unsupported transform mode: {mode}")


# Mask utilities

def build_valid_mask_np(
    clean_depth: np.ndarray,
    aug_depth: np.ndarray,
    gt_depth: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    max_depth: float = 80.0,
) -> np.ndarray:
    
    valid = np.isfinite(clean_depth) & np.isfinite(aug_depth)
    valid = valid & (clean_depth > 0) & (aug_depth > 0)

    if gt_depth is not None:
        valid = valid & np.isfinite(gt_depth) & (gt_depth > 0) & (gt_depth < max_depth)

    if mask is not None:
        valid = valid & mask.astype(bool)

    return valid


# Offline NumPy bi-pseudo loss

def huber_np(error: np.ndarray, delta: float = 1.0) -> np.ndarray:
    """
    Elementwise Huber loss from prediction error.
    """
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def bi_pseudo_loss_np(
    clean_depth: np.ndarray,
    aug_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, float]:

    if clean_depth.shape != aug_depth.shape:
        raise ValueError(f"Shape mismatch: clean {clean_depth.shape}, aug {aug_depth.shape}")

    if mask is None:
        mask = np.isfinite(clean_depth) & np.isfinite(aug_depth) & (clean_depth > 0) & (aug_depth > 0)
    else:
        mask = mask.astype(bool)

    valid_pixels = int(mask.sum())

    if valid_pixels == 0:
        return {"loss": float("nan"), "valid_pixels": 0}

    clean_v = clean_depth[mask]
    aug_v = aug_depth[mask]

    loss_type = loss_type.lower()

    if loss_type == "l1":
        loss = np.mean(np.abs(clean_v - aug_v))
    elif loss_type == "mse":
        loss = np.mean((clean_v - aug_v) ** 2)
    elif loss_type == "rmse":
        loss = np.sqrt(np.mean((clean_v - aug_v) ** 2))
    elif loss_type == "huber":
        loss = np.mean(huber_np(clean_v - aug_v, delta=huber_delta))
    elif loss_type == "silog":
        d = np.log(clean_v + eps) - np.log(aug_v + eps)
        loss = np.sqrt(np.maximum(np.mean(d ** 2) - 0.85 * (np.mean(d) ** 2), eps))
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    return {"loss": float(loss), "valid_pixels": valid_pixels}


def per_frame_bi_pseudo_loss_np(
    clean_depth: np.ndarray,
    aug_depth: np.ndarray,
    mask: Optional[np.ndarray],
    loss_type: str = "huber",
    huber_delta: float = 1.0,
) -> Tuple[list[Dict[str, float]], Dict[str, float]]:
    """
    Compute per-frame and overall bi-pseudo loss.
    """
    rows = []

    if mask is None:
        mask = np.isfinite(clean_depth) & np.isfinite(aug_depth) & (clean_depth > 0) & (aug_depth > 0)

    for i in range(clean_depth.shape[0]):
        result = bi_pseudo_loss_np(
            clean_depth=clean_depth[i:i+1],
            aug_depth=aug_depth[i:i+1],
            mask=mask[i:i+1],
            loss_type=loss_type,
            huber_delta=huber_delta,
        )
        rows.append({"frame_index": i, "loss": result["loss"], "valid_pixels": result["valid_pixels"]})

    overall = bi_pseudo_loss_np(
        clean_depth=clean_depth,
        aug_depth=aug_depth,
        mask=mask,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )

    return rows, overall


# Optional PyTorch training-time bi-pseudo loss

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    def transform_depth_torch(depth, mode: str = "identity", params: Optional[Dict[str, Any]] = None, eps: float = 1e-6):
    
        if params is None:
            params = {}

        mode = mode.lower()

        if mode == "identity":
            return depth
        if mode == "affine":
            return float(params.get("a", 1.0)) * depth + float(params.get("b", 0.0))
        if mode == "power":
            return float(params.get("a", 1.0)) * torch.clamp(depth, min=eps).pow(float(params.get("b", 1.0))) + float(params.get("c", 0.0))
        if mode == "exponential":
            return float(params.get("a", 1.0)) * torch.exp(float(params.get("b", 1.0)) * depth) + float(params.get("c", 0.0))
        if mode == "log":
            return float(params.get("a", 1.0)) * torch.log(torch.clamp(depth + float(params.get("b", 1.0)), min=eps)) + float(params.get("c", 0.0))
        if mode == "inverse":
            return float(params.get("a", 1.0)) / (depth + float(params.get("b", eps)) + eps) + float(params.get("c", 0.0))

        raise ValueError(f"Unsupported torch transform mode: {mode}")


    def bi_pseudo_loss_torch(
        clean_depth,
        aug_depth,
        mask=None,
        loss_type: str = "huber",
        huber_delta: float = 1.0,
        bidirectional: bool = True,
        eps: float = 1e-6,
    ):
       
        if clean_depth.shape != aug_depth.shape:
            raise ValueError(f"Shape mismatch: clean {clean_depth.shape}, aug {aug_depth.shape}")

        if mask is None:
            valid = torch.isfinite(clean_depth) & torch.isfinite(aug_depth)
            valid = valid & (clean_depth > 0) & (aug_depth > 0)
        else:
            valid = mask.bool()
            valid = valid & torch.isfinite(clean_depth) & torch.isfinite(aug_depth)
            valid = valid & (clean_depth > 0) & (aug_depth > 0)

        if valid.sum() == 0:
            return None

        clean_v = clean_depth[valid]
        aug_v = aug_depth[valid]
        loss_type = loss_type.lower()

        def one_direction_loss(pred, pseudo_target):
            pseudo_target = pseudo_target.detach()

            if loss_type == "l1":
                return torch.abs(pred - pseudo_target).mean()
            if loss_type == "mse":
                return torch.mean((pred - pseudo_target) ** 2)
            if loss_type == "rmse":
                return torch.sqrt(torch.mean((pred - pseudo_target) ** 2) + eps)
            if loss_type == "huber":
                return F.smooth_l1_loss(pred, pseudo_target, beta=huber_delta)
            if loss_type == "silog":
                d = torch.log(pred + eps) - torch.log(pseudo_target + eps)
                return torch.sqrt(torch.clamp(torch.mean(d ** 2) - 0.85 * (torch.mean(d) ** 2), min=eps))

            raise ValueError(f"Unsupported loss type: {loss_type}")

        forward_loss = one_direction_loss(aug_v, clean_v)

        if not bidirectional:
            return forward_loss

        backward_loss = one_direction_loss(clean_v, aug_v)
        return 0.5 * (forward_loss + backward_loss)


# CSV output

def save_loss_csv(rows: list[Dict[str, float]], overall: Dict[str, float], output_csv: str | Path) -> None:
    """
    Save per-frame and overall bi-pseudo loss results to CSV.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "loss", "valid_pixels"])
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

        writer.writerow({
            "frame_index": "OVERALL",
            "loss": overall["loss"],
            "valid_pixels": overall["valid_pixels"],
        })


# Command line interface

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute modular bi-pseudo consistency loss between clean and augmented depth outputs."
    )

    parser.add_argument("--clean", required=True, help="Path to clean depth .npy file.")
    parser.add_argument("--aug", required=True, help="Path to augmented depth .npy file.")
    parser.add_argument("--gt", default=None, help="Optional ground-truth depth .npy file. Used for mask or median_scale_to_gt.")
    parser.add_argument("--mask", default=None, help="Optional boolean mask .npy file.")

    parser.add_argument(
        "--transform",
        default="identity",
        choices=["identity", "affine", "power", "exponential", "log", "inverse", "minmax", "median_scale_to_gt"],
        help="Optional transformation applied to both clean and augmented depth before loss.",
    )

    parser.add_argument(
        "--params-json",
        default=None,
        help='JSON string for transform parameters, e.g. \'{"a":47.59,"b":3.46,"c":9.75}\'.',
    )

    parser.add_argument("--params-file", default=None, help="Path to JSON file containing transform parameters.")

    parser.add_argument(
        "--loss",
        default="huber",
        choices=["l1", "mse", "rmse", "huber", "silog"],
        help="Consistency loss type.",
    )

    parser.add_argument("--huber-delta", type=float, default=1.0, help="Delta parameter for Huber loss.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Maximum valid ground-truth depth if GT is provided.")
    parser.add_argument("--output-csv", default=None, help="Optional output CSV path for per-frame and overall loss.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params_json, args.params_file)

    clean = canonical_depth_shape(load_npy(args.clean), name="clean")
    aug = canonical_depth_shape(load_npy(args.aug), name="aug")

    if clean.shape != aug.shape:
        raise ValueError(f"Clean and augmented depth shape mismatch: {clean.shape} vs {aug.shape}")

    gt = None
    if args.gt is not None:
        gt = canonical_depth_shape(load_npy(args.gt), name="gt")
        if gt.shape != clean.shape:
            raise ValueError(f"GT and prediction shape mismatch: {gt.shape} vs {clean.shape}")

    mask = None
    if args.mask is not None:
        mask = canonical_depth_shape(load_npy(args.mask), name="mask").astype(bool)
        if mask.shape != clean.shape:
            raise ValueError(f"Mask and prediction shape mismatch: {mask.shape} vs {clean.shape}")

    print("Loaded arrays:")
    print(f"  clean shape: {clean.shape}, range: [{np.nanmin(clean):.6f}, {np.nanmax(clean):.6f}]")
    print(f"  aug   shape: {aug.shape}, range: [{np.nanmin(aug):.6f}, {np.nanmax(aug):.6f}]")
    if gt is not None:
        print(f"  gt    shape: {gt.shape}, range: [{np.nanmin(gt):.6f}, {np.nanmax(gt):.6f}]")
    if mask is not None:
        print(f"  mask  shape: {mask.shape}, valid pixels: {int(mask.sum())}")

    clean_final = transform_depth_np(clean, mode=args.transform, params=params, gt_depth=gt)
    aug_final = transform_depth_np(aug, mode=args.transform, params=params, gt_depth=gt)

    valid_mask = build_valid_mask_np(
        clean_depth=clean_final,
        aug_depth=aug_final,
        gt_depth=gt,
        mask=mask,
        max_depth=args.max_depth,
    )

    rows, overall = per_frame_bi_pseudo_loss_np(
        clean_depth=clean_final,
        aug_depth=aug_final,
        mask=valid_mask,
        loss_type=args.loss,
        huber_delta=args.huber_delta,
    )

    print("\nBi-pseudo consistency result:")
    print(f"  transform:    {args.transform}")
    print(f"  params:       {params}")
    print(f"  loss type:    {args.loss}")
    print(f"  overall loss: {overall['loss']:.8f}")
    print(f"  valid pixels: {overall['valid_pixels']}")

    if args.output_csv is not None:
        save_loss_csv(rows, overall, args.output_csv)
        print(f"\nSaved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
