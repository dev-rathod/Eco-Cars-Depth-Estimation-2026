import argparse
import io
from pathlib import Path
import zipfile

import cv2
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Video Depth Anything predictions against a Waymo depth stack."
    )
    parser.add_argument(
        "--pred-depth-dir",
        type=Path,
        default=None,
        help="Directory containing Video Depth Anything *_depth.npy files.",
    )
    parser.add_argument("--pred-stack", type=Path, default=None, help="Stacked Video Depth Anything .npy file.")
    parser.add_argument(
        "--pred-stems-zip",
        type=Path,
        default=None,
        help="Zip of input images used to infer stems for --pred-stack.",
    )
    parser.add_argument("--gt-stack", type=Path, default=None, help="Ground-truth depth_stack.npy.")
    parser.add_argument("--gt-stems", type=Path, default=None, help="Ground-truth depth_stack_stems.npy.")
    parser.add_argument(
        "--gt-depth-zip",
        type=Path,
        default=None,
        help="Zip containing per-frame ground-truth .npy files.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=5,
        help="Number of matched frames to use for calibration before evaluation.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip this many sorted matched stems before calibration/evaluation.",
    )
    parser.add_argument("--max-depth", type=float, default=40.0, help="Maximum valid GT depth in meters.")
    parser.add_argument(
        "--formula",
        choices=["hyperbolic", "power", "log", "exponential"],
        default=None,
        help="Calibration formula to force. By default, the lowest calibration RMSE model is used.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=100_000,
        help="Maximum number of cleaned points passed to scipy curve_fit.",
    )
    return parser.parse_args()


def stems_from_zip(zip_path, suffixes):
    with zipfile.ZipFile(zip_path) as zf:
        stems = []
        for name in zf.namelist():
            path = Path(name)
            if path.name.startswith("._"):
                continue
            if path.suffix.lower() in suffixes:
                stems.append(path.stem)
    return sorted(set(stems))


def load_prediction_dict(pred_depth_dir):
    pred_files = sorted(pred_depth_dir.glob("*_depth.npy"))
    if not pred_files:
        raise FileNotFoundError(f"No *_depth.npy files found in {pred_depth_dir}")

    pred_dict = {}
    for path in pred_files:
        stem = path.stem.removesuffix("_depth")
        pred_dict[stem] = np.load(path).astype(np.float32)
    return pred_dict


def load_prediction_stack(stack_path, stems_zip):
    stack = np.load(stack_path, mmap_mode="r")
    stems = stems_from_zip(stems_zip, {".png", ".jpg", ".jpeg"})
    if len(stems) != stack.shape[0]:
        raise RuntimeError(f"{len(stems)} stems found, but prediction stack has {stack.shape[0]} frames.")
    return {stem: stack[i].astype(np.float32) for i, stem in enumerate(stems)}


def load_gt_from_zip(gt_depth_zip, stems):
    with zipfile.ZipFile(gt_depth_zip) as zf:
        zip_names = {Path(name).stem: name for name in zf.namelist() if name.endswith(".npy")}
        gt_dict = {}
        for stem in stems:
            if stem not in zip_names:
                continue
            with zf.open(zip_names[stem]) as file:
                gt_dict[stem] = np.load(io.BytesIO(file.read())).astype(np.float32)
    return gt_dict


def resize_to_gt(pred, gt_shape):
    if pred.shape == gt_shape:
        return pred
    return cv2.resize(pred, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)


def noise_handler(pred, gt):
    ground_truth = gt.flatten()
    predictions = pred.flatten()

    total_bins = 20
    bin_edges = np.linspace(predictions.min(), predictions.max(), total_bins + 1)
    bin_index = np.digitize(predictions, bin_edges) - 1
    bins_inlier = np.zeros(len(predictions), dtype=bool)

    minimum_entries = 25
    for bin_id in range(total_bins):
        in_bin = bin_index == bin_id
        if in_bin.sum() < minimum_entries:
            continue

        gt_entry = ground_truth[in_bin]
        lower_quartile, upper_quartile = np.percentile(gt_entry, [25, 75])
        iqr = upper_quartile - lower_quartile
        bins_inlier[in_bin] = (gt_entry >= lower_quartile - 1.5 * iqr) & (
            gt_entry <= upper_quartile + 1.5 * iqr
        )

    p_clean = predictions[bins_inlier]
    g_clean = ground_truth[bins_inlier]
    if len(p_clean) < 10:
        raise RuntimeError("Too few calibration points remain after IQR filtering.")

    poly_graph = np.polyfit(p_clean, g_clean, deg=4)
    residuals = g_clean - np.polyval(poly_graph, p_clean)
    zscores = np.abs(stats.zscore(residuals, nan_policy="omit"))
    deviations_mask = np.isfinite(zscores) & (zscores < 2.75)

    return p_clean[deviations_mask], g_clean[deviations_mask]


def model_hyperbolic(p, a, b, c):
    return a / (p + b) + c


def model_power(p, a, b, c):
    return a * np.power(np.clip(p, 1e-6, None), b) + c


def model_log(p, a, b, c):
    return a * np.log(np.clip(p + b, 1e-6, None)) + c


def model_exponential(p, a, b, c):
    return a * np.exp(b * p) + c


FN_MAP = {
    "hyperbolic": model_hyperbolic,
    "power": model_power,
    "log": model_log,
    "exponential": model_exponential,
}


def fit_all_models(p_clean, g_clean, n_sample):
    if len(p_clean) > n_sample:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(p_clean), n_sample, replace=False)
        p_fit = p_clean[idx]
        g_fit = g_clean[idx]
    else:
        p_fit, g_fit = p_clean, g_clean

    models = {
        "hyperbolic": (
            model_hyperbolic,
            [-5.0, -0.85, 35.0],
            ([-np.inf, -1.0 + 1e-6, -np.inf], [np.inf, -1e-6, np.inf]),
        ),
        "power": (model_power, [10.0, 0.5, 0.0], ([-np.inf] * 3, [np.inf] * 3)),
        "log": (model_log, [-20.0, 1.0, 40.0], ([-np.inf] * 3, [np.inf] * 3)),
        "exponential": (model_exponential, [2.2, 2.97, 5.6], ([-np.inf] * 3, [np.inf] * 3)),
    }

    results = {}
    for name, (fn, p0, bounds) in models.items():
        try:
            popt, _ = curve_fit(fn, p_fit, g_fit, p0=p0, bounds=bounds, maxfev=50000)
            g_pred = fn(p_clean, *popt)
            rmse = np.sqrt(np.mean((g_clean - g_pred) ** 2))
            ss_res = np.sum((g_clean - g_pred) ** 2)
            ss_tot = np.sum((g_clean - g_clean.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            results[name] = {"params": popt, "RMSE": rmse, "R2": r2}
        except Exception as exc:
            print(f"{name}: failed - {exc}")
    return results


def compute_metrics(pred, gt, max_depth):
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0.0) & (gt <= max_depth)
    if not mask.any():
        raise RuntimeError("No valid pixels remain for evaluation.")

    pred_v = np.clip(pred[mask], 1e-6, None)
    gt_v = gt[mask]
    absrel = np.mean(np.abs(pred_v - gt_v) / gt_v)
    rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta1 = np.mean(ratio < 1.25)
    return {"AbsRel": absrel, "RMSE": rmse, "delta1": delta1, "N": len(gt_v)}


def main():
    args = parse_args()
    if args.pred_stack:
        if not args.pred_stems_zip:
            raise RuntimeError("--pred-stems-zip is required with --pred-stack.")
        pred_dict = load_prediction_stack(args.pred_stack, args.pred_stems_zip)
    elif args.pred_depth_dir:
        pred_dict = load_prediction_dict(args.pred_depth_dir)
    else:
        raise RuntimeError("Provide either --pred-stack or --pred-depth-dir.")

    if args.gt_depth_zip:
        gt_dict = load_gt_from_zip(args.gt_depth_zip, pred_dict.keys())
    elif args.gt_stack and args.gt_stems:
        gt_stack = np.load(args.gt_stack, mmap_mode="r")
        gt_stems = np.load(args.gt_stems, allow_pickle=True).astype(str)
        gt_dict = {stem: gt_stack[i].astype(np.float32) for i, stem in enumerate(gt_stems)}
    else:
        raise RuntimeError("Provide either --gt-depth-zip or both --gt-stack and --gt-stems.")

    common_stems = sorted(set(pred_dict) & set(gt_dict))
    if args.start_index:
        common_stems = common_stems[args.start_index :]
    if not common_stems:
        raise RuntimeError("No matching stems between Video Depth Anything predictions and ground truth.")

    pred_aligned = []
    gt_aligned = []
    for stem in common_stems:
        gt = gt_dict[stem]
        pred_aligned.append(resize_to_gt(pred_dict[stem], gt.shape))
        gt_aligned.append(gt)

    pred_aligned = np.stack(pred_aligned, axis=0)
    gt_aligned = np.stack(gt_aligned, axis=0)

    n_calib = min(args.calibration_frames, len(common_stems) - 1)
    if n_calib <= 0:
        raise RuntimeError("Need at least two matched frames: one for calibration and one for evaluation.")

    calib_mask = (
        np.isfinite(pred_aligned[:n_calib])
        & np.isfinite(gt_aligned[:n_calib])
        & (gt_aligned[:n_calib] > 0)
        & (gt_aligned[:n_calib] <= args.max_depth)
    )
    p_clean, g_clean = noise_handler(pred_aligned[:n_calib][calib_mask], gt_aligned[:n_calib][calib_mask])
    fit_results = fit_all_models(p_clean, g_clean, args.sample_points)
    if not fit_results:
        raise RuntimeError("All calibration models failed.")

    best_name = args.formula or min(fit_results, key=lambda name: fit_results[name]["RMSE"])
    best = fit_results[best_name]
    calibrated_eval = FN_MAP[best_name](pred_aligned[n_calib:], *best["params"])
    metrics = compute_metrics(calibrated_eval, gt_aligned[n_calib:], args.max_depth)

    print(f"Matched stems          : {len(common_stems)}")
    print(f"Calibration stems      : {common_stems[:n_calib]}")
    print(f"Evaluation stems       : {common_stems[n_calib:]}")
    print(f"Prediction shape       : {pred_aligned.shape}")
    print(f"Ground-truth shape     : {gt_aligned.shape}")
    print()
    print("Calibration fits:")
    for name, result in fit_results.items():
        params = np.round(result["params"], 4)
        print(f"  {name:12s} R2={result['R2']:.4f} RMSE={result['RMSE']:.4f}m params={params}")
    print()
    print(f"Selected model         : {best_name}")
    print(f"Evaluation pixels      : {metrics['N']:,}")
    print(f"AbsRel                 : {metrics['AbsRel']:.4f}")
    print(f"RMSE                   : {metrics['RMSE']:.4f} m")
    print(f"delta1                 : {metrics['delta1']:.4f}")


if __name__ == "__main__":
    main()
