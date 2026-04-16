import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch
from scipy import stats
from scipy.optimize import curve_fit


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}

COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified Video Depth Anything pipeline for inference, evaluation, and report generation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="Run Video Depth Anything on a directory of frames.")
    add_infer_args(infer)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against Waymo-style depth data.")
    add_evaluate_args(evaluate)

    report = subparsers.add_parser("report", help="Generate calibration reports for multiple segment zips.")
    add_report_args(report)

    infer_eval = subparsers.add_parser(
        "infer-evaluate",
        help="Run frame-directory inference and then evaluate the generated prediction stack.",
    )
    add_infer_args(infer_eval)
    add_evaluate_args(infer_eval, require_prediction_input=False)
    infer_eval.add_argument(
        "--eval-output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation summary as JSON.",
    )
    return parser


def add_infer_args(parser):
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing input frames.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save inference outputs.")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for frame discovery.")
    parser.add_argument("--input_size", type=int, default=518, help="Model input size.")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint override path.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Optional limit on the number of frames.")
    parser.add_argument("--metric", action="store_true", help="Use the metric model checkpoint.")
    parser.add_argument("--fp32", action="store_true", help="Run inference in float32 instead of autocast.")
    parser.add_argument(
        "--target_fps",
        type=int,
        default=24,
        help="Target sequence FPS passed to Video Depth Anything inference.",
    )
    parser.add_argument(
        "--save_heatmaps",
        action="store_true",
        help="Save blended heatmap overlays for each frame.",
    )
    parser.add_argument(
        "--save_individual_depths",
        action="store_true",
        help="Save one .npy file per frame under output_dir/depth_arrays.",
    )
    parser.add_argument(
        "--save_stack",
        action="store_true",
        help="Save the full depth tensor as output_dir/video_depth_anything_depths.npy.",
    )
    parser.add_argument(
        "--heatmap_alpha",
        type=float,
        default=0.5,
        help="Blend ratio for the original frame when saving heatmaps.",
    )
    parser.add_argument(
        "--heatmap_colormap",
        type=str,
        default="jet",
        choices=sorted(COLORMAPS),
        help="OpenCV colormap used for heatmap overlays.",
    )


def add_evaluate_args(parser, require_prediction_input=True):
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
    if require_prediction_input:
        parser.add_argument(
            "--require-prediction-input",
            action="store_true",
            help=argparse.SUPPRESS,
        )


def add_report_args(parser):
    parser.add_argument("--gt-batch-zip", type=Path, required=True, help="Zip containing nested GT segment zips.")
    parser.add_argument("--pred-zip-dir", type=Path, required=True, help="Directory containing per-segment prediction zips.")
    parser.add_argument("--image-zip-dir", type=Path, required=True, help="Directory containing per-segment image zips.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where reports will be written.")
    parser.add_argument(
        "--segments",
        nargs="*",
        default=None,
        help="Optional explicit segment zip filenames to process.",
    )
    parser.add_argument("--calibration-frames", type=int, default=5, help="Frames used for calibration.")
    parser.add_argument("--max-depth", type=float, default=40.0, help="Maximum valid GT depth in meters.")
    parser.add_argument("--sample-points", type=int, default=100_000, help="Max points used for curve fitting.")


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint
    checkpoint_name = "metric_video_depth_anything" if args.metric else "video_depth_anything"
    return Path(f"./checkpoints/{checkpoint_name}_{args.encoder}.pth")


def load_frames(input_dir, pattern, max_frames):
    frame_paths = sorted(input_dir.glob(pattern))
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"No frames matched pattern '{pattern}' in '{input_dir}'.")

    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frame_paths, np.stack(frames, axis=0)


def build_model(encoder, metric, checkpoint_path, device):
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=metric)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"), strict=True)
    return model.to(device).eval()


def normalize_depth_to_uint8(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth - depth_min) / (depth_max - depth_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def save_depth_outputs(frame_paths, frames_rgb, depths, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = output_dir / "depth_arrays"
    heatmap_dir = output_dir / "heatmaps"

    if args.save_individual_depths:
        depth_dir.mkdir(parents=True, exist_ok=True)
    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        colormap = COLORMAPS[args.heatmap_colormap]
        heatmap_beta = max(0.0, 1.0 - args.heatmap_alpha)

    for index, (frame_path, frame_rgb, depth) in enumerate(zip(frame_paths, frames_rgb, depths)):
        stem = frame_path.stem
        if args.save_individual_depths:
            np.save(depth_dir / f"{stem}_depth.npy", depth.astype(np.float32))
        if args.save_heatmaps:
            depth_uint8 = normalize_depth_to_uint8(depth)
            heatmap = cv2.applyColorMap(depth_uint8, colormap)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            blend = cv2.addWeighted(frame_bgr, args.heatmap_alpha, heatmap, heatmap_beta, 0)
            cv2.imwrite(str(heatmap_dir / f"{stem}_heatmap.png"), blend)
        if index % 20 == 0 or index == len(frame_paths) - 1:
            print(f"[{index + 1}/{len(frame_paths)}] processed {stem}")

    stack_path = output_dir / "video_depth_anything_depths.npy"
    if args.save_stack:
        np.save(stack_path, depths.astype(np.float32))
    return stack_path


def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = resolve_checkpoint(args)

    if not any([args.save_heatmaps, args.save_individual_depths, args.save_stack]):
        args.save_heatmaps = True
        args.save_individual_depths = True

    frame_paths, frames = load_frames(args.input_dir, args.pattern, args.max_frames)
    print(f"Loaded {len(frame_paths)} frames from {args.input_dir}")

    model = build_model(args.encoder, args.metric, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    depths, returned_fps = model.infer_video_depth(
        frames,
        target_fps=args.target_fps,
        input_size=args.input_size,
        device=device,
        fp32=args.fp32,
    )
    print(f"Depth stack shape: {depths.shape}")
    print(f"Returned FPS: {returned_fps}")

    stack_path = save_depth_outputs(frame_paths, frames, depths, args.output_dir, args)
    print(f"Saved outputs to {args.output_dir}")
    return {
        "frame_paths": frame_paths,
        "stack_path": stack_path,
        "returned_fps": returned_fps,
    }


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
    return {path.stem.removesuffix("_depth"): np.load(path).astype(np.float32) for path in pred_files}


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

    for bin_id in range(total_bins):
        in_bin = bin_index == bin_id
        if in_bin.sum() < 25:
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
    p_clean = p_clean[deviations_mask]
    g_clean = g_clean[deviations_mask]
    if len(p_clean) < 10:
        raise RuntimeError("Too few calibration points remain after z-score filtering.")
    return p_clean, g_clean


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
            rmse = float(np.sqrt(np.mean((g_clean - g_pred) ** 2)))
            ss_res = float(np.sum((g_clean - g_pred) ** 2))
            ss_tot = float(np.sum((g_clean - g_clean.mean()) ** 2))
            r2 = float(1 - ss_res / ss_tot)
            results[name] = {"params": [float(x) for x in popt], "RMSE": rmse, "R2": r2}
        except Exception as exc:
            print(f"{name}: failed - {exc}")
    return results


def compute_metrics(pred, gt, max_depth):
    mask = np.isfinite(pred) & np.isfinite(gt) & (gt > 0.0) & (gt <= max_depth)
    if not mask.any():
        raise RuntimeError("No valid pixels remain for evaluation.")

    pred_v = np.clip(pred[mask], 1e-6, None)
    gt_v = gt[mask]
    absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
    rmse = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))
    ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
    delta1 = float(np.mean(ratio < 1.25))
    return {"AbsRel": absrel, "RMSE": rmse, "delta1": delta1, "N": int(len(gt_v))}


def resolve_prediction_inputs(args):
    if args.pred_stack:
        if not args.pred_stems_zip:
            raise RuntimeError("--pred-stems-zip is required with --pred-stack.")
        return load_prediction_stack(args.pred_stack, args.pred_stems_zip)
    if args.pred_depth_dir:
        return load_prediction_dict(args.pred_depth_dir)
    raise RuntimeError("Provide either --pred-stack or --pred-depth-dir.")


def resolve_ground_truth_inputs(args, stems):
    if args.gt_depth_zip:
        return load_gt_from_zip(args.gt_depth_zip, stems)
    if args.gt_stack and args.gt_stems:
        gt_stack = np.load(args.gt_stack, mmap_mode="r")
        gt_stems = np.load(args.gt_stems, allow_pickle=True).astype(str)
        return {stem: gt_stack[i].astype(np.float32) for i, stem in enumerate(gt_stems)}
    raise RuntimeError("Provide either --gt-depth-zip or both --gt-stack and --gt-stems.")


def evaluate_predictions(args):
    pred_dict = resolve_prediction_inputs(args)
    gt_dict = resolve_ground_truth_inputs(args, pred_dict.keys())

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

    summary = {
        "matched_stems": len(common_stems),
        "calibration_stems": common_stems[:n_calib],
        "evaluation_stems": common_stems[n_calib:],
        "prediction_shape": tuple(pred_aligned.shape),
        "ground_truth_shape": tuple(gt_aligned.shape),
        "selected_model": best_name,
        "fits": fit_results,
        **metrics,
    }

    print(f"Matched stems          : {summary['matched_stems']}")
    print(f"Calibration stems      : {summary['calibration_stems']}")
    print(f"Evaluation stems       : {summary['evaluation_stems']}")
    print(f"Prediction shape       : {summary['prediction_shape']}")
    print(f"Ground-truth shape     : {summary['ground_truth_shape']}")
    print()
    print("Calibration fits:")
    for name, result in fit_results.items():
        params = np.round(result["params"], 4)
        print(f"  {name:12s} R2={result['R2']:.4f} RMSE={result['RMSE']:.4f}m params={params}")
    print()
    print(f"Selected model         : {best_name}")
    print(f"Evaluation pixels      : {summary['N']:,}")
    print(f"AbsRel                 : {summary['AbsRel']:.4f}")
    print(f"RMSE                   : {summary['RMSE']:.4f} m")
    print(f"delta1                 : {summary['delta1']:.4f}")
    return summary


def read_npy_from_zip_bytes(zip_bytes, entry_names):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {entry.filename for entry in zf.infolist()}
        for entry_name in entry_names:
            if entry_name in names:
                with zf.open(entry_name) as handle:
                    return np.load(io.BytesIO(handle.read()), allow_pickle=True)
    raise KeyError(f"None of the expected entries were found: {entry_names}")


def stems_from_image_zip(path):
    with zipfile.ZipFile(path) as zf:
        stems = []
        for name in zf.namelist():
            p = Path(name)
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and not p.name.startswith("._"):
                stems.append(p.stem)
    stems = sorted(set(stems))
    if not stems:
        raise RuntimeError(f"No image stems found in {path}")
    return stems


def read_gt_segment_from_outer_zip(outer_zip_path, inner_name):
    with zipfile.ZipFile(outer_zip_path) as outer:
        with outer.open(inner_name) as inner_handle:
            inner_bytes = inner_handle.read()

    gt_stack = read_npy_from_zip_bytes(inner_bytes, ["depth_stack.npy", "all_depths.npy"]).astype(np.float32)
    gt_stems = read_npy_from_zip_bytes(inner_bytes, ["depth_stack_stems.npy", "all_depths_stems.npy"]).astype(str)
    return gt_stack, gt_stems


def read_pred_segment(pred_zip_path, image_zip_path):
    with zipfile.ZipFile(pred_zip_path) as pred_zip:
        with pred_zip.open("all_depths.npy") as handle:
            pred_stack = np.load(io.BytesIO(handle.read()), allow_pickle=True).astype(np.float32)

    stems = stems_from_image_zip(image_zip_path)
    if len(stems) != pred_stack.shape[0]:
        raise RuntimeError(
            f"Prediction stack frame count {pred_stack.shape[0]} does not match image stem count {len(stems)}"
        )
    return pred_stack, np.array(stems, dtype=str)


def write_segment_report(report_path, result):
    interpretation_lines = [
        "Interpretation:",
        "- Lower AbsRel and RMSE are better.",
        "- Higher delta1 is better and measures the fraction of pixels within a 1.25x error ratio.",
        f"- The selected model for this segment was `{result['selected_model']}` because it had the lowest calibration RMSE.",
    ]
    if result["AbsRel"] == 0.0 and result["RMSE"] == 0.0 and result["delta1"] == 1.0:
        interpretation_lines.append(
            "- This segment produced a perfect score in this run. Treat that cautiously because it can also indicate that the prediction and GT inputs were already numerically aligned."
        )

    lines = [
        f"Segment: {result['segment']}",
        f"Matched stems: {result['matched_stems']}",
        f"Calibration frames: {result['calibration_frames']}",
        f"Evaluation pixels: {result['evaluation_pixels']:,}",
        "",
        *interpretation_lines,
        "",
        "Calibration fits:",
    ]
    for name, fit in result["fits"].items():
        params = ", ".join(f"{p:.6f}" for p in fit["params"])
        lines.append(f"- {name}: R2={fit['R2']:.4f} RMSE={fit['RMSE']:.4f}m params=[{params}]")
    lines.extend(
        [
            "",
            f"Selected model: {result['selected_model']}",
            f"AbsRel: {result['AbsRel']:.4f}",
            f"RMSE: {result['RMSE']:.4f} m",
            f"delta1: {result['delta1']:.4f}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(output_dir, results):
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "segment",
                "matched_stems",
                "calibration_frames",
                "evaluation_pixels",
                "selected_model",
                "AbsRel",
                "RMSE",
                "delta1",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    aggregate = {
        "segments_processed": len(results),
        "avg_AbsRel": float(np.mean([r["AbsRel"] for r in results])) if results else 0.0,
        "avg_RMSE": float(np.mean([r["RMSE"] for r in results])) if results else 0.0,
        "avg_delta1": float(np.mean([r["delta1"] for r in results])) if results else 0.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


def generate_reports(args):
    output_dir = args.output_dir
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.gt_batch_zip) as outer:
        gt_inner_names = sorted(
            name
            for name in outer.namelist()
            if name.endswith(".zip") and Path(name).name.startswith("segment-")
        )
    if args.segments:
        wanted = set(args.segments)
        gt_inner_names = [name for name in gt_inner_names if Path(name).name in wanted]

    results = []
    for inner_name in gt_inner_names:
        segment_name = Path(inner_name).name
        pred_zip = args.pred_zip_dir / segment_name
        image_zip = args.image_zip_dir / segment_name
        if not pred_zip.exists():
            raise FileNotFoundError(f"Missing prediction zip: {pred_zip}")
        if not image_zip.exists():
            raise FileNotFoundError(f"Missing image zip: {image_zip}")

        gt_stack, gt_stems = read_gt_segment_from_outer_zip(args.gt_batch_zip, inner_name)
        pred_stack, pred_stems = read_pred_segment(pred_zip, image_zip)

        gt_dict = {stem: gt_stack[i] for i, stem in enumerate(gt_stems)}
        pred_dict = {stem: pred_stack[i] for i, stem in enumerate(pred_stems)}
        common_stems = sorted(set(gt_dict) & set(pred_dict))
        if len(common_stems) < 2:
            raise RuntimeError(f"{segment_name}: need at least 2 matched frames, got {len(common_stems)}")

        pred_aligned = np.stack([pred_dict[s] for s in common_stems], axis=0)
        gt_aligned = np.stack([gt_dict[s] for s in common_stems], axis=0)

        n_calib = min(args.calibration_frames, len(common_stems) - 1)
        calib_mask = (
            np.isfinite(pred_aligned[:n_calib])
            & np.isfinite(gt_aligned[:n_calib])
            & (gt_aligned[:n_calib] > 0)
            & (gt_aligned[:n_calib] <= args.max_depth)
        )
        p_clean, g_clean = noise_handler(pred_aligned[:n_calib][calib_mask], gt_aligned[:n_calib][calib_mask])
        fits = fit_all_models(p_clean, g_clean, args.sample_points)
        if not fits:
            raise RuntimeError(f"{segment_name}: all calibration models failed")

        best_name = min(fits, key=lambda name: fits[name]["RMSE"])
        best_params = np.array(fits[best_name]["params"], dtype=np.float32)
        calibrated_eval = FN_MAP[best_name](pred_aligned[n_calib:], *best_params)
        metrics = compute_metrics(calibrated_eval, gt_aligned[n_calib:], args.max_depth)

        result = {
            "segment": segment_name.replace(".zip", ""),
            "matched_stems": len(common_stems),
            "calibration_frames": n_calib,
            "evaluation_pixels": metrics["N"],
            "selected_model": best_name,
            "AbsRel": metrics["AbsRel"],
            "RMSE": metrics["RMSE"],
            "delta1": metrics["delta1"],
            "fits": fits,
        }
        results.append(result)
        write_segment_report(reports_dir / f"{result['segment']}.md", result)
        (reports_dir / f"{result['segment']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote report for {result['segment']}")

    write_summary(output_dir, results)
    print(f"Wrote batch summary to {output_dir}")


def run_infer_evaluate(args):
    infer_result = run_inference(args)
    args.pred_stack = infer_result["stack_path"]
    if not args.pred_stems_zip:
        raise RuntimeError("--pred-stems-zip is required for infer-evaluate.")
    summary = evaluate_predictions(args)
    if args.eval_output_json:
        args.eval_output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote evaluation summary to {args.eval_output_json}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "infer":
        run_inference(args)
    elif args.command == "evaluate":
        evaluate_predictions(args)
    elif args.command == "report":
        generate_reports(args)
    elif args.command == "infer-evaluate":
        run_infer_evaluate(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
