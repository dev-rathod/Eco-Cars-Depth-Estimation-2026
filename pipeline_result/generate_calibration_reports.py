import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate calibration reports for a batch of segment zips using the GitHub calibration pipeline logic."
    )
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
    return parser.parse_args()


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
        except Exception:
            continue
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


def read_npy_from_zip_bytes(zip_bytes, entry_names):
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = {entry.filename for entry in zf.infolist()}
        for entry_name in entry_names:
            if entry_name in names:
                with zf.open(entry_name) as handle:
                    return np.load(io.BytesIO(handle.read()), allow_pickle=True)
    raise KeyError(f"None of the expected entries were found: {entry_names}")


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
    lines = [
        f"Segment: {result['segment']}",
        f"Matched stems: {result['matched_stems']}",
        f"Calibration frames: {result['calibration_frames']}",
        f"Evaluation pixels: {result['evaluation_pixels']:,}",
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


def main():
    args = parse_args()
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

    write_summary(output_dir, results)


if __name__ == "__main__":
    main()
