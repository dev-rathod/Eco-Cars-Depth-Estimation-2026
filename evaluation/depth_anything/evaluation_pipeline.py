"""
Compute hyperparameters for a power function across multiple data segments.

This script:
- Iterates through all files in a given directory
- Filters noise using IQR and standard deviation methods
- Computes power-function hyperparameters per segment
- Generates and stores relative pixel (u, v) density graphs against ground truth
- Applies log transformations to computed metrics
- Evaluates performance using RMSE (for good, bad, and fitted data)
- Calculates bad pixel rates for each segment
- Saves all results into a consolidated CSV file
"""

import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import shutil
import tempfile
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR       = "/home/jiale/Documents/eco_cars/data/uploads/scale_aligned_output/small" 
INPUT_DIR        = "/home/jiale/Documents/eco_cars/data/depth_anything/small/all_depths"     
GROUND_TRUTH_DIR = "/home/jiale/Documents/eco_cars/data/uploads/groundtruth/depth"      
CSV_OUTPUT_DIR   = "/home/jiale/Documents/eco_cars/data/uploads/scale_aligned_output/small/scale_alignment_results.csv"
PLOTS_DIR        = "/home/jiale/Documents/eco_cars/data/uploads/scale_aligned_output/small/scale_alignment"

# Pre-fitted power params from notebook (used as initial guess for curve_fit)
POWER_A_INIT = 47.0671
POWER_B_INIT =  3.7100
POWER_C_INIT =  8.7821

# ── Power model (only model used for large-scale deployment) ───────────────────
def model_power(p, a, b, c):
    return a * np.power(np.clip(p, 1e-6, None), b) + c

# ── IQR + z-score noise handler ────────────────────────────────────────────────
def noiseHandler(pred40, gt40):
    groundTruth = gt40.flatten()
    predictions = pred40.flatten()

    # Using iqr to remove the outliers for the graph taking 20 outliers
    totalBins        = 20
    numberOfBins     = np.linspace(0, 1, totalBins + 1)
    numberOfBins_idx = np.digitize(predictions, numberOfBins) - 1
    binsInlier       = np.zeros(len(predictions), dtype=bool)
    minimum_entries  = 25

    for individualBins in range(totalBins):
        binsTotalEntries = (numberOfBins_idx == individualBins)
        if binsTotalEntries.sum() < minimum_entries:
            continue

        # Getting the individual entries in the area of the graph
        gt_entry = groundTruth[binsTotalEntries]
        # Getting the upper and lower quartile of the curve: 25 and 75 percent
        lowerQuartile, upperQuartile = np.percentile(gt_entry, [25, 75])
        iqr = upperQuartile - lowerQuartile
        binsInlier[binsTotalEntries] = (
            (gt_entry >= lowerQuartile - 1.5 * iqr) &
            (gt_entry <= upperQuartile + 1.5 * iqr)
        )

    # Mask after getting the inliers
    pred_c, gt_c = predictions[binsInlier], groundTruth[binsInlier]
    # Isolating the values outside the standard deviation
    polyGraph      = np.polyfit(pred_c, gt_c, deg=4)
    residuals      = gt_c - np.polyval(polyGraph, pred_c)
    # Having variable for sensitivity of the readings
    sensitivity    = 4
    deviationsMask = np.abs(stats.zscore(residuals)) < sensitivity
    return pred_c[deviationsMask], gt_c[deviationsMask]

# ── Fit power model on cleaned data ───────────────────────────────────────────
def fit_power_model(p_clean, g_clean, n_sample=100_000):
    # Subsample — curve_fit doesn't need 3M points, 100k is plenty
    if len(p_clean) > n_sample:
        idx   = np.random.choice(len(p_clean), n_sample, replace=False)
        p_fit = p_clean[idx]
        g_fit = g_clean[idx]
    else:
        p_fit, g_fit = p_clean, g_clean

    popt, _ = curve_fit(
        model_power, p_fit, g_fit,
        p0=[POWER_A_INIT, POWER_B_INIT, POWER_C_INIT],
        maxfev=50_000
    )
    g_pred = model_power(p_clean, *popt)
    ss_res = np.sum((g_clean - g_pred) ** 2)
    ss_tot = np.sum((g_clean - g_clean.mean()) ** 2)
    r2   = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((g_clean - g_pred) ** 2))
    return popt, r2, rmse

# ── Compute all metrics including pixel density ────────────────────────────────
def compute_metrics(pred, gt, popt):
    eval_mask    = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (gt <= 40)
    gt_eval      = gt[eval_mask]
    pred_aligned = model_power(pred[eval_mask], *popt)
    residuals    = np.abs(pred_aligned - gt_eval)

    rmse_all  = np.sqrt(np.mean((pred_aligned - gt_eval) ** 2))

    # RMSE excluding >5m error pixels (good pixels)
    good_mask = residuals <= 5.0
    rmse_good = np.sqrt(np.mean(residuals[good_mask] ** 2)) if good_mask.any() else np.nan
    bad_pct   = 100.0 * (~good_mask).mean()
    good_pct  = 100.0 - bad_pct

    # Pixel density: fraction of GT pixels with valid LiDAR measurements
    pixel_density_pct = 100.0 * (np.isfinite(gt) & (gt > 0)).sum() / gt.size

    return rmse_all, rmse_good, bad_pct, good_pct, pixel_density_pct

# ── Save diagnostic scatter plot for each segment ─────────────────────────────
def save_scatter_plot(pred, gt, popt, segment_id, plots_dir):
    eval_mask  = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (gt <= 40)
    x          = pred[eval_mask].flatten()
    y          = gt[eval_mask].flatten()
    sample_idx = np.random.choice(len(x), min(5_000, len(x)), replace=False)
    x_line     = np.linspace(x.min(), x.max(), 200)
    y_fitted   = model_power(x_line, *popt)
    a, b, c    = popt

    plt.figure(figsize=(6, 6))
    plt.scatter(x[sample_idx], y[sample_idx], s=2, alpha=0.3, label="data sample")
    plt.plot(x_line, y_fitted, "r-", lw=2,
             label=f"Power: {a:.2f}·x^{b:.2f}+{c:.2f}")
    plt.xlabel("Predicted Depth (relative)")
    plt.ylabel("Ground Truth Depth (m)")
    plt.title(segment_id)
    plt.legend(fontsize=8)
    plt.grid(True)
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{segment_id}.png")
    plt.savefig(plot_path, dpi=100)
    plt.close()
    return plot_path

# ── CSV helpers ────────────────────────────────────────────────────────────────
COLUMNS = [
    "file_id", "input_shape", "output_shape",
    "formula", "a", "b", "c",
    "R2", "RMSE_fit",
    "RMSE_all_pixels", "RMSE_good_pixels",
    "bad_pixels_pct", "good_pixels_pct",
    "pixel_density_pct",
    "output_image_id",
]

def load_or_create_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    df = pd.DataFrame(columns=COLUMNS)
    df.to_csv(path, index=False)
    return df

def append_csv_row(path, row_dict):
    pd.DataFrame([row_dict]).to_csv(path, mode="a", header=False, index=False)

# ── Archive / extraction helpers ───────────────────────────────────────────────
def unzip_to_temp(zip_path):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    return tmp

def zip_aligned_output(npy_path, segment_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{segment_id}_aligned.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(npy_path, arcname=os.path.basename(npy_path))
    return zip_path

# ── Main pipeline ──────────────────────────────────────────────────────────────
# V1_CSV = "/home/jiale/Documents/eco_cars/data/uploads/scale_aligned_output_V1/scale_alignment_results.csv"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_or_create_csv(CSV_OUTPUT_DIR)

    already_done = set()
    # if os.path.exists(V1_CSV):
    #     v1_df = pd.read_csv(V1_CSV)
    #     already_done = set(v1_df["file_id"].dropna().tolist())
    #     print(f"Loaded {len(already_done)} already-processed segment(s) from V1 CSV — will skip.\n")

    # DO os.walk in both the input folder
    input_zips = sorted(Path(INPUT_DIR).glob("*.zip"))
    print(f"Found {len(input_zips)} segment(s) to process.\n")

    for zip_path in input_zips:
        segment_id  = zip_path.stem
        gt_zip_path = Path(GROUND_TRUTH_DIR) / zip_path.name

        print(f"── {segment_id}")

        if segment_id in already_done:
            print(f"   [SKIP] Already processed in V1 — skipping.\n")
            continue

        if not gt_zip_path.exists():
            print(f"   [SKIP] No matching ground-truth zip found.\n")
            continue

        # In the for loop extract the data frames (unzip both archives)
        pred_tmp = unzip_to_temp(str(zip_path))
        gt_tmp   = unzip_to_temp(str(gt_zip_path))

        try:
            pred = np.load(os.path.join(pred_tmp, f"{segment_id}_depths.npy"))
            gt   = np.load(os.path.join(gt_tmp,   "all_depths.npy"))

            if pred.ndim == 4 and pred.shape[-1] == 1:
                pred = pred[..., 0]
            if gt.ndim == 4 and gt.shape[-1] == 1:
                gt = gt[..., 0]

            assert pred.shape == gt.shape, \
                f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"

            print(f"   pred shape: {pred.shape}  gt shape: {gt.shape}")
            print(f"   pred [{np.nanmin(pred):.4f}, {np.nanmax(pred):.4f}]  "
                  f"gt [{np.nanmin(gt):.4f}, {np.nanmax(gt):.4f}]")

            # Run calibration analysis for each utility
            mask   = np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (gt < 40)
            pred40 = pred[mask]
            gt40   = gt[mask]

            p_clean, g_clean = noiseHandler(pred40, gt40)
            print(f"   Noise handler: {len(pred40.flatten()):,} → {len(p_clean):,} points")

            popt, r2, rmse_fit = fit_power_model(p_clean, g_clean)
            a, b, c = popt
            formula = f"GT = {a:.4f} * x^{b:.4f} + {c:.4f}"
            print(f"   {formula}  R²={r2:.4f}  RMSE_fit={rmse_fit:.4f}m")

            rmse_all, rmse_good, bad_pct, good_pct, pixel_density_pct = \
                compute_metrics(pred, gt, popt)
            print(f"   RMSE all={rmse_all:.4f}m  good={rmse_good:.4f}m  "
                  f"bad={bad_pct:.1f}%  pixel_density={pixel_density_pct:.2f}%")

            # Store all the details in the csv file and load the image in the output
            plot_path       = save_scatter_plot(pred, gt, popt, segment_id, PLOTS_DIR)
            output_image_id = os.path.basename(plot_path)

            # Produce the scale-aligned depth array and zip it to the output folder
            aligned     = model_power(pred, a, b, c)
            aligned_npy = os.path.join(pred_tmp, f"{segment_id}_aligned_depths.npy")
            np.save(aligned_npy, aligned)
            zip_out = zip_aligned_output(aligned_npy, segment_id, OUTPUT_DIR)
            print(f"   Saved → {zip_out}")

            append_csv_row(CSV_OUTPUT_DIR, {
                "file_id":           segment_id,
                "input_shape":       str(pred.shape),
                "output_shape":      str(aligned.shape),
                "formula":           formula,
                "a":                 (float(a), 6),
                "b":                 (float(b), 6),
                "c":                 (float(c), 6),
                "R2":                (float(r2), 6),
                "RMSE_fit":          (float(rmse_fit), 6),
                "RMSE_all_pixels":   (float(rmse_all), 6),
                "RMSE_good_pixels":  (float(rmse_good), 6),
                "bad_pixels_pct":    (float(bad_pct), 4),
                "good_pixels_pct":   (float(good_pct), 4),
                "pixel_density_pct": (float(pixel_density_pct), 4),
                "output_image_id":   output_image_id,
            })

        except Exception as e:
            print(f"   [ERROR] {e}")

        finally:
            # Delete the unzipped files permanently to save space
            shutil.rmtree(pred_tmp, ignore_errors=True)
            shutil.rmtree(gt_tmp,   ignore_errors=True)
            # Flush all large arrays and matplotlib state from memory
            try:
                del pred, gt, aligned, pred40, gt40, p_clean, g_clean
            except NameError:
                pass
            plt.close("all")
            gc.collect()

        # Go to the next file (loop continues)

    print(f"\nDone. Results written to {CSV_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
