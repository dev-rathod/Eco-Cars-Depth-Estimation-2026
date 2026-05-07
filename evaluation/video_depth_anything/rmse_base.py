import os
import zipfile
import numpy as np
import csv
import random
from scipy.optimize import curve_fit
from scipy import stats

PRED_DIR   = os.path.expanduser('~/video_depth/base/all_depths')
GT_DIR     = os.path.expanduser('~/groundtruth/depth')
OUT_CSV    = os.path.expanduser('~/rmse_results_base.csv')
SEED       = 42
SEEN_RATIO = 0.30

def load_segment(pred_zip, gt_zip):
    with zipfile.ZipFile(pred_zip) as pz, zipfile.ZipFile(gt_zip) as gz:
        pred_depths_file = [f for f in pz.namelist() if f.endswith('_depths.npy')][0]
        with pz.open(pred_depths_file) as f:
            pred_depths = np.load(f)
        with gz.open('all_depths.npy') as f:
            gt_depths = np.load(f)
    return pred_depths, gt_depths

def noiseHandler(pred40, gt40):
    predictions = pred40.flatten()
    groundTruth = gt40.flatten()
    totalBins = 20
    numberOfBins = np.linspace(0, 1, totalBins+1)
    numberOfBins_index = np.digitize(predictions, numberOfBins) - 1
    binsInlier = np.zeros(len(predictions), dtype=bool)
    minimum_entries = 25
    for individualBins in range(totalBins):
        binsTotalEntries = (numberOfBins_index == individualBins)
        if binsTotalEntries.sum() < minimum_entries:
            continue
        goundtruth_entry = groundTruth[binsTotalEntries]
        lowerQuartile, upperQuartile = np.percentile(goundtruth_entry, [25, 75])
        iqr = upperQuartile - lowerQuartile
        binsInlier[binsTotalEntries] = (
            (goundtruth_entry >= lowerQuartile - 1.5 * iqr) &
            (goundtruth_entry <= upperQuartile + 1.5 * iqr)
        )
    predictions_cleaned = predictions[binsInlier]
    groundTruth_cleaned = groundTruth[binsInlier]
    polyGraph = np.polyfit(predictions_cleaned, groundTruth_cleaned, deg=4)
    residual_entries = groundTruth_cleaned - np.polyval(polyGraph, predictions_cleaned)
    standardDeviations = np.abs(stats.zscore(residual_entries))
    sensitivity = 4
    devitionsMask = standardDeviations < sensitivity
    return predictions_cleaned[devitionsMask], groundTruth_cleaned[devitionsMask]

def model_power(p, a, b, c):
    return a * np.power(np.clip(p, 1e-6, None), b) + c

pred_segs = set(os.listdir(PRED_DIR))
gt_segs   = set(os.listdir(GT_DIR)) - {'data.csv'}
segments  = sorted(pred_segs & gt_segs)
print(f"Total matched segments: {len(segments)}")

random.seed(SEED)
shuffled = segments[:]
random.shuffle(shuffled)
n_seen      = int(len(shuffled) * SEEN_RATIO)
seen_segs   = shuffled[:n_seen]
unseen_segs = shuffled[n_seen:]
print(f"Seen: {len(seen_segs)}  Unseen: {len(unseen_segs)}")

print("\nPooling seen segments for calibration...")
all_pred = []
all_gt   = []
for idx, seg in enumerate(seen_segs):
    print(f"  [{idx+1}/{len(seen_segs)}] {seg[:50]}...", flush=True)
    pred_depths, gt_depths = load_segment(
        os.path.join(PRED_DIR, seg), os.path.join(GT_DIR, seg))
    n_frames = min(len(pred_depths), len(gt_depths))
    for i in range(n_frames):
        gt_f   = gt_depths[i]
        pred_f = pred_depths[i]
        valid  = ~np.isnan(gt_f) & np.isfinite(pred_f) & (gt_f > 0) & (gt_f <= 40)
        if valid.sum() < 10:
            continue
        all_pred.append(pred_f[valid])
        all_gt.append(gt_f[valid])

all_pred = np.concatenate(all_pred)
all_gt   = np.concatenate(all_gt)
print(f"Total calibration pixels: {len(all_pred):,}")

print("\nCleaning calibration data...")
p_clean, g_clean = noiseHandler(all_pred, all_gt)
print(f"Pixels before cleaning: {len(all_pred):,}")
print(f"Pixels after  cleaning: {len(p_clean):,}")

print("\nFitting power curve...")
popt, _ = curve_fit(model_power, p_clean, g_clean,
                    p0=[10.0, 0.5, 0.0], maxfev=50000)
a, b, c = popt
print(f"Calibration formula: GT = {a:.4f} * x^{b:.4f} + {c:.4f}")

g_pred_fit = model_power(p_clean, a, b, c)
ss_res = np.sum((g_clean - g_pred_fit) ** 2)
ss_tot = np.sum((g_clean - g_clean.mean()) ** 2)
r2   = 1 - ss_res / ss_tot
rmse_fit = np.sqrt(np.mean((g_clean - g_pred_fit) ** 2))
print(f"R²={r2:.4f}  RMSE on fit={rmse_fit:.4f}m")

print("\nComputing RMSE on unseen segments...")
results = []
for idx, seg in enumerate(unseen_segs):
    print(f"  [{idx+1}/{len(unseen_segs)}] {seg[:50]}...", flush=True)
    try:
        pred_depths, gt_depths = load_segment(
            os.path.join(PRED_DIR, seg), os.path.join(GT_DIR, seg))
        n_frames = min(len(pred_depths), len(gt_depths))
        rmse_list = []
        for i in range(n_frames):
            gt_f   = gt_depths[i]
            pred_f = pred_depths[i]
            valid  = ~np.isnan(gt_f) & np.isfinite(pred_f) & (gt_f > 0) & (gt_f <= 40)
            if valid.sum() < 10:
                continue
            pred_cal = model_power(pred_f[valid], a, b, c)
            rmse = float(np.sqrt(np.mean((pred_cal - gt_f[valid]) ** 2)))
            rmse_list.append(rmse)
        mean_rmse = float(np.mean(rmse_list))
        results.append({'segment': seg, 'split': 'unseen',
                        'mean_rmse': mean_rmse, 'n_frames': len(rmse_list)})
        print(f"           RMSE={mean_rmse:.4f}m  frames={len(rmse_list)}")
    except Exception as e:
        print(f"           ERROR: {e}")

with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['segment','split','mean_rmse','n_frames'])
    writer.writeheader()
    writer.writerows(results)

unseen_rmse = np.mean([r['mean_rmse'] for r in results])
print(f"\n{'='*40}")
print(f"Calibration formula: GT = {a:.4f} * x^{b:.4f} + {c:.4f}")
print(f"Calibration R²     : {r2:.4f}")
print(f"Unseen RMSE        : {unseen_rmse:.4f} m")
print(f"Results saved to {OUT_CSV}")
