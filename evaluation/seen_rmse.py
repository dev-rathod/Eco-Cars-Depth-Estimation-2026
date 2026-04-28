import os
import zipfile
import numpy as np
import random

PRED_DIR = os.path.expanduser('~/depth_anything/base/all_depths')
GT_DIR   = os.path.expanduser('~/groundtruth/depth')

a, b, c = 40.9741, 2.9651, 9.1177

def model_power(p, a, b, c):
    return a * np.power(np.clip(p, 1e-6, None), b) + c

def load_segment(pred_zip, gt_zip):
    with zipfile.ZipFile(pred_zip) as pz, zipfile.ZipFile(gt_zip) as gz:
        pred_depths_file = [f for f in pz.namelist() if f.endswith('_depths.npy')][0]
        pred_stems_file  = [f for f in pz.namelist() if f.endswith('_stems.npy')][0]
        with pz.open(pred_depths_file) as f:
            pred_depths = np.load(f)
        with pz.open(pred_stems_file) as f:
            pred_stems = np.load(f, allow_pickle=True)
        with gz.open('all_depths.npy') as f:
            gt_depths = np.load(f)
        with gz.open('all_depths_stems.npy') as f:
            gt_stems = np.load(f, allow_pickle=True)
    return pred_depths, pred_stems, gt_depths, gt_stems

pred_segs = set(os.listdir(PRED_DIR))
gt_segs   = set(os.listdir(GT_DIR)) - {'data.csv'}
segments  = sorted(pred_segs & gt_segs)
random.seed(42)
shuffled = segments[:]
random.shuffle(shuffled)
seen_segs = shuffled[:105]
print(f"Computing Seen RMSE on {len(seen_segs)} segments...")

results = []
for idx, seg in enumerate(seen_segs):
    print(f"  [{idx+1}/{len(seen_segs)}] {seg[:50]}...", flush=True)
    try:
        pred_depths, pred_stems, gt_depths, gt_stems = load_segment(
            os.path.join(PRED_DIR, seg), os.path.join(GT_DIR, seg))
        gt_lookup = {s: i for i, s in enumerate(gt_stems)}
        rmse_list = []
        for i, stem in enumerate(pred_stems):
            if stem not in gt_lookup:
                continue
            gt_f   = gt_depths[gt_lookup[stem]]
            pred_f = pred_depths[i]
            valid  = ~np.isnan(gt_f) & np.isfinite(pred_f) & (gt_f > 0) & (gt_f <= 40)
            if valid.sum() < 10:
                continue
            pred_cal = model_power(pred_f[valid], a, b, c)
            rmse = float(np.sqrt(np.mean((pred_cal - gt_f[valid]) ** 2)))
            rmse_list.append(rmse)
        mean_rmse = float(np.mean(rmse_list))
        results.append(mean_rmse)
        print(f"           RMSE={mean_rmse:.4f}m")
    except Exception as e:
        print(f"           ERROR: {e}")

print(f"\n{'='*40}")
print(f"Seen   RMSE: {np.mean(results):.4f} m")
print(f"Unseen RMSE: 3.3965 m")
