# Waymo RMSE Comparison

This records the RMSE calculation used to compare Depth Anything V2 and Video Depth Anything on the same Waymo frame set.

## Method

- Ground truth: Waymo `groundtruth.npy`.
- Predictions:
  - Depth Anything V2 `depthanythingv2_depths.npy`
  - Video Depth Anything `video_depth_anything_depths.npy`
- Target comparison resolution: `1280 x 1920`, matching the Depth Anything V2 output shape.
- Ground truth and Video Depth Anything predictions are resized to the target resolution with bilinear interpolation when their native resolution differs.
- Valid mask: finite ground truth, ground truth greater than zero, finite prediction, prediction greater than zero.
- Both raw RMSE and linearly aligned RMSE are reported.
- Linear alignment uses least squares fit from prediction to ground truth:

```text
ground_truth = slope * prediction + intercept
```

## Command

Example:

```bash
python benchmark/eval/compute_waymo_rmse.py \
  --groundtruth /path/to/groundtruth.npy \
  --predictions /path/to/depthanythingv2_depths.npy /path/to/video_depth_anything_depths.npy \
  --names "Depth Anything V2" "Video Depth Anything" \
  --target-shape 1280 1920
```

## Result

Using the Google Drive files from the project shared drive:

```text
Depth Anything V2
  target_shape: (1280, 1920)
  frames: 199
  valid_pixels: 1780372
  linear_fit_slope: 54.28378866297587
  linear_fit_intercept: -6.080567074853338
  rmse_before_alignment: 39.167256
  rmse_after_alignment: 12.294337

Video Depth Anything
  target_shape: (1280, 1920)
  frames: 199
  valid_pixels: 1780390
  linear_fit_slope: 67.98521140229272
  linear_fit_intercept: -18.628891216351423
  rmse_before_alignment: 39.145489
  rmse_after_alignment: 11.638902
```

With this setup, Video Depth Anything has the lower aligned RMSE:

```text
Depth Anything V2:     12.294337
Video Depth Anything:  11.638902
```
