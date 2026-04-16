# Video Depth Anything RMSE Evaluation

This evaluation uses the Google Drive artifacts:

- Predictions: `depth_estimators/VideoDepthAnything/video_depth_anything_depths.npy`
- Input frame stems: `depth_estimators/waymo_open/output/images.zip`
- Ground truth: `depth_estimators/waymo_open/output/depth.zip`

The first ten ground-truth frames in the Drive zip are on a different scale than the rest of the segment. To match the usable portion of the segment, evaluation starts at `frame_00010`, calibrates on `frame_00010` through `frame_00014`, and evaluates on `frame_00015` through `frame_00198`.

Notebook workflow:

- Open `VideoDepthAnything_Pipeline.ipynb`
- Go to `Section 2` and set:
  - `PRED_STACK = G:\...\VideoDepthAnything\video_depth_anything_depths.npy`
  - `PRED_STEMS_ZIP = G:\...\waymo_open\output\images.zip`
  - `GT_DEPTH_ZIP = G:\...\waymo_open\output\depth.zip`
  - `START_INDEX = 10`
- Run `Section 4`

Results:

```text
Matched stems     : 189
Calibration stems : frame_00010 through frame_00014
Evaluation stems  : frame_00015 through frame_00198
Selected model    : exponential
Evaluation pixels : 3,282,059
AbsRel            : 0.0630
RMSE              : 2.2860 m
delta1            : 0.9714
```

Metric notes:

- `AbsRel` is mean absolute relative error. Lower is better.
- `RMSE` is root mean squared error in meters. Lower is better.
- `delta1` is the fraction of valid pixels within a `1.25x` prediction-to-ground-truth ratio. Higher is better.

Calibration report notes:

- A separate batch calibration run was generated for 22 downloaded segment zips and stored under `pipeline_result/`.
- Those per-segment reports use the same calibration idea: calibrate on the first 5 matched frames, fit `hyperbolic`, `power`, `log`, and `exponential`, then keep the model with the lowest calibration RMSE.
- In that 22-segment batch, every segment selected `power` and reported `AbsRel = 0.0`, `RMSE = 0.0`, and `delta1 = 1.0`.
- Those perfect scores should be interpreted carefully. They can indicate genuine agreement, but they can also mean the prediction arrays and GT arrays were already numerically identical or already aligned before calibration.
