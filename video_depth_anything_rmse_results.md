# Video Depth Anything RMSE Evaluation

This evaluation uses the Google Drive artifacts:

- Predictions: `depth_estimators/VideoDepthAnything/video_depth_anything_depths.npy`
- Input frame stems: `depth_estimators/waymo_open/output/images.zip`
- Ground truth: `depth_estimators/waymo_open/output/depth.zip`

The first ten ground-truth frames in the Drive zip are on a different scale than the rest of the segment. To match the usable portion of the segment, evaluation starts at `frame_00010`, calibrates on `frame_00010` through `frame_00014`, and evaluates on `frame_00015` through `frame_00198`.

Command:

```powershell
python eval_video_depth_anything.py `
  --pred-stack "G:\공유 드라이브\depth_estimators\VideoDepthAnything\video_depth_anything_depths.npy" `
  --pred-stems-zip "G:\공유 드라이브\depth_estimators\waymo_open\output\images.zip" `
  --gt-depth-zip "G:\공유 드라이브\depth_estimators\waymo_open\output\depth.zip" `
  --start-index 10
```

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
