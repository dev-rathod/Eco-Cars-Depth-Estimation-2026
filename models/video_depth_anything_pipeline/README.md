# Video Depth Anything Pipeline

This folder contains the Eco Cars team's lightweight workflow for running Video Depth Anything on extracted frames and evaluating the resulting depth stack against Waymo ground truth. The upstream model source is intentionally not vendored into this repository.

## Contents

- `extract_frames.py`: extracts PNG frames from a video.
- `infer_frames.py`: runs a local clone of Video Depth Anything on a frame directory.
- `eval_video_depth_anything.py`: calibrates Video Depth Anything predictions against Waymo depth and reports AbsRel, RMSE, and delta1.
- `RMSE_RESULTS.md`: recorded evaluation command template and metrics.
- `results/`: summary plots from the evaluation workflow.

## Setup

Clone the upstream model repository outside this repo or under an ignored external directory:

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything.git external/Video-Depth-Anything
pip install -r external/Video-Depth-Anything/requirements.txt
pip install -r models/video_depth_anything_pipeline/requirements.txt
```

Download the Video Depth Anything checkpoint you plan to use and place it under:

```text
external/Video-Depth-Anything/checkpoints/
```

For example, the base relative-depth checkpoint should be:

```text
external/Video-Depth-Anything/checkpoints/video_depth_anything_vitb.pth
```

## Frame Inference

Extract video frames if needed:

```bash
python models/video_depth_anything_pipeline/extract_frames.py \
  --input_video path/to/input.mp4 \
  --output_dir data/video_depth_anything/frames
```

Run depth inference:

```bash
python models/video_depth_anything_pipeline/infer_frames.py \
  --vda_root external/Video-Depth-Anything \
  --input_dir data/video_depth_anything/frames \
  --output_dir data/video_depth_anything/predictions \
  --encoder vitb \
  --save_stack \
  --save_heatmaps
```

## Waymo Evaluation

Evaluate the stacked predictions against Waymo depth files:

```bash
python models/video_depth_anything_pipeline/eval_video_depth_anything.py \
  --pred-stack data/video_depth_anything/predictions/video_depth_anything_depths.npy \
  --pred-stems-zip path/to/images.zip \
  --gt-depth-zip path/to/depth.zip \
  --start-index 10
```

The recorded run in `RMSE_RESULTS.md` starts at `frame_00010`, calibrates on five frames, and evaluates the remaining matched frames.
