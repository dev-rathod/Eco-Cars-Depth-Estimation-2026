import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Video Depth Anything on a directory of frames and export heatmaps."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input frames.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference outputs.")
    parser.add_argument("--pattern", type=str, default="*.png", help="Glob pattern for frame discovery.")
    parser.add_argument("--input_size", type=int, default=518, help="Model input size.")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override path.")
    parser.add_argument("--max_frames", type=int, default=-1, help="Optional limit on the number of frames.")
    parser.add_argument("--metric", action="store_true", help="Use the metric model checkpoint.")
    parser.add_argument("--fp32", action="store_true", help="Run inference in float32 instead of autocast.")
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
        choices=["jet", "inferno", "magma", "plasma", "turbo", "viridis"],
        help="OpenCV colormap used for heatmap overlays.",
    )
    return parser.parse_args()


def resolve_checkpoint(args):
    if args.checkpoint:
        return args.checkpoint

    checkpoint_name = "metric_video_depth_anything" if args.metric else "video_depth_anything"
    return f"./checkpoints/{checkpoint_name}_{args.encoder}.pth"


def load_frames(input_dir, pattern, max_frames):
    frame_paths = sorted(Path(input_dir).glob(pattern))
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

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    model = VideoDepthAnything(**model_configs[encoder], metric=metric)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
    return model.to(device).eval()


def normalize_depth_to_uint8(depth):
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min:
        return np.zeros(depth.shape, dtype=np.uint8)
    scaled = (depth - depth_min) / (depth_max - depth_min)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def get_colormap(name):
    return {
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }[name]


def save_depth_outputs(frame_paths, frames_rgb, depths, output_dir, args):
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    depth_dir = output_root / "depth_arrays"
    heatmap_dir = output_root / "heatmaps"

    if args.save_individual_depths:
        depth_dir.mkdir(parents=True, exist_ok=True)

    if args.save_heatmaps:
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        colormap = get_colormap(args.heatmap_colormap)
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

    if args.save_stack:
        np.save(output_root / "video_depth_anything_depths.npy", depths.astype(np.float32))


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = resolve_checkpoint(args)

    if not any([args.save_heatmaps, args.save_individual_depths, args.save_stack]):
        args.save_heatmaps = True
        args.save_individual_depths = True

    frame_paths, frames = load_frames(args.input_dir, args.pattern, args.max_frames)
    print(f"Loaded {len(frame_paths)} frames from {args.input_dir}")

    model = build_model(args.encoder, args.metric, checkpoint_path, device)
    print(f"Loaded checkpoint: {checkpoint_path}")

    depths, _ = model.infer_video_depth(
        frames,
        target_fps=1,
        input_size=args.input_size,
        device=device,
        fp32=args.fp32,
    )
    print(f"Depth stack shape: {depths.shape}")

    save_depth_outputs(frame_paths, frames, depths, args.output_dir, args)
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
