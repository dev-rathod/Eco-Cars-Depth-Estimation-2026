import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Extract video frames as PNG files.")
    parser.add_argument("--input_video", type=Path, required=True, help="Path to the source video.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save PNG frames.")
    parser.add_argument("--prefix", default="frame", help="Output filename prefix.")
    parser.add_argument("--start_index", type=int, default=0, help="Starting frame index for output names.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")

    frame_idx = args.start_index
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = args.output_dir / f"{args.prefix}_{frame_idx:05d}.png"
        cv2.imwrite(str(out_path), frame)
        frame_idx += 1
        saved_count += 1

    cap.release()
    print(f"Done. Saved {saved_count} PNG frames to: {args.output_dir}")


if __name__ == "__main__":
    main()
