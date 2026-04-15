from pathlib import Path
import cv2

video_path = Path(r"C:\Users\User\Documents\Video-Depth-Anything\outputs_test\davis_rollercoaster_vis.mp4")
output_dir = Path(r"C:\Users\User\Documents\Video-Depth-Anything\outputs_test\vis_frames")
output_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {video_path}")

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out_path = output_dir / f"frame_{frame_idx:05d}.png"
    cv2.imwrite(str(out_path), frame)
    frame_idx += 1

cap.release()
print(f"Done. Saved {frame_idx} PNG frames to: {output_dir}")