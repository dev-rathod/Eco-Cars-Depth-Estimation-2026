import os
import sys
import shutil
import subprocess
import zipfile
import time
from pathlib import Path

import cv2
import numpy as np

# CONFIGS

INPUT_ROOT  = Path("/home/jiale/Documents/eco_cars/data/output_images")
OUTPUT_ROOT = Path("/home/jiale/Documents/eco_cars/data/depth_anything/small")
TEMP_ROOT   = OUTPUT_ROOT / "temp"

DA_V2_DIR    = Path("/home/jiale/Documents/eco_cars/Depth-Anything-V2")
RUN_TENSORRT = DA_V2_DIR / "run_tensorrt.py"

ENCODER = "vits"   # vits/vitb fit in 8 GB VRAM; vitl needs ~20 GB RAM to compile

# Output buckets
IMAGES_ZIP_DIR     = OUTPUT_ROOT / "images"      # heatmap PNGs per segment
DEPTHS_ZIP_DIR     = OUTPUT_ROOT / "depth"       # per-frame normalised .npy per segment
ALL_DEPTHS_ZIP_DIR = OUTPUT_ROOT / "all_depths"  # stacked (N,H,W) array per segment

for d in (IMAGES_ZIP_DIR, DEPTHS_ZIP_DIR, ALL_DEPTHS_ZIP_DIR, TEMP_ROOT):
    d.mkdir(parents=True, exist_ok=True)


# HELPERS

def normalised_numpy_depths(inference_dir: Path, npy_dir: Path) -> None:
    """Convert *_gray.png depth outputs to normalised + inverted float32 .npy files."""
    npy_dir.mkdir(parents=True, exist_ok=True)
    gray_pngs = sorted(
        p for p in inference_dir.iterdir()
        if p.suffix.lower() == ".png" and p.stem.endswith("_gray")
    )
    assert gray_pngs, f"No *_gray.png files found in {inference_dir}"

    for path in gray_pngs:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] unreadable: {path.name}")
            continue
        img_norm      = img.astype(np.float32) / 255.0
        img_inverted  = 1.0 - img_norm
        original_stem = path.stem.replace("_gray", "")
        np.save(npy_dir / f"{original_stem}_depth.npy", img_inverted)


def stack_numpy(npy_dir: Path, output_dir: Path, segment_name: str) -> tuple[Path, Path, tuple]:
    """Stack all per-frame *_depth.npy files into a single (N, H, W) float32 array."""
    pred_npy_files = sorted(npy_dir.glob("*_depth.npy"))
    assert pred_npy_files, f"No *_depth.npy files in {npy_dir}"

    frames = [np.load(f) for f in pred_npy_files]
    stems  = [f.stem.replace("_depth", "") for f in pred_npy_files]

    pred_stack = np.stack(frames, axis=0)   # (N, H, W)
    pred_stems = np.array(stems)

    stack_path = output_dir / f"{segment_name}_depths.npy"
    stems_path = output_dir / f"{segment_name}_stems.npy"
    np.save(stack_path, pred_stack)
    np.save(stems_path, pred_stems)

    print(f"[POST] {segment_name}  stack shape={pred_stack.shape}")
    return stack_path, stems_path, pred_stack.shape


def zip_files(zip_path: Path, files: list[Path]) -> None:
    """Pack a list of files into a zip archive (low compression for speed)."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


# ================= MAIN =================

def main(encoder: str = ENCODER) -> None:
    segment_folders = sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(segment_folders)} segment folder(s)\n")

    for folder in segment_folders:
        segment_name = folder.name
        seg_temp     = TEMP_ROOT / segment_name
        infer_dir    = seg_temp / "inference"
        npy_dir      = seg_temp / "depths"
        infer_dir.mkdir(parents=True, exist_ok=True)

        t_seg = time.perf_counter()
        print(f"[SEG] {segment_name}")

        # Run TRT inference — outputs *_heatmap.png + *_gray.png
        cmd = [
            sys.executable, str(RUN_TENSORRT),
            "--encoder",  encoder,
            "--img-path", str(folder),
            "--outdir",   str(infer_dir),
            "--pred-only",
            "--grayscale",
        ]
        result = subprocess.run(cmd, cwd=str(DA_V2_DIR), text=True, timeout=1800)
        if result.returncode != 0:
            print(f"[ERROR] Inference failed for {segment_name} — skipping.\n")
            shutil.rmtree(seg_temp, ignore_errors=True)
            continue

        # Convert *_gray.png → normalised float32 .npy
        normalised_numpy_depths(infer_dir, npy_dir)

        # Stack per-frame .npy into (N, H, W) array
        stack_path, stems_path, stack_shape = stack_numpy(npy_dir, seg_temp, segment_name)

        # Write three zips
        heatmaps  = sorted(infer_dir.glob("*_heatmap.png"))
        npy_files = sorted(npy_dir.glob("*_depth.npy"))

        zip_files(IMAGES_ZIP_DIR / f"{segment_name}.zip", heatmaps)
        print(f"[ZIP]  images      {len(heatmaps)} heatmaps")

        zip_files(DEPTHS_ZIP_DIR / f"{segment_name}.zip", npy_files)
        print(f"[ZIP]  depth       {len(npy_files)} npy files")

        zip_files(ALL_DEPTHS_ZIP_DIR / f"{segment_name}.zip", [stack_path, stems_path])
        print(f"[ZIP]  all_depths  stack {stack_shape}")

        # Delete temp dir for this segment to free disk space
        shutil.rmtree(seg_temp, ignore_errors=True)

        elapsed = time.perf_counter() - t_seg
        print(f"[DONE] {segment_name}  {elapsed:.1f}s\n")

    print("ALL DONE")

if __name__ == "__main__":
    main()
