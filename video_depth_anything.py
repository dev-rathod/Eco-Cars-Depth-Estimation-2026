import os
import sys
import shutil
import subprocess
import zipfile
import time
import cv2
import numpy as np
from pathlib import Path

INPUT_ROOT  = Path("/home/jiale/Documents/eco_cars/data/output_images")
OUTPUT_ROOT = Path("/home/jiale/Documents/eco_cars/data/video_depth/large")

VDA_DIR = Path("/home/jiale/Documents/eco_cars/Video-Depth-Anything")
RUN_TRT = VDA_DIR / "run_trt.py"

ENCODER = "vitl"

IMAGES_ZIP_DIR     = OUTPUT_ROOT / "images"
DEPTHS_ZIP_DIR     = OUTPUT_ROOT / "depth"
ALL_DEPTHS_ZIP_DIR = OUTPUT_ROOT / "all_depths"

for d in (IMAGES_ZIP_DIR, DEPTHS_ZIP_DIR, ALL_DEPTHS_ZIP_DIR):
    d.mkdir(parents=True, exist_ok=True)


def normalised_numpy_depths(inference_dir: Path, npy_dir: Path):
    npy_dir.mkdir(parents=True, exist_ok=True)

    gray_pngs = sorted(inference_dir.glob("*_gray.png"))
    assert gray_pngs, f"No gray depth images in {inference_dir}"

    for path in gray_pngs:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        img = 1.0 - img
        np.save(npy_dir / f"{path.stem.replace('_gray','')}_depth.npy", img)


def stack_numpy(npy_dir: Path, output_dir: Path, segment_name: str):
    files = sorted(npy_dir.glob("*_depth.npy"))
    arr   = np.stack([np.load(f) for f in files], axis=0)

    stack_path = output_dir / f"{segment_name}_depths.npy"
    np.save(stack_path, arr)
    return stack_path


def zip_files(zip_path: Path, files):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def main():
    segments = sorted(p for p in INPUT_ROOT.iterdir() if p.is_dir())

    print(f"Found {len(segments)} segments\n")

    for folder in segments:
        segment_name = folder.name
        seg_out = OUTPUT_ROOT / segment_name
        infer_dir = seg_out / "infer"
        npy_dir   = seg_out / "npy"

        infer_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SEG] {segment_name}")
        t0 = time.perf_counter()

        # ===== RUN TRT DIRECTLY ON FRAMES =====
        cmd = [
            sys.executable, str(RUN_TRT),
            "--input_dir", str(folder),
            "--output_dir", str(infer_dir),
            "--encoder", ENCODER,
            "--grayscale"
        ]

        result = subprocess.run(cmd, cwd=str(VDA_DIR))
        if result.returncode != 0:
            print(f"[FAIL] {segment_name}")
            continue

        depth_dir = infer_dir / f"{segment_name}_depths"

        # ===== POST =====
        normalised_numpy_depths(depth_dir, npy_dir)
        stack_path = stack_numpy(npy_dir, seg_out, segment_name)

        # ===== ZIP =====
        zip_files(IMAGES_ZIP_DIR / f"{segment_name}.zip",
                  list(depth_dir.glob("*_heatmap.png")))

        zip_files(DEPTHS_ZIP_DIR / f"{segment_name}.zip",
                  list(npy_dir.glob("*.npy")))

        zip_files(ALL_DEPTHS_ZIP_DIR / f"{segment_name}.zip",
                  [stack_path])

        shutil.rmtree(seg_out, ignore_errors=True)

        print(f"[DONE] {segment_name}  {time.perf_counter()-t0:.1f}s\n")

    print("ALL DONE")


if __name__ == "__main__":
    main()