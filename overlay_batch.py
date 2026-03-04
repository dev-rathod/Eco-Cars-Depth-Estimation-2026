import os
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"RGB not found or unreadable: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_depth_npy(path: Path) -> np.ndarray:
    d = np.load(str(path))
    d = np.squeeze(d)  # (1,1,192,640) -> (192,640)
    if d.ndim != 2:
        raise ValueError(f"Unexpected depth shape {d.shape} from {path}")
    return d.astype(np.float32)


def make_overlay(rgb: np.ndarray, depth: np.ndarray, alpha: float = 0.55):
    H, W = rgb.shape[:2]

    # resize depth to match RGB
    depth_rs = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    # normalize ONLY for visualization (depth_rs 자체는 float로 따로 저장)
    d_min, d_max = float(depth_rs.min()), float(depth_rs.max())
    if d_max - d_min < 1e-8:
        d_norm = np.zeros_like(depth_rs, dtype=np.uint8)
    else:
        d_norm = ((depth_rs - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

    overlay = (
        rgb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha
    ).clip(0, 255).astype(np.uint8)

    stats = {
        "min": d_min,
        "max": d_max,
        "mean": float(depth_rs.mean()),
        "finite_ratio": float(np.isfinite(depth_rs).mean()),
    }
    return overlay, depth_rs, stats


def main():
    # ✅ 네 폴더 구조 기준
    images_dir = Path("test_frames/images")
    npy_dir = Path("test_frames/images")          # frame_00011_disp.npy 있는 폴더 (필요시 수정)

    out_overlay_dir = Path("outputs/overlays")
    out_depth_dir = Path("outputs/depth_npy")
    out_overlay_dir.mkdir(parents=True, exist_ok=True)
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    # ✅ RGB만: *_disp.png(시각화 결과) 제외
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs += sorted(images_dir.glob(ext))
    imgs = [p for p in imgs if not p.stem.endswith("_disp")]

    if not imgs:
        raise FileNotFoundError(f"No RGB images found in {images_dir.resolve()}")

    depth_stack = []
    stems = []

    for i, img_path in enumerate(imgs):
        stem = img_path.stem  # frame_00011
        npy_path = npy_dir / f"{stem}_disp.npy"

        if not npy_path.exists():
            print(f"[SKIP] missing npy: {npy_path}")
            continue

        rgb = load_rgb(img_path)
        depth = load_depth_npy(npy_path)

        overlay, depth_rs, stats = make_overlay(rgb, depth, alpha=0.55)

        out_png = out_overlay_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(out_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        out_npy = out_depth_dir / f"{stem}_depth_rs.npy"
        np.save(str(out_npy), depth_rs.astype(np.float32))

        depth_stack.append(depth_rs.astype(np.float32))
        stems.append(stem)

        if (len(stems)) % 20 == 0:
            print(f"[{len(stems)}] saved {out_png.name} | min/max/mean={stats['min']:.4f}/{stats['max']:.4f}/{stats['mean']:.4f}")

    if not depth_stack:
        raise RuntimeError("No depth maps were processed (all skipped). Check npy_dir/images_dir.")

    stack = np.stack(depth_stack, axis=0)  # (N,H,W)
    np.save("outputs/all_depth_rs.npy", stack)
    np.save("outputs/all_depth_rs_stems.npy", np.array(stems))

    print("\nDONE")
    print(f"overlays -> {out_overlay_dir.resolve()}")
    print(f"depth npy -> {out_depth_dir.resolve()}")
    print(f"stack -> outputs/all_depth_rs.npy  shape={stack.shape}")


if __name__ == "__main__":
    main()