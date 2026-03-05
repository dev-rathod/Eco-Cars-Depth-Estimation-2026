from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIG (여기만 조절하면 됨)
# =========================
IMAGES_DIR = Path("test_frames/images")   # frame_00000.png 있는 곳
DISP_DIR   = Path("test_frames/images")   # frame_00000_disp.npy 있는 곳
OUT_DIR    = Path("outputs/sparse_overlays")

USE_INVERSE = True        # monodepth2 disp -> depth-like 로 바꿀지 (보통 True 권장)
EPS = 1e-6

# dense를 "sparse처럼" 보이게 다운샘플링 (점 개수 줄이기)
STRIDE = 1                # 1=모든 픽셀(너무 빽빽), 2~4 추천

# ROI 마스킹 (원하는 영역만 보여주기)
ROI_Y_MIN_FRAC = 0.25     # 위에서부터 몇 % 아래부터 보여줄지 (너무 끊기면 0.10~0.20으로 낮춰)
ROI_Y_MAX_FRAC = 0.98     # 아래쪽 끝 (거의 끝까지)

# "도로 제외" (하단 몇 %는 버리기) 0이면 끔
EXCLUDE_BOTTOM_FRAC = 0.00  # 예: 0.15면 맨 아래 15%는 안 그림

# "멀면 안봄" (meters처럼 보이게 depth-like 만들었을 때 cutoff)
# None이면 cutoff 안 함
DEPTH_MAX_M = None        # 예: 30.0, 50.0 등
DEPTH_MIN_M = None

# 시각화 스타일
POINT_SIZE = 1.0
ALPHA = 0.9
CMAP = "jet"            

# vmin/vmax 자동(퍼센타일) — 컬러바 스케일 안정화
VMIN_PCTL = 1
VMAX_PCTL = 99

FIGSIZE = (10, 6)
DPI = 150


# =========================
# Utils
# =========================
def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"RGB not found: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_disp(path: Path) -> np.ndarray:
    d = np.load(str(path)).astype(np.float32)
    d = np.squeeze(d)  # (1,1,H,W)->(H,W) or already (H,W)
    if d.ndim != 2:
        raise ValueError(f"disp must be 2D, got {d.shape} from {path}")
    return d


def disp_to_depth_like(disp: np.ndarray, use_inverse: bool = True) -> np.ndarray:
    # monodepth2 disp: 보통 "클수록 가까움"
    # depth-like: "클수록 멀어짐" 으로 만들려면 inverse가 직관적임
    if not use_inverse:
        return disp
    return 1.0 / (disp + EPS)


def plot_sparse_overlay(
    rgb: np.ndarray,
    depth_like: np.ndarray,
    out_path: Path,
    title: str = "Sparse Depth Overlay on RGB (valid pixels only)",
    depth_unit: str = "Depth (meters)",
):
    H_img, W_img = rgb.shape[:2]
    H_d, W_d = depth_like.shape

    # ---- valid mask ----
    valid = np.isfinite(depth_like) & (depth_like > 0)

    # ROI y-range on depth-map coords
    y0 = int(ROI_Y_MIN_FRAC * H_d)
    y1 = int(ROI_Y_MAX_FRAC * H_d)
    valid[:y0, :] = False
    valid[y1:, :] = False

    # exclude bottom band (road removal-ish)
    if EXCLUDE_BOTTOM_FRAC and EXCLUDE_BOTTOM_FRAC > 0:
        cut = int((1.0 - EXCLUDE_BOTTOM_FRAC) * H_d)
        valid[cut:, :] = False

    # depth cutoff (far ignore)
    if DEPTH_MIN_M is not None:
        valid &= (depth_like >= float(DEPTH_MIN_M))
    if DEPTH_MAX_M is not None:
        valid &= (depth_like <= float(DEPTH_MAX_M))

    # stride downsample (sparse look)
    if STRIDE and STRIDE > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::STRIDE, ::STRIDE] = True
        valid &= stride_mask

    ys, xs = np.where(valid)
    vals = depth_like[ys, xs].astype(np.float32)

    if vals.size == 0:
        raise RuntimeError("No valid depth pixels after masking. ROI/DEPTH_MAX 너무 빡셈.")

    # depth coords -> RGB coords (resize depth 자체는 안 함)
    scale_y = H_img / H_d
    scale_x = W_img / W_d
    ys_rgb = ys.astype(np.float32) * scale_y
    xs_rgb = xs.astype(np.float32) * scale_x

    # vmin/vmax robust
    vmin = float(np.percentile(vals, VMIN_PCTL))
    vmax = float(np.percentile(vals, VMAX_PCTL))
    if vmax <= vmin:
        vmin, vmax = float(vals.min()), float(vals.max())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.imshow(rgb)

    sc = ax.scatter(
        xs_rgb,
        ys_rgb,
        c=vals,
        s=POINT_SIZE,
        cmap=CMAP,
        alpha=ALPHA,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )

    ax.set_title(title)
    ax.axis("off")

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(depth_unit)

    # ----- stats -----
    valid_pixels = int(vals.size)
    total_pixels = int(depth_like.size)
    density = valid_pixels / total_pixels

    stats_text = (
        f"Depth shape: {depth_like.shape}\n"
        f"Valid pixels: {valid_pixels}\n"
        f"Total pixels: {total_pixels}\n"
        f"Density: {density}\n"
        f"Min depth (valid): {float(vals.min())}\n"
        f"Max depth (valid): {float(vals.max())}\n"
        f"Mean depth (valid): {float(vals.mean())}"
    )

    plt.figtext(
        0.02, 0.02,
        stats_text,
        fontsize=9,
        ha="left",
        va="bottom"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # RGB 파일들: *_disp.png 제외하고 frame_XXXXX.* 만
    rgb_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        rgb_files += sorted(IMAGES_DIR.glob(ext))
    rgb_files = [p for p in rgb_files if not p.stem.endswith("_disp")]

    if not rgb_files:
        raise FileNotFoundError(f"No RGB found in {IMAGES_DIR.resolve()}")

    n_done = 0
    for rgb_path in rgb_files:
        stem = rgb_path.stem  # frame_00000
        disp_path = DISP_DIR / f"{stem}_disp.npy"
        if not disp_path.exists():
            print(f"[SKIP] missing {disp_path.name}")
            continue

        rgb = load_rgb(rgb_path)
        disp = load_disp(disp_path)
        depth_like = disp_to_depth_like(disp, use_inverse=USE_INVERSE)

        out_path = OUT_DIR / f"{stem}_sparse_overlay.png"
        plot_sparse_overlay(rgb, depth_like, out_path)

        n_done += 1
        if n_done % 20 == 0:
            print(f"[{n_done}] saved {out_path.name}")

    print("\nDONE")
    print(f"saved -> {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()