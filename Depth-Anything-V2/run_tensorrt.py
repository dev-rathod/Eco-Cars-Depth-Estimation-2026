import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F

from depth_anything_v2.dpt import DepthAnythingV2


def build_or_load_engine(depth_anything, encoder, input_size, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    engine_path = os.path.join(cache_dir, f'depth_anything_v2_{encoder}_fp16_{input_size}.pt2')

    try:
        import torch_tensorrt
    except ImportError:
        print('ERROR: torch_tensorrt is not installed.')
        print('  pip install torch-tensorrt --index-url https://download.pytorch.org/whl/cu130')
        sys.exit(1)

    if os.path.exists(engine_path):
        print(f'Loading cached TensorRT engine: {engine_path}')
        return torch_tensorrt.load(engine_path).module()

    print(f'Compiling TensorRT engine for {encoder} at {input_size}x{input_size}...')
    print('This runs once and takes ~2-10 minutes depending on encoder size.')
    torch.cuda.empty_cache()

    # Static shape: DINOv2 ViT bakes num_patches² as a constant at trace time,
    # so dynamic H/W ranges cause ConstraintViolationError.
    inputs = [
        torch_tensorrt.Input(
            shape=[1, 3, input_size, input_size],
            dtype=torch.float16,
        )
    ]

    model_trt = torch_tensorrt.compile(
        depth_anything,
        inputs=inputs,
        enabled_precisions={torch.float16},
        use_explicit_typing=False,
        workspace_size=1 * 1024 ** 3,
        truncate_long_and_double=True,
    )

    torch_tensorrt.save(model_trt, engine_path, inputs=inputs)
    print(f'Engine cached to: {engine_path}')
    return model_trt


def infer(model, depth_anything, raw_image, input_size):
    h, w = raw_image.shape[:2]
    # Force square so the static TRT engine (compiled at input_size×input_size) gets exact dims.
    # Squashing to square is fine for depth estimation; we rescale back to original size after.
    square = cv2.resize(raw_image, (input_size, input_size))
    image, _ = depth_anything.image2tensor(square, input_size)
    image = image.half()

    with torch.no_grad():
        depth = model(image)

    depth = F.interpolate(
        depth[:, None], (h, w), mode='bilinear', align_corners=True
    )[0, 0]

    return depth.cpu().float().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 — TensorRT FP16')

    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitb',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='vitb recommended for TRT on 8 GB VRAM; vitl needs ~20 GB RAM to compile')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true',
                        help='Save only the depth prediction, not side-by-side')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true',
                        help='Also save grayscale depth alongside the heatmap')
    parser.add_argument('--trt-cache', type=str, default='./trt_cache',
                        help='Directory where compiled TRT engines are stored')
    parser.add_argument('--rebuild', action='store_true',
                        help='Delete cached engine and recompile from scratch')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: CUDA GPU not found. TensorRT requires an NVIDIA GPU.')
        sys.exit(1)

    DEVICE = 'cuda'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,   96,   192,  384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96,   192,  384,  768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256,  512,  1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    print(f'Loading {args.encoder} weights...')
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(
        torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu')
    )
    depth_anything = depth_anything.to(DEVICE).eval().half()

    if args.rebuild:
        stale = os.path.join(
            args.trt_cache,
            f'depth_anything_v2_{args.encoder}_fp16_{args.input_size}.pt2'
        )
        if os.path.exists(stale):
            os.remove(stale)
            print(f'Deleted stale engine: {stale}')

    model_trt = build_or_load_engine(
        depth_anything, args.encoder, args.input_size, args.trt_cache
    )

    print('Warming up TensorRT engine (3 passes)...')
    dummy = torch.randn(1, 3, args.input_size, args.input_size, device=DEVICE, dtype=torch.float16)
    with torch.no_grad():
        for i in range(3):
            t = time.perf_counter()
            model_trt(dummy)
            torch.cuda.synchronize()
            print(f'  warmup {i+1}/3  {(time.perf_counter()-t)*1000:.0f} ms')
    print('Warmup done.\n')

    # Collect input files
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    elif os.path.isdir(args.img_path):
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    else:
        print(f'ERROR: --img-path {args.img_path!r} is not a file or directory.')
        sys.exit(1)

    filenames = [
        f for f in filenames
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    print(f'Found {len(filenames)} image(s)\n')

    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    total_inference_ms = 0.0
    total_image_store_time = 0.0
    processed = 0

    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f'[{k+1}/{len(filenames)}] Skipping (unreadable): {filename}')
            continue

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        depth = infer(model_trt, depth_anything, raw_image, args.input_size)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        total_inference_ms += elapsed_ms
        processed += 1

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_uint8 = depth_norm.astype(np.uint8)

        stem = os.path.splitext(os.path.basename(filename))[0]
        heatmap = (cmap(depth_uint8)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, stem + '_heatmap.png'), heatmap)
            if args.grayscale:
                gray = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
                cv2.imwrite(os.path.join(args.outdir, stem + '_gray.png'), gray)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            cv2.imwrite(
                os.path.join(args.outdir, stem + '_heatmap.png'),
                cv2.hconcat([raw_image, split_region, heatmap]),
            )
            if args.grayscale:
                gray = np.repeat(depth_uint8[..., np.newaxis], 3, axis=-1)
                cv2.imwrite(
                    os.path.join(args.outdir, stem + '_gray.png'),
                    cv2.hconcat([raw_image, split_region, gray]),
                )

        # print(f'[{k+1}/{len(filenames)}] {os.path.basename(filename)} | Model Inference Speed —  {elapsed_ms:.1f} ms')

    if processed > 0:
        avg_inference_ms = total_inference_ms / processed
        avg_image_store_ms = total_image_store_time / processed
        print(f'\nFinished {processed} image(s).')
        print(f"Model type : {args.encoder}")
        print(f'Avg inference time : {avg_inference_ms:.1f} ms/image')
        print(f'Effective Inference FPS      : {1000/avg_inference_ms:.1f} FPS')
        print(f'Output saved to    : {args.outdir}')
