import torch


def configure_inference_runtime(device, use_tf32=True):
    device_type = torch.device(device).type
    if device_type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def maybe_compile_model(model, enable_compile=False, mode="reduce-overhead"):
    if not enable_compile:
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch 2.0 or newer.")

    return torch.compile(model, mode=mode)
