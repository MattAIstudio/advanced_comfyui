#!/usr/bin/env python3
"""
Week 02 Lab - Script 1: ComfyUI Tensor Layout, Device & Dtype Bug Reproducer
Environment: Windows 11 / RTX 5090 / PyTorch
"""

import sys
import torch
import torch.nn as nn


def inspect_tensor(name: str, t: torch.Tensor) -> None:
    print(
        f"  [{name:<18}] shape={str(tuple(t.shape)):<20} "
        f"dtype={str(t.dtype):<14} device={str(t.device):<8} "
        f"contig={str(t.is_contiguous()):<5} stride={t.stride()}"
    )


def run_case(title: str, fn) -> None:
    print(f"\n{'=' * 78}\nCASE: {title}\n{'=' * 78}")
    fn()


def case_1_bhwc_vs_bchw() -> None:
    # ComfyUI IMAGE: [B, H, W, C], float32, [0, 1]
    comfy_image = torch.rand(1, 1024, 1024, 3, dtype=torch.float32)
    conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

    inspect_tensor("ComfyUI IMAGE", comfy_image)

    print("-> Reproducing bug: feeding BHWC directly into nn.Conv2d...")
    try:
        _ = conv(comfy_image)
    except RuntimeError as exc:
        print(f"   [EXPECTED CRASH] {exc}")

    print("-> Fix: movedim(-1, 1) (BHWC -> BCHW) before Conv2d, movedim(1, -1) after...")
    bchw = comfy_image.movedim(-1, 1)  # Equivalent to .permute(0, 3, 1, 2)
    inspect_tensor("BCHW input", bchw)
    out_bchw = conv(bchw)
    out_bhwc = out_bchw.movedim(1, -1)
    inspect_tensor("Fixed BHWC output", out_bhwc)


def case_2_mask_shape_and_broadcast() -> None:
    # ComfyUI IMAGE: [1, 512, 512, 3], MASK often arrives as [H, W] or [B, H, W]
    image = torch.ones(1, 512, 512, 3, dtype=torch.float32)
    squeezed_mask = torch.rand(512, 512, dtype=torch.float32)

    inspect_tensor("IMAGE", image)
    inspect_tensor("Squeezed MASK", squeezed_mask)

    print("-> Reproducing bug: multiplying [1, 512, 512, 3] by [512, 512]...")
    try:
        _ = image * squeezed_mask
    except RuntimeError as exc:
        print(f"   [EXPECTED CRASH] {exc}")

    print("-> Fix: normalize MASK shape cleanly to [B, H, W, 1] for BHWC broadcasting...")
    mask = squeezed_mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)        # [1, H, W]
    if mask.ndim == 3:
        mask = mask.unsqueeze(-1)       # [B, H, W, 1]
    inspect_tensor("Aligned MASK", mask)
    masked_image = image * mask
    inspect_tensor("Masked IMAGE", masked_image)


def case_3_contiguous_after_permute() -> None:
    image = torch.rand(1, 256, 256, 3, dtype=torch.float32)
    permuted = image.permute(0, 3, 1, 2)
    inspect_tensor("Permuted BCHW", permuted)

    print("-> Reproducing bug: calling .view() on non-contiguous permuted tensor...")
    try:
        _ = permuted.view(1, 3, -1)
    except RuntimeError as exc:
        print(f"   [EXPECTED CRASH] {exc}")

    print("-> Fix: call .contiguous().view(...) or .reshape(...)...")
    fixed = permuted.contiguous().view(1, 3, -1)
    inspect_tensor("Contiguous view", fixed)


def case_4_device_and_dtype_mismatch() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Active accelerator: {device}")

    # Model layer sitting in bfloat16 on GPU (typical ComfyUI FLUX/modern VAE weight)
    conv_bf16 = nn.Conv2d(3, 4, kernel_size=3, padding=1).to(
        device=device, dtype=torch.bfloat16
    )
    # User-created tensor defaults to CPU + float32
    x_cpu_f32 = torch.rand(1, 3, 128, 128, dtype=torch.float32, device="cpu")
    inspect_tensor("conv weight", conv_bf16.weight)
    inspect_tensor("x_cpu_f32", x_cpu_f32)

    print("-> Reproducing bug: feeding CPU/float32 tensor into target device/bf16 Conv2d...")
    try:
        _ = conv_bf16(x_cpu_f32)
    except RuntimeError as exc:
        print(f"   [EXPECTED CRASH] {exc}")

    print("-> Fix: explicit alignment to weight device & dtype + safe NumPy export...")
    x_aligned = x_cpu_f32.to(device=conv_bf16.weight.device, dtype=conv_bf16.weight.dtype)
    out = conv_bf16(x_aligned)
    inspect_tensor("Aligned output", out)

    # Attempting out.detach().cpu().numpy() on bfloat16 fails in NumPy!
    try:
        _ = out.detach().cpu().numpy()
    except TypeError as exc:
        print(f"   [NUMPY BF16 CRASH] {exc}")

    np_ready = out.detach().cpu().float().numpy()
    print(f"   [FIXED NUMPY] shape={np_ready.shape}, dtype={np_ready.dtype}")


def main() -> int:
    print(f"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    case_1_bhwc_vs_bchw()
    case_2_mask_shape_and_broadcast()
    case_3_contiguous_after_permute()
    case_4_device_and_dtype_mismatch()
    print("\nAll diagnostic cases completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())