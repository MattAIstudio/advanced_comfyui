#!/usr/bin/env python3
"""
Week 02 Lab - Script 2: ComfyUI VAE Roundtrip Loss & Telemetry Benchmark
Directly invokes ComfyUI's backend (VAELoader / VAEEncode / VAEDecode) with 0 KSampler steps.

Example (PowerShell on Windows 11):
  python week02_vae_roundtrip_bench.py --comfy-dir D:\ComfyUI --out-dir D:\ComfyUI\week02_bench_out
  python week02_vae_roundtrip_bench.py --comfy-dir D:\ComfyUI --vae ae.safetensors --image .\mattbook_ports.png
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def create_hardware_test_plate(path: Path, size: int = 1024) -> Path:
    """
    Generates a crisp synthetic 'MattBook Pro 16' QA inspection plate with:
    - 1px CNC anodized edge lines
    - Fine USB-C / Thunderbolt laser-etched port labels
    - High-contrast connector pins & micro-checkerboard
    """
    img = Image.new("RGB", (size, size), (28, 32, 38))
    draw = ImageDraw.Draw(img)

    # Brushed anodized chassis slab
    draw.rectangle([64, 180, size - 64, size - 180], fill=(182, 188, 196), outline=(235, 240, 245), width=2)
    # 1px CNC chamfer highlight
    draw.line([(66, 182), (size - 66, 182)], fill=(255, 255, 255), width=1)
    draw.line([(66, size - 182), (size - 66, size - 182)], fill=(90, 96, 104), width=1)

    # Two recessed USB-C ports with internal gold pins
    for idx, x0 in enumerate([220, 560]):
        pw, ph = 180, 72
        y0 = 470
        draw.rounded_rectangle([x0, y0, x0 + pw, y0 + ph], radius=32, fill=(18, 20, 24), outline=(70, 75, 82), width=3)
        # Internal tongue & 12 gold pins
        draw.rectangle([x0 + 26, y0 + 30, x0 + pw - 26, y0 + 42], fill=(45, 48, 54))
        for p in range(12):
            px = x0 + 34 + p * 10
            draw.rectangle([px, y0 + 25, px + 4, y0 + 30], fill=(225, 182, 72))
            draw.rectangle([px, y0 + 42, px + 4, y0 + 47], fill=(225, 182, 72))

    # Crisp micro-text labels & resolution test bars
    try:
        font_lg = ImageFont.truetype("arial.ttf", 26)
        font_sm = ImageFont.truetype("arial.ttf", 16)
        font_xs = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font_lg = font_sm = font_xs = ImageFont.load_default()

    draw.text((220, 425), "USB-C / TB4  40Gbps", fill=(35, 38, 44), font=font_lg)
    draw.text((560, 425), "PD 140W  CHASSIS-QA", fill=(35, 38, 44), font=font_lg)
    draw.text((220, 560), "MATTBOOK PRO 16 | TOLERANCE +/-0.05mm | SN: M26-00492", fill=(45, 48, 56), font=font_sm)
    draw.text((220, 588), "1PX HAIRLINE & MICRO-TEXT CHECK: 0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ", fill=(55, 60, 68), font=font_xs)

    # High-frequency 1px & 2px Nyquist test gratings
    for i in range(0, 128, 2):
        draw.line([(220 + i, 640), (220 + i, 700)], fill=(20, 20, 20), width=1)
    for i in range(0, 128, 4):
        draw.line([(400 + i, 640), (400 + i, 700)], fill=(20, 20, 20), width=2)

    img.save(path, format="PNG")
    return path


def load_image_as_comfy_bhwc(path: Path) -> torch.Tensor:
    pil_img = Image.open(path).convert("RGB")
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    # ComfyUI IMAGE format: [B, H, W, C], float32, [0, 1]
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def compute_psnr(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    mse = torch.mean((img_a - img_b) ** 2).item()
    if mse <= 1e-12:
        return 99.99
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_bchw(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
    """Pure-PyTorch SSIM on [B, C, H, W] float32 tensors in [0, 1]."""
    c = img1.shape[1]
    coords = torch.arange(window_size, dtype=torch.float32, device=img1.device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window = window_2d.expand(c, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=c)
    mu2 = F.conv2d(img2, window, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


def save_l1_heatmap(orig_bhwc: torch.Tensor, dec_bhwc: torch.Tensor, out_path: Path, gain: float = 8.0) -> Dict[str, float]:
    diff = torch.abs(orig_bhwc[0] - dec_bhwc[0]).mean(dim=-1)  # [H, W]
    mean_l1 = float(diff.mean().item())
    max_l1 = float(diff.max().item())

    norm = torch.clamp(diff * gain, 0.0, 1.0).cpu().numpy()
    # Inferno-style false color mapping (black -> blue -> red -> yellow -> white)
    r = np.clip(norm * 2.5, 0.0, 1.0)
    g = np.clip((norm - 0.35) * 2.2, 0.0, 1.0)
    b = np.clip(np.sin(norm * math.pi) * 0.85, 0.0, 1.0)
    rgb = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)
    return {"mean_l1": mean_l1, "max_l1": max_l1, "heatmap_gain": gain}


def save_zoom_comparison(
    orig_bhwc: torch.Tensor,
    dec_bhwc: torch.Tensor,
    out_path: Path,
    crop_box: Tuple[int, int, int, int] = (200, 415, 456, 615),
    zoom: int = 4,
) -> None:
    x1, y1, x2, y2 = crop_box
    orig_np = (torch.clamp(orig_bhwc[0], 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    dec_np = (torch.clamp(dec_bhwc[0], 0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)

    orig_crop = Image.fromarray(orig_np).crop((x1, y1, x2, y2))
    dec_crop = Image.fromarray(dec_np).crop((x1, y1, x2, y2))

    zw, zh = orig_crop.width * zoom, orig_crop.height * zoom
    orig_zoom = orig_crop.resize((zw, zh), resample=Image.NEAREST)
    dec_zoom = dec_crop.resize((zw, zh), resample=Image.NEAREST)

    canvas = Image.new("RGB", (zw * 2 + 24, zh + 48), (15, 18, 22))
    canvas.paste(orig_zoom, (8, 40))
    canvas.paste(dec_zoom, (zw + 16, 40))

    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"ORIGINAL ({zoom * 100}% NEAREST)", fill=(120, 235, 180))
    draw.text((zw + 20, 12), f"VAE ROUNDTRIP ({zoom * 100}% NEAREST)", fill=(255, 175, 95))
    canvas.save(out_path)


def tensor_stats(t: torch.Tensor) -> Dict[str, object]:
    t_f32 = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
        "nbytes_mb": round(t.numel() * t.element_size() / (1024 ** 2), 4),
        "min": round(float(t_f32.min().item()), 6),
        "max": round(float(t_f32.max().item()), 6),
        "has_nan": bool(torch.isnan(t_f32).any().item()),
        "has_inf": bool(torch.isinf(t_f32).any().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 02 ComfyUI VAE Roundtrip Benchmark")
    parser.add_argument("--comfy-dir", type=Path, default=Path(r"D:\ComfyUI"), help="Path to ComfyUI root")
    parser.add_argument("--vae", type=str, nargs="*", default=None, help="VAE filename(s) inside models/vae")
    parser.add_argument("--image", type=Path, default=None, help="Optional input RGB image path")
    parser.add_argument("--out-dir", type=Path, default=Path(r"D:\ComfyUI\week02_bench_out"), help="Output directory")
    args = parser.parse_args()

    comfy_dir = args.comfy_dir.resolve()
    if not comfy_dir.exists():
        print(f"[ERROR] ComfyUI directory not found: {comfy_dir}")
        return 1

    # Attach ComfyUI backend modules
    sys.path.insert(0, str(comfy_dir))
    import folder_paths  # type: ignore
    import nodes         # type: ignore

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    available_vaes: List[str] = folder_paths.get_filename_list("vae")
    if not available_vaes:
        print(f"[ERROR] No VAE files found in {comfy_dir / 'models' / 'vae'}.")
        return 1

    target_vaes = args.vae if args.vae else available_vaes[:2]
    print(f"Found {len(available_vaes)} VAEs. Benchmarking: {target_vaes}")

    if args.image is not None and args.image.exists():
        input_path = args.image.resolve()
    else:
        input_path = out_dir / "mattbook_pro_16_test_plate.png"
        create_hardware_test_plate(input_path, size=1024)
        print(f"Generated synthetic QA plate: {input_path}")

    pixels = load_image_as_comfy_bhwc(input_path)
    orig_stats = tensor_stats(pixels)
    print(f"\nInput IMAGE Tensor: {orig_stats}")

    vae_loader = nodes.VAELoader()
    vae_encoder = nodes.VAEEncode()
    vae_decoder = nodes.VAEDecode()

    manifest = {
        "comfy_dir": str(comfy_dir),
        "pytorch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "input_image": str(input_path),
        "input_tensor": orig_stats,
        "runs": [],
    }

    for vae_name in target_vaes:
        if vae_name not in available_vaes:
            print(f"[SKIP] '{vae_name}' not found in models/vae.")
            continue

        print(f"\n{'-' * 74}\nBenchmarking VAE: {vae_name}\n{'-' * 74}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        (vae_obj,) = vae_loader.load_vae(vae_name)
        load_ms = (time.perf_counter() - t0) * 1000.0

        with torch.no_grad():
            t1 = time.perf_counter()
            (latent_dict,) = vae_encoder.encode(vae_obj, pixels)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            (decoded_pixels,) = vae_decoder.decode(vae_obj, latent_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - t2) * 1000.0

        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)
            if torch.cuda.is_available()
            else 0.0
        )

        latent_tensor = latent_dict["samples"]
        latent_stats = tensor_stats(latent_tensor)
        decoded_stats = tensor_stats(decoded_pixels)
        compression_ratio = round(pixels.numel() / latent_tensor.numel(), 2)

        orig_cpu = pixels.detach().cpu().float()
        dec_cpu = torch.clamp(decoded_pixels.detach().cpu().float(), 0.0, 1.0)
        psnr_db = round(compute_psnr(orig_cpu, dec_cpu), 3)
        ssim_val = round(
            compute_ssim_bchw(orig_cpu.movedim(-1, 1), dec_cpu.movedim(-1, 1)), 5
        )

        safe_stem = Path(vae_name).stem.replace(" ", "_")
        dec_out = out_dir / f"{safe_stem}_decoded.png"
        heatmap_out = out_dir / f"{safe_stem}_heatmap.png"
        zoom_out = out_dir / f"{safe_stem}_crops_400pct.png"

        dec_uint8 = (dec_cpu[0].numpy() * 255.0).astype(np.uint8)
        Image.fromarray(dec_uint8).save(dec_out)
        err_metrics = save_l1_heatmap(orig_cpu, dec_cpu, heatmap_out, gain=8.0)
        save_zoom_comparison(orig_cpu, dec_cpu, zoom_out, zoom=4)

        run_record = {
            "vae_name": vae_name,
            "load_ms": round(load_ms, 2),
            "encode_ms": round(enc_ms, 2),
            "decode_ms": round(dec_ms, 2),
            "peak_vram_mb": peak_vram_mb,
            "compression_ratio": f"{compression_ratio}x",
            "latent_tensor": latent_stats,
            "decoded_tensor": decoded_stats,
            "psnr_db": psnr_db,
            "ssim": ssim_val,
            **err_metrics,
            "artifacts": {
                "decoded": str(dec_out),
                "heatmap": str(heatmap_out),
                "zoom_crops": str(zoom_out),
            },
        }
        manifest["runs"].append(run_record)

        print(f"  LATENT shape     : {latent_stats['shape']} ({compression_ratio}x compression)")
        print(f"  DECODED shape    : {decoded_stats['shape']} | dtype={decoded_stats['dtype']}")
        print(f"  Timing           : load={load_ms:.1f}ms | enc={enc_ms:.1f}ms | dec={dec_ms:.1f}ms | VRAM={peak_vram_mb}MB")
        print(f"  Quality Metrics  : PSNR={psnr_db} dB | SSIM={ssim_val} | Mean L1={err_metrics['mean_l1']:.5f}")

    manifest_path = out_dir / "week02_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved complete telemetry manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())