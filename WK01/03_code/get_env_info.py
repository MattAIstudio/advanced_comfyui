"""Collect reproducibility information for a ComfyUI workflow.

The running ComfyUI server is the authoritative source for its version and
launch arguments. Hardware information is collected locally as a fallback,
so this script is still useful when the server is not running.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "env_info.json"
DEFAULT_SERVER = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
UNKNOWN = "Unknown"

# File-valued inputs used by ComfyUI's model loader nodes. Limiting extraction
# to these keys avoids treating sampler_name and filename_prefix as model files.
MODEL_INPUT_KEYS = {
    "ckpt_name", "clip_name", "clip_name1", "clip_name2", "clip_name3",
    "clip_vision_name", "control_net_name", "diffusion_model_name",
    "gligen_name", "lora_name", "model_name", "style_model_name",
    "unet_name", "upscale_model_name", "vae_name",
}


def _find_default_workflow() -> Path:
    """Find workflow_api.json without assuming a fixed project layout."""
    candidates = (
        SCRIPT_DIR / "workflow_api.json",
        SCRIPT_DIR.parent / "workflow_api.json",
        Path.cwd() / "workflow_api.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


DEFAULT_WORKFLOW = _find_default_workflow()


def _system_stats(server: str, timeout: float) -> dict[str, Any]:
    """Return ComfyUI's public system statistics, or an empty mapping."""
    try:
        with urlopen(
            f"{server.rstrip('/')}/system_stats", timeout=timeout
        ) as response:
            value = json.load(response)
        return value if isinstance(value, dict) else {}
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return {}


def _load_workflow(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
        digest = hashlib.sha256(raw).hexdigest()
        return (value if isinstance(value, dict) else {}, digest)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, UNKNOWN


def _model_files(workflow: dict[str, Any]) -> list[str]:
    files: set[str] = set()
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if key in MODEL_INPUT_KEYS and isinstance(value, str) and value:
                files.add(value)
    return sorted(files, key=str.casefold)


def _windows_ram_bytes() -> int | None:
    if platform.system() != "Windows":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        # Unlike GlobalMemoryStatusEx, this reports installed memory and does
        # not subtract hardware-reserved regions (96 GB should remain 96 GB).
        installed_kib = ctypes.c_ulonglong()
        if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(
            ctypes.byref(installed_kib)
        ):
            return installed_kib.value * 1024

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys if ok else None
    except (AttributeError, OSError):
        return None


def _system_ram_gb() -> int | float | str:
    total = _windows_ram_bytes()
    if total is None:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            return UNKNOWN
    return round(total / (1024**3))


def _os_name() -> str:
    if platform.system() == "Windows":
        try:
            return (
                "Windows 11"
                if sys.getwindowsversion().build >= 22000
                else "Windows 10"
            )
        except (AttributeError, OSError):
            pass
    return f"{platform.system()} {platform.release()}".strip() or UNKNOWN


def _torch_info() -> tuple[str, str, str, int | None]:
    try:
        import torch
    except (ImportError, OSError):
        return UNKNOWN, UNKNOWN, UNKNOWN, None

    pytorch = str(torch.__version__)
    cuda_runtime = str(torch.version.cuda or UNKNOWN)
    gpu_name, vram = UNKNOWN, None
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            gpu_name, vram = props.name, int(props.total_memory)
        except (RuntimeError, AssertionError):
            pass
        # torch.version.cuda is the runtime bundled with the PyTorch build.
        # Query cudart when possible to report the actually loaded runtime.
        try:
            version = int(torch.cuda.cudart().cudaRuntimeGetVersion())
            cuda_runtime = f"{version // 1000}.{(version % 1000) // 10}"
        except (AttributeError, RuntimeError, ValueError):
            pass
    return pytorch, cuda_runtime, gpu_name, vram


def _nvidia_smi() -> tuple[str, str, int | None]:
    command = [
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits", "--id=0",
    ]
    try:
        line = subprocess.check_output(
            command, stderr=subprocess.DEVNULL, text=True, timeout=5
        ).splitlines()[0]
        name, memory_mib, driver = (part.strip() for part in line.split(",", 2))
        return name, driver, round(float(memory_mib) * 1024**2)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return UNKNOWN, UNKNOWN, None


def _gpu_label(name: str, vram_bytes: int | None) -> str:
    # Server names can look like "cuda:0 NVIDIA GeForce RTX 5090 : cudaMallocAsync".
    name = re.sub(r"^cuda:\d+\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*:\s*(cudaMallocAsync|native)\s*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^NVIDIA\s+(GeForce\s+)?", "", name, flags=re.IGNORECASE).strip()
    if not name or name == UNKNOWN:
        return UNKNOWN
    return f"{name} {round(vram_bytes / (1024**3))}GB" if vram_bytes else name


def _server_device(stats: dict[str, Any]) -> tuple[str, int | None]:
    devices = stats.get("devices", [])
    if not isinstance(devices, list):
        return UNKNOWN, None
    for device in devices:
        if isinstance(device, dict) and str(device.get("type", "")).lower() != "cpu":
            name = str(device.get("name") or UNKNOWN)
            try:
                vram = int(device.get("vram_total"))
            except (TypeError, ValueError):
                vram = None
            return name, vram
    return UNKNOWN, None


def _launch_args(system: dict[str, Any]) -> list[str]:
    argv = system.get("argv")
    if isinstance(argv, list):
        # sys.argv[0] is main.py, not a launch argument.
        return [str(value) for value in argv[1:]]
    configured = os.environ.get("COMFYUI_LAUNCH_ARGS", "")
    if not configured:
        return []
    return shlex.split(configured, posix=platform.system() != "Windows")


def _attention_backend(args: list[str]) -> str:
    mappings = (
        ("--use-sage-attention", "SageAttention"),
        ("--use-flash-attention", "FlashAttention"),
        ("--use-pytorch-cross-attention", "PyTorch SDPA"),
        ("--use-split-cross-attention", "split cross-attention"),
        ("--use-quad-cross-attention", "sub-quadratic cross-attention"),
    )
    for flag, label in mappings:
        if flag in args:
            return label
    if "--disable-xformers" in args:
        return "PyTorch SDPA (xFormers disabled)"
    return "auto"


def _precision(args: list[str], workflow: dict[str, Any]) -> str:
    # Ordered from whole-program overrides to component-specific flags.
    flags = (
        ("--force-fp32", "fp32"), ("--force-fp16", "fp16"),
        ("--bf16-unet", "bf16 (UNet)"), ("--fp16-unet", "fp16 (UNet)"),
        ("--fp8_e4m3fn-unet", "fp8_e4m3fn (UNet)"),
        ("--fp8_e5m2-unet", "fp8_e5m2 (UNet)"),
    )
    for flag, label in flags:
        if flag in args:
            return label

    values: set[str] = set()
    for node in workflow.values():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if not isinstance(inputs, dict):
            continue
        for key in ("weight_dtype", "dtype"):
            value = inputs.get(key)
            if (
                isinstance(value, str)
                and value.lower() not in {"", "default", "auto"}
            ):
                values.add(value)
    if values:
        return ", ".join(sorted(values))

    # With dtype=default ComfyUI preserves the checkpoint's native weight
    # type. Many distributed model files state that type in their filename.
    filename_types: set[str] = set()
    for filename in _model_files(workflow):
        normalized = filename.lower().replace("-", "_")
        for marker, label in (
            ("bf16", "bf16"), ("fp16", "fp16"),
            ("fp8_e4m3fn", "fp8_e4m3fn"), ("fp8_e5m2", "fp8_e5m2"),
        ):
            if marker in normalized:
                filename_types.add(label)
    return ", ".join(sorted(filename_types)) if filename_types else "auto"


def get_env_info(
    workflow_path: str | Path = DEFAULT_WORKFLOW,
    server: str = DEFAULT_SERVER,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Collect the stable environment manifest requested by the benchmark."""
    resolved_workflow = Path(workflow_path).expanduser().resolve()
    server = server.rstrip("/")
    workflow, workflow_sha256 = _load_workflow(resolved_workflow)
    stats = _system_stats(server, timeout)
    system = stats.get("system", {}) if isinstance(stats.get("system"), dict) else {}
    launch_args = _launch_args(system)

    local_pytorch, cuda_runtime, local_gpu, local_vram = _torch_info()
    smi_gpu, driver, smi_vram = _nvidia_smi()
    server_gpu, server_vram = _server_device(stats)
    gpu_name = server_gpu if server_gpu != UNKNOWN else local_gpu
    gpu_vram = server_vram or local_vram
    if gpu_name == UNKNOWN:
        gpu_name, gpu_vram = smi_gpu, smi_vram

    pytorch = str(system.get("pytorch_version") or local_pytorch)
    # If the collector and server use different Python environments, prefer
    # the CUDA build encoded in the server's PyTorch version (e.g. +cu128).
    cuda_tag = re.search(r"\+cu(\d{2,3})", pytorch)
    if cuda_tag:
        digits = cuda_tag.group(1)
        cuda_runtime = f"{int(digits[:-1])}.{digits[-1]}"

    return {
        "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "comfyui_connected": bool(stats),
        "comfyui_server": server,
        "workflow_path": str(resolved_workflow),
        "workflow_found": resolved_workflow.is_file(),
        "gpu": _gpu_label(gpu_name, gpu_vram),
        "system_ram_gb": _system_ram_gb(),
        "os": _os_name(),
        "python_version": platform.python_version(),
        "driver": driver,
        "pytorch": pytorch,
        "cuda_runtime": cuda_runtime,
        "comfyui_version": str(system.get("comfyui_version") or UNKNOWN),
        "workflow_sha256": workflow_sha256,
        "model_files": _model_files(workflow),
        "launch_args": launch_args,
        "attention_backend": _attention_backend(launch_args),
        "precision": _precision(launch_args, workflow),
    }


def _save_report(report: dict[str, Any], output_path: Path) -> Path:
    """Write formatted JSON atomically and return its absolute path."""
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    print("Collecting environment information...")
    report = get_env_info(args.workflow, args.server, args.timeout)
    destination = _save_report(report, args.output)

    if report["comfyui_connected"]:
        print("ComfyUI server is connected.")
    else:
        print("ComfyUI server is not running. Local information was collected.")
    if not report["workflow_found"]:
        print("Warning: Workflow file was not found.")
    print("Collection complete.")
    print(f"Saved report: {destination}")


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(f"Collection failed: {exc}")
        raise SystemExit(1)
