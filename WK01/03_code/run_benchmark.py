import argparse
import hashlib
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import websocket


DEFAULT_SERVER_ADDRESS = "127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKFLOW_PATH = (
    SCRIPT_DIR / "image_qwen_image_2_1_t2i_workflow_api.json"
)
DEFAULT_MANIFEST_PATH = SCRIPT_DIR.parent / "04_output" / "manifest.json"
GIB = 1024**3


def utc_now():
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def request_json(server_address, path, data=None, timeout=10):
    """Send an HTTP request to ComfyUI and decode its JSON response."""
    request = urllib.request.Request(
        f"http://{server_address}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def queue_prompt(server_address, prompt):
    """Submit a workflow to the ComfyUI execution queue."""
    payload = {"prompt": prompt, "client_id": CLIENT_ID}
    return request_json(
        server_address,
        "/prompt",
        json.dumps(payload).encode("utf-8"),
    )


def get_system_stats(server_address):
    """Read system, software, and GPU information from ComfyUI."""
    return request_json(server_address, "/system_stats", timeout=3)


def cuda_version_from_pytorch(pytorch_version):
    """Extract a readable CUDA version from a PyTorch build string."""
    match = re.search(r"\+cu(\d+)", pytorch_version)
    if not match:
        return "N/A"

    digits = match.group(1)
    return f"{digits[:-1]}.{digits[-1]}"


def get_gpu_driver_version():
    """Return the NVIDIA driver version reported by nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return versions[0] if versions else "N/A"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "N/A"


def detect_cache_mode(server_arguments):
    """Identify the ComfyUI cache mode from its startup arguments."""
    if "--cache-none" in server_arguments:
        return "none"
    if "--cache-classic" in server_arguments:
        return "classic"
    if "--cache-lru" in server_arguments:
        index = server_arguments.index("--cache-lru")
        value = server_arguments[index + 1] if index + 1 < len(server_arguments) else ""
        return f"lru:{value}"
    return "default"


def monitor_vram(server_address, stop_event, state, poll_interval):
    """Poll ComfyUI in the background and track peak VRAM usage."""
    while not stop_event.is_set():
        try:
            stats = get_system_stats(server_address)
            device = stats["devices"][0]
            used_bytes = device["vram_total"] - device["vram_free"]
            state["peak_used_bytes"] = max(state["peak_used_bytes"], used_bytes)
            state["samples"] += 1
        except (OSError, KeyError, IndexError, json.JSONDecodeError):
            state["failed_samples"] += 1
        stop_event.wait(poll_interval)


def wait_for_prompt(websocket_client, prompt_id, request_started):
    """Wait for one prompt to finish and return its timing details."""
    execution_started = None

    while True:
        output = websocket_client.recv()
        if not isinstance(output, str):
            continue

        message = json.loads(output)
        message_type = message.get("type")
        data = message.get("data", {})
        if data.get("prompt_id") != prompt_id:
            continue

        if message_type == "execution_start":
            execution_started = time.perf_counter()
        elif message_type == "execution_error":
            error = data.get("exception_message", "Unknown ComfyUI execution error")
            raise RuntimeError(f"Prompt {prompt_id} failed: {error}")
        elif message_type == "execution_interrupted":
            raise RuntimeError(f"Prompt {prompt_id} was interrupted.")
        elif message_type == "executing" and data.get("node") is None:
            completed = time.perf_counter()
            queue_seconds = (
                execution_started - request_started
                if execution_started is not None
                else None
            )
            execution_seconds = (
                completed - execution_started
                if execution_started is not None
                else None
            )
            return completed, queue_seconds, execution_seconds


def image_dimensions(image_bytes):
    """Read width and height directly from PNG image bytes."""
    png_signature = b"\x89PNG\r\n\x1a\n"
    if image_bytes.startswith(png_signature) and len(image_bytes) >= 24:
        return {
            "width": int.from_bytes(image_bytes[16:20], "big"),
            "height": int.from_bytes(image_bytes[20:24], "big"),
        }
    return {}


def collect_outputs(
    server_address,
    prompt_id,
    local_output_directory,
    local_filename_prefix,
):
    """Download generated images locally and describe each saved file."""
    history = {}
    for _ in range(10):
        history = request_json(
            server_address,
            f"/history/{urllib.parse.quote(prompt_id)}",
            timeout=3,
        )
        if prompt_id in history:
            break
        time.sleep(0.1)

    prompt_history = history.get(prompt_id, {})
    collected = []
    local_output_directory.mkdir(parents=True, exist_ok=True)
    for node_id, node_output in prompt_history.get("outputs", {}).items():
        for image in node_output.get("images", []):
            query = urllib.parse.urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            with urllib.request.urlopen(
                f"http://{server_address}/view?{query}",
                timeout=30,
            ) as response:
                image_bytes = response.read()

            output_index = len(collected) + 1
            source_name = Path(image["filename"]).name
            local_filename = (
                f"{local_filename_prefix}_{output_index:02d}_{source_name}"
            )
            local_path = local_output_directory / local_filename
            with open(local_path, "wb") as file:
                file.write(image_bytes)

            collected.append(
                {
                    "node_id": node_id,
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                    "local_path": str(local_path),
                    "size_bytes": len(image_bytes),
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    **image_dimensions(image_bytes),
                }
            )
    return collected


def run_prompt(
    server_address,
    websocket_client,
    prompt_workflow,
    samplers,
    seed,
    phase,
    run_number,
    benchmark_id,
    local_output_directory,
):
    """Execute one workflow run and collect timing and output metadata."""
    for sampler in samplers:
        sampler["inputs"]["seed"] = seed

    request_started = time.perf_counter()
    prompt_response = queue_prompt(server_address, prompt_workflow)
    prompt_id = prompt_response["prompt_id"]
    completed, queue_seconds, execution_seconds = wait_for_prompt(
        websocket_client,
        prompt_id,
        request_started,
    )
    duration_seconds = completed - request_started
    local_filename_prefix = (
        f"benchmark_{benchmark_id[:8]}_{phase}_{run_number:02d}_{seed}"
    )
    outputs = collect_outputs(
        server_address,
        prompt_id,
        local_output_directory,
        local_filename_prefix,
    )

    result = {
        "phase": phase,
        "run": run_number,
        "prompt_id": prompt_id,
        "seed": seed,
        "status": "completed",
        "duration_seconds": round(duration_seconds, 3),
        "queue_seconds": round(queue_seconds, 3) if queue_seconds is not None else None,
        "execution_seconds": (
            round(execution_seconds, 3) if execution_seconds is not None else None
        ),
        "outputs": outputs,
    }
    print(
        f"  {phase.title()} {run_number}: {duration_seconds:.3f} s "
        f"(seed {seed}, {len(outputs)} outputs)"
    )
    return result


def percentile(values, percent):
    """Calculate a percentile using linear interpolation."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values):
    """Build common descriptive statistics for duration values."""
    if not values:
        return {}

    return {
        "average_seconds": round(statistics.fmean(values), 3),
        "median_seconds": round(statistics.median(values), 3),
        "minimum_seconds": round(min(values), 3),
        "maximum_seconds": round(max(values), 3),
        "standard_deviation_seconds": round(statistics.pstdev(values), 3),
        "p95_seconds": round(percentile(values, 95), 3),
    }


def workflow_details(prompt_workflow, workflow_path):
    """Summarize workflow identity, nodes, models, and generation settings."""
    model_keys = {
        "ckpt_name",
        "unet_name",
        "clip_name",
        "vae_name",
        "lora_name",
        "control_net_name",
    }
    models = []
    samplers = []
    resolution = []

    for node_id, node in prompt_workflow.items():
        class_type = node.get("class_type", "Unknown")
        inputs = node.get("inputs", {})
        selected_models = {key: inputs[key] for key in model_keys if key in inputs}
        if selected_models:
            models.append(
                {"node_id": node_id, "class_type": class_type, **selected_models}
            )
        if class_type == "KSampler":
            samplers.append(
                {
                    "node_id": node_id,
                    **{
                        key: inputs.get(key)
                        for key in (
                            "steps",
                            "cfg",
                            "sampler_name",
                            "scheduler",
                            "denoise",
                        )
                    },
                }
            )
        if class_type in {"ResolutionSelector", "EmptyLatentImage"}:
            resolution.append(
                {
                    "node_id": node_id,
                    "class_type": class_type,
                    **{
                        key: value
                        for key, value in inputs.items()
                        if key in {"width", "height", "batch_size", "aspect_ratio", "megapixels", "multiple"}
                    },
                }
            )

    workflow_bytes = workflow_path.read_bytes()
    class_counts = Counter(
        node.get("class_type", "Unknown") for node in prompt_workflow.values()
    )
    return {
        "file": str(workflow_path),
        "sha256": hashlib.sha256(workflow_bytes).hexdigest(),
        "node_count": len(prompt_workflow),
        "node_class_counts": dict(sorted(class_counts.items())),
        "models": models,
        "samplers": samplers,
        "resolution": resolution,
    }


def run_benchmark(
    server_address,
    workflow_path,
    manifest_path,
    measured_runs=5,
    warmup_runs=1,
    sampling_steps=20,
    poll_interval=0.1,
    base_seed=None,
):
    """Run the complete benchmark and write images plus a JSON manifest."""
    if measured_runs < 1 or warmup_runs < 0:
        raise ValueError("Runs must be >= 1 and warm-up runs must be >= 0.")

    benchmark_id = str(uuid.uuid4())
    started_at = utc_now()
    wall_started = time.perf_counter()
    base_seed = base_seed if base_seed is not None else uuid.uuid4().int % (2**63)
    local_output_directory = manifest_path.parent
    local_output_directory.mkdir(parents=True, exist_ok=True)

    with open(workflow_path, "r", encoding="utf-8") as file:
        prompt_workflow = json.load(file)

    samplers = [
        node
        for node in prompt_workflow.values()
        if node.get("class_type") == "KSampler"
    ]
    if not samplers:
        raise RuntimeError("The workflow does not contain a KSampler node.")
    for sampler in samplers:
        sampler["inputs"]["steps"] = sampling_steps

    initial_stats = get_system_stats(server_address)
    server_info = initial_stats["system"]
    device_info = initial_stats["devices"][0]
    pytorch_version = server_info["pytorch_version"]
    cuda_version = cuda_version_from_pytorch(pytorch_version)
    server_arguments = server_info.get("argv", [])
    baseline_used_bytes = device_info["vram_total"] - device_info["vram_free"]

    total_runs = warmup_runs + measured_runs
    print(f"Starting benchmark: {warmup_runs} warm-up + {measured_runs} measured runs")
    print(f"Server:   http://{server_address}")
    print(f"Workflow: {workflow_path}")
    print(f"Device:   {device_info['name']}")
    print(f"PyTorch:  {pytorch_version}")
    print(f"CUDA:     {cuda_version}")
    print(f"Cache:    {detect_cache_mode(server_arguments)}")
    print(f"Settings: {sampling_steps} steps\n")

    websocket_client = websocket.WebSocket()
    websocket_client.connect(f"ws://{server_address}/ws?clientId={CLIENT_ID}")

    vram_state = {
        "peak_used_bytes": baseline_used_bytes,
        "samples": 0,
        "failed_samples": 0,
    }
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_vram,
        args=(server_address, monitor_stop, vram_state, poll_interval),
        daemon=True,
    )
    monitor_thread.start()

    warmup_results = []
    measured_results = []
    try:
        for index in range(total_runs):
            seed = base_seed + index
            if index < warmup_runs:
                result = run_prompt(
                    server_address,
                    websocket_client,
                    prompt_workflow,
                    samplers,
                    seed,
                    "warmup",
                    index + 1,
                    benchmark_id,
                    local_output_directory,
                )
                warmup_results.append(result)
            else:
                measured_index = index - warmup_runs + 1
                result = run_prompt(
                    server_address,
                    websocket_client,
                    prompt_workflow,
                    samplers,
                    seed,
                    "measured",
                    measured_index,
                    benchmark_id,
                    local_output_directory,
                )
                measured_results.append(result)
    finally:
        monitor_stop.set()
        monitor_thread.join(timeout=max(3, poll_interval + 1))
        websocket_client.close()

    durations = [result["duration_seconds"] for result in measured_results]
    queue_durations = [
        result["queue_seconds"]
        for result in measured_results
        if result["queue_seconds"] is not None
    ]
    execution_durations = [
        result["execution_seconds"]
        for result in measured_results
        if result["execution_seconds"] is not None
    ]
    output_count = sum(len(result["outputs"]) for result in measured_results)
    total_measured_seconds = sum(durations)
    images_per_minute = (
        output_count / total_measured_seconds * 60 if total_measured_seconds else 0
    )

    peak_used_bytes = vram_state["peak_used_bytes"]
    completed_at = utc_now()
    manifest = {
        "schema_version": "2.0",
        "benchmark": {
            "benchmark_id": benchmark_id,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "wall_clock_seconds": round(time.perf_counter() - wall_started, 3),
            "server": f"http://{server_address}",
            "client_id": CLIENT_ID,
            "local_output_directory": str(local_output_directory),
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "sampling_steps": sampling_steps,
            "base_seed": base_seed,
            "cache_mode": detect_cache_mode(server_arguments),
        },
        "environment": {
            "os": server_info.get("os"),
            "comfyui_version": server_info.get("comfyui_version"),
            "python_version": server_info.get("python_version"),
            "pytorch_version": pytorch_version,
            "cuda_version": cuda_version,
            "gpu_driver_version": get_gpu_driver_version(),
            "embedded_python": server_info.get("embedded_python"),
            "server_arguments": server_arguments,
            "device": {
                "name": device_info.get("name"),
                "type": device_info.get("type"),
                "index": device_info.get("index"),
                "vram_total_bytes": device_info.get("vram_total"),
                "vram_total_gb": round(device_info["vram_total"] / GIB, 2),
            },
            "system_ram_total_gb": round(server_info.get("ram_total", 0) / GIB, 2),
        },
        "workflow": workflow_details(prompt_workflow, workflow_path),
        "runs": {
            "warmup": warmup_results,
            "measured": measured_results,
        },
        "performance": {
            "end_to_end": summarize(durations),
            "queue": summarize(queue_durations),
            "execution": summarize(execution_durations),
            "measured_duration_seconds": round(total_measured_seconds, 3),
            "output_count": output_count,
            "images_per_minute": round(images_per_minute, 3),
            "p95_note": (
                "Treat p95 as indicative only when fewer than 20 measured runs are used."
            ),
        },
        "memory": {
            "vram_total_gb": round(device_info["vram_total"] / GIB, 2),
            "baseline_vram_used_gb": round(baseline_used_bytes / GIB, 2),
            "peak_vram_used_gb": round(peak_used_bytes / GIB, 2),
            "benchmark_vram_delta_gb": round(
                max(0, peak_used_bytes - baseline_used_bytes) / GIB,
                2,
            ),
            "poll_interval_ms": round(poll_interval * 1000),
            "successful_samples": vram_state["samples"],
            "failed_samples": vram_state["failed_samples"],
            "note": (
                "Sampled from ComfyUI /system_stats; values include baseline "
                "usage and other GPU processes, and short spikes may be missed."
            ),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)

    stats = manifest["performance"]["end_to_end"]
    memory = manifest["memory"]
    print("\nBenchmark completed successfully.")
    print(f"Average:     {stats['average_seconds']:.3f} s")
    print(f"Median:      {stats['median_seconds']:.3f} s")
    print(f"Min / Max:   {stats['minimum_seconds']:.3f} / {stats['maximum_seconds']:.3f} s")
    print(f"Std dev:     {stats['standard_deviation_seconds']:.3f} s")
    print(f"P95:         {stats['p95_seconds']:.3f} s")
    print(f"Throughput:  {images_per_minute:.3f} images/min")
    print(f"Peak VRAM:   {memory['peak_vram_used_gb']:.2f} GB")
    print(f"VRAM delta:  {memory['benchmark_vram_delta_gb']:.2f} GB")
    print(f"Images:      {local_output_directory}")
    print(f"Manifest:    {manifest_path}")


def parse_arguments():
    """Parse command-line options for benchmark configuration."""
    parser = argparse.ArgumentParser(
        description="Benchmark a ComfyUI API workflow and write a JSON manifest."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER_ADDRESS)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--runs", type=int, default=5, help="Measured runs (default: 5).")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warm-up runs excluded from statistics (default: 1).",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="VRAM polling interval in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        help="First seed. A random seed is used when omitted.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_benchmark(
        server_address=args.server,
        workflow_path=args.workflow.resolve(),
        manifest_path=args.manifest.resolve(),
        measured_runs=args.runs,
        warmup_runs=args.warmup_runs,
        sampling_steps=args.steps,
        poll_interval=args.poll_interval,
        base_seed=args.base_seed,
    )
