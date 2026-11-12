import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import httpx
    from comfy_sdk import Comfy
    from comfy_sdk.exceptions import ComfyError
except ModuleNotFoundError as exc:
    print(
        f"Missing dependency: {exc.name}. Install it with the same Python "
        "interpreter:\n"
        f'python -m pip install -r "{SCRIPT_DIR / "requirements_comfyui_api_02.txt"}"'
    )
    raise SystemExit(1)


DEFAULT_API_BASE_URL = os.environ.get(
    "COMFY_BASE_URL", "http://127.0.0.1:8189"
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("COMFYUI_OUTPUT_DIR", SCRIPT_DIR / "output")
)
REQUEST_TIMEOUT = 30.0

DEFAULT_PROMPT = (
    "masterpiece best quality chinese woman walking on beach, "
    "20+years, plump figure, bikini"
)
DEFAULT_SEED = 5


def find_default_workflow():
    """Find workflow_api.json in common locations without a fixed layout."""
    candidates = (
        SCRIPT_DIR / "workflow_api.json",
        SCRIPT_DIR.parent / "workflow_api.json",
        Path.cwd() / "workflow_api.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def find_node_id(workflow, class_types, title_text=None):
    """Find a workflow node without relying on exported numeric node IDs."""
    matches = [
        (node_id, node)
        for node_id, node in workflow.json.items()
        if node.get("class_type") in class_types
    ]
    if title_text:
        for node_id, node in matches:
            title = node.get("_meta", {}).get("title", "")
            if title_text.lower() in title.lower():
                return node_id
    return matches[0][0] if matches else None


def configure_workflow(workflow, text, seed):
    text_node_id = find_node_id(workflow, {"CLIPTextEncode"}, "Prompt")
    sampler_node_id = find_node_id(
        workflow, {"KSampler", "KSamplerAdvanced"}
    )
    if not text_node_id or not sampler_node_id:
        raise RuntimeError("Required prompt or sampler node was not found.")
    workflow.set_input(text_node_id, "text", text)
    workflow.set_input(sampler_node_id, "seed", seed)


def get_save_node_ids(workflow):
    """Return all SaveImage node IDs in the workflow."""
    return [
        node_id
        for node_id, node in workflow.json.items()
        if node.get("class_type") == "SaveImage"
    ]


def save_outputs(job, node_ids, output_dir):
    """Download SDK job outputs and return their absolute local paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for node_id in node_ids:
        for output in job.get_outputs(node_id):
            destination = output_dir / Path(output.name).name
            output.to_file(destination)
            saved_paths.append(destination.resolve())

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run a workflow with Comfy API v2.")
    parser.add_argument("--base-url", default=DEFAULT_API_BASE_URL)
    parser.add_argument("--workflow", type=Path, default=find_default_workflow())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    workflow_path = args.workflow.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    os.environ["COMFY_BASE_URL"] = base_url

    print("Starting image generation with Comfy SDK...")
    print(f"Loading workflow: {workflow_path}")

    if not workflow_path.is_file():
        raise RuntimeError(
            f"Workflow file not found: {workflow_path}. "
            "Use --workflow to specify its location."
        )

    print(f"Checking ComfyUI API v2 service: {base_url}")
    with Comfy(timeout=REQUEST_TIMEOUT) as client:
        workflow = client.workflows.from_file(workflow_path)
        configure_workflow(workflow, args.prompt, args.seed)

        save_node_ids = get_save_node_ids(workflow)
        if not save_node_ids:
            raise RuntimeError("No SaveImage node was found in the workflow.")

        job = client.submit(workflow)
        print("ComfyUI API v2 service is ready.")
        print(f"Task submitted. ID: {job.id}")
        print("Processing... Please wait.")

        job.result()
        output_paths = save_outputs(job, save_node_ids, output_dir)

    if not output_paths:
        raise RuntimeError("The workflow finished, but no output image was found.")

    print("Processing complete.")
    for path in output_paths:
        print(f"Saved image: {path}")


if __name__ == "__main__":
    try:
        main()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        print(
            "Processing failed: Cannot connect to ComfyUI API v2. "
            "Check --base-url, start the service, and try again."
        )
        raise SystemExit(1)
    except httpx.TimeoutException:
        print("Processing failed: The ComfyUI API request timed out.")
        raise SystemExit(1)
    except ComfyError as exc:
        if exc.http_status in {502, 503, 504}:
            print(
                "Processing failed: ComfyUI is not running or the API v2 "
                "service cannot reach it. Please start ComfyUI and try again."
            )
        else:
            print(f"Processing failed: {exc}")
        raise SystemExit(1)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Processing failed: {exc}")
        raise SystemExit(1)
