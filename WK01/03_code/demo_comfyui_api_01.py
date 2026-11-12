import argparse
import json
import os
import time
from pathlib import Path
from urllib import error, parse, request


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("COMFYUI_OUTPUT_DIR", SCRIPT_DIR / "output")
)
POLL_INTERVAL = 1.0
REQUEST_TIMEOUT = 10
DEFAULT_PROMPT = (
    "masterpiece best quality chinese woman walking on beach, "
    "20+years, plump figure, bikini"
)
DEFAULT_SEED = 5

#This is the ComfyUI api prompt format.

#If you want it for a specific workflow you can "File -> Export (API)" in the interface.

#this is the one for the default workflow
prompt_text = """
{
  "39": {
    "inputs": {
      "clip_name": "qwen_3_4b.safetensors",
      "type": "lumina2",
      "device": "default"
    },
    "class_type": "CLIPLoader",
    "_meta": {
      "title": "Load CLIP"
    }
  },
  "40": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "41": {
    "inputs": {
      "width": 1088,
      "height": 1920,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage",
    "_meta": {
      "title": "EmptySD3LatentImage"
    }
  },
  "42": {
    "inputs": {
      "conditioning": [
        "45",
        0
      ]
    },
    "class_type": "ConditioningZeroOut",
    "_meta": {
      "title": "Conditioning Zero Out"
    }
  },
  "43": {
    "inputs": {
      "samples": [
        "44",
        0
      ],
      "vae": [
        "40",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "44": {
    "inputs": {
      "seed": 977768513915292,
      "steps": 9,
      "cfg": 1,
      "sampler_name": "res_multistep",
      "scheduler": "simple",
      "denoise": 1,
      "model": [
        "47",
        0
      ],
      "positive": [
        "45",
        0
      ],
      "negative": [
        "42",
        0
      ],
      "latent_image": [
        "41",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "45": {
    "inputs": {
      "text": "",
      "clip": [
        "39",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "46": {
    "inputs": {
      "unet_name": "z_image_turbo_bf16.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "47": {
    "inputs": {
      "shift": 3,
      "sampling": "flow",
      "model": [
        "46",
        0
      ]
    },
    "class_type": "ModelSamplingAuraFlow",
    "_meta": {
      "title": "ModelSamplingAuraFlow"
    }
  },
  "48": {
    "inputs": {
      "images": [
        "43",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "51": {
    "inputs": {
      "clean_file_cache": true,
      "clean_processes": true,
      "clean_dlls": true,
      "retry_times": 3,
      "anything": [
        "43",
        0
      ]
    },
    "class_type": "RAMCleanup",
    "_meta": {
      "title": "🎈RAM-Cleanup"
    }
  },
  "53": {
    "inputs": {
      "offload_model": true,
      "offload_cache": true,
      "anything": [
        "51",
        0
      ]
    },
    "class_type": "VRAMCleanup",
    "_meta": {
      "title": "🎈VRAM-Cleanup"
    }
  },
  "54": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "43",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}
"""


def find_node_id(prompt, class_types, title_text=None):
    """Find a workflow node without relying on exported numeric node IDs."""
    matches = [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") in class_types
    ]
    if title_text:
        for node_id, node in matches:
            title = node.get("_meta", {}).get("title", "")
            if title_text.lower() in title.lower():
                return node_id
    return matches[0][0] if matches else None


def configure_prompt(prompt, text, seed):
    text_node_id = find_node_id(prompt, {"CLIPTextEncode"}, "Prompt")
    sampler_node_id = find_node_id(prompt, {"KSampler", "KSamplerAdvanced"})
    if not text_node_id or not sampler_node_id:
        raise RuntimeError("Required prompt or sampler node was not found.")
    prompt[text_node_id]["inputs"]["text"] = text
    prompt[sampler_node_id]["inputs"]["seed"] = seed


def check_comfyui_server(server_url):
    """Show a clear message when the local ComfyUI server is unavailable."""
    try:
        with request.urlopen(
            f"{server_url}/system_stats", timeout=REQUEST_TIMEOUT
        ):
            pass
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "ComfyUI is not running. Please start ComfyUI and try again."
        ) from exc


def queue_prompt(prompt, server_url):
    payload = {"prompt": prompt}

    # If the workflow contains API nodes, you can add a Comfy API key to the `extra_data`` field of the payload.
    # payload["extra_data"] = {
    #     "api_key_comfy_org": "comfyui-87d01e28d*******************************************************"  # replace with real key
    # }
    # See: https://docs.comfy.org/tutorials/api-nodes/overview
    # Generate a key here: https://platform.comfy.org/login

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{server_url}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        result = json.load(response)
    return result["prompt_id"]


def wait_for_completion(prompt_id, server_url):
    while True:
        with request.urlopen(
            f"{server_url}/history/{prompt_id}", timeout=REQUEST_TIMEOUT
        ) as response:
            history = json.load(response)

        if prompt_id in history:
            result = history[prompt_id]
            status = result.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError("ComfyUI could not complete the workflow.")
            return result

        time.sleep(POLL_INTERVAL)


def save_output_images(history, prompt, server_url, output_dir):
    """Download images from the SaveImage node and return their local paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    save_node_ids = {
        node_id
        for node_id, node in prompt.items()
        if node.get("class_type") == "SaveImage"
    }

    for node_id in save_node_ids:
        for image in history.get("outputs", {}).get(node_id, {}).get("images", []):
            query = parse.urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            destination = output_dir / Path(image["filename"]).name
            with request.urlopen(
                f"{server_url}/view?{query}", timeout=REQUEST_TIMEOUT
            ) as response:
                destination.write_bytes(response.read())
            saved_paths.append(destination.resolve())

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run a workflow with ComfyUI API v1.")
    parser.add_argument("--server", default=DEFAULT_SERVER_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    server_url = args.server.rstrip("/")
    output_dir = args.output_dir.expanduser().resolve()

    print("Starting image generation...")
    print(f"Checking ComfyUI server: {server_url}")
    check_comfyui_server(server_url)
    print("ComfyUI server is ready.")

    prompt = json.loads(prompt_text)
    configure_prompt(prompt, args.prompt, args.seed)

    prompt_id = queue_prompt(prompt, server_url)
    print(f"Task submitted. ID: {prompt_id}")
    print("Processing... Please wait.")

    history = wait_for_completion(prompt_id, server_url)
    output_paths = save_output_images(
        history, prompt, server_url, output_dir
    )
    if not output_paths:
        raise RuntimeError("The workflow finished, but no output image was found.")

    print("Processing complete.")
    for path in output_paths:
        print(f"Saved image: {path}")


if __name__ == "__main__":
    try:
        main()
    except (TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Processing failed: {exc}")
        raise SystemExit(1)
    except error.URLError:
        print("Processing failed: Lost connection to ComfyUI.")
        raise SystemExit(1)

