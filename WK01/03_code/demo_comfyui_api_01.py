import json
from urllib import request

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

def queue_prompt(prompt):
    p = {"prompt": prompt}

    # If the workflow contains API nodes, you can add a Comfy API key to the `extra_data`` field of the payload.
    # p["extra_data"] = {
    #     "api_key_comfy_org": "comfyui-87d01e28d*******************************************************"  # replace with real key
    # }
    # See: https://docs.comfy.org/tutorials/api-nodes/overview
    # Generate a key here: https://platform.comfy.org/login

    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)


prompt = json.loads(prompt_text)
#set the text prompt for our positive CLIPTextEncode
prompt["45"]["inputs"]["text"] = "masterpiece best quality chinese woman walking on beach, 20+years, plump figure, bikini"

#set the seed for our KSampler node
prompt["44"]["inputs"]["seed"] = 5

# print(prompt)


queue_prompt(prompt)

