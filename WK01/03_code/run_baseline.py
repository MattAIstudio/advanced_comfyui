import json
import time
import urllib.request
from pathlib import Path
import pynvml
import torch

SERVER_ADDRESS = "127.0.0.1:8188"
RUNS = 3
SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = SCRIPT_DIR / "workflow_api.json"
RESULT_DIR = SCRIPT_DIR / "baseline_result"

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"http://{SERVER_ADDRESS}/prompt", data=data)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode('utf-8'))['prompt_id']

def wait_for_completion(prompt_id):
    while True:
        with urllib.request.urlopen(f"http://{SERVER_ADDRESS}/history/{prompt_id}") as resp:
            history = json.loads(resp.read().decode('utf-8'))
            if prompt_id in history:
                return history[prompt_id]
        time.sleep(0.5)

# 显存监控辅助函数
def get_gpu_memory_used():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    return info.used / (1024 ** 2)  # MB

def run_benchmark():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with WORKFLOW_PATH.open("r", encoding="utf-8") as f:
        workflow = json.load(f)

    # 1. 固定 Seed (根据你的工作流节点 ID 进行替换，通常是 KSampler 节点)
    for node_id, node in workflow.items():
        if node.get("class_type") in ["KSampler", "KSamplerAdvanced"]:
            node["inputs"]["seed"] = 42

    records = []
    
    for i in range(RUNS):
        print(f"--- Running Iteration {i+1}/{RUNS} ---")
        start_mem = get_gpu_memory_used()
        start_time = time.time()
        
        prompt_id = queue_prompt(workflow)
        history = wait_for_completion(prompt_id)
        
        duration = time.time() - start_time
        peak_mem = get_gpu_memory_used() # 或在循环中采样得到的最大值

        # 从 history 中提取输出图像文件名
        output_images = []
        for node_id, out in history.get('outputs', {}).items():
            if 'images' in out:
                for img in out['images']:
                    output_images.append(img['filename'])

        records.append({
            "run_id": i + 1,
            "duration_sec": round(duration, 2),
            "peak_vram_mb": round(peak_mem, 2),
            "output_images": output_images
        })
        print(f"Run {i+1} done: {duration:.2f}s, Peak VRAM: {peak_mem:.2f} MB")

    # 2. 生成 manifest.json
    manifest = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware_and_env": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "config": {
            "seed": 42,
            "steps": 20, # 根据实际填入
            "resolution": "512x512"
        },
        "metrics": records,
        "summary": {
            "cold_start_time": records[0]["duration_sec"],
            "avg_warm_time": round((records[1]["duration_sec"] + records[2]["duration_sec"]) / 2, 2)
        }
    }

    manifest_path = RESULT_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)
        
    print(f"Baseline finished. Results saved to {manifest_path}")

if __name__ == "__main__":
    run_benchmark()
