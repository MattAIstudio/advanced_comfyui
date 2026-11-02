import os
from comfy_sdk import Comfy

# 1. 指向本地 API v2 代理地址（默认 8189 端口）
os.environ["COMFY_BASE_URL"] = "http://127.0.0.1:8189"

# 2. 本地自建实例无需传 api_key
client = Comfy()

# 3. 加载本地工作流并执行
wf = client.workflows.from_file(r"D:\Coding\comfyui\wk01\workflow_api.json")

job = client.run(wf)

# 4. 获取并保存节点 9 的输出图片
for output in job.get_outputs("9"):
    output.to_file(output.name)