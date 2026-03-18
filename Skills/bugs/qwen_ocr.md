# Qwen2-VL OCR 训练 Bug 和代码更改记录

## Bug 1: CUDA cupti 符号缺失导致 torch 无法导入

**错误信息**:
```
ImportError: /data/lishichao/env/llamaF/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so:
undefined symbol: cuptiActivityEnableDriverApi, version libcupti.so.12
```

**原因**:
conda 环境中之前安装了 torch 2.10.0（较新版本），其 `libtorch_cpu.so` 链接了更高版本的 cupti 符号。卸载后重装 torch 2.6.0+cu124 时，`.so` 文件残留导致不兼容。系统自带的 cupti `/usr/lib/x86_64-linux-gnu/libcupti.so` 版本太旧（CUDA 11.x），不包含 `cuptiActivityEnableDriverApi` 这个符号。

**解决方案**:
1. 使用 `pip install --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` 彻底重装 torch
2. 在启动脚本 `run_train.sh` 中设置 `LD_LIBRARY_PATH`，确保使用 pip 安装的 NVIDIA 库而非系统库：
```bash
export LD_LIBRARY_PATH=/data/lishichao/env/llamaF/lib/python3.12/site-packages/nvidia/cuda_cupti/lib:\
/data/lishichao/env/llamaF/lib/python3.12/site-packages/nvidia/cublas/lib:\
/data/lishichao/env/llamaF/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
```

**涉及文件**: `LLaMA-Factory/run_train.sh`（新建）

---

## Bug 2: pip install 后依赖版本冲突

**错误信息**:
```
torch 2.6.0+cu124 requires nvidia-cuda-cupti-cu12==12.4.127, but you have nvidia-cuda-cupti-cu12 12.9.79
trl 0.9.6 requires numpy<2.0.0, but you have numpy 2.3.5
datasets 3.2.0 requires fsspec<=2024.9.0, but you have fsspec 2025.12.0
gradio 5.12.0 requires pillow<12.0,>=8.0, but you have pillow 12.0.0
```

**原因**: `--force-reinstall` torch 时拉取了最新版的传递依赖（numpy, fsspec, pillow 等），与 LLaMA-Factory 的版本约束冲突。

**解决方案**:
```bash
pip install "numpy<2.0.0" "markupsafe~=2.0" "pillow<12.0,>=8.0" "fsspec<=2024.9.0,>=2023.1.0"
```

---

## Bug 3: Qwen2-VL-7B 模型 shard 缓存不完整

**错误信息**:
```
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it
in the cached files and it looks like Qwen/Qwen2-VL-7B is not the path to a directory containing
a file named model-00003-of-00005.safetensors.
```

**原因**: HuggingFace cache 中 Qwen2-VL-7B 只有 2/5 个 shard 下载完成，训练脚本在 proxy 断开时无法下载剩余部分。

**解决方案**:
```python
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2-VL-7B', resume_download=True)
```
需要开启 proxy (`proxy_on`) 并等待全部 5 个 shard (~15GB) 下载完成。

---

## 代码更改汇总

### 新建文件

| 文件 | 说明 |
|---|---|
| `LLaMA-Factory/scripts/prepare_ocr_data.py` | OCR 数据准备脚本，从 LLaVAR 提取 OCR 样本并转为 LLaMA-Factory sharegpt 格式 |
| `LLaMA-Factory/data/dataset_info.json` | 数据集注册配置 |
| `LLaMA-Factory/data/llavar_ocr.json` | 处理后的 19,800 条 OCR 训练数据 |
| `LLaMA-Factory/configs/qwen2vl_ocr_lora_sft.yaml` | OCR LoRA 训练配置（rank=8） |
| `LLaMA-Factory/configs/qwen2vl_ocr_lora_sft_small.yaml` | OCR LoRA 训练配置（rank=4，显存不足时用） |
| `LLaMA-Factory/configs/qwen2vl_ocr_merge_lora.yaml` | LoRA 合并到基座模型的 export 配置 |
| `LLaMA-Factory/run_train.sh` | 训练启动脚本，含 LD_LIBRARY_PATH 修复和 proxy 设置 |
| `LLaMA-Factory/scripts/train_ocr.sh` | 训练封装脚本（自动准备数据 + 检测 GPU 数量） |
