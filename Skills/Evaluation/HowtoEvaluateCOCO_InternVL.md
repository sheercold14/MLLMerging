# InternVL RefCOCO Evaluation

## 1. 环境

已安装好的 conda 环境：`/data/shichao/envs/eval_coco`

如需重新安装：
```bash
cd InternVL
pip install -r requirements/internvl_chat.txt
pip install timm
pip install 'bitsandbytes>=0.43'  # requirements 里的 0.42.0 不兼容 CUDA 12.8
```

## 2. 代码修改（已完成，无需重复操作）

未安装 `flash_attn` 时，模型 config 中 `llm_config._attn_implementation` 硬编码为 `flash_attention_2` 会报错。已做两处修改：

1. `internvl_chat/internvl/model/internvl_chat/modeling_internvl_chat.py` 第 65 行后新增一行：
```python
config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'  # 新增
```

2. `internvl_chat/internvl/model/__init__.py` 的 `from_pretrained` 调用中加了 `attn_implementation='eager'`。

## 3. 数据准备（已完成，无需重复操作）

代码中需要的路径（相对于 `internvl_chat/`，定义在 `eval/refcoco/evaluate_grounding.py` 的 `ds_collections`）：
- 标注文件：`data/refcoco/*.jsonl`
- COCO 图片：`data/coco/train2014/`（jsonl 中 image 字段引用此路径）

已通过符号链接映射到实际数据位置：
```bash
internvl_chat/data/refcoco -> /data/shichao/data/coco/refcoco
internvl_chat/data/coco    -> /data/shichao/data/coco
```

## 4. 运行评测

`evaluate.sh` 会将第一个参数拼接为本地绝对路径 `$(pwd)/${CHECKPOINT}`，所以 checkpoint 必须是相对于 `internvl_chat/` 的本地路径，不能直接用 HuggingFace model ID。

### 评测 HuggingFace 缓存中的模型

先创建符号链接，再运行：

```bash
cd /data/shichao/era-2026/MLLMerging/InternVL/internvl_chat

# 1. 创建符号链接（以 OpenGVLab/InternVL2_5-1B 为例）
#    先找到缓存中的 snapshot hash
ls ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-1B/snapshots/
#    然后创建链接
mkdir -p OpenGVLab
ln -s ~/.cache/huggingface/hub/models--OpenGVLab--InternVL2_5-1B/snapshots/<hash> OpenGVLab/InternVL2_5-1B

# 2. 运行评测（必须指定 eval_coco 环境的 PATH）
PATH="/data/shichao/envs/eval_coco/bin:$PATH" CUDA_VISIBLE_DEVICES=2,3 GPUS=2 bash evaluate.sh OpenGVLab/InternVL2_5-1B refcoco --dynamic
```

### 评测本地模型（如 merge 后的模型）

支持绝对路径和相对路径（已修改 `evaluate.sh` 支持绝对路径）：

```bash
cd /data/shichao/era-2026/MLLMerging/InternVL/internvl_chat

# 绝对路径
PATH="/data/shichao/envs/eval_coco/bin:$PATH" CUDA_VISIBLE_DEVICES=2,3 GPUS=2 bash evaluate.sh /data/shichao/models/my_merged_model refcoco --dynamic

# 相对路径（相对于 internvl_chat/）
PATH="/data/shichao/envs/eval_coco/bin:$PATH" CUDA_VISIBLE_DEVICES=2,3 GPUS=2 bash evaluate.sh merged_models/my_model refcoco --dynamic
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `PATH="/data/shichao/envs/eval_coco/bin:$PATH"` | **必须**，指定使用 eval_coco 环境 |
| `CUDA_VISIBLE_DEVICES` | 指定使用哪些 GPU |
| `GPUS` | GPU 数量，需与 CUDA_VISIBLE_DEVICES 一致 |
| 第1个参数 | 模型路径（相对于 internvl_chat/） |
| 第2个参数 | `refcoco` 会同时评测 refcoco_val, refcoco+_val, refcocog_val |
| `--dynamic` | 启用动态分辨率 |

### 结果

结果输出到 `internvl_chat/results/<model_name>.txt`。

## 5. 参考结果（InternVL2_5-1B, eager attention）

| 数据集 | Precision @ 1 |
|--------|--------------|
| refcoco_val | 74.40% |
| refcoco+_val | 67.81% |
| refcocog_val | 71.32% |
