# Qwen2-VL-7B OCR LoRA Fine-tuning 训练记录

## 基本信息

- **日期**: 2026-03-16
- **目标**: 复现 OptMerge 论文中 Qwen2-VL OCR LoRA fine-tuning 实验
- **基座模型**: `Qwen/Qwen2-VL-7B`（基座版，非 Instruct 版）
- **训练框架**: LLaMA-Factory 0.9.3.dev0
- **硬件**: 单卡 NVIDIA RTX 4090 (24GB)，GPU 3
- **Conda 环境**: `/data/lishichao/env/llamaF`（Python 3.12, PyTorch 2.6.0+cu124）

## 训练数据

- **数据集**: LLaVAR OCR 子集
- **来源**: 从 `llava_instruct_150k_llavar_20k.json` 中过滤 OCR 专用样本（image ID 以 `1000...` 开头）
- **样本数**: 19,800
- **数据格式**: ShareGPT 格式（multi-turn conversations），包含 `<image>` token
- **图片目录**: `/data/lishichao/data/Optmerge/OCR/LLaVAR/`（442,115 张 JPG 图片）
- **数据准备脚本**: `LLaMA-Factory/scripts/prepare_ocr_data.py`
- **处理后文件**: `LLaMA-Factory/data/llavar_ocr.json`，注册在 `LLaMA-Factory/data/dataset_info.json`

## 训练超参数

| 参数 | 值 |
|---|---|
| finetuning_type | lora |
| lora_rank | 8 |
| lora_target | all |
| stage | sft |
| template | qwen2_vl |
| per_device_train_batch_size | 1 |
| gradient_accumulation_steps | 8 |
| effective_batch_size | 8 |
| learning_rate | 1e-4 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.1 |
| num_train_epochs | 3 |
| cutoff_len | 2048 |
| bf16 | true |
| gradient_checkpointing | true |
| image_max_pixels | 262144 |

## 训练结果

| 指标 | 值 |
|---|---|
| train_loss | 0.5695 |
| 最终 epoch loss | ~0.43 |
| total_steps | 7,425 |
| trainable_parameters | 20,185,088 (~20M) |
| train_runtime | 5h 04m 27s |
| train_samples_per_second | 3.252 |
| train_steps_per_second | 0.406 |
| GPU 显存占用 | ~18.3GB / 24GB |

## 输出文件

- **LoRA adapter**: `LLaMA-Factory/saves/qwen2vl-7b-ocr/lora/sft/adapter_model.safetensors` (78MB)
- **训练配置**: `LLaMA-Factory/configs/qwen2vl_ocr_lora_sft.yaml`
- **Loss 曲线**: `LLaMA-Factory/saves/qwen2vl-7b-ocr/lora/sft/training_loss.png`
- **训练日志**: `LLaMA-Factory/train_ocr.log`
- **Checkpoints**: checkpoint-6500, checkpoint-7000, checkpoint-7425

## 启动命令

```bash
cd /data/lishichao/project/era-2026/MLLMerging/LLaMA-Factory
CUDA_VISIBLE_DEVICES=3 nohup bash run_train.sh > train_ocr.log 2>&1 &
```

`run_train.sh` 设置了 `LD_LIBRARY_PATH` 修复 CUDA cupti 兼容性问题，详见 bugs 记录。

## 合并 LoRA 到基座模型

```bash
# 合并配置: configs/qwen2vl_ocr_merge_lora.yaml
# 输出目录: saves/qwen2vl-7b-ocr/merged
llamafactory-cli export configs/qwen2vl_ocr_merge_lora.yaml
```

## 备注

- 论文未公开详细训练超参数，当前参数参考 LLaMA-Factory 的 Qwen2-VL LoRA example config
- 论文 HuggingFace 上传的是合并后的 full model（`yongxianwei/Qwen2-VL-7B-OCR`，16.6GB），不是 LoRA adapter
- 还有一个 rank=4 的小配置 `configs/qwen2vl_ocr_lora_sft_small.yaml`，用于显存不足时测试
- 当前仅使用 LLaVAR 数据集；TextVQA、DocVQA、OCRVQA 等其他 OCR 数据集部分未下载完整，后续可补充
