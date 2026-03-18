# VLMEvalKit 评测流程文档

## 基本信息

- **日期**: 2026-03-17
- **目标**: 使用 VLMEvalKit 评测 Qwen2-VL-7B-OCR（OCR LoRA 合并后模型）
- **评测框架**: VLMEvalKit (from OptMerge repo)
- **Conda 环境**: `/data/lishichao/env/eval-kit`（Python 3.12, PyTorch 2.10.0+cu128）
- **硬件**: 3x NVIDIA RTX 4090 (24GB), `CUDA_VISIBLE_DEVICES=0,1,2`

## 环境配置

### 关键依赖版本

| 包 | 版本 | 备注 |
|---|---|---|
| torch | 2.10.0+cu128 | 系统 CUDA 12.4 |
| transformers | 4.57.6 | 5.x 不兼容 VLMEvalKit |
| huggingface-hub | 0.36.2 | 1.x 不兼容 transformers 4.x |
| accelerate | 1.13.0 | 模型加载必需 |
| torchvision | 0.25.0 | |
| qwen-vl-utils | 0.0.14 | Qwen2-VL 推理必需 |
| nvidia-cusparse-cu12 | 12.3.1.170 | 必须降级，12.5.x 与系统 CUDA 12.4 不兼容 |

### 安装步骤

```bash
conda activate /data/lishichao/env/eval-kit
cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit
pip install -e .

# 修复依赖兼容性
pip install 'transformers>=4.45,<5.0' --no-deps
pip install 'huggingface-hub>=0.34,<1.0'
pip install accelerate torchvision qwen-vl-utils pyarrow

# 每次 pip install 后 cusparse 可能被升级回 12.5，需要手动降级
pip install nvidia-cusparse-cu12==12.3.1.170 --no-deps
```

### 已知环境问题

- **cusparse 版本反复被覆盖**: 每次 `pip install` 带依赖的包（torchvision、accelerate 等），cusparse 会被升回 12.5.x，导致 `ImportError: undefined symbol: __nvJitLinkCreate_12_8`。需要在装完后手动 `pip install nvidia-cusparse-cu12==12.3.1.170 --no-deps`

## 数据准备

### 数据目录

```
/data/lishichao/data/Optmerge/Eval/LMUData/
├── ChartQA_TEST.tsv           (2500 rows)
├── GQA_TestDev_Balanced.tsv   (12578 rows)
├── MathVista_MINI.tsv         (1000 rows)
├── MathVision_MINI.tsv        (304 rows)
├── TextVQA_VAL.tsv            (3333 rows)
├── OCRVQA_TEST.tsv            (100424 rows)
├── OCRVQA_TESTCORE.tsv        (1000 rows)
├── VizWiz.tsv                 (4319 rows)
└── images/
    ├── ChartQA_TEST/          (69M)
    ├── GQA_TestDev_Balanced/  (64M)
    ├── MathVista_MINI/        (138M)
    ├── MathVision_MINI/       (9.7M)
    ├── TextVQA_VAL/           (583M)
    ├── OCRVQA/                (898M)
    └── VizWiz/                (2.0G)
```

### 数据来源

TSV 由 `VLMEvalKit/convert_local_to_tsv.py` 从本地 HuggingFace parquet 数据转换而来。TSV 只含 `image_path`（绝对路径），不含 base64 图片数据。

### ⚠️ 与官方 TSV 的差异（重要）

**我们的 TSV 不等同于 VLMEvalKit 官方 TSV**，存在以下差异：

| 数据集 | 我们的 | 官方预期 | 差异说明 |
|--------|--------|----------|----------|
| TextVQA_VAL | 3333 条 | ~5000 条 | HF parquet 的 val split 只有 3333 条 |
| OCRVQA_TESTCORE | 随机采样 1000 条 | 固定子集 1000 条 | 官方是精选的 core subset，不是随机采样 |
| VizWiz | val split 4319 条 | test split ~8000 条 | 用错了 split |
| GQA_TestDev_Balanced | 12578 条 | 12578 条 | 数量一致，内容待确认 |
| ChartQA_TEST | 2500 条 | 2500 条 | 数量一致，内容待确认 |
| MathVista_MINI | 1000 条 | 1000 条 | 数量一致，内容待确认 |
| MathVision_MINI | 304 条 | 304 条 | 数量一致，内容待确认 |

**结论**: 要和论文结果对齐，必须使用官方 TSV。官方 TSV 从 `https://opencompass.openxlab.space/utils/VLMEval/` 下载，当前网络不通，需要代理。

## 模型注册

在 `vlmeval/config.py` 的 `qwen2vl_series` 中添加：

```python
'Qwen2-VL-7B-OCR': partial(Qwen2VLChat,
    model_path='/data/lishichao/data/model/Qwen2-VL-7B-OCR',
    min_pixels=256*28*28, max_pixels=1280*28*28),
```

## VLMEvalKit 代码修改

仅修改了一处（非破坏性）：

**`vlmeval/dataset/image_base.py`** — `prepare_tsv()` 方法去掉了 MD5 校验：

```python
# 修改前:
if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
# 修改后:
if osp.exists(data_path):
```

目的：当本地已有同名 TSV 时直接使用，不校验 MD5，避免因我们自制 TSV 格式不同而触发重新下载。对官方 TSV 无影响。

## 运行评测

```bash
source activate /data/lishichao/env/eval-kit
export LMUData=/data/lishichao/data/Optmerge/Eval/LMUData
export CUDA_VISIBLE_DEVICES=0,1,2
cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc-per-node=3 run.py \
    --data MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST \
    --model Qwen2-VL-7B-OCR \
    --verbose --reuse
```

### 结果汇总

```bash
python results.py outputs/Qwen2-VL-7B-OCR
```

结果输出目录: `./outputs/Qwen2-VL-7B-OCR/T{date}_G{commit}/`

## 评测结果（非官方 TSV）

| 数据集 | 准确率 | 状态 |
|--------|--------|------|
| TextVQA_VAL | 71.57% | 完成 |
| OCRVQA_TESTCORE | 69.90% | 完成 |
| ChartQA_TEST | 60.52% | 完成 |
| GQA_TestDev_Balanced | 38.07% | 完成 |
| MathVista_MINI | - | 需要 OpenAI API key |
| MathVision_MINI | - | 需要 OpenAI API key |
| VizWiz | - | 评测阶段失败 |

**注意**: 以上结果基于自制 TSV（非官方），仅供参考，不可直接与论文对比。

## 待办

- [ ] 通过代理下载官方 TSV，替换自制 TSV 后重新评测
- [ ] 配置 OpenAI API key 以评测 MathVista 和 MathVision
- [ ] 排查 VizWiz 评测失败原因
- [ ] 使用论文中的 merge 后模型进行评测，与论文结果对比
