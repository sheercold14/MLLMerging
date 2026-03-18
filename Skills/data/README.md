# OptMerge 数据集下载指南

论文: **OptMerge: Unifying Multimodal LLM Capabilities and Modalities via Model Merging**

## 快速开始

### 环境准备

```bash
pip install huggingface_hub
# 可选: 登录 HuggingFace (部分数据集可能需要)
huggingface-cli login
```

### 下载训练数据 (VQA + OCR)

```bash
bash download_all.sh /path/to/save
```

默认保存到脚本同目录下的 `Optmerge/`，总计约 **100GB**。

### 下载评估数据

```bash
# 先下载训练数据，再下载评估数据 (会自动创建软链接复用已有数据)
bash download_eval.sh /path/to/save
```

总计约 **18GB**。

## 数据集清单

### VQA 训练数据 (588K samples)

| 数据集 | HF Repo | 大小 | 图像 |
|--------|---------|------|------|
| GQA | `lmms-lab/GQA` | ~29G | parquet 内嵌 |
| VQAv2 | `HuggingFaceM4/VQAv2` + 官方 annotations | ~662M | 需 COCO train2014/val2014 |
| OKVQA | `Multimodal-Fatima/OK-VQA_train` | ~230M | parquet 内嵌 |
| LLaVA-Instruct | `liuhaotian/LLaVA-Instruct-150K` | ~559M | 需 COCO train2014 |
| CogVLM (single+multi) | `THUDM/CogVLM-SFT-311K` | ~7.7G zip | 含图像, 需解压 |

### OCR 训练数据 (238K samples)

| 数据集 | HF Repo | 大小 | 图像 |
|--------|---------|------|------|
| OCRVQA | `howard-hou/OCR-VQA` | ~2G | parquet 内嵌 |
| TextCaps | `lmms-lab/TextCaps` | ~2G | parquet 内嵌 |
| SynthDoG | `naver-clova-ix/synthdog-en` | ~40G | parquet 内嵌 |
| LLaVAR | `SALT-NLP/LLaVAR` | ~14G | zip 需解压 |
| ST-VQA | `lmms-lab/ST-VQA` | ~313M | parquet 内嵌 |
| TextVQA | `lmms-lab/textvqa` | ~7G | parquet 内嵌 |
| DocVQA | `lmms-lab/DocVQA` | ~2G | parquet 内嵌 |
| TabFact | GitHub `wenhuchen/Table-Fact-Checking` | ~650M | 纯文本/表格 |
| DeepForm | 私有 repo, 已跳过 | - | - |
| KLC | 私有 repo, 已跳过 | - | - |

### 评估数据

| 任务 | 数据集 | HF Repo | 大小 |
|------|--------|---------|------|
| VQA | VizWiz | `lmms-lab/VizWiz-VQA` | ~5.7G |
| VQA | GQA (test) | 复用训练数据 | - |
| OCR | TextVQA (val) | 复用训练数据 | - |
| OCR | OCRVQA (test) | 复用训练数据 | - |
| Chart | ChartQA | `lmms-lab/ChartQA` | ~70M |
| Geometry | MathVista | `AI4Math/MathVista` | ~1.7G |
| Geometry | MATH-Vision | `MathLLMs/MathVision` | ~111M |
| Grounding | RefCOCO | `lmms-lab/RefCOCO` | ~2.2G |
| Grounding | RefCOCO+ | `lmms-lab/RefCOCOplus` | ~503M |
| Grounding | RefCOCOg | `lmms-lab/RefCOCOg` | ~2.0G |
| Audio-Visual | AVQA | `harryhsing/AVQA-R1-6K` | ~5.1G |
| Audio-Visual | MUSIC-AVQA | GitHub `GeWu-Lab/MUSIC-AVQA` | ~642M |

## 目录结构

```
Optmerge/
├── VQA/
│   ├── GQA/                      # parquet 含图像
│   ├── VQAv2/                    # annotations + COCO 软链接
│   ├── OKVQA/                    # parquet 含图像
│   ├── LLaVA-Instruct/           # JSON + COCO 软链接
│   └── CogVLM-SFT-311K/         # 含 single/multi round 图像
├── OCR/
│   ├── OCRVQA/                   # parquet 含图像
│   ├── TextCaps/                 # parquet 含图像
│   ├── SynthDoG/                 # parquet 含合成图像
│   ├── LLaVAR/                   # jpg 图像 + JSON 标注
│   ├── ST-VQA/                   # parquet 含图像
│   ├── TextVQA/                  # parquet 含图像
│   ├── DocVQA/                   # parquet 含图像
│   └── TabFact/                  # 表格数据
├── Eval/
│   ├── VizWiz/
│   ├── GQA -> ../VQA/GQA
│   ├── TextVQA -> ../OCR/TextVQA
│   ├── OCRVQA -> ../OCR/OCRVQA
│   ├── ChartQA/
│   ├── MathVista/
│   ├── MATH-Vision/
│   ├── RefCOCO/
│   ├── RefCOCOplus/
│   ├── RefCOCOg/
│   ├── AVQA/
│   └── MUSIC-AVQA/
└── images/
    └── coco/                     # COCO 图像 (或软链接)
```

## 备注

- VQAv2 和 LLaVA-Instruct 需要 COCO 图像, 脚本会提示是否下载
- 如已有 COCO 数据, 手动创建软链接: `ln -s /your/coco Optmerge/images/coco`
- DeepForm 和 KLC 为 HF 私有 repo, 在 OCR 238K 中占比小, 已跳过
- 论文仓库只提供合并/评估代码, 不含训练脚本, 复现训练需自行编写 LLaMA-Factory 配置
