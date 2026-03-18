# 本地数据转 VLMEvalKit TSV 格式

## 背景

VLMEvalKit 使用自己的 TSV 格式存储评测数据，官方 TSV 从 `opencompass.openxlab.space` 下载。由于网络不通，我们从本地 HuggingFace parquet 数据转换生成了 TSV。

## 转换脚本

**路径**: `VLMEvalKit/convert_local_to_tsv.py`

### 使用方法

```bash
cd /data/lishichao/project/era-2026/MLLMerging/VLMEvalKit

# 转换全部数据集
python convert_local_to_tsv.py

# 转换指定数据集
python convert_local_to_tsv.py chartqa vizwiz textvqa
```

### 支持的数据集

| 参数名 | 数据集 | 源路径 |
|--------|--------|--------|
| chartqa | ChartQA_TEST | `Eval/ChartQA/data/` |
| vizwiz | VizWiz | `Eval/VizWiz/data/` (val split) |
| mathvista | MathVista_MINI | `Eval/MathVista/data/` (testmini) |
| mathvision | MathVision_MINI | `Eval/MATH-Vision/data/` (testmini) |
| textvqa | TextVQA_VAL | `Eval/TextVQA/TextVQA/data/` (val) |
| ocrvqa | OCRVQA_TEST/TESTCORE | `Eval/OCRVQA/OCRVQA/data/` (test) |
| gqa | GQA_TestDev_Balanced | `Eval/GQA/GQA/testdev_balanced_*/` |

### 输出

- TSV 文件: `/data/lishichao/data/Optmerge/Eval/LMUData/{DatasetName}.tsv`
- 图片: `/data/lishichao/data/Optmerge/Eval/LMUData/images/{DatasetName}/`
- TSV 中 `image_path` 为绝对路径，不含 base64 图片数据

## 已知问题和修复记录

### 问题 1: numpy array 序列化

**现象**: TextVQA 的 answer 列存成了 numpy array 的 repr 字符串（`"['a' 'b']"`），而非 Python list 字符串（`"['a', 'b']"`）。

**原因**: HF parquet 中 answers 字段是 numpy array，`str()` 输出不含逗号。

**修复**: 转换后手动用正则提取并转为 Python list 字符串格式，VLMEvalKit 的 `process_line()` 使用 `eval()` 解析。

### 问题 2: OCRVQA flatten 失败

**现象**: OCRVQA 每行存了完整的 questions/answers 列表，而不是单个 QA 对。

**原因**: parquet 中 questions/answers 是 numpy array，`isinstance(x, list)` 返回 False，flatten 分支没走到。

**修复**: 添加 `hasattr(x, 'tolist')` 判断，先转为 Python list 再 flatten。已直接在环境中重新生成了 OCRVQA TSV。

### 问题 3: 与官方 TSV 样本不一致

详见 `eval_vlmevalkit.md` 中"与官方 TSV 的差异"章节。核心问题：
- TextVQA: HF val split 3333 条 vs 官方 ~5000 条
- OCRVQA_TESTCORE: 我们随机采样 vs 官方固定子集
- VizWiz: 我们用 val split vs 官方用 test split

**建议**: 下载官方 TSV 后可直接替换，不需要改代码。
