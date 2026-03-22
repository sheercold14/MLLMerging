# Qwen + VLMEvalKit 排查经验

## 背景

- 目标：复现论文 `OptMerge: Unifying Multimodal LLM Capabilities and Modalities via Model Merging` 中 Qwen2-VL 相关评测。
- 关注模型：
  - `/data/lishichao/data/model/Qwen2-VL-7B-OCR`
  - `/data/lishichao/data/model/Qwen_merged/Qwen_merged_all`
- 评测框架：
  - `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit`
- 重点输出目录：
  - `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/Qwen2-VL-7B-OCR/T20260317_G8f205d4e`

## 先确认的事实

- 当前讨论对应的是官方 TSV，不是自制 TSV。
- `Qwen2-VL-7B-OCR` 是论文 Table 3 对应的 OCR 专项模型。
- 论文 Table 3 中 `Individual OCR` 的 Qwen2-VL 指标：
  - `VizWiz`: `28.38`
  - `GQA (test)`: `37.53`
  - `MathVista (mini)`: `31.81`
  - `MATH-Vision (mini)`: `13.16`
  - `ChartQA (test)`: `57.40`
  - `TextVQA (val)`: `70.50`
  - `OCRVQA (test)`: `64.68`

## 当前复现现象

- `Qwen2-VL-7B-OCR` 的下列任务基本对齐论文：
  - `GQA`: `39.22` vs `37.53`
  - `TextVQA`: `72.63` vs `70.50`
  - `OCRVQA_TESTCORE`: `64.13` vs `64.68`
- 但两项异常严重：
  - `VizWiz`: `0.64` vs `28.38`
  - `ChartQA`: `2.64` vs `57.40`

这说明：

- 模型大概率没有下错。
- 整体推理链也不是全错。
- 问题大概率集中在 `VizWiz` 和 `ChartQA` 的评测调用方式。

## 模板和模型文件结论

- `Qwen2-VL-7B-OCR` 的实际 chat template 文件是：
  - `/data/lishichao/data/model/Qwen2-VL-7B-OCR/chat_template.json`
- `merge_all` 当前 instruct 模板文件是：
  - `/data/lishichao/data/model/Qwen_merged/Qwen_merged_all/chat_template.json`
- `merge_all` 目录里也保留了 base 模板：
  - `/data/lishichao/data/model/Qwen_merged/Qwen_merged_all/chat_template_base.json`

注意：

- `tokenizer_config.json` 里可能仍看到 base 风格模板。
- 但 `Qwen2VLProcessor` 实际更可能读取模型目录里的独立 `chat_template.json`。
- 因此判断模板是否生效，不能只看 `tokenizer_config.json`。

## 为什么不能先怪 merge

- `Qwen2-VL-7B-OCR` 是现成论文模型，不是 merge 产物。
- 它在 `GQA/TextVQA/OCRVQA` 上已接近论文。
- 所以如果只有 `VizWiz/ChartQA` 崩，不应优先怀疑 merge 代码。

正确顺序应是：

1. 先对齐 OCR 模型。
2. OCR 基线稳定后，再看 `merge_all`。

## 直接观察到的异常输出

### VizWiz

在 `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/Qwen2-VL-7B-OCR/T20260317_G8f205d4e/Qwen2-VL-7B-OCR_VizWiz.xlsx` 中，出现大量无意义短串：

- `xps`
- `kpix`
- `kpis`
- `kips`
- `kyps`
- `kxpy`
- `kxwv`

统计特征：

- `alpha_only<=6` 的超短英文串约 `1929 / 4319`
- 明确统计的典型异常串约 `356 / 4319`

这类输出不像正常 VQA，更像 OCR 偏置下对局部噪声字符的误读。

### ChartQA

在 `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/outputs/Qwen2-VL-7B-OCR/T20260317_G8f205d4e/Qwen2-VL-7B-OCR_ChartQA_TEST.xlsx` 中，出现明显题型错位：

- 数值题回答 `Yes.` / `No.`
- 是非题回答数字或百分比
- 个别样本输出异常长小数串

典型例子：

- `How many bars are shown in the chart? -> No.`
- `What percent ... as Dangerous? -> Yes.`
- `Is the sum value of Madagascar more then Fiji? -> 23`

这不是普通误差，而是模型没有稳定遵守 `ChartQA` 的回答类型。

## 真正的可疑点：VLMEvalKit 的 prompt 路径

关键代码：

- 推理入口：
  - `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/vlmeval/inference.py`
- Qwen2-VL prompt：
  - `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/vlmeval/vlm/qwen2_vl/prompt.py`
- VQA 数据集定义：
  - `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/vlmeval/dataset/image_vqa.py`

行为链路：

1. `inference.py` 中，如果模型 `use_custom_prompt(dataset_name)` 返回 `True`，则优先调用模型自己的 `build_prompt(...)`。
2. 对 `Qwen2VLChat`，`prompt.py` 里几乎所有 `VQA` 数据集都会返回 `True`。
3. 这导致 `ChartQA` 和 `VizWiz` 不走数据集层的 prompt，而是走 Qwen2-VL 的通用 VQA prompt。

当前 Qwen2-VL 的通用 VQA prompt 是：

- `Please try to answer the question with short words or phrases if possible.`

这句对通用 Instruct 模型通常还能工作，但对 OCR 专项模型不够强。

## 为什么 Instruct 指标正常

`Qwen2-VL-Instruct` 是通用指令对齐模型，对“粗糙 prompt”更鲁棒：

- 更会按问题类型答题
- 更容易在不确定时避免乱码式输出
- 对 `VizWiz` 和 `ChartQA` 这种任务泛化更强

而 OCR 专项模型更可能出现：

- 看到模糊字符就强行读文本
- 对题型约束不敏感

所以同一套 prompt 下：

- `Instruct` 正常
- `OCR` 在 `VizWiz/ChartQA` 容易出问题

这不说明 OCR 模型错，而说明 OCR 模型对 prompt 更敏感。

## 目前最合理的判断

- 论文没有充分公开 `VizWiz` 和 `ChartQA` 的评测 prompt 细节。
- 当前 `VLMEvalKit -> Qwen2VLChat` 的默认 prompt 路径，很可能不是论文当时那条完整路径。
- 对多数任务影响不大。
- 但对 `VizWiz` 和 `ChartQA` 影响极大。

## 已做过的代码改动

为了方便 A/B 验证，已在下列文件加入显式 `chat_template` 覆盖能力：

- `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/vlmeval/vlm/qwen2_vl/model.py`
- `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/vlmeval/config.py`

新增的模型别名：

- `merge_all_base_template`
- `merge_all_instruct_template`
- `Qwen2-VL-7B-OCR-base-template`
- `Qwen2-VL-7B-OCR-instruct-template`

这部分主要用于后续对照实验，不代表模板就是唯一根因。

## 最建议的后续动作

### 1. 不先看 merge，先只盯 OCR

只围绕 `Qwen2-VL-7B-OCR` 定位问题，避免混淆：

- `VizWiz`
- `ChartQA`

### 2. 直接检查 prompt 是否需要特化

最值得修改的位置：

- `/data/lishichao/project/era-2026/MLLMerging/VLMEvalKit/vlmeval/vlm/qwen2_vl/prompt.py`

建议考虑新增两个数据集专门 prompt：

- `ChartQA_TEST`
  - 明确要求只输出最终答案
  - 数值题只输出数字
  - 是非题只输出 `Yes` 或 `No`
  - 不要解释
- `VizWiz`
  - 明确要求看不清或无法判断时输出 `unanswerable`
  - 不要猜测模糊字符碎片
  - 其余情况给简短自然答案

### 3. 只复跑两个最敏感数据集

优先复跑：

- `ChartQA_TEST`
- `VizWiz`

不要一开始就全量重跑。

### 4. 观察是否出现以下改善

- `VizWiz` 中 `xps/kpix/kips/...` 明显下降
- `ChartQA` 中题型错位明显下降
- 分数向论文值靠近：
  - `VizWiz -> 28.38`
  - `ChartQA -> 57.40`

## 简短结论

这次排查最重要的结论不是“模型错了”，而是：

- OCR 模型本身大概率没问题。
- 论文结果和当前 `VLMEvalKit` 默认 Qwen2-VL prompt 路径之间，很可能缺了 `VizWiz` / `ChartQA` 的任务特化设置。
- 先把 OCR 的这两个数据集对齐，再分析 `merge_all` 才有意义。
