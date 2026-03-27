# Experiment Log: ETVD & TA Scaling Sweep + Variant Experiments

**Date**: 2026-03-22 ~ 2026-03-23
**Status**: PAUSED (进程已停止，可续)
**Stopped at**: 2026-03-23 ~03:00 UTC

---

## 1. 实验目标

对 InternVL2.5-1B 的 5 个 fine-tuned experts (OCR, VQA, Geometry, Chart, Grounding) 进行系统性的合并方法比较：

1. **TA (Task Arithmetic)** 不同 scaling coefficient: 0.1, 0.2, 0.3, 0.5
2. **ETVD (Ensemble Task Vector Decomposition)** 不同 scaling coefficient: 0.1, 0.2, 0.3, 0.5
3. **TA-norm / ETVD-norm**: 加入 per-layer norm matching 的变体
4. **ETVD-v2**: 只对 conflict 方向干预的改进版（共享方向保持 TA 全量求和）
5. **ta_no_geometry**: 去掉 Geometry 专家的 TA 合并（基于干扰分析）
6. **ta_weighted_v1**: 每个专家不同权重的 TA 合并

基线对比：wudi2 方法 (merge_ours_internvl) 和 InternVL2_5-1B 原始模型。

## 2. 实验设置

### 2.1 模型合并

**合并代码**:
- `Skills/theory/scaling_sweep.py` — TA/ETVD/norm 变体的批量 scaling sweep
- `Skills/theory/merge_variants.py` — ta_no_geometry, ta_weighted 等变体（CPU-only，避免 GPU 争用）
- `InternVL/internvl_chat/etvd_merge.py` — 核心合并算法实现

**合并模型保存位置**: `/data/shichao/data/InternVL_merged/`

| 模型名 | 方法 | Scaling | 专家 | 创建时间 |
|--------|------|---------|------|----------|
| ta_s01_all5 | Task Arithmetic | 0.1 | 5 experts | 03-22 17:31 |
| ta_s02_all5 | Task Arithmetic | 0.2 | 5 experts | 03-22 17:32 |
| ta_s03_all5 | Task Arithmetic | 0.3 | 5 experts | 03-22 17:32 |
| ta_s05_all5 | Task Arithmetic | 0.5 | 5 experts | 03-22 17:32 |
| etvd_s01_all5 | ETVD | 0.1 | 5 experts | 03-22 17:35 |
| etvd_s02_all5 | ETVD | 0.2 | 5 experts | 03-22 17:38 |
| etvd_s03_all5 | ETVD | 0.3 | 5 experts | 03-22 17:41 |
| etvd_s05_all5 | ETVD | 0.5 | 5 experts | 03-22 17:44 |
| etvd_norm_mean_s10_all5 | ETVD + norm matching | 1.0 | 5 experts | 03-22 17:49 |
| ta_norm_mean_s10_all5 | TA + norm matching | 1.0 | 5 experts | 03-22 20:55 |
| etvd_v2_s01_all5 | ETVD-v2 (conflict-only) | 0.1 | 5 experts | 03-23 00:06 |
| ta_no_geometry_s01 | Task Arithmetic | 0.1 | 4 experts (no Geo) | 03-23 00:34 |
| ta_no_geometry_s015 | Task Arithmetic | 0.15 | 4 experts (no Geo) | 03-23 00:37 |
| ta_no_geometry_s02 | Task Arithmetic | 0.2 | 4 experts (no Geo) | 03-23 00:39 |
| ta_weighted_v1 | Weighted TA | per-expert | 5 experts | 03-23 00:45 |
| etvd_norm_median_s10_all5 | ETVD + norm (median) | 1.0 | 5 experts | 03-22 17:56 |

**ta_weighted_v1 权重**: OCR=0.1, VQA=0.1, Geometry=0.05, Chart=0.15, Grounding=0.05

### 2.2 评测配置

**评测框架**: VLMEvalKit (`VLMEvalKit/`)
**模型注册**: `VLMEvalKit/vlmeval/config.py` 中添加对应 entry
**Benchmarks**: TextVQA_VAL, ChartQA_TEST, OCRVQA_TESTCORE, GQA_TestDev_Balanced, MathVista_MINI, MathVision_MINI, VizWiz
**Judge model**: gpt-4o-mini (用于 MathVista/MathVision)
**关键参数**: `--reuse` (支持断点续跑), `--verbose`, `--nproc-per-node=1`

**评测脚本**:
- `Skills/Evaluation/eval_sweep.sh` — GPU 1 上跑 TA scaling sweep
- `Skills/Evaluation/eval_sweep_textvqa.sh` — GPU 1 上 TextVQA 专项
- `Skills/Evaluation/eval_parallel_gpu1.sh` — GPU 1 并行评测
- `Skills/Evaluation/eval_etvd.sh` — GPU 2 上跑 ETVD
- `Skills/Evaluation/eval_gpu3_batch.sh` — GPU 3 批量评测
- `Skills/Evaluation/eval_new_models_gpu3.sh` — 新模型评测脚本

### 2.3 GPU 分配

| GPU | CUDA_VISIBLE_DEVICES | 模型 |
|-----|---------------------|------|
| GPU 0 | - | 其他用户 cali_x_mask.py (勿动!) |
| GPU 1 | 1 | ta_s01, ta_s02, ta_s03, ta_s05, ta_no_geometry_s01 (x2 重复) |
| GPU 2 | 2 | etvd_s01, etvd_s02, etvd_norm_mean, ta_norm_mean, etvd_v2_s01 |
| GPU 3 | 3 | etvd_s03, etvd_s05 |

## 3. 已完成结果

### 3.1 主要结果表

```
Model                             TextVQA  ChartQA   OCRVQA      GQA   VizWiz
------------------------------------------------------------------------------
InternVL2_5-1B (base)               74.23    69.72    41.76    54.69    29.01
merge_ours_internvl (wudi2)         76.01    68.44    46.29    57.14    30.88

TA s=0.1                            76.13    62.40    44.82      ---      ---
TA s=0.2                            74.07    64.00      ---      ---      ---
TA s=0.3                            74.06      ---    43.52      ---      ---
TA s=0.5                            74.02    63.92    43.65      ---      ---
TA s=1.0 (ta_merge_all5)            52.49      ---    27.12      ---    24.04

ETVD s=0.1                          74.06    64.36      ---      ---      ---
ETVD s=0.2                          73.75    65.04      ---      ---      ---
ETVD s=0.3                          73.94      ---      ---      ---      ---
ETVD s=0.5                          73.74      ---      ---      ---      ---
ETVD s=1.0 (etvd_merge_all5)        53.91      ---    27.60      ---    24.52

ETVD-norm (s=1.0)                   69.88    66.48      ---      ---      ---
TA-norm (s=1.0)                     70.08    66.64      ---      ---      ---
```

### 3.2 wudi2 Exclude-One 实验 (干扰分析)

```
Model                             TextVQA  ChartQA   OCRVQA      GQA   VizWiz  MathVista  MathVision
----------------------------------------------------------------------------------------------------
wudi2 all-5                         76.01    68.44    46.29    57.14    30.88     47.60      16.12
wudi2 -Geometry                     76.20    70.32    46.48    57.20    31.26     44.10      16.78
wudi2 -OCR                          75.49    70.60    44.63    57.10    30.78     47.00      20.72
wudi2 -VQA                          75.95    70.68    46.84    55.54    30.46     47.30      17.11
wudi2 -Chart                        75.92    62.96    46.74    56.89    31.10     46.40      16.78
wudi2 -Grounding                    76.00    69.24    46.52    57.24    31.04     47.00      16.78
```

## 4. 暂停时的进度快照 (2026-03-23 03:00 UTC)

### 4.1 正在进行的评测

| 模型 | GPU | 当前 benchmark | 进度 | 已完成 benchmarks |
|------|-----|---------------|------|-------------------|
| ta_s01_all5 | 1 | GQA_TestDev_Balanced | 4560/12578 (36%) | TextVQA, ChartQA, OCRVQA |
| ta_s02_all5 | 1 | OCRVQA→GQA(并行) | OCRVQA 120/3072, GQA 7400/12578 | TextVQA, ChartQA |
| ta_s03_all5 | 1 | ChartQA→GQA(并行) | ChartQA 150/2500, GQA 6440/12578 | TextVQA, OCRVQA |
| ta_s05_all5 | 1 | GQA_TestDev_Balanced | 3590/12578 (29%) | TextVQA, ChartQA, OCRVQA |
| etvd_s01_all5 | 2 | OCRVQA_TESTCORE | 1740/3072 (57%) | TextVQA, ChartQA |
| etvd_s02_all5 | 2 | OCRVQA_TESTCORE | 690/3072 (22%) | TextVQA, ChartQA |
| etvd_s03_all5 | 3 | ChartQA_TEST | 1710/2500 (68%) | TextVQA |
| etvd_s05_all5 | 3 | ChartQA_TEST | 1730/2500 (69%) | TextVQA |
| etvd_norm_mean | 2 | OCRVQA_TESTCORE | 980/3072 (32%) | TextVQA, ChartQA |
| ta_norm_mean | 2 | OCRVQA→开始 | 20/3072 (1%) | TextVQA, ChartQA |
| ta_no_geometry_s01 | 1 | TextVQA_VAL | ~1500/5000 (30%) | (无) |
| etvd_v2_s01_all5 | 2 | TextVQA_VAL | ~1400/5000 (28%) | (无) |

### 4.2 尚未开始评测的模型

- ta_no_geometry_s015
- ta_no_geometry_s02
- ta_weighted_v1
- etvd_norm_median_s10_all5

## 5. 关键发现 (截至暂停)

### 5.1 TA vs ETVD 的 TextVQA-ChartQA Trade-off

- **TA s=0.1 是 TextVQA 最佳** (+1.9 vs base), 但 ChartQA 大幅下降 (-7.3)
- **ETVD 在 ChartQA 上一致优于 TA** (64.36 vs 62.40 at s=0.1; 65.04 vs 64.00 at s=0.2)
- **ETVD 在 TextVQA 上接近 base** (~74), 低于 TA s=0.1 (76.13)
- **wudi2 最均衡**: TextVQA +1.78, ChartQA -1.28 — 任何 data-free 方法都未能同时做到

### 5.2 Per-layer Norm Matching 有害

- TA-norm: TextVQA 70.08 (-4.15), ChartQA 66.64 (-3.08)
- ETVD-norm: TextVQA 69.88 (-4.35), ChartQA 66.48 (-3.24)
- 结论：**均匀 norm 归一化破坏了层间 scaling 关系**，不可取

### 5.3 Cross-Expert 干扰

wudi2 exclude-one 实验揭示：
- **Geometry 干扰 Chart**: 去掉 Geometry 后 ChartQA 从 68.44 → 70.32 (+1.88)
- **OCR/VQA 也干扰 Chart**: 去掉 OCR → 70.60, 去掉 VQA → 70.68
- **Chart expert 本身是必要的**: 去掉 Chart → 62.96 (-5.48)
- 这些发现驱动了 ta_no_geometry 和 ta_weighted_v1 实验

### 5.4 Scaling 敏感性

- s=0.1 → 0.2 TextVQA 骤降 2 分（76.13 → 74.07）
- s=0.2 ~ 0.5 范围 TextVQA 基本持平（~74）
- s=1.0 灾难性（52-54）
- **最优窗口极窄**，主要在 s=0.1 附近

### 5.5 ETVD 理论分析

- 86.2% unique 方向, 5.6% shared, 8.2% conflict (threshold=0.3)
- VQA expert 有最大 task vector norm (27.3), Grounding 最小 (13.0)
- OCR-VQA 子空间重叠最高, Geometry 最孤立
- Energy ratio 1.092 确认 net over-accumulation

## 6. 如何续上实验

### 6.1 断点续跑命令

VLMEvalKit 的 `--reuse` 参数支持断点续跑。部分完成的推理结果保存在 PKL 文件中：
`outputs/<model>/T20260322_Gcddedc20/01_<benchmark>.pkl`

**续跑原则**: 使用相同的命令重新启动即可，`--reuse` 会自动跳过已完成的 benchmark 并从 PKL 断点继续。

### 6.2 推荐的续跑策略

**问题**: 之前同时跑了 14 个 eval 进程，40 核 CPU load 到 111，导致 GPU 利用率极低（0-11%）。另外 GPU 0 被 cali_x_mask.py 占用。

**推荐**: 每个 GPU 同时跑不超过 2 个 eval 进程。分批进行。

#### 批次 1: 完成最接近完成的任务 (GPU 2 + GPU 3)

```bash
export LMUData=/data/shichao/data/Merge_Evalkit/LMUData
source /home/shichao/miniconda3/etc/profile.d/conda.sh
conda activate eval-kit
cd /data/shichao/era-2026/MLLMerging/VLMEvalKit

# GPU 3: etvd_s03 和 etvd_s05 的 ChartQA 已经 68-69% 完成
export CUDA_VISIBLE_DEVICES=3
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
nohup torchrun --master-port $PORT --nproc-per-node=1 run.py \
  --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz \
  --model etvd_s03_all5 --verbose --reuse --judge gpt-4o-mini \
  > outputs/etvd_s03_resume.log 2>&1 &

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
nohup torchrun --master-port $PORT --nproc-per-node=1 run.py \
  --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz \
  --model etvd_s05_all5 --verbose --reuse --judge gpt-4o-mini \
  > outputs/etvd_s05_resume.log 2>&1 &
```

#### 批次 2: 关键新模型 (GPU 2)

```bash
export CUDA_VISIBLE_DEVICES=2
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
nohup torchrun --master-port $PORT --nproc-per-node=1 run.py \
  --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE \
  --model ta_no_geometry_s01 --verbose --reuse \
  > outputs/ta_no_geometry_s01_resume.log 2>&1 &

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
nohup torchrun --master-port $PORT --nproc-per-node=1 run.py \
  --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE \
  --model etvd_v2_s01_all5 --verbose --reuse \
  > outputs/etvd_v2_resume.log 2>&1 &
```

#### 批次 3: TA scaling sweep 剩余 benchmarks (GPU 1)

```bash
export CUDA_VISIBLE_DEVICES=1
# 逐个跑，避免过载
for model in ta_s01_all5 ta_s02_all5 ta_s03_all5 ta_s05_all5; do
  PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
  torchrun --master-port $PORT --nproc-per-node=1 run.py \
    --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE GQA_TestDev_Balanced MathVista_MINI MathVision_MINI VizWiz \
    --model $model --verbose --reuse --judge gpt-4o-mini
done
```

#### 批次 4: 尚未评测的新模型

```bash
# 等批次 2 完成后
for model in ta_no_geometry_s015 ta_no_geometry_s02 ta_weighted_v1; do
  PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
  torchrun --master-port $PORT --nproc-per-node=1 run.py \
    --data TextVQA_VAL ChartQA_TEST OCRVQA_TESTCORE \
    --model $model --verbose --reuse
done
```

### 6.3 注意事项

1. **GPU 0 不要用** — 有别人的 cali_x_mask.py 在跑
2. **每 GPU 最多 2 个 eval 进程**，否则 CPU 会成为瓶颈
3. **ta_no_geometry_s01 之前有重复进程** (两组 torchrun worker)，续跑时只需启动一个
4. 所有合并好的模型已保存在 `/data/shichao/data/InternVL_merged/`，不需要重新合并
5. 模型已在 `VLMEvalKit/vlmeval/config.py` 中注册

## 7. 相关文件索引

| 文件 | 说明 |
|------|------|
| `Skills/theory/etvd_merge.py` | ETVD/TA/norm 核心合并算法 |
| `Skills/theory/scaling_sweep.py` | 批量 scaling 合并脚本 |
| `Skills/theory/merge_variants.py` | ta_no_geometry/weighted 合并脚本 (CPU-only) |
| `Skills/theory/etvd_analysis.py` | ETVD 理论分析脚本 |
| `Skills/theory/results/analysis_report.md` | ETVD 分析报告 (含 8 张图) |
| `Skills/theory/ensemble_task_vector_decomposition.md` | ETVD 理论推导 |
| `VLMEvalKit/vlmeval/config.py` | 模型注册 (已添加所有新模型) |
| `VLMEvalKit/results.py` | 结果汇总脚本 |
| `Skills/Evaluation/eval_*.sh` | 各评测启动脚本 |
