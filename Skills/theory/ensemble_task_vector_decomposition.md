# Ensemble Task Vector Decomposition: A Unified Framework for MLLM Merging

> 讨论记录 | 2026-03-22 | OptMerge 理论构建

---

## 1. 问题背景

### 1.1 实验设置

基于 InternVL2.5-1B，有 5 个 fine-tuned experts：

| Expert | 来源 | 目标能力 |
|--------|------|----------|
| OCR | `yongxianwei/InternVL2_5-1B_OCR` | 文字识别 |
| VQA | `yongxianwei/InternVL2_5-1B_VQA` | 视觉问答 |
| Geometry | `yongxianwei/InternVL2_5-1B_Geometry` | 几何推理 |
| Chart | `yongxianwei/InternVL2_5-1B_Chart` | 图表理解 |
| Grounding | `yongxianwei/InternVL2_5-1B_Grounding` | 视觉定位 |

评估 7 个 benchmarks：TextVQA, OCRVQA, VizWiz, GQA, ChartQA, MathVision, MathVista。

当前主要 merge 方法为 `wudi2`（SVD 分解 + 梯度优化干扰最小化的 merging vector）。

### 1.2 实验结果

| Model | TextVQA | OCRVQA | VizWiz | GQA | ChartQA | MathVision | MathVista |
|---|---|---|---|---|---|---|---|
| **Base (InternVL2.5-1B)** | 74.23 | 41.76 | 29.01 | 54.69 | 69.72 | 17.43 | 46.40 |
| **All 5 experts (wudi2)** | **76.01** | **46.29** | **30.88** | **57.14** | 68.44 ↓ | 16.12 ↓ | **47.60** |
| exclude_chart | 75.92 | 46.74 | 31.10 | 56.89 | 62.96 ↓↓ | 16.78 | 46.40 |
| exclude_geometry | 76.20 | 46.48 | 31.26 | 57.20 | **70.32** | 16.78 | 44.10 ↓ |
| exclude_grounding | 76.00 | 46.52 | 31.04 | 57.24 | 69.24 | 16.78 | 47.00 |
| exclude_ocr | 75.49 | 44.63 ↓ | 30.78 | 57.10 | **70.60** | **20.72** ↑↑ | 47.00 |
| exclude_vqa | 75.95 | **46.84** | 30.46 | 55.54 ↓ | **70.68** | 17.11 | 47.30 |

**Sequential add-back 实验**（merge 4 experts 后以 merged model 为 base 再 add 1 expert）：

| Model | TextVQA | OCRVQA | MathVision | MathVista |
|---|---|---|---|---|
| exclude_chart → add_chart | 67.93 ↓↓ | 40.63 | 14.47 ↓ | 39.10 ↓↓ |
| exclude_add_chart | 55.13 ↓↓↓ | 35.94 ↓ | 13.49 ↓ | 36.20 ↓↓ |
| exclude_add_vqa | 64.75 ↓↓ | 36.88 ↓ | 15.13 | 34.10 ↓↓ |

### 1.3 关键观察

1. **OCR ↔ MathVision 负干扰**：去掉 OCR 后 MathVision 从 16.12→20.72 (+4.6)，OCR task vector 与 Math 推理方向存在显著冲突
2. **Chart ↔ ChartQA 正依赖但有溢出**：去掉 Chart 后 ChartQA 暴跌至 62.96，但全部合并时 ChartQA 也只有 68.44（低于 base 的 69.72），其他 expert 的 task vector 干扰了 Chart 的信号
3. **VQA ↔ GQA 强关联**：去掉 VQA 后 GQA 降为 55.54
4. **Geometry 对 MathVista 有贡献**：去掉后 MathVista 从 47.6→44.1
5. **Grounding 近似中性**：去掉 Grounding 几乎无变化，说明其 task vector 与其他任务正交
6. **串行 add-back 灾难性退化**：两步 merge（4个→再加1个）远差于一步 merge 5个

核心问题：**不同 expert 之间存在复杂的图依赖关系（协同/冲突/中性），是否存在基于第一性原理的方法，通过分析 task vector 的结构来预测和控制这些依赖？**

---

## 2. 干扰量化方法的层次分析

### 2.1 三个量化层次

#### 层次一：参数空间度量（无需数据，纯几何）

**1a. 逐层 Cosine Similarity Gram 矩阵**

```
G^l_{ij} = cos(τ_i^l, τ_j^l)
```

每层一个 N×N 矩阵。高 cosine → 方向重叠 → 叠加时该方向被放大。

局限：cosine 不区分"协同"和"冲突"——两个方向一致可能是共同增强某能力，也可能是重复叠加导致 over-scaling。

**1b. Magnitude-weighted Sign Conflict Rate**

```
Conflict(i,j) = Σ_p |τ_i(p)·τ_j(p)| · 1[sign(τ_i(p)) ≠ sign(τ_j(p))]
Synergy(i,j)  = Σ_p |τ_i(p)·τ_j(p)| · 1[sign(τ_i(p)) = sign(τ_j(p))]
```

比 TIES 的简单 sign vote 更精细——大 magnitude 参数的符号冲突影响更大。

**1c. Principal Angle Analysis（子空间重叠度）**

对每个 expert 的 task vector 矩阵做 SVD 得到主子空间，计算两个子空间间的 principal angles：

```
cos(θ_k) = σ_k(U_i^T · U_j)
```

Principal angle 接近 0 → 子空间高度重叠 → 相互干扰的结构性基础。

**共同局限**：这些度量都是参数空间的几何关系，完全不知道下游 task 的 loss landscape。两个 cosine=0.9 的 task vector，可能对某个 benchmark 是协同的，对另一个是冲突的——参数空间的欧氏距离/角度和功能空间的影响之间存在非线性映射。

#### 层次二：梯度-Task Vector 对齐（需少量数据，一阶预测）

对 benchmark t，base model 参数 θ₀，应用 task vector τᵢ 后的 loss 变化的一阶 Taylor 展开：

```
L_t(θ₀ + τᵢ) ≈ L_t(θ₀) + ⟨∇_θ L_t(θ₀), τᵢ⟩
```

定义 Task-Expert Alignment：

```
A(t, i) = -⟨∇_θ L_t(θ₀), τᵢ⟩
```

- A(t,i) > 0 → expert i 减小 task t 的 loss（有益）
- A(t,i) < 0 → expert i 增大 task t 的 loss（有害）

给出 T×N 的 alignment 矩阵（T 个 benchmarks × N 个 experts），直接预测每个 expert 对每个 task 的影响方向和强度。

**Pairwise 干扰的二阶分析**（同时应用 τᵢ + τⱼ 时）：

```
L_t(θ₀ + τᵢ + τⱼ) ≈ L_t(θ₀) + ⟨∇L_t, τᵢ+τⱼ⟩ + ½(τᵢ+τⱼ)ᵀ H_t (τᵢ+τⱼ)
```

交叉项 `τᵢᵀ H_t τⱼ` 是二阶干扰项。可用 Fisher Information Matrix 对角近似：

```
F_t(p) = E_x[(∂L_t/∂θ_p)²]
Interference(t, i, j) ≈ Σ_p F_t(p) · τᵢ(p) · τⱼ(p)
```

在 task t 敏感的参数位置上，τᵢ 和 τⱼ 同号 → 叠加放大（可能 over-shoot），异号 → 相互抵消（信号丢失）。

计算成本：每个 benchmark 采样 ~100-200 条样本，在 base model 上做一次 forward+backward，得到梯度后与 task vectors 做内积。总成本 ≈ 7 次 forward-backward pass（7 个 benchmark）。

#### 层次三：Activation-Space 功能干扰（最精确，成本最高）

**CKA（Centered Kernel Alignment）**：对同一组 probe 输入，比较 base+τᵢ 和 base+τⱼ 在中间层的 representation similarity。

**Cross-Expert Activation Interference**：逐层度量 expert 引起的 activation shift。

成本：O(N×T) 次 forward pass。

### 2.2 层次对比

| 方法 | 理论预测力 | 计算成本 | Task-aware | 可推广性 |
|------|-----------|---------|------------|---------|
| 参数空间 Gram 矩阵 | 低 | 极低 | 否 | 高 |
| 梯度-TaskVector 对齐 | 中-高 | 低 | 是 | 高 |
| Fisher 加权干扰 | 高 | 中 | 是 | 高 |
| Activation CKA | 高 | 高 | 是 | 中 |

### 2.3 Data-free vs Data-dependent 的策略抉择

**方案 A：坚持 Data-free 路线**

优势：
- 干净的比较对手群：Task Arithmetic, TIES, DARE, TSV-Merge, SVC
- 更强的实用性叙事（不需要 validation data）
- wudi2 已经在这个阵营

需要回答的审稿人问题：
- "你的干扰度量和 TSV 的 STI 有什么区别？"
- "和 SVC 比，你是否也考虑了 spectral over-accumulation？"

**方案 B：引入少量数据**

优势：
- 梯度对齐给出 task-aware 的干扰度量，预测力更强
- 可以做 per-expert per-layer 的系数优化

风险：
- 需要和 AdaMerging (ICLR 2024), ESM, RegMean, Fisher Merging 比较
- 审稿人会问"你用了数据为什么不直接 multi-task fine-tune？"
- 叙事变复杂，contribution 边界模糊

**结论：方案 A 更稳健。** 2025-2026 年这个领域的趋势清晰——TSV、SVC、MDA 都在证明纯参数空间的结构性分析（SVD/奇异向量视角）是一条富矿。

---

## 3. 现有方法论图景

### 3.1 TSV — Task Singular Vectors (CVPR 2025, Gargiulo et al.)

- **论文**: [arxiv.org/abs/2412.00081](https://arxiv.org/abs/2412.00081)
- **代码**: [github.com/AntoAndGar/task_singular_vectors](https://github.com/AntoAndGar/task_singular_vectors)

核心贡献：定义了 Singular Task Interference (STI) 度量：

```
STI({Δᵢ}) = ||(UᵀU - I) Σ (VᵀV - I)||₁
```

其中 U, V 是各 task vector 逐层做 SVD 后奇异向量的拼接矩阵。UᵀU → I（奇异向量空间正交）时 STI → 0，无干扰。

关键发现：task matrices 天然低秩（保留 10% 奇异分量即可保持 99% 精度）。

方法：TSV-Compress（低秩压缩）+ TSV-Merge（白化变换去相关）。

局限：
- STI 是把所有 task 拼接后算一个 global scalar，丢失了 pairwise 结构
- 白化是无差别去相关，不区分"有益的共享"和"有害的重叠"
- 完全 data-free

### 3.2 SVC — Singular Value Calibration (arXiv 2602.05536, 南京大学, 2026)

- **论文**: [arxiv.org/abs/2602.05536](https://arxiv.org/abs/2602.05536)
- **代码**: [github.com/lyymuwu/SVC](https://github.com/lyymuwu/SVC)

核心发现：Spectral Over-Accumulation——当多个 task vectors 共享对齐的奇异方向时，线性叠加重复累积这些方向，导致奇异值膨胀，merged model 偏向共享子空间而丧失 task-specific 能力。

方法：data-free 后处理——量化子空间重叠度，缩放膨胀的奇异值恢复平衡谱。仅修改奇异值即让 Task Arithmetic 提升 13%。

与我们实验现象的对应：5 expert 合并后 ChartQA 下降、MathVision 下降——大概率是 OCR/VQA 等 expert 共享方向的 over-accumulation 淹没了 Chart/Math 的 task-specific 信号。

局限：
- 检测到某个奇异值膨胀了，但不知道是哪些 expert 贡献的
- 不知道这个方向对哪些 downstream task 重要
- 无法回答"这个共享方向是 beneficial sharing 还是 harmful accumulation？"
- 完全 data-free

### 3.3 ESM — Essential Subspace Merging (arXiv 2602.20208, 东南大学+华为, 2026)

- **论文**: [arxiv.org/abs/2602.20208](https://arxiv.org/abs/2602.20208)

对 feature shifts（activation 变化）做 PCA 找 essential subspace，再把 task vector 投影到这个子空间做低秩分解后合并。加入 multi-level polarized scaling 策略放大关键知识、抑制冗余参数。

关键区别：用 activation 数据定义"essential"而非纯参数几何。

局限：
- 需要 forward pass 数据，属于 data-dependent
- 每个 task 的 essential subspace 独立定义，不直接建模 inter-task 子空间关系

### 3.4 MDA / DC-Merge (CVPR 2026)

- **相关论文**: [arxiv.org/abs/2512.00391](https://arxiv.org/abs/2512.00391)（MDA: From Coefficients to Directions）

基于 Neural Collapse 的 ETF (Equiangular Tight Frame) 几何框架，在参数空间和特征空间同时做方向对齐。

核心思想：fine-tuned model 的 classifier weights 和 class-mean features 趋向 ETF 结构（Neural Collapse 现象），合并时应维持这种方向一致性。

局限：
- ETF 理论源自分类任务，是否适用于 MLLM 的 generation 任务存疑
- 对齐目标是外部施加的几何结构，而非从 task vectors 本身自然推导

### 3.5 其他相关工作

- **AdaMerging** (ICLR 2024)：学习 per-task per-layer 合并系数，data-dependent
- **TIES-Merging** (NeurIPS 2023)：trim + elect sign + disjoint merge
- **DARE** (2024)：random drop + rescale task vector 参数
- **AdaRank** (ICLR 2026)：自适应秩选择
- **Orthogonal Model Merging** (2026)

---

## 4. 核心 Insight：统一的三类 Failure Mode

### 4.1 问题的本质

现有方法对"干扰"的理解是碎片化的：

- **TSV** 问："各 expert 的奇异向量空间是否正交？" → 度量 non-orthogonality
- **SVC** 问："共享方向的奇异值是否被叠加膨胀了？" → 度量 over-accumulation

但这两个问题是**同一个底层现象的两个侧面**。问题的根源是：

> **当 N 个 task vectors 线性叠加时，merged vector 的奇异值谱相对于每个 individual task 需要的谱产生了系统性畸变——共享方向被放大 k 倍（k 为该方向的 occupancy），unique 方向保持 1 倍，两者的相对权重被扭曲。**

### 4.2 三类 Failure Mode

Naive sum（如 Task Arithmetic）不区分"一个方向被多少个 expert 占据"，导致三类 failure mode：

| Failure Mode | 机制 | 现有方法的处理 |
|---|---|---|
| **Over-accumulation** | 多个 expert 共享的方向被叠加放大 k 倍 | SVC 发现并缩放 |
| **Under-representation** | Expert 独有方向的相对权重被共享方向的放大所淹没 | 无人显式处理 |
| **Sign conflict** | Expert 间在同一方向上符号相反，相互抵消 | TIES/TSV 部分处理 |

**这三个 failure mode 是统一的**：都是因为 naive sum 没有利用"各方向的 occupancy 结构"。

---

## 5. 提出的理论框架：Ensemble Task Vector Decomposition (ETVD)

### 5.1 核心操作：对 Task Vector Ensemble 做联合 SVD

区别于所有现有方法（先对各 task vector 独立 SVD 再分析交互），我们对整个 task vector ensemble 做联合分解。

将 N 个 task vectors 逐层堆叠成矩阵 T ∈ ℝ^{N×d}，直接对 T 做 SVD：

```
T = U Σ Vᵀ
```

- **V 的列**：task vector 空间的自然正交基底（不是某个 individual task 的基底，而是所有 task 共同张成的空间的正交基）
- **U 的列** u_k：N 个 expert 在基底方向 v_k 上的坐标/投影系数
- **Σ 的对角元** σ_k：各基底方向的能量/重要性

### 5.2 从 U 矩阵直接读出三类结构

| U 的第 k 列的结构 | 含义 | 合并策略 |
|---|---|---|
| 只有 1 个元素显著 | **Unique direction**：仅属于某个 expert | 完整保留 |
| 多个元素同号 | **Shared beneficial**：多个 expert 的共识方向 | 保留但 normalize by occupancy（计一次而非 k 次） |
| 多个元素异号 | **Conflict direction**：expert 间的对抗方向 | Resolve（majority vote / 抑制） |

**Naive sum（Task Arithmetic）等价于直接取 Σᵢ uᵢ 作为合并系数，忽略了 occupancy 信息。**

### 5.3 Occupancy-Aware Spectrum Reconstruction

合并规则：

```
τ_merged^l = Σ_k  w_k · σ_k · v_k
```

其中 w_k 由 u_k 的结构决定：

```python
for each basis direction k:
    occupancy_k = count(|u_{ik}| > threshold for i in 1..N)
    sign_consistency_k = sign(Σ_i u_{ik})

    if occupancy_k == 1:
        # Unique direction: preserve at full strength
        w_k = u_{ik}  (the single significant entry)
    elif sign_consistent:
        # Shared beneficial: normalize to prevent over-accumulation
        w_k = mean(u_{ik} for significant i)  # or sum / occupancy
    else:
        # Conflict: resolve by majority or suppress
        w_k = sign_consistency_k * mean(|u_{ik}| for agreeing i)
```

### 5.4 与各方法的理论对比

| | 分解对象 | 基底来源 | 知道"谁占据了哪个方向" | 区分 shared/unique/conflict | Data-free |
|---|---|---|---|---|---|
| **TSV** | 各 task 独立 SVD | 每个 task 自己的 | 否（只知道空间重叠度） | 否 | 是 |
| **SVC** | 合并后的 merged vector SVD | merged vector 的 | 否（只看奇异值是否膨胀） | 否 | 是 |
| **ESM** | 各 task 的 activation PCA | activation feature shift 的 | 是（per-task essential subspace） | 部分（essential vs non-essential） | 否 |
| **MDA** | 对齐到外部 ETF | 外部施加的 | N/A | N/A | 部分 |
| **ETVD（ours）** | 所有 task vectors 联合 SVD | task vector 空间的自然正交基 | **是（U 矩阵直接给出）** | **是（通过 U 的稀疏性和符号）** | **是** |

### 5.5 核心理论区分的一句话表述

> 现有方法要么分析个体再比较交互（TSV/SVC），要么引入外部目标做对齐（MDA），而 ETVD 从 task vector ensemble 的联合结构中自然导出基底、occupancy 和冲突模式，用一个统一的分解解释 over-accumulation、under-representation 和 sign conflict 三类 failure mode。

---

## 6. 框架能力边界分析

### 6.1 Pairwise 依赖：完整刻画

两个 expert 的关系完全由它们在 U 中的坐标向量决定。对于基底方向 v_k：

```
u_{ik} 和 u_{jk} 同号且大 → 该方向上 i,j 协同（合并时 over-accumulate）
u_{ik} 和 u_{jk} 异号且大 → 该方向上 i,j 冲突（合并时 cancel）
一个大一个小            → 该方向是某个 expert 的 unique 方向
```

所有 pairwise 关系浓缩在 Gram 矩阵 G = UΣ²Uᵀ = TTᵀ 中。

### 6.2 多 Expert 交互：线性假设下可分解

在线性框架下，合并 experts {1,2,3} 在基底方向 v_k 上的效果是：

```
effect_k = (u_{1k} + u_{2k} + u_{3k}) · σ_k
```

具体例子——假设某方向 v_k 上：

```
OCR:       u_{OCR,k}  = +0.6
VQA:       u_{VQA,k}  = +0.5
Geometry:  u_{Geo,k}  = -0.8
```

- 只合并 OCR+VQA: effect = 1.1σ_k（过度放大）
- 加入 Geometry: effect = 0.3σ_k（被 Geometry 大幅抵消）
- 只合并 VQA+Geometry: effect = -0.3σ_k（方向反转）

从 ablation 实验看，"加入 Geometry 后 OCR benchmark 下降"这种现象，看起来像三体交互，但实际上由 U 矩阵中三个 expert 在该方向上的坐标完全解释。

**在 task arithmetic 的线性框架下，任意 k-expert 交互都可以精确分解为各 expert 在共同基底上的坐标叠加，不存在真正的高阶项。Ensemble SVD 的 U 矩阵给出了完备的描述。**

### 6.3 根本性局限：参数空间线性 ≠ 功能空间线性

神经网络是参数的非线性函数：

```
L(θ₀ + Σᵢτᵢ) = L(θ₀) + Σᵢ∇L·τᵢ + ½Σᵢⱼ τᵢᵀHτⱼ + O(τ³)
```

二阶项 τᵢᵀHτⱼ 是 Hessian 加权的 pairwise 交互，三阶项是真正的三体交互。这些不是 U 矩阵能捕捉的。

**但这个局限是所有 data-free 参数空间方法共有的**——TSV、SVC、TIES、DARE、Task Arithmetic 全部隐式假设了线性叠加。

### 6.4 Sequential merge 退化的解释

4-expert merge 再 add 1 的灾难性退化，原因不是"高阶干扰"，而是 task vector 语义污染：

```
τ_VQA_new = θ_VQA_finetuned - θ_merged_4
           = (θ_base + τ_VQA_original) - (θ_base + δ_merged_4)
           = τ_VQA_original - δ_merged_4
```

第二步的 "task vector" 不再是纯粹的 VQA 能力增量，而是混入了 4-expert merged delta 的负像。优化方法在这个 contaminated vector 上做优化，结果崩坏。这是操作顺序导致的 task vector 语义污染，和高阶干扰是不同的问题。

### 6.5 能力边界总结

| 能建模 | 不能建模 | 但这个局限是... |
|---|---|---|
| 两两依赖的方向和强度 | 功能空间的非线性交互 | 所有 data-free 方法共有的 |
| 任意子集合并的线性效果预测 | Hessian 加权的真实 pairwise 影响 | 需要 data 才能解决 |
| Over-accumulation / under-representation / sign conflict 的统一机制 | 优化方法的路径依赖性 | 优化方法本身的问题 |

---

## 7. 当前 wudi2 方法在此框架中的位置

### 7.1 wudi2 的核心操作

```python
# 对每个 2D 权重层:
# 1. 各 expert task vector 做 SVD，保留低秩部分
# 2. 以 sum 为初始点，优化 merging vector 最小化:
loss = Σ_i || (m - τ̃_i)ᵀ · V_i ||² / ||τ_i||²
```

### 7.2 与 ETVD 框架的对比

| 特性 | TSV | SVC | wudi2 | 缺什么 |
|------|-----|-----|-------|--------|
| 低秩 SVD 分解 | 是 | 是 | 是 | - |
| 干扰度量 | STI 公式化 | 子空间重叠度 | 隐式（loss 函数） | **无显式度量** |
| Over-accumulation 处理 | 无 | 奇异值校准 | 无 | **缺失** |
| 优化方式 | 白化变换 | 后处理缩放 | 梯度优化 | 优化目标可改进 |

wudi2 的独特之处：通过梯度优化找到干扰最小的 merging vector，比 TSV 的白化和 SVC 的后处理更灵活。

wudi2 的缺失：
1. 没有显式量化干扰——优化在参数级别进行，不知道干扰的结构化来源
2. 没有处理 over-accumulation——优化起点（sum of vectors）已经 over-accumulated
3. 合并系数不是 per-expert 的——所有 expert 等权叠加后优化一个向量

---

## 8. 研究路线

### 8.1 Paper 定位

> Data-free 框架下，基于 SVD 子空间结构的干扰感知合并：
> 1. 用 ensemble SVD + occupancy 分析 → 构造显式干扰分解（诊断工具，统一解释三类 failure mode）
> 2. 基于分解结构 → 设计 occupancy-aware spectrum reconstruction 策略
> 3. 理论贡献：证明 over-accumulation 和 under-representation 是合并退化的两个独立 failure mode，并提供统一的处理框架

相对于现有工作的定位：
- 比 TSV 更进一步（pairwise + occupancy structure + unified explanation）
- 比 SVC 更深（不仅是后处理校准，而是结构化分解）
- 保持 data-free（不需要和 AdaMerging/ESM 竞争）
- 在 MLLM 场景下验证（TSV/SVC 还没覆盖的）

### 8.2 潜在理论贡献

Occupancy-aware reconstruction 可以被证明在某种意义上是最优的：如果假设每个 task vector 对 merged model 的贡献应该是等权的，那么 occupancy normalization 是唯一满足"每个基底方向的总贡献与占据它的 expert 数量无关"这个公平性约束的线性合并策略。

### 8.3 未来扩展方向

**路径 A：逐层传播分析（Data-free）**

网络是 f = f_L ∘ ... ∘ f₁ 的复合。即使每层的 task vector 是线性叠加，层间的复合引入了非线性。逐层分析 U 矩阵的结构变化——如果某对 expert 在浅层正交但在深层共线，说明干扰在深层累积放大。

**路径 B：二阶近似（Data-dependent，留作 future work）**

用少量数据 + Hessian-vector product 估算 τᵢᵀHτⱼ，量化功能空间的真实 pairwise 干扰。

---

## 9. 参考文献

- [Task Singular Vectors: Reducing Task Interference in Model Merging](https://arxiv.org/abs/2412.00081) (CVPR 2025)
- [When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging](https://arxiv.org/abs/2602.05536) (arXiv 2026)
- [Model Merging in the Essential Subspace](https://arxiv.org/abs/2602.20208) (arXiv 2026)
- [From Coefficients to Directions: Rethinking Model Merging with Directional Alignment](https://arxiv.org/abs/2512.00391) (MDA, 2025; DC-Merge, CVPR 2026)
- [Awesome Model Merging Survey](https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications) (ACM Computing Surveys 2026)
- [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089) (ICLR 2023)
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) (NeurIPS 2023)
- [AdaMerging: Adaptive Model Merging for Multi-Task Learning](https://arxiv.org/abs/2310.02575) (ICLR 2024)
