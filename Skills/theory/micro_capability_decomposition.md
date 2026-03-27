# Micro-Capability Decomposition: Beyond Atomic Expert Merging

> 讨论记录 | 2026-03-23 | ETVD 理论演进

---

## 1. 当前 ETVD 的局限：Expert 是原子的

### 1.1 回顾

现在 ETVD 对每个 2D 参数层做的事：

```
5 个 expert 的 task vectors τ₁,...,τ₅ ∈ R^{m×n}
flatten → 5 个 d 维行向量（d = m×n）
堆叠 → T ∈ R^{5×d}
SVD  → U(5×5) Σ(5×5) Vᵀ(5×d)
```

只有 **5 个基向量** v₁,...,v₅。U 矩阵告诉我们每个 expert 在这 5 个方向上的投影系数。

### 1.2 为什么只有 5 个是不够的

每个 expert 的 task vector τᵢ ∈ R^{m×n} 本身是一个矩阵，有自己的 SVD：

```
τᵢ = Σⱼ σᵢⱼ · uᵢⱼ · vᵢⱼᵀ
```

有效秩可能有几十到几百。也就是说，一个 expert 内部包含 **很多独立的修改方向**（称为"微能力"）。

当我们把整个 τᵢ flatten 成一个向量做 ETVD，相当于把 expert 的所有微能力 **打包成一个原子**。我们只能回答"OCR 和 VQA 整体上有多重叠"，不能回答"OCR 的哪些微能力和 VQA 重叠，哪些不重叠"。

### 1.3 这导致什么问题

实际场景中，一个 OCR expert 可能：
- **微能力 A**：学会了识别印刷体字符 → 对 TextVQA 有益
- **微能力 B**：学会了文本语义理解 → 和 VQA expert 共享
- **微能力 C**：为了 OCR 调整了某些空间特征 → 干扰 Chart 理解

5 方向的 ETVD 把 A、B、C 混在一起。如果我们判定 "OCR 方向和 Chart 方向冲突"，就只能整体抑制 OCR 的贡献——但这也丢掉了无辜的微能力 A。

**理想的做法**：保留 A，合理处理 B（共享部分不重复累积），去掉 C。这需要在微能力层面操作。

---

## 2. 核心 Insight：微能力分解

### 2.1 从 "N 个原子 expert" 到 "K 个微能力"

不再把每个 expert 当成一个原子。对每个参数层：

**Step 1：Individual SVD — 提取每个 expert 的微能力**

```
τᵢ ∈ R^{m×n}  →  SVD  →  τᵢ = Σⱼ₌₁^{rᵢ} σᵢⱼ · uᵢⱼ · vᵢⱼᵀ
```

保留 top-kᵢ 个奇异分量（按能量占比，如 90%），得到 kᵢ 个微能力。

每个微能力 (uᵢⱼ, σᵢⱼ, vᵢⱼ) 的含义：
- vᵢⱼ ∈ R^n：输入空间中被修改的方向（"这个微能力关注什么输入特征"）
- uᵢⱼ ∈ R^m：输出空间中产生的变化（"这个微能力产生什么输出变化"）
- σᵢⱼ：修改的强度

**Step 2：Cross-Expert 微能力关系分析**

5 个 expert 共产生 K = Σᵢ kᵢ 个微能力。分析它们之间的关系：

```
对任意两个微能力 (i,j) 和 (i',j')：
  输入重叠：cos_in = |vᵢⱼᵀ · vᵢ'ⱼ'|
  输出重叠：cos_out = |uᵢⱼᵀ · uᵢ'ⱼ'|
```

根据输入/输出重叠的组合，分为几种关系：

| 输入重叠 | 输出重叠 | 关系 | 含义 |
|---------|---------|------|------|
| 低 | 低 | **独立** | 两个微能力互不干扰 |
| 高 | 高，同向 | **冗余** | 相同的修改，叠加会 over-accumulate |
| 高 | 高，反向 | **冲突** | 对相同输入产生相反输出 |
| 高 | 低 | **输入竞争** | 修改同一输入特征但方向不同 |
| 低 | 高 | **输出竞争** | 不同输入但修改同一输出通道 |

### 2.2 和当前 ETVD 的对比

```
当前 ETVD：
  5 expert → flatten → 5×d 矩阵 → SVD → 5 个方向
  U(5×5) 给出 expert 级别的 occupancy

微能力分解：
  5 expert → 各自 SVD → K 个微能力方向 (K >> 5)
  分析 K 个方向之间的 input/output 重叠关系
  → 更细粒度的冲突/冗余/独立分类
```

关键区别：
- **基向量数量**：5 vs K（K 可能是几十到几百）
- **分析维度**：一维 occupancy vs 二维 (input overlap × output overlap)
- **操作粒度**：整个 expert vs 单个 rank-1 修改

### 2.3 数学上的关系

Individual SVD 和 Joint SVD 之间有联系。令 T_mat 为所有微能力在 flatten 空间中的表示：

```
Joint ETVD:
  T_flat = [vec(τ₁); ...; vec(τ₅)] ∈ R^{5×mn}
  → rank ≤ 5

Individual SVD pooling:
  每个 vec(σᵢⱼ · uᵢⱼ · vᵢⱼᵀ) = σᵢⱼ · (uᵢⱼ ⊗ vᵢⱼ) ∈ R^{mn}   (Kronecker product)
  → K 个方向，span 和 T_flat 相同
  → 但提供了不同的、更结构化的基底
```

两者张成同一个子空间（维度 ≤ 5），但 individual SVD 的基底保留了 input/output 的因子结构，而 joint ETVD 把这个结构 flatten 掉了。

---

## 3. 这带来什么新的可能

### 3.1 精细化的冲突解决

**当前**：检测到 "OCR 和 Chart 在方向 v₃ 上冲突" → 整体抑制这个方向。

**新方法**：
- 识别出 OCR 的微能力 o₇ 和 Chart 的微能力 c₃ 在输入空间上高度重叠但输出方向冲突
- 只去掉 o₇（OCR 中干扰 Chart 的那个微能力），保留 OCR 的其他微能力
- Chart 的 c₃ 完整保留

### 3.2 虚拟专家重组

从 K 个微能力中，按兼容性选择子集，组成"虚拟专家"：

```
Virtual Expert A = {OCR 独有微能力} ∪ {VQA 独有微能力} ∪ {OCR-VQA 共享微能力（去重）}
Virtual Expert B = {Chart 独有微能力} ∪ {Geometry 独有微能力（去掉和 Chart 冲突的）}
```

虚拟专家之间的干扰远小于原始专家，因为冲突微能力已经在重组阶段被移除。

### 3.3 和 wudi2 优化的结合

wudi2 在 d 维空间里做优化（每层 d 个参数）。微能力分解提供了一个中间方案：

```
方案对比：
  TA:     0 个可调参数（fixed sum）
  ETVD:   5 个隐式系数（由 occupancy 规则决定）
  微能力: K 个系数（每个微能力一个保留/丢弃/缩放决策）
  wudi2:  d 个参数（全空间优化）
```

K << d，但 K >> 5。在微能力系数空间做优化，比 wudi2 更高效，比 ETVD 更灵活。

且微能力系数有明确的语义：αᵢⱼ 控制的是 "expert i 的第 j 个修改方向的保留程度"。优化出来的结果是可解释的。

---

## 4. 待解决的问题

### 4.1 有效秩怎么选？

每个 expert 保留多少个微能力？

候选方案：
- **固定比例**：保留贡献 90%/95%/99% Frobenius norm 的前 k 个奇异值
- **自适应**：根据奇异值谱的"肘部"自动确定
- **统一**：所有 expert 保留相同数量 k

实验要回答：有效秩的分布是什么样的？不同 expert 差异大吗？

### 4.2 跨 expert 的微能力怎么比较？

两个来自不同 expert 的微能力，衡量"重叠"应该看：
- **只看右奇异向量 v**（输入空间方向）？
- **只看左奇异向量 u**（输出空间方向）？
- **同时看 u 和 v**？用什么组合度量？

直觉上，两个微能力"冲突"要求：修改同一输入方向（v 重叠高），但产生相反的输出变化（u 反向）。只看 v 或只看 u 都不完整。

可能的度量：

```
interference(a, b) = (vₐᵀ vᵦ) · (uₐᵀ uᵦ) · σₐ · σᵦ
```

- 正值 → 冗余（同向修改同输入）
- 负值 → 冲突（反向修改同输入）
- 接近零 → 独立

### 4.3 微能力之间的关系用什么数据结构表示？

K 个微能力之间形成一个 K×K 的关系矩阵。K 可能几百。

可能的结构化方法：
- **图**：微能力为节点，冲突/冗余为边，用图算法找最大兼容集
- **聚类**：把相似微能力聚成"能力簇"，每簇内去冗余
- **二部图**：input-overlap 图 + output-overlap 图

### 4.4 优化目标

如果我们要在微能力系数空间做优化，目标函数是什么？

候选：
1. **wudi2 式干扰最小化**（在微能力子空间中）
2. **最大独立集**：找到最多互不冲突的微能力子集
3. **带约束优化**：最大化保留的总能量（Σ αᵢⱼ² σᵢⱼ²），约束冲突微能力不能同时保留

### 4.5 非 2D 层怎么处理？

Individual SVD 天然只适用于 2D 权重矩阵。对 bias、LayerNorm 等 1D 参数，需要回退到简单策略（average 或 TA）。

---

## 5. wudi2 的分解及其局限

### 5.1 wudi vs wudi2 的核心区别

两者共享同一个优化框架：初始化 `m = Σ τ_i`，Adam 300 步。区别在 loss 的构造。

**wudi (v1)：原始空间**

```
loss = Σ_i || (m - τ_i) @ τ_i^T ||²_F / ||τ_i||²_F
```

- 扰动 `(m - τ_i)`：merged 和 raw task vector 的差
- 投影 `τ_i^T`：投影到 expert i 的完整行空间
- 含义：merged vector 对每个 expert 的干扰，投影到该 expert 自己的空间上，应尽量小

**wudi2 (v2)：拆分共享/独有 + 主子空间**

```python
avg = mean(τ_1, ..., τ_5)                              # 共享部分
τ̃_i = SVD_truncated(τ_i - avg, rank=m/N) + avg          # 独有低秩近似 + 共享
L_i = diag(s_i[:m/N]) @ V_i                             # expert i 的 top-1/N 奇异值子空间

loss = Σ_i || (m - τ̃_i) @ L_i^T ||²_F / ||τ_i||²_F
```

- 扰动目标 `τ̃_i`：不是原始 task vector，而是"清洗版"（top 独有方向 + 共享均值）
- 投影 `L_i^T`：投影到 expert i 的**主子空间**（top 1/N 奇异值）
- 含义：merged vector 对每个 expert 清洗版的干扰，在主子空间上应尽量小

**wudi2 的改进本质：不重要的方向上的干扰无所谓，重要的方向上才需要保护。**

### 5.2 wudi2 的共享/独有分解只是 0 阶

wudi2 的分解：

```
τ_i = avg + (τ_i - avg)
      ↑          ↑
  5 人平均     个体偏差
```

问题：假设 OCR 和 VQA 共享"文本理解"，Chart 和 Geometry 共享"空间推理"。5 人平均把这两个不同的共享混在一起了：

- `avg` 对 OCR 来说太多了（混入了 Chart/Geometry 的空间推理）
- `avg` 对 Chart 来说也太多了（混入了 OCR/VQA 的文本理解）
- `τ_i - avg` 不是纯粹的"独有"，还包含 "子组共享 - 全局平均" 的残留

这是最粗糙的 1-level 分解。

### 5.3 搜索空间：PiSSA 类比

wudi2 用 SVD 构造了更好的 loss 函数，但**搜索空间仍然是全参数空间**——merging_vector 有 m×n 个自由参数，优化器可以任意调整每个值。

```
                搜索空间           利用 SVD 的方式      可调参数量
────────────────────────────────────────────────────────────────
LoRA 类比:
  LoRA          随机低秩子空间       不利用             2rn
  PiSSA         主奇异值子空间       构造搜索空间        2rn

Merging 类比:
  wudi2         全参数空间 (m×n)     构造 loss 函数      m×n (~420万)
  微能力优化     微能力系数空间       构造搜索空间         K (~几百)
```

PiSSA 之于 LoRA 的 insight：不是搜索空间大小的区别（两者参数量相同），而是**起始点和搜索方向更合理**——主奇异值方向承载了最重要的信息。

类比：微能力优化不只是参数更少，而是每个参数都有明确语义（某个 expert 的某个修改方向的保留程度），优化在正确的结构化空间中进行。

### 5.4 wudi2 没有跨 expert 的组合能力

wudi2 的 loss 是 **per-expert 独立计算**再求和：

```
loss = Σ_i || (m - τ̃_i) @ L_i^T ||²
```

每一项只关心 merged vector 和 expert i 的关系。它不分析 expert i 和 expert j 之间的微能力重叠——如果 OCR 的某个方向和 Chart 的某个方向冲突，wudi2 只是在两个相矛盾的约束之间做折中，而不是识别冲突然后精准解决。

---

## 6. ETVD Joint SVD 的价值：发现分组结构

### 6.1 U 矩阵天然编码分组信息

ETVD 的 joint SVD 虽然只有 5 个方向，但 U 矩阵揭示了 expert 间的分组结构：

```
T = [τ₁; ...; τ₅] = UΣVᵀ

U 矩阵 (5×5) 的每一列 u_k：
  u_k = [0.9, 0.8, 0.0, 0.0, 0.1]  →  方向 k 被 {OCR, VQA} 共享
  u_k = [0.0, 0.0, 0.7, 0.6, 0.0]  →  方向 k 被 {Geometry, Chart} 共享
  u_k = [0.0, 0.0, 0.0, 0.0, 0.9]  →  方向 k 是 Grounding 独有
```

**不需要预设 group，U 矩阵自动从数据中发现哪些 expert 共享哪些方向。** 这比 wudi2 的全局 avg 有意义得多。

### 6.2 ETVD 作为分析管线的起点

ETVD 的 joint SVD 不应该只是一个 merging 方法。它的更大价值是作为**发现分组结构的手段**，为后续的精细分解提供指导。

### 6.3 层级分解：从 0 阶到多阶

wudi2 的分解是 0 阶（全局均值）。基于 ETVD 发现的分组，可以做更精细的 ANOVA 式层级分解：

```
Level 0:  μ = grand mean (所有 5 expert 共享)
Level 1:  μ_g = group mean (子组共享，由 ETVD 的 U 矩阵发现 group)
Level 2:  τ_i - μ_g(i) = individual residual (个体私有)

τ_i = μ + (μ_g(i) - μ) + (τ_i - μ_g(i))
      ↑         ↑                ↑
   全局共享   子组共享         个体私有
```

wudi2 只有 Level 0 和 Level 2，跳过了 Level 1。但 **Level 1 可能是最关键的**——正是子组间的差异导致了干扰。

具体来说：如果 U 矩阵显示在某个方向上 {OCR, VQA} 的系数大而 {Chart, Geometry, Grounding} 接近零，那么：

- 这个方向上的共享不应该被 5 人平均稀释（wudi2 的做法）
- 而应该只在 {OCR, VQA} 内部去冗余
- 其他 expert 不参与这个方向的平均

### 6.4 统一管线

```
Step 1: ETVD joint SVD (5 方向)
        → U 矩阵 → 发现 expert 分组结构
        → 哪些 expert 共享哪些宏观方向

Step 2: 基于分组，层级分解
        → τ_i = 全局共享 + 子组共享 + 个体私有
        → 比 wudi2 的 avg 更精确的共享/私有分离

Step 3: 对各组分，individual SVD 提取微能力
        → K 个微能力的 (u, σ, v) triplet
        → 分析跨 expert 的 input/output 重叠

Step 4: 微能力系数空间优化
        → α ∈ R^K 控制各微能力保留程度
        → 结构化搜索空间 + 可解释结果
```

ETVD 的 joint SVD 是 Step 1——整个分析流程的入口。

---

## 7. 发现 compositional 能力的三条路线

### 7.1 路线 A：右奇异向量空间的跨 expert 聚类

不 flatten task vector，直接分析每个 expert SVD 基向量之间的重叠：

```
Expert i: τ_i = U_i Σ_i V_i^T
取 top-k 右奇异向量: V_i ∈ R^{n × k}

池化所有 expert 的右奇异向量:
V_pool = [v_{1,1}, ..., v_{1,k}, v_{2,1}, ..., v_{5,k}] ∈ R^{n × 5k}

计算 5k × 5k 的 cosine similarity 矩阵
→ 聚类/谱分析 → 发现共享的输入方向
```

同样对左奇异向量 U_i 做。交叉分析：

- OCR 的 v₃ 和 VQA 的 v₇ cosine=0.92 → 同一输入特征被两个 expert 修改
- 再看对应的 u：同向 → 冗余修改；反向 → 冲突
- 给出 K×K 的微能力关系图（K = 5k >> 5）

**优点**：最直接，保留了 input/output 的因子结构。
**问题**：不同 expert 的奇异向量不天然对齐（SVD 的旋转不确定性），cosine similarity 可能不稳定。

### 7.2 路线 B：保持矩阵结构的张量分解

当前 ETVD 把 m×n 矩阵 flatten 成 mn 维向量再堆叠，丢掉了 input/output 因子结构。

保持矩阵结构，构造 3-way 张量：

```
T ∈ R^{N × m × n}    (5 × 2048 × 2048)

Tucker decomposition: T ≈ G ×₁ A ×₂ B ×₃ C
  A ∈ R^{N × r₁}:   expert 组合系数 (哪个 expert 有这个能力)
  B ∈ R^{m × r₂}:   output 基向量 (产生什么输出变化)
  C ∈ R^{n × r₃}:   input 基向量 (关注什么输入特征)
  G ∈ R^{r₁ × r₂ × r₃}: 核心张量 (input-output-expert 的耦合)
```

**"compositional capability" = 一个 (input direction, output direction) pair。**

Tucker 分解天然给出这种组合结构：
- 核心张量 G 的每个元素 G[a,b,c] 表示："expert 组合模式 a"在"输出方向 b"和"输入方向 c"之间的耦合强度
- A 矩阵告诉我们每个 expert 怎么参与各种组合模式

**优点**：数学上最干净，天然分离 expert/input/output 三个维度。
**问题**：计算成本高（2048×2048 的张量分解）；Tucker 分解不唯一，需要额外约束（如非负性、稀疏性）。

### 7.3 路线 C：ETVD-guided 层级 SVD

利用 ETVD 的 5 方向分解作为 scaffold，逐步细化：

```
Step 1: Joint SVD → 5 个宏观方向 + U 矩阵的分组信息
Step 2: 对每个宏观方向，取贡献最大的 expert 子组
Step 3: 在子组内，对对应的 task vector 分量做 individual SVD
        → 展开为子组内的微能力
Step 4: 分析子组间微能力的重叠（此时只需要比较不同子组，而非全体两两比较）
```

**优点**：层层递进，计算量可控，每一步都有 ETVD 的理论支撑。
**连接**：Step 1 发现分组 → Step 2-3 是层级分解（Section 6.3）→ Step 4 是微能力关系分析。

### 7.4 三条路线的对比

```
                   基向量数量    保留矩阵结构    需要的计算    和 ETVD 的关系
───────────────────────────────────────────────────────────────────────────
A: V-pooling       5k (~几百)   是（分别分析u,v） 中等         独立
B: Tucker          r₁×r₂×r₃    是（天然分离）    高           替代 ETVD
C: ETVD-guided     逐步展开     部分             低→中        扩展 ETVD
```

**初步判断**：路线 C 最稳妥——以 ETVD 为起点逐步深入，每一步都可验证。路线 A 作为补充分析工具。路线 B 理论上最优雅但实现复杂度高，可作为后续方向。

---

## 8. 和现有工作的关系

### 8.1 和 TSV 的关系

TSV (CVPR 2025) 也对每个 expert 做 individual SVD。但 TSV 只用它来度量 STI（子空间非正交度），不做微能力级别的分析。TSV 的合并策略是白化变换——一刀切地去相关，不区分"有益的共享"和"有害的冲突"。

微能力分解更进一步：识别出具体哪些 (expert, direction) pair 在冲突，精准去除。

### 8.2 和 SVC 的关系

SVC 关注 merged vector 的奇异值膨胀。微能力分解可以回答 SVC 回答不了的问题：膨胀的奇异值是哪些 expert 的哪些微能力贡献的。

### 8.3 和 wudi2 的关系

详见 Section 5。wudi2 用 SVD 构造了更好的 loss，但搜索空间仍为全空间。微能力分解构造了更好的搜索空间本身。wudi2 没有跨 expert 的组合分析能力。

---

## 9. 第一步实验计划

### 9.1 分析实验（不涉及 merge，纯分析）

1. **有效秩分布**：对 5 个 expert 的所有 2D 层，统计 individual SVD 的有效秩
   - 90%/95%/99% energy 对应多少个奇异值？
   - 不同 expert 之间差异多大？
   - 不同层深度有什么规律？

2. **跨 expert 微能力重叠分析**：
   - 对每一层，计算所有 K 个微能力两两之间的 interference 度量
   - 统计冲突/冗余/独立的比例（和 ETVD 的 8% conflict 比较）
   - 识别主要的冲突 pair：是哪些 expert 的哪些微能力在冲突

3. **验证**：用微能力分析的预测和 exclude-one 实验结果对比
   - 如果 OCR 和 Chart 之间的冲突微能力最多，应该和 exclude-OCR 后 ChartQA 提升的实验结果一致

### 9.2 合并实验

在分析的基础上：

1. **微能力过滤**：移除被识别为冲突的微能力，直接重建 task vector，然后做 TA
2. **微能力系数优化**：用 wudi2 式的目标函数，但在 K 维系数空间优化而非 d 维
3. 和 wudi2、TA、ETVD 对比

---

## 10. 开放讨论

- 微能力分解后的最优组合问题，本质上是一个 **combinatorial optimization**（从 K 个微能力中选子集），还是 **continuous optimization**（每个微能力一个连续系数）？
- input overlap 和 output overlap 在功能上的含义是否对等？还是其中一个更重要？
- 不同层的微能力结构是否有系统性差异？（early layers 更共享？deep layers 更 unique？）
- 微能力的数量 K 和 expert 数量 N 之间是什么关系？K 随 N 线性增长？
