# MLLM Expert Merging: 从第一性原理到实验验证的完整推导

> 整理于 2026-03-23 | OptMerge 项目理论课

## 目录

1. [问题定义：为什么需要 Model Merging](#1-问题定义)

2. [Task Arithmetic：最简单的合并方法](#2-task-arithmetic)

3. [SVD 分解基础回顾](#3-svd-分解基础)

4. [Ensemble Task Vector Decomposition (ETVD) 的完整推导](#4-etvd-完整推导)

5. [三类 Failure Mode 的数学推导](#5-三类-failure-mode)

6. [wudi/wudi2 方法的数学分析](#6-wudi-方法分析)

7. [实验数据分析与图表解读](#7-实验分析)

8. [从 ETVD 到微能力分解](#8-微能力分解)

9. [方法间的统一视角与理论边界](#9-统一视角)

10. [开放问题与研究方向](#10-开放问题)

---

## 1. 问题定义

### 1.1 设置

有一个预训练的 base model，参数为 $\theta_0 \in \mathbb{R}^D$（$D$ 是总参数量）。

在 $N$ 个不同的下游任务上分别 fine-tune，得到 $N$ 个 expert model：

$$\theta_i = \theta_0 + \tau_i, \quad i = 1, 2, \ldots, N$$

其中 $\tau_i = \theta_i - \theta_0$ 称为第 $i$ 个 expert 的 **task vector**（任务向量）。

在我们的实验中：

- Base model: InternVL2.5-1B（约 10 亿参数）

- $N = 5$ 个 experts: OCR, VQA, Geometry, Chart, Grounding

- 参数层: 170 个 2D 权重矩阵（attention Q/K/V/O + MLP up/gate/down × 24 层）

**目标**：找到一个 merged model $\theta_{\text{merged}}$，使其在所有 $N$ 个任务上都表现良好。

### 1.2 为什么不直接 multi-task fine-tune？

- 需要所有任务的训练数据（数据隐私/可及性问题）

- 计算成本高（要重新训练整个模型）

- 每次新增/删除任务都要从头训练

Model merging 的核心价值：**data-free, training-free**，只操作参数本身。

---

## 2. Task Arithmetic

### 2.1 最朴素的想法

Task Arithmetic（TA, Ilharco et al., ICLR 2023）直接把所有 task vector 加起来：

$$\theta_{\text{TA}} = \theta_0 + \lambda \sum_{i=1}^{N} \tau_i$$

其中 $\lambda$ 是一个全局缩放系数（scaling coefficient）。

**直觉**：如果 fine-tuning 在参数空间中沿某个方向移动了模型，那么叠加多个方向就应该同时获得多种能力。

### 2.2 为什么需要 $\lambda < 1$？

实验发现 $\lambda = 1$ 通常导致灾难性结果。在我们的实验中：

- $\lambda = 1.0$: 所有方法都崩溃

- $\lambda = 0.1$: TA 在 TextVQA 上达到 76.13（比 base 74.23 提升）

- wudi2 使用 $\lambda = 0.1$

**原因**：$N$ 个 task vector 叠加后，merged task vector 的 norm 远大于任何单个 task vector：

$$\left\| \sum_{i=1}^N \tau_i \right\| \geq \left\| \tau_i \right\| \quad \text{（通常远大于）}$$

如果 task vectors 互相正交：

$$\left\| \sum_{i=1}^N \tau_i \right\|^2 = \sum_{i=1}^N \|\tau_i\|^2 \quad \text{（勾股定理）}$$

$$\left\| \sum_{i=1}^N \tau_i \right\| = \sqrt{\sum_{i=1}^N \|\tau_i\|^2}$$

对 $N=5$，如果各 expert norm 相近，合并后 norm 约为单个的 $\sqrt{5} \approx 2.24$ 倍。如果有对齐成分，会更大。$\lambda < 1$ 是对此的粗糙补偿。

### 2.3 TA 的本质局限

$\lambda$ 是一个 **全局标量**，对所有方向同等缩放。但不同方向的情况完全不同：

- **独有方向**（只有一个 expert 贡献）：理想的缩放应该是 1

- **共享方向**（多个 expert 同向贡献）：被叠加放大了 $k$ 倍（$k$ 为占据该方向的 expert 数），需要缩小

- **冲突方向**（expert 间异号）：需要决议，而不是简单缩放

一个全局 $\lambda$ 无法同时满足这三种需求。**这是 TA 的根本缺陷。**

---

## 3. SVD 分解基础

在推导 ETVD 之前，先回顾 SVD 的关键性质。

### 3.1 矩阵的 SVD

对任意矩阵 $A \in \mathbb{R}^{m \times n}$（假设 $m \leq n$），存在分解：

$$A = U \Sigma V^T$$

其中：

- $U \in \mathbb{R}^{m \times m}$：左奇异向量矩阵，$U^T U = I$

- $\Sigma \in \mathbb{R}^{m \times n}$：对角矩阵，$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$，$r = \operatorname{rank}(A)$

- $V \in \mathbb{R}^{n \times n}$：右奇异向量矩阵，$V^T V = I$

**经济版 SVD**（我们实际使用的）：

$$A = U_r \Sigma_r V_r^T$$

其中 $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{r \times n}$。

### 3.2 SVD 的几何解释

$$A = \sum_{k=1}^{r} \sigma_k \, u_k \, v_k^T$$

每个 $\sigma_k u_k v_k^T$ 是一个 rank-1 矩阵，表示：

- $v_k$：输入空间中被"关注"的方向

- $u_k$：输出空间中产生的变化方向

- $\sigma_k$：这个关注-变化对的强度

**类比**：$A$ 将输入 $x$ 映射为 $Ax$。SVD 说这个映射可以分解为 $r$ 个独立的"通道"：先投影到 $v_k$ 方向，放大 $\sigma_k$ 倍，再映射到 $u_k$ 方向。

### 3.3 低秩近似（Eckart-Young 定理）

保留 top-$k$ 个奇异值的近似是所有 rank-$k$ 矩阵中 Frobenius norm 最优的：

$$A_k = \sum_{j=1}^{k} \sigma_j u_j v_j^T = \arg\min_{\operatorname{rank}(B) \leq k} \|A - B\|_F$$

$$\|A - A_k\|_F^2 = \sum_{j=k+1}^{r} \sigma_j^2$$

**关键事实**：TSV 论文发现 task vector 矩阵通常是低秩的——保留 10% 的奇异分量就能保持 99% 的 Frobenius 能量。

### 3.4 向量堆叠矩阵的 SVD

对于 ETVD，我们将 $N$ 个 $d$ 维向量堆叠成矩阵 $T \in \mathbb{R}^{N \times d}$（$N \ll d$）。

此时 SVD 给出：

$$T = U \Sigma V^T, \quad U \in \mathbb{R}^{N \times r}, \quad \Sigma \in \mathbb{R}^{r \times r}, \quad V^T \in \mathbb{R}^{r \times d}$$

其中 $r \leq \min(N, d) = N$。

**关键解释**：

- $V$ 的各列 $v_1, \ldots, v_r$：$d$ 维空间中的正交基底，张成 $N$ 个向量的子空间

- $U$ 的各行 $u_{i,:}$：第 $i$ 个原始向量在新基底下的坐标

- $U$ 的各列 $u_{:,k}$：所有 $N$ 个向量在第 $k$ 个基底方向上的投影系数

- $\Sigma$ 的对角元 $\sigma_k$：第 $k$ 个基底方向承载的能量

**原始向量的重建**：

$$T_{i,:} = \sum_{k=1}^{r} U_{ik} \sigma_k V_{k,:}$$

即第 $i$ 个向量等于各基底方向 $v_k$ 的加权和，权重为 $U_{ik} \sigma_k$。

---

## 4. ETVD 完整推导

### 4.1 第一步：构造 Ensemble 矩阵

对模型的每一个 2D 权重层 $l$，有 $N$ 个 task vector $\tau_i^l \in \mathbb{R}^{m_l \times n_l}$。

将每个 task vector 矩阵展平为向量：

$$\operatorname{vec}(\tau_i^l) \in \mathbb{R}^{d_l}, \quad d_l = m_l \times n_l$$

堆叠为 ensemble 矩阵：

$$T^l = \begin{bmatrix} \operatorname{vec}(\tau_1^l)^T \\ \operatorname{vec}(\tau_2^l)^T \\ \vdots \\ \operatorname{vec}(\tau_N^l)^T \end{bmatrix} \in \mathbb{R}^{N \times d_l}$$

在我们的实验中：$N = 5$，$d_l$ 从几千到几百万不等（如 $2048 \times 2048 = 4194304$）。

### 4.2 第二步：对 Ensemble 矩阵做 SVD

$$T^l = U^l \Sigma^l (V^l)^T$$

由于 $N = 5 \ll d_l$，我们得到：

- $U^l \in \mathbb{R}^{5 \times 5}$：5 个 expert 在 5 个基底方向上的坐标

- $\Sigma^l \in \mathbb{R}^{5 \times 5}$：5 个基底方向的能量

- $(V^l)^T \in \mathbb{R}^{5 \times d_l}$：5 个 $d_l$ 维正交基向量

**为什么是 5 个基底方向？**

$\operatorname{rank}(T^l) \leq \min(N, d_l) = N = 5$

5 个向量最多张成 5 维子空间，所以最多 5 个正交基底就能完全表示这些向量（无近似误差）。

### 4.3 第三步：从 $U$ 矩阵读取结构

$U^l$ 矩阵是 $5 \times 5$，每一列对应一个基底方向，每一行对应一个 expert。

$$U^l = \begin{bmatrix} u_{11} & u_{12} & u_{13} & u_{14} & u_{15} \\ u_{21} & u_{22} & u_{23} & u_{24} & u_{25} \\ u_{31} & u_{32} & u_{33} & u_{34} & u_{35} \\ u_{41} & u_{42} & u_{43} & u_{44} & u_{45} \\ u_{51} & u_{52} & u_{53} & u_{54} & u_{55} \end{bmatrix}$$

行：[OCR, VQA, Geometry, Chart, Grounding]

列：[基底方向 1, 2, 3, 4, 5]

**定义占据度（Occupancy）**：

对第 $k$ 个基底方向，定义阈值 $\epsilon$（实验中取 0.3）：

$$\operatorname{occ}(k) = \left| \{ i : |U_{ik}| > \epsilon \} \right|$$

即有多少个 expert 在该方向上有显著贡献。

**定义符号一致性（Sign Consistency）**：

对第 $k$ 个基底方向中所有显著的 expert：

$$\text{signs}_k = \{ \operatorname{sign}(U_{ik}) : |U_{ik}| > \epsilon \}$$

如果所有符号相同，则该方向是"符号一致"的。

### 4.4 第四步：方向分类

基于占据度和符号一致性，将每个基底方向分为三类：

| 条件 | 类型 | 含义 |
|------|------|------|
| $\operatorname{occ}(k) = 1$ | **Unique** | 仅属于一个 expert |
| $\operatorname{occ}(k) \geq 2$ 且符号一致 | **Shared** | 多个 expert 的共识方向 |
| $\operatorname{occ}(k) \geq 2$ 且符号不一致 | **Conflict** | expert 之间在此方向上冲突 |

**实验结果**（170 层 × 5 方向 = 850 个方向）：

- Unique: 733 (86.2%)

- Shared: 51 (6.0%)

- Conflict: 66 (7.8%)

### 4.5 第五步：占据度感知的谱重建

**TA 的等价形式**：

先推导 TA 在 SVD 基底下的表示。TA 的 merged task vector 是所有行的和：

$$\tau_{\text{TA}}^l = \sum_{i=1}^N \operatorname{vec}(\tau_i^l) = \sum_{i=1}^N \sum_{k=1}^r U_{ik} \sigma_k v_k = \sum_{k=1}^r \underbrace{\left(\sum_{i=1}^N U_{ik}\right)}_{c_k^{\text{TA}}} \sigma_k v_k$$

即 TA 在每个基底方向 $k$ 上的系数为：

$$c_k^{\text{TA}} = \sum_{i=1}^N U_{ik}$$

这是 $U$ 矩阵第 $k$ 列的元素之和，**不区分占据度**。

**ETVD 的重建**：

$$\tau_{\text{ETVD}}^l = \sum_{k=1}^r w_k \cdot \sigma_k \cdot v_k$$

其中 $w_k$ 根据方向类型确定：

**情况 1：Unique 方向**（$\operatorname{occ}(k) = 1$，只有 expert $j$ 显著）

$$w_k = \sum_{i=1}^N U_{ik} \approx U_{jk}$$

实际上由于只有 $j$ 显著，其他项接近 0，所以 $w_k \approx U_{jk}$。保留完整的贡献。

**情况 2：Shared 方向**（$\operatorname{occ}(k) \geq 2$，所有显著 expert 同号）

$$w_k = \frac{\sum_{i=1}^N U_{ik}}{\operatorname{occ}(k)}$$

TA 的系数被占据度除，**防止 over-accumulation**。

直觉：如果 3 个 expert 都在同一方向贡献了 $+0.5$，TA 会得到 $1.5$，但实际上这只是同一个改进被重复计了 3 次。ETVD 将其归一化为 $0.5$（或 $1.5/3 = 0.5$）。

**情况 3：Conflict 方向**（$\operatorname{occ}(k) \geq 2$，符号不一致）

多数投票策略：

$$w_k = \operatorname{sign}\left(\sum_{i \in \mathcal{S}} \operatorname{sign}(U_{ik})\right) \cdot \operatorname{mean}_{i \in \mathcal{S}^+}(|U_{ik}|) \cdot \frac{|\text{majority}|}{\operatorname{occ}(k)}$$

其中 $\mathcal{S} = \{i : |U_{ik}| > \epsilon\}$ 是显著 expert 的集合，$\mathcal{S}^+$ 是多数一方的 expert 集合。

### 4.6 理解 ETVD 与 TA 的区别

用一个具体例子说明。假设某层某基底方向 $k$：

$$U_{:,k} = [0.6, \; 0.5, \; -0.8, \; 0.0, \; 0.0]$$

即 OCR=+0.6, VQA=+0.5, Geometry=-0.8, Chart≈0, Grounding≈0。

- **TA 的系数**：$c_k^{\text{TA}} = 0.6 + 0.5 + (-0.8) + 0 + 0 = 0.3$

- **ETVD 的分析**：

- $\operatorname{occ}(k) = 3$（OCR, VQA, Geometry 都显著）

- 符号不一致（OCR/VQA 为正，Geometry 为负）→ **Conflict 方向**

- 多数方为正（2 vs 1），agreement ratio = 2/3

- $w_k = +1 \times \frac{0.6+0.5}{2} \times \frac{2}{3} = 0.55 \times 0.667 = 0.367$

在这个例子中 ETVD 和 TA 的系数相近（0.367 vs 0.3），但语义不同：

- TA 是盲目求和

- ETVD 是识别出冲突后，根据多数意见做出的决策

**更极端的例子**：shared 方向

$$U_{:,k} = [0.5, \; 0.4, \; 0.6, \; 0.0, \; 0.0]$$

- **TA 的系数**：$c_k^{\text{TA}} = 0.5 + 0.4 + 0.6 = 1.5$

- **ETVD**：$\operatorname{occ}(k) = 3$, 同号 → Shared

- $w_k = 1.5 / 3 = 0.5$

ETVD 把系数从 1.5 压缩到 0.5——**去除了 3 倍的 over-accumulation**。

---

## 5. 三类 Failure Mode 的数学推导

### 5.1 统一框架：为什么 naive sum 有系统性缺陷

令 $\tau_{\text{sum}} = \sum_{i=1}^N \tau_i$ 是所有 task vector 的直接求和。对 $\tau_{\text{sum}}$ 做 SVD：

$$\tau_{\text{sum}} = \sum_{k} \sigma_k^{\text{sum}} \, u_k^{\text{sum}} \, (v_k^{\text{sum}})^T$$

问题是：$\sigma_k^{\text{sum}}$ 相对于个体 task vector 的奇异值谱 $\sigma_k^{(i)}$ 发生了系统性畸变。

### 5.2 Failure Mode I: Over-Accumulation

**定义**：当多个 expert 的 task vector 在某个方向上同向对齐时，该方向的奇异值在求和后被放大。

**数学推导**：

考虑 $d$ 维空间中 $N$ 个 task vector。假设方向 $v$ 是所有 $N$ 个 task vector 共享的，即：

$$\langle \tau_i, v \rangle = a_i > 0, \quad \forall i$$

则求和后在该方向的投影为：

$$\left\langle \sum_{i=1}^N \tau_i, \, v \right\rangle = \sum_{i=1}^N a_i$$

如果每个 $a_i \approx a$，则投影值 $\approx Na$，而每个个体的投影只有 $a$。

**放大倍率**：共享方向的能量被放大了 $N$ 倍（或更准确地说，放大了占据度 $k$ 倍，$k \leq N$）。

**对 benchmark 的影响**：共享方向通常编码了任务间的通用能力提升（如通用的语言理解）。过度放大这些方向导致模型偏向通用能力，而任务特定的能力（编码在独有方向上）被相对稀释。

**SVC 论文的发现**（arXiv 2602.05536）：这正是 "Spectral Over-Accumulation" 现象。他们用 $\sigma_k^{\text{sum}} / \sqrt{\sum_i (\sigma_k^{(i)})^2}$ 来量化放大程度。比值 $>1$ 表示 over-accumulation。

我们的实验数据：

- 总能量放大比：$\frac{\|{\tau_{\text{sum}}}\|_F^2}{\sum_i \|\tau_i\|_F^2} = \frac{2039.78}{1868.60} = 1.092$

- Top 奇异值放大比均值：1.033，最大值：1.607

### 5.3 Failure Mode II: Under-Representation

**定义**：独有方向（仅属于一个 expert）的相对权重被共享方向的放大所淹没。

**数学推导**：

假设有两类方向：

- 共享方向 $v_s$，被 $k$ 个 expert 占据，每个贡献 $a$

- 独有方向 $v_u$，仅被 1 个 expert 占据，贡献 $b$

求和后：

- 共享方向能量：$(ka)^2 = k^2 a^2$

- 独有方向能量：$b^2$

**独有方向的相对权重**：

$$\frac{b^2}{k^2 a^2 + b^2}$$

如果 $a \approx b$（两类方向同等重要），则相对权重约为 $\frac{1}{k^2 + 1}$。对 $k=3$，独有方向的权重仅为原来的 $\frac{1}{10}$。

**对 benchmark 的影响**：任务特定能力被稀释。例如 Chart 理解的专有知识在合并后被 OCR/VQA 的共享方向淹没，导致 ChartQA 下降。

**TA 中 $\lambda < 1$ 的处理方式**：$\lambda$ 缩小了所有方向，包括共享和独有。这部分补偿了 over-accumulation，但**同时也削弱了独有方向**。$\lambda$ 越小，独有方向损失越大——这解释了为什么太小的 $\lambda$ 也不好。

### 5.4 Failure Mode III: Sign Conflict

**定义**：不同 expert 在同一方向上贡献了相反符号的修改，相互抵消。

**数学推导**：

假设方向 $v_c$ 上：

$$\langle \tau_A, v_c \rangle = +a, \quad \langle \tau_B, v_c \rangle = -a$$

求和后：

$$\langle \tau_A + \tau_B, v_c \rangle = 0$$

两个 expert 在该方向上的改进完全被抹去。即使第三个 expert $C$ 也在该方向有贡献 $+c$，效果也被严重削弱。

**magnitude-weighted sign conflict rate**：

$$\text{Conflict}(i,j) = \frac{\sum_p |\tau_i(p) \cdot \tau_j(p)| \cdot \mathbb{1}[\operatorname{sign}(\tau_i(p)) \neq \operatorname{sign}(\tau_j(p))]}{\sum_p |\tau_i(p) \cdot \tau_j(p)|}$$

我们的实验数据：

- 大部分 pair 的 conflict rate 接近 0.5（近乎随机）

- OCR-Grounding: 0.39（最低，因为它们最对齐）

- Geometry 和所有人的 conflict rate 都 ~0.497（最独立）

**直觉**：conflict rate 接近 0.5 意味着两个 task vector 几乎正交——它们在共同参数上的符号是随机的。这正是 cosine similarity 接近 0 的另一种表述。

### 5.5 三类 Failure Mode 的统一

这三个问题都源于同一个根因：**naive sum 不利用占据结构（occupancy structure）**。

$$c_k^{\text{TA}} = \sum_{i=1}^N U_{ik} \quad \text{vs} \quad c_k^{\text{ETVD}} = f(U_{:,k}, \operatorname{occ}(k), \text{signs})$$

ETVD 根据 $U$ 矩阵提供的占据和符号信息，对三类方向施加不同的处理，在一个统一的框架内解决三个问题。

---

## 6. wudi/wudi2 方法的数学分析

### 6.1 wudi (v1) 的推导

**优化目标**：找到 merging vector $m \in \mathbb{R}^{m \times n}$，使其对每个 expert 的干扰最小。

**干扰的定义**：$m$ 和 $\tau_i$ 的差异 $(m - \tau_i)$ 投影到 $\tau_i$ 的行空间上应尽量小。

逐步推导：

**第一步**：定义扰动（disturbance）向量：

$$\delta_i = m - \tau_i$$

这是 merged vector 偏离 expert $i$ 的方向。

**第二步**：将扰动投影到 expert $i$ 的行空间：

$$\delta_i \cdot \tau_i^T \in \mathbb{R}^{m \times m}$$

矩阵乘法 $(m - \tau_i) \cdot \tau_i^T$：逐行来看，$\delta_i$ 的每行 $\delta_{i,j,:} \in \mathbb{R}^n$ 和 $\tau_i$ 的各行做内积。如果 $\delta_i$ 在 $\tau_i$ 行空间上的投影大，说明 $m$ 破坏了 expert $i$ 的重要方向。

**第三步**：loss 函数（归一化后求和）：

$$\mathcal{L}_{\text{wudi}} = \sum_{i=1}^N \frac{\| (m - \tau_i) \cdot \tau_i^T \|_F^2}{\|\tau_i\|_F^2}$$

分母 $\|\tau_i\|_F^2$ 使不同 expert 的贡献归一化。

**第四步**：优化：

$$m^* = \arg\min_m \mathcal{L}_{\text{wudi}}$$

初始化 $m_0 = \sum_i \tau_i$（TA 的结果），Adam 优化器，lr=1e-5，300 步。

### 6.2 wudi2 (v2) 的改进

wudi2 做了两个关键改进：

**改进一：清洗 task vector（共享/独有分解）**

$$\bar{\tau} = \frac{1}{N} \sum_{i=1}^N \tau_i \quad \text{（全局均值）}$$

对每个 expert 的"个体偏差"做 SVD：

$$\tau_i - \bar{\tau} = U_i' \Sigma_i' (V_i')^T$$

保留 top $m/N$ 个奇异分量（$m$ 为行数，$N$ 为 expert 数），重建清洗后的 target：

$$\tilde{\tau}_i = U_i'[:, :m/N] \cdot \Sigma_i'[:m/N, :m/N] \cdot (V_i')[:m/N, :] + \bar{\tau}$$

**直觉**：$\bar{\tau}$ 是所有 expert 共享的部分。$\tau_i - \bar{\tau}$ 是 expert $i$ 的独有偏差。只保留独有偏差中最重要的方向（top $m/N$ 个奇异值），然后加回共享部分。这样：

- 去掉了噪声（小奇异值的尾巴）

- 分离了共享和独有成分

**改进二：投影到主子空间**

不再投影到 $\tau_i$ 的完整行空间，而是投影到 top $m/N$ 个奇异值对应的子空间：

$$L_i = \operatorname{diag}(s_i[:m/N]) \cdot V_i[:m/N, :]$$

loss 变为：

$$\mathcal{L}_{\text{wudi2}} = \sum_{i=1}^N \frac{\| (m - \tilde{\tau}_i) \cdot L_i^T \|_F^2}{\|\tau_i\|_F^2}$$

**含义**：只在 expert 的主子空间上最小化干扰。不重要的方向（小奇异值）上的偏差被忽略。

### 6.3 wudi2 的优化过程详解

输入：5 个 task vector $\tau_1, \ldots, \tau_5 \in \mathbb{R}^{m \times n}$

```

1. 计算 avg = mean(τ₁, ..., τ₅)

2. 对每个 i = 1..5:

a. SVD(τᵢ - avg) = U'ᵢ Σ'ᵢ V'ᵢᵀ

b. rank_cut = m / N

c. τ̃ᵢ = U'ᵢ[:,:rank_cut] @ Σ'ᵢ[:rank_cut,:rank_cut] @ V'ᵢ[:rank_cut,:] + avg

d. Lᵢ = diag(s'ᵢ[:rank_cut]) @ V'ᵢ[:rank_cut,:]

3. 初始化 m = Σᵢ τᵢ (TA 的结果)

4. Adam 优化 300 步:

loss = Σᵢ ‖(m - τ̃ᵢ) @ Lᵢᵀ‖²_F / ‖τᵢ‖²_F

```

### 6.4 wudi2 的理论局限

**局限 1：共享/独有分解太粗糙**

wudi2 用全局均值 $\bar{\tau}$ 做分解。但如果只有 OCR 和 VQA 共享"文本理解"，Chart 和 Geometry 共享"空间推理"，那么 5 人平均 $\bar{\tau}$ 混合了这两个不同的共享成分：

$$\bar{\tau} = \frac{1}{5}(\underbrace{\tau_{\text{OCR}} + \tau_{\text{VQA}}}_{\text{文本理解组}} + \underbrace{\tau_{\text{Geo}} + \tau_{\text{Chart}}}_{\text{空间推理组}} + \tau_{\text{Ground}})$$

结果是 $\bar{\tau}$ 对任何一组来说都不是好的共享估计。

**局限 2：搜索空间太大**

$m \in \mathbb{R}^{m \times n}$ 有 $m \times n$ 个自由参数（如 $2048 \times 2048 = 420$ 万）。这是全空间优化。虽然有 SVD 指导的 loss，但搜索空间本身没有被结构化。

**局限 3：没有跨 expert 的组合分析**

wudi2 的 loss 是 **per-expert 独立** 后求和：

$$\mathcal{L} = \sum_{i=1}^N \mathcal{L}_i(m, \tilde{\tau}_i, L_i)$$

每一项只看 $m$ 和 expert $i$ 的关系。如果 OCR 和 Chart 在某个方向上冲突，wudi2 只是在两个矛盾约束之间折中，不会识别出冲突并主动解决。

### 6.5 ETVD vs wudi2 的结构性对比

| | ETVD | wudi2 |
|--|------|-------|
| **共享/独有分解** | Joint SVD 自然发现分组 | 全局均值（0 阶粗糙分解） |
| **干扰识别** | 显式分类（U 矩阵 → unique/shared/conflict） | 隐式（loss 中的优化） |
| **搜索空间** | 5 个方向的系数（由规则确定） | $m \times n$ 全空间 |
| **计算成本** | 一次 SVD（$O(N^2 d)$） | 300 次 Adam 迭代 |
| **Over-accumulation 处理** | 显式除以 occupancy | 无显式处理 |
| **可解释性** | 每个方向的处理策略可解释 | 黑盒优化结果 |

---

## 7. 实验分析与图表解读

### 7.1 Cosine Similarity Heatmap

**数据**（aggregated over all 170 2D layers）：

| | OCR | VQA | Geometry | Chart | Grounding |
|--|-----|-----|----------|-------|-----------|
| OCR | 1.000 | 0.031 | 0.004 | 0.017 | **0.141** |
| VQA | 0.031 | 1.000 | 0.003 | 0.011 | 0.032 |
| Geometry | 0.004 | 0.003 | 1.000 | 0.008 | 0.003 |
| Chart | 0.017 | 0.011 | 0.008 | 1.000 | 0.010 |
| Grounding | **0.141** | 0.032 | 0.003 | 0.010 | 1.000 |

**解读**：

1. **所有 off-diagonal 值都非常小**（最大 0.141），说明 5 个 task vector 基本生活在正交子空间中。这是 ETVD 中 86% 方向为 Unique 的底层原因。

2. **OCR-Grounding 的亲和性最高**（0.141，远高于其他 pair）。两者都涉及精细的空间定位（OCR 定位文字位置，Grounding 定位物体位置），可能共享了一些低级视觉特征的调整。

3. **Geometry 最孤立**（和所有人的 cosine 都 < 0.008）。几何推理使用的参数修改和其他任务完全不同。

4. **OCR-VQA 的相似度**（0.031）远低于 OCR-Grounding（0.141），尽管 OCR 和 VQA 在功能上更相近（都涉及文本）。这说明参数空间的几何关系不直接反映功能语义。

### 7.2 Task Vector Magnitude

| Expert | L2 Norm |
|--------|---------|
| VQA | **27.31** |
| OCR | 21.64 |
| Geometry | 16.09 |
| Chart | 15.18 |
| Grounding | **12.83** |

**解读**：

VQA 的 norm 是 Grounding 的 2.1 倍。这意味着在 naive sum 中，VQA 的贡献被自然放大——不是因为 VQA "更重要"，而是因为它的 fine-tuning 修改幅度更大。

$$\text{VQA 在总 norm 中的能量占比} = \frac{27.31^2}{27.31^2 + 21.64^2 + 16.09^2 + 15.18^2 + 12.83^2} = \frac{745.8}{1868.6} = 39.9\%$$

VQA 一个 expert 就占了总能量的 40%。这对 GQA benchmark 有利（VQA 直接相关），但对 ChartQA 不利（Chart 的 15.18 能量被淹没）。

### 7.3 Gram Matrix: $G = T T^T$

Gram 矩阵 $G_{ij} = \langle \operatorname{vec}(\tau_i), \operatorname{vec}(\tau_j) \rangle$（aggregated over layers）：

- **对角线**：即 $\|\tau_i\|^2$，和 L2 Norm 平方一致（VQA: 745.74, OCR: 468.45, ...）

- **Off-diagonal**：

- OCR-Grounding: 39.18（最高 off-diagonal 值）

- OCR-VQA: 18.28

- VQA-Grounding: 11.22

Gram 矩阵 = cosine similarity × norm 乘积。OCR-Grounding 有最高的 Gram 值，因为它们既有最高的 cosine（0.141）又有较大的 norm 乘积（21.64 × 12.83 = 277.6）。

**Gram 矩阵与 ETVD 的联系**：

$$G = T T^T = U \Sigma V^T V \Sigma^T U^T = U \Sigma^2 U^T$$

Gram 矩阵完全由 $U$ 矩阵和奇异值决定。所有 pairwise 关系都浓缩在 ETVD 的 $U\Sigma^2 U^T$ 中。

### 7.4 Sign Conflict Heatmap

大部分 pair 的冲突率在 0.475-0.498 之间（接近 0.5 = 完全随机），再次确认近正交性。

**OCR-Grounding 的冲突率最低**（0.390），意味着在它们有重叠的参数位置上，61% 的情况同号（synergy），只有 39% 异号。

**推论**：OCR 和 Grounding 不仅方向相近（cosine=0.141），而且在重叠方向上倾向于同向修改（synergy > conflict）。这解释了为什么去掉 Grounding 几乎无影响——OCR 已经"涵盖"了 Grounding 的共享修改。

### 7.5 Pairwise STI (Singular Task Interference)

TSV 论文定义的 STI 度量子空间重叠度（通过奇异向量的内积）：

| | OCR | VQA | Geometry | Chart | Grounding |
|--|-----|-----|----------|-------|-----------|
| OCR | 0 | **1.000** | 0.456 | 0.687 | 0.688 |
| VQA | **1.000** | 0 | 0.497 | 0.625 | 0.709 |
| Geometry | 0.456 | 0.497 | 0 | 0.536 | 0.349 |
| Chart | 0.687 | 0.625 | 0.536 | 0 | 0.472 |
| Grounding | 0.688 | 0.709 | **0.349** | 0.472 | 0 |

**与 cosine similarity 的对比**：

注意 STI 和 cosine 的排序不同！

- Cosine 最高：OCR-Grounding (0.141)

- STI 最高：OCR-VQA (1.000)

**为什么？** Cosine 度量的是 **整体方向对齐**（所有参数一起算一个角度），而 STI 度量的是 **奇异向量子空间的重叠**。OCR 和 VQA 的整体方向不同（cosine 只有 0.031），但它们的 SVD 子空间高度重叠（STI=1.0）——意味着它们修改的参数子空间类似，只是在子空间内的具体方向不同。

**直觉类比**：OCR 和 VQA 可能都在修改同一组注意力头的同一组输入/输出通道（子空间重叠），但修改的具体数值不同（整体 cosine 低）。

### 7.6 U-Matrix Structure 可视化

U-Matrix 热力图展示了三个不同层（early/middle/late）的 $U \times \Sigma$ 矩阵结构。

**Early Layer (self_attn.q_proj.weight)**:

```

k=0 k=1 k=2 k=3 k=4

OCR -0.28 0.98 0.25 -0.05 -0.07

VQA -1.38 -0.22 -0.01 0.00 -0.05

Geo -0.02 0.10 -0.13 0.74 -0.02

Chart -0.06 0.25 -0.91 -0.12 -0.05

Ground-0.14 0.12 -0.05 0.01 0.64

```

解读 k=0 方向：VQA 主导（-1.38），其他弱。这是 VQA 的 unique 方向。

解读 k=1 方向：OCR 主导（0.98），其他弱。这是 OCR 的 unique 方向。

解读 k=3 方向：Geometry 主导（0.74）。Geometry 的 unique 方向。

解读 k=2 方向：Chart 主导（-0.91），OCR 有部分贡献（0.25）且异号 → **Conflict 方向**！

这个 early layer 展示了清晰的方向分离，但也有局部冲突（Chart-OCR 在 k=2 上）。

**Late Layer (mlp1.3.weight)**:

```

k=0 k=1 k=2

OCR -0.24 0.66 0.03

VQA -0.83 -0.22 0.01

Geo -0.02 0.03 -0.62

Chart -0.07 0.07 -0.05

Ground-0.11 0.15 0.00

```

k=0：VQA + OCR 共同占据（-0.83, -0.24 同号）→ **Shared 方向**，over-accumulation 风险。

### 7.7 Spectral Over-Accumulation

**左图（直方图）**：Top singular value ratio（实际/正交假设预期）的分布。

- 绿色虚线：ratio=1.0（完全正交，无 over-accumulation）

- 红色线：均值=1.033

- 分布横跨 0.82-1.61

约 60% 的层的 ratio > 1.0（存在 over-accumulation），约 40% 的层 ratio < 1.0（某些方向因 sign conflict 而 under-accumulated）。

**右图（频谱对比）**：以 q_proj.weight 为例，比较：

- 红色：sum of TVs 的奇异值谱

- 绿色虚线：正交假设下的预期谱（$\sqrt{\sum_i \sigma_{i,k}^2}$）

- 浅色：各个 expert 的个体谱

在 top-3 奇异值位置，红色线高于绿色虚线 → **确认 over-accumulation 存在于最重要的方向上**。这些方向正是 ETVD 中被分类为 "Shared" 的方向。

### 7.8 Layer Depth Analysis: U-shaped Pattern

| Layer | Mean Cosine |
|-------|-------------|
| 0 (first) | **0.0464** |
| 4-5 (middle) | 0.0234-0.0237 |
| 12 | **0.0397** |
| 23 (last) | **0.0460** |

**U-shaped pattern**：

- **早期层**（layer 0）：高 cosine → 共享修改。早期层学习低级视觉/语言特征（边缘检测、token embedding），各 expert 的修改相似。

- **中间层**（layer 4-5）：最低 cosine → 最独立。中间层是任务特定的推理逻辑所在。

- **晚期层**（layer 23）：高 cosine → 共享修改。晚期层负责输出格式化和最终预测，各 expert 可能学到了类似的输出习惯。

**启示**：per-layer 的 merge 策略应该区分对待——中间层可以大胆保留（干扰小），早期/晚期层需要更谨慎的 over-accumulation 控制。

### 7.9 Component Analysis: Attention vs MLP

| Component | Mean Cosine |
|-----------|-------------|
| attn_k | **0.0449** |
| attn_v | **0.0404** |
| attn_q | 0.0343 |
| attn_o | 0.0313 |
| mlp_down | 0.0265 |
| mlp_up | 0.0265 |
| mlp_gate | **0.0207** |

**Attention 层的 cross-expert 相似度系统性高于 MLP 层**，特别是 K/V projection。

**解释**：K/V 层决定了"关注什么"和"记住什么"。不同 expert 可能都在调整模型对类似视觉/文本特征的注意力分配，导致在 K/V 空间中产生共享修改。

MLP gate（最低 cosine）控制信息流的非线性门控，各 expert 的修改最独立。

**启示**：merge 策略可以 per-component 差异化——attention K/V 需要更强的 over-accumulation 控制，MLP gate 可以直接叠加。

### 7.10 Benchmark 结果综合分析

| Model | TextVQA | OCRVQA | VizWiz | GQA | ChartQA |
|---|---|---|---|---|---|
| Base | 74.23 | 41.76 | 29.01 | 54.69 | 69.72 |
| wudi2 (s=0.1) | **76.01** | **46.29** | **30.88** | **57.14** | 68.44 ↓ |
| TA (s=0.1) | **76.13** | 44.82 | - | - | 62.40 ↓↓ |
| ETVD (s=0.1) | 74.06 | - | - | - | 64.36 |
| ETVD (s=0.2) | 73.75 | - | - | - | 65.04 |

**关键发现**：

1. **TA s=0.1 在 TextVQA 上最好**（76.13），但在 ChartQA 上崩溃（62.40，比 base 降 7.3 分）。这正是 over-accumulation + under-representation 的典型表现——VQA/OCR 的共享方向在 TA 中被放大，TextVQA 受益（这些共享方向有助于文本识别），但 ChartQA 受损（Chart 的独有方向被淹没）。

2. **ETVD 在 ChartQA 上比 TA 好**（s=0.1: 64.36 vs 62.40, s=0.2: 65.04 vs 64.00），因为 occupancy normalization 抑制了共享方向的 over-accumulation，保护了 Chart 的独有方向。但 ETVD 在 TextVQA 上不如 TA（74.06 vs 76.13），因为抑制共享方向也减弱了对 TextVQA 有益的共享改进。

3. **wudi2 是综合最好的**：TextVQA 76.01, ChartQA 68.44, OCRVQA 46.29。wudi2 通过优化找到了更好的平衡点，但仍然在 ChartQA 上低于 base（69.72）。

4. **ETVD-norm 和 TA-norm（per-layer norm matching）是有害的**：TextVQA 从 74.23 降到 ~70。per-layer 强制匹配 norm 太激进，破坏了层间的相对比例。

### 7.11 wudi2 Exclude-One 实验的深层解读

| Excluded Expert | ChartQA | 解释 |
|----------------|---------|------|
| None (all 5) | 68.44 | 基线 |
| -Geometry | **70.32** | ↑ Geometry 和 Chart 冲突 |
| -OCR | **70.60** | ↑ OCR 的大 norm 干扰了 Chart |
| -VQA | **70.68** | ↑ VQA 最大 norm，最大干扰源 |
| -Grounding | 69.24 | 轻微改善 |
| -Chart | 62.96 | ↓ 去掉相关 expert 则大降 |

**核心发现**：去掉 OCR/VQA/Geometry 中的任何一个，ChartQA 都 **超过 base（69.72）**。这意味着 Chart expert 本身可以提升 ChartQA，但 **其他 expert 的干扰拖了后腿**。

**解释**：

- VQA norm 最大（27.31），OCR 次之（21.64），它们的合并贡献淹没了 Chart（15.18）的信号

- Geometry 和 Chart 在某些方向冲突（可能共享"空间推理"相关的参数但方式不同）

- 去掉 Grounding 改善最小，因为 Grounding norm 最小（12.83）且与 Chart 重叠最少

---

## 8. 从 ETVD 到微能力分解

### 8.1 ETVD 的粒度局限

ETVD 将每个 expert 的 task vector（一个 $m \times n$ 矩阵）flatten 成一个 $d$ 维向量，堆叠成 $N \times d$ 矩阵做 SVD。结果只有 $N = 5$ 个基底方向。

但每个 task vector 矩阵本身是有结构的：

$$\tau_i \in \mathbb{R}^{m \times n}, \quad \tau_i = \sum_{j=1}^{r_i} \sigma_{ij} \, u_{ij} \, v_{ij}^T$$

有效秩 $r_i$ 可能有几十到几百。也就是说，一个 expert 包含 **多个独立的修改方向**（微能力/micro-capabilities）。

ETVD 的 5 个基底方向把所有微能力打包成原子，只能做 expert 级别的分析。

### 8.2 微能力分解的数学

**第一步：每个 expert 的 individual SVD**

$$\tau_i = U_i \Sigma_i V_i^T = \sum_{j=1}^{r_i} \sigma_{ij} \, u_{ij} \, v_{ij}^T$$

每个 rank-1 成分 $(\sigma_{ij}, u_{ij}, v_{ij})$ 是一个"微能力"：

- $v_{ij} \in \mathbb{R}^n$：输入方向（"关注什么输入特征"）

- $u_{ij} \in \mathbb{R}^m$：输出方向（"产生什么输出变化"）

- $\sigma_{ij}$：强度

**第二步：跨 expert 微能力关系度量**

对来自不同 expert 的两个微能力 $(i,j)$ 和 $(i',j')$：

$$\text{interference}((i,j), (i',j')) = (v_{ij}^T v_{i'j'}) \cdot (u_{ij}^T u_{i'j'}) \cdot \sigma_{ij} \cdot \sigma_{i'j'}$$

这个度量的因子结构：

| 输入重叠 $v^T v$ | 输出重叠 $u^T u$ | 意义 | 结果 |
|:---:|:---:|---|---|
| 低 | 低 | 独立 | interference ≈ 0 |
| 高 | 高，同号 | 冗余 | interference > 0（大）|
| 高 | 高，反号 | 冲突 | interference < 0（大）|
| 高 | 低 | 输入竞争 | interference ≈ 0 |
| 低 | 高 | 输出竞争 | interference ≈ 0 |

**关键区别**：

- ETVD 只知道"OCR 和 Chart 整体上 cosine = 0.017"

- 微能力分解能知道"OCR 的第 3 个微能力和 Chart 的第 7 个微能力在输入方向上高度重叠（v 的 cosine = 0.85），输出方向冲突（u 的 cosine = -0.72），interference = -0.61 × σ₃ × σ₇"

### 8.3 Individual SVD 和 Joint SVD 的数学关系

将每个微能力 flatten 到参数空间中：

$$\operatorname{vec}(\sigma_{ij} \, u_{ij} \, v_{ij}^T) = \sigma_{ij} \cdot (u_{ij} \otimes v_{ij}) \in \mathbb{R}^{mn}$$

其中 $\otimes$ 是 Kronecker 积。

Joint ETVD 和 Individual SVD 张成同一个子空间：

$$\operatorname{span}\{\operatorname{vec}(\tau_1), \ldots, \operatorname{vec}(\tau_N)\} = \operatorname{span}\{\sigma_{ij} \cdot (u_{ij} \otimes v_{ij}) : \text{all } i, j\}$$

但 Individual SVD 的基底 $\{u_{ij} \otimes v_{ij}\}$ 保留了输入/输出的因子结构，而 Joint ETVD 的基底 $\{v_k\}$ 把这个结构 flatten 掉了。

**类比**：Joint ETVD 给的是"混合颜色"（把红蓝混成紫），Individual SVD 给的是原始"色素分量"（红和蓝分开看）。两者描述同一个色域，但分量的物理意义不同。

### 8.4 从 ETVD 到微能力的层级管线

```

Step 1: ETVD Joint SVD (5 个宏观方向)

→ U 矩阵 → 发现 expert 分组结构

→ 哪些 expert 共享哪些宏观方向

Step 2: 基于分组，ANOVA 式层级分解

→ τ_i = 全局共享 μ + 子组偏差 (μ_g - μ) + 个体私有 (τ_i - μ_g)

→ 比 wudi2 的全局 avg 更精确

Step 3: 对各组分，Individual SVD 提取微能力

→ K 个微能力的 (u, σ, v) triplet

→ K >> 5, 但 K << d

Step 4: 跨 expert 微能力关系分析

→ K×K interference 矩阵

→ 精确识别冲突 pair

```

---

## 9. 统一视角与理论边界

### 9.1 各方法在 "分析 → 合并" 光谱上的位置

```

纯分析 纯优化

←──────────────────────────────────────────────→

TSV(度量) SVC(度量+后处理) ETVD(分析+规则) wudi2(优化)

```

- **TSV**：定义了 STI 度量，但合并策略（白化）与度量脱节

- **SVC**：度量了 over-accumulation，后处理缩放奇异值（比 TSV 更 actionable）

- **ETVD**：分析 U 矩阵的 occupancy 结构，直接推导合并规则（分析和合并统一）

- **wudi2**：不做显式分析，直接在全空间优化（最灵活但最不可解释）

### 9.2 线性假设下的理论完备性

在 Task Arithmetic 的线性框架下（$f(\theta_0 + \delta) \approx f(\theta_0) + Jf \cdot \delta$），任意 $k$-expert 交互都可以精确分解：

$$\sum_{i \in S} \tau_i = \sum_{k=1}^r \left(\sum_{i \in S} U_{ik}\right) \sigma_k v_k$$

不存在真正的高阶交互项——一切都是 U 矩阵中各 expert 在共同基底上的坐标叠加。

**ETVD 的 U 矩阵在线性框架下是完备的描述。**

### 9.3 根本性局限：参数空间 ≠ 功能空间

神经网络是参数的高度非线性函数。Loss 的 Taylor 展开：

$$L(\theta_0 + \sum_i \tau_i) = L(\theta_0) + \sum_i \nabla L^T \tau_i + \frac{1}{2} \sum_{i,j} \tau_i^T H \tau_j + O(\tau^3)$$

- 一阶项 $\nabla L^T \tau_i$：task-expert alignment（需要梯度数据）

- 二阶项 $\tau_i^T H \tau_j$：Hessian 加权的 pairwise 干扰（需要 Hessian）

- 三阶项：真正的三体交互

ETVD 的 U 矩阵捕捉的是**参数空间的几何**，不是功能空间的影响。两个 cosine=0.9 的 task vector，可能对某个 benchmark 是协同的，对另一个是冲突的——这需要 task-specific 的梯度信息才能区分。

**但这个局限是所有 data-free 方法共有的**——TSV、SVC、TIES、DARE、TA 全部隐式做了线性假设。

### 9.4 Sequential Merge 退化的解释

实验发现：先 merge 4 个 expert，再把 merged model 当 base 加第 5 个 expert，结果灾难性退化（TextVQA 从 76 降到 55-68）。

**数学原因**：第二步的 "task vector" 语义被污染。

$$\tau_5^{\text{new}} = \theta_5 - \theta_{\text{merged4}} = (\theta_0 + \tau_5) - (\theta_0 + \delta_{\text{merged4}}) = \tau_5 - \delta_{\text{merged4}}$$

第二步的 task vector 不再是纯粹的第 5 个能力增量，而是**混入了第一步 merged delta 的负像**。这是操作顺序导致的 task vector 语义污染，不是高阶干扰。

### 9.5 Data-free 方法的竞争格局

| 方法 | 显式干扰度量 | Over-accumulation | Sign Conflict | Under-representation | 核心操作 |
|------|:---:|:---:|:---:|:---:|---|
| TA | -- | -- | -- | -- | 直接求和 |
| TIES | -- | -- | 部分 | -- | Trim + elect sign |
| DARE | -- | -- | -- | -- | Random drop + rescale |
| TSV | STI | -- | 部分 | -- | 白化 |
| SVC | -- | 是 | -- | -- | 奇异值缩放 |
| **ETVD** | U-matrix | 是 | 是 | 是 | Occupancy-aware recon |

ETVD 是唯一同时处理三类 failure mode 并提供统一框架的方法。

---

## 10. 开放问题与研究方向

### 10.1 ETVD 的实验弱点

当前 ETVD 的实验结果不如 wudi2（尽管理论更清晰）。可能的原因：

1. **Occupancy 阈值敏感性**：$\epsilon = 0.3$ 是经验选择。阈值太高 → 误判 shared 为 unique；太低 → 误判 unique 为 shared。

2. **Conflict resolution 策略太粗糙**：majority vote 在少数 expert 正确的情况下会犯错。

3. **全局 scaling $\lambda$ 与 occupancy normalization 的交互**：ETVD 在 occupancy 内部做了归一化，外部还有 $\lambda$ 缩放，两层缩放可能不协调。

4. **Non-2D 层的处理**：ETVD 只处理 2D 权重矩阵，其他参数用简单平均，可能遗漏重要信息。

### 10.2 微能力分解的关键技术问题

1. **有效秩选择**：保留 90%/95%/99% 能量分别对应多少个微能力？不同 expert 的有效秩分布是否有系统性差异？

2. **SVD 旋转不确定性**：不同 expert 的奇异向量不天然对齐，直接用 cosine 比较可能不稳定。可能需要 Procrustes alignment。

3. **计算可扩展性**：K 个微能力的两两比较是 $O(K^2)$。如果 K=200（每 expert 40 个 × 5），需要 20000 次比较。

### 10.3 理论贡献的方向

1. **最优性证明**：如果假设"每个 expert 对 merged model 的贡献应等权"，可以证明 occupancy normalization 是唯一满足此公平性约束的线性合并策略。

2. **干扰的信息论界**：task vector 的互信息或 KL 散度视角。

3. **逐层传播分析**：即使每层线性叠加，层间复合引入非线性。分析 U 矩阵结构随层深度的变化如何影响最终输出。

### 10.4 ETVD-v2 和变体实验

| 变体 | 核心修改 | 假设 |
|------|---------|------|
| ETVD-v1 | shared 方向除以 occupancy | shared 是重复计算 |
| ETVD-v2 | 只干预 conflict，shared 保持 TA sum | shared 是有益共识 |
| ta_no_geometry | TA 去掉 Geometry | Geometry 主要贡献干扰 |
| ta_weighted_v1 | 每个 expert 不同权重 | norm 差异需要补偿 |

ETVD-v2 的逻辑：v1 对 shared 方向除以 occupancy 可能过于激进——如果所有 expert 一致认为某个方向应该加强，那这可能确实是一个重要的改进，不应该被缩小。v2 只对 conflict 方向做干预。

---

## 附录 A：关键公式汇总

**Task Arithmetic**:

$$\theta_{\text{TA}} = \theta_0 + \lambda \sum_{i=1}^N \tau_i$$

**ETVD Ensemble SVD**:

$$T = U\Sigma V^T, \quad T_{ij} = \operatorname{vec}(\tau_j)_i$$

**ETVD Reconstruction**:

$$\tau_{\text{ETVD}} = \sum_k w_k \sigma_k v_k, \quad w_k = \begin{cases} \sum_i U_{ik} & \text{unique} \\ \frac{\sum_i U_{ik}}{\operatorname{occ}(k)} & \text{shared} \\ \text{majority vote} & \text{conflict} \end{cases}$$

**wudi2 Loss**:

$$\mathcal{L} = \sum_i \frac{\|(m - \tilde{\tau}_i) L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

**Micro-capability Interference**:

$$I((i,j),(i',j')) = (v_{ij}^T v_{i'j'}) \cdot (u_{ij}^T u_{i'j'}) \cdot \sigma_{ij} \cdot \sigma_{i'j'}$$

**Spectral Over-Accumulation Ratio**:

$$R = \frac{\|\sum_i \tau_i\|_F^2}{\sum_i \|\tau_i\|_F^2} \quad (= 1 \text{ if orthogonal, } > 1 \text{ if over-accumulated})$$

---

## 附录 B：实验设置速查

- Base model: `OpenGVLab/InternVL2_5-1B`

- 5 experts: `yongxianwei/InternVL2_5-1B_{OCR,VQA,Geometry,Chart,Grounding}`

- 170 2D weight layers, 850 basis directions (5 per layer)

- Evaluation: TextVQA, OCRVQA, VizWiz, GQA, ChartQA, MathVision, MathVista

- VLMEvalKit for evaluation

- Merged models saved to `/data/shichao/data/InternVL_merged/`

- Code: `Skills/theory/{etvd_merge.py, etvd_analysis.py, scaling_sweep.py, merge_variants.py}`

---

## 附录 C：文件索引

| 文件 | 功能 |
|------|------|
| `ensemble_task_vector_decomposition.md` | ETVD 原始讨论记录（包含完整文献综述） |
| `micro_capability_decomposition.md` | 微能力分解的讨论记录 |
| `etvd_analysis.py` | 分析脚本（cosine, Gram, STI, over-accumulation, U-matrix） |
| `etvd_merge.py` | ETVD/TA/TA-norm merge 实现 |
| `merge_variants.py` | 实验变体（weighted TA, no-geometry 等） |
| `scaling_sweep.py` | 缩放系数 sweep |
| `results/analysis_report.md` | 完整分析报告 |
| `results/figures/*.png` | 所有分析图表 |
| `InternVL/internvl_chat/model_merging.py` | wudi/wudi2 原始实现 |

❯ 可以先记录一节，但是我们 concatenated task vectors 然后去做分析这一个环节感觉还挺有意义的，wudi2 是通过各自减去
avg，我们能够用更高阶的统计来组合共有和私有，或者说 compositional 能力，这个如何去激发

● 你说到点子上了。让我想清楚这个问题。

  wudi2 的分解只是 0 阶

  wudi2:  τ_i = avg + (τ_i - avg)
                ↑          ↑
           5 人平均     个体偏差

  这是最粗糙的分解。问题在哪：

  假设 OCR 和 VQA 共享"文本理解"，Chart 和 Geometry 共享"空间推理"。5 人平均把这两个不同的共享混在一起了。OCR 减去
  avg 后，它的"独有部分"其实还混着 "文本理解 - 平均" 的残留。

  avg 对 OCR 来说太多了（混入了 Chart/Geometry 的空间推理），对 Chart 来说也太多了（混入了 OCR/VQA
  的文本理解）。这不是一个好的共享/私有分解。

  我们的 ETVD 堆叠分析其实更有信息

  回看 ETVD 的 joint SVD：

  T = [τ_1; ...; τ_5] = UΣV^T

  U 矩阵 (5×5) 的每一列 u_k 告诉我们：
    u_k = [0.9, 0.8, 0.0, 0.0, 0.1]  →  方向 k 被 OCR 和 VQA 共享
    u_k = [0.0, 0.0, 0.7, 0.6, 0.0]  →  方向 k 被 Geometry 和 Chart 共享
    u_k = [0.0, 0.0, 0.0, 0.0, 0.9]  →  方向 k 是 Grounding 独有

  U 矩阵天然编码了分组信息——不需要预定义 group，它自动从数据中发现哪些 expert 共享哪些方向。这比 wudi2 的 avg
  有意义得多。

  但问题是只有 5 个方向。5 个方向够分组吗？对于 5 个 expert 来说，理论上刚好够（5 个方向、5 个 expert）。但它把每个
  expert 的内部结构压缩成了一个标量系数，丢掉了微能力的细节。

  真正的问题：怎么做到比 5 方向更丰富，同时保留分组发现的能力

  我觉得有几条路：

  路线 A：右奇异向量空间的跨 expert 分析

  不 flatten task vector，而是分析每个 expert 的 SVD 基向量之间的重叠：

  Expert i: τ_i = U_i Σ_i V_i^T
  取 top-k 右奇异向量: V_i ∈ R^{n × k}

  把所有 expert 的 V_i 列向量池化:
  V_pool = [v_{1,1}, ..., v_{1,k}, v_{2,1}, ..., v_{5,k}] ∈ R^{n × 5k}

  计算 5k × 5k 的 cosine similarity 矩阵
  → 聚类发现共享的输入方向

  同样对左奇异向量 U_i 做。这样就能发现：
  - OCR 的 v_3 和 VQA 的 v_7 cosine=0.92 → 同一个输入特征被两个 expert 修改
  - 再看对应的 u：方向一致 → 冗余修改；方向相反 → 冲突

  这给出了 K×K 的微能力关系图（K = 5k），比 ETVD 的 5×5 丰富得多。

  路线 B：层级分解（类似 ANOVA）

  不是简单 avg，而是多级分解：

  Level 0: μ = grand mean (所有 5 expert 的共有能力)
  Level 1: μ_g = group mean (子组共享能力)
            比如 {OCR, VQA} 共享文本理解, {Chart, Geometry} 共享空间推理
  Level 2: τ_i - μ_g(i) = individual private (个体独有能力)

  τ_i = μ + (μ_g(i) - μ) + (τ_i - μ_g(i))
        ↑         ↑                ↑
     全局共享   子组共享         个体私有

  wudi2 只有 Level 0 和 Level 2，跳过了 Level 1。但 Level 1 可能是最关键的——正是子组间的差异导致了干扰。

  问题：group 怎么定义？不用预设——用 ETVD 的 U 矩阵来发现！

  U 的列 u_k = [0.9, 0.8, 0.0, 0.0, 0.1]
  → 在方向 k 上，{expert 1, 2} 构成一个 group

  所以 ETVD 的 U 矩阵不只是分类工具，它可以作为发现 group 结构的手段，进而驱动层级分解。

  路线 C：保持矩阵结构的张量分解

  当前 ETVD 把 m×n 的 task vector flatten 成 mn 维向量再堆叠。这丢掉了 input/output 的因子结构。

  如果保持矩阵结构：

  T ∈ R^{N × m × n}   (3-way tensor)
  Tucker decomposition: T ≈ G ×₁ A ×₂ B ×₃ C
    A ∈ R^{N × r₁}:  expert 的组合系数
    B ∈ R^{m × r₂}:  output 基向量
    C ∈ R^{n × r₃}:  input 基向量
    G ∈ R^{r₁ × r₂ × r₃}: 核心张量

  这直接给出：
  - r₃ 个共享 input 方向（哪些输入特征被修改）
  - r₂ 个共享 output 方向（产生什么输出变化）
  - A 矩阵告诉我们每个 expert 怎么组合这些 input-output pair
  - G 编码了 input-output 方向之间的耦合

  "compositional capability" = 一个 (input direction, output direction) 的组合。 Tucker 分解天然给出这种组合结构。

  我的想法

  这三条路线不矛盾，可以统一成一个框架：

  Step 1: ETVD 的 joint SVD (5 方向) → 发现 expert 分组结构 (U 矩阵)
  Step 2: 基于分组，做层级分解 → 分离 全局共享 / 子组共享 / 个体私有
  Step 3: 对各组分，用 individual SVD 提取微能力 → 精细的 input/output 方向分析
  Step 4: 在微能力系数空间做优化 → 结构化的 merging

  ETVD 的 joint SVD 不只是一个 merging
  方法，它是整个分析流程的起点——发现分组结构。然后层级分解和微能力分析在此基础上深入。

## 11. 超越 wudi2：三条改进路线的完整推导

> **核心立场**：我们的分析发现（occupancy 结构、三类 failure mode、U-shaped depth pattern 等）已经足够构成实验贡献。但在方法上要超过 wudi2，不必追求闭式解——可以做优化，甚至直接在 wudi2 上改进。关键是用结构性分析来指导优化：更好的分解 → 更好的 loss → 更好的搜索空间。

### 11.0 wudi2 的三个可改进环节

回顾 wudi2 的完整流程：

```

输入: τ₁, ..., τ₅ ∈ R^{m×n}

(1) 分解: avg = mean(τᵢ), 个体偏差 = τᵢ - avg

(2) 构造 loss 的组件:

τ̃ᵢ = low-rank(τᵢ - avg) + avg ← 清洗后的 target

Lᵢ = top singular subspace of (τᵢ - avg) ← 投影方向

(3) 优化: m* = argmin Σᵢ ‖(m - τ̃ᵢ) Lᵢᵀ‖²/‖τᵢ‖² ← 全空间搜索

```

三个环节分别对应三个改进方向：

| 环节 | wudi2 的做法 | 问题 | 改进路线 |
|------|------------|------|---------|
| **(1) 分解** | 全局均值 avg（0 阶） | 混合了不同子组的共享 | **路线 B：ANOVA 层级分解** |
| **(2) Loss 构造** | per-expert 独立，不知跨 expert 关系 | 对冲突方向盲目折中 | **路线 A：微能力感知 loss** |
| **(3) 搜索空间** | 全参数空间 $m \times n$ | 冗余自由度太多 | **路线 C：结构化子空间搜索** |

下面逐一做完整推导。

---

### 11.1 路线 A：跨 Expert 微能力分析与干扰感知 Loss

#### 11.1.1 动机

wudi2 的 loss 函数对每个 expert **独立**计算干扰：

$$\mathcal{L}_{\text{wudi2}} = \sum_{i=1}^N \frac{\|(m - \tilde{\tau}_i) L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

每一项 $\mathcal{L}_i$ 只关心 $m$ 和 expert $i$ 之间的偏差。当 OCR 和 Chart 在某个方向上冲突时，wudi2 同时收到两个矛盾的梯度信号——"往 +v 方向走"和"往 -v 方向走"——然后在两者之间做盲目折中。

**如果我们事先知道哪些微能力是冲突的**，就可以在 loss 中显式处理：给冲突方向更低的权重（让优化器不在这里纠结），或在 target 中预先解决冲突（让优化器看到一致的目标）。

#### 11.1.2 第一步：提取微能力

对每个 expert $i$，每个 2D 权重层：

$$\tau_i = U_i \Sigma_i V_i^T = \sum_{j=1}^{r_i} \sigma_{ij} \, u_{ij} \, v_{ij}^T$$

保留 top-$k_i$ 个奇异分量（按累积能量比选择阈值，如 95%）：

$$\tau_i^{(k_i)} = \sum_{j=1}^{k_i} \sigma_{ij} \, u_{ij} \, v_{ij}^T$$

每个被保留的 rank-1 成分 $(\sigma_{ij}, u_{ij}, v_{ij})$ 就是一个微能力。$N$ 个 expert 共产生 $K = \sum_i k_i$ 个微能力。

**注意**：这里 $u_{ij} \in \mathbb{R}^m$ 是左奇异向量（输出方向），$v_{ij} \in \mathbb{R}^n$ 是右奇异向量（输入方向）。它们是 task vector 矩阵 $\tau_i$ 的 SVD，不是 ETVD 的 joint SVD。

#### 11.1.3 第二步：构建 K×K 微能力干扰矩阵

**定义微能力干扰度量**：

对任意两个微能力 $a = (i, j)$（expert $i$ 的第 $j$ 个微能力）和 $b = (i', j')$：

$$I_{ab} = \underbrace{(v_a^T v_b)}_{\text{输入重叠}} \cdot \underbrace{(u_a^T u_b)}_{\text{输出重叠}} \cdot \underbrace{\sigma_a \sigma_b}_{\text{能量权重}}$$

**逐步推导为什么这是正确的干扰度量**：

考虑两个 rank-1 修改 $\Delta_a = \sigma_a u_a v_a^T$ 和 $\Delta_b = \sigma_b u_b v_b^T$ 同时叠加。它们的合并结果 $\Delta_a + \Delta_b$ 作用于输入 $x$：

$$(\Delta_a + \Delta_b) x = \sigma_a (v_a^T x) u_a + \sigma_b (v_b^T x) u_b$$

**情况 1：输入正交**（$v_a^T v_b = 0$）。

两个修改关注不同的输入方向。对任何输入 $x$，$v_a^T x$ 和 $v_b^T x$ 是独立的投影——两个修改互不影响。干扰 = 0。

**情况 2：输入重叠高但输出正交**（$|v_a^T v_b| \approx 1$, $u_a^T u_b \approx 0$）。

两个修改关注同一输入特征但产生正交的输出变化。等效于在同一个输入上施加两个不同维度的输出修改——各走各的，不干扰。干扰 = 0。

**情况 3：输入和输出都重叠**（$v_a^T v_b \approx 1$, $u_a^T u_b \approx 1$）。

两个修改关注同一输入特征且产生同方向的输出变化：

$$\Delta_a x + \Delta_b x \approx (\sigma_a + \sigma_b)(v_a^T x) u_a$$

效果被叠加放大——**冗余**。$I_{ab} > 0$ 且大。

**情况 4：输入重叠但输出反向**（$v_a^T v_b \approx 1$, $u_a^T u_b \approx -1$）。

$$\Delta_a x + \Delta_b x \approx (\sigma_a - \sigma_b)(v_a^T x) u_a$$

效果被相互抵消——**冲突**。$I_{ab} < 0$ 且绝对值大。

所以 $I_{ab}$ 的符号和大小精确编码了两个微能力之间的干扰类型和强度。

**矩阵形式**：

定义右奇异向量矩阵（所有微能力的输入方向）：

$$\mathcal{V} = [v_1, v_2, \ldots, v_K] \in \mathbb{R}^{n \times K}$$

定义左奇异向量矩阵（所有微能力的输出方向）：

$$\mathcal{U} = [u_1, u_2, \ldots, u_K] \in \mathbb{R}^{m \times K}$$

定义奇异值对角矩阵：

$$\mathcal{S} = \operatorname{diag}(\sigma_1, \ldots, \sigma_K) \in \mathbb{R}^{K \times K}$$

则 $K \times K$ 干扰矩阵为：

$$\boxed{I = \mathcal{S} \cdot (\mathcal{V}^T \mathcal{V}) \odot (\mathcal{U}^T \mathcal{U}) \cdot \mathcal{S}}$$

其中 $\odot$ 是 Hadamard（逐元素）乘积。

**计算复杂度**：$\mathcal{V}^T \mathcal{V}$ 和 $\mathcal{U}^T \mathcal{U}$ 各需要 $O(K^2 n)$ 和 $O(K^2 m)$。如果 $K = 200$（每 expert 平均 40 个微能力 × 5），$n = m = 2048$，则 $\sim 2 \times 200^2 \times 2048 \approx 1.6 \times 10^8$ 次乘法。可在 GPU 上秒级完成。

#### 11.1.4 第三步：干扰矩阵的谱分析与聚类

干扰矩阵 $I \in \mathbb{R}^{K \times K}$ 编码了所有微能力之间的关系。可以用谱方法提取结构：

**对 $I$ 做特征分解**：

$$I = Q \Lambda Q^T$$

由于 $I$ 是实对称矩阵（$I_{ab} = I_{ba}$），特征值 $\lambda_1 \geq \lambda_2 \geq \cdots$ 是实数。

- **正特征值对应的特征向量**：编码了"冗余簇"（一组互相冗余的微能力）

- **负特征值对应的特征向量**：编码了"冲突簇"（一组互相冲突的微能力）

- **接近零的特征值**：对应的微能力组彼此独立

**图聚类视角**：

将 $I$ 视为一个带符号加权图（signed weighted graph）：

- 节点 = 微能力

- 边权 = $I_{ab}$

- 正边 = 冗余

- 负边 = 冲突

用 signed graph clustering（如 signed spectral clustering, SPONGE 算法）将 K 个微能力分成若干簇，簇内边权为正（互相冗余），簇间边权为负（互相冲突）。

#### 11.1.5 第四步：改进 wudi2 的 loss

有了微能力干扰矩阵，有三种方式改进 wudi2：

**方案 A：干扰感知的 target 清洗**

wudi2 的 target $\tilde{\tau}_i$ 是通过减去全局 avg 再低秩近似得到的。我们可以用微能力干扰矩阵来做更精准的清洗：

对 expert $i$ 的每个微能力 $j$，计算它和**其他 expert** 的微能力之间的总冲突：

$$\text{conflict}_j^{(i)} = \sum_{\substack{(i', j') \\ i' \neq i}} \max(0, -I_{(i,j),(i',j')})$$

如果 $\text{conflict}_j^{(i)}$ 很大，说明这个微能力和其他 expert 严重冲突。在 target 中抑制它：

$$\tilde{\tau}_i^{\text{new}} = \sum_{j=1}^{k_i} \underbrace{(1 - \beta \cdot \hat{c}_j^{(i)})}_{\text{冲突抑制系数}} \cdot \sigma_{ij} \, u_{ij} \, v_{ij}^T$$

其中 $\hat{c}_j^{(i)} = \text{conflict}_j^{(i)} / \max_j \text{conflict}_j^{(i)}$ 是归一化的冲突度，$\beta \in [0, 1]$ 是抑制强度超参数。

**直觉**：在给 wudi2 的优化器呈现 target 之前，先把和其他 expert 冲突最严重的微能力削弱。这样优化器看到的 5 个 target 之间的矛盾更少，更容易找到好的折中。

**方案 B：干扰感知的投影空间**

wudi2 把干扰投影到 $L_i^T$（expert $i$ 的主子空间）上。我们可以用干扰矩阵修改投影权重：

$$L_i^{\text{new}} = \operatorname{diag}(w_1^{(i)}, \ldots, w_{k_i}^{(i)}) \cdot L_i$$

其中权重 $w_j^{(i)}$ 根据该微能力的冗余度和冲突度确定：

$$w_j^{(i)} = \frac{\sigma_{ij}}{1 + \gamma \cdot \text{redundancy}_j^{(i)}}$$

$$\text{redundancy}_j^{(i)} = \sum_{\substack{(i', j') \\ i' \neq i}} \max(0, I_{(i,j),(i',j')})$$

**直觉**：如果一个微能力和其他 expert 高度冗余，那么在 loss 中降低它的投影权重——即使 $m$ 在这个方向上偏离 $\tilde{\tau}_i$，也没关系，因为其他 expert 已经覆盖了。这让优化器把注意力集中在 expert $i$ 真正独有的微能力上。

**方案 C：对冲突微能力做预解决（pre-resolution）**

不修改 loss 函数，而是在**进入优化之前**预处理 task vector：

对被识别为冲突的微能力对 $(a, b)$（$I_{ab} < -\text{threshold}$）：

1. 如果 $\sigma_a > \sigma_b$：保留 $a$，从 expert $b$ 的 task vector 中移除 $b$ 的投影

2. 如果 $\sigma_a \approx \sigma_b$：两者都缩放 50%

3. 或者：取两者的加权平均方向

移除微能力 $b$ 从 $\tau_{i'}$ 中的操作：

$$\tau_{i'}^{\text{cleaned}} = \tau_{i'} - \sigma_{i'j'} \, u_{i'j'} \, v_{i'j'}^T$$

用清洗后的 $\{\tau_i^{\text{cleaned}}\}$ 代替原始 $\{\tau_i\}$ 输入 wudi2。

**这是最简单的改进路径——不改 wudi2 的代码，只改它的输入。**

#### 11.1.6 完整算法：微能力感知 wudi2（MC-wudi2）

```

输入: base_params θ₀, expert_params θ₁...θ₅

超参: k_i (每 expert 保留微能力数), β (冲突抑制强度)

对每个 2D 权重层:

1. 计算 task vectors: τᵢ = θᵢ - θ₀

2. Individual SVD + 微能力提取:

对每个 i: τᵢ = Σⱼ σᵢⱼ uᵢⱼ vᵢⱼᵀ, 保留 top-kᵢ

3. 构建 K×K 干扰矩阵:

I_{ab} = (vₐᵀvᵦ)(uₐᵀuᵦ)σₐσᵦ

4. 预解决冲突:

对每个 expert i 的每个微能力 j:

conflict_j = Σ_{other (i',j')} max(0, -I_{(i,j),(i',j')})

α_j = max(0, 1 - β · conflict_j / max_conflict)

τ̃ᵢ = Σⱼ αⱼ σᵢⱼ uᵢⱼ vᵢⱼᵀ ← 冲突感知的清洗 target

5. 构造 wudi2 的优化组件（用 τ̃ᵢ 替代原始 τᵢ 的角色）:

avg = mean(τ̃₁, ..., τ̃₅)

Lᵢ = top subspace of (τ̃ᵢ - avg)

6. wudi2 优化:

m* = argmin Σᵢ ‖(m - τ̃ᵢ) Lᵢᵀ‖²/‖τ̃ᵢ‖²

(Adam, 300 steps, lr=1e-5)

7. merged_param = θ₀ + λ · m*

```

---

### 11.2 路线 B：ANOVA 式层级分解

#### 11.2.1 动机

wudi2 的共享/独有分解是：

$$\tau_i = \underbrace{\bar{\tau}}_{\text{全局均值}} + \underbrace{(\tau_i - \bar{\tau})}_{\text{个体偏差}}$$

这是 0 阶分解——假设只有一个共享成分。但我们的 ETVD 分析发现 U 矩阵中存在**分组结构**：某些方向只被 {OCR, VQA} 共享，另一些只被 {Chart, Geometry} 共享。

类比统计学中的 ANOVA（方差分析），我们可以做多级分解：

$$\tau_i = \underbrace{\mu}_{\text{Level 0: 全局}} + \underbrace{(\mu_{g(i)} - \mu)}_{\text{Level 1: 子组效应}} + \underbrace{(\tau_i - \mu_{g(i)})}_{\text{Level 2: 个体残差}}$$

其中 $g(i)$ 是 expert $i$ 所属的组，$\mu_{g}$ 是组均值。

#### 11.2.2 第一步：用 ETVD 的 U 矩阵发现分组

对每个层的 $U^l$ 矩阵（$5 \times 5$），分析各列的结构来发现哪些 expert 构成子组。

**形式化定义**：

ETVD 给出 $T^l = U^l \Sigma^l (V^l)^T$。每一列 $u_{:,k}$ 定义了一个基底方向 $v_k$ 上的 expert 参与模式。

定义 **方向-Expert 关联矩阵** $A \in \{0, 1\}^{5 \times 5}$：

$$A_{ik} = \mathbb{1}[|U_{ik}| > \epsilon]$$

（和 occupancy 分析用的相同阈值 $\epsilon$）。

对方向 $k$，参与的 expert 集合为 $\mathcal{S}_k = \{i : A_{ik} = 1\}$。

**分组发现**：将占据度 $\geq 2$ 的方向（即 shared 和 conflict 方向）的 expert 集合 $\mathcal{S}_k$ 提取出来，这些集合就是"子组"的候选。

**跨层聚合**：单层的 U 矩阵可能有噪声。跨所有 170 层聚合：

$$\text{affinity}(i, i') = \frac{1}{L} \sum_{l=1}^{L} \sum_{k: \text{occ}(k) \geq 2} A_{ik}^l \cdot A_{i'k}^l$$

这给出一个 $5 \times 5$ 的 expert 亲和矩阵：$\text{affinity}(i, i')$ 计数的是 expert $i$ 和 $i'$ 在多少个层的多少个方向上共同出现。

对这个亲和矩阵做层次聚类（hierarchical clustering），得到 expert 的树状分组。

**实验中的预期**（根据分析报告）：

根据 cosine similarity 和 STI 数据，预期的分组为：

- 组 1: {OCR, Grounding}（cosine=0.141，最高亲和）

- 组 2: {VQA}（和所有人都有一定关联但不强烈）

- 组 3: {Chart}

- 组 4: {Geometry}（最孤立）

或者更粗糙地：

- 组 A: {OCR, VQA, Grounding}（文本/定位相关）

- 组 B: {Chart, Geometry}（空间推理相关）

具体的分组会由数据决定，不需要预设。

#### 11.2.3 第二步：层级分解的数学推导

假设 ETVD 发现了 $G$ 个子组 $\mathcal{G}_1, \ldots, \mathcal{G}_G$。定义各级均值：

**Level 0：全局共享**

$$\mu = \frac{1}{N} \sum_{i=1}^N \tau_i$$

**Level 1：子组均值**

$$\mu_g = \frac{1}{|\mathcal{G}_g|} \sum_{i \in \mathcal{G}_g} \tau_i, \quad g = 1, \ldots, G$$

**分解**：

$$\tau_i = \mu + (\mu_{g(i)} - \mu) + (\tau_i - \mu_{g(i)})$$

验证等式成立：

$$\mu + (\mu_{g(i)} - \mu) + (\tau_i - \mu_{g(i)}) = \tau_i \quad \checkmark$$

**三个组分的含义**：

- $\mu$：所有 5 个 expert 都做的修改。这是"通用能力增强"。

- $\mu_{g(i)} - \mu$：expert $i$ 所在子组相对于全局均值的偏差。这是"子组共享的特殊能力"——比如 {OCR, Grounding} 组共享的空间定位增强，扣掉了全局共享的部分。

- $\tau_i - \mu_{g(i)}$：expert $i$ 相对于自己组均值的偏差。这是纯粹的"个体私有能力"。

**与 wudi2 的对比**：

wudi2 的分解：

$$\tau_i = \bar{\tau} + (\tau_i - \bar{\tau})$$

ANOVA 式分解：

$$\tau_i = \mu + \underbrace{(\mu_{g(i)} - \mu)}_{\text{wudi2 缺失的层}} + (\tau_i - \mu_{g(i)})$$

wudi2 把 Level 1（子组效应）混入了 Level 2（个体残差）。结果是：

1. 个体残差被子组效应污染——低秩近似 $\tau_i - \bar{\tau}$ 中包含了子组共享成分

2. 全局均值被子组差异稀释——$\bar{\tau}$ 不是任何一个子组的好估计

#### 11.2.4 第三步：正交性证明

**命题**：三个组分之间两两正交（在 Frobenius 内积意义下）。

**证明**：

定义 Frobenius 内积 $\langle A, B \rangle_F = \operatorname{tr}(A^T B) = \sum_{pq} A_{pq} B_{pq}$。

**$\mu$ 和 $(\mu_g - \mu)$ 的正交性**：

$$\sum_{i=1}^N \langle \mu, \mu_{g(i)} - \mu \rangle_F = N \langle \mu, \frac{1}{N}\sum_i \mu_{g(i)} - \mu \rangle_F$$

需要证明 $\frac{1}{N}\sum_i \mu_{g(i)} = \mu$。

$$\frac{1}{N}\sum_{i=1}^N \mu_{g(i)} = \frac{1}{N} \sum_{g=1}^G |\mathcal{G}_g| \cdot \mu_g = \frac{1}{N} \sum_{g=1}^G |\mathcal{G}_g| \cdot \frac{1}{|\mathcal{G}_g|}\sum_{i \in \mathcal{G}_g} \tau_i = \frac{1}{N}\sum_{i=1}^N \tau_i = \mu \quad \checkmark$$

所以加权平均意义下，Level 1 与 Level 0 正交。

**类似地**，$(\mu_g - \mu)$ 和 $(\tau_i - \mu_g)$ 在组内均值意义下正交。

**意义**：这个正交分解保证了三个层级的能量可加——不会因为分解方式引入伪信号或丢失真实信号：

$$\sum_{i=1}^N \|\tau_i\|_F^2 = N\|\mu\|_F^2 + \sum_{g=1}^G |\mathcal{G}_g| \|\mu_g - \mu\|_F^2 + \sum_{i=1}^N \|\tau_i - \mu_{g(i)}\|_F^2$$

#### 11.2.5 第四步：将层级分解注入 wudi2

**改进 1：更精确的 target**

wudi2 的 target 是 $\tilde{\tau}_i = \text{low-rank}(\tau_i - \bar{\tau}) + \bar{\tau}$。替换为：

$$\tilde{\tau}_i^{\text{ANOVA}} = \text{low-rank}(\tau_i - \mu_{g(i)}) + \mu_{g(i)}$$

这里 $\tau_i - \mu_{g(i)}$ 是更纯净的个体残差（不含子组效应），低秩近似更精确。$\mu_{g(i)}$ 是更精确的共享估计（只包含同组 expert）。

**改进 2：分层合并策略**

不同层级可以用不同的合并策略：

$$m = \underbrace{\alpha_0 \cdot \mu}_{\text{全局共享：直接加}} + \underbrace{\sum_g \alpha_g \cdot (\mu_g - \mu)}_{\text{子组效应：每组一个系数}} + \underbrace{m_{\text{residual}}}_{\text{个体残差：wudi2 优化}}$$

其中：

- $\alpha_0$：全局共享的缩放系数。所有 expert 都同意的修改应该完整保留，所以 $\alpha_0$ 可以接近 1。

- $\alpha_g$：子组效应的缩放系数。每个子组各自控制。

- $m_{\text{residual}}$：只对个体残差部分做 wudi2 优化。

**关键优势**：wudi2 的搜索空间从 $\mathbb{R}^{m \times n}$ 缩小到个体残差的子空间。全局共享和子组效应由分析直接确定，不需要优化。

**改进 3：组内 over-accumulation 控制**

在每个子组内部，仍然可能有 over-accumulation。对子组 $g$ 内的 $|\mathcal{G}_g|$ 个 expert，做子组内的 ETVD：

$$T_g = [\operatorname{vec}(\tau_i - \mu_{g})]_{i \in \mathcal{G}_g} \in \mathbb{R}^{|\mathcal{G}_g| \times d}$$

$$T_g = U_g \Sigma_g V_g^T$$

$U_g$ 矩阵给出子组内的 occupancy 结构，可以对组内共享方向做 occupancy normalization。

#### 11.2.6 完整算法：ANOVA-wudi2

```

输入: base_params θ₀, expert_params θ₁...θ₅

超参: ε (occupancy 阈值), α₀ (全局缩放)

预处理（所有层一起做）:

1. 对每个 2D 层 l:

a. ETVD joint SVD → U^l 矩阵

b. 计算关联矩阵 A^l_{ik} = 1[|U^l_{ik}| > ε]

2. 跨层聚合 affinity 矩阵 → 层次聚类 → 发现 G 个子组

对每个 2D 权重层:

3. 三级分解:

μ = mean(τ₁, ..., τ₅)

μ_g = mean(τᵢ for i ∈ G_g)

residual_i = τᵢ - μ_{g(i)}

4. 确定性合并全局和子组成分:

m_shared = α₀·μ + Σ_g α_g·(μ_g - μ)

5. 对残差做 wudi2 优化:

（对 {residual_i} 执行 wudi2 的 SVD + 优化流程）

→ m_residual

6. merged_param = θ₀ + m_shared + λ·m_residual

```

---

### 11.3 路线 C：结构化子空间搜索（Tucker 张量分解）

#### 11.3.1 动机

wudi2 在 $\mathbb{R}^{m \times n}$ 全空间搜索 $m$，有 $m \times n$ 个自由参数。这远远超过实际的自由度——task vector 天然低秩（top 10% 奇异分量承载 99% 能量），搜索空间的大部分是无用的。

关键 insight：**搜索空间应该被限制在 task vectors 张成的子空间中**，因为 merged vector 不应该在 task vectors 的正交补空间中引入新的修改方向——那里没有任何 expert 提供的信号。

#### 11.3.2 第一步：矩阵结构的联合低秩基底

当前 ETVD 把 $\tau_i \in \mathbb{R}^{m \times n}$ flatten 成向量，丢掉了矩阵结构。我们可以保留矩阵结构，构造联合的低秩基底。

**定义联合输出基底**：

将 $N$ 个 task vector 的左奇异向量池化。每个 expert 的 SVD：

$$\tau_i = U_i \Sigma_i V_i^T$$

提取 top-$k$ 左奇异向量（输出方向）并池化：

$$\mathcal{U}_{\text{pool}} = [U_1^{(k)}, U_2^{(k)}, \ldots, U_N^{(k)}] \in \mathbb{R}^{m \times Nk}$$

其中 $U_i^{(k)} = [u_{i1}, \ldots, u_{ik}] \in \mathbb{R}^{m \times k}$。

对 $\mathcal{U}_{\text{pool}}$ 做 SVD 或 QR 分解，得到正交化的联合输出基底：

$$\mathcal{U}_{\text{pool}} = P_U \cdot R_U, \quad P_U \in \mathbb{R}^{m \times r_U}$$

其中 $r_U \leq Nk$ 是联合输出子空间的秩，$P_U$ 的列构成正交基底。

**类似地定义联合输入基底**：

$$\mathcal{V}_{\text{pool}} = [V_1^{(k)}, \ldots, V_N^{(k)}] \in \mathbb{R}^{n \times Nk}$$

$$\mathcal{V}_{\text{pool}} = P_V \cdot R_V, \quad P_V \in \mathbb{R}^{n \times r_V}$$

#### 11.3.3 第二步：将搜索空间限制为联合基底上的矩阵

任何由 task vectors 张成的矩阵都可以表示为：

$$m = P_U \cdot C \cdot P_V^T$$

其中 $C \in \mathbb{R}^{r_U \times r_V}$ 是在联合基底下的坐标矩阵。

**搜索空间大小对比**：

| 方法 | 搜索空间维度 | 例子 ($m=n=2048$, $N=5$, $k=40$) |
|------|-----------|------|
| wudi2 | $m \times n$ | $2048 \times 2048 = 4,194,304$ |
| 联合基底 | $r_U \times r_V$ | $\leq 200 \times 200 = 40,000$ |

缩小了 **100 倍**！且没有丢失信息——merged vector 不应该在 task vectors 子空间之外有分量。

#### 11.3.4 第三步：Tucker 张量分解的完整推导

将 $N$ 个 task vector 视为 3-way 张量 $\mathcal{T} \in \mathbb{R}^{N \times m \times n}$：

$$\mathcal{T}_{i,:,:} = \tau_i$$

**Tucker 分解**：

$$\mathcal{T} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$

逐步展开这个公式。

**符号定义**：

- $A \in \mathbb{R}^{N \times r_1}$：expert 模式矩阵（哪个 expert 参与什么组合）

- $B \in \mathbb{R}^{m \times r_2}$：输出基底矩阵（$r_2$ 个输出方向）

- $C \in \mathbb{R}^{n \times r_3}$：输入基底矩阵（$r_3$ 个输入方向）

- $\mathcal{G} \in \mathbb{R}^{r_1 \times r_2 \times r_3}$：核心张量（编码三者的耦合）

**n-mode product 定义**：

$\mathcal{T} \times_1 A$ 表示沿第 1 个模式（expert 维度）用矩阵 $A$ 做线性变换：

$$(\mathcal{T} \times_1 A)_{j,p,q} = \sum_{i=1}^N A_{ij} \mathcal{T}_{ipq}$$

完整的 Tucker 展开：

$$\mathcal{T}_{i,p,q} \approx \sum_{\alpha=1}^{r_1} \sum_{\beta=1}^{r_2} \sum_{\gamma=1}^{r_3} \mathcal{G}_{\alpha\beta\gamma} \cdot A_{i\alpha} \cdot B_{p\beta} \cdot C_{q\gamma}$$

即第 $i$ 个 task vector 的 $(p, q)$ 元素由 $(A, B, C, \mathcal{G})$ 四个因子的组合给出。

**将张量分量翻译回矩阵语言**：

$$\tau_i \approx \sum_{\alpha=1}^{r_1} A_{i\alpha} \cdot \underbrace{\left(\sum_{\beta, \gamma} \mathcal{G}_{\alpha\beta\gamma} \, b_\beta \, c_\gamma^T\right)}_{\text{compositional capability } \alpha}$$

即每个 task vector 是 $r_1$ 个 **组合能力**（compositional capability）的线性组合，$A_{i\alpha}$ 是 expert $i$ 对第 $\alpha$ 个能力的参与系数。

每个组合能力本身是 $r_2 \times r_3$ 个 input-output pair 的组合：

$$\text{capability}_\alpha = \sum_{\beta=1}^{r_2} \sum_{\gamma=1}^{r_3} \mathcal{G}_{\alpha\beta\gamma} \cdot b_\beta \cdot c_\gamma^T$$

**和 ETVD 的关系**：

ETVD 的 joint SVD 相当于 Tucker 分解在特殊情况下的退化：

$$\text{ETVD}: \quad \tau_i \approx \sum_{\alpha=1}^{r} U_{i\alpha} \sigma_\alpha v_\alpha^T$$

对比 Tucker：

$$\text{Tucker}: \quad \tau_i \approx \sum_{\alpha=1}^{r_1} A_{i\alpha} \cdot \text{capability}_\alpha$$

ETVD 中每个"方向" $v_\alpha$ 是 $d = mn$ 维向量（flatten 后），而 Tucker 中每个 capability 是 $m \times n$ 矩阵，保留了输入/输出的因子结构。

**当 $r_2 = r_3 = 1$（每个 capability 只有一个 input-output pair）时，Tucker 退化为 CP 分解**：

$$\tau_i \approx \sum_{\alpha} A_{i\alpha} \cdot \mathcal{G}_\alpha \cdot b_\alpha \cdot c_\alpha^T$$

此时每个 capability 就是一个 rank-1 矩阵 $b_\alpha c_\alpha^T$，和 ETVD 的 $v_k$（reshape 成矩阵后）形式一致。

#### 11.3.5 第四步：在 Tucker 坐标下优化

在 Tucker 基底下，merged task vector 可以表示为：

$$m = B \cdot X \cdot C^T, \quad X \in \mathbb{R}^{r_2 \times r_3}$$

优化变量是 $X$（$r_2 \times r_3$ 个参数），而不是 $m$（$m \times n$ 个参数）。

**wudi2 的 loss 在 Tucker 坐标下的重写**：

$$\mathcal{L}(X) = \sum_{i=1}^N \frac{\|(BXC^T - \tilde{\tau}_i) L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

展开 $\tilde{\tau}_i$ 在 Tucker 基底下的坐标 $X_i$（即 $\tilde{\tau}_i \approx B X_i C^T$）：

$$\mathcal{L}(X) = \sum_{i=1}^N \frac{\|B(X - X_i)C^T L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

利用 $B$ 正交（$B^T B = I_{r_2}$）：

$$= \sum_{i=1}^N \frac{\|(X - X_i) C^T L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

这是一个 $r_2 \times r_3$ 维空间中的优化问题，比原始的 $m \times n$ 小得多。

**梯度推导**：

$$\frac{\partial \mathcal{L}}{\partial X} = 2 \sum_{i=1}^N \frac{(X - X_i) C^T L_i^T L_i C}{\|\tau_i\|_F^2}$$

令梯度为零（闭式解，如果 loss 是凸的）：

$$X^* \left(\sum_{i=1}^N \frac{C^T L_i^T L_i C}{\|\tau_i\|_F^2}\right) = \sum_{i=1}^N \frac{X_i C^T L_i^T L_i C}{\|\tau_i\|_F^2}$$

定义 $H = \sum_{i} \frac{C^T L_i^T L_i C}{\|\tau_i\|_F^2}$ 和 $b = \sum_{i} \frac{X_i C^T L_i^T L_i C}{\|\tau_i\|_F^2}$，则：

$$\boxed{X^* = b \cdot H^{-1}}$$

**这是一个矩阵方程，可以直接求解！不需要迭代优化。**

当然，这只在 Tucker 近似足够精确时才成立。实际中可以：

1. 先用闭式解得到 $X^*$ 作为初始化

2. 回到原空间 $m_0 = B X^* C^T$

3. 在 $m_0$ 附近做 wudi2 的 fine-tuning（原空间的局部优化）

#### 11.3.6 联合基底的实际构造

**方案 1：直接池化**（推荐，最简单）

```

对每个 2D 权重层:

1. 对每个 expert i 做 SVD: τᵢ = Uᵢ Σᵢ Vᵢᵀ

2. 取 top-k: Uᵢ^(k) ∈ R^{m×k}, Vᵢ^(k) ∈ R^{n×k}

3. 池化: U_pool = [U₁^(k), ..., U₅^(k)] ∈ R^{m×5k}

V_pool = [V₁^(k), ..., V₅^(k)] ∈ R^{n×5k}

4. 正交化: B = orth(U_pool) ∈ R^{m×r₂} (r₂ ≤ 5k)

C = orth(V_pool) ∈ R^{n×r₃} (r₃ ≤ 5k)

5. 投影各 expert 到联合基底: Xᵢ = Bᵀ τᵢ C ∈ R^{r₂×r₃}

```

**方案 2：HOSVD（Higher-Order SVD）**

直接对 3-way 张量 $\mathcal{T} \in \mathbb{R}^{5 \times m \times n}$ 做 HOSVD：

1. 沿模式 1（expert）展开：$T_{(1)} \in \mathbb{R}^{5 \times mn}$ → SVD 得 $A$

2. 沿模式 2（output）展开：$T_{(2)} \in \mathbb{R}^{m \times 5n}$ → SVD 得 $B$

3. 沿模式 3（input）展开：$T_{(3)} \in \mathbb{R}^{n \times 5m}$ → SVD 得 $C$

4. 核心张量：$\mathcal{G} = \mathcal{T} \times_1 A^T \times_2 B^T \times_3 C^T$

模式 2 展开的具体过程：

$$T_{(2)} = [\tau_1 | \tau_2 | \cdots | \tau_5] \in \mathbb{R}^{m \times 5n}$$

即把 5 个 $m \times n$ 矩阵水平拼接。对 $T_{(2)}$ 做 SVD：

$$T_{(2)} = B \Sigma_B W_B^T$$

$B$ 的 top-$r_2$ 列构成联合输出基底。

**注意**：HOSVD 的模式 1 SVD 和 ETVD 的 joint SVD 完全一致——$T_{(1)} = T_{\text{flat}}$（5×mn 矩阵），$A$ 就是 ETVD 的 $U$ 矩阵。所以 Tucker 分解是 ETVD 的自然推广。

#### 11.3.7 完整算法：Tucker-wudi2

```

输入: base_params θ₀, expert_params θ₁...θ₅

超参: k (per-expert 保留秩), rank_cut_ratio

对每个 2D 权重层:

1. 计算 task vectors: τᵢ = θᵢ - θ₀

2. 构造联合基底:

对每个 i: SVD(τᵢ) → top-k 左/右奇异向量

B = orth([U₁^(k), ..., U₅^(k)]) ∈ R^{m × r₂}

C = orth([V₁^(k), ..., V₅^(k)]) ∈ R^{n × r₃}

3. 投影到联合基底:

Xᵢ = Bᵀ τᵢ C ∈ R^{r₂ × r₃} (各 expert 的坐标)

Lᵢ_proj = Bᵀ Lᵢ C^(-T) (wudi2 投影子空间的坐标)

4a. 闭式解（如果 loss 是凸的）:

X* = (Σᵢ Xᵢ Hᵢ) (Σᵢ Hᵢ)⁻¹ where Hᵢ = C^T Lᵢ^T Lᵢ C / ‖τᵢ‖²

4b. 或 Adam 优化（在 r₂×r₃ 空间中）:

X = Parameter(mean(X₁,...,X₅)) ← 初始化为均值

loss = Σᵢ ‖(X - Xᵢ) Cᵀ Lᵢᵀ‖² / ‖τᵢ‖²

→ 300 步 Adam

5. 重建:

m = B X* Cᵀ ∈ R^{m × n}

6. merged_param = θ₀ + λ · m

```

---

### 11.4 三条路线的融合：统一方法设计

三条路线解决 wudi2 的不同环节，**可以叠加**：

```

┌──────────────────────────────────────────────────────┐

│ Unified Pipeline │

│ │

│ ┌─路线 B (ANOVA)──────────────┐ │

│ │ ETVD U-matrix → 发现子组 │ │

│ │ τᵢ = μ + (μ_g-μ) + residual│ │

│ │ → 更精确的 target 分解 │ │

│ └──────────┬──────────────────┘ │

│ ↓ residual_i │

│ ┌─路线 A (微能力)─────────────┐ │

│ │ Individual SVD → K 微能力 │ │

│ │ 干扰矩阵 I → 冲突检测 │ │

│ │ → 预解决冲突的 target │ │

│ └──────────┬──────────────────┘ │

│ ↓ cleaned residual_i │

│ ┌─路线 C (Tucker)─────────────┐ │

│ │ 联合基底 B, C → 低维搜索 │ │

│ │ X ∈ R^{r₂×r₃} 代替全空间 │ │

│ │ → 高效优化 / 闭式解 │ │

│ └──────────┬──────────────────┘ │

│ ↓ X* → m = BX*Cᵀ │

│ merged = θ₀ + λ(m_shared + m_residual) │

└──────────────────────────────────────────────────────┘

```

#### 11.4.1 融合算法的完整伪代码

```python

def structured_merge(base_sd, expert_sds, expert_names):

"""

Structured Merge: ANOVA + 微能力 + Tucker 联合方法

"""

N = len(expert_sds)

# ===== 全局预处理：发现 expert 分组 =====

# 对所有 2D 层做 ETVD，聚合 affinity 矩阵

affinity = zeros(N, N)

for layer in all_2d_layers:

T = stack([flatten(expert_sd[layer] - base_sd[layer]) for expert_sd])

U, S, Vt = svd(T)

for k in range(rank):

significant = abs(U[:, k]) > threshold

if significant.sum() >= 2:

# 这对 expert 在方向 k 上共享

for i, j in pairs(where(significant)):

affinity[i, j] += 1

# 层次聚类发现子组

groups = hierarchical_clustering(affinity) # e.g., [{0,4}, {1}, {2}, {3}]

# ===== 逐层处理 =====

merged_sd = {}

for layer in all_2d_layers:

tau = [expert_sd[layer] - base_sd[layer] for expert_sd]

# --- 路线 B: ANOVA 分解 ---

mu = mean(tau)

mu_g = {g: mean([tau[i] for i in g]) for g in groups}

residuals = [tau[i] - mu_g[group_of(i)] for i in range(N)]

# --- 路线 A: 微能力分析 + 冲突预解决 ---

micro_caps = [] # [(expert_idx, sigma, u, v), ...]

for i in range(N):

U_i, S_i, Vt_i = svd(residuals[i])

for j in range(top_k):

micro_caps.append((i, S_i[j], U_i[:, j], Vt_i[j, :]))

# 构建干扰矩阵

K = len(micro_caps)

I_matrix = zeros(K, K)

for a in range(K):

for b in range(a+1, K):

v_overlap = dot(micro_caps[a].v, micro_caps[b].v)

u_overlap = dot(micro_caps[a].u, micro_caps[b].u)

I_matrix[a, b] = v_overlap * u_overlap * micro_caps[a].sigma * micro_caps[b].sigma

I_matrix[b, a] = I_matrix[a, b]

# 冲突抑制

for a in range(K):

expert_a = micro_caps[a].expert_idx

conflict_score = sum(max(0, -I_matrix[a, b])

for b in range(K) if micro_caps[b].expert_idx != expert_a)

suppression = max(0, 1 - beta * conflict_score / max_conflict)

micro_caps[a].sigma *= suppression

# 重建清洗后的 residual

cleaned_residuals = reconstruct_from_micro_caps(micro_caps, N)

# --- 路线 C: Tucker 基底搜索 ---

# 构造联合基底

U_pool = hstack([svd(cleaned_residuals[i]).U[:, :k] for i in range(N)])

V_pool = hstack([svd(cleaned_residuals[i]).V[:, :k] for i in range(N)])

B = orth(U_pool) # R^{m × r₂}

C = orth(V_pool) # R^{n × r₃}

# 投影到低维空间

X_targets = [B.T @ cleaned_residuals[i] @ C for i in range(N)]

# 在低维空间优化

X = Parameter(mean(X_targets))

optimizer = Adam([X], lr=1e-4)

for step in range(300):

loss = sum(norm(B @ (X - X_targets[i]) @ C.T @ L_i.T)**2

/ norm(tau[i])**2 for i in range(N))

loss.backward()

optimizer.step()

# 重建

m_residual = B @ X.data @ C.T

# --- 合并 ---

m_shared = alpha_0 * mu + sum(alpha_g * (mu_g[g] - mu) for g in groups)

merged_sd[layer] = base_sd[layer] + lambda_ * (m_shared + m_residual)

return merged_sd

```

#### 11.4.2 各路线可以独立使用

三条路线是模块化的，可以单独使用或任意组合：

| 组合 | 改进来源 | 复杂度 | 预期效果 |
|------|---------|--------|---------|
| 仅 A（微能力预解决 + 原版 wudi2） | 更好的 input | 低 | 温和改进 |
| 仅 B（ANOVA 分解 + 原版 wudi2） | 更好的 target | 低 | 温和改进 |
| 仅 C（Tucker 子空间 + wudi2 loss） | 更好的搜索空间 | 中 | 显著加速 |
| A + B（微能力 + ANOVA + wudi2） | 更好的 input + target | 中 | 较大改进 |
| A + B + C（完整管线） | 全方位改进 | 中 | 最大改进 |

**建议的实验优先级**：

1. **先做 A（最简单）**：只需在 wudi2 之前加一步微能力干扰分析 + 冲突预解决。不改 wudi2 代码，只改输入。

2. **再做 B（中等难度）**：替换 wudi2 的 avg 分解为 ANOVA 分解。需要改 wudi2 的 target 构造。

3. **最后做 C（最复杂但潜力最大）**：限制搜索空间到 Tucker 子空间。需要重写优化过程。

#### 11.4.3 与现有工作的差异化

| 现有方法 | 我们的改进 | 差异化 |
|---------|-----------|-------|
| TSV（白化去相关） | 微能力级别的精准冲突解决 | 不是一刀切去相关，而是区分冗余/冲突/独立 |
| SVC（奇异值缩放） | ANOVA 层级分解 | 不只是缩放，而是结构化分解 + 差异化处理 |
| wudi2（全空间优化） | Tucker 子空间搜索 + 微能力感知 loss | 结构化搜索空间 + 显式跨 expert 关系建模 |
| ETVD（5 方向规则） | 微能力分解（K 方向优化） | 从 5 个方向扩展到 K 个，从规则到优化 |

**定位**：我们是在 wudi2 的优化框架上，注入了 ETVD 分析发现的结构性 insight。不是替代 wudi2，而是增强它。

---

### 11.5 理论分析：为什么这些改进应该有效

#### 11.5.1 ANOVA 分解减小了 target 噪声

**命题**：ANOVA 的个体残差 $\tau_i - \mu_{g(i)}$ 的 norm 严格小于 wudi2 的个体偏差 $\tau_i - \bar{\tau}$。

**证明**：

$$\|\tau_i - \mu_{g(i)}\|_F^2 = \|\tau_i - \bar{\tau} - (\mu_{g(i)} - \bar{\tau})\|_F^2$$

$$= \|\tau_i - \bar{\tau}\|_F^2 - 2\langle \tau_i - \bar{\tau}, \mu_{g(i)} - \bar{\tau}\rangle_F + \|\mu_{g(i)} - \bar{\tau}\|_F^2$$

由于 $\mu_{g(i)} - \bar{\tau}$ 是 $\tau_i - \bar{\tau}$ 在组均值方向上的投影分量（对组内 expert 取平均后的方向），所以：

$$\langle \tau_i - \bar{\tau}, \mu_{g(i)} - \bar{\tau}\rangle_F \geq \|\mu_{g(i)} - \bar{\tau}\|_F^2$$

（因为 $\mu_{g(i)} - \bar{\tau}$ 就是 $\{\tau_j - \bar{\tau}\}_{j \in \mathcal{G}_{g(i)}}$ 的均值，$\tau_i - \bar{\tau}$ 是其中一个，和均值的内积不小于均值的自内积。）

因此：

$$\|\tau_i - \mu_{g(i)}\|_F^2 \leq \|\tau_i - \bar{\tau}\|_F^2 - \|\mu_{g(i)} - \bar{\tau}\|_F^2 \leq \|\tau_i - \bar{\tau}\|_F^2$$

等号当且仅当 $\mu_{g(i)} = \bar{\tau}$（子组均值等于全局均值，即分组没有效果）。

**意义**：ANOVA 残差更小 → 低秩近似更精确 → wudi2 的 target $\tilde{\tau}_i$ 更准确 → 优化更容易收敛到好的解。

#### 11.5.2 微能力预解决减少了 loss 中的矛盾

**命题**：冲突预解决后，wudi2 loss 的 Hessian 矩阵的条件数降低。

**直觉**（非严格证明）：

wudi2 的 loss 是 $N$ 个二次函数之和：

$$\mathcal{L}(m) = \sum_i \|m - \tilde{\tau}_i\|_{A_i}^2$$

其中 $\|x\|_{A_i}^2 = x^T A_i x$，$A_i = L_i^T L_i / \|\tau_i\|_F^2$。

Hessian 为 $H = 2\sum_i A_i$（和 $m$ 无关，因为 loss 是二次的）。

如果 $\tilde{\tau}_1$ 和 $\tilde{\tau}_2$ 在某个方向 $v$ 上严重冲突（$v^T \tilde{\tau}_1 \gg 0$, $v^T \tilde{\tau}_2 \ll 0$），那么 $v$ 方向上的 loss 曲面有两个相互竞争的极小点——优化器在这个方向上会振荡或收敛到一个"谁都不满意"的折中点。

冲突预解决把冲突微能力从 target 中移除或抑制，消除了这种矛盾。结果：

- Loss landscape 更光滑

- 极小值更接近所有 expert 的共同最优

- 优化器收敛更快、结果更好

#### 11.5.3 Tucker 子空间搜索的最优性

**命题**：如果 merged vector $m^*$ 的最优解在 task vectors 张成的子空间中，则 Tucker 子空间搜索不损失最优性。

**证明**：

将 $\mathbb{R}^{m \times n}$ 分解为 task vectors 张成的子空间 $\mathcal{S}$ 和正交补 $\mathcal{S}^\perp$：

$$m = m_\mathcal{S} + m_{\mathcal{S}^\perp}, \quad m_\mathcal{S} \in \mathcal{S}, \quad m_{\mathcal{S}^\perp} \in \mathcal{S}^\perp$$

wudi2 的 loss 中，$\tilde{\tau}_i \in \mathcal{S}$（因为 $\tilde{\tau}_i$ 是由 task vectors 构造的），$L_i$ 的行空间也在 $\mathcal{S}$ 中（因为 $L_i$ 来自 $\tau_i - \bar{\tau}$ 的 SVD）。

$$\|(m - \tilde{\tau}_i) L_i^T\| = \|(m_\mathcal{S} + m_{\mathcal{S}^\perp} - \tilde{\tau}_i) L_i^T\|$$

由于 $L_i$ 的行空间在 $\mathcal{S}$ 中，$m_{\mathcal{S}^\perp} \cdot L_i^T = 0$（$m_{\mathcal{S}^\perp}$ 的行和 $L_i$ 的列正交）。

对于矩阵乘法 $(m - \tilde{\tau}_i) L_i^T$，严格来说需要更仔细的分析。但核心思想是：$L_i^T$ 的作用是投影到 expert $i$ 的主子空间，而这个子空间完全在 $\mathcal{S}$ 内。所以 $m_{\mathcal{S}^\perp}$ 对 loss 没有贡献——在 $\mathcal{S}^\perp$ 中搜索是浪费的。

**因此，将搜索限制在 $\mathcal{S}$ 中（即 Tucker 子空间）不损失最优性，但减小了搜索空间维度，加速了收敛并减少了过拟合风险。**

---

### 11.6 实验设计建议

#### 11.6.1 消融实验设计

为了验证各组件的贡献，建议以下消融实验：

| 实验 | B (ANOVA) | A (微能力) | C (Tucker) | 对比 |
|------|:---------:|:----------:|:----------:|------|
| Baseline: wudi2 | -- | -- | -- | 复现 |
| + ANOVA only | 是 | -- | -- | B 的单独效果 |
| + MC only | -- | 是 | -- | A 的单独效果 |
| + Tucker only | -- | -- | 是 | C 的单独效果 |
| + ANOVA + MC | 是 | 是 | -- | A+B 的组合效果 |
| Full pipeline | 是 | 是 | 是 | 完整方法 |

#### 11.6.2 关键超参数

| 超参 | 含义 | 建议范围 |
|------|------|---------|
| $k$ (per-expert 保留秩) | 微能力数量 | 10-100（按 95% 能量确定） |
| $\beta$ (冲突抑制强度) | 预解决的激进程度 | 0.3-0.8 |
| $\epsilon$ (occupancy 阈值) | 分组灵敏度 | 0.2-0.4 |
| $\alpha_0$ (全局共享缩放) | 共享成分保留程度 | 0.8-1.0 |
| $\lambda$ (最终缩放) | 和 wudi2 相同 | 0.1 |

#### 11.6.3 计算成本估算

以 $m = n = 2048$, $N = 5$, $k = 40$ 为例：

| 操作 | 计算量 | 时间估计 |
|------|--------|---------|
| 5 个 individual SVD ($2048 \times 2048$) | $O(5 \times 2048^3)$ | ~1s/层 |
| 干扰矩阵 ($200 \times 200$) | $O(200^2 \times 2048)$ | ~0.1s/层 |
| 联合基底构造 | $O((5 \times 40)^2 \times 2048)$ | ~0.1s/层 |
| Tucker 优化 (300 步, $200 \times 200$) | $O(300 \times 200^2)$ | ~0.01s/层 |
| **总计 170 层** | | **~3-5 分钟** |

对比 wudi2 原始（全空间优化）：每层 300 步 × $2048^2$ 参数 → 明显更慢。Tucker 子空间搜索不仅效果更好，计算也更快。

---

## 附录 A：关键公式汇总

**Task Arithmetic**:

$$\theta_{\text{TA}} = \theta_0 + \lambda \sum_{i=1}^N \tau_i$$

**ETVD Ensemble SVD**:

$$T = U\Sigma V^T, \quad T_{ij} = \operatorname{vec}(\tau_j)_i$$

**ETVD Reconstruction**:

$$\tau_{\text{ETVD}} = \sum_k w_k \sigma_k v_k, \quad w_k = \begin{cases} \sum_i U_{ik} & \text{unique} \\ \frac{\sum_i U_{ik}}{\operatorname{occ}(k)} & \text{shared} \\ \text{majority vote} & \text{conflict} \end{cases}$$

**wudi2 Loss**:

$$\mathcal{L} = \sum_i \frac{\|(m - \tilde{\tau}_i) L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

**Micro-capability Interference**:

$$I((i,j),(i',j')) = (v_{ij}^T v_{i'j'}) \cdot (u_{ij}^T u_{i'j'}) \cdot \sigma_{ij} \cdot \sigma_{i'j'}$$

**Spectral Over-Accumulation Ratio**:

$$R = \frac{\|\sum_i \tau_i\|_F^2}{\sum_i \|\tau_i\|_F^2} \quad (= 1 \text{ if orthogonal, } > 1 \text{ if over-accumulated})$$

**Micro-capability Interference Matrix** (Section 11.1):

$$I = \mathcal{S} \cdot (\mathcal{V}^T \mathcal{V}) \odot (\mathcal{U}^T \mathcal{U}) \cdot \mathcal{S}$$

**ANOVA Hierarchical Decomposition** (Section 11.2):

$$\tau_i = \mu + (\mu_{g(i)} - \mu) + (\tau_i - \mu_{g(i)})$$

**Tucker Subspace Optimization** (Section 11.3):

$$m = B \cdot X^* \cdot C^T, \quad X^* = \arg\min_X \sum_i \frac{\|B(X - X_i)C^T L_i^T\|_F^2}{\|\tau_i\|_F^2}$$

---

## 附录 B：实验设置速查

- Base model: `OpenGVLab/InternVL2_5-1B`

- 5 experts: `yongxianwei/InternVL2_5-1B_{OCR,VQA,Geometry,Chart,Grounding}`

- 170 2D weight layers, 850 basis directions (5 per layer)

- Evaluation: TextVQA, OCRVQA, VizWiz, GQA, ChartQA, MathVision, MathVista

- VLMEvalKit for evaluation

- Merged models saved to `/data/shichao/data/InternVL_merged/`

- Code: `Skills/theory/{etvd_merge.py, etvd_analysis.py, scaling_sweep.py, merge_variants.py}`

---

## 附录 C：文件索引

| 文件 | 功能 |
|------|------|
| `ensemble_task_vector_decomposition.md` | ETVD 原始讨论记录（包含完整文献综述） |
| `micro_capability_decomposition.md` | 微能力分解的讨论记录 |
| `etvd_analysis.py` | 分析脚本（cosine, Gram, STI, over-accumulation, U-matrix） |
| `etvd_merge.py` | ETVD/TA/TA-norm merge 实现 |
| `merge_variants.py` | 实验变体（weighted TA, no-geometry 等） |
| `scaling_sweep.py` | 缩放系数 sweep |
| `results/analysis_report.md` | 完整分析报告 |
| `results/figures/*.png` | 所有分析图表 |
| `InternVL/internvl_chat/model_merging.py` | wudi/wudi2 原始实现 |
