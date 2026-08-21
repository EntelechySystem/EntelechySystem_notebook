# 世界模型之理论：Entity 本体的形式化与工程表达

> **作者**：Ethan Lin
> **日期**：2026-07-22
> **关联项目**：scenekit / RECS / EntelechySystem / ComplexIntelligenceSystem
> **设计文档**：[[设计：场景工具包]]

---

## 摘要

本文提出一种统一的实体本体论（Entity Ontology），作为场景工具包（scenekit, WMK）的理论地基。核心命题为：**"世界"不是空间、时间、实体的三元组，而是以 Entity 为唯一本原、以 parent 嵌套为空间结构、以 Relation 为交互载体、以 Tag 为语义标签的统一体系**。不存在独立的"空间"类——空间只是两个 Entity 之间 `parent` 关系的人类可读标签。本文从范畴论、黎曼几何、图论三个数学视角严格论证这一命题，并将结论映射到 WMK 的工程设计中。

---

## 一、引言：为什么要从本体论开始

### 1.1 现有 ABM 框架的本体论缺陷

NetLogo、Agents.jl、Mesa 等 ABM 框架共享一个隐含的本体论假设：

$$
\text{World} = \text{Space} \times \text{Time} \times \text{Agent}
$$

其中 Space 是容纳 Agent 的容器，Time 是推进的刻度，Agent 是容器中的质点。三者在代码层面是不同的类型：`patches` vs `turtles`（NetLogo），`Model.space` vs `Model.agents`（Mesa）。这种分离导致：

1. **空间类型的硬编码**：你无法把地球表面（S² 球面）当作 Agent 的"空间"，因为框架只提供了 $\mathbb{R}^2$ 网格
2. **交互的维度受限**：质点之间的交互（Point-Point）是唯一的交互类型，无法表达"道路段与河道的交叉"（Curve-Curve）
3. **嵌套的不可表达**：老鼠站在大象背上（大象是老鼠的"空间"）在传统 ABM 中需要手动维护相对坐标

### 1.2 本文的命题

> **一切皆 Entity。Entity 的嵌套树就是"空间"。没有 Space 类——只有 Entity 之间的 `parent` 关系。**

这个命题的推论十分激进：

- "地球表面"和"一辆车"在代码的意义上**是同一类东西**——都是一个 Entity
- 车的 `parent` 是"交通网络"，交通网络的 `parent` 是"地球表面"，地球表面的 `parent` 是"三维欧几里得空间"
- 这个链条上的每一个东西，都只是一个 Entity，只不过它们的 `geometry` 字段不同

---

## 二、形式化地基：五种数学语言

### 2.1 范畴论视角：Entity 作为对象，Relation 作为态射

如果世界由 Entity 和 Relation 构成，范畴论提供了最精确的语言。

令 $\mathcal{E}$ 为所有 Entity 构成的类。对任意两个 Entity $A, B \in \mathcal{E}$，它们之间的**关系**构成态射集 $\text{Hom}(A, B)$。特别地，`parent` 关系是一个特殊的态射：

$$
p: A \to B \quad \text{（A 是 child，B 是 parent）}
$$

**坐标求解链对应态射的复合**。如果车辆 C 的 parent 是道路网络 R，道路网络 R 的 parent 是地球表面 S，则：

$$
\text{world\_pos}(C) = (p_{S \to \mathbb{R}^3} \circ p_{R \to S} \circ p_{C \to R})(t_C)
$$

其中 $p_{C \to R}$ 将车辆在道路上的参数坐标 $t$ 映射为道路的局部坐标，$p_{R \to S}$ 将道路坐标映射为球面 UV 坐标，$p_{S \to \mathbb{R}^3}$ 将球面坐标映射为三维世界坐标。

**这恰恰是范畴论中的态射复合**。WMK 的 `resolve_world_position()` 函数实现的就是这个复合。

更一般地，如果我们将 `geometry` 视为一个**函子** $G: \mathcal{E} \to \mathbf{Man}$（从 Entity 范畴到流形范畴），则 parent 关系定义了拉回（pullback）：

$$
G(A) \xrightarrow{p} G(B) \quad \Longrightarrow \quad G(A) = p^* G(B)
$$

A 的几何（例如"质点"）是通过 parent 的几何（例如"路径"）的拉回来定义的。A 的所有运动自由度都由 $G(B)$ 的切空间约束。

### 2.2 黎曼几何视角：空间 = 带度规的子流形

当 Entity $B$ 的 `geometry = "surface"` 时，$B$ 实际上是一个**黎曼流形** $(\mathcal{M}_B, g_B)$，其中 $g_B$ 是度规张量。$B$ 通过嵌入映射 $\varphi: \mathcal{M}_B \to \mathcal{M}_{\text{parent}}$ 嵌入到其 parent 的流形中。$B$ 上的诱导度规为拉回度规：

$$
g_B = \varphi^* g_{\text{parent}}
$$

这意味着：$B$ 上的每一个 Entity，其"距离"概念是由 $B$ 的几何决定的。如果 $B$ 是 $S^2$（球面），那么 $B$ 上两点之间的"距离"是大圆航线长度，而不是三维欧几里得直线距离。

**例 1：黎曼子流形链**。考虑如下嵌套：

```
三维欧几里得空间 (ℝ³, δ_ij)          ← 平坦度规
  │
  └── 地球 S²(r=6371)               ← 紧致黎曼子流形
       │  φ: S² → ℝ³（球面坐标嵌入）
       │  g_S² = φ*δ（拉回度规 = 大圆测地线）
       │
       └── 岛屿 D ⊂ S²               ← 带边紧致子流形（闭圆盘）
            │  ∂D = S¹（硬边界）
            │  g_D = g_S²|_D（限制）
            │
            └── 大象质点 p(t)         ← 0 维质点
                 │  ṗ ∈ T_{p(t)}D（切空间运动）
                 │  边界反弹（测地线在 ∂D 处切向反射）
                 │
                 └── 鼠质点 q_i(t)    ← 0 维质点
                      q_i 在大象的测地球 B_g(p, r) 内
```

**关键洞察**：

1. **这不是拓扑空间，这是黎曼流形**。$S^2$ 和 $D$ 都有度规 $g$，大象和鼠之间的"距离" $d_g(p, q) = \inf_\gamma \int \sqrt{g_{ij} \dot{\gamma}^i \dot{\gamma}^j} \, dt$ 是测地线长度，不是逻辑规则中的 `if distance < threshold`。
2. **约束是几何的，不是逻辑的**。大象不能走出 $D$，因为 $D$ 的边界 $\partial D \cong S^1$ 是一个一维子流形，大象到达边界时速度被切向反弹（`bounce`）。这是**带边流形上的测地线运动**——一个微分几何问题，不是 if-else。
3. **递归嵌入 = 拉回度规链 + 自由度逐级降维**。鼠的自由度（在大象的测地球内移动）由大象的位置决定；大象的自由度（在岛屿的闭圆盘内移动）由岛屿的几何决定；岛屿的自由度（在球面上）由球面的曲率决定。每一步嵌入都伴随着度规的拉回和自由度的降维。

#### 2.2.4 连续流形的拓扑结构：同伦与不变性

黎曼度规 $g$ 决定了流形上的"刚性的"距离和曲率。但在度规之下，有一层更基础的结构：**拓扑结构**——它在连续的拉伸、压缩、弯曲下保持不变。

考虑例 1 中的岛屿 $D$。它是一个闭圆盘，边界 $\partial D \cong S^1$。如果我们将 $D$ 变形——拉伸成椭圆、弯曲成马鞍面、甚至捏成星形——只要不撕裂、不粘合，它的**同伦类型**不变：

$$
D \simeq \{\text{点}\}, \quad \partial D \simeq S^1
$$

其**基本群**（fundamental group）为：

$$
\pi_1(D) \cong \{1\} \quad \text{（单连通，可缩）}, \qquad \pi_1(\partial D) \cong \mathbb{Z}
$$

这就是"橡胶圈几何"的数学本质。WMK 中的 Entity 如果 `geometry="surface"` 且 `boundary="closed"`，则它同胚于 $S^2$——亏格为 0（没有洞）。一个亏格为 $g$ 的曲面（如 $g=1$ 的环面 $\mathbb{T}^2$）在拓扑上与球面有本质不同的交互行为：球面上的任何闭合曲线都可缩为一点，环面上则存在不可缩的闭合曲线（绕洞的那一圈）。

**拓扑不变量对 WMK 的工程意义**：当 Entity $A$ 的 geometry 是亏格为 $g$ 的曲面时，$A$ 上的 entity 运动被限制在亏格为 $g$ 的约束空间内。这意味着：

- 可缩空间（$g=0$）：任何闭合路径可连续收缩
- 非可缩空间（$g \geq 1$）：存在"绕洞"的周期轨迹，agent 在此类轨迹上可以无限循环而不重复位置
- 边界类型（无边 $\cong$ 紧致无边流形 / 有边 $\cong$ 带边流形）：决定 agent 是否会被反弹

**度规 $g$ vs 拓扑**：度规告诉你"弯多少"，拓扑告诉你"有没有洞"。两者独立：你可以有一个平坦的环面（$\mathbb{T}^2$ 配平坦度规——周期边界条件），也可以有一个弯曲的球面（$S^2$ 配标准圆度规）。WMK 通过 `geometry` 的边界模式（`"clamp"` / `"toroidal"` / `"bounce"`）控制拓扑，通过 `parent` 的度规（`g_S²`、`g_D` 等）控制几何。**拓扑定义"什么运动可能"**（有没有不可缩的循环），**度规定义"运动有多快/多远"**（沿测地线的距离）。

### 2.3 图论视角：离散空间的拓扑

#### 2.3.1 parent 森林与 Relation 图

如果只考虑空间嵌入，Entity 通过 `parent` 形成的是一个**有根森林**（每个 Entity 最多一个 parent）。多叉树的原因是：一个 parent 可以有多个 children（地球表面上有多个城市、多个道路网络）。

但跨 parent 的连接需要另一种结构。两座城市的道路网络在边界处相连——这不是"道路 A 的 parent 是城市 B"，而是两个同级的 entity 之间的**关系**。这种多对多关系用 **RECS Relation 表**表达，它是一个有向或无向图：

$$
G_{\text{relation}} = (V, E), \quad V = \mathcal{E}, \quad E \subseteq \mathcal{E} \times \mathcal{E} \times \{\text{label}\}
$$

**树负责空间，图负责关系。** 两者在 RECS 里各有各的存储：`parent` 列负责树（坐标求解链），`Relation` 表负责图（任意多对多的语义关系）。这就是为什么 WMK 不需要额外的"空间对象"——因为它同时拥有树和图，足以表达从物理嵌入到语义关联的全部结构。

#### 2.3.2 离散空间的拓扑不变量：从单纯复形到同调

图（或超图）并不是"没有拓扑"。离散空间拥有自己的拓扑结构——通过**单纯复形**（simplicial complex）或**胞腔复形**（cell complex）的语言严格表达。

令 $\mathcal{H} = (V, \mathcal{E})$ 为一个超图，其中 $\mathcal{E}$ 是超边的集合。如果 $\mathcal{E}$ 在包含关系下封闭（即超边的子集也是超边），则 $\mathcal{H}$ 形成一个抽象单纯复形，可定义：

- **0-维同调群** $H_0(\mathcal{H})$：连通分量数。$H_0 \cong \mathbb{Z}^k$ 表示有 $k$ 个互不连通的子图
- **1-维同调群** $H_1(\mathcal{H})$：独立循环（"洞"）的数量。$H_1 \cong \mathbb{Z}^b$ 的秩 $b$ 就是环空间（cycle space）的维数 $\beta_1$，即第一贝蒂数
- **贝蒂数** $\beta_k$：第 $k$ 维贝蒂数 = $\text{rank}(H_k)$

对 WMK 的工程意义：

- **连通性**（$\beta_0$）：如果 $\beta_0 > 1$，entity 无法从一个连通分量走到另一个——这应当是一个 bug（表示你创建了不连通的空间），或是故意的设计（孤立的子世界）
- **环结构**（$\beta_1 > 0$）：离散空间中的"洞"——电网的冗余回路、道路网络的环形线、社会网络中的闭合三角。$\beta_1$ 的存在意味着存在多条路径连接同一对节点，这是冗余和容错性的拓扑基础
- **孔洞**（$\beta_2 > 0$）：三维离散空间中的封闭空腔，如建筑的房间或生物细胞内部空间

**例：电网拓扑验证**。假设你用一个超图表示电网（顶点 = 母线，超边 = 输电线路）。通过计算 $H_0$ 可以验证"所有母线是否电气连通"（$\beta_0 = 1$ 表示全连通）；通过计算 $H_1$ 可以检测"是否存在环形冗余路径"（$\beta_1 > 0$ 表示有冗余回路，$\beta_1 = 0$ 表示纯放射状）。这比单纯的"检查是否有孤岛"深一个层次——它是**拓扑层面的完整性验证**。

**关键区分**：连续空间的拓扑使用同伦（homotopy）和单纯同调（simplicial homology in $\mathbb{R}^n$），离散空间的拓扑使用抽象单纯同调（abstract simplicial homology）。二者在数学上同源（都是代数拓扑），但在 WMK 的语境中对应不同的 geometry 类型：`surface`/`volume` 走连续同伦，`hypergraph` 走离散同调。

### 2.4 混合空间：连续与离散的共存

#### 2.4.1 同一个 Entity 树中不同 geometry 的共存

WMK 的核心设计决策之一是：**不同 geometry 类型的 Entity 可以共存于同一棵树中**。这在形式化上意味着同一个 $\mathcal{W}$ 中同时存在连续结构（黎曼子流形）和离散结构（超图/图），且它们之间通过 `parent` 和 `Relation` 交叉连接。

```
三维欧几里得 volume (连续，ℝ³)
  │
  ├── 地球 surface (连续，S²)
  │     │
  │     └── 道路网络 path (连续 1D 网络)
  │           │
  │           └── 车辆 point (连续质点)
  │
  └── 电网拓扑 hypergraph (离散)
        │
        └── 母线 point (离散顶点)
```

**关键问题**：当一个 point entity 穿越连续空间和离散空间的边界时，它的运动规则从连续力学的 $dx/dt = v$ 切换到离散步进 $v \to v'$（沿超边）。这一过渡在理论上是**非平凡的**——它涉及到从 $\mathbb{R}^n$-值状态到离散状态的映射。

在 WMK 的工程实现中，这一过渡由 `geometry` 类型的分发逻辑处理：

```python
def move(entity_id, delta):
    parent = entities[entity_id].parent
    if parent.geometry == "surface":
        entities[entity_id].u += delta[0]  # 连续 UV 位移
        entities[entity_id].v += delta[1]
    elif parent.geometry == "hypergraph":
        edge = parent.hyperedges[delta[0]]  # 离散步进
        entities[entity_id].vertex = edge.target
```

#### 2.4.2 混合空间的交互：连续几何 ↔ 离散拓扑

混合空间最深刻的理论问题出现在**交互**层面。考虑：

- 一个连续流形上的 point entity（飞机）飞越一个离散超图 entity（行政区域划分）
- 飞机的轨迹是连续曲线 $\gamma(t)$，行政区域是离散超图上的顶点
- "飞机进入城市 B" 等价于 $\gamma(t)$ 穿过了超图顶点 B 的对应区域

这要求 WMK 能够检测**连续几何与离散拓扑之间的交集**。这在数学上不是标准操作——传统的微分几何和代数拓扑各自独立发展，很少有交叉。但 WMK 的工程需求强制了这种交叉：

| 交互类型 | 连续侧 | 离散侧 | 检测方式 |
|---|---|---|---|
| Point-in-Hyperedge | point entity 的连续坐标 | 超图 vertex 的区域映射 | 映射连续坐标到最近的 vertex |
| Curve-crosses-Region | path entity 的连续曲线 | 超图 vertex 的边界 | 曲线采样 + 超边归属判定 |
| Surface-covers-Hypergraph | surface entity 的连续 UV | 超图 vertex 的嵌入坐标 | 最近邻 + 距离阈值 |

#### 2.4.3 橡胶圈在混合空间：拓扑同伦与离散同调的交叉

最有理论深度的混合场景是：如果我们将连续流形和离散超图**对偶**起来——连续流形上的一个环（$\gamma: S^1 \to \mathcal{M}$）离散化后对应超图上的一个闭合路径，那么连续同伦 $\pi_1(\mathcal{M})$ 与离散同调 $H_1(\mathcal{H})$ 之间是否存在函子关系？

如果离散化是一个保持拓扑的映射（如三角剖分），那么存在自然同构：

$$
H_k(\mathcal{M}) \cong H_k(K)
$$

其中 $K$ 是 $\mathcal{M}$ 的三角剖分（一个单纯复形），$H_k$ 表示第 $k$ 阶同调群。这意味着：**连续流形的"洞"和离散超图的"环"在拓扑层面是同一类东西**——只是分别用不同的数学语言描述。WMK 的混合空间使得我们可以在代码层面同时操作这两者，而不需要像传统 ABM 那样硬编码为两套独立的类。

### 2.5 高维与分数维：黎曼流形、分形与信息几何的统一视角

#### 2.5.1 高维流形与投影链

当前的 geometry 六元组隐含一个假设：Entity 的嵌入维度等于可视化维度。但**黎曼嵌入定理**（Nash, 1956）告诉我们：任何 $n$ 维可微黎曼流形都可以**等距嵌入**到 $\mathbb{R}^m$ 中，其中 $m \leq n(3n+11)/2$（光滑情形）或 $m \leq n(n+1)(3n+11)/2$（$C^1$ 情形）。这意味着：任何 Entity 的几何在原则上都可以嵌入一个足够高维的欧几里得空间。

**例：克莱因瓶**。$K^2$ 是一个二维流形——局部像 $\mathbb{R}^2$，但整体不可定向。它可以无自交地嵌入 $\mathbb{R}^4$，但不能无自交地嵌入 $\mathbb{R}^3$。我们日常见到的三维克莱因瓶图像实际上是 $\mathbb{R}^4 \to \mathbb{R}^3$ 的有自交投影，而眼睛看到的二维图像是 $\mathbb{R}^4 \to \mathbb{R}^3 \to \mathbb{R}^2$ 的两层投影。

用 WMK 的语言表达：

```
四维欧几里得空间 (ℝ⁴, δ_ij)          ← 根，4D
  │
  └── 克莱因瓶 K²                      ← 2D 黎曼子流形
       │  φ: K² → ℝ⁴（无自交嵌入）
       │  π₁: ℝ⁴ → ℝ³（可视化投影层 1）
       │  π₂: ℝ³ → ℝ²（可视化投影层 2，屏幕）
       │  g_K = φ*δ（拉回度规，在 K² 内部定义测地线距离）
       │
       └── 动点 p(t) ∈ K²              ← 0D 质点，沿 K² 自身的测地线运动
```

**关键洞察**：Entity 的 `dim`（自身维度）与 parent 的 `dim`（嵌入空间维度）是独立的。一个 2D surface 可以嵌入 4D volume 中。一个 1D path 可以嵌入 3D volume 中（如高速公路的立体交叉——在 2D 地图上看似相交，在 3D 空间中实则错开）。投影链 $\pi_2 \circ \pi_1 \circ \varphi: K^2 \to \mathbb{R}^2$ 是嵌套树中坐标求解链的自然延伸——每一个投影步骤都是一个**满射态射**（epimorphism），只不过方向与 parent-child 的嵌入态射相反。

在 WMK 架构中，这意味着 `geometry` 需要一个 `dim` 字段，且它独立于所嵌入的 parent 的维度：

```python
EntityKind("klein_bottle",
    geometry="surface",
    dim=2,           # 自身是 2D 流形
    parent_dim=4,    # 嵌入在 4D 空间中
)
```

#### 2.5.2 分形几何：非整数维度的空间

经典黎曼流形要求每个点都有一个切空间 $T_p\mathcal{M}$，这要求流形在局部像 $\mathbb{R}^n$。分形在局部**不像**任何 $\mathbb{R}^n$——每一层级放大后看到的仍是无限嵌套的自我重复结构。因此分形不是流形。但这不意味着分形不在 WMK 的版图里。有三条路径将分形整合进 Entity 体系。

**路径 1：分形作为"可测空间"，用 Hausdorff 维度替代拓扑维度**

分形有 Hausdorff 维度 $d_H$（Falconer, 2014），可以是非整数。经典例子：

| 分形 | $d_H$ | 嵌入空间 $\mathbb{R}^n$ |
|---|---|---|
| Koch 曲线 | $\log_3 4 \approx 1.261$ | $\mathbb{R}^2$ |
| Sierpinski 三角形 | $\log_2 3 \approx 1.585$ | $\mathbb{R}^2$ |
| Menger 海绵 | $\log_3 20 \approx 2.727$ | $\mathbb{R}^3$ |
| 英国海岸线 | $\approx 1.25$ | $\mathbb{R}^2$ |
| Mandelbrot 集边界 | $2$（Hausdorff 维度恰为 2！） | $\mathbb{C} \cong \mathbb{R}^2$ |

如果 WMK 接受 `dim` 为实数类型：

```python
EntityKind("koch_boundary",
    geometry="path",       # 直觉是 1D 路径
    dim=1.261,             # 但 Hausdorff 维度是分数
    generator="koch",      # 分形生成器类型
    iterations=8,          # 离散近似层级
)
```

工程含义：分形边界上的 entity 运动不是沿切向量（切线不存在），而是沿**分形生成器的迭代步进**——步长不是均匀的，而是遵循分形的尺度律（scaling law）。分形的 `metric(p, q)` 需要通过迭代逼近计算（如 Mandelbrot 集边界上的测地线距离没有闭式公式），这构成了 SoA 向量化的性能挑战——一个实用绕行方案是预计算分形的第 $k$ 次迭代，将其存储为 `path` entity 的密集控制点，然后用标准 1D 测地线逻辑处理运动。分形变成"生成时昂贵，运行时便宜"。

**路径 2：分形作为动力系统的吸引子——Takens 嵌入定理**

**Takens 嵌入定理**（Takens, 1981）提供了分形重吸收回流形框架的桥梁。定理陈述：对于一个 $d$ 维紧致光滑流形上的动力系统，通过时间延迟坐标将观测时间序列 $\{x(t), x(t-\tau), \dots, x(t-(2d)\tau)\}$ 嵌入到 $\mathbb{R}^{2d+1}$ 中，得到的嵌入在**微分同胚**意义下与原始系统等价。

推论：如果一个分形是某个动力系统的**奇异吸引子**（如 Lorenz 吸引子），则通过 Takens 嵌入，它可以用高维欧几里得空间中的微分同胚结构近似。在 WMK 的框架下：

```
ℝ²ᵈ⁺¹（高维欧几里得 volume entity）
  │
  └── 嵌入吸引子（dim ≈ d，recovers fractal structure）
       │  实际上是一个可微子流形的"影子"
       │
       └── agent（point entity，在嵌入空间中的运动轨迹重新构成分形动力学）
```

这使分形在 WMK 中不必是"特例"——它可以通过高维嵌入回流形的理论管辖。

**路径 3：分形作为 entity 的属性而非空间**

最务实的做法不是把分形当作独立的 `geometry` 类型，而是作为 entity 的**几何修饰符**：

```python
EntityKind("fractal_terrain",
    geometry="surface",
    dim=2.3,                     # Hausdorff 维度
    fractal_generator="diamond_square",  # 地形分形生成算法
    fractal_iterations=8,
)
```

在这种模式下，`fractal_surface` 仍然实现三个几何接口——`metric`（通过离散近似的最近点）、`move`（沿离散三角面片的边）、`neighborhood`（半径搜索在离散面上）——只是这些查询的精度随分形迭代次数指数提高，而 smooth UV 参数化不再可用。

#### 2.5.3 多个交叉前沿

分形几何并非孤立发展。以下是正在活跃的交叉领域：

| 领域 | 核心问题 | 关键文献 | 与 WMK 的关系 |
|---|---|---|---|
| **分形上的分析** | 能否在分形上定义拉普拉斯算子和扩散方程？ | Kigami (2001), *Analysis on Fractals*, Cambridge University Press | `move()` 在分形 entity 上的语义——不是测地线运动，而是分形扩散 |
| **谱几何与分形弦** | 分形边界的频谱与其分形维度的关系 | Lapidus & van Frankenhuijsen (2006), *Fractal Geometry, Complex Dimensions and Zeta Functions*, Springer | 分形 entity 上的"距离"可通过谱度量定义，而不是通过嵌入空间的度规 |
| **分形微分方程** | 如何严格定义分形上的微分算子 | Strichartz (2006), *Differential Equations on Fractals: A Tutorial*, Princeton University Press | 分形 entity 上的 `step_for` 行为——动力系统在分形上需要重定义 |
| **信息几何** | 统计流形的度规是 Fisher 信息矩阵；在参数空间上可能具有非整数有效的维度 | Amari (2016), *Information Geometry and Its Applications*, Springer | WMK 中的 `geometry="statistical_manifold"`——一个统计模型的参数空间作为一个 entity |
| **拓扑数据分析** | 从高维点云中提取持续同调（persistent homology），检测多个尺度的拓扑特征 | Edelsbrunner & Harer (2010), *Computational Topology*, AMS | WMK 中的 `hypergraph` 可通过持续同调检测跨尺度不变量 |

#### 2.5.4 统一的几何接口

这些看似迥异的数学对象——黎曼流形、分形、统计流形、离散超图——在 WMK 的框架下被统一为 **`geometry` 的六种实现**。它们共享同一套接口：

```python
class Geometry(ABC):
    dim: float        # 可以为整数的拓扑维度或分数的 Hausdorff 维度
    def metric(p, q)          # 两点间的距离（测地线 / 分形迭代 / 离散步进 / Fisher）
    def move(p, v, dt)        # 沿方向移动（切向量 / 分形生成器步进 / 超边步进）
    def neighborhood(p, r)    # 邻域（测地球 / 离散邻接 / 统计置信域）
    def contains(point)       # 点在不在这个几何里
```

EME 不关心 `geometry` 的具体实现——黎曼流形用测地线测度规，分形用离散近似测度规，统计流形用 Fisher 矩阵测度规。它们对 WMK 而言都是 **同样的抽象，不同的底层算法**。

**黎曼嵌入定理给予了一个统一的上界**：任何 $n$ 维几何（不论它是光滑流形还是分形的 Takens 嵌入）都可以浸入 $\mathbb{R}^{n(n+1)/2 + n}$ 维的欧几里得空间。WMK 的 root entity 只需要足够高维，就可以容纳任何子 entity 的几何——不管它是球面、环面、克莱因瓶，还是 Koch 雪花的三维分形近似。

**信息几何为 Tag 系统提供了更深的理论基础**。如果将 Entity 的 `tags` 视为分布的参数——例如 `"energy"` tag 不是标量，而是 energy 分布的统计参数 $(\mu, \sigma)$，那么 Entity 的几何就是这些参数构成的统计流形。Entity 的 `move()` 不是物理位移，而是在 Fisher-Rao 度规下的自然梯度。这是将 WMK 从物理仿真推广到**认知建模**的理论路径——一个信念 entity 在信息流形上的运动就是一个认知更新过程。

---

## 三、Entity 的本体论：Entity × Relation × Tag

### 3.1 为什么只有三个东西

经过对现有 ABM 生态的系统审视和对 CIS/AWS/ES 理论体系的反思，我们得出结论：世界模型的**最小完备本体**只需要三个原语：

$$
\mathcal{W} = (\mathcal{E}, \mathcal{R}, \mathcal{T})
$$

其中：

- $\mathcal{E}$：Entity 的集合。每个 Entity 是 RECS EntityPool 中的一个 SoA 行，有 `geometry`（几何形态）、`parent`（空间宿主）、`position`（在宿主上的参数坐标）、`tags`（领域属性字典）。
- $\mathcal{R}$：Relation 的集合。包括特殊的 `parent` 关系（$\mathcal{R}_{\text{parent}} \subset \mathcal{E} \times \mathcal{E}$，单值）和一般的 `Relation` 表（$\mathcal{R}_{\text{general}} \subset \mathcal{E} \times \mathcal{E} \times \text{Label}$，多对多）。
- $\mathcal{T}$：Tag 的集合。Tag 是附加在 Entity 或 Relation 上的语义标签，形如 `(key, dtype)`。`"energy": float32` 是一个 Tag，`"team": int32` 也是一个 Tag。

**"空间"不在这个本体里。** 它只是 $\mathcal{R}_{\text{parent}}$ 关系的**人类可读标签**。当我们说"地球表面是车辆的宿主空间"时，我们的意思是：车辆 Entity 的 `parent` 字段指向地球表面 Entity。不多不少。

### 3.2 Entity 的几何形态：geometry 六元组

| geometry | 数学对象 | 内部参数 | children 的自由度 |
|---|---|---|---|
| `point` | 0 维质点 | — | N/A（通常无 children） |
| `segment` | 1 维线段 | 两个端点 | 1 DOF（沿 t ∈ [0,1]） |
| `path` | 1 维折线网络 | 控制点 + 连接 | 1 DOF（沿边步进，可路由） |
| `surface` | 2 维黎曼子流形 | UV 参数网格 | 2 DOF（沿 UV） |
| `volume` | 3 维可微流形 | UVW 参数 | 3 DOF |
| `hypergraph` | 离散超图 | 顶点集 + 超边集 | 离散步进 |

**geometry 决定了 "我是谁" 和 "我为 children 提供什么"**。一个 Entity 是 `volume` 还是 `surface`，不是它自身的分类标签，而是它为 children 提供的自由度。geometry 是 Entity 的**函子**：它把 Entity 映射到流形范畴中的对象，然后 children 的运动被约束在这个流形的切丛上。

### 3.3 Tag 系统：领域语义的落脚点

传统 ABM 框架预设了大量领域属性：NetLogo 的 `turtles-own`，Mesa 的 `Agent` 基类属性，甚至 Agents.jl 的 `@agent` 宏。WMK 拒绝这种做法——因为一旦预设 `energy`，就意味着"所有世界都有能量概念"，这显然是领域偏见。

取而代之的是 **Tag 系统**：

```python
EntityKind("prey",
    geometry="point",
    tags={
        "energy": np.float32,       # 饿了减、吃了增
        "fear_level": np.float32,   # 捕食者距离 → 恐惧上升
        "diet": "U16",               # "herbivore" / "carnivore"
    }
)
```

Tag 的本质是 **Entity 上的自由语义标注**。从数学角度看，一个 Tag $t = (k, d)$ 是一个从 Entity 集合到数据类型 $d$ 的**部分函数**：

$$
t: \mathcal{E} \dashrightarrow \text{values}(d)
$$

并非所有 Entity 都有某个 Tag。`tagged("energy")` 返回所有**定义了该 Tag** 的 Entity。这允许跨 kind 的空间查询：

```python
# 捕食者找所有 "可食用的" 实体（不管什么 kind）
prey_candidates = model.within_radius(
    predator_ids,
    target_kinds=model.kinds_with_tag("edible"),
    radius=perception_radius
)
```

---

## 四、动态 Entity：世界不是静态背景

### 4.1 传统 ABM 的 "活的Agent / 死的空间" 二分

NetLogo、Mesa 等框架隐含地将世界分为两类：`turtles`（活的、有行为的 Agent）和 `patches`（死的、静态的空间）。这种二分是方便的教学抽象，但在真实世界里是错误的——地球板块在漂移、交通网络在扩建、河流在改道。

### 4.2 "活的 entity" 架构含义

在 WMK 中，`type` 字段（`AGENT` / `ENTITY` / `OBJECT` / `ENV`）**不是"能不能动"的规定**。任何 entity——不管它的 geometry 是 point 还是 surface 还是 hypergraph——**都可以注册自己的 `@step_for`**。

```
三维欧几里得空间 (root, geometry=volume)
  │
  ├── 地球 S² (geometry=surface)
  │     │  @step_for: 板块漂移 ← "曲面"有行为
  │     │
  │     ├── 道路网络 (geometry=path)
  │     │     │  @step_for: 路网扩建 ← "折线网络"有行为
  │     │     │
  │     │     └── 车辆 (geometry=point)
  │     │           @step_for: IDM 跟驰 ← "质点"有行为
```

**`type` 只是调度优先级和渲染分组的语义提示**，不是禁锢。如果一个 entity 在某个仿真中没有演化行为，不注册 `step_for` 即可——它作为背景存在，仍为 children 提供几何约束。

### 4.3 Entity 的拆分准则

一个重要设计问题是：什么情况下该把一个 entity 拆成多个 entity？

> **按行为拆分，而非按空间拆分。** 如果把地球按地理区域（华北平原/黄土高原/四川盆地）拆分，但不给每个区域独立的 `@step_for`——那么拆分没有意义，徒增复杂度。如果给每个 tectonic_plate 注册了独立的地质运动规则——那么拆分才是有意义的。entity 的边界不是地理边界，而是"谁有独立的演化规则"。

---

## 五、Entity 之间的交互理论

### 5.1 全维度交互矩阵

WMK 的交互不是 Point-Point 距离检测（那是传统 ABM 的唯一窗口）。Entity 的 geometry 决定它如何与其它 entity 交互：

| A \ B | point | segment | path | surface | volume |
|---|---|---|---|---|---|
| **point** | 测地线距离 < r | 点到线段最近点 | 沿线位置 | 点在面内 | 点在体内 |
| **segment** | — | 线段相交 | — | 穿面检测 | — |
| **path** | — | — | 路径交叉 | — | — |
| **surface** | — | — | — | 面面相交（交线） | — |

两维度越低，交互越基础；维度越高，交互越昂贵（计算复杂度从 $O(N)$ 到 $O(N \log N)$ 到 $O(N^2)$）。WMK 提供统一的 `model.intersects(entity_a, entity_b)` 和 `model.intersecting_pairs(kind_a, kind_b)` 入口，底层根据 geometry 类型分发到不同的向量化算法。

### 5.2 代理交互 vs 直接交互

借鉴 Lenia 文献中的分类（Chan, 2020），WMK 区分两种交互模式：

| 模式 | 机制 | 示例 |
|---|---|---|
| **直接交互** | Entity A 直接修改 Entity B 的状态（碰撞、捕食） | predator 吃掉 prey |
| **代理交互** | Entity A 通过修改共享环境间接影响 Entity B | 蚂蚁留下信息素，其他蚂蚁感知 |

在 WMK 的实现中，直接交互通过 `model.within_radius()` 和 `model.attr()` 写入完成；代理交互通过 `model.add_field("pheromone")` 和 Entity 对场的读写完成。两者可以共存——一个 entity 的 `@step_for` 可以同时包含直接交互和代理交互。

### 5.3 从微观到宏观的涌现

一个关键的理论洞察来自 ES 的复杂智能体理论：**微观的 agent-agent 交互在宏观上可能涌现出新的 entity 行为**。WMK 不做涌现检测（那是 ES 的工作），但 WMK 的数据采集系统通过 `model.add_metric("entropy", fn)` 为涌现度量提供了数据接口。ES 可以在 WMK 上构建涌现度量（熵、复杂度、互信息、因果涌现），而 WMK 只负责采集原始数据。

---

## 六、世界坐标求解的形式化

### 6.1 递归求解链

在 WMK 中，每个 Entity 的 `position` 不存储世界坐标——只存储相对于 `parent` 的**参数坐标**。世界坐标通过沿 parent 链的递归求解得到：

```python
def resolve_world_position(entity_id):
    e = entities[entity_id]
    if e.parent_id == -1:
        return param_to_cartesian(e.geometry, e.position)  # 根：参数→直角
    parent_world = resolve_world_position(e.parent_id)
    local_cart = e.parent_geometry.param_to_local(e.geometry, e.position)
    return parent_world + local_cart
```

**复杂度**：$O(N \times D)$，其中 $N$ 是 entity 总数，$D$ 是最大嵌套深度。地球交通仿真中 $D \leq 5$（三维空间 → 球面 → 岛 → 路网 → 车）。

### 6.2 优化：批量刷新

坐标求解不需要在每次 `move()` 时触发。嵌入树很少变化——只在 `spawn`（新 entity 入树）、`kill`（entity 出树）、`reattach`（entity 换 parent）时更新。世界坐标在每 tick 的 `step()` 结束后**批量执行一次**，然后缓存供渲染/数据采集使用。

---

## 七、与 WMK 工程设计的映射

### 7.1 理论 → 代码 对照表

| 理论概念 | 数学对象 | WMK 实现 |
|---|---|---|
| Entity 集合 $\mathcal{E}$ | SoA 行 | RECS EntityPool |
| parent 关系 | 有根森林 | EntityPool 的 `parent_id` 列 |
| 态射复合（坐标求解） | 函子拉回 $p^*$ | `resolve_world_position()` |
| 一般 Relation | 有向/无向图 | RECS Relation 表 |
| Tag | 部分函数 $t: \mathcal{E} \dashrightarrow V$ | `tags: dict[str, dtype]` |
| geometry (流形类型) | 函子 $G: \mathcal{E} \to \mathbf{Man}$ | `EntityKind.geometry` |
| 约束运动 | 切丛 $T\mathcal{M}$ 上的测地线流 | `model.move(ids, delta)` |
| 全维度交互 | 子流形交集检测 | `model.intersects()` |
| 连续拓扑不变性 | 同伦 $\simeq$、基本群 $\pi_1$、亏格 $g$ | `geometry` 的边界模式 (`clamp`/`toroidal`/`bounce`) |
| 离散拓扑不变性 | 贝蒂数 $\beta_k$、同调群 $H_k$ | `hypergraph` 的连通性和环结构检测 |
| 混合空间过渡 | 连续 $\mathbb{R}^n \leftrightarrow$ 离散顶点 | `move()` 按 parent.geometry 分发 |
| 高维嵌入 | Nash 等距嵌入；Takens 延迟嵌入 | `dim` 独立于 `parent_dim`；坐标投影链 |
| 分形几何 | Hausdorff 维度 $d_H$；分形生成器 | `dim: float`；分形迭代预计算 |
| 信息几何 | Fisher 信息矩阵 $g_{ij}$；自然梯度 | `geometry="statistical_manifold"`；Tag→分布参数

### 7.2 为什么这种设计能覆盖全部消费者

| 消费者 | 使用的理论能力 |
|---|---|
| **CSL 交通仿真** | 黎曼子流形链（道路是 S² 上的 1D path） + 质点 Agent（车辆） + 全维度交互（点-path） + 离散拓扑（道路网络的环结构 $\beta_1$ 表示冗余路径） |
| **YGE 游戏引擎** | 多叉树嵌套（地图 → 区域 → 单位） + 树+图（parent 坐标 + Relation 联盟） + 亏格控制（环面地图 vs 球面地图） |
| **ES 涌现实验** | Tag 跨 kind 查询 + 数据采集管道 + 活的 entity（环境也有 step_for） + 混合空间：连续质点 + 离散信念网络 |
| **ME 数学引擎** | geometry=curve/surface 的数学对象嵌入 + path-surface 交线 + 拓扑不变量（亏格、基本群）的可视化 |

---

## 八、结语

本文提出的 Entity 本体论虽然是对 WMK 工程设计的理论支撑，但其意义不止于一个工具包。它试图回答一个更根本的问题：**当我们说"世界模型"时，我们到底在建模什么？**

传统 ABM 的回答是：建模 Space × Time × Agent 三元组。但我们的回答是：建模 Entity 之间的 parent 嵌套、Relation 关联和 Tag 标注。世界不是容器——世界就是一个 Entity。Entity 的 children 是它的"内容"。Root Entity 的 children 是它包含的"世界"。

从范畴论看，这是态射复合和函子拉回。从黎曼几何看，这是递归子流形链和诱导度规。从代数拓扑看，这是同伦不变性和同调不变量——连续流形的"洞"和离散超图的"环"原来是同一类东西。从高维嵌入看，Nash 定理和 Takens 定理将分形重吸收回流形的理论管辖，而信息几何又将 Tag 系统从领域标注提升为 Fisher-Rao 度规下的统计流形。从图论看，这是树和图的分工。从工程上看，这只需要三列：`parent_id`、`geometry`、`tags`。

**少即是多。** 三个原语，构成全部世界。五种数学语言——范畴、度规、拓扑、分形、信息——从不同视角描述同一个本体：Entity 的嵌套、关联与标注。统一的几何接口(`metric`/`move`/`neighborhood`/`contains`)，不同的底层实现(测地线/分形迭代/离散步进/Fisher 梯度)。

---

## 参考文献

- Amari, S. (2016). *Information Geometry and Its Applications*. Springer. Applied Mathematical Sciences, vol. 194.
- Chan, B. W.-C. (2020). Lenia and Expanded Universe. *The 2020 Conference on Artificial Life*, 221–229.
- do Carmo, M. P. (1992). *Riemannian Geometry*. Birkhäuser.
- Edelsbrunner, H., & Harer, J. L. (2010). *Computational Topology: An Introduction*. American Mathematical Society.
- Falconer, K. (2014). *Fractal Geometry: Mathematical Foundations and Applications* (3rd ed.). Wiley.
- Hofstadter, D., & Sander, E. (2013). *Surfaces and Essences: Analogy as the Fuel and Fire of Thinking*. Basic Books.
- Kigami, J. (2001). *Analysis on Fractals*. Cambridge Tracts in Mathematics 143. Cambridge University Press.
- Lapidus, M. L., & van Frankenhuijsen, M. (2006). *Fractal Geometry, Complex Dimensions and Zeta Functions: Geometry and Spectra of Fractal Strings*. Springer.
- Mandelbrot, B. B. (1982). *The Fractal Geometry of Nature*. W. H. Freeman.
- Nash, J. (1956). The Imbedding Problem for Riemannian Manifolds. *Annals of Mathematics*, 63(1), 20–63.
- Signorelli, C. M., Wang, Q., & Coecke, B. (2021). Reasoning about Conscious Experience with Axiomatic and Graphical Mathematics. *Consciousness and Cognition*, 95, 103168.
- Signorelli, C. M., Wang, Q., & Khan, I. (2021). A Compositional Model of Consciousness Based on Consciousness-Only. *Entropy*, 23(3), 308.
- Spivak, D. I. (2014). *Category Theory for the Sciences*. MIT Press.
- Strichartz, R. S. (2006). *Differential Equations on Fractals: A Tutorial*. Princeton University Press.
- Takens, F. (1981). Detecting Strange Attractors in Turbulence. In D. A. Rand & L.-S. Young (Eds.), *Dynamical Systems and Turbulence*, Lecture Notes in Mathematics, vol. 898 (pp. 366–381). Springer.
