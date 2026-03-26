# 统一索引与查询_API_简并语法糖_设计说明

本文档描述：在本类 ECS（SoA 实体表 + Relation 边表）项目中，如何把“多种二维数组形态（实体表/关系表、稠密/稀疏）”的操作接口**简并**为两类：

- **索引（index）**：只负责“返回位置/掩码”，便于组合，不材料化数据。
- **查询（query）**：在索引的基础上，按用户指定的返回类型材料化（或返回轻量视图）。

目标是：
- 写法尽可能接近 NumPy（简洁、可组合）。
- 后台仍保持高性能：默认自动路由，性能敏感路径可显式指定后端以跳过判断。

---

## 1. 背景：我们到底在操作什么“二维数组”？

在本项目中实际上有两类“二维数组”概念：

1) **第一类：实体表（Entity Table）**
- 语义：一类实体的属性表。
- 存储：SoA（列式）—— `attr_name -> 1D ndarray`。
- “二维”来自“概念上的表格”：行=实体实例（pos），列=属性。

2) **第二类：关系数组（Relation）**
- 语义：两类实体之间某个单一属性（或多属性）的关系。
- 存储有两种形态：
  - 稠密：`mat[src_pos, dst_pos]` 形式的矩阵（`np.ndarray`）。
  - 稀疏：边表（SoA）：`src_uid[] / dst_uid[] / amount[] / ...`。

同一类数据在不同场景下会呈现稠密或稀疏，用户不希望因此学习 8 套 API。

---

## 2. 核心概念：索引 vs 查询（ECS Query 与 NumPy Indexing 的关系）

### 2.1 通俗解释
- **索引（index）**：你提供一个条件或选择器，系统帮你算出“哪些位置命中”。
  - 输出通常是：`indices`（整数位置数组）或 `mask`（布尔掩码）。
  - 这一步尽量轻量，便于用户把多个条件做交/并/差，再继续组合。

- **查询（query）**：你拿到“命中的位置”，再把需要的数据取出来。
  - 输出可能是：`records`（材料化拷贝）、`view`（轻量视图）、或“子 Relation/子矩阵”。

### 2.2 NumPy 的启发
NumPy 非常强调：
- `mask = (A > 0)`（先得到掩码）
- `idx = np.where(mask)[0]`（再得到位置）
- `A[idx]`（最后用位置取值）

我们希望把这种可组合的思想，迁移到 ECS 的查询。

### 2.3 传统 ECS Query 的启发
传统 ECS 的 Query 常见形式：
- “筛选出拥有某些组件的实体集合”
- “再在这些实体上做批量系统计算”

在我们的 SoA ECS 中：
- “拥有某组件”可以等价为“某列 present==True / 或列存在且 active==True”。
- “系统计算”尽量写成 NumPy 向量化。

---

## 3. 统一 API 的产品级约定（对用户只暴露两类操作）

### 3.1 统一入口（建议）
对用户只推荐两类方法：

- `index(...) -> indices|mask`
- `query(...) -> indices|mask|view|records|relation|dense`

其余方法（如 `take`, `where`, `select`）保留为内部后端实现细节或兼容层，不作为主文档推荐入口。

### 3.2 它们操作哪些数据类型？
- **实体表**：`ECSEngine` / `EntityPool`（兼容层对象）。
- **关系表**：`Relation`（边表关系）。

（将来可以扩展：接受 `DenseEntityTable/SparseEntityTable/DenseRelation/SparseRelationEdges`，但对用户主路径尽量隐藏。）

### 3.3 统一方法签名（第一版建议）

#### 实体表：index
```python
idx = engine.index(pred=None, *, include_active_only=False, return_='indices')
```
- pred：
  - None：全选
  - mask：布尔掩码（len==size）
  - callable(pool)->mask
- return_：`'indices'|'mask'`

#### 实体表：query
```python
res = engine.query(pred=None, *, include_active_only=False, return_='indices')
```
- return_：`'indices'|'mask'|'view'|'records'`
- view/copy 语义：
  - indices/mask：轻量
  - view：轻量（只保存 idx + 引用）
  - records：材料化拷贝

#### 关系表：index
```python
edge_idx = rel.index(*, src=None, dst=None, edge_pred=None, return_='indices')
```
- src/dst：支持 None、单个 uid、多个 uid（OR）
- edge_pred：None/mask/callable(rel)->mask
- return_：`'indices'|'mask'`

#### 关系表：query
```python
res = rel.query(*, src=None, dst=None, edge_pred=None, return_='indices')
```
- return_：`'indices'|'mask'|'view'|'relation'`
- view/copy：
  - view：轻量 edge 视图（只存 edge_idx + 引用）
  - relation：子 Relation（拷贝）

### 3.4 自动路由 vs 显式后端提示
为了兼顾易用与性能：

- 默认 `backend='auto'`：
  - 内部根据对象类型决定是实体表/关系表。
  - 根据内部列/存储决定 dense/sparse。

- 性能敏感路径提供显式提示：
  - `engine.query(..., backend='dense')`
  - `rel.query(..., backend='sparse')`

第一版可以先不暴露 backend 参数，只把“可扩展点”写进文档；等我们有实测后再决定是否需要把 backend 真正做成公开参数。

---

## 4. 常用操作清单（我们产品应该支持什么？）

### 4.1 实体表（Entity Table）
- 过滤：`pred(pool)->mask`（向量化比较 + 逻辑与/或）
- active 过滤：`include_active_only=True`
- 返回：indices/mask/view/records

### 4.2 关系表（Relation）
- 按端点过滤：src/dst uid 单个或多个（OR）
- 按边属性过滤：edge_pred（mask 或 callable）
- 返回：edge indices/mask/view/subrelation

### 4.3 组合（index 层最重要）
- 交：`np.intersect1d(a, b)`
- 并：`np.union1d(a, b)`
- 差：`np.setdiff1d(a, b)`
- 取反：`~mask`

---

## 5. 迁移与兼容策略

- 旧接口（如 `EntityPool.query`、`Relation.query`）保留，但主文档推荐走 `index/query`。
- demo 统一改写：只展示 `index/query` 两类入口，其他只作为底层实现说明。

---

## 6. 与现有代码的映射（便于开发）

- 实体表：当前已有 `EntityPool.query(return_='indices|mask|records|view')`，再补一个 `EntityPool.index`（或在 `ECSEngine` 上补 `index/query` 代理）。
- 关系表：当前 `Relation.query(return_='indices|mask|relation')` 已存在；需要补：
  - `Relation.index(...)`：只返回 indices/mask（语义上就是 query 的子集）
  - `RelationView`：与 `EntityPoolView` 对称（轻量边视图）
  - `Relation.query(return_='view')`

第一版重构的“最小闭环”是：
- 新增 `engine.index / engine.query`
- 新增 `relation.index / relation.query(view)`
- demo2 改成只用 `index/query`


