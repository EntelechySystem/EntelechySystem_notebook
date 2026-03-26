# ECS查询功能



# ECS Query（查询）——通俗说明

## 一句话概述
ECS 的 Query（查询）是把“基于实体属性的条件”转换为“边/关系的选择”，常见流程是：实体谓词 → 实体位置掩码/索引 → 把实体位置映射到边表的 src/dst → 得到边掩码或索引，最后返回边集合或子 Relation。

## 输入 / 输出
- 输入：
  - 实体层的谓词（例如“资产 > 1000”、“状态为 active”），可以是布尔掩码、整型索引数组或可调用的 predicate(pool)。
  - 目标 Relation（边表）。
- 输出（可选）：
  - 边索引（int 数组）、边布尔掩码（length == relation.size）、或子 Relation（边表切片）。

## 处理流程（通俗步骤）
1. 在实体池上执行谓词（向量化计算），得到长度为实体数的布尔掩码或整型索引。
2. 用实体的 uid 列表构建 uid->pos 映射（`build_uid_to_pos`），pos 指的是池内位置（0..n-1）。
3. 将 Relation 中的 `src_uid` / `dst_uid` 向量化映射为 src_pos / dst_pos（使用 `_map_uids_to_pos`），得到每条边对应的源/目标位置，未映射的 uid 返回 -1。
4. 仅对有效边（src_pos/dst_pos >= 0）在 pos 维度上查询实体掩码：edge_mask = src_mask[src_pos] & dst_mask[dst_pos]（或单侧掩码）。
5. 根据需要返回索引、掩码或 `relation.take(idxs)`。

## 常见优化建议（工程化要点）
- 向量化优先：所有映射和布尔组合用 NumPy 操作实现，避免 Python 循环。
- uid\->pos 用稠密数组（`np.full(max_uid+1, -1)`）或 dict（稀疏 uid 空间）二选一，按场景选择。
- 对布尔字段使用位集合（bitset / `np.packbits` / `bitarray`）做位运算再展开为索引，能显著节省内存并加速并/交运算。
- Relation 为稀疏主场景：把边在内存中以 SoA（edge-list）存储，按需构造 CSR/CSC（用于矩阵式分析）并缓存，更新时延迟失效重建。
- 缓存常用映射（uid\->pos）、常用掩码和构造好的稀疏矩阵，修改 Relation 时标记失效。

## 与已实现 API 的对应（映射说明）
- 实体谓词前端：
  - `EntityPool.match(pred)`：把 predicate 规范化，返回长度为 size 的布尔掩码（或整型索引）。这是 Query 的统一前端。
- 边选择后端：
  - `Relation.query_by_entity_filters(src_pool, dst_pool, src_filter, dst_filter, return_type)`：把实体掩码映射到边表并返回 `indices` / `mask` / `relation`。
- 辅助工具：
  - `Relation.build_uid_to_pos(uid_array)`：构造 uid->pos 映射。
  - `Relation._map_uids_to_pos(uids, uid_to_pos)`：把边表的 uid 向量化映射为 pos 向量。

## 简单示例（伪代码）
```python
# 在实体池上计算谓词（向量化）
bank_mask = banks_pool.match(lambda p: p.get_attr('A_IB_all') > 1000)

# 把实体谓词映射到边（查询边表）
edge_idxs = A_IB_relation.query_by_entity_filters(
    src_pool=banks_pool,
    dst_pool=banks_pool,
    src_filter=bank_mask,
    dst_filter=None,
    return_type='indices'
)

# 获取子 Relation
sub_rel = A_IB_relation.take(edge_idxs)

