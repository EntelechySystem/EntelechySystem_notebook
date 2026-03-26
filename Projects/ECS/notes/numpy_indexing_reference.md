# numpy_indexing_reference

本文件总结 NumPy 对二维数组提供的灵活且高性能的索引/切片/聚合操作，并给出示例与性能注意事项，目的是作为 `Relation`（边表）在设计索引 API 时的行为规范参考（支持行/列的多索引、集合运算、视图/拷贝语义等）。

目录
- 基本语义（切片 / 标量 / 视图 vs 拷贝）
- 高级索引（整型数组 / 布尔掩码 / 混合索引）
- 行列交叉选择（np.ix_）
- 批量取/写（np.take / np.put / np.compress）
- 筛选到索引（np.where）与集合运算（交/并/差/取反）
- 排序、Top-K（argsort / argpartition）
- 聚合（按行/按列 sum/mean/max）
- 内存布局与性能要点
- 在 `Relation` 中的映射建议

---

## 1. 基本语义（切片 / 标量 / 视图 vs 拷贝）

- 切片（slice）示例：`arr[a:b, c:d]`、`arr[:, j]`、`arr[i, :]`。
  - 通常返回视图（若内存布局允许），支持原地修改，开销最小。
- 标量索引：`arr[i, j]` 直接访问单元素，代价最小。
- 要点：优先使用连续切片以避免产生拷贝。在 Relation 上实现切片时，若能返回对底层列的视图或轻量索引数组更优。

示例：
```python
sub = arr[2:10, 3:6]  # 视图（若可能）
val = arr[1, 2]       # 标量访问
```

## 2. 高级索引（整型数组 / 布尔掩码 / 混合索引）

- 整型数组索引（Fancy indexing）：`arr[[i1,i2,...], :]`、`arr[[r1,r2],[c1,c2]]`。
  - 结果为拷贝（非视图）。灵活但会产生内存分配。对于大量重复或多次取同一子集，优先用 `np.take` 或缓存索引位置。
- 布尔掩码索引：`arr[row_mask, :]`、`arr[:, col_mask]`、或一维布尔掩码 `arr[mask2d]`（展平）。
  - 也会产生拷贝，方便表达条件筛选。
- 混合索引（slice + fancy）：一旦包含高级索引，相应轴会产生拷贝；注意形状规则和广播行为。

示例：
```python
rows = [0,2,5]
sub = arr[np.array(rows), :]            # 拷贝
mask = (arr[:,0] > 0)
rows2 = np.where(mask)[0]
sub2 = arr[rows2, :]
```

## 3. 行列交叉选择（np.ix_）

- `arr[np.ix_(rows, cols)]` 用于高效构造行/列交叉的子矩阵。
- 相比嵌套循环更简洁且通常比重复的 fancy 索引更可读。

示例：
```python
sub = arr[np.ix_([0,2,5], [1,3,4])]
```

## 4. 批量取/写（np.take / np.put / np.compress）

- `np.take(arr, indices, axis=0)`：C 实现，通常比 fancy 索引更快。
- `np.put(arr, indices, values, axis=?)`：按位置写入，可用于原地批量更新。
- `np.compress(condition, arr, axis=0)`：按布尔条件选择行/列（返回拷贝或视图取决实现）。

建议：在 Relation 实现中暴露 `take`/`set`/`put` 等接口以便对列进行高性能批量读写。

示例：
```python
rows_taken = np.take(arr, [0,3,5], axis=0)
np.put(arr[:,1], [0,2], [9.0, 8.0])  # 直接写入第1列的指定位置
```

## 5. 筛选到索引（np.where）与集合运算（交/并/差/取反）

- `np.where(cond)` 返回满足条件的行/列索引（整型数组），便于后续 `take` 或集合运算。
- 常用集合操作：`np.intersect1d(a,b)`, `np.union1d(a,b)`, `np.setdiff1d(a,b)`, `np.setxor1d(a,b)`。
- 取反可用布尔取反 `~mask` 或集合差集。

示例：
```python
r, c = np.where(arr > thresh)
rows = np.unique(r)
sel = np.intersect1d(rows_a, rows_b)
inv_mask = ~mask
```

## 6. 排序、Top-K（argsort / argpartition）

- `np.argsort(arr, axis=? )`：返回完整排序的索引数组，复杂度 O(n log n)。可选择稳定性（`kind='mergesort'`）。
- `np.argpartition(arr, k)`：返回分区索引，可在平均 O(n) 时间找到 Top-k（未完全排序）。
- 在 Relation 场景中，Top-K 常用于取最大/最小边权的若干条边。

示例：
```python
order = np.argsort(values)[::-1]       # 降序
topk_idxs = np.argpartition(values, -k)[-k:]
```

## 7. 聚合（按行/按列 sum/mean/max）

- 使用 `arr.sum(axis=1)` / `arr.sum(axis=0)` 等 NumPy 聚合函数，底层用 C 加速。
- 对于边表（稀疏表示），等价操作是先通过 `np.where` 或边的 uid->pos 映射找到行/列，然后使用 `np.bincount` 或分组聚合以获得按源/目的的加和。

示例（Relation 映射场景）：
```python
# 使用 uid_to_pos 将 src_uid 映射到行索引 pos，然后 np.bincount(pos, weights=vals)
```

## 8. 内存布局与性能要点

- 保持 C-contiguous 布局，沿最后一维（连续维）做向量化操作更快。
- 尽量使用切片和内置 ufunc/聚合（sum/mean/max），避免 Python 层循环。
- 避免在高频路径产生大量高级索引拷贝；若必须，多次重复读取建议缓存索引或使用 `np.take`。
- 对大型表格数据使用 `np.argsort`/`np.argpartition` 时注意内存峰值，必要时分批处理。

## 9. 在 `Relation` 中的映射建议

为了让 `Relation` 的索引语义接近 NumPy 的二维数组（高性能且灵活），建议实现并暴露以下能力：

1. 行/列通用索引接口
   - 支持：int, slice, list/ndarray[int], 布尔掩码 (length == relation.size)
   - 语义：`relation[row_sel, col_sel]` 返回一个新 `Relation` 或一个轻量视图/索引对象，表示满足条件的边集合。

2. 行列交集 / 子矩阵支持
   - 支持 `relation.select_rows(rows).select_cols(cols)` 或 `relation[np.ix_(rows, cols)]` 风格的 API。

3. 索引集合运算
   - 提供帮助函数：`indices_intersect(a,b)`, `indices_union(a,b)`, `indices_diff(a,b)`, `indices_invert(mask)`。
   - 在内部用 `np.intersect1d/union1d/setdiff1d` 或布尔逻辑来实现高效操作。

4. 批量取/写接口
   - `take(idxs)`：按边索引取子 Relation（目前已经实现 `take`）。
   - `set_attr(name, idxs, values)`：按索引批量写入边属性（目前已实现 `set_attr`）。
   - `filter_mask(mask)` / `remove_by_mask(mask)`：按布尔掩码批量筛选或删除（已实现 `remove_by_mask`）。

5. 行/列按 uid 索引与映射
   - 支持直接按 `src_uid` / `dst_uid` 做筛选：`relation.indices_from_uid(uid)` / `relation.indices_to_uid(uid)`（已实现）。
   - 提供 `build_uid_to_pos(uid_array)` 辅助，便于高效做行/列聚合（已实现）。

6. 聚合与分组
   - `row_sum(attr, uid_to_pos, n_rows)` / `col_sum(attr, uid_to_pos, n_cols)`（已实现），以及 `row_count`/`col_count`。

7. 排序与 Top-K
   - `argsort_by(attr, ascending=True)` / `sort_by(attr, ascending=True)`（已实现）。
   - `topk_by(attr, k)`：可以基于 `argpartition` 高效实现（建议补充）。

8. 视图 vs 拷贝的明确语义
   - 明确文档化哪些操作返回 view（如按列 slice、get_attr 返回视图）哪些返回 copy（如 fancy indexing、布尔筛选、take）。
   - 尽量提供原地写入 API（如 `set_attr` / `put_attr`）以避免创建临时大拷贝。

9. 性能与错误模式
   - 在大规模表上，布尔索引会产生拷贝并增加内存压力；为批量操作推荐先计算整型索引并使用 `take` / `np.bincount` 等原语。
   - 提供 `reserve` 接口避免插入时频繁扩容。

---

### 参考用法示例（伪代码）

```python
# 按行范围与列列表交叉选择
rows = slice(0, 10)
cols = [0, 2, 5]
sub_rel = relation.select_rows(rows).select_cols(cols)

# 布尔掩码 + 行集合交集
mask = relation.get_attr('amount') > 1000
idxs_A = relation.indices_from_uid(bank_uid)
idxs_B = np.where(mask)[0]
sel_idxs = np.intersect1d(idxs_A, idxs_B)
sub = relation.take(sel_idxs)

# Top-K
order = relation.argsort_by('amount', ascending=False)
topk = relation.take(order[:k])
```

---

文档维护：将本文件放在 `Projects/ECS/notes`，后续可把要实现的 API 条目逐项映射到 `ecs/ecsengine.py` 的 `Relation` 方法并写入单元测试（pytest）。
