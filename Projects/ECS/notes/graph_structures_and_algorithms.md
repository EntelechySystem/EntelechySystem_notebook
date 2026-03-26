# graph_structures_and_algorithms

本文件介绍常见的图数据结构及其算法，包括基本特征、工作原理、极简 Python 伪代码，并对它们的性能与实用性进行对比分析。

---

## 1. 邻接矩阵（Adjacency Matrix）

**基本特征**：
- 用二维数组存储顶点间的连接关系。
- 适合稠密图，空间复杂度 $O(N^2)$。
- 查询边是否存在 $O(1)$，遍历所有边 $O(N^2)$。

**工作原理**：
- 行列分别代表顶点，`mat[i][j]=1` 表示有边。

**伪代码**：
```python
N = 4
mat = [[0]*N for _ in range(N)]
mat[0][1] = 1  # 添加边0->1
if mat[0][1]:
    print("0到1有边")
```

---

## 2. 邻接表（Adjacency List）

**基本特征**：
- 用字典或列表存储每个顶点的邻居。
- 适合稀疏图，空间复杂度 $O(N+E)$。
- 查询某节点的所有邻居 $O(度)$。

**工作原理**：
- 每个节点维护一个邻居列表。

**伪代码**：
```python
adj = {i: [] for i in range(N)}
adj[0].append(1)  # 添加边0->1
for v in adj[0]:
    print("0的邻居:", v)
```

---

## 3. 边表（Edge List）

**基本特征**：
- 直接存储所有边的列表。
- 空间复杂度 $O(E)$。
- 适合边属性丰富、动态增删边的场景。

**工作原理**：
- 每条边为(src, dst, ...属性)。

**伪代码**：
```python
edges = []
edges.append((0, 1))
for src, dst in edges:
    print(f"{src}->{dst}")
```

---

## 4. 反向索引（Inverted Index）

**基本特征**：
- 实体到节点的倒排映射。
- 适合快速查找“某实体关联了哪些节点”。

**工作原理**：
- 字典：key为实体，value为节点列表。

**伪代码**：
```python
inv = {}
inv.setdefault("tagA", []).append(0)
print(inv["tagA"])  # 输出所有有tagA的节点
```

---

## 5. 稀疏矩阵（Sparse Matrix）

**基本特征**：
- 只存储非零元素，节省空间。
- 适合大规模稀疏图。

**工作原理**：
- 用三元组(row, col, value)或专用库如scipy.sparse。

**伪代码**：
```python
from scipy.sparse import coo_matrix
rows, cols, data = [0], [1], [1]
mat = coo_matrix((data, (rows, cols)), shape=(N, N))
```

---

## 6. 位图/位集合（Bitset）

**基本特征**：
- 用位数组表示集合。
- 适合大规模集合的高效并/交/查。

**工作原理**：
- 每个位代表一个元素是否存在。

**伪代码**：
```python
from bitarray import bitarray
bits = bitarray(N)
bits.setall(0)
bits[0] = 1  # 节点0激活
```

---

## 7. ECS风格：EntityPool + Relation

**基本特征**：
- SoA结构，实体池+边表，属性可扩展。
- 支持动态增删、属性扩展、激活/禁用。
- 适合仿真、游戏、复杂系统。

**工作原理**：
- EntityPool存实体属性，Relation存(src_uid, dst_uid, ...)。

**伪代码**：
```python
class EntityPool:
    def add(self, **attrs): ...
class Relation:
    def add(self, src_uid, dst_uid, **attrs): ...
```

---

## 性能对比与实用性分析

| 结构         | 查询边 | 遍历邻居 | 动态增删 | 空间效率 | 适用场景           |
|--------------|--------|----------|----------|----------|--------------------|
| 邻接矩阵     | O(1)   | O(N)     | 差       | 差       | 小型稠密图         |
| 邻接表       | O(度)  | O(度)    | 好       | 优       | 大型稀疏图         |
| 边表         | O(E)   | O(E)     | 优       | 优       | 动态边/属性丰富    |
| 反向索引     | O(1)   | O(1)     | 好       | 优       | 倒排检索           |
| 稀疏矩阵     | O(1)   | O(非零)  | 一般     | 优       | 大型稀疏图分析     |
| 位图         | O(1)   | O(N)     | 优       | 优       | 大规模集合运算     |
| EntityPool+Relation | O(度) | O(度) | 优 | 优 | 动态ECS系统         |

- **静态分析/批量运算**：邻接矩阵、稀疏矩阵更快
- **动态增删/属性扩展**：边表、EntityPool+Relation更优
- **ECS/仿真/游戏**：EntityPool+Relation最适合

---

**结论**
- 静态图分析优先用邻接矩阵/稀疏矩阵
- 动态系统、ECS、仿真优先用EntityPool+Relation
- 边表适合属性丰富、频繁变动的关系
- 反向索引/位图适合集合运算和倒排检索

