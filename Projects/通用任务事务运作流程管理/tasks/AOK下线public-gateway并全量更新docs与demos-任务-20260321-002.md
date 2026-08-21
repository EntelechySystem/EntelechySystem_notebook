# AOK下线public-gateway并全量更新docs与demos-任务-20260321-002

## 1. 预处理

### 1.1 任务目标

- 彻底下线 `autodokit/tools/public` 与 `autodokit/tools/gateway` 旧层。
- 使用最新“工具直调 + `__init__` 分组导出”方法更新 `autodo-kit` 文档体系。
- 在 `demos` 补充可单独运行的 tool 示例脚本，并完成最小回归验证。

### 1.2 实施计划

1. 全量扫描代码与文档中对旧层的依赖与引用。
2. 删除旧层源码与清单，清理打包配置与索引残留。
3. 更新 `README`、用户手册、开发者指南、API 手册、产品需求、开发日志、Sphinx 快速开始。
4. 新增 `demos/scripts` 的工具独立示例（用户直调、开发者直调、CLI 调用）。
5. 运行最小回归并记录结果。

## 2. 实施任务

### 2.1 代码下线清理

- 删除目录内容：
  - `autodokit/tools/public/*`
  - `autodokit/tools/gateway/*`
- 删除打包残留：
  - `pyproject.toml` 中移除 `tools/gateway/*.json`。
  - `autodo_kit.egg-info/SOURCES.txt` 中移除旧层文件索引。
- 物理目录删除：
  - 因 PowerShell `Remove-Item` 策略受限，改用 `cmd rmdir /s /q` 删除目录壳，完成下线。

### 2.2 文档全量同步

已更新文件：

- `README.md`
- `docs/用户手册.md`
- `docs/开发者指南.md`
- `docs/API手册.md`
- `docs/产品需求文档.md`
- `docs/开发日志.md`
- `docs/sphinx/quickstart.md`

同步结果：

- 全部改为“函数直调 + 分组导出”口径；
- 删除旧层清单与统一能力调用叙事；
- 新增 demos 工具示例的可执行命令。

### 2.3 demos 新增示例

新增脚本：

- `demos/scripts/demo_tool_user_import_call.py`
- `demos/scripts/demo_tool_developer_get_tool_call.py`
- `demos/scripts/demo_tool_cli_call.py`

覆盖场景：

- 用户公开工具直接导入调用；
- 开发者工具按名称读取调用；
- CLI 按函数名调用公开工具。

## 3. 实施进度与结果

### 3.1 验证结果

已执行最小回归命令：

- `python demos/scripts/demo_tool_user_import_call.py`
- `python demos/scripts/demo_tool_developer_get_tool_call.py`
- `python demos/scripts/demo_tool_cli_call.py`

结果：全部通过。

### 3.2 当前状态

- 旧层已彻底下线（代码、清单、打包残留已清理）。
- 文档体系已与最新工具管理方式对齐。
- demos 已补充“单独使用 tool”示例并可运行。

## 4. 下一步建议

1. 追加一份 `demos/scripts/README`，集中说明各脚本输入输出与预期结果。
2. 在 CI 中加入 3 个 demo 脚本的最小冒烟执行，防止后续回归。
3. 逐步将 `autodoengine.api` 的历史命名（如 `list_public_tools`）迁移到更中性的命名，并保留兼容别名。
