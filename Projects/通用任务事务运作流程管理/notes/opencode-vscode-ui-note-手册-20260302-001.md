# opencode-vscode-ui-note-手册-20260302-001

本笔记把你截图中 OpenCode 插件在 VS Code 欢迎页（TUI）上显示的主要文本、快捷键与选项逐条解释，并说明如何在工程中找到对应的配置文件以便进一步查看或自定义。

**概览**

- 该界面是 OpenCode 在 VS Code 中的交互式“欢迎/控制台”界面，允许你直接输入自然语言提示与指令，选择交互变体（variants）或代理（agents），并管理会话（sessions）。

**界面元素解释**

- 输入占位提示："Ask anything... 'Fix a TODO in the codebase'" — 输入框提示你在此处键入问题或任务（示例提示：在代码库修 TODO）。通常在输入后按回车发送请求。
- 变体列表（示例：`Build / Big Pickle / OpenCode / Zen`）— 这些是不同的运行/交互预设（variants），通常控制模型/系统提示/工具权限或响应风格。
- 底部快捷键提示（截图可见）：`ctrl+t variants  tab agents  ctrl+p commands` — 直观提醒了常用的内部快捷键：
  - `Ctrl+T`：打开或切换 **variants**（变体/预设）列表。
  - `Tab`：切换或浏览 **agents**（代理/角色集合）。
  - `Ctrl+P`：打开 opencode 的 **commands** 面板（或命令面板）。
- 时间线 / 跳转提示："Tip Press Ctrl+X G or /timeline to jump to specific messages" — 两种跳转消息的方法：
  - 按快捷键组合 `Ctrl+X` 然后 `G`（和弦快捷键），或
  - 在输入框输入命令 `/timeline` 来通过时间线跳转到特定消息。
- 右侧会话面板（Sessions / New Session）— 展示当前会话历史与会话列表，点 `New Session` 可新建独立对话线程以便分离不同任务。

**常见快捷键与交互（截图中可见）**

- `Enter`：发送当前输入（标准行为）。
- `Ctrl+T`：打开 variants 面板，选择不同的预设（如 Build、Zen）。
- `Tab`：在界面控件或 agents 列表间切换。
- `Ctrl+P`：打开 opencode 命令面板（可执行更细粒度命令）。
- `Ctrl+X` 然后 `G`：跳到指定消息（和弦按键）。
- `/timeline`：在输入框内输入以调用时间线跳转功能。

注：插件可能还定义其它快捷键（例如展开/折叠、复制消息、停止请求等），建议在 VS Code 的键盘快捷方式中搜索 `opencode` 以获取完整列表。

**如何在项目中找到对应配置与定义**

- `opencode.json`：通常包含全局配置、默认 variant、端口和其它运行设置。界面上出现的 variant 名称（例如 `Build`、`Zen`）很可能来源于这里或由其引用的配置文件定义。请打开并搜索变体名称以确认。
- `.opencode/agents/`：这个目录通常包含若干 agent 定义（角色/能力说明），对应界面中的 agents 列表。查看该目录下的 `*.md` 或配置文件可以了解每个 agent 的职责与示例提示词。

建议的查看操作（在 VS Code 中运行）：

1. 打开 `opencode.json`：检查 `variants` / `defaults` 字段，找到界面上列出的预设名称。
2. 打开 `.opencode/agents/` 目录（或仓库中 `.opencode` 下的子目录）：查看各 agent 文档，理解其用途与限制。
3. 在键盘快捷方式（`Preferences: Open Keyboard Shortcuts`）中搜索 `opencode`，查看并可自定义快捷键映射。

**常见操作示例**

在工作区根目录（VS Code 终端）你也可以通过本地 OpenCode 服务查询会话或消息，例如（仅示例，需本地服务启动）：

```powershell
# 查询本地 OpenCode 服务的会话（示例）
$base='http://localhost:29000'
$dir=(Resolve-Path .).Path
$q=[uri]::EscapeDataString($dir)
Invoke-RestMethod "$base/question?directory=$q"
```

**后续建议**

- 如果希望将界面上的 variant 名称和值一一映射进笔记，我可以帮你自动提取 `opencode.json` 中的 `variants` 字段并补充到本笔记中（需要我现在去读 `opencode.json` 并更新此文件吗？）。
- 我也可以列出当前已注册的所有快捷键（从 VS Code 设置中筛选 `opencode`），并加入到笔记中作为完整参考。

**参考文档**

- OpenCode 官方中文文档：https://opencode.ai/docs/zh-cn/ — 可在该站内查找“Variants、Agents、Commands、Timeline、Sessions”等章节以获得更详细说明。

----

笔记文件路径：results/opencode-vscode-ui-note.md
