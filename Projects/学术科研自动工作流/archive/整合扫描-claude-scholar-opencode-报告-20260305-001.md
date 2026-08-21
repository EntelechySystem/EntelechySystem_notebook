# 整合扫描-claude-scholar-opencode-报告-20260305-001

## 1. 扫描范围与方法

本报告对以下对象进行对照扫描：

1. 来源侧：`claude-scholar`（`opencode` 分支）
- `commands/`
- `skills/`
- `plugins/`
- `opencode.jsonc`（agent 与 mcp 配置）

2. 目标侧：`AcademicResearch-auto-workflow`
- `.opencode/commands/`
- `.opencode/skills/`
- `.opencode/agents/`
- `.opencode/rules/`
- `.opencode/workflows/`
- `.opencode/governance/`

3. 证据来源
- 文件系统扫描（目录与文件存在性）
- 协议文本抽样核查（命令、技能、规则、代理）

## 2. 总体结论

结论：**已完成“结构性整合”，达到可运行的融合骨架；尚未完成“全量等价迁移”。**

说明：
1. 关键学术主链路（构思、综述、笔记、结果分析、rebuttal、录用后产出）已通过桥接命令接入。
2. 高价值学术技能已迁移首批 6 个，并添加本项目路径与留痕约束。
3. P3 协同与优先级治理已落地，`orchestrator` 已升级为统一调度入口。
4. 非核心通用开发能力（如 build-fix、tdd、sc 指令体系、TS plugins）尚未迁移或按本项目形态重构。

## 3. 已整合内容（按层）

### 3.1 Commands 层

`claude-scholar` 研究类命令已整合为桥接命令：

1. `research-init` -> `run-scholar-research-init`
2. `zotero-review` -> `run-scholar-zotero-review`
3. `zotero-notes` -> `run-scholar-zotero-notes`
4. `analyze-results` -> `run-scholar-analyze-results`
5. `rebuttal` -> `run-scholar-rebuttal`
6. `presentation/poster/promote` -> `run-scholar-post-acceptance`

并新增端到端命令：
- `run-fusion-p4-e2e`

评估：
- 完整度：高（研究主链覆盖）
- 本地化程度：高（路径与审计约束已绑定）

### 3.2 Skills 层

已迁移并本地化的 `claude-scholar` 同名高价值技能：

1. `research-ideation`
2. `results-analysis`
3. `ml-paper-writing`
4. `paper-self-review`
5. `review-response`
6. `citation-verification`

每个技能都增加了：
- 本项目路径约束
- 强制每步留痕
- 与现有代理协同说明

评估：
- 完整度：中高（首批核心能力已落地）
- 与现有实证技能兼容性：高

### 3.3 Agent 协同层

已完成 P3 升级：

1. 新增协同矩阵：`.opencode/governance/agent-collaboration-matrix.md`
2. 新增优先级规则：`.opencode/rules/agent-collaboration-priority.md`
3. 升级总调度：`.opencode/agents/orchestrator.md` 引用上述规则并加入冲突裁决门禁

评估：
- 冲突治理能力：中高
- 可审计性：高

### 3.4 Workflows 层

已存在并可联动：

1. `empirical-paper-factory.md`
2. `rag-econometrics-autopipeline.md`
3. 新增 `fusion-p4-e2e-validation.md`

评估：
- 从规则到执行的闭环可形成

### 3.5 Rules/Governance 层

新增或强化：

1. `agent-collaboration-priority.md`
2. `agent-collaboration-matrix.md`
3. 与既有规则联动：`methods-policy.md`、`workspace-structure.md`、`mandatory-step-logging-supervision.md`

评估：
- 治理强度高于来源项目通用配置

## 4. 未整合或部分整合内容

### 4.1 未整合命令（来源有，目标未迁移）

代表项：
- `build-fix`
- `checkpoint`
- `code-review`
- `commit`
- `create_project`
- `learn`
- `plan`
- `refactor-clean`
- `setup-pm`
- `tdd`
- `update-github`
- `update-memory`
- `update-readme`
- `verify`
- `sc/*` 套件

原因：
- 当前阶段聚焦学术主链，不做通用开发链全量搬运。

### 4.2 未整合技能（来源有，目标未迁移）

代表项：
- `writing-anti-ai`
- `post-acceptance`（已通过桥接命令实现功能，但技能目录未同名迁移）
- `daily-paper-generator`
- `doc-coauthoring`
- `planning-with-files`
- `uv-package-manager`
- `verification-loop`
- `kaggle-learner`
- `frontend-design`、`ui-ux-pro-max`、`web-design-reviewer`
- 以及 plugin/command 开发类技能

原因：
- 与当前实证主线相关性次级，或已有本地替代能力。

### 4.3 Plugins 机制差异

`claude-scholar(opencode)` 的 TypeScript plugins（如 `skill-eval.ts`、`session-start.ts`）尚未按同机制迁移到当前项目。当前项目采用规则与命令治理为主。

影响：
- 不影响主流程执行。
- 会影响“会话级自动提示与自动守卫”的体验一致性。

## 5. 优势与缺口评估

### 5.1 当前整合后的优势

1. 实证主线稳定：命令和技能都绑定项目目录职责。
2. 治理更强：优先级链路和双审计已进入规则层。
3. 可追溯性强：每步留痕要求已内建。
4. 可扩展性好：桥接策略允许后续渐进迁移。

### 5.2 当前缺口

1. 通用开发链路未整合（对代码工程向任务支持较弱）。
2. 来源项目的插件化自动化体验尚未对齐。
3. P4 尚待真实课题实跑验证，当前仍以静态验证为主。

## 6. 建议（按优先级）

1. 高优先级：执行一次 `run-fusion-p4-e2e` 真实课题试运行，形成 `v1` 基线。
2. 中优先级：补充 `post-acceptance` 与 `writing-anti-ai` 技能目录迁移，提升投稿后链路与文本质控。
3. 中优先级：对接最小插件能力（至少 session-start 与 skill-eval 的等价策略）。
4. 低优先级：按需引入 `sc/*` 与开发类命令，避免复杂度提前膨胀。

## 7. 结论

当前整合状态可定义为：
- **“学术主流程可运行 + 治理链可审计 + 扩展面待增量迁移”。**

即：已具备可投入实跑的融合框架，但若目标是“与 `claude-scholar(opencode)` 全量等价”，仍需继续推进命令、技能与插件三条补齐路线。