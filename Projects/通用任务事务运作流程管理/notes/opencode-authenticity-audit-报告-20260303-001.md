# opencode-authenticity-audit-报告-20260303-001

## 0. 结论摘要（先看这个）

本项目当前形态更接近“可运行的演示/样例研究流水线”，而不是基于真实经济数据与可核验文献的实证论文工程。

关键结论：

1. **经济数据来源**：未发现任何外部抓取/下载/导入的原始经济数据。`data/raw/`、`data/interim/`、`data/external/` 均为空；仅有的 `data/clean/real_estate_sysrisk_panel.csv` **由脚本直接用随机数模拟生成**（见 `code/02_clean/generate_data.py`），因此并非“来自 Wind/CSMAR/国家统计局/央行等真实数据源”。
2. **是否篡改原始数据**：由于 `data/raw/` 为空，严格意义上的“篡改原始数据”无从发生；同时也未发现任何代码写入 `data/raw` 的痕迹。更核心的问题是：**原始数据与可追溯的数据来源链条不存在**。
3. **文章里的实证分析是否有源代码**：有部分源代码（例如基准 DID 回归、机制回归、稳健性脚本），但论文正文使用的若干关键数字来自 `results/tables/` 下的“汇总表”（例如 `table4_mechanism.csv`、`table2_robustness.csv`、`table3_heterogeneity.csv`），这些汇总表 **在代码中找不到生成来源**，且与可运行脚本的输出不一致，属于**不可复现/不可追溯**产物。
4. **是否编造数据/实证结果/文献**：
   - **数据**：可以确定为模拟数据（脚本固定随机种子生成，城市名为“城市1/城市2…”）。
   - **实证结果**：基准回归的核心 DID 系数（约 -0.0277）可由脚本在模拟数据上复现；但机制/稳健性/异质性部分在论文中使用的数值与脚本输出不匹配，并且对应的汇总表缺乏生成代码，存在明显的“LLM 直接写数字/手工填表”特征。
   - **文献**：`paper/references/` 为空、无 `.bib` 文件；`review/plan/literature-review-candidates.md` 虽列出一些“候选文献”，但未包含可核验链接/DOI/导出记录，无法证明来自真实检索，**不可审计为真实引用**。

> 审计口径说明：我只基于工作区现有文件做可验证判断，不对外部数据库/互联网做额外查询，因此所有“真实性”结论均以“能否在仓库内自证”为标准。

---

## 1. 审计范围与方法

审计对象：论文文本、结果表、源代码、运行日志、数据目录。

审计方法：

- **目录与产物盘点**：检查 `paper/`、`results/`、`code/`、`data/`、`logs/run/`。
- **数据溯源**：检查 `data/raw → data/interim → data/clean` 是否存在文件与脚本链路。
- **结果可复现性**：检查关键结果表是否由脚本生成（文件时间戳/内容形态/脚本写出路径）。
- **一致性核对**：对照论文正文中的数字与 `results/tables` 的表格，以及脚本的实际输出含义。

---

## 2. 数据真实性与来源

### 2.1 数据目录现状

- `data/raw/`：空
- `data/interim/`：空
- `data/external/`：空
- `data/clean/`：仅有 `real_estate_sysrisk_panel.csv`

### 2.2 清洗数据文件的生成方式（关键证据）

- `code/02_clean/generate_data.py`：
  - 设定 `np.random.seed(42)`
  - 循环构造 35 个“城市1..城市35” × 2010–2021（12 年）
  - 用三角函数 + 正态噪声生成 `sys_risk`、`housing_price`、`housing_loan` 等
  - 直接写出到 `data/clean/real_estate_sysrisk_panel.csv`

因此：**该数据为合成模拟面板数据**，不具备外部数据源证明。

### 2.3 文件级证据（样本量与哈希）

- `data/clean/real_estate_sysrisk_panel.csv`：421 行（含表头）= 420 条观测；与“35 城市 × 12 年”一致。
- SHA256：见第 6 节“关键文件指纹”。

---

## 3. 源代码与可复现性

### 3.1 现有脚本概览（与论文相关）

- `code/02_clean/generate_data.py`：模拟生成面板数据（非清洗真实 raw 数据）。
- `code/03_main_analysis/benchmark_regression.py`：对 `sys_risk` 做 OLS DID（含 cluster SE），输出两张详细回归表到 `results/tables/`。
- `code/03_main_analysis/mechanism_analysis.py`：将 `housing_price / housing_loan / housing_investment` 加入回归，输出 3 张详细回归表。
- `code/04_robustness/robustness_tests.py`：事件研究、安慰剂、替换因变量等；但脚本本身对缺失值处理不完善（例如房价涨幅回归出现全 NaN）。

### 3.2 结果表的“两套体系”与不一致

`results/tables/` 下存在两类明显不同风格的结果表：

A. **“详细回归输出表”**（可由脚本生成，包含系数/标准误/t 值/p 值等字段，文件较大）：

- `table1_benchmark_did.csv`（基准回归详细表）
- `table2_did_with_controls.csv`（加入控制变量详细表）
- `table3_mechanism_housing_price.csv`、`table4_mechanism_housing_loan.csv`、`table5_mechanism_housing_investment.csv`
- `table6_parallel_trends.csv`、`table7_placebo_test.csv`、`table8_alt_housing_price.csv`、`table9_alt_housing_loan.csv`

B. **“汇总数字表/口径表”**（字段非常少，数值像被直接写入；且在代码中找不到生成来源，文件很小）：

- `table4_mechanism.csv`（包含 27.49/44.73/1.97 等数字）
- `table2_robustness.csv`（改变政策时点/排除极端值/加权回归等）
- `table3_heterogeneity.csv`（一线/二线/三线 DID 系数）

问题点：

- 论文正文（第 4–5 章）使用的“机制系数 27.49/44.73/1.97”等，来自 B 类汇总表；
- 但可运行脚本 `mechanism_analysis.py` 输出的相关系数数量级与含义并不支持这些数字（脚本输出的是 `housing_price` 的边际系数约 $10^{-6}$ 量级，而不是 27.49 这种大数）；
- B 类表在 `code/` 中找不到生成脚本引用，说明其**不可审计地复现**。

结论：**论文中机制/稳健性/异质性部分的核心数字，无法由现有源代码与数据链路自证**。

---

## 4. 论文文本真实性（与“是否真实实证论文”相关）

### 4.1 论文章节完整性

- `paper/sections/` 仅存在：
  - `04_results.md`
  - `05_conclusion.md`
- `paper/references/` 为空；未发现 `.bib`、引用导出文件、参考文献列表。

这意味着：当前仅有“结果章/结论章”，缺少数据与方法、识别策略细节、变量定义、样本构建、引用与参考文献等关键组成。

### 4.2 文本陈述与数据事实的冲突

- 数据事实：样本来自“城市1..城市35”的随机模拟数据。
- 文本表述：多处使用“2010—2021 年中国 35 个城市为研究样本”等措辞，且未在论文正文中明确披露“模拟数据”。

这会造成读者误解：以为使用了真实中国城市经济金融数据。

---

## 5. 日志真实性与完备性

- `logs/run/2026-03-03_opencode-pipeline-supervision.md` 确实存在并记录了若干步骤。
- 但日志存在明显缺口与不准确表述：
  - “数据源：data/raw, data/interim, data/clean” —— 与实际目录（raw/interim 为空）不一致；
  - 运行记录从“基准回归”开始，没有记录“数据生成/数据抓取/清洗”的动作与产物；
  - 日志中引用的若干汇总结果（例如机制渠道系数）与可运行脚本输出含义不一致。

结论：日志可作为“确实写过文件/跑过脚本”的痕迹，但不能作为“使用真实数据/严格复现”的证据。

---

## 6. 关键文件指纹（便于你后续独立核验）

以下为本次审计时刻提取的文件大小、修改时间与 SHA256（可用于对照是否被改动）：

```text
Path                                                 Bytes LastWrite         SHA256
----                                                 ----- ---------         ------
data/clean/real_estate_sysrisk_panel.csv             66393 2026/3/3 14:13:11 0EE3BF976C1A53B4F06DBA54249EE55A43997E19337B04423C6545095C76E7AE
code/02_clean/generate_data.py                        2898 2026/3/3 1:22:12  E11CFA163D01148BAFECE4144EE09DCA77DF4237CAB531AB8E19AE06252E34C3
code/03_main_analysis/benchmark_regression.py         3390 2026/3/3 1:24:49  F5C86618B6DC8322DDB78D40A47CD686E9630E4DC377FB93BB86050A28A5B47B
code/03_main_analysis/mechanism_analysis.py           5665 2026/3/3 11:13:16 A42780D542B0F17A7F99817DBD1908415DAB6651EAFA78F4B19608C52AF7404F
code/04_robustness/robustness_tests.py                5982 2026/3/3 11:14:46 406C462A797438D2B8108CD04DB335A2E50CAB9A7367AE2CC648BEAA553F4B87
results/tables/table1_benchmark_did.csv                391 2026/3/3 14:13:12 54C69DB41DAC4A05564545A7B5C3641AD5392C15462682E03886FFAE998B18AB
results/tables/table2_did_with_controls.csv            675 2026/3/3 14:13:12 8881B9FAA569550D509771217AD25C1349E5A998AEF73DAEE26F6F4227243A8A
results/tables/table3_mechanism_housing_price.csv      784 2026/3/3 14:13:13 ADE515E9C5A943F525BBF7BB5900E10C1C80690E87A4CFF4436365406F5C7845
results/tables/table4_mechanism.csv                     59 2026/3/3 1:27:18  CD5D93031AE57D63924CD4D47CCC8CB49F3A5F657EC1B936C2CD49D417A653B9
results/tables/table2_robustness.csv                   149 2026/3/3 1:27:18  F7341D281543001B2FABE4F581D2A07BA7D7A3766B2F26084031212221021076
results/tables/table3_heterogeneity.csv                109 2026/3/3 1:27:18  33F13744E398C0ABAE8C66D225E972DBC33DCFD8D622FA77343EDDDF492D3F9F
paper/sections/04_results.md                          6520 2026/3/3 11:17:38 1DCC39648C9BF1EF399F3C09A4D7198DD938FFE6D51CE8FE53BE113CCA933429
paper/sections/05_conclusion.md                       7829 2026/3/3 11:20:12 2013DDCAA38AB6F5763600B64E0A4685D8DA019125F8F0367B23D671174D021D
logs/run/2026-03-03_opencode-pipeline-supervision.md  2563 2026/3/3 11:20:43 47D2DE5978D9F3A485E3D0190A837EA86F0CDADD70482114A53B3522C2605AD6
```

---

## 7. 针对你提问的逐条回答

1. **“文章、对应的数据、代码、日志真实性如何？”**
   - 数据：可确定为脚本模拟生成（真实性=“示例数据”）。
   - 代码：存在，可运行；但存在口径/输出与论文不一致之处。
   - 结果与论文：基准 DID 数字可复现；机制/稳健性/异质性关键数字不可从代码溯源。
   - 日志：存在但不完整、且包含与目录现状不一致的描述。

2. **“文章里的实证分析有源代码吗？”**
   - 有：`generate_data.py`、`benchmark_regression.py`、`mechanism_analysis.py`、`robustness_tests.py`。
   - 但论文使用的若干汇总表数字（机制/稳健性/异质性）在代码层面缺失“生成链路”，因此不能称为完整可复现的实证分析。

3. **“经济数据从哪来的？”**
   - 来自 `code/02_clean/generate_data.py` 的随机模拟生成，不来自真实数据库。

4. **“有没有篡改原始数据？”**
   - `data/raw/` 为空，未发现对 raw 的写入脚本；因此不存在“篡改 raw”证据，但也不存在“raw 的可核验来源”。

5. **“文献从哪里来的？”**
   - 仓库内只有 `review/plan/literature-review-candidates.md` 的候选清单式文本；缺少引用导出、DOI、URL、`.bib`、参考文献章节，无法审计为真实检索。

6. **“是否编造的数据、实证结果、文献？”**
   - 数据：是模拟数据。
   - 实证结果：部分（基准回归）可由模拟数据+代码复现；但机制/稳健性/异质性等关键数字有明显不可复现/不可溯源问题。
   - 文献：缺乏可核验出处，无法确认真实性；在审计口径下等同“不可证明为真”。

---

## 8. 建议（如果你要把它变成“真实可发表研究工程”）

- 数据：补齐 `code/01_fetch/`（下载/抓取）与 `data/raw/`（原始文件），并在 `results/` 增加 `data_manifest.json`（来源、下载日期、变量说明、哈希）。
- 清洗：把 `generate_data.py` 改为“从 raw 清洗到 clean”，并保留可重复的清洗日志。
- 结果：删除/替换不可追溯的“汇总数字表”，所有用于论文的数字必须由脚本自动生成（建议 `code/06_tables/` 写表格生成器）。
- 论文：在数据章节明确披露数据来源；若继续用模拟数据，应在摘要/数据章节明确标注“模拟”。
- 文献：引入 `.bib` 或参考文献清单，附 DOI/URL；保留检索记录（例如 CNKI/WoS 导出）。

