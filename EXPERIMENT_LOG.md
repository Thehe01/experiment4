# v6 实验台账

## 创建记录

- **包版本**：`v6`
- **创建日期**：2026-09-15
- **来源**：`experiments/v5` 的 `dataset_freeze_manifest_v5.json`
- **标注协议**：`4.6-mcpu-mention-fact-dual-layer-v1`
- **抽取模式**：`chapter3-no-capec-v1`
- **边界契约**：`chapter3-boundary-sync-v2`
- **划分**：官方 `v7`，train/dev/test = 63/21/21
- **Gold**：105 篇，已完成人工复标与分歧裁决，存放于 `data/annotations/gold/`
- **源文本**：物理复制到 `data/sources/`

## 重置原则

1. **绝对隔离**：`v6` 为全新独立的实验包，不读取 `v5` 的运行结果、历史测试预测或历史 APO 候选。
2. **命名规范**：`v6` 运行产物统一使用 `v6_*` 前缀；`full` 只能复用同一次 `v6_multipass` 原始预测。
3. **APO 重置**：APO 必须在 `v6` 环境下的 v7 train/dev 重新搜索与调优。
4. **盲法隔离与测试集门禁**：MCPU v2 的规则形成审计覆盖了现有 406 个 Configuration，因此原 test 降级为开发审计用途；正式测试须从未参与协议形成的文档另行预留并冻结，且只在提示词晋级后使用一次。
5. **Gold 与实验有效性**：Gold 已完成人工复标与分歧裁决；人工 IAA 尚无可核验结果，台账中的“有效”仅表示冻结数据上的自动受控实验有效。

## 2026-09-19 Configuration MCPU v2 升级

- 在不读取模型预测的前提下全量审计 105 篇文档中的 406 个 Configuration mention，并把边界规则收敛为最小规范产品单元（MCPU）。
- 同一连续产品名词短语中的紧邻厂商保留；跨表格列、并列项、列表项、句子或非连续文本不得补入厂商。官方产品/组件标识保留，纯类别或部署形态后缀删除。
- 普通发布版本不进入表面跨度，普通自然语言 mention 的 CPE 版本字段统一为 `*`；完整字面 CPE 和产品身份所需的代际/型号标识例外。
- 明确修订 15 个跨度和 7 个 CPE，涉及 17 个不重复 mention；完整裁决台账由 `configuration_boundary_mcpu_v2_review.*` 保存。
- 由于规则形成审计覆盖了原 test，旧 63/21/21 划分只用于开发与可追溯审计；在新未见 test 冻结之前，受控 test 与正式实验门禁保持关闭。

## 2026-09-16 ProTeGi Stage 1 事后审计与修复

- 首次 Stage 1 产物 `results/protegi_optimization/entity_protegi/` 已封存为**审计失败样本**，不得晋级 Stage 2，也不得作为论文中的有效 ProTeGi 结果。
- 主要问题：候选可改写冻结边界与 few-shot；14/24 个示例 offset 错误；Stage 1 混入关系输出；任务/优化模型角色未完整留痕；UCB 候选看到不同的文件前缀批次且最低拉动不足；最终 Dev 候选、分类型结果和原始预测未落盘；Stage 2 缓存缺少关系 Gold 并存在回退 Gold 实体的路径。
- 已实施代码级修复：冻结 `chapter3-boundary-sync-v1` 契约，只允许 Muse 改写 guidance；正式配置显式绑定任务模型 `hy3`（temperature=0, reasoning effort=none）与优化模型 `muse-spark-1.3-contributor`（temperature=0.1, reasoning effort=high）；UCB 采用每轮固定洗牌的共享 candidate-local 批次且每候选至少 2 pulls；完整记录生成/拒绝/未抽样/评估候选；最终 Dev 保存逐类型指标与原始预测；缓存增加内容哈希与 Gold 关系门禁；Stage 2 删除 Gold 实体静默回退；最终产物增加全目录哈希清单。
- 依用户要求，本次仅修复，**未运行单元测试、模型调用或新一轮 Stage 1**。修复版应使用新的空输出目录运行，旧目录不会被覆盖。

## 2026-09-16 ProTeGi 提示搜索范围双分支

- 新增等预算 `prompt_scope=constrained` 与 `prompt_scope=unconstrained` 两个实验臂，配对标识为 `protegi-prompt-scope-v1`。
- 有约束臂仅允许 Muse 改写 `OPTIMIZABLE_GUIDANCE`，候选必须与 P0 的冻结边界契约、few-shot 和 JSON schema 精确一致。
- 无约束臂允许 Muse 改写完整语义提示、任务定义和 few-shot；程序只冻结 `{text}`／`{entities}` 输入占位符及评价器所需 JSON 字段。候选同时记录冻结契约精确匹配结果，供后续边界漂移审计，不将其用于自动选优。
- 两臂正式配置除 `prompt_scope` 外保持相同的 P0、模型、随机种子、轮数、Beam、生成规模与 Task Model pull 预算；输出目录和实体缓存按实验臂隔离，并在 summary、Dev 记录和缓存 manifest 中绑定分支标识。该设计保证等调用与等 Task 评估窗口，不预设 Muse token 成本相等；后者依据 `call_stats` 单独报告。
- 依用户要求，本次仅增加代码、配置、静态测试用例与文档，**未运行单元测试、冒烟测试、模型调用或实验**。

## 2026-09-16 ProTeGi 双分支小规模 pilot 准备

- 冻结 `data/protegi_prompt_scope_pilot_v1.json`：Train 4 篇（预计 24 windows），Dev 4 篇（预计 24 windows），两侧均覆盖四类抽取实体，Test 为空；选样仅依据官方 Train/Dev 归属、Gold 类型覆盖与文档规模，不参考模型输出。两侧均形成 3 个完整的 8-window 批次，无不足 8 个窗口的尾批次。
- 新增 `protegi_pilot_constrained.yaml` 与 `protegi_pilot_unconstrained.yaml`：2 轮、Beam=2、每父代保留 1 个后继、每轮 8 pulls、每次 8 windows、每候选至少 2 pulls、seed=42；除 `prompt_scope` 外配置相同。
- 所有 ProTeGi 配置新增 `task_max_workers=8`；hy3 窗口预测在 Train、最终 Dev 和实体缓存生成时统一按最多 8 路并发执行，并使用线程锁维护调用与 token 统计。Muse 搜索依赖链不并行化。
- 新增 `run_protegi_scope_pilot.ps1`，在未来执行时先检查 split 哈希、文档归属、Test 隔离、配置等价和输出目录，再运行纯逻辑测试与两个实验臂；失败时不继续下一阶段，不覆盖既有结果。
- 新增 `compare_protegi_scope_pilot.py`，未来仅离线读取两臂产物，核对共同 P0、配置和 split 后输出总体/分类型指标、调用量及预注册晋级状态。
- 预计两臂合计最多约 464 次 hy3 窗口调用和 18 次 Muse 调用；这只是方向性 pilot，不用于正式有效性结论。
- 依用户要求，**本次未运行测试、pilot、比较脚本或任何模型调用**。

## 2026-09-17 Configuration 定向有约束 ProTeGi pilot 准备

- 新增隔离实验清单 data/protegi_configuration_pilot_v1.json：Train 沿用 4 篇官方 Train 文档，Dev 改用 3 篇此前 prompt-scope pilot 未观察的官方 Dev 文档，Test 为空。实际 build_text_windows 计数为 Train 25、Dev 34；不再沿用旧 pilot 将两侧估算为 24 windows 的长度公式。
- 增加 explicit-abbreviation-context-v1：只从全文显式 Long Form (SF) 或 SF (Long Form) 结构恢复词汇对应关系，不调用模型、不判定 Configuration、不自动回填实体。该固定上下文对 P0 与所有候选一致；没有显式定义时为 no-op。
- ProTeGi 批评输入仅保留 Train 上的 Configuration FN、同类型边界重叠错误和非 Gold FP；UCB 奖励与最终首要目标均改为 Configuration Strict F1。Jaccard 不低于 0.5 的 overlap F1 只作诊断，绝不参与选优。
- 最终新鲜 Dev 决选允许 P0 自动胜出。候选须先通过 Vulnerability、Weakness、AttackTechnique 相对 P0 最多下降 0.02 的门禁，再按 Configuration Strict F1、总体 Strict Micro-F1、总体 Strict Precision、提示词长度和 candidate ID 确定性排序。
- CPE 受控链接明确排除在本轮之外；normalized_id 仅保留既有接口，不新增词典、检索、约束解码或基于 CPE 的评分。
- 固定 task model=hy3、task_max_workers=8、optimizer model=muse-spark-1.3-contributor。启动脚本为 scripts/run_protegi_configuration_pilot.ps1，输出目录为 results/protegi_optimization/configuration_pilot_v1/entity_constrained。
- 已完成离线验证：tests/test_v6_package.py 11/11、tests/test_protegi.py 32/32，Configuration pilot preflight 通过；model_calls=0、network_calls=0。依用户要求，**尚未运行模型实验**。

---

## 待办与执行状态表

| 实验项目 | 状态 | 产物路径 / 对应脚本 |
|---|:---:|---|
| **v6 数据集冻结清单** | ✅ 已完成 | `data/dataset_freeze_manifest_v6.json` |
| **Gold 语义与格式审计** | ✅ 已通过 | `results/gold_strategy_audit_v6_v7.json` |
| **数据划分防泄露审计** | ✅ 已通过 | `results/split_leakage_audit_v6_v7.json` |
| **包边界单元测试** | ✅ 11/11 通过（2026-09-17） | `tests/test_v6_package.py` |
| **规则基线 (Rule)** | ⏳ 待运行 | `python -X utf8 experiments/v6/scripts/run_v6_experiment.py --method rule` |
| **Multipass 基线 (P0)** | ⏳ 待运行 | `python -X utf8 experiments/v6/scripts/run_v6_experiment.py --method multipass` |
| **Full 确定性增强** | ⏳ 待运行 | `python -X utf8 experiments/v6/scripts/run_v6_experiment.py --method full` |
| **基线结果冻结** | ⏳ 待冻结 | `python -X utf8 experiments/v6/scripts/freeze_baseline_methods_v6.py --create` |
| **ProTeGi Stage 1 首次运行** | ❌ 审计失败并封存 | `results/protegi_optimization/entity_protegi/`（不得晋级） |
| **ProTeGi 修复版静态/单元检查** | ✅ 32/32 通过（2026-09-17） | `python -B -X utf8 experiments/v6/tests/test_protegi.py` |
| **ProTeGi Stage 1 有约束臂** | ⏳ 待执行 | `protegi_formal_constrained_muse.yaml`；新空输出目录 |
| **ProTeGi Stage 1 无约束臂** | ⏳ 待执行 | `protegi_formal_unconstrained_muse.yaml`；新空输出目录 |
| **ProTeGi 双分支小规模 pilot** | 🧰 已准备、未运行 | `run_protegi_scope_pilot.ps1`；固定清单 `protegi_prompt_scope_pilot_v1.json` |
| **Configuration 定向有约束 ProTeGi pilot** | 🧰 已准备且离线预检通过、未运行模型 | `scripts/run_protegi_configuration_pilot.ps1`；固定清单 `data/protegi_configuration_pilot_v1.json` |
| **ProTeGi Stage 2** | 🔒 待修复版 Stage 1 与缓存通过门禁 | `run_protegi.py --stage relation ...` |
| **Test 一次性评估** | 🔒 继续锁定 | 仅在 P_E*、实体缓存与 P_R* 全部冻结后执行 |
| **Full 映射补全层评估** | ⏳ 待运行 | `python -X utf8 experiments/v6/scripts/evaluate_mapping_completion.py` |
| **文档级配对 Bootstrap 检验** | ⏳ 待运行 | `python -X utf8 experiments/v6/scripts/paired_bootstrap_v6.py` |

---

## 实验结果汇总表 (待填充，Test = 21 篇, 131 条关系)

| 方法 | 实体 F1 | 严格关系 F1 | 规范化事实 F1 | 预测证据歧义率 | affects F1 | exploited_by F1 | instantiates F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rule-based** | - | - | - | - | - | - | - |
| **LLM-Multipass (P0)** | - | - | - | - | - | - | - |
| **Full 增强方案** | - | - | - | - | - | - | - |
| **ProTeGi** | - | - | - | - | - | - | - |
| **ProTeGi_Full** | - | - | - | - | - | - | - |
