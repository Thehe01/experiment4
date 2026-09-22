# ProTeGi 提示搜索范围配对实验协议

状态：代码与配置已建立，尚未运行。

## 1. 实验问题

在相同 P0、数据、模型与搜索预算下，允许 ProTeGi 改写完整语义提示，是否比仅改写
operational guidance 获得更好的冻结 Dev 抽取效果；其增益是否伴随第三章实体/关系
边界漂移。

## 2. 实验臂

| 实验臂 | 配置 | Muse 搜索空间 | 自动准入门禁 |
|---|---|---|---|
| constrained | `protegi_formal_constrained_muse.yaml` | 仅 `OPTIMIZABLE_GUIDANCE` | P0 冻结契约精确匹配 |
| unconstrained | `protegi_formal_unconstrained_muse.yaml` | 完整语义提示、任务定义和 few-shot | 固定输入占位符与 JSON 运行接口 |

两个实验臂统一绑定 `experiment_pair_id=protegi-prompt-scope-v1`。

## 3. 固定条件

- 相同的实体/关系 P0；
- 相同的 v7 Train/Dev 文档与窗口构造；
- 相同的 `hy3` 任务模型参数；
- 所有 hy3 窗口调用统一使用 `task_max_workers=8`；该并发设置覆盖 Train 候选评估、
  最终 Dev 评估以及后续实体缓存构建，并保持输入顺序汇总；
- 相同的 `muse-spark-1.3-contributor` 批评与改写参数；
- 相同随机种子、轮数、Beam、每父代后继数、最低 pulls 和每轮 Task pull 总预算；
- 搜索只使用 Train；完成全部轮次后，每个实验臂只进行一次统一 Dev 决选；
- 程序自动生成和选择候选，不进行人工候选挑选或改写；
- Test 在方法和提示冻结前始终隔离。

等预算指调用次数、候选规模和 Task 评估窗口相同。由于无约束元提示包含完整候选提示，
Muse 输入 token 可能更多，必须根据 `call_stats` 单独报告，而不能宣称等 token 成本。

## 4. 报告指标

- 严格实体或关系 Micro Precision、Recall、F1；
- 各实体或关系类型的分类型指标；
- 实体规范化准确率及其适用样本数；
- 生成候选数、运行接口拒绝数、重复数和实际评估数；
- Task Model 与 Muse 的调用数和估算 token；
- 最终提示的 `runtime_interface_valid` 与 `frozen_contract_exact_match`；
- 无约束最终提示相对于 `chapter3-boundary-sync-v2` MCPU 契约的边界漂移审计结果。

## 5. 分支选择与晋级规则

1. 两个实验臂分别完成自动 Train 搜索和冻结 Dev 决选后，才查看臂间结果。
2. 无约束臂最终提示必须通过第三章边界、文本抽取层范围、关系方向和直接证据语义审计；
   运行接口有效不能替代该审计。
3. 若无约束臂未通过边界审计，则不得晋级，即使 Dev F1 更高；该结果作为任务定义漂移案例报告。
4. 若两臂均符合第三章边界，则以同口径冻结 Dev 严格 F1 选择；相同 F1 时依次比较
   Precision、提示长度，仍相同时优先 constrained。
5. 选定实验臂及 P_E* 后冻结配置、提示哈希与实体缓存；另一实验臂不进入 Test。
6. Stage 2 沿用选定的 `prompt_scope` 和独立缓存；P_E*、P_R* 均冻结后才允许一次 Test 评价。

## 6. 产物隔离

- 有约束输出：`results/protegi_optimization/entity_protegi_constrained/`
- 无约束输出：`results/protegi_optimization/entity_protegi_unconstrained/`
- 实体缓存目录必须按 `prompt_scope` 分开，缓存 manifest 同时绑定提示哈希、任务模型与
  `prompt_scope`。

已有 `results/protegi_optimization/entity_protegi/` 是审计失败的历史样本，不属于本次
配对实验，也不得作为任一实验臂的续跑目录。

## 7. 小规模 Stage 1 pilot（已准备，未运行）

- 固定清单：`data/protegi_prompt_scope_pilot_v1.json`；仅从官方 v7 Train/Dev 中按
  Gold 实体类型覆盖选取，不参考模型输出，`test=[]`。
- Train：4 篇、实际 25 个窗口；Vulnerability/Configuration/Weakness/
  AttackTechnique 分别为 49/26/4/67 个提及。
- Dev：4 篇、实际 25 个窗口；四类实体分别为 78/22/5/40 个提及。
- 两臂统一参数：2 轮、Beam=2、每父代保留 1 个后继、minibatch=8、每轮 8 pulls、每次 8 windows、
  每候选至少 2 pulls、seed=42、hy3 任务模型 8 路并发。
- Train 与 Dev 均形成 4 个批次（8/8/8/1），尾批次保留；
  selector 按每批实际窗口数记录 `samples_seen`。
- 配置：`protegi_pilot_constrained.yaml` 与 `protegi_pilot_unconstrained.yaml`。
- 一键入口：`scripts/run_protegi_scope_pilot.ps1`。入口先核对官方划分哈希、文档归属、
  Test 为空、配置等预算和新输出目录；默认先运行纯逻辑测试，通过后才按顺序调用两臂。
- 离线汇总：`scripts/compare_protegi_scope_pilot.py`。只有两臂均完成后才比较共同 P0、
  配置、split 哈希、总体和分类型指标、调用量及契约审计，并生成 JSON/Markdown。

实际调用量以两臂 summary 中的 `call_stats` 和真实尾批次记录为准；它可能因没有错误样本、
候选格式不合格或候选去重而下降。该规模只用于端到端可执行性与
方向性信号判断，不能替代后续扩大样本、稳定性分析或冻结 Test 评价。
