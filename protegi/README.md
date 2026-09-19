# ProTeGi 提示词优化框架 (ProTeGi for Structured Vulnerability Extraction)

面向网络安全漏洞知识抽取任务的两阶段提示词自动优化系统。

双分支的固定比较口径与晋级规则见 `PROMPT_SCOPE_EXPERIMENT.md`。

---

## 1. 方法定位与学术规范说明

> **重要学术说明**：  
> ProTeGi（Prompt Optimization with Textual Gradients and Beam Search）是 Pryzant 等人（EMNLP 2023）提出的通用黑盒提示词优化算法。**本项目不将 ProTeGi 声称为“本文提出的算法”**。  
> 
> 本文的核心工作与领域适配包括：
> 1. **结构化网络安全抽取适配**：将原用于文本分类的 ProTeGi 迁移至复杂的实体边界抽取与关系三元组抽取，并预注册有约束与无约束两个提示搜索范围；
> 2. **严格 Micro-F1 奖励机制**：优化目标采用无偏且严谨的 Strict Entity Micro-F1 与 Strict Relation Micro-F1；
> 3. **两阶段独立解耦搜索**：实体（Stage 1）与关系（Stage 2）拥有各自独立的 Prompt 搜索空间、Beam 集束、文本梯度批评与验证选优；
> 4. **冻结上游实体缓存保证公平性**：Stage 2 关系提示词搜索严格运行在由最优实体提示词 $P_E^*$ 预先生成的固定实体预测上；缓存绑定 P_E*、内容哈希、样本顺序和关系 Gold，缺失时直接终止；
> 5. **严谨的消融基准矩阵**：对比初始提示词 (`initial`)、无梯度的释义扩展 (`mc`)、贪心搜索 (`greedy_protegi`) 以及均匀分配评估预算 (`protegi_uniform`)，系统性分析自适应分配与梯度引导的贡献。

### 1.1 提示搜索范围实验臂

| `prompt_scope` | Muse 可改写内容 | 候选准入条件 | 研究定位 |
| :--- | :--- | :--- | :--- |
| `constrained` | 仅 `OPTIMIZABLE_GUIDANCE` | 冻结契约、few-shot、标签边界及 JSON schema 与 P0 精确一致 | 主实验候选，确保优化前后任务定义不变 |
| `unconstrained` | 完整语义提示，包括任务定义、规则和 few-shot | 保留 `{text}`／`{entities}` 输入占位符及评价器需要的 JSON 字段 | 搜索空间消融；最终结果必须另做边界漂移审计 |

两个实验臂使用相同 P0、Train/Dev、随机种子、模型参数、轮数、Beam、候选数和
Task Model pull 预算。`summary.json`、`final_dev_evaluations.json` 与实体缓存清单均
记录 `prompt_scope`；无约束候选另行记录 `frozen_contract_exact_match`，但该项仅供
审计，不参与其 Train/Dev 自动选择。两臂的模型调用次数预算相同，但无约束元提示
包含完整候选提示词，Muse 的实际输入 token 可能更多，因此必须依据 `call_stats`
另行报告优化模型 token 消耗，不能把“等调用数”表述为“等 token 成本”。

---

## 2. 总体架构与两阶段流水线

```
[Stage 1: 实体抽取优化]
P_E0 (种子实体提示词)
  │
  ▼ ProTeGi 搜索 (Beam=4, 错误样本 -> 文本梯度批评 -> 针对性重写 -> 释义扩展 -> UCB自适应评估)
P_E* (最优实体提示词)
  │
  ├─> 冻结 P_E*，构建 Train/Dev 实体预测缓存 (带 SHA-256 签名校验)
  │
  ▼
[Stage 2: 关系抽取优化]
P_R0 (种子关系提示词) + 固定的 P_E* 实体输入
  │
  ▼ ProTeGi 搜索 (Beam=4, 关系错误样本 -> 文本梯度批评 -> 针对性重写 -> 释义扩展 -> UCB自适应评估)
P_R* (最优关系提示词)

[最终端到端推理]
Text ───(P_E*)───> Entities ───(P_R*)───> Relations
```

---

## 3. 对照实验与消融矩阵及公平预算设计

为消除 Task Model API 评估窗口数量对实验结论的干扰，除 `initial`（零步 baseline）外，所有 4 种搜索方法在每一优化轮次中拥有**严格相等且完全一致的任务模型评估总预算（total_pull_budget_per_round = 64 pulls，每次 8 windows，单轮共计 512 评估窗口）**：

| 方法标识 | 方法描述 | 集束 (Beam) | 每轮产生候选数 $K$ | 每轮 Optimizer 模型调用 | 每轮 Task 候选评估 Pulls | 每轮 Task 评估窗口数 | 候选评估与选择机制 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `initial` | 零优化基线 | 1 | 1 ($P_0$) | 0 | 0 | 0 | 仅在最终阶段独立评测 P0 的 Dev/Test 表现 |
| `mc` | 释义消融基线 | 4 | $\le 20$ | 16 (R1: 4) | **64 pulls** | **512 windows** | 纯蒙特卡洛释义 (无梯度)，UCB 自适应分配 64 pulls |
| `greedy_protegi` | 贪心消融基线 | 1 | $\le 5$ | 13 (固定) | **64 pulls** | **512 windows** | 梯度批评 + guidance 重写 + 释义，UCB 自适应分配 64 pulls |
| `protegi` | **主方法 (Full ProTeGi)** | 4 | $\le 20$ | 52 (R1: 13) | **64 pulls** | **512 windows** | 每候选先在相同共享批次上获得至少 2 pulls，再由 UCB 自适应分配余量 ($c=2.0$) |
| `protegi_uniform` | 均匀分配消融基线 | 4 | $\le 20$ | 52 (R1: 13) | **64 pulls** | **512 windows** | 相同共享批次、相同总预算与最低 pulls，仅将余量改为均匀轮转 |

> **注**：正式配置显式固定任务模型 `hy3`（temperature=0, reasoning effort=none）与优化模型 `muse-spark-1.3-contributor`（temperature=0.1, reasoning effort=xhigh）。每轮 Train 窗口经固定种子洗牌，candidate-local pull index 对应同一共享批次；不再使用固定文件前缀。
> 所有 ProTeGi 配置显式设置 `task_max_workers=8`。任务模型窗口在 Train、Dev 和实体缓存生成时统一以最多 8 路并发执行；Muse 的梯度—编辑—释义依赖链仍按搜索顺序执行。

---

## 4. 模块结构

```
experiments/v6/protegi/
├── __init__.py           # 包定义与对外导出接口
├── models.py             # PromptCandidate, PromptGradient, ErrorExample, CallStats 等核心数据类
├── prompts_p0.py         # 冻结契约 + 可优化 guidance + 已校验 reference examples
├── templates.py          # constrained / unconstrained 两组批评、编辑与释义元提示词
├── contract_validator.py # 冻结契约门禁与无约束臂固定运行接口门禁
├── metrics.py            # 严格 Micro-F1 评估累加器 (复用 eval_metrics.py 严谨匹配逻辑)
├── evaluator.py          # 任务模型推理与错误样本收集执行器 (直接使用完整 Prompt，绕开 base+guidance)
├── mutators.py           # 梯度生成器 (GradientGenerator)、编辑重写器 (PromptEditor)、释义器 (MonteCarloParaphraser)
├── selectors.py          # UCB 候选臂选择器 (UCBPromptSelector) 与 均匀分配选择器 (UniformPromptSelector)
├── entity_cache.py       # Stage 2 冻结实体预测缓存与 SHA-256 哈希完整性管理器
├── lineage.py            # 提示词演化谱系追踪器 (支持导出 JSON 树与 Graphviz .dot 图)
├── logging_utils.py      # 每轮产物归档器 (round_X/beam.json, candidates.json, optimization_curve.csv)
├── optimizer.py          # 两阶段提示词搜索核心调度引擎
└── configs/
    ├── protegi_formal_constrained_muse.yaml   # 等预算有约束臂
    ├── protegi_formal_unconstrained_muse.yaml # 等预算无约束臂
    ├── protegi_pilot_constrained.yaml         # 小规模配对 pilot 有约束臂
    ├── protegi_pilot_unconstrained.yaml       # 小规模配对 pilot 无约束臂
    ├── protegi_dryrun_constrained.yaml        # 有约束联调配置
    ├── protegi_dryrun_unconstrained.yaml      # 无约束联调配置
    └── protegi_formal_muse.yaml / protegi_dryrun.yaml # 兼容别名，均为 constrained
```

---

## 5. 快速运行指引

运行环境需配置 OpenCode Go 或兼容的 OpenAI 接口密钥：

#### 5.1 联调冒烟测试 (Dry-run)
```bash
# 有约束 Stage 1 冒烟测试
python scripts/run_protegi.py --stage entity --method protegi \
  --config protegi/configs/protegi_dryrun_constrained.yaml \
  --dry-run --output-dir results/protegi_optimization/pilot_entity_constrained

# 无约束 Stage 1 冒烟测试（等预算；使用另一个空目录）
python scripts/run_protegi.py --stage entity --method protegi \
  --config protegi/configs/protegi_dryrun_unconstrained.yaml \
  --dry-run --output-dir results/protegi_optimization/pilot_entity_unconstrained
```

### 5.2 小规模等预算对比（Stage 1）

固定 pilot 清单覆盖四类实体且不包含 Test。准备好的入口为：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_protegi_scope_pilot.ps1
```

该入口会先执行纯逻辑测试，再顺序运行两个实验臂并离线生成比较报告；不会覆盖已有目录。
当前只完成准备，尚未执行。

### 5.3 阶段一：实体提示词优化 (Stage 1)
```bash
# 运行有约束臂
python scripts/run_protegi.py --stage entity --method protegi \
  --config protegi/configs/protegi_formal_constrained_muse.yaml \
  --output-dir results/protegi_optimization/entity_protegi_constrained

# 运行无约束臂（相同预算）
python scripts/run_protegi.py --stage entity --method protegi \
  --config protegi/configs/protegi_formal_unconstrained_muse.yaml \
  --output-dir results/protegi_optimization/entity_protegi_unconstrained

# 运行释义消融
python scripts/run_protegi.py --stage entity --method mc

# 运行均匀评估消融
python scripts/run_protegi.py --stage entity --method protegi_uniform
```
必须使用新的空输出目录；两个实验臂不得共用输出目录或实体缓存目录。为避免混合不同配置和随机状态，当前实现不自动续跑旧 checkpoint。产物除最终提示词外，还包括所有生成候选、最终 Dev 原始预测、逐类型指标、完整谱系、提示范围审计和 `artifact_manifest.json`。

### 5.4 阶段过渡：构建冻结实体缓存
在 Stage 1 产出 $P_E^*$ 后，对 Train 和 Dev 执行离线预测固化：
```bash
python scripts/run_protegi.py \
  --stage build_entity_cache \
  --entity-prompt-file results/protegi_optimization/entity_protegi_repaired/final_entity_prompt.txt \
  --entity-cache-dir results/protegi_optimization/entity_cache_repaired
```

### 5.5 阶段二：关系提示词优化 (Stage 2)
```bash
python scripts/run_protegi.py \
  --stage relation \
  --method protegi \
  --entity-prompt-file results/protegi_optimization/entity_protegi_repaired/final_entity_prompt.txt \
  --entity-cache-dir results/protegi_optimization/entity_cache_repaired \
  --output-dir results/protegi_optimization/relation_protegi_repaired
```

---

## 6. 测试集隔离红线

所有提示词搜索、批评、重写与 Dev 验证过程中，**严禁加载或评估 Test 集（21 篇独立文档）**。  
只有当 Stage 1 最优提示词 $P_E^*$ 与 Stage 2 最优提示词 $P_R^*$ 均最终确定并冻结后，方可受控执行一次性独立测试集评估。
