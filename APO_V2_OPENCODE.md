# APO-v2（OpenCode）实验说明

## 目的

当前 APO-v2 从固定 Multipass/P0 开始，先增加 1 轮自动化 Stage 1
`Configuration`/CPE 优化，再依次针对 `affects`、`instantiates` 和
`exploited_by` 做 3 轮 Stage 2 原子优化。旧的关系-only preset 和运行结果保留，
只用于历史复现；当前 Tuning/Formal 入口均采用新的 1+3 轮协议。

该流程仅使用 train 反馈和 dev 选优。现有 test Gold、冻结的 Rule、
Multipass/P0、Full 预测以及正式 APO 提示均不会被改写或重新评价。

当前第三章边界契约为 `chapter3-boundary-sync-v2`，Configuration 已按 MCPU v2 完成预测盲法全量审计与重新冻结；此前绑定 v1 的 APO 产物仅作历史记录。
程序现允许只读取 train/dev 的 Smoke、Tuning 和 Formal 优化；所有 APO 运行仍须
从固定 P0 重新开始，旧候选不得续跑。只有非 P0 候选通过预设的完整 dev 晋级门并
冻结后，才允许对 test 做一次性评价。

## 已验证参数

| 角色 | 模型与端点 | 参数 |
|---|---|---|
| 任务模型 | `hy3`，`/v1/chat/completions` | temperature=0，top_p=0.95，thinking=disabled，reasoning_effort=none，max_tokens=4096 |
| Critic | `muse-spark-1.3-contributor`，`/v1/responses` | temperature=0.1，top_p=1，reasoning_effort=xhigh，max_output_tokens=16384 |
| Editor | `muse-spark-1.3-contributor`，`/v1/responses` | temperature=0.1，top_p=1，reasoning_effort=xhigh，max_output_tokens=8192 |

4096 的 Editor 预算在真实候选提示上出现过推理预算耗尽，因此没有采用。
Pilot 中 Critic 的一次 8192 high 推理也触及预算上限并由重试恢复，因此最终
将 Critic 提高至 16384。该配置须通过 Configuration/CPE Stage 1 与三个关系
类型的 TRAIN-only Critic/Editor 语义预检后才能由统一入口启动实验。

## 选择与安全门槛

- 第 1 轮只编辑 Stage 1，目标固定为 `Configuration`；候选按
  mention/boundary、canonical CPE fields、唯一父族投影和歧义弃权四类
  预注册策略池生成，Muse 不得自行改换目标。低成本 Tuning 使用前两类各
  1 个候选，Formal 使用四类各 1 个候选。
- Stage 1 以候选实体层的端到端 CPE NA 选优；只有精确跨度与 canonical CPE
  `normalized_id` 同时命中才计为正确，漏检、错边界和错 CPE 均计为错误。
- 随后三轮只编辑 Stage 2，目标依次为 `affects`、`instantiates`、
  `exploited_by`；Stage 1 保持为程序自动选择的父候选。
- 14 文档 selection-dev 覆盖 dev 中全部 `exploited_by` 正例文档和全部
  `instantiates` 正例文档；最终晋级仍在完整 21 文档 dev 上做三次配对复核。
- 单个实体/关系类型 F1 相对 P0 最多下降 0.01。
- Configuration 候选召回不得低于 P0，Stage 1 的 CPE NA 在最终门上至少提高
  0.005；同一阈值用于各关系轮的原子目标 F1。
- 严格关系 micro-F1 和规范化关系 micro-F1 相对 P0 最多下降 0.005。
- 原子目标 F1 在完整 dev 上至少提高 0.005，并满足重复方向一致性。
- P0 始终保留为回退；正式 preset 不冻结提示，也不触碰 test。
- 未启用完整 dev 复核的小规模运行也必须通过原子目标增益门槛；不能因
  非目标关系的随机上升选择候选。

## 分阶段开发协议

小规模阶段用于调试 APO 算法和注册参数，不用于人工挑选提示词。每次
Tuning 都从固定 P0 重新开始，由 Muse 自动批评和改写，由程序自动选择；
不得把上一次运行中人工看过的候选复制为下一次初始提示。允许调整的对象
仅限候选数、beam、固定轮数、样本规模、自动目标函数和停止条件。

推荐顺序为：Smoke（历史关系-only 接口检查）→ Tuning（6 篇
selection-dev，1+3 轮）→
扩大规模的开发运行 → Formal → 冻结提示 → test 一次性评价。只有进入
Formal 前才锁死全部元参数；Tuning 结果不得写成 APO 的正式效果结论。

## 运行入口

在 `D:\BRON` 下运行：

```powershell
# 重新执行参数探针和 Stage-1 + 关系 Critic/Editor 语义预检
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode Validate -RefreshPreflight

# 只检查数据边界和运行配置
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode Validate

# 查看最低成本 smoke、历史 pilot、小规模 tuning 或正式开发实验的请求数上界
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode SmokeEstimate
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode PilotEstimate
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode TuningEstimate
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode Estimate

# 真实运行；小规模 Tuning 达到预注册门槛后才扩大规模
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode Smoke
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode Tuning
experiments\v5\scripts\run_apo_v2_opencode.ps1 -Mode Formal
```

当前预算上界：历史 Smoke 为 156 次 hy3 与 6 次 Muse；历史 Pilot 为 1666 次
hy3 与 6 次 Muse；新增 Stage 1 后的 6 文档 Tuning 为 2710 次 hy3 与 8 次
Muse；Formal 为 42036 次 hy3 与 14 次 Muse。上界未扣除缓存命中、无效候选
和结构早停。新的 1+3 轮 Tuning 尚未执行。

首轮 6 文档 Tuning 已完成：三轮共自动评价 6 个候选，实际调用 1360 次
hy3 与 6 次 Muse，任务阶段失败为 0。`affects` 两个候选均降低目标 F1；
`instantiates` 候选一降一平；`exploited_by` 候选均未改变目标 F1。程序最终
回退 P0，未生成可冻结 APO 提示。运行后发现小规模最终选择仍沿用探索阶段
的零增益阈值；现已修正为同样执行 0.005 的注册最终增益门槛。由于本轮所有
候选的目标增益均不大于 0，该修正不会改变本轮 P0 结论。

边界复裁后的第二次 6 文档 Tuning 从固定 P0 重新开始，三轮自动评价 6 个候选，
实际执行 1523 次 hy3 与 7 次 Muse 调用尝试，任务阶段失败为 0。程序最终选择
第一轮 `affects` 候选 `p2`：严格关系 micro-F1 从 0.6107 升至 0.6260，
`affects` F1 从 0.5556 升至 0.5753，规范化关系 micro-F1 保持 0.5905。
最后一轮 beam 候选 `p6` 因 `exploited_by` 目标增益为 0 被最终目标门排除；
`p1/p3/p4` 因规范化关系 F1=0.5849 低于 0.5855 下限被排除。该运行只有一次
selection-dev 重复且未做完整 dev 复核，`p2` 未冻结，test 未加载。

CPE Gold 审计并重新冻结为 v8 后，关系-only 的最新 6 文档 Tuning 再次从固定 P0 开始，
三轮自动评价 6 个候选，实际执行 1529 次 hy3 与 7 次 Muse 调用尝试，任务阶段
失败为 0。程序最终选择第二轮 `instantiates` 候选 `p4`：严格关系 micro-F1
从 0.6000 升至 0.6154，`instantiates` F1 从 0.8571 升至 0.9143；规范化关系
micro-F1 保持 0.6476，`affects` 与 `exploited_by` 不变。离线复算显示，增益来自
`aa23-213a` 中一条关系改绑到同句 CVE mention；前后 mention 的规范化 CVE ID
相同，因此没有规范化事实增益。本次仍只有一次 selection-dev 重复且未做完整
dev 复核，`p4` 未冻结，test 未加载。完整复核见该运行目录的 `RUN_SUMMARY.md`。

## 当前 smoke 结论

首次端到端 smoke 已完成三轮关系路由，无失败的任务阶段调用，且 P0 回退
正确拒绝了退化候选。由于它只有一个 train 文档和一个 dev 文档，不应用于
论文效果比较。下一步应先运行 Pilot；若完整门槛未通过，则保留 P0 并继续
改进错误路由或候选策略，不得据此读取或调整 test。

## 当前 Pilot 结果

4 文档、单次 selection-dev Pilot 已选择非 P0 候选 `p1`，其 Stage 1 指导
保持为空，只增加一条 `affects` 精度控制规则。与同批 P0 相比：实体 F1
均为 0.9202；严格关系 F1 从 0.5789 提升到 0.6071；关系类型宏 F1 从
0.582467 提升到 0.609867；规范化关系 F1 均为 0.6087。逐类关系 F1 中，
`affects` 为 0.4746（不变），`exploited_by` 为 0.5455（不变），
`instantiates` 从 0.7273 提升到 0.8095。

该结果是目标增益硬门槛加入前的探索性开发信号，不是最终实验结论。
由于候选目标为 `affects` 而该关系没有提升，按当前自动门槛它会回退到 P0；
因此不得人工保留 p1，也不得将其作为后续 APO 的初始提示。
Pilot 共执行 1095 次 hy3 调用尝试和 7 次 Muse 调用；任务阶段失败为 0。
Critic 在 8192 预算下有一次推理预算耗尽并由重试恢复，之后同一长提示在
16384 下单次通过，故最终入口采用 Critic 16384。Pilot 没有加载 test、
没有生成 test 预测，也没有冻结为正式提示。
