# Configuration 定向有约束 ProTeGi 小规模实验协议

## 目的与边界

本分支只验证：在第三章冻结实体定义不变的前提下，有约束 ProTeGi 能否通过自动批评和自动改写，提高 Configuration 的严格字符跨度 F1。它不引入人工挑选候选，不修改 Gold，不在优化中访问 Test，也不把 CPE 链接纳入 Stage 1。

normalized_id 字段继续保留在既有输出接口中，但本轮不增加 CPE 词典、候选检索、Trie 约束解码或基于 CPE 的候选评分。CPE 对齐属于后处理/实体链接实验。

## 文档缩写上下文

程序可在切窗前扫描全文中显式出现的 Long Form (SF) 或 SF (Long Form) 定义，并把这些字符串对作为固定的文档级词汇上下文复制到该文档的各窗口。该扫描：

1. 不调用模型；
2. 不预识别 Configuration；
3. 不对未显式定义的缩写做推断；
4. 不自动回填实体；
5. 不改变任何预测跨度；
6. 对 P0 和所有 ProTeGi 候选完全相同。

即使短写出现在该上下文中，模型仍须依据当前局部文本和第三章冻结定义，独立判断该次出现是否属于 Configuration。

## 数据隔离

- Train：沿用 protegi_prompt_scope_pilot_v1 的 4 篇官方 Train 文档。Train 错误本来就是 ProTeGi 的可见优化信号。
- Dev：改用此前提示词范围 pilot 未使用的 3 篇官方 Dev 文档。
- Test：清单必须为空，实验脚本也不提供 Test 路径。
- 已观察的旧 pilot Dev 只允许作为问题诊断材料，不参与本轮选优。

运行时窗口数必须由 build_text_windows(max_chars=3000, overlap=400) 实际生成并与 manifest 核对，不再使用字符长度估算公式。

## 自动优化与选优

从固定 P0 开始，运行 2 轮、Beam=2 的有约束 ProTeGi。Muse 只能重写 OPTIMIZABLE_GUIDANCE；实体类型、边界、示例和 JSON 契约保持冻结。

每轮仅向批评/改写模型提供当前 Train minibatch 中的 Configuration 错误：

- 严格漏检（FN）；
- 同类型、Jaccard 不低于 0.5 但边界不完全相同的错误；
- 非 Gold Configuration 预测（FP）。

UCB 搜索奖励使用 Train 上的 Configuration Strict F1。搜索轮次结束前不得读取 Dev。

最终 Beam 与 P0 在新鲜 Dev 上各评估一次，由程序自动决选：

1. 首先要求 Vulnerability、Weakness、AttackTechnique 的 Strict F1 相对 P0 均不得下降超过 0.02；
2. 合格候选按 Configuration Strict F1 降序；
3. 再按总体 Strict Micro-F1、总体 Strict Precision、较短提示词和 candidate ID 确定性打破平局；
4. 若改写候选未超过 P0，或无法通过门禁，P0 可以自动胜出。

同类型跨度 Jaccard 不低于 0.5 的 overlap 指标只用于解释边界误差，不进入 UCB、门禁或最终排序。

## 可复现参数

- Task model：hy3
- Task model 并发：8
- 批评与改写模型：muse-spark-1.3-contributor
- Seed：42
- 配置：protegi/configs/protegi_configuration_pilot_v1.yaml
- 切分：data/protegi_configuration_pilot_v1.json
- 输出：results/protegi_optimization/configuration_pilot_v1/entity_constrained

本协议准备完成后不自动调用模型；必须由人工明确启动脚本。
