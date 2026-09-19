# 从 v4.5 迁移到 v5

v5 是实验运行包的重置，不是一次新的标注策略变更。数据内容来自 v4.5/v13
冻结快照，仍遵循 `4.5-mention-fact-dual-layer-v1`；复制后的 Gold 与源文本
通过 `dataset_freeze_manifest_v5.json` 重新绑定哈希。

已迁移：

- 105 篇 Gold 与 102 份源文本文件；
- 官方 `train_dev_test_split_v7.json`；
- v4.5 Gold 审计、泄漏审计和 7 篇 moved-dev 复核收据的只读副本；
- 当前 schema、规则基线、LLM 抽取、APO 优化器及端到端运行器。

未迁移：

- v4.5 的 `results/` 历史结果和 `raw_predictions/`；
- `iaa_v4_blind`、旧 adjudication 批处理脚本和旧提示候选；
- 任何 API key 文件。

v5 使用 `v5_*` 输出命名，并将 APO 产物固定为
`results/apo_optimization_v5/final_prompt.json`。代码仍接受 `V3_*` 环境变量作为
兼容桥，同时推荐新运行使用 `V5_*`；兼容桥只在 `V3_*` 未显式设置时生效。
