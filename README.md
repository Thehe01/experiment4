# v6 受控实验包

`v6` 是从 `experiments/v5` 独立分离出的崭新实验包。它使用
`chapter3-no-capec-v1` 抽取模式和 `4.6-mcpu-mention-fact-dual-layer-v1` 标注协议；
以 `chapter3-boundary-sync-v2` 为上位契约。

## 当前状态

- **Gold**：`data/annotations/gold/` 中的 105 篇保留人工复标与分歧裁决来源；在 v1 复裁基础上，406 个 Configuration mention 已完成 MCPU v2 全量审计，修订 15 个跨度和 7 个普通版本 CPE。
- **划分**：`data/train_dev_test_split_v7.json`，63/21/21 篇现有开发划分。由于 MCPU 规则形成覆盖了原 test，正式评价还需另行预留未见 test。
- **冻结清单**：`data/dataset_freeze_manifest_v6.json`，绑定 Gold、划分和审计证据哈希。
- **语义与划分门禁**：当前边界审计通过（`gold_strategy_audit_v6_v7.json` 与 `split_leakage_audit_v6_v7.json`）。
- **运行器与隔离**：`run_v6_experiment.py`，结果使用 `v6_*` 前缀；最终 test 门禁在新未见 test 冻结前保持关闭。
- **ProTeGi 优化器**：`scripts/run_protegi.py`。首次 Stage 1 产物已在 2026-09-16 事后审计中判为不可晋级；修复版现包含等预算 `constrained`（仅优化 guidance）与 `unconstrained`（优化完整语义提示、固定运行接口）两个实验臂，并补齐公平选择、Dev 留痕、缓存与哈希门禁，但依用户要求尚未运行。
- **当前门禁**：ProTeGi 的 Test 评价保持锁定；必须在新空目录重跑 Stage 1，审核 P_E* 后构建实体缓存，再运行 Stage 2。

## 目录结构

```text
v6/
├─ ANNOTATION_GUIDELINE.md
├─ CHAPTER3_EXPERIMENT_ALIGNMENT.md 第三章驱动的第四章实验执行稿
├─ EXPERIMENT_LOG.md                v6 实验台账
├─ config/runtime_profile.json
├─ data/
│  ├─ annotations/gold/             105 篇冻结 Gold 副本
│  ├─ sources/                      对应源文本副本
│  ├─ train_dev_test_split_v7.json  v7 划分
│  ├─ dataset_freeze_manifest_v6.json
│  └─ review_status.json
├─ scripts/
│  ├─ run_v6_experiment.py          端到端运行入口
│  ├─ freeze_baseline_methods_v6.py  Rule/P0/Full 冻结与离线校验
│  ├─ apo_optimizer.py              历史 APO 实现（当前主实验不使用）
│  ├─ run_protegi.py                两阶段 ProTeGi 运行入口
│  ├─ promote_apo_v6.py             APO 晋级门禁
│  ├─ paired_bootstrap_v6.py        文档级配对 bootstrap 检验
│  ├─ schema.py / eval_metrics.py
│  ├─ rule_baseline.py / llm_methods.py
│  └─ audit_*.py                    Gold、划分与契约门禁
├─ tests/test_v6_package.py         包边界与冻结一致性测试
└─ results/                         v6 新结果，不承接旧 v5 结果
```

## 执行流程

从仓库根目录运行：

```powershell
# 1. 运行包边界与一致性单元测试（0 API 调用）
python -X utf8 experiments/v6/tests/test_v6_package.py

# 2. 运行规则基线
python -X utf8 experiments/v6/scripts/run_v6_experiment.py --method rule --split test

# 3. 运行 Multi-stage 基线 (P0)
python -X utf8 experiments/v6/scripts/run_v6_experiment.py --method multipass --split test

# 4. 运行 Full 确定性增强
python -X utf8 experiments/v6/scripts/run_v6_experiment.py --method full --split test

# 5. 冻结三种基线结果
python -X utf8 experiments/v6/scripts/freeze_baseline_methods_v6.py --create

# 6a. 有约束 ProTeGi Stage 1（必须使用新的空输出目录）
python -X utf8 experiments/v6/scripts/run_protegi.py --stage entity --method protegi --config experiments/v6/protegi/configs/protegi_formal_constrained_muse.yaml --output-dir experiments/v6/results/protegi_optimization/entity_protegi_constrained

# 6b. 无约束 ProTeGi Stage 1（与 6a 等预算，必须使用另一个空目录）
python -X utf8 experiments/v6/scripts/run_protegi.py --stage entity --method protegi --config experiments/v6/protegi/configs/protegi_formal_unconstrained_muse.yaml --output-dir experiments/v6/results/protegi_optimization/entity_protegi_unconstrained

# 7. 完成 Dev 比较并冻结一个 P_E* 后，使用其同一 prompt_scope 配置构建独立实体缓存
# （此处路径待选择实验臂后填写；不得在两个实验臂之间混用缓存）

# 8. 缓存门禁通过后，沿用所选实验臂配置运行 Stage 2（不存在 Gold 实体回退）
```
