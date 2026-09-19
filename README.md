# v6 受控实验包 (Experiment 4)

`v6` 是从 `experiments/v5` 独立分离出的独立实验仓库（Experiment 4）。它使用
`chapter3-no-capec-v1` 抽取模式和 `4.6-mcpu-mention-fact-dual-layer-v1` 标注协议；
以 `chapter3-boundary-sync-v2` 为上位契约。

## 当前状态与门禁机制

- **Gold 标注**：`data/annotations/gold/` 中的 105 篇保留人工复标与分歧裁决来源；406 个 Configuration mention 已完成 MCPU v2 全量审计，修订 15 个跨度和 7 个普通版本 CPE。本次任务严格遵循“零修改 Gold 标注”原则。
- **划分**：`data/train_dev_test_split_v7.json`，63/21/21 篇开发划分。
- **冻结清单**：`data/dataset_freeze_manifest_v6.json`，绑定 Gold 聚合哈希、划分文件哈希及全部审计证据。
- **最终测试门禁（Fail-Closed）**：`controlled_test_rerun_ready` 严格保持 `false`。`scripts/run_v6_experiment.py` 在 `split in {"test", "all"}` 时最先触发 `_assert_run_allowed` 门禁并立即阻断（Predictor 实际调用次数恒为 0）。在人类 IAA 复核完成且新预留未见测试集冻结前，严禁对 test 执行任何推理与评估。
- **旧 APO 产物隔离**：APO (`apo`, `apo_full`) 明确标记为历史遗留基线方法，严禁通过重命名或封装冒充 ProTeGi。`promote_protegi_v6.py` 和 `load_protegi_final_artifact` 设置了严格的反向探查与签名校验。
- **两阶段 ProTeGi 架构**：Stage 1 优化实体提示词 $P_E^*$；使用 $P_E^*$ 固化 Train/Dev 实体预测缓存（深度绑定 Freeze Manifest 和 Split 哈希）；Stage 2 优化关系提示词 $P_R^*$，严格以缓存预测实体作为输入，绝对禁止回退到 Gold entities。

## 目录结构

```text
experiment4/ (v6 repo root)
├─ ANNOTATION_GUIDELINE.md
├─ CHAPTER3_EXPERIMENT_ALIGNMENT.md 第三章驱动的第四章实验执行稿
├─ EXPERIMENT_LOG.md                v6 实验台账
├─ config/runtime_profile.json
├─ data/
│  ├─ annotations/gold/             105 篇冻结 Gold 副本 (100% 保持只读)
│  ├─ sources/                      对应源文本副本
│  ├─ train_dev_test_split_v7.json  v7 划分
│  ├─ dataset_freeze_manifest_v6.json
│  └─ review_status.json
├─ protegi/                         ProTeGi 两阶段优化模块与契约验证器
├─ scripts/
│  ├─ run_v6_experiment.py          端到端运行入口（含 Final Test 门禁）
│  ├─ run_protegi.py                两阶段 ProTeGi 运行入口
│  ├─ promote_protegi_v6.py         正式 ProTeGi 产物晋级门禁（防伪与全哈希校验）
│  ├─ freeze_dataset_manifest_v6.py 冻结清单生成与校验
│  ├─ freeze_baseline_methods_v6.py Rule/P0/Full 基线冻结
│  ├─ paired_bootstrap_v6.py        文档级配对 bootstrap 检验
│  ├─ schema.py / eval_metrics.py
│  ├─ rule_baseline.py / llm_methods.py
│  └─ audit_*.py                    Gold、划分、泄漏与契约门禁
├─ tests/
│  ├─ test_v6_package.py            包边界、冻结清单与最终测试门禁测试
│  └─ test_protegi.py               ProTeGi 逻辑、缓存绑定与晋级准入测试
└─ results/                         v6 正式产物与预测结果
```

## 正式实验标准执行流程 (8 步工作流)

直接在独立仓库根目录执行命令：

```powershell
# 1. 审计与门禁预检 (0 API 调用)
python -X utf8 scripts/audit_gold_strategy.py
python -X utf8 scripts/audit_split_leakage.py

# 2. 冻结清单与单元测试全量验证
python -X utf8 scripts/freeze_dataset_manifest_v6.py --check
python -X utf8 tests/test_v6_package.py
python -X utf8 tests/test_protegi.py
# 或使用 pytest:
pytest -q

# 3. ProTeGi Stage 1: 实体提示词优化 (在 Dev 集搜索最优 P_E*)
# 有约束实验臂 (constrained, 仅优化 guidance):
python -X utf8 scripts/run_protegi.py --stage entity --method protegi --config protegi/configs/protegi_formal_constrained_muse.yaml --output-dir results/protegi_optimization/entity_protegi_constrained
# 或无约束实验臂 (unconstrained, 优化完整语义提示):
python -X utf8 scripts/run_protegi.py --stage entity --method protegi --config protegi/configs/protegi_formal_unconstrained_muse.yaml --output-dir results/protegi_optimization/entity_protegi_unconstrained

# 4. 构建与固化实体预测缓存 (使用胜出的 P_E*，深度绑定冻结清单与当前划分哈希)
python -X utf8 scripts/run_protegi.py --stage entity_cache --method protegi --config protegi/configs/protegi_formal_constrained_muse.yaml --entity-cache-dir results/protegi_optimization/entity_cache_constrained

# 5. ProTeGi Stage 2: 关系提示词优化 (在 Dev 集以冻结实体缓存为输入搜索最优 P_R*，严禁 Gold 回退)
python -X utf8 scripts/run_protegi.py --stage relation --method protegi --config protegi/configs/protegi_formal_constrained_muse.yaml --entity-cache-dir results/protegi_optimization/entity_cache_constrained --output-dir results/protegi_optimization/relation_protegi_constrained

# 6. 正式 ProTeGi 产物晋级 (执行全哈希绑定校验与契约准入，阻断旧 APO 产物与自定义切分)
python -X utf8 scripts/promote_protegi_v6.py `
  --entity-prompt results/protegi_optimization/entity_protegi_constrained/final_entity_prompt.txt `
  --relation-prompt results/protegi_optimization/relation_protegi_constrained/final_relation_prompt.txt `
  --entity-summary results/protegi_optimization/entity_protegi_constrained/summary.json `
  --relation-summary results/protegi_optimization/relation_protegi_constrained/summary.json `
  --entity-cache-train-manifest results/protegi_optimization/entity_cache_constrained/entity_cache_train_manifest.json `
  --entity-cache-dev-manifest results/protegi_optimization/entity_cache_constrained/entity_cache_dev_manifest.json `
  --output results/protegi_final/protegi_final_artifact.json

# 7. 最终测试门禁检查 (当前处于锁定状态，运行将安全报错)
# python -X utf8 scripts/run_v6_experiment.py --method protegi --split test
# [注意] 当前 controlled_test_rerun_ready=false，此命令将严格抛出 RuntimeError 并阻断执行。

# 8. 正式评价与显著性检验 (待人类 IAA 与新未见测试集冻结开启后执行)
# python -X utf8 scripts/paired_bootstrap_v6.py
# python -X utf8 scripts/audit_chapter3_experiment_alignment.py
```

