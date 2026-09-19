# 第三章边界同步复裁清单

契约版本：`chapter3-boundary-sync-v1`
复裁日期：`2026-09-14`
复裁人：`expert_adjudicator`

该清单依据 `chapter3-boundary-sync-v1` 契约完成 118 个去重复裁对象的正式人工复裁与 Gold 安全修订。

## 门禁状态与复裁决议汇总

- 审计文档：105 篇
- 去重后复裁对象：118 个
- 阻断性语义问题修复：64 项全部解决
- 需重新裁决的 exploited_by：58 条全部完成裁决（54 条重新认证通过，3 条利用后序列剔除，1 条语义错配剔除）

## 分类处理统计

| 类别 | 数量 | 裁决决议 | 说明 |
|---|---:|:---:|---|
| `weakness_without_local_explicit_cwe` | 58 | **REJECT** | 描述性弱点短语缺少局部明示 CWE，自正式 Gold 移除并归档至可恢复隔离文件；安全级联删除 47 条 `instantiates` 关系 |
| `configuration_release_bearing_span` | 2 | **REVISE_SPAN** | 裁剪普通发布版本号，表面跨度收敛为最小独立识别产品基名 `Adobe ColdFusion` |
| `explicit_post_exploitation_sequence` | 3 | **REJECT** | 明确出现利用后行为（执行 RAT/C2 或任意代码执行），剔除出正式 Gold |
| `exploited_by_boundary_recertification_required` | 54 | **ACCEPT** | 逐条核验证实为直接利用/入口语义，更新 `adjudication_basis`，写入关系特定的句/表/列表作用域和端点绑定依据 |
| `exploited_by` 语义错配 | 1 | **REJECT** | `aa24-290a/R6` 的 T1202 表项描述相邻的间接命令执行行为，未证明 CVE-2020-1472 直接实例化 T1202；仅保留同文 R5 的 T1068 直接利用关系 |

## 裁决规则合规核查

1. Weakness 只有在同一事实块存在唯一对应的明示 CWE 时保留；其余 58 项描述性短语及其 47 条级联关系已保存至 `data/annotations/quarantine/chapter3-boundary-sync-v1/rejected_items.json`，不进入正式 Gold。
2. Configuration 普通发布版本已全部删除，最小独立识别产品跨度已规范化。
3. `exploited_by` 逐条按行为角色判断：3 条利用后活动和 1 条 T1202 语义错配已剔除，54 条合法直接利用关系已重新认证。
4. 13 条原先只覆盖端点或截断列表项的证据已扩展为最短自足句/表格行/受支配列表块；所有 54 条保留关系的 `adjudication_basis` 均写入契约版本、具体作用域、文档/关系编号及端点绑定。
