# ProTeGi 小规模提示范围对比

- 配对标识：`protegi-prompt-scope-pilot-v1`
- 范围：固定 pilot Train/Dev，仅用于流程与正向信号判断，不是正式 Test 结论。

| 实验臂 | Precision | Recall | F1 | 最终提示长度 | Task 并发上限 | Task 调用 | Muse 调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| constrained | 0.8182 | 0.8727 | 0.8446 | 4199 | 8 | 196 | 8 |
| unconstrained | 0.8375 | 0.8121 | 0.8246 | 6039 | 8 | 221 | 9 |

## 分类型 F1

| 实体类型 | constrained | unconstrained |
|---|---:|---:|
| AttackTechnique | 0.9583 | 0.9388 |
| Configuration | 0.2963 | 0.2667 |
| Vulnerability | 0.9551 | 0.9123 |
| Weakness | 0.7692 | 0.7273 |

## 预注册晋级判断

- 状态：`provisional_complete`
- 暂定选择：`constrained`
- 依据：有约束臂 Dev F1 更高；按预注册规则选择 constrained

> 本报告不读取 Test，也不证明统计显著性。无约束提示的运行接口有效不等于第三章边界一致。
