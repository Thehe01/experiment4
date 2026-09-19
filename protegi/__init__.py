"""面向漏洞知识抽取任务的 ProTeGi (Prompt Optimization with Textual Gradients and Beam Search) 适配包。

包含独立的两阶段提示词自动优化流程：
- Stage 1: 实体抽取优化 (目标: Strict Entity Micro-F1)
- Stage 2: 关系抽取优化 (目标: Strict Relation Micro-F1, 输入固定为冻结 P_E* 预测)
"""

__version__ = "1.3.0"
