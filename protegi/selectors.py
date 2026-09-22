"""ProTeGi 候选提示词选择器 (Prompt Selectors).

实现两类评估预算分配选择策略：
1. UCBPromptSelector (主方法): 基于 UCB 在固定总预算下自适应分配评估批次；
2. UniformPromptSelector (消融方法): 在相同固定总预算下均匀轮转分配评估批次。

核心公平性准则：
- 单轮总评估预算严格定义为 total_pull_budget_per_round (默认 64 pulls);
- 每次 pull 处理 eval_batch_size (默认 8 windows);
- UCB 与 Uniform 消耗完全相同的 Task Model 评估窗口数 (64 * 8 = 512 windows);
- 每个候选至少获得 ``min_pulls_per_candidate`` 个共享批次；
- 同一 candidate-local pull index 对应同一批样本，消除候选顺序偏差；
- 严禁任何静默截断；最低拉动预算不足时直接显式报错。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from protegi.metrics import aggregate_micro_f1
from protegi.models import EvaluationResult, PromptCandidate


def _actual_sample_count(result: EvaluationResult, fallback: int) -> int:
    """尾批不足 batch_size 时记录真实暴露窗口数。"""
    value = result.details.get("num_samples")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    sample_ids = result.details.get("sample_ids")
    if isinstance(sample_ids, list):
        return len(sample_ids)
    return fallback


def selection_counts_from_result(
    eval_result: EvaluationResult,
    objective_entity_type: Optional[str],
) -> Tuple[int, int, int]:
    """Project an evaluation result to the preregistered search objective."""
    if not objective_entity_type:
        return eval_result.tp, eval_result.fp, eval_result.fn
    payload = eval_result.details.get("by_type", {}).get(objective_entity_type)
    if not isinstance(payload, dict):
        raise ValueError(
            f"评估结果缺少 selection_entity_type={objective_entity_type!r} 的按类型指标"
        )
    return int(payload.get("tp", 0)), int(payload.get("fp", 0)), int(payload.get("fn", 0))


class UCBPromptSelector:
    """基于 UCB (Upper Confidence Bound) 的自适应提示词候选选择器。"""

    def __init__(
        self,
        c: float = 2.0,
        total_pull_budget_per_round: int = 64,
        batch_size: int = 8,
        min_pulls_per_candidate: int = 1,
        objective_entity_type: Optional[str] = None,
    ):
        self.c = c
        self.total_pull_budget_per_round = total_pull_budget_per_round
        self.batch_size = batch_size
        self.min_pulls_per_candidate = min_pulls_per_candidate
        self.objective_entity_type = objective_entity_type

    def update_candidate_with_batch(
        self,
        candidate: PromptCandidate,
        eval_result: EvaluationResult,
        samples_count: int,
    ) -> None:
        """用新评估批次的累加指标更新 candidate 的臂状态。"""
        candidate.num_evaluations += 1
        candidate.samples_seen += samples_count
        tp, fp, fn = selection_counts_from_result(
            eval_result, self.objective_entity_type
        )
        candidate.tp += tp
        candidate.fp += fp
        candidate.fn += fn

        # 重新根据全部已见样本计算真实的 Strict Micro-F1
        overall_metric = aggregate_micro_f1(candidate.tp, candidate.fp, candidate.fn)
        candidate.estimated_reward = overall_metric.f1

    def compute_ucb_scores(self, candidates: List[PromptCandidate], total_t: int) -> None:
        """为所有候选计算 UCB 得分。"""
        for cand in candidates:
            if cand.num_evaluations == 0:
                cand.ucb_score = float("inf")
            else:
                bonus = self.c * math.sqrt(math.log(max(1, total_t)) / cand.num_evaluations)
                cand.ucb_score = cand.estimated_reward + bonus

    def select_next_arm_to_evaluate(self, candidates: List[PromptCandidate], total_t: int) -> PromptCandidate:
        """选择当前 UCB 得分最高的臂进行下一轮采样评估。"""
        self.compute_ucb_scores(candidates, total_t)
        return max(
            candidates,
            key=lambda c: (
                c.ucb_score,
                c.estimated_reward,
                -len(c.prompt_text),
                c.candidate_id,
            ),
        )

    def rank_and_select_top_k(
        self,
        candidates: List[PromptCandidate],
        top_k: int = 4,
    ) -> List[PromptCandidate]:
        """严格按照估计出的 Strict Micro-F1 选择 Top-k 候选晋级。

        确定性平局决断规则:
        1. 更高 estimated_reward (Strict Micro-F1);
        2. 更高 Precision (tp / (tp + fp));
        3. 更短 Prompt 长度;
        4. candidate_id 字典序。
        """
        def sort_key(c: PromptCandidate):
            precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) > 0 else 0.0
            return (c.estimated_reward, precision, -len(c.prompt_text), c.candidate_id)

        sorted_cands = sorted(candidates, key=sort_key, reverse=True)
        for i, c in enumerate(sorted_cands):
            c.selection_status = "selected" if i < top_k else "dropped"
        return sorted_cands[:top_k]

    def execute_evaluation_budget(
        self,
        candidates: List[PromptCandidate],
        eval_batch_fn: Callable[[PromptCandidate, int], EvaluationResult],
    ) -> List[dict]:
        """在严格 total_pull_budget_per_round 约束下执行 UCB 评估循环。

        每 round 评估消耗严格等于 total_pull_budget_per_round：
        1. 每个候选先在相同序号的共享批次上评估至少 M 次；
        2. 剩余预算按 UCB 自适应动态分配。
        """
        k = len(candidates)
        if k == 0:
            return []
        required_initial_pulls = k * self.min_pulls_per_candidate
        if required_initial_pulls > self.total_pull_budget_per_round:
            raise ValueError(
                f"候选数 K={k} × min_pulls_per_candidate={self.min_pulls_per_candidate} "
                f"超过单轮总拉动预算 {self.total_pull_budget_per_round}！"
            )

        history: List[dict] = []
        pulls_executed = 0

        # 阶段 1: 所有候选在 candidate-local index 相同的共享批次上完成最低评估。
        for local_pull_idx in range(self.min_pulls_per_candidate):
            for cand in candidates:
                eval_res = eval_batch_fn(cand, local_pull_idx)
                self.update_candidate_with_batch(
                    cand, eval_res, _actual_sample_count(eval_res, self.batch_size)
                )
                pulls_executed += 1
                history.append({
                    "type": "initial_pull",
                    "pull_index": pulls_executed,
                    "candidate_pull_index": local_pull_idx,
                    "candidate_id": cand.candidate_id,
                    "batch_sample_ids": eval_res.details.get("sample_ids", []),
                    "reward": cand.estimated_reward,
                    "objective_entity_type": self.objective_entity_type,
                })

        # 阶段 2: 剩余预算按 UCB 贪心自适应拉动
        total_t = pulls_executed
        while pulls_executed < self.total_pull_budget_per_round:
            arm = self.select_next_arm_to_evaluate(candidates, total_t)
            local_pull_idx = arm.num_evaluations
            eval_res = eval_batch_fn(arm, local_pull_idx)
            self.update_candidate_with_batch(
                arm, eval_res, _actual_sample_count(eval_res, self.batch_size)
            )
            pulls_executed += 1
            total_t += 1
            history.append({
                "type": "ucb_pull",
                "pull_index": pulls_executed,
                "candidate_pull_index": local_pull_idx,
                "candidate_id": arm.candidate_id,
                "batch_sample_ids": eval_res.details.get("sample_ids", []),
                "ucb_score": arm.ucb_score,
                "reward": arm.estimated_reward,
                "objective_entity_type": self.objective_entity_type,
            })

        assert pulls_executed == self.total_pull_budget_per_round, (
            f"UCB 评估拉动次数 ({pulls_executed}) 不等于设定总预算 ({self.total_pull_budget_per_round})！"
        )
        return history


class UniformPromptSelector:
    """均匀评估预算分配选择器（用于与 UCB 的严格公平消融对照）。

    与 UCB 共享完全相同的 total_pull_budget_per_round 总预算，
    将总 pull 预算尽可能均等地 round-robin 分配给 K 个候选：
    - 每个候选基础 pull 数 = total_pull_budget_per_round // K
    - 余数 pulls 确定性按顺序分配给前 R 个候选
    - 任意两候选之间的 pull 数量差不超过 1。
    """

    def __init__(
        self,
        total_pull_budget_per_round: int = 64,
        batch_size: int = 8,
        min_pulls_per_candidate: int = 1,
        objective_entity_type: Optional[str] = None,
    ):
        self.total_pull_budget_per_round = total_pull_budget_per_round
        self.batch_size = batch_size
        self.min_pulls_per_candidate = min_pulls_per_candidate
        self.objective_entity_type = objective_entity_type

    def update_candidate_with_batch(
        self,
        candidate: PromptCandidate,
        eval_result: EvaluationResult,
        samples_count: int,
    ) -> None:
        """更新 Candidate 的累加评价指标。"""
        candidate.num_evaluations += 1
        candidate.samples_seen += samples_count
        tp, fp, fn = selection_counts_from_result(
            eval_result, self.objective_entity_type
        )
        candidate.tp += tp
        candidate.fp += fp
        candidate.fn += fn

        overall_metric = aggregate_micro_f1(candidate.tp, candidate.fp, candidate.fn)
        candidate.estimated_reward = overall_metric.f1
        candidate.ucb_score = candidate.estimated_reward

    def rank_and_select_top_k(
        self,
        candidates: List[PromptCandidate],
        top_k: int = 4,
    ) -> List[PromptCandidate]:
        """预算均匀耗尽后，严格按照估计出的 Strict Micro-F1 选择 Top-k 候选晋级。"""
        def sort_key(c: PromptCandidate):
            precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) > 0 else 0.0
            return (c.estimated_reward, precision, -len(c.prompt_text), c.candidate_id)

        sorted_cands = sorted(candidates, key=sort_key, reverse=True)
        for i, c in enumerate(sorted_cands):
            c.selection_status = "selected" if i < top_k else "dropped"
        return sorted_cands[:top_k]

    def execute_evaluation_budget(
        self,
        candidates: List[PromptCandidate],
        eval_batch_fn: Callable[[PromptCandidate, int], EvaluationResult],
    ) -> List[dict]:
        """在严格与 UCB 相同的 total_pull_budget_per_round 约束下执行均匀分配评估。"""
        k = len(candidates)
        if k == 0:
            return []
        if k * self.min_pulls_per_candidate > self.total_pull_budget_per_round:
            raise ValueError(
                f"候选数 K={k} × min_pulls_per_candidate={self.min_pulls_per_candidate} "
                f"超过单轮总拉动预算 {self.total_pull_budget_per_round}！"
            )

        history: List[dict] = []
        pulls_executed = 0

        # 确定性轮转 (Round-Robin) 分配剩余 pulls，直至达到 total_pull_budget_per_round
        while pulls_executed < self.total_pull_budget_per_round:
            cand_idx = pulls_executed % k
            cand = candidates[cand_idx]
            local_pull_idx = cand.num_evaluations
            eval_res = eval_batch_fn(cand, local_pull_idx)
            self.update_candidate_with_batch(
                cand, eval_res, _actual_sample_count(eval_res, self.batch_size)
            )
            pulls_executed += 1
            history.append({
                "type": "uniform_pull",
                "pull_index": pulls_executed,
                "candidate_pull_index": local_pull_idx,
                "candidate_id": cand.candidate_id,
                "batch_sample_ids": eval_res.details.get("sample_ids", []),
                "reward": cand.estimated_reward,
                "objective_entity_type": self.objective_entity_type,
            })

        assert pulls_executed == self.total_pull_budget_per_round, (
            f"Uniform 评估拉动次数 ({pulls_executed}) 不等于设定总预算 ({self.total_pull_budget_per_round})！"
        )
        return history
