"""ProTeGi 核心数据结构与实体模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptCandidate:
    """Prompt 候选对象，记录文本、血统、评估状态与 UCB 指标。"""
    candidate_id: str
    prompt_text: str
    parent_id: Optional[str] = None
    generation_type: str = "initial"  # "initial", "gradient_edit", "paraphrase"
    gradient_id: Optional[str] = None
    round_idx: int = 0
    num_evaluations: int = 0
    samples_seen: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    estimated_reward: float = 0.0
    ucb_score: float = 0.0
    selection_status: str = "pending"  # "selected", "dropped", "pending"
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "generation_type": self.generation_type,
            "gradient_id": self.gradient_id,
            "round_idx": self.round_idx,
            "num_evaluations": self.num_evaluations,
            "samples_seen": self.samples_seen,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "estimated_reward": round(self.estimated_reward, 6),
            "ucb_score": round(self.ucb_score, 6),
            "selection_status": self.selection_status,
            "prompt_text": self.prompt_text,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PromptCandidate:
        return cls(
            candidate_id=data["candidate_id"],
            prompt_text=data["prompt_text"],
            parent_id=data.get("parent_id"),
            generation_type=data.get("generation_type", "initial"),
            gradient_id=data.get("gradient_id"),
            round_idx=data.get("round_idx", 0),
            num_evaluations=data.get("num_evaluations", 0),
            samples_seen=data.get("samples_seen", 0),
            tp=data.get("tp", 0),
            fp=data.get("fp", 0),
            fn=data.get("fn", 0),
            estimated_reward=data.get("estimated_reward", 0.0),
            ucb_score=data.get("ucb_score", 0.0),
            selection_status=data.get("selection_status", "pending"),
            metrics=data.get("metrics", {}),
        )


@dataclass
class PromptGradient:
    """文本梯度对象，由批评模型根据错误样本生成。"""
    gradient_id: str
    parent_prompt_id: str
    error_group_id: str
    gradient_text: str
    round_idx: int

    def to_dict(self) -> dict:
        return {
            "gradient_id": self.gradient_id,
            "parent_prompt_id": self.parent_prompt_id,
            "error_group_id": self.error_group_id,
            "gradient_text": self.gradient_text,
            "round_idx": self.round_idx,
        }


@dataclass
class ErrorExample:
    """错误样本，记录输入文本、标注真值与当前错误预测。"""
    sample_id: str
    input_text: str
    gold_output: Dict[str, Any]
    predicted_output: Dict[str, Any]
    error_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "input_text": self.input_text,
            "gold_output": self.gold_output,
            "predicted_output": self.predicted_output,
            "error_details": self.error_details,
        }


@dataclass
class EvaluationResult:
    """评估指标累加结果。"""
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "details": self.details,
        }


@dataclass
class CallStats:
    """实验运行的 API 与 Token 统计。"""
    task_model_calls: int = 0
    optimizer_model_calls: int = 0
    task_input_tokens: int = 0
    task_output_tokens: int = 0
    optimizer_input_tokens: int = 0
    optimizer_output_tokens: int = 0
    num_generated_candidates: int = 0
    num_evaluated_candidates: int = 0
    num_duplicate_candidates: int = 0
    num_evaluated_samples: int = 0
    # 运行稳定性计数（加性字段，不影响历史语义）。
    transient_api_retries: int = 0
    budget_exhaustion_retries: int = 0
    evaluation_cache_hits: int = 0

    def to_dict(self) -> dict:
        return {
            "task_model_calls": self.task_model_calls,
            "optimizer_model_calls": self.optimizer_model_calls,
            "task_input_tokens": self.task_input_tokens,
            "task_output_tokens": self.task_output_tokens,
            "optimizer_input_tokens": self.optimizer_input_tokens,
            "optimizer_output_tokens": self.optimizer_output_tokens,
            "num_generated_candidates": self.num_generated_candidates,
            "num_evaluated_candidates": self.num_evaluated_candidates,
            "num_duplicate_candidates": self.num_duplicate_candidates,
            "num_evaluated_samples": self.num_evaluated_samples,
            "transient_api_retries": self.transient_api_retries,
            "budget_exhaustion_retries": self.budget_exhaustion_retries,
            "evaluation_cache_hits": self.evaluation_cache_hits,
        }
