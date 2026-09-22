"""ProTeGi 提示词优化搜索核心引擎 (ProTeGi Optimizer).

支持两阶段独立搜索与五种对照实验方法：
- initial: 零优化对照基线 (仅评估 P0)
- mc: 蒙特卡洛释义消融基线 (无梯度批评，仅做语义保持变体搜索)
- greedy_protegi: 贪心消融基线 (Beam=1)
- protegi: Full ProTeGi (主方法: Beam=4, UCB 自适应臂分配)
- protegi_uniform: 均匀预算分配消融基线 (Beam=4, Uniform 评估分配)
"""

from __future__ import annotations

import json
import random
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm_methods import (
    build_text_windows,
    merge_window_predictions,
    make_extractor,
)
from schema import EXTRACTION_ENTITY_TYPES, EXTRACTION_RELATION_TYPES
from protegi.contract_validator import PromptContractValidator
from protegi.document_context import (
    extract_explicit_abbreviation_pairs,
    format_abbreviation_context,
)
from protegi.entity_cache import EntityCacheManager, compute_prompt_hash
from protegi.evaluator import TaskEvaluator, focus_entity_error_examples
from protegi.runtime_contract import (
    SELECTION_WINDOW_OWNERSHIP,
    validate_task_runtime,
)
from protegi.output_expansion_guard import OutputExpansionGuard
from protegi.output_expansion_guard import REASON as EXPANSION_REASON
from protegi.search_stability import (
    INVALID_BUDGET_EXHAUSTED,
    INVALID_OUTPUT_AMPLIFICATION,
    EVALUATION_STATUS_VALID,
    CandidateBudgetExhaustedError,
    CandidateEvalCache,
    RoundSelectionAborted,
    WindowBudgetExhaustedError,
    classify_search_error,
    eval_cache_key,
    implementation_hashes,
    load_search_checkpoint,
    new_stability_counters,
    rng_state_from_json,
    rng_state_to_json,
    sample_ids_hash,
    save_search_checkpoint,
    validate_checkpoint_bindings,
)
from protegi.lineage import PromptLineageTracker
from protegi.logging_utils import ProTeGiLogger
from protegi.metrics import (
    aggregate_micro_f1,
    calc_strict_entity_sample_counts,
    calc_strict_relation_sample_counts,
)
from protegi.models import (
    CallStats,
    ErrorExample,
    EvaluationResult,
    PromptCandidate,
    PromptGradient,
)
from protegi.mutators import (
    GradientGenerator,
    MonteCarloParaphraser,
    PromptEditor,
    deduplicate_and_sample_successors,
)
from protegi.prompts_p0 import ENTITY_PROMPT_P0, RELATION_PROMPT_P0
from protegi.selectors import (
    UCBPromptSelector,
    UniformPromptSelector,
    selection_counts_from_result,
)


def load_split_doc_ids(split_file: Path) -> Dict[str, List[str]]:
    """加载切分清单中的 doc_id 列表。"""
    data = json.loads(Path(split_file).read_text(encoding="utf-8"))
    return {
        "train": data.get("train", []),
        "dev": data.get("dev", []),
        "test": data.get("test", []),
    }


def _window_evaluation_regions(windows: List[dict], text_length: int) -> List[tuple[int, int]]:
    """把重叠窗口划成互斥评价区，避免同一 mention 被重复计分。"""
    if not windows:
        return []
    boundaries = [0]
    for previous, current in zip(windows, windows[1:]):
        overlap_left = int(current["start"])
        overlap_right = int(previous["end"])
        boundary = (
            (overlap_left + overlap_right) // 2
            if overlap_left < overlap_right
            else overlap_left
        )
        boundaries.append(boundary)
    boundaries.append(text_length)
    return [
        (
            max(int(window["start"]), boundaries[index]),
            min(int(window["end"]), boundaries[index + 1]),
        )
        for index, window in enumerate(windows)
    ]


def _span_owned_by_region(
    start: int, end: int, region_start: int, region_end: int
) -> bool:
    """按 span 中点把 mention 唯一分配给一个互斥评价区。"""
    midpoint_twice = start + end
    return 2 * region_start <= midpoint_twice < 2 * region_end


def prepare_stage1_window_samples(
    doc_ids: List[str],
    gold_dir: Path,
    max_chars: int = 3000,
    overlap: int = 400,
    max_docs: Optional[int] = None,
    include_document_abbreviations: bool = False,
    snap_sentence_boundary: bool = False,
    dense_run_split: Optional[bool] = None,
    dense_min_ids: Optional[int] = None,
    dense_min_span: Optional[int] = None,
    dense_gap: Optional[int] = None,
    dense_max_ids: Optional[int] = None,
    dense_seam: Optional[int] = None,
) -> List[dict]:
    """将 Gold 文档按字符窗口切分，并映射局部实体真值标注。

    dense_* 为 None 时沿用环境默认（默认关闭，window-split-v1 行为）。
    """
    samples: List[dict] = []
    target_doc_ids = doc_ids[:max_docs] if max_docs else doc_ids

    allowed_types = set(EXTRACTION_ENTITY_TYPES)

    for doc_id in target_doc_ids:
        doc_path = Path(gold_dir) / f"{doc_id}.json"
        if not doc_path.is_file():
            continue
        doc_data = json.loads(doc_path.read_text(encoding="utf-8"))
        full_text = doc_data.get("text", "")
        abbreviation_pairs = (
            extract_explicit_abbreviation_pairs(full_text)
            if include_document_abbreviations
            else []
        )
        abbreviation_context = (
            format_abbreviation_context(abbreviation_pairs)
            if include_document_abbreviations
            else None
        )
        gold_entities = [
            e for e in doc_data.get("entities", [])
            if e.get("type") in allowed_types
        ]

        windows = build_text_windows(
            full_text,
            max_chars=max_chars,
            overlap=overlap,
            snap_sentence_boundary=snap_sentence_boundary,
            dense_run_split=dense_run_split,
            dense_min_ids=dense_min_ids,
            dense_min_span=dense_min_span,
            dense_gap=dense_gap,
            dense_max_ids=dense_max_ids,
            dense_seam=dense_seam,
        )
        evaluation_regions = _window_evaluation_regions(windows, len(full_text))
        for w_idx, win in enumerate(windows):
            w_start, w_end = win["start"], win["end"]
            w_text = win["text"]
            evaluation_start, evaluation_end = evaluation_regions[w_idx]

            # 过滤落在当前窗口内的实体并转换为窗口相对偏移
            local_entities = []
            for e in gold_entities:
                e_start = e.get("start")
                e_end = e.get("end")
                if e_start is not None and e_end is not None:
                    if (
                        e_start >= w_start
                        and e_end <= w_end
                        and _span_owned_by_region(
                            e_start,
                            e_end,
                            evaluation_start,
                            evaluation_end,
                        )
                    ):
                        rel_e = dict(e)
                        rel_e["start"] = e_start - w_start
                        rel_e["end"] = e_end - w_start
                        local_entities.append(rel_e)

            sample = {
                "sample_id": f"{doc_id}_w{w_idx}",
                "doc_id": doc_id,
                "window_index": w_idx,
                "window_start": w_start,
                "window_end": w_end,
                "evaluation_start": evaluation_start - w_start,
                "evaluation_end": evaluation_end - w_start,
                "text": w_text,
                "gold_entities": local_entities,
            }
            if include_document_abbreviations:
                sample["document_abbreviations"] = abbreviation_context
                sample["document_abbreviation_pairs"] = abbreviation_pairs
            samples.append(sample)

    return samples


def prepare_stage2_window_samples(
    doc_ids: List[str],
    gold_dir: Path,
    fixed_entity_predictions: Optional[Dict[str, List[dict]]] = None,
    max_chars: int = 3000,
    overlap: int = 400,
    max_docs: Optional[int] = None,
    include_document_abbreviations: bool = False,
    snap_sentence_boundary: bool = False,
    dense_run_split: Optional[bool] = None,
    dense_min_ids: Optional[int] = None,
    dense_min_span: Optional[int] = None,
    dense_gap: Optional[int] = None,
    dense_max_ids: Optional[int] = None,
    dense_seam: Optional[int] = None,
) -> List[dict]:
    """为 Stage 2 准备带有固定实体输入的窗口样本。

    fixed_entity_predictions: {sample_id: pred_entities} 或 {doc_id: pred_entities}

    dense_* 为 None 时沿用环境默认（默认关闭，window-split-v1 行为）。
    """
    samples: List[dict] = []
    target_doc_ids = doc_ids[:max_docs] if max_docs else doc_ids

    allowed_ent_types = set(EXTRACTION_ENTITY_TYPES)
    allowed_rel_types = set(EXTRACTION_RELATION_TYPES)

    for doc_id in target_doc_ids:
        doc_path = Path(gold_dir) / f"{doc_id}.json"
        if not doc_path.is_file():
            continue
        doc_data = json.loads(doc_path.read_text(encoding="utf-8"))
        full_text = doc_data.get("text", "")
        abbreviation_pairs = (
            extract_explicit_abbreviation_pairs(full_text)
            if include_document_abbreviations
            else []
        )
        abbreviation_context = (
            format_abbreviation_context(abbreviation_pairs)
            if include_document_abbreviations
            else None
        )
        gold_entities = [
            e for e in doc_data.get("entities", [])
            if e.get("type") in allowed_ent_types
        ]
        gold_relations = [
            r for r in doc_data.get("relations", [])
            if r.get("type") in allowed_rel_types
        ]

        windows = build_text_windows(
            full_text,
            max_chars=max_chars,
            overlap=overlap,
            snap_sentence_boundary=snap_sentence_boundary,
            dense_run_split=dense_run_split,
            dense_min_ids=dense_min_ids,
            dense_min_span=dense_min_span,
            dense_gap=dense_gap,
            dense_max_ids=dense_max_ids,
            dense_seam=dense_seam,
        )
        evaluation_regions = _window_evaluation_regions(windows, len(full_text))
        entity_by_id = {entity["id"]: entity for entity in gold_entities}
        for w_idx, (win, evaluation_region) in enumerate(
            zip(windows, evaluation_regions)
        ):
            w_start, w_end = win["start"], win["end"]
            evaluation_start, evaluation_end = evaluation_region
            w_text = win["text"]
            sample_id = f"{doc_id}_w{w_idx}"

            # 对应局部金标实体与关系
            local_gold_ents = []
            local_ent_ids = set()
            for e in gold_entities:
                e_start, e_end = e.get("start"), e.get("end")
                if e_start is not None and e_end is not None:
                    if e_start >= w_start and e_end <= w_end:
                        rel_e = dict(e)
                        rel_e["start"] = e_start - w_start
                        rel_e["end"] = e_end - w_start
                        local_gold_ents.append(rel_e)
                        local_ent_ids.add(e["id"])

            local_gold_rels = []
            for relation in gold_relations:
                if (
                    relation.get("head") not in local_ent_ids
                    or relation.get("tail") not in local_ent_ids
                ):
                    continue
                head = entity_by_id.get(relation.get("head"))
                tail = entity_by_id.get(relation.get("tail"))
                if head is None or tail is None:
                    continue
                # 关系归属仅由端点 mention 决定；这与预测端可观测的
                # 信息一致，也避免较长 Gold 证据跨过窗口 seam 时丢标签。
                ownership_start = min(head["start"], tail["start"])
                ownership_end = max(head["end"], tail["end"])
                if not _span_owned_by_region(
                    ownership_start,
                    ownership_end,
                    evaluation_start,
                    evaluation_end,
                ):
                    continue
                local_relation = dict(relation)
                evidence_start = relation.get("evidence_start")
                evidence_end = relation.get("evidence_end")
                if isinstance(evidence_start, int) and isinstance(evidence_end, int):
                    local_evidence_start = max(evidence_start, w_start)
                    local_evidence_end = min(evidence_end, w_end)
                    local_relation["evidence_start"] = local_evidence_start - w_start
                    local_relation["evidence_end"] = local_evidence_end - w_start
                    local_relation["evidence"] = full_text[
                        local_evidence_start:local_evidence_end
                    ]
                    if (
                        local_evidence_start != evidence_start
                        or local_evidence_end != evidence_end
                    ):
                        local_relation["evidence_window_truncated"] = True
                        local_relation["original_evidence_start"] = evidence_start
                        local_relation["original_evidence_end"] = evidence_end
                local_gold_rels.append(local_relation)

            # 固定的前序实体输入。禁止静默使用 Gold 实体冒充 Stage 1 预测。
            fixed_ents = (
                fixed_entity_predictions.get(sample_id, [])
                if fixed_entity_predictions
                else []
            )

            sample = {
                "sample_id": sample_id,
                "doc_id": doc_id,
                "window_index": w_idx,
                "window_start": w_start,
                "window_end": w_end,
                "evaluation_start": evaluation_start - w_start,
                "evaluation_end": evaluation_end - w_start,
                "text": w_text,
                "fixed_entities": fixed_ents,
                "gold_entities": local_gold_ents,
                "gold_relations": local_gold_rels,
            }
            if include_document_abbreviations:
                sample["document_abbreviations"] = abbreviation_context
                sample["document_abbreviation_pairs"] = abbreviation_pairs
            samples.append(sample)

    return samples


def select_final_candidate(
    candidates: List[PromptCandidate],
    dev_results_by_id: Dict[str, EvaluationResult],
    *,
    p0_candidate_id: str,
    objective_entity_type: Optional[str] = None,
    guardrail_entity_types: Optional[List[str]] = None,
    guardrail_max_f1_drop: float = 0.0,
) -> Tuple[PromptCandidate, Dict[str, dict]]:
    """Apply a deterministic, preregistered Dev selection policy.

    Strict per-type metrics are the only objective and guardrail inputs.
    Auxiliary overlap diagnostics remain visible in ``EvaluationResult`` but
    are intentionally ignored here.
    """
    if not candidates:
        raise ValueError("final candidate list is empty")
    if p0_candidate_id not in dev_results_by_id:
        raise ValueError("P0 Dev evaluation is required for guardrails")
    if guardrail_max_f1_drop < 0:
        raise ValueError("guardrail_max_f1_drop must be non-negative")

    guardrail_entity_types = list(guardrail_entity_types or [])
    p0_result = dev_results_by_id[p0_candidate_id]
    p0_by_type = p0_result.details.get("by_type", {})
    audits: Dict[str, dict] = {}

    for candidate in candidates:
        result = dev_results_by_id[candidate.candidate_id]
        by_type = result.details.get("by_type", {})
        objective_payload = by_type.get(objective_entity_type, {}) if objective_entity_type else {}
        objective_f1 = (
            float(objective_payload.get("f1") or 0.0)
            if objective_entity_type
            else result.f1
        )
        guardrails = {}
        eligible = True
        for entity_type in guardrail_entity_types:
            baseline_payload = p0_by_type.get(entity_type, {})
            candidate_payload = by_type.get(entity_type, {})
            baseline_f1 = baseline_payload.get("f1")
            candidate_f1 = candidate_payload.get("f1")
            applicable = bool(baseline_payload.get("applicable"))
            passed = (
                not applicable
                or (
                    candidate_f1 is not None
                    and float(candidate_f1)
                    >= float(baseline_f1 or 0.0) - guardrail_max_f1_drop
                )
            )
            guardrails[entity_type] = {
                "applicable": applicable,
                "p0_strict_f1": baseline_f1,
                "candidate_strict_f1": candidate_f1,
                "max_allowed_drop": guardrail_max_f1_drop,
                "passed": passed,
            }
            eligible = eligible and passed
        audits[candidate.candidate_id] = {
            "eligible": eligible,
            "objective": (
                f"strict_{objective_entity_type}_f1"
                if objective_entity_type
                else "strict_micro_f1"
            ),
            "objective_f1": objective_f1,
            "overall_strict_f1": result.f1,
            "overall_strict_precision": result.precision,
            "guardrails": guardrails,
            "overlap_metrics_used_for_selection": False,
        }

    eligible_candidates = [
        candidate
        for candidate in candidates
        if audits[candidate.candidate_id]["eligible"]
    ]
    if not eligible_candidates:
        raise RuntimeError("no final candidate satisfies the preregistered guardrails")

    def sort_key(candidate: PromptCandidate):
        audit = audits[candidate.candidate_id]
        return (
            audit["objective_f1"],
            audit["overall_strict_f1"],
            audit["overall_strict_precision"],
            -len(candidate.prompt_text),
            candidate.candidate_id,
        )

    return max(eligible_candidates, key=sort_key), audits


class ProTeGiOptimizer:
    """ProTeGi 两阶段、双提示搜索范围的自动优化控制器。"""

    def __init__(
        self,
        stage: str,  # "entity" or "relation"
        method: str,  # "initial", "mc", "greedy_protegi", "protegi", "protegi_uniform"
        config: dict,
        output_dir: Path,
        gold_dir: Path,
        split_file: Path,
        entity_cache_dir: Optional[Path] = None,
        task_client=None,
        optimizer_client=None,
        freeze_bindings: Optional[dict] = None,
        resume: bool = False,
        resume_bindings: Optional[dict] = None,
    ):
        self.stage = stage.lower()
        self.method = method.lower()
        self.config = config
        if "prompt_scope" not in config:
            raise ValueError(
                "配置文件必须显式声明 prompt_scope: constrained 或 unconstrained"
            )
        self.prompt_scope = str(config["prompt_scope"]).strip().lower()
        if self.prompt_scope not in {"constrained", "unconstrained"}:
            raise ValueError(
                f"未知 prompt_scope: {self.prompt_scope!r}；必须为 constrained 或 unconstrained"
            )
        self.experiment_pair_id = config.get("experiment_pair_id")
        self.output_dir = Path(output_dir)
        self.gold_dir = Path(gold_dir)
        self.split_file = Path(split_file)
        self.entity_cache_dir = Path(entity_cache_dir) if entity_cache_dir else output_dir / "entity_cache"
        self.freeze_bindings = dict(freeze_bindings or {})
        # 运行稳定性状态（计数器随检查点持久化，随 resume 恢复）。
        self.stability = new_stability_counters()
        self._last_synced_eval_transient = 0
        self._last_synced_eval_budget = 0
        self._invalid_candidate_ids: set[str] = set()
        self._resumed_checkpoint: Optional[dict] = None
        self._restored_rng_state = None
        if resume:
            self._init_from_checkpoint(resume_bindings)
        else:
            self._preexisting_output_entries = (
                sorted(path.name for path in self.output_dir.iterdir())
                if self.output_dir.exists()
                else []
            )
            if self._preexisting_output_entries:
                raise RuntimeError(
                    "输出目录在本次启动前非空。为保证可审计性，请指定新的空 "
                    f"--output-dir；检测到: {self._preexisting_output_entries[:8]}"
                )

        # 校验合法性
        valid_methods = {"initial", "mc", "greedy_protegi", "protegi", "protegi_uniform"}
        if self.method not in valid_methods:
            raise ValueError(f"未知方法: {self.method}，必须在 {valid_methods} 之中")

        self.seed = int(config.get("seed", 42))
        self.beam_width = int(config.get("beam_width", 4 if self.method not in {"initial", "greedy_protegi"} else 1))
        if self.method == "greedy_protegi":
            self.beam_width = 1

        self.optimization_steps = int(config.get("optimization_steps", 6 if self.method != "initial" else 0))
        self.minibatch_size = int(config.get("minibatch_size", 64))
        self.eval_batch_size = int(config.get("eval_batch_size", 8))
        self.total_pull_budget_per_round = int(
            config.get("total_pull_budget_per_round", config.get("total_eval_budget", 64))
        )
        self.errors_per_group = int(config.get("errors_per_group", 4))
        self.gradients_per_error_group = int(config.get("gradients_per_error_group", 4))
        self.max_error_groups = int(config.get("max_error_groups", 1))
        self.edits_per_gradient = int(config.get("edits_per_gradient", 1))
        self.paraphrases_per_edit = int(config.get("paraphrases_per_edit", 2))
        self.successors_per_parent = int(config.get("successors_per_parent", 8))
        self.min_pulls_per_candidate = int(config.get("min_pulls_per_candidate", 1))
        self.ucb_c = float(config.get("ucb_c", 2.0))
        self.document_abbreviation_context = bool(
            config.get("document_abbreviation_context", False)
        )
        self.selection_entity_type = config.get("selection_entity_type")
        self.error_focus_entity_type = config.get("error_focus_entity_type")
        self.final_selection_entity_type = config.get("final_selection_entity_type")
        self.include_p0_in_final_selection = bool(
            config.get("include_p0_in_final_selection", False)
        )
        self.guardrail_entity_types = list(
            config.get("guardrail_entity_types", []) or []
        )
        self.guardrail_max_f1_drop = float(
            config.get("guardrail_max_f1_drop", 0.0)
        )
        if self.stage != "entity" and any((
            self.selection_entity_type,
            self.error_focus_entity_type,
            self.final_selection_entity_type,
            self.guardrail_entity_types,
        )):
            raise ValueError("实体类型定向配置只能用于 entity 阶段")
        for config_name, entity_type in (
            ("selection_entity_type", self.selection_entity_type),
            ("error_focus_entity_type", self.error_focus_entity_type),
            ("final_selection_entity_type", self.final_selection_entity_type),
        ):
            if entity_type and entity_type not in EXTRACTION_ENTITY_TYPES:
                raise ValueError(f"{config_name} 不是合法实体类型: {entity_type}")
        invalid_guardrails = sorted(
            set(self.guardrail_entity_types) - set(EXTRACTION_ENTITY_TYPES)
        )
        if invalid_guardrails:
            raise ValueError(f"guardrail_entity_types 含非法类型: {invalid_guardrails}")
        if self.guardrail_max_f1_drop < 0:
            raise ValueError("guardrail_max_f1_drop 必须为非负数")

        # 统计与日志器
        self.call_stats = CallStats()
        self._evaluated_candidate_ids: set[str] = set()
        self.logger = ProTeGiLogger(output_dir=self.output_dir, stage=self.stage, method=self.method)
        self.lineage_tracker = PromptLineageTracker()

        # 模型客户端
        if task_client is None:
            required_task_keys = {
                "task_model",
                "task_max_workers",
                "task_temperature",
                "task_thinking",
                "task_reasoning_effort",
                "task_top_p",
                "task_max_tokens",
            }
            missing_task_keys = sorted(
                key for key in required_task_keys
                if key not in config or config.get(key) is None
            )
            if missing_task_keys:
                raise ValueError(f"任务模型配置不完整: {missing_task_keys}")
        task_max_tokens = int(config["task_max_tokens"]) if config.get("task_max_tokens") else None
        task_top_p = float(config["task_top_p"]) if config.get("task_top_p") is not None else None
        self.evaluator = TaskEvaluator(
            task_client=task_client,
            max_workers=int(config.get("task_max_workers", 8)),
            task_model=config.get("task_model"),
            task_temperature=float(config.get("task_temperature", 0.0)),
            task_thinking=config.get("task_thinking", "disabled"),
            task_max_tokens=task_max_tokens,
            task_top_p=task_top_p,
            task_reasoning_effort=config.get("task_reasoning_effort", "none"),
            vulnerability_anchored_backfill=bool(config.get("vulnerability_anchored_backfill", False)),
        )
        opt_max_tokens = int(config["optimizer_max_tokens"]) if config.get("optimizer_max_tokens") else None
        opt_top_p = float(config["optimizer_top_p"]) if config.get("optimizer_top_p") else None
        opt_reasoning = config.get("optimizer_reasoning_effort")
        if optimizer_client is None:
            required_optimizer_keys = {
                "optimizer_model",
                "optimizer_temperature",
                "optimizer_thinking",
                "optimizer_reasoning_effort",
                "optimizer_top_p",
                "optimizer_max_tokens",
            }
            missing_optimizer_keys = sorted(
                key for key in required_optimizer_keys
                if key not in config or config.get(key) is None
            )
            if missing_optimizer_keys:
                raise ValueError(f"优化模型配置不完整: {missing_optimizer_keys}")
        self.opt_client = optimizer_client or make_extractor(
            model=config.get("optimizer_model"),
            temperature=float(config.get("optimizer_temperature", 0.1)),
            thinking=config.get("optimizer_thinking", "disabled"),
            max_tokens=opt_max_tokens,
            top_p=opt_top_p,
            reasoning_effort=opt_reasoning,
        )

        # 突变器
        self.gradient_generator = GradientGenerator(
            self.opt_client,
            self.call_stats,
            prompt_scope=self.prompt_scope,
        )
        self.prompt_editor = PromptEditor(
            self.opt_client,
            self.call_stats,
            prompt_scope=self.prompt_scope,
        )
        self.paraphraser = MonteCarloParaphraser(
            self.opt_client,
            self.call_stats,
            prompt_scope=self.prompt_scope,
        )

        # 选择器 (统一使用 total_pull_budget_per_round 保证总预算严格公平)
        if self.method == "protegi_uniform":
            self.selector = UniformPromptSelector(
                total_pull_budget_per_round=self.total_pull_budget_per_round,
                batch_size=self.eval_batch_size,
                min_pulls_per_candidate=self.min_pulls_per_candidate,
                objective_entity_type=self.selection_entity_type,
            )
        else:
            self.selector = UCBPromptSelector(
                c=self.ucb_c,
                total_pull_budget_per_round=self.total_pull_budget_per_round,
                batch_size=self.eval_batch_size,
                min_pulls_per_candidate=self.min_pulls_per_candidate,
                objective_entity_type=self.selection_entity_type,
            )

        self.cache_manager = EntityCacheManager(self.entity_cache_dir)
        self.eval_cache = CandidateEvalCache(self.output_dir)
        if self._resumed_checkpoint is not None:
            # 应用检查点恢复：call_stats/lineage/曲线修剪（logger 与
            # lineage_tracker 在上文已按 fresh 创建，此处覆盖为恢复值）。
            for field, value in self._restored_call_stats.items():
                setattr(self.call_stats, field, int(value))
            self.lineage_tracker.nodes = dict(
                self._restored_lineage.get("nodes") or {}
            )
            self.lineage_tracker.edges = list(
                self._restored_lineage.get("edges") or []
            )
            self.logger.prune_rounds_from(self._resume_next_round)

    # ---------- 运行稳定性：检查点 / 恢复 / 评估缓存 ----------

    def _task_runtime_for_key(self) -> dict:
        """评估缓存 key 用的 task runtime：优先 recorded effective 值。"""
        eff = self.config.get("effective_task_runtime")
        if isinstance(eff, dict):
            return eff
        return {
            "_config_snapshot": hashlib.sha256(
                json.dumps(
                    {k: v for k, v in self.config.items() if not str(k).startswith("_")},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest(),
        }

    def _eval_cache_key_for(
        self,
        candidate: PromptCandidate,
        batch_samples: List[dict],
        collect_errors: bool,
        capture_predictions: bool,
    ) -> str:
        sample_ids = [
            sample.get("sample_id") or sample.get("id") or "unknown"
            for sample in batch_samples
        ]
        return eval_cache_key(
            stage=self.stage,
            prompt_hash=compute_prompt_hash(candidate.prompt_text),
            sample_ids=sample_ids,
            task_runtime=self._task_runtime_for_key(),
            split_sha256=self.freeze_bindings.get("split_sha256"),
            gold_aggregate_sha256=self.freeze_bindings.get("gold_aggregate_sha256"),
            freeze_manifest_sha256=self.freeze_bindings.get(
                "freeze_manifest_sha256"
            ),
            collect_errors=collect_errors,
            capture_predictions=capture_predictions,
        )

    def _sync_evaluator_counters(self) -> None:
        """同步 evaluator 累计计数到稳定性计数器与 call_stats（增量）。

        防御性语义：evaluator 计数器永不回退；若观测到负增量（例如
        resume 后水位陈旧或 evaluator 被复用/重置），只把水位重置到
        当前值并记 failure_log，绝不把负数累加进 stability/call_stats。
        """
        transient_delta = (
            self.evaluator.transient_api_retries - self._last_synced_eval_transient
        )
        budget_delta = (
            self.evaluator.budget_exhaustion_retries - self._last_synced_eval_budget
        )
        if transient_delta < 0:
            self._record_failure(
                where="counter_sync_negative_transient",
                reason=(
                    f"evaluator transient went backwards "
                    f"(current={self.evaluator.transient_api_retries}, "
                    f"watermark={self._last_synced_eval_transient}); "
                    "watermark reset, delta dropped"
                ),
            )
            self._last_synced_eval_transient = self.evaluator.transient_api_retries
            transient_delta = 0
        if budget_delta < 0:
            self._record_failure(
                where="counter_sync_negative_budget",
                reason=(
                    f"evaluator budget went backwards "
                    f"(current={self.evaluator.budget_exhaustion_retries}, "
                    f"watermark={self._last_synced_eval_budget}); "
                    "watermark reset, delta dropped"
                ),
            )
            self._last_synced_eval_budget = self.evaluator.budget_exhaustion_retries
            budget_delta = 0
        if transient_delta:
            self._last_synced_eval_transient = self.evaluator.transient_api_retries
            self.stability["transient_api_retries"] += transient_delta
            self.call_stats.transient_api_retries += transient_delta
        if budget_delta:
            self._last_synced_eval_budget = self.evaluator.budget_exhaustion_retries
            self.stability["budget_exhaustion_retries"] += budget_delta
            self.call_stats.budget_exhaustion_retries += budget_delta

    def _record_failure(self, *, where: str, reason: str, round_idx: Any = None) -> None:
        entry = {"where": where, "reason": str(reason)[:300]}
        if round_idx is not None:
            entry["round_idx"] = round_idx
        self.stability["failure_log"].append(entry)

    def _mark_candidate_invalid(
        self,
        candidate: PromptCandidate,
        error: CandidateBudgetExhaustedError,
        *,
        round_idx: Any = None,
    ) -> None:
        """将预算耗尽候选隔离：记 INVALID，不写任何 F1，不进 UCB/beam/winner。"""
        candidate.selection_status = INVALID_BUDGET_EXHAUSTED
        candidate.estimated_reward = -1.0
        candidate.ucb_score = -1.0
        candidate.metrics["evaluation_status"] = INVALID_BUDGET_EXHAUSTED
        candidate.metrics["failure_reason"] = error.failure_reason
        for sample_id in error.sample_ids:
            if sample_id not in self.stability["budget_exhausted_samples"]:
                self.stability["budget_exhausted_samples"].append(sample_id)
        if candidate.candidate_id not in self._invalid_candidate_ids:
            self._invalid_candidate_ids.add(candidate.candidate_id)
            self.stability["candidates_invalid_budget_exhausted"] += 1
        self._record_failure(
            where="candidate_invalid_budget_exhausted",
            reason=f"{candidate.candidate_id}: {error.failure_reason}",
            round_idx=round_idx,
        )
        self.lineage_tracker.register_candidate(candidate)

    def _fail_p0_budget_exhausted(
        self,
        candidate: PromptCandidate,
        error: CandidateBudgetExhaustedError,
        *,
        round_idx: Any = None,
    ) -> "NoReturn":
        """P0 任一路径持续预算耗尽：记 INVALID 留痕后整个 run hard fail。

        P0 是全 run 的 viability 证明：init / round parent 评估 /
        UCB 选择评估 / Dev 任一路径持续超限，都说明 frozen runtime 下
        P0 不可用。此时禁止把 P0 当普通候选淘汰后继续搜索（继续出的
        任何 winner 都失去基线意义），必须 loud hard fail。
        """
        self._mark_candidate_invalid(candidate, error, round_idx=round_idx)
        raise RuntimeError(
            f"P0 种子候选持续输出预算耗尽，整个 run hard fail "
            f"(round={round_idx})：{error.failure_reason}"
        ) from error

    def _is_p0_candidate(
        self,
        candidate: PromptCandidate,
        p0_candidate: Optional[PromptCandidate],
    ) -> bool:
        """按 candidate_id 认定 P0 身份（恢复前后 ID 稳定为 P_E0/P_R0）。"""
        return (
            p0_candidate is not None
            and candidate.candidate_id == p0_candidate.candidate_id
        )

    def _admit_or_reject_output_amplification(
        self,
        candidate: PromptCandidate,
        *,
        round_idx: Any = None,
    ) -> bool:
        """Output Expansion Guard 准入：拒绝则记 INVALID，不调用 Task Model。

        调用方必须在现有 PromptContractValidator 通过之后调用；返回 False
        时该候选已隔离（不进 UCB/beam/winner），调用方直接 continue。
        """
        result = OutputExpansionGuard.validate(
            candidate.prompt_text, stage=self.stage
        )
        if result:
            return True
        candidate.selection_status = INVALID_OUTPUT_AMPLIFICATION
        candidate.estimated_reward = -1.0
        candidate.ucb_score = -1.0
        candidate.metrics["evaluation_status"] = INVALID_OUTPUT_AMPLIFICATION
        candidate.metrics["failure_reason"] = EXPANSION_REASON
        candidate.metrics["expansion_guard_reasons"] = list(result.reasons)
        if candidate.candidate_id not in self._invalid_candidate_ids:
            self._invalid_candidate_ids.add(candidate.candidate_id)
            self.stability["candidates_invalid_output_amplification"] += 1
        self._record_failure(
            where="candidate_invalid_output_amplification",
            reason=f"{candidate.candidate_id}: {result.error_message[:200]}",
            round_idx=round_idx,
        )
        self.lineage_tracker.register_candidate(candidate)
        return False

    def _drop_invalid_from_beam(
        self, beam: List[PromptCandidate]
    ) -> List[PromptCandidate]:
        """candidate 一旦变 INVALID，立即从 active beam 清掉，不靠后过滤兜底。

        checkpoint 里的 beam 是搜索状态的一部分，绝不允许残留已 INVALID
        的 active member；清理后若无有效 incumbent，调用方按既定语义
        明确 hard fail。
        """
        return [
            c for c in beam
            if c.selection_status
            not in (INVALID_BUDGET_EXHAUSTED, INVALID_OUTPUT_AMPLIFICATION)
        ]

    def _checkpoint_bindings(self) -> dict:
        """当前运行的恢复绑定（由 run_protegi 的实时文件/配置构造）。

        implementation 取启动快照（非保存时重算）：运行中代码变更
        不得污染检查点，否则 resume 校验会被漂移后的值蒙混过关。
        """
        startup = getattr(self, "_startup_implementation", None)
        return {
            "stage": self.stage,
            "method": self.method,
            "prompt_scope": self.prompt_scope,
            "experiment_pair_id": self.experiment_pair_id,
            "config_file_sha256": self.config.get("_config_file_sha256"),
            "split_sha256": self.freeze_bindings.get("split_sha256"),
            "gold_aggregate_sha256": self.freeze_bindings.get(
                "gold_aggregate_sha256"
            ),
            "freeze_manifest_sha256": self.freeze_bindings.get(
                "freeze_manifest_sha256"
            ),
            "effective_task_runtime": self.config.get("effective_task_runtime"),
            "implementation": (
                dict(startup) if startup is not None
                else implementation_hashes(ROOT)
            ),
        }

    def _save_checkpoint(
        self,
        *,
        phase: str,
        next_round: int,
        beam: List[PromptCandidate],
        p0_candidate: Optional[PromptCandidate],
        train_samples: List[dict],
        dev_samples: List[dict],
        rng_state: Any = None,
        reason: str = "",
    ) -> Path:
        """保存可恢复检查点（transient 耗尽或每轮结束调用）。"""
        beam = self._drop_invalid_from_beam(beam)
        payload = {
            **self._checkpoint_bindings(),
            "phase": phase,
            "next_round": int(next_round),
            "beam": [c.to_dict() for c in beam],
            "p0_candidate": p0_candidate.to_dict() if p0_candidate else None,
            "p0_done": p0_candidate is not None
            and p0_candidate.metrics.get("evaluation_status", EVALUATION_STATUS_VALID)
            == EVALUATION_STATUS_VALID
            and bool(p0_candidate.metrics.get("train_f1") is not None),
            "lineage": {
                "nodes": self.lineage_tracker.nodes,
                "edges": self.lineage_tracker.edges,
            },
            "call_stats": self.call_stats.to_dict(),
            "evaluated_candidate_ids": sorted(self._evaluated_candidate_ids),
            "invalid_candidate_ids": sorted(self._invalid_candidate_ids),
            "rng_state": rng_state_to_json(rng_state) if rng_state is not None else None,
            "stability": self.stability,
            "train_sample_ids": [
                s.get("sample_id") or s.get("id") or "unknown" for s in train_samples
            ],
            "dev_sample_ids": [
                s.get("sample_id") or s.get("id") or "unknown" for s in dev_samples
            ],
            "train_sample_ids_sha256": sample_ids_hash([
                s.get("sample_id") or s.get("id") or "unknown" for s in train_samples
            ]),
            "dev_sample_ids_sha256": sample_ids_hash([
                s.get("sample_id") or s.get("id") or "unknown" for s in dev_samples
            ]),
            "reason": reason,
        }
        path = save_search_checkpoint(self.output_dir, payload)
        print(f"[ProTeGi Checkpoint] 已保存检查点: {path} (phase={phase}, next_round={next_round})")
        return path

    def _init_from_checkpoint(self, resume_bindings: Optional[dict]) -> None:
        """从检查点恢复搜索状态；绑定不一致则拒绝恢复。"""
        if resume_bindings is None:
            raise ValueError("--resume 需要调用方提供 resume_bindings，拒绝盲恢复")
        checkpoint = load_search_checkpoint(self.output_dir)
        expected = dict(resume_bindings)
        validate_checkpoint_bindings(checkpoint, expected)
        self._resumed_checkpoint = checkpoint
        # 向后兼容：旧检查点缺新计数键时用默认值补齐。
        self.stability = {
            **new_stability_counters(),
            **(checkpoint.get("stability") or {}),
        }
        self.stability["resume_count"] = int(self.stability.get("resume_count", 0)) + 1
        stats = checkpoint.get("call_stats") or {}
        self._restored_call_stats = {
            field: int(stats.get(field, 0))
            for field in (
                "task_model_calls", "optimizer_model_calls", "task_input_tokens",
                "task_output_tokens", "optimizer_input_tokens", "optimizer_output_tokens",
                "num_generated_candidates", "num_evaluated_candidates",
                "num_duplicate_candidates", "num_evaluated_samples",
                "transient_api_retries", "budget_exhaustion_retries",
                "evaluation_cache_hits",
            )
        }
        self._last_synced_eval_transient = 0
        self._last_synced_eval_budget = 0
        # 注意：累计总数保留在 stability/call_stats 恢复值里；同步水位必须
        # 以新 evaluator 当前计数（通常 0）为准，之后只累加本进程增量，
        # 否则恢复后首次同步即产生负增量，污染正式 summary/provenance。
        self._evaluated_candidate_ids = set(
            checkpoint.get("evaluated_candidate_ids") or []
        )
        # INVALID 身份随检查点持久化：已从 beam 清掉的 INVALID 候选不在
        # beam 里，必须靠 stored 列表保留，否则恢复后集合丢失、重复计数。
        # 旧检查点无该键时回退到 beam 派生（兼容 run8 之前产物）。
        stored_invalid = set(checkpoint.get("invalid_candidate_ids") or [])
        beam_invalid = {
            c["candidate_id"]
            for c in checkpoint.get("beam", [])
            if c.get("selection_status")
            in (INVALID_BUDGET_EXHAUSTED, INVALID_OUTPUT_AMPLIFICATION)
        }
        self._invalid_candidate_ids = stored_invalid | beam_invalid
        lineage = checkpoint.get("lineage") or {}
        self._restored_lineage = {
            "nodes": lineage.get("nodes") or {},
            "edges": lineage.get("edges") or [],
        }
        rng_state = checkpoint.get("rng_state")
        self._restored_rng_state = (
            rng_state_from_json(rng_state) if rng_state is not None else None
        )
        self._resume_next_round = int(checkpoint.get("next_round", 1))
        self._resume_phase = str(checkpoint.get("phase", "search"))
        print(
            f"[ProTeGi Resume] 已从检查点恢复 "
            f"(phase={checkpoint.get('phase')}, next_round={checkpoint.get('next_round')}, "
            f"resume_count={self.stability['resume_count']})"
        )

    def _guarded_optimizer_call(
        self,
        fn,
        *,
        phase: str,
        next_round: int,
        beam: List[PromptCandidate],
        p0_candidate: Optional[PromptCandidate],
        round_idx: Any,
        train_samples: List[dict],
        dev_samples: List[dict],
        rng: Any,
    ):
        """Optimizer 侧模型调用守卫：transient 落检查点后退出，budget 硬失败。

        ValueError/assert/程序 bug 直接 hard fail，不写检查点。
        """
        try:
            return fn()
        except Exception as exc:
            self._sync_evaluator_counters()
            kind = classify_search_error(exc)
            if kind == "transient":
                self._record_failure(
                    where=f"transient_api_exhausted_{phase}",
                    reason=f"{type(exc).__name__}: {exc}",
                    round_idx=round_idx,
                )
                self._save_checkpoint(
                    phase=phase, next_round=next_round, beam=beam,
                    p0_candidate=p0_candidate, train_samples=train_samples,
                    dev_samples=dev_samples, rng_state=rng.getstate(),
                    reason=f"{type(exc).__name__}: {exc}",
                )
                raise exc
            if kind == "budget":
                raise RuntimeError(
                    f"optimizer 模型输出预算耗尽，无法生成候选，run hard fail：{exc}"
                ) from exc
            raise

    def _validate_candidate_for_arm(self, prompt_text: str):
        """按当前实验臂执行准入校验。"""
        return PromptContractValidator.validate_candidate(
            self.stage,
            prompt_text,
            prompt_scope=self.prompt_scope,
        )

    def _attach_prompt_scope_audit(self, candidate: PromptCandidate) -> None:
        """同时记录运行接口与冻结契约状态，避免把二者混为一谈。"""
        runtime_result = PromptContractValidator.validate_candidate(
            self.stage,
            candidate.prompt_text,
            prompt_scope="unconstrained",
        )
        frozen_result = PromptContractValidator.validate_candidate(
            self.stage,
            candidate.prompt_text,
            prompt_scope="constrained",
        )
        candidate.metrics["prompt_scope_audit"] = {
            "configured_scope": self.prompt_scope,
            "runtime_interface_valid": bool(runtime_result),
            "runtime_interface_reasons": runtime_result.reasons,
            "frozen_contract_exact_match": bool(frozen_result),
            "frozen_contract_reasons": frozen_result.reasons,
        }

    def _get_p0_text(self) -> str:
        """获取当前阶段的种子 P0 提示词。"""
        if self.stage == "entity":
            return ENTITY_PROMPT_P0
        elif self.stage == "relation":
            return RELATION_PROMPT_P0
        else:
            raise ValueError(f"未知阶段: {self.stage}")

    def _evaluate_candidate_batch(
        self,
        candidate: PromptCandidate,
        batch_samples: List[dict],
        collect_errors: bool = True,
        capture_predictions: bool = False,
    ) -> Tuple[EvaluationResult, List[ErrorExample]]:
        """在样本批次上评估单个候选。

        稳定性语义：
        - 成功评估先查 eval cache，命中则零 task 调用复用；
        - WindowBudgetExhausted → 转为 CandidateBudgetExhausted（调用方隔离）；
        - transient 错误直接向上传播（调用方保存检查点后退出）；
        - 其余错误直接 hard fail，禁止伪造 F1。
        """
        from protegi.models import ErrorExample as _ErrorExample

        cache_key = self._eval_cache_key_for(
            candidate, batch_samples, collect_errors, capture_predictions
        )
        cached = self.eval_cache.lookup(cache_key)
        if cached is not None:
            self.stability["evaluation_cache_hits"] += 1
            self.call_stats.evaluation_cache_hits += 1
            self.call_stats.num_evaluated_samples += len(batch_samples)
            stored_eval = cached["evaluation"]
            result = EvaluationResult(
                tp=int(stored_eval["tp"]),
                fp=int(stored_eval["fp"]),
                fn=int(stored_eval["fn"]),
                precision=float(stored_eval["precision"]),
                recall=float(stored_eval["recall"]),
                f1=float(stored_eval["f1"]),
                details=dict(stored_eval.get("details", {})),
            )
            errors = [
                _ErrorExample(
                    sample_id=item["sample_id"],
                    input_text=item["input_text"],
                    gold_output=item["gold_output"],
                    predicted_output=item["predicted_output"],
                    error_details=item.get("error_details", {}),
                )
                for item in cached.get("errors", [])
            ]
            return result, errors

        before_calls = self.evaluator.call_count
        before_input_tokens = self.evaluator.input_tokens_est
        before_output_tokens = self.evaluator.output_tokens_est
        try:
            if self.stage == "entity":
                res, errs = self.evaluator.evaluate_stage1_batch(
                    samples=batch_samples,
                    full_entity_prompt=candidate.prompt_text,
                    collect_errors=collect_errors,
                    capture_predictions=capture_predictions,
                )
            else:
                res, errs = self.evaluator.evaluate_stage2_batch(
                    samples=batch_samples,
                    full_relation_prompt=candidate.prompt_text,
                    collect_errors=collect_errors,
                    capture_predictions=capture_predictions,
                )
        except WindowBudgetExhaustedError as exc:
            self._sync_evaluator_counters()
            raise CandidateBudgetExhaustedError(
                candidate_id=candidate.candidate_id,
                stage=self.stage,
                prompt_hash=compute_prompt_hash(candidate.prompt_text),
                sample_ids=[
                    s.get("sample_id") or s.get("id") or "unknown"
                    for s in batch_samples
                ],
                failure_reason=(
                    f"budget exhausted on sample {exc.sample_id} "
                    f"(max_tokens={exc.max_tokens}, retries_used={exc.retries_used})"
                ),
            ) from exc
        except Exception as exc:
            self._sync_evaluator_counters()
            kind = classify_search_error(exc)
            if kind == "budget":
                raise CandidateBudgetExhaustedError(
                    candidate_id=candidate.candidate_id,
                    stage=self.stage,
                    prompt_hash=compute_prompt_hash(candidate.prompt_text),
                    sample_ids=[
                        s.get("sample_id") or s.get("id") or "unknown"
                        for s in batch_samples
                    ],
                    failure_reason=f"budget exhausted: {type(exc).__name__}: {exc}",
                ) from exc
            raise
        self._sync_evaluator_counters()

        self.call_stats.task_model_calls += self.evaluator.call_count - before_calls
        self.call_stats.task_input_tokens += self.evaluator.input_tokens_est - before_input_tokens
        self.call_stats.task_output_tokens += self.evaluator.output_tokens_est - before_output_tokens
        self.call_stats.num_evaluated_samples += len(batch_samples)
        res.details["sample_ids"] = [
            sample.get("sample_id") or sample.get("id") or "unknown"
            for sample in batch_samples
        ]
        self.eval_cache.store(
            cache_key, res.to_dict(), [e.to_dict() for e in errs]
        )
        return res, errs

    def _evaluate_candidate_on_dev(
        self,
        candidate: PromptCandidate,
        dev_samples: List[dict],
    ) -> EvaluationResult:
        """在完整或选定的 Dev 集上评估单个候选。"""
        res, _ = self._evaluate_candidate_batch(
            candidate,
            dev_samples,
            collect_errors=False,
            capture_predictions=True,
        )
        return res

    def run_optimization(
        self,
        train_samples: List[dict],
        dev_samples: List[dict],
    ) -> PromptCandidate:
        """运行完整的 ProTeGi 提示词搜索循环。"""
        start_time_utc = datetime.now(timezone.utc).isoformat()
        rng = random.Random(self.seed)
        if not train_samples:
            raise ValueError("Train 样本为空，禁止启动 ProTeGi")
        if not dev_samples:
            raise ValueError("Dev 样本为空，禁止启动最终候选决选")
        # 启动时实现快照：结束时比对，运行中代码变更即 provenance 受损。
        self._startup_implementation = implementation_hashes(ROOT)
        if self.document_abbreviation_context:
            missing_context = [
                sample.get("sample_id") or sample.get("id") or "unknown"
                for sample in list(train_samples) + list(dev_samples)
                if "document_abbreviations" not in sample
            ]
            if missing_context:
                raise ValueError(
                    "document_abbreviation_context=true 但样本缺少固定上下文: "
                    f"{missing_context[:5]}"
                )

        def _save_transient_checkpoint(
            exc: Exception,
            *,
            phase: str,
            next_round: int,
            beam: List[PromptCandidate],
            p0_candidate: Optional[PromptCandidate],
            round_idx: Any = None,
        ) -> None:
            """Transient 重试耗尽：落检查点后由调用方退出（禁止静默继续）。"""
            self._sync_evaluator_counters()
            self._record_failure(
                where=f"transient_api_exhausted_{phase}",
                reason=f"{type(exc).__name__}: {exc}",
                round_idx=round_idx,
            )
            self._save_checkpoint(
                phase=phase,
                next_round=next_round,
                beam=beam,
                p0_candidate=p0_candidate,
                train_samples=train_samples,
                dev_samples=dev_samples,
                rng_state=rng.getstate(),
                reason=f"{type(exc).__name__}: {exc}",
            )

        def _fail_transient(
            exc: Exception,
            *,
            phase: str,
            next_round: int,
            beam: List[PromptCandidate],
            p0_candidate: Optional[PromptCandidate],
            round_idx: Any = None,
        ) -> "NoReturn":
            _save_transient_checkpoint(
                exc, phase=phase, next_round=next_round, beam=beam,
                p0_candidate=p0_candidate, round_idx=round_idx,
            )
            raise exc

        # Resume 样本一致性校验（与检查点记录的样本集合比对）。
        resumed = self._resumed_checkpoint is not None
        if resumed:
            ckpt = self._resumed_checkpoint
            live_train_ids = [
                s.get("sample_id") or s.get("id") or "unknown" for s in train_samples
            ]
            live_dev_ids = [
                s.get("sample_id") or s.get("id") or "unknown" for s in dev_samples
            ]
            if sample_ids_hash(live_train_ids) != ckpt.get("train_sample_ids_sha256"):
                raise ValueError("恢复的 train 样本集合与检查点不一致，拒绝恢复")
            if sample_ids_hash(live_dev_ids) != ckpt.get("dev_sample_ids_sha256"):
                raise ValueError("恢复的 dev 样本集合与检查点不一致，拒绝恢复")
            if self._restored_rng_state is not None:
                rng.setstate(self._restored_rng_state)
            beam = [PromptCandidate.from_dict(c) for c in ckpt.get("beam", [])]
            # 纵深清理：旧检查点可能残留 INVALID active member，加载时清掉；
            # INVALID 身份取 stored 并集（_init 已恢复），不因 beam 已清而丢失。
            beam = self._drop_invalid_from_beam(beam)
            self._invalid_candidate_ids = set(self._invalid_candidate_ids) | {
                c["candidate_id"]
                for c in ckpt.get("beam", [])
                if c.get("selection_status")
                in (INVALID_BUDGET_EXHAUSTED, INVALID_OUTPUT_AMPLIFICATION)
            } | set(ckpt.get("invalid_candidate_ids") or [])
            p0_restored = ckpt.get("p0_candidate")
            p0_candidate = (
                PromptCandidate.from_dict(p0_restored) if p0_restored else None
            )
            for candidate in beam:
                self.lineage_tracker.register_candidate(candidate)
            if p0_candidate is not None:
                self.lineage_tracker.register_candidate(p0_candidate)
            resume_phase = self._resume_phase
            start_round = int(self._resume_next_round)
            p0_done = start_round >= 1 or resume_phase == "dev"
        else:
            beam = []
            p0_candidate = None
            resume_phase = "search"
            start_round = 1
            p0_done = False

        if not p0_done:
            # 1. 种子初始化与结构契约检查
            p0_text = self._get_p0_text()
            p0_frozen_val = PromptContractValidator.validate_candidate(
                self.stage,
                p0_text,
                prompt_scope="constrained",
            )
            if not p0_frozen_val:
                raise ValueError(
                    f"{self.stage} 种子提示词 P0 未通过冻结契约校验: "
                    f"{p0_frozen_val.error_message}"
                )
            # P0 身份按 frozen hash 认定（非 round/类型）；冻结 P0 自带
            # 输出膨胀语义属于代码/契约错误，直接 HARD FAIL，不得标 invalid。
            frozen_p0 = (
                ENTITY_PROMPT_P0 if self.stage == "entity" else RELATION_PROMPT_P0
            )
            if compute_prompt_hash(p0_text) != compute_prompt_hash(frozen_p0):
                raise ValueError(
                    f"{self.stage} 种子提示词 P0 与 frozen P0 hash 不一致，"
                    "拒绝启动（P0 身份按 hash 认定）"
                )
            p0_expansion = OutputExpansionGuard.validate(
                p0_text, stage=self.stage
            )
            if not p0_expansion:
                raise RuntimeError(
                    f"冻结 P0 自身触发输出膨胀守卫，属代码/契约错误，run hard fail："
                    f"{p0_expansion.error_message}"
                )

            p0_candidate = PromptCandidate(
                candidate_id=f"P_{self.stage[0].upper()}0",
                prompt_text=p0_text,
                parent_id=None,
                generation_type="initial",
                round_idx=0,
            )

            self.lineage_tracker.register_candidate(p0_candidate)

            # 未实现输入/配置/随机状态完整绑定前，禁止静默续跑并混合两次实验。
            # 初始训练集评估使用固定种子的随机共享批次，不采用文件顺序前缀。
            p0_init_pool = list(train_samples)
            random.Random(self.seed).shuffle(p0_init_pool)
            p0_init_batch = p0_init_pool[: self.eval_batch_size]
            try:
                p0_train_eval, _ = self._evaluate_candidate_batch(
                    p0_candidate,
                    p0_init_batch,
                    collect_errors=False,
                )
            except CandidateBudgetExhaustedError as exc:
                self._sync_evaluator_counters()
                self._mark_candidate_invalid(p0_candidate, exc, round_idx=0)
                raise RuntimeError(
                    f"P0 种子候选持续输出预算耗尽，整个 run hard fail：{exc.failure_reason}"
                ) from exc
            except Exception as exc:
                self._sync_evaluator_counters()
                if classify_search_error(exc) == "transient":
                    _fail_transient(
                        exc, phase="search", next_round=0, beam=[],
                        p0_candidate=p0_candidate, round_idx=0,
                    )
                raise
            self._evaluated_candidate_ids.add(p0_candidate.candidate_id)
            self.call_stats.num_evaluated_candidates = len(self._evaluated_candidate_ids)
            p0_selection_counts = selection_counts_from_result(
                p0_train_eval, self.selection_entity_type
            )
            p0_selection_eval = aggregate_micro_f1(*p0_selection_counts)
            p0_candidate.tp, p0_candidate.fp, p0_candidate.fn = p0_selection_counts
            p0_candidate.estimated_reward = p0_selection_eval.f1
            p0_candidate.selection_status = "selected"
            p0_candidate.metrics = {
                "evaluation_status": EVALUATION_STATUS_VALID,
                "train_f1": p0_train_eval.f1,
                "train_precision": p0_train_eval.precision,
                "train_recall": p0_train_eval.recall,
            "train_selection_objective": (
                f"strict_{self.selection_entity_type}_f1"
                if self.selection_entity_type
                else "strict_micro_f1"
            ),
            "train_selection_f1": p0_selection_eval.f1,
            "train_sample_ids": p0_train_eval.details.get("sample_ids", []),
        }
            self._attach_prompt_scope_audit(p0_candidate)

            beam = [p0_candidate]
            self.lineage_tracker.register_candidate(p0_candidate)
            self.logger.log_round(
                round_idx=0,
                beam=beam,
                candidates=[p0_candidate],
                generated_candidates=[p0_candidate],
                gradients=[],
                call_stats=self.call_stats,
                stability=dict(self.stability),
            )
            self._save_checkpoint(
                phase="search", next_round=1, beam=beam,
                p0_candidate=p0_candidate,
                train_samples=train_samples, dev_samples=dev_samples,
                rng_state=rng.getstate(),
                reason="round 0 (P0 init) completed",
            )
            start_round = 1

        # 若方法为 initial，则对 P0 进行 Dev 评估并输出
        if self.method == "initial" or self.optimization_steps <= 0:
            try:
                dev_eval_p0 = self._evaluate_candidate_on_dev(p0_candidate, dev_samples)
            except CandidateBudgetExhaustedError as exc:
                self._sync_evaluator_counters()
                self._mark_candidate_invalid(p0_candidate, exc, round_idx="dev")
                raise RuntimeError(
                    f"P0 在 Dev 评估中持续输出预算耗尽，整个 run hard fail：{exc.failure_reason}"
                ) from exc
            except Exception as exc:
                self._sync_evaluator_counters()
                if classify_search_error(exc) == "transient":
                    _fail_transient(
                        exc, phase="dev", next_round=self.optimization_steps + 1,
                        beam=beam, p0_candidate=p0_candidate, round_idx="dev",
                    )
                raise
            p0_candidate.metrics["dev_f1"] = dev_eval_p0.f1
            p0_candidate.metrics["dev_precision"] = dev_eval_p0.precision
            p0_candidate.metrics["dev_recall"] = dev_eval_p0.recall
            p0_candidate.metrics["dev_by_type"] = dev_eval_p0.details.get("by_type", {})
            if "normalization" in dev_eval_p0.details:
                p0_candidate.metrics["dev_normalization"] = dev_eval_p0.details["normalization"]
            p0_candidate.selection_status = "final_winner"
            self.lineage_tracker.register_candidate(p0_candidate)
            self.logger.save_json(
                {
                    "stage": self.stage,
                    "prompt_scope": self.prompt_scope,
                    "experiment_pair_id": self.experiment_pair_id,
                    "selection_scope": "frozen_dev_once_after_search",
                    "winner_candidate_id": p0_candidate.candidate_id,
                    "candidates": [{
                        "candidate_id": p0_candidate.candidate_id,
                        "prompt_sha256": compute_prompt_hash(p0_candidate.prompt_text),
                        "prompt_scope_audit": p0_candidate.metrics.get(
                            "prompt_scope_audit", {}
                        ),
                        "selected": True,
                        "evaluation": dev_eval_p0.to_dict(),
                    }],
                },
                "final_dev_evaluations.json",
            )
            self.logger.save_json(
                [p0_candidate.to_dict()],
                "final_beam_dev.json",
            )
            final_prompt_name = f"final_{self.stage}_prompt.txt"
            self.logger.save_final_prompt(p0_candidate.prompt_text, filename=final_prompt_name)
            self._save_final_artifacts(
                p0_candidate,
                start_time_utc,
                p0_metrics=p0_candidate.metrics,
            )
            return p0_candidate

        # 2. 搜索迭代循环 (从 start_round 到 Round T)
        for r_idx in range(start_round, self.optimization_steps + 1):
            round_candidates: List[PromptCandidate] = []
            round_generated: List[PromptCandidate] = []
            round_gradients: List[PromptGradient] = []
            round_error_examples: List[dict] = []
            round_gradient_minibatches: List[dict] = []
            selector_history: List[dict] = []
            round_rng_state = rng.getstate()

            def _guarded(fn):
                return self._guarded_optimizer_call(
                    fn, phase="search", next_round=r_idx, beam=beam,
                    p0_candidate=p0_candidate, round_idx=r_idx,
                    train_samples=train_samples, dev_samples=dev_samples, rng=rng,
                )

            # (1) 候选扩展生成阶段 (Expansion)
            for parent_idx, parent in enumerate(beam):
                generated_before_parent = len(round_generated)
                # 抽取错误样本 minibatch (严格来自 train_samples)
                minibatch = rng.sample(train_samples, min(self.minibatch_size, len(train_samples)))
                try:
                    _, errors = self._evaluate_candidate_batch(parent, minibatch, collect_errors=True)
                except CandidateBudgetExhaustedError as exc:
                    self._sync_evaluator_counters()
                    if self._is_p0_candidate(parent, p0_candidate):
                        self._fail_p0_budget_exhausted(
                            parent, exc, round_idx=r_idx
                        )
                    self._mark_candidate_invalid(parent, exc, round_idx=r_idx)
                    beam = self._drop_invalid_from_beam(beam)
                    continue
                except Exception as exc:
                    self._sync_evaluator_counters()
                    if classify_search_error(exc) == "transient":
                        self._record_failure(
                            where="transient_api_exhausted_search",
                            reason=f"{type(exc).__name__}: {exc}",
                            round_idx=r_idx,
                        )
                        self._save_checkpoint(
                            phase="search", next_round=r_idx, beam=beam,
                            p0_candidate=p0_candidate,
                            train_samples=train_samples, dev_samples=dev_samples,
                            rng_state=round_rng_state,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    raise
                if self.error_focus_entity_type:
                    errors = focus_entity_error_examples(
                        errors, self.error_focus_entity_type
                    )
                round_gradient_minibatches.append({
                    "parent_candidate_id": parent.candidate_id,
                    "sample_ids": [
                        sample.get("sample_id") or sample.get("id") or "unknown"
                        for sample in minibatch
                    ],
                    "error_sample_ids": [error.sample_id for error in errors],
                    "error_focus_entity_type": self.error_focus_entity_type,
                })
                for error in errors:
                    record = error.to_dict()
                    record["parent_candidate_id"] = parent.candidate_id
                    round_error_examples.append(record)

                parent_successors: List[PromptCandidate] = []

                if self.method == "mc":
                    # MC 基线：无梯度生成，仅做释义扩展
                    paras = _guarded(
                        lambda: self.paraphraser.paraphrase_prompt(
                            base_candidate=parent,
                            num_paraphrases=self.successors_per_parent,
                            id_prefix=f"c_r{r_idx}_p{parent_idx}_mc",
                        )
                    )
                    for para in paras:
                        round_generated.append(para)
                        para_val = self._validate_candidate_for_arm(para.prompt_text)
                        self._attach_prompt_scope_audit(para)
                        if not para_val:
                            para.selection_status = "invalid_contract"
                            para.metrics["rejection_reason"] = para_val.error_message
                            self.lineage_tracker.register_candidate(para)
                            continue
                        if not self._admit_or_reject_output_amplification(
                            para, round_idx=r_idx
                        ):
                            continue
                        parent_successors.append(para)
                else:
                    # ProTeGi 模式：梯度批评 -> 针对性重写 -> 释义扩展
                    # 生成文本梯度 (max_error_groups=1 确保单父代调用严格受控)
                    grads = _guarded(
                        lambda: self.gradient_generator.generate_gradients(
                            parent_candidate=parent,
                            errors=errors,
                            errors_per_group=self.errors_per_group,
                            gradients_per_error_group=self.gradients_per_error_group,
                            max_error_groups=self.max_error_groups,
                        )
                    )
                    round_gradients.extend(grads)

                    for g_idx, grad in enumerate(grads):
                        # 编辑重写
                        edit_cand = _guarded(
                            lambda: self.prompt_editor.edit_prompt(
                                parent_candidate=parent,
                                gradient=grad,
                                errors=errors,
                                errors_per_group=self.errors_per_group,
                                next_candidate_id=f"c_r{r_idx}_p{parent_idx}_g{g_idx}_edit",
                            )
                        )
                        if edit_cand:
                            round_generated.append(edit_cand)
                            # 结构契约校验 (Contract Validation)
                            contract_val = self._validate_candidate_for_arm(
                                edit_cand.prompt_text
                            )
                            self._attach_prompt_scope_audit(edit_cand)
                            if not contract_val:
                                edit_cand.selection_status = "invalid_contract"
                                edit_cand.metrics["rejection_reason"] = contract_val.error_message
                                self.lineage_tracker.register_candidate(edit_cand, gradient_text=grad.gradient_text)
                                # 不合规候选直接淘汰，不得调用任务模型，也不对其进行释义
                                continue

                            if not self._admit_or_reject_output_amplification(
                                edit_cand, round_idx=r_idx
                            ):
                                # 膨胀候选直接淘汰，不得调用任务模型，也不对其进行释义
                                continue
                            parent_successors.append(edit_cand)
                            # 对合规的编辑版本进行释义扩充
                            paras = _guarded(
                                lambda: self.paraphraser.paraphrase_prompt(
                                    base_candidate=edit_cand,
                                    num_paraphrases=self.paraphrases_per_edit,
                                    id_prefix=f"c_r{r_idx}_p{parent_idx}_g{g_idx}_para",
                                )
                            )
                            for para in paras:
                                round_generated.append(para)
                                para_val = self._validate_candidate_for_arm(
                                    para.prompt_text
                                )
                                self._attach_prompt_scope_audit(para)
                                if not para_val:
                                    para.selection_status = "invalid_contract"
                                    para.metrics["rejection_reason"] = para_val.error_message
                                    self.lineage_tracker.register_candidate(
                                        para,
                                        gradient_text=grad.gradient_text,
                                    )
                                    continue
                                if not self._admit_or_reject_output_amplification(
                                    para, round_idx=r_idx
                                ):
                                    continue
                                parent_successors.append(para)

                # 去重与后继采样控制
                sampled_succs, dup_stats = deduplicate_and_sample_successors(
                    parent_candidate=parent,
                    successors=parent_successors,
                    max_successors=self.successors_per_parent,
                    seed=self.seed + r_idx * 100 + parent_idx,
                    call_stats=None,
                )

                self.call_stats.num_generated_candidates += (
                    len(round_generated) - generated_before_parent
                )
                self.call_stats.num_duplicate_candidates += dup_stats["duplicate_count"]
                sampled_ids = {candidate.candidate_id for candidate in sampled_succs}
                gradient_text_by_id = {
                    gradient.gradient_id: gradient.gradient_text
                    for gradient in round_gradients
                }
                seen_signatures = {"".join(parent.prompt_text.split())}
                for generated in parent_successors:
                    signature = "".join(generated.prompt_text.split())
                    if signature in seen_signatures:
                        generated.selection_status = "duplicate"
                    elif generated.candidate_id not in sampled_ids:
                        generated.selection_status = "not_sampled"
                    seen_signatures.add(signature)
                    self.lineage_tracker.register_candidate(
                        generated,
                        gradient_text=gradient_text_by_id.get(generated.gradient_id),
                    )

                for succ in sampled_succs:
                    succ.round_idx = r_idx
                    succ.selection_status = "pending"
                    round_candidates.append(succ)
                    self.lineage_tracker.register_candidate(
                        succ,
                        gradient_text=gradient_text_by_id.get(succ.gradient_id),
                    )

            # 将上一轮 Beam 也作为候选之一参与本轮竞争 (允许保留优质父代)
            # INVALID 候选（预算耗尽 / 输出膨胀）永不进入 UCB/beam：先过滤。
            all_pool = [
                c for c in list(round_candidates) + list(beam)
                if c.selection_status
                not in (INVALID_BUDGET_EXHAUSTED, INVALID_OUTPUT_AMPLIFICATION)
            ]
            if not all_pool and (round_candidates or beam):
                raise RuntimeError(
                    f"Round {r_idx} 无有效候选可评估（新候选与 incumbent "
                    "全部 INVALID），明确失败，禁止从无效候选中选择。"
                )

            # (2) 候选评估与选择阶段 (Selection / Bandits - 严格在 train_samples 上进行)
            if not all_pool:
                # 容错：若无有效候选生成，保留当前 beam
                selector_history = []
            else:
                # 每轮候选臂状态重新开始，禁止继承上一轮不同样本上的 UCB 统计。
                for candidate in all_pool:
                    candidate.num_evaluations = 0
                    candidate.samples_seen = 0
                    candidate.tp = 0
                    candidate.fp = 0
                    candidate.fn = 0
                    candidate.estimated_reward = 0.0
                    candidate.ucb_score = 0.0
                    candidate.selection_status = "pending"

                # 每轮固定种子洗牌并覆盖完整 Train；candidate-local pull index
                # 映射到同一共享批次，避免候选顺序和文件前缀偏差。
                round_train_samples = list(train_samples)
                random.Random(self.seed + r_idx * 10000).shuffle(round_train_samples)
                eval_batches = [
                    round_train_samples[i : i + self.eval_batch_size]
                    for i in range(0, len(round_train_samples), self.eval_batch_size)
                ]
                if not eval_batches:
                    raise ValueError("Train 样本为空，无法运行候选选择")

                def eval_batch_fn(cand: PromptCandidate, candidate_pull_idx: int) -> EvaluationResult:
                    eval_b = eval_batches[candidate_pull_idx % len(eval_batches)]
                    try:
                        res, _ = self._evaluate_candidate_batch(cand, eval_b, collect_errors=False)
                    except CandidateBudgetExhaustedError as exc:
                        self._sync_evaluator_counters()
                        if self._is_p0_candidate(cand, p0_candidate):
                            self._fail_p0_budget_exhausted(
                                cand, exc, round_idx=r_idx
                            )
                        self._mark_candidate_invalid(cand, exc, round_idx=r_idx)
                        raise RoundSelectionAborted(
                            invalid_candidate_id=cand.candidate_id,
                            round_idx=r_idx,
                        ) from exc
                    return res

                try:
                    selector_history = self.selector.execute_evaluation_budget(all_pool, eval_batch_fn)
                except RoundSelectionAborted as exc:
                    # 中止本轮选择：先清 INVALID，再落检查点归档，保留有效 incumbent。
                    beam = self._drop_invalid_from_beam(beam)
                    self._record_failure(
                        where="round_selection_aborted",
                        reason=str(exc),
                        round_idx=r_idx,
                    )
                    self._save_checkpoint(
                        phase="search", next_round=r_idx, beam=beam,
                        p0_candidate=p0_candidate,
                        train_samples=train_samples, dev_samples=dev_samples,
                        rng_state=round_rng_state,
                        reason=str(exc),
                    )
                    self.logger.log_round(
                        round_idx=r_idx,
                        beam=beam,
                        candidates=all_pool,
                        generated_candidates=round_generated,
                        gradients=round_gradients,
                        selector_history=[],
                        call_stats=self.call_stats,
                        error_examples=round_error_examples,
                        gradient_minibatches=round_gradient_minibatches,
                        stability=dict(self.stability),
                    )
                    continue
                except Exception as exc:
                    self._sync_evaluator_counters()
                    if classify_search_error(exc) == "transient":
                        self._record_failure(
                            where="transient_api_exhausted_search",
                            reason=f"{type(exc).__name__}: {exc}",
                            round_idx=r_idx,
                        )
                        self._save_checkpoint(
                            phase="search", next_round=r_idx, beam=beam,
                            p0_candidate=p0_candidate,
                            train_samples=train_samples, dev_samples=dev_samples,
                            rng_state=round_rng_state,
                            reason=f"{type(exc).__name__}: {exc}",
                        )
                    raise
                self._evaluated_candidate_ids.update(
                    candidate.candidate_id for candidate in all_pool
                    if candidate.selection_status != INVALID_BUDGET_EXHAUSTED
                )
                self.call_stats.num_evaluated_candidates = len(
                    self._evaluated_candidate_ids
                )
                beam = self.selector.rank_and_select_top_k(all_pool, top_k=self.beam_width)
                for candidate in all_pool:
                    self.lineage_tracker.register_candidate(candidate)

            # (3) 中间过程归档 (Dev 集完全隔离，仅记录由 Train UCB/Uniform 估计的奖励)
            self.logger.log_round(
                round_idx=r_idx,
                beam=beam,
                candidates=all_pool,
                generated_candidates=round_generated,
                gradients=round_gradients,
                selector_history=selector_history,
                call_stats=self.call_stats,
                error_examples=round_error_examples,
                gradient_minibatches=round_gradient_minibatches,
                stability=dict(self.stability),
            )
            # 每轮结束落检查点：任何中断都可从下一轮恢复，无需 Round 0 重跑。
            self._save_checkpoint(
                phase="search",
                next_round=r_idx + 1,
                beam=beam,
                p0_candidate=p0_candidate,
                train_samples=train_samples,
                dev_samples=dev_samples,
                rng_state=rng.getstate(),
                reason=f"round {r_idx} completed",
            )

        # 3. 搜索结束，对最终 Beam (B_T) 执行全流程唯一一次 Dev 验证集统一评估决选
        # Dev is used only in the final selection phase after all optimization rounds are completed.
        # Budget 耗尽的候选直接剔除（不评分）；若无有效候选则明确失败。
        # Transient 耗尽则落检查点 (phase=dev) 后退出，可恢复。
        print(f"\n[ProTeGi 搜索结束] 共完成 {self.optimization_steps} 轮 Train 搜索。Dev is used only in the final selection phase after all optimization rounds are completed. 开始对最终 Beam ({len(beam)} 个候选) 执行统一 Dev 评估选优...")
        dev_results_by_id: Dict[str, EvaluationResult] = {}
        for cand in beam:
            if cand.selection_status in (
                INVALID_BUDGET_EXHAUSTED,
                INVALID_OUTPUT_AMPLIFICATION,
            ):
                continue
            try:
                dev_eval = self._evaluate_candidate_on_dev(cand, dev_samples)
            except CandidateBudgetExhaustedError as exc:
                self._sync_evaluator_counters()
                self._mark_candidate_invalid(cand, exc, round_idx="dev")
                continue
            except Exception as exc:
                self._sync_evaluator_counters()
                if classify_search_error(exc) == "transient":
                    self._record_failure(
                        where="transient_api_exhausted_dev",
                        reason=f"{type(exc).__name__}: {exc}",
                        round_idx="dev",
                    )
                    self._save_checkpoint(
                        phase="dev", next_round=self.optimization_steps + 1,
                        beam=beam, p0_candidate=p0_candidate,
                        train_samples=train_samples, dev_samples=dev_samples,
                        rng_state=rng.getstate(),
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                raise
            dev_results_by_id[cand.candidate_id] = dev_eval
            dev_results_by_id[cand.candidate_id] = dev_eval
            cand.metrics["dev_f1"] = dev_eval.f1
            cand.metrics["dev_precision"] = dev_eval.precision
            cand.metrics["dev_recall"] = dev_eval.recall
            cand.metrics["dev_by_type"] = dev_eval.details.get("by_type", {})
            if "normalization" in dev_eval.details:
                cand.metrics["dev_normalization"] = dev_eval.details["normalization"]

        # 单独跑 P0 的 Dev baseline 用于记录初始对比，绝不参与 Full ProTeGi 候选决赛
        dev_eval_p0 = dev_results_by_id.get(p0_candidate.candidate_id)
        if dev_eval_p0 is None:
            try:
                dev_eval_p0 = self._evaluate_candidate_on_dev(p0_candidate, dev_samples)
            except CandidateBudgetExhaustedError as exc:
                self._sync_evaluator_counters()
                self._mark_candidate_invalid(p0_candidate, exc, round_idx="dev")
                raise RuntimeError(
                    f"P0 在 Dev 评估中持续输出预算耗尽，整个 run hard fail：{exc.failure_reason}"
                ) from exc
            except Exception as exc:
                self._sync_evaluator_counters()
                if classify_search_error(exc) == "transient":
                    self._record_failure(
                        where="transient_api_exhausted_dev",
                        reason=f"{type(exc).__name__}: {exc}",
                        round_idx="dev",
                    )
                    self._save_checkpoint(
                        phase="dev", next_round=self.optimization_steps + 1,
                        beam=beam, p0_candidate=p0_candidate,
                        train_samples=train_samples, dev_samples=dev_samples,
                        rng_state=rng.getstate(),
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                raise
            dev_results_by_id[p0_candidate.candidate_id] = dev_eval_p0
        p0_candidate.metrics["dev_f1"] = dev_eval_p0.f1
        p0_candidate.metrics["dev_precision"] = dev_eval_p0.precision
        p0_candidate.metrics["dev_recall"] = dev_eval_p0.recall
        p0_candidate.metrics["dev_by_type"] = dev_eval_p0.details.get("by_type", {})
        if "normalization" in dev_eval_p0.details:
            p0_candidate.metrics["dev_normalization"] = dev_eval_p0.details["normalization"]

        # 默认保留旧实验“P0 仅作 baseline”的行为；定向分支显式允许 P0
        # 参与自动决选，使“没有合格改进”能够成为可审计结果。
        # INVALID 候选永不进入决赛；若无有效候选则明确失败。
        finalists = [
            c for c in beam
            if c.selection_status
            not in (INVALID_BUDGET_EXHAUSTED, INVALID_OUTPUT_AMPLIFICATION)
            and c.candidate_id in dev_results_by_id
        ]
        if self.include_p0_in_final_selection:
            if all(c.candidate_id != p0_candidate.candidate_id for c in finalists):
                if (
                    p0_candidate.selection_status
                    not in (
                        INVALID_BUDGET_EXHAUSTED,
                        INVALID_OUTPUT_AMPLIFICATION,
                    )
                    and p0_candidate.candidate_id in dev_results_by_id
                ):
                    finalists.append(p0_candidate)
        else:
            non_initial = [c for c in finalists if c.generation_type != "initial"]
            finalists = non_initial or finalists
        if not finalists:
            raise RuntimeError(
                "Dev 决选中无有效候选（全部因预算耗尽 INVALID），"
                "明确失败，禁止从无效候选中选择 winner。"
            )

        winner, final_selection_audits = select_final_candidate(
            finalists,
            dev_results_by_id,
            p0_candidate_id=p0_candidate.candidate_id,
            objective_entity_type=self.final_selection_entity_type,
            guardrail_entity_types=self.guardrail_entity_types,
            guardrail_max_f1_drop=self.guardrail_max_f1_drop,
        )
        for candidate in finalists:
            candidate.metrics["final_selection"] = final_selection_audits[
                candidate.candidate_id
            ]
            candidate.selection_status = (
                "final_winner" if candidate.candidate_id == winner.candidate_id
                else "final_not_selected"
            )
            self.lineage_tracker.register_candidate(candidate)
        self.lineage_tracker.register_candidate(p0_candidate)

        dev_records = []
        dev_candidates = list(beam)
        if all(candidate.candidate_id != p0_candidate.candidate_id for candidate in dev_candidates):
            dev_candidates.append(p0_candidate)
        for candidate in dev_candidates:
            # INVALID 候选在 Dev 评估中被跳过，无评估结果：记状态，不 KeyError。
            dev_eval_result = dev_results_by_id.get(candidate.candidate_id)
            dev_records.append({
                "candidate_id": candidate.candidate_id,
                "generation_type": candidate.generation_type,
                "prompt_sha256": compute_prompt_hash(candidate.prompt_text),
                "prompt_scope_audit": candidate.metrics.get("prompt_scope_audit", {}),
                "selected": candidate.candidate_id == winner.candidate_id,
                "is_p0_baseline": candidate.candidate_id == p0_candidate.candidate_id,
                "evaluation_status": candidate.metrics.get(
                    "evaluation_status", EVALUATION_STATUS_VALID
                ),
                "failure_reason": candidate.metrics.get("failure_reason"),
                "selection_audit": final_selection_audits.get(candidate.candidate_id),
                "evaluation": (
                    dev_eval_result.to_dict() if dev_eval_result is not None else None
                ),
            })
        self.logger.save_json(
            {
                "stage": self.stage,
                "prompt_scope": self.prompt_scope,
                "experiment_pair_id": self.experiment_pair_id,
                "selection_scope": "frozen_dev_once_after_all_train_search_rounds",
                "selection_policy": {
                    "objective": (
                        f"strict_{self.final_selection_entity_type}_f1"
                        if self.final_selection_entity_type
                        else "strict_micro_f1"
                    ),
                    "include_p0": self.include_p0_in_final_selection,
                    "guardrail_entity_types": self.guardrail_entity_types,
                    "guardrail_max_f1_drop": self.guardrail_max_f1_drop,
                    "overlap_metrics_used_for_selection": False,
                },
                "winner_candidate_id": winner.candidate_id,
                "candidates": dev_records,
            },
            "final_dev_evaluations.json",
        )
        self.logger.save_json(
            [candidate.to_dict() for candidate in dev_candidates],
            "final_beam_dev.json",
        )
        final_prompt_name = f"final_{self.stage}_prompt.txt"
        self.logger.save_final_prompt(winner.prompt_text, filename=final_prompt_name)
        self._save_final_artifacts(winner, start_time_utc, p0_metrics=p0_candidate.metrics)
        return winner

    def _save_final_artifacts(
        self,
        winner: PromptCandidate,
        start_time_utc: str,
        p0_metrics: Optional[dict] = None,
    ) -> None:
        """导出谱系跟踪图与最终总结摘要。"""
        self.lineage_tracker.register_candidate(winner)
        # 导出 Lineage JSON 与 DOT
        self.lineage_tracker.export_json(
            self.output_dir / "prompt_lineage.json",
            final_candidate_id=winner.candidate_id,
        )
        self.lineage_tracker.export_dot(
            self.output_dir / "prompt_lineage.dot",
            final_candidate_id=winner.candidate_id,
        )

        final_prompt_filename = f"final_{self.stage}_prompt.txt"
        final_prompt_path = self.output_dir / final_prompt_filename

        def public_client_config(client) -> dict:
            return {
                key: value
                for key, value in getattr(client, "config", {}).items()
                if key not in {"api_key", "authorization", "headers"}
            }

        def file_hash(path: Path) -> Optional[str]:
            if not path.is_file():
                return None
            return hashlib.sha256(path.read_bytes()).hexdigest()

        freeze_manifest_path = ROOT / "data" / "dataset_freeze_manifest_v6.json"
        scope_protocol_path = ROOT / "protegi" / "PROMPT_SCOPE_EXPERIMENT.md"

        # 结束漂移检查：运行中实现文件变更即 provenance 受损，直接 hard fail。
        startup_implementation = getattr(self, "_startup_implementation", None)
        end_implementation = implementation_hashes(ROOT)
        drifted_files = sorted(
            rel
            for rel, digest in end_implementation.items()
            if startup_implementation is not None
            and startup_implementation.get(rel) != digest
        )
        if drifted_files:
            raise RuntimeError(
                "检测到运行中实现漂移，provenance 已受损，拒绝封存产物："
                f"{drifted_files}"
            )
        implementation_paths = [
            Path(__file__),
            ROOT / "protegi" / "prompts_p0.py",
            ROOT / "protegi" / "templates.py",
            ROOT / "protegi" / "mutators.py",
            ROOT / "protegi" / "contract_validator.py",
            ROOT / "protegi" / "document_context.py",
            ROOT / "protegi" / "selectors.py",
            ROOT / "protegi" / "evaluator.py",
            ROOT / "protegi" / "entity_cache.py",
            ROOT / "protegi" / "logging_utils.py",
            ROOT / "protegi" / "lineage.py",
        ]

        summary = {
            "stage": self.stage,
            "method": self.method,
            "prompt_scope": self.prompt_scope,
            "experiment_pair_id": self.experiment_pair_id,
            "candidate_admission_policy": (
                "frozen_contract_exact_match"
                if self.prompt_scope == "constrained"
                else "runtime_interface_only"
            ),
            "search_stability": {
                "evaluation_status": "completed",
                "failure_reason": None,
                "budget_exhausted_samples": list(
                    self.stability["budget_exhausted_samples"]
                ),
                "budget_exhaustion_retries": int(
                    self.stability["budget_exhaustion_retries"]
                ),
                "candidates_valid": len(self._evaluated_candidate_ids),
                "candidates_invalid_budget_exhausted": int(
                    self.stability["candidates_invalid_budget_exhausted"]
                ),
                "candidates_invalid_output_amplification": int(
                    self.stability.get(
                        "candidates_invalid_output_amplification", 0
                    )
                ),
                "transient_api_retries": int(
                    self.stability["transient_api_retries"]
                ),
                "evaluation_cache_hits": int(
                    self.stability.get("evaluation_cache_hits", 0)
                ),
                "resume_count": int(self.stability.get("resume_count", 0)),
                "failure_log": list(self.stability.get("failure_log", [])),
            },
            "winner_candidate_id": winner.candidate_id,
            "winner_round": winner.round_idx,
            "winner_metrics": winner.metrics,
            "winner_prompt_sha256": compute_prompt_hash(winner.prompt_text),
            "winner_prompt_sha256_raw_bytes": file_hash(final_prompt_path),
            "winner_prompt_scope_audit": winner.metrics.get(
                "prompt_scope_audit", {}
            ),
            "window_construction": self.config.get("window_construction"),
            "selection_window_ownership": self.config.get(
                "selection_window_ownership", SELECTION_WINDOW_OWNERSHIP
            ),
            "initial_p0_metrics": p0_metrics or {},
            "total_candidates_registered": len(self.lineage_tracker.nodes),
            "call_stats": self.call_stats.to_dict(),
            "config": self.config,
            "runtime": {
                "task_client": public_client_config(self.evaluator.client),
                "task_max_workers": self.evaluator.max_workers,
                "optimizer_client": public_client_config(self.opt_client),
                "token_counts_are_estimates": True,
            },
            "input_bindings": {
                "split_file": str(self.split_file),
                "split_file_sha256_raw_bytes": file_hash(self.split_file),
                "gold_dir": str(self.gold_dir),
                "entity_cache_dir": str(self.entity_cache_dir),
                "entity_cache_train_manifest_sha256_raw_bytes": file_hash(
                    self.entity_cache_dir / "entity_cache_train_manifest.json"
                ),
                "entity_cache_dev_manifest_sha256_raw_bytes": file_hash(
                    self.entity_cache_dir / "entity_cache_dev_manifest.json"
                ),
                "dataset_freeze_manifest": str(freeze_manifest_path),
                "dataset_freeze_manifest_sha256_raw_bytes": file_hash(freeze_manifest_path),
                "prompt_scope_experiment_protocol": str(scope_protocol_path),
                "prompt_scope_experiment_protocol_sha256_raw_bytes": file_hash(
                    scope_protocol_path
                ),
                "config_file": self.config.get("_config_file"),
                "config_file_sha256_raw_bytes": self.config.get("_config_file_sha256"),
                "implementation_sha256_raw_bytes": {
                    str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path)
                    for path in implementation_paths
                },
            },
            "formal_eligible": bool(self.config.get("formal_eligible", True)),
            "implementation_snapshot_at_start": startup_implementation,
            "implementation_drift_detected": False,
            "start_time_utc": start_time_utc,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.logger.save_summary(summary)
        self.logger.create_artifact_manifest(
            final_prompt_filename=final_prompt_filename,
            canonical_prompt_sha256=compute_prompt_hash(winner.prompt_text),
        )
        # 成功完成：删除检查点（恢复仅用于未完成的 run；eval 缓存保留复用）。
        try:
            checkpoint_file = self.output_dir / "search_checkpoint.json"
            if checkpoint_file.is_file():
                checkpoint_file.unlink()
        except OSError:
            pass
