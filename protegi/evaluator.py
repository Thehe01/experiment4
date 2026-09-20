"""ProTeGi 任务模型评估与推理执行器。

彻底绕开 APO-v2 的 _append_apo_guidance 与 base+guidance 机制，
将 ProTeGi Candidate 中的完整 Prompt 直接注入 Task Model 执行推理。
"""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from llm_methods import (
    _loose_json,
    parse_entity_mentions,
    parse_relations,
    make_extractor,
)
from schema import EXTRACTION_ENTITY_TYPES, EXTRACTION_RELATION_TYPES
from protegi.models import ErrorExample, EvaluationResult
from protegi.metrics import (
    calc_strict_entity_sample_counts,
    calc_same_type_jaccard_overlap_counts,
    calc_strict_relation_sample_counts,
    aggregate_micro_f1,
)
from protegi.document_context import render_document_context
from protegi.entity_backfill import vulnerability_anchored_backfill
from protegi.entity_cache import compute_prompt_hash
from protegi.retry_utils import retry_api_call
from protegi.search_stability import WindowBudgetExhaustedError


def _is_budget_exhausted(exc: Exception) -> bool:
    """判定是否为输出预算耗尽（类型或名称双重判定，避免 import  fragile）。"""
    if type(exc).__name__ == "ModelOutputBudgetExhaustedError":
        return True
    try:
        from llm_extractor import ModelOutputBudgetExhaustedError

        return isinstance(exc, ModelOutputBudgetExhaustedError)
    except Exception:
        return False


def focus_entity_error_examples(
    errors: List[ErrorExample],
    entity_type: str,
) -> List[ErrorExample]:
    """Project Train errors to one entity type and rank them deterministically.

    The projection prevents the critic from optimizing unrelated labels merely
    because they co-occur in the same window.  It uses predictions from the
    current Train minibatch only; Dev predictions never enter gradient input.
    """
    focused: List[ErrorExample] = []
    for error in errors:
        gold_entities = [
            entity
            for entity in error.gold_output.get("entities", [])
            if entity.get("type") == entity_type
        ]
        pred_entities = [
            entity
            for entity in error.predicted_output.get("entities", [])
            if entity.get("type") == entity_type
        ]
        strict_tp, strict_fp, strict_fn = calc_strict_entity_sample_counts(
            pred_entities,
            gold_entities,
            allowed_types={entity_type},
        )
        if strict_fp == 0 and strict_fn == 0:
            continue
        overlap_tp, _, _ = calc_same_type_jaccard_overlap_counts(
            pred_entities,
            gold_entities,
            allowed_types={entity_type},
            threshold=0.5,
        )
        focused.append(ErrorExample(
            sample_id=error.sample_id,
            input_text=error.input_text,
            gold_output={"entities": gold_entities},
            predicted_output={"entities": pred_entities},
            error_details={
                "focus_entity_type": entity_type,
                "strict_tp": strict_tp,
                "strict_fp": strict_fp,
                "strict_fn": strict_fn,
                "boundary_overlap_pairs": max(0, overlap_tp - strict_tp),
                "overlap_threshold": 0.5,
                "source_split": "train",
                "document_abbreviations": error.error_details.get(
                    "document_abbreviations"
                ),
            },
        ))

    return sorted(
        focused,
        key=lambda error: (
            -int(error.error_details["strict_fn"]),
            -int(error.error_details["boundary_overlap_pairs"]),
            -int(error.error_details["strict_fp"]),
            error.sample_id,
        ),
    )


class TaskEvaluator:
    """任务模型评估器，支持 Stage 1 (实体) 与 Stage 2 (关系) 的直接完整 Prompt 评估。"""

    def __init__(
        self,
        task_client=None,
        max_workers: int = 8,
        *,
        task_model: Optional[str] = None,
        task_temperature: float = 0.0,
        task_thinking: str = "disabled",
        task_max_tokens: Optional[int] = None,
        task_top_p: Optional[float] = None,
        task_reasoning_effort: Optional[str] = "none",
        vulnerability_anchored_backfill: bool = False,
    ):
        if task_client is None and not task_model:
            raise ValueError("TaskEvaluator 必须显式指定 task_model，禁止继承环境默认模型")
        self.client = task_client or make_extractor(
            model=task_model,
            temperature=task_temperature,
            thinking=task_thinking,
            max_tokens=task_max_tokens,
            top_p=task_top_p,
            reasoning_effort=task_reasoning_effort,
        )
        self.max_workers = int(max_workers)
        if self.max_workers <= 0:
            raise ValueError("TaskEvaluator max_workers 必须为正整数")
        self.vulnerability_anchored_backfill = bool(vulnerability_anchored_backfill)
        self.call_count = 0
        self.input_tokens_est = 0
        self.output_tokens_est = 0
        # 运行稳定性计数（锁保护，供 optimizer 同步）。
        self.budget_exhaustion_retries = 0
        self.transient_api_retries = 0
        self._stats_lock = threading.Lock()

    def _note_transient_retry(self, exc: Exception, attempt: int) -> None:
        with self._stats_lock:
            self.transient_api_retries += 1

    def _call_task_model_once(
        self,
        *,
        stage: str,
        prompt_content: str,
        system_content: str,
        sample_id: str,
        full_prompt_for_hash: str,
    ) -> str:
        """单次任务模型调用；预算耗尽时在完全相同 runtime 下最多再试 1 次。

        仍超限则抛 WindowBudgetExhaustedError（不伪造空预测）。
        Transient 错误沿用现有 retry_api_call 策略（参数未动）。
        """
        with self._stats_lock:
            self.call_count += 1
            self.input_tokens_est += (len(system_content) + len(prompt_content)) // 4

        def _invoke() -> str:
            return retry_api_call(
                self.client.call_fn,
                prompt=prompt_content,
                system_prompt=system_content,
                config=self.client.config,
                on_retry=self._note_transient_retry,
            )

        try:
            raw_output = _invoke()
        except Exception as exc:
            if not _is_budget_exhausted(exc):
                raise
            with self._stats_lock:
                self.budget_exhaustion_retries += 1
            try:
                raw_output = _invoke()
            except Exception as exc2:
                if not _is_budget_exhausted(exc2):
                    raise
                raise WindowBudgetExhaustedError(
                    stage=stage,
                    prompt_hash=compute_prompt_hash(full_prompt_for_hash),
                    sample_id=sample_id,
                    max_tokens=(self.client.config or {}).get("max_tokens"),
                    retries_used=1,
                ) from exc2
        with self._stats_lock:
            self.output_tokens_est += len(raw_output) // 4
        return raw_output

    def _parallel_map(self, fn, items: List[Any]) -> List[Any]:
        """按输入顺序返回结果；所有任务模型批量路径统一受 max_workers 控制。"""
        if not items:
            return []
        if len(items) == 1:
            return [fn(items[0])]
        worker_count = min(self.max_workers, len(items))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(fn, items))

    def predict_stage1_window(
        self,
        text: str,
        full_entity_prompt: str,
        document_abbreviations: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> List[dict]:
        """使用完整的 Entity Prompt 在单窗口文本上抽取实体。

        绕开任何 base + guidance 拼接，full_entity_prompt 直接作为指令。
        """
        if "{text}" in full_entity_prompt:
            prompt_content = full_entity_prompt.replace("{text}", text)
            system_content = "You are a helpful cybersecurity intelligence annotation expert. Output valid JSON only."
        else:
            prompt_content = f"<text>\n{text}\n</text>\nReturn JSON only."
            system_content = full_entity_prompt

        if document_abbreviations is not None:
            prompt_content = (
                render_document_context(document_abbreviations)
                + "\n\n"
                + prompt_content
            )

        raw_output = self._call_task_model_once(
            stage="entity",
            prompt_content=prompt_content,
            system_content=system_content,
            sample_id=sample_id or "window_unknown",
            full_prompt_for_hash=full_entity_prompt,
        )

        parsed = _loose_json(raw_output)
        raw_entities = parsed.get("entities", [])
        if not isinstance(raw_entities, list):
            raw_entities = []

        entities, _ = parse_entity_mentions(text, raw_entities)
        return [
            {k: v for k, v in entity.items() if not k.startswith("_")}
            for entity in entities
        ]

    def predict_stage1_texts(
        self,
        texts: List[str],
        full_entity_prompt: str,
        document_abbreviations: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
    ) -> List[List[dict]]:
        """使用统一并发上限批量执行 Stage 1 窗口预测并保持输入顺序。"""
        texts = list(texts)
        if sample_ids is not None and len(sample_ids) != len(texts):
            raise ValueError("sample_ids 与 texts 数量不一致")
        if document_abbreviations is not None:
            if len(document_abbreviations) != len(texts):
                raise ValueError("document_abbreviations 与 texts 数量不一致")
            indexed = list(enumerate(zip(texts, document_abbreviations)))
            return self._parallel_map(
                lambda item: self.predict_stage1_window(
                    item[1][0],
                    full_entity_prompt,
                    item[1][1],
                    sample_id=(
                        sample_ids[item[0]] if sample_ids is not None else None
                    ),
                ),
                indexed,
            )
        indexed = list(enumerate(texts))
        return self._parallel_map(
            lambda item: self.predict_stage1_window(
                item[1],
                full_entity_prompt,
                sample_id=(
                    sample_ids[item[0]] if sample_ids is not None else None
                ),
            ),
            indexed,
        )

    def predict_stage2_window(
        self,
        text: str,
        entities: List[dict],
        full_relation_prompt: str,
        sample_id: Optional[str] = None,
    ) -> List[dict]:
        """使用完整的 Relation Prompt 和固定的实体列表在单窗口文本上抽取关系。

        绕开任何 base + guidance 拼接，full_relation_prompt 直接作为指令。
        """
        public_entities = [
            {k: v for k, v in e.items() if not k.startswith("_")}
            for e in entities
        ]
        entities_str = json.dumps(public_entities, ensure_ascii=False)

        if "{text}" in full_relation_prompt and "{entities}" in full_relation_prompt:
            prompt_content = full_relation_prompt.replace("{text}", text).replace("{entities}", entities_str)
            system_content = "You are a helpful cybersecurity relation annotation expert. Output valid JSON only."
        elif "{text}" in full_relation_prompt:
            prompt_content = full_relation_prompt.replace("{text}", text) + f"\n\nEntities:\n<entities>\n{entities_str}\n</entities>"
            system_content = "You are a helpful cybersecurity relation annotation expert. Output valid JSON only."
        else:
            prompt_content = f"<text>\n{text}\n</text>\n<entities>\n{entities_str}\n</entities>\nReturn JSON only."
            system_content = full_relation_prompt

        raw_output = self._call_task_model_once(
            stage="relation",
            prompt_content=prompt_content,
            system_content=system_content,
            sample_id=sample_id or "window_unknown",
            full_prompt_for_hash=full_relation_prompt,
        )

        parsed = _loose_json(raw_output)
        raw_relations = parsed.get("relations", [])
        if not isinstance(raw_relations, list):
            raw_relations = []

        identity_aliases = {e["id"]: [e["id"]] for e in public_entities}
        relations = parse_relations(
            text,
            raw_relations,
            public_entities,
            identity_aliases,
        )
        return relations

    def predict_stage2_inputs(
        self,
        inputs: List[Tuple[str, List[dict]]],
        full_relation_prompt: str,
        sample_ids: Optional[List[str]] = None,
    ) -> List[List[dict]]:
        """使用统一并发上限批量执行 Stage 2 窗口预测并保持输入顺序。"""
        if sample_ids is not None and len(sample_ids) != len(inputs):
            raise ValueError("sample_ids 与 inputs 数量不一致")
        indexed = list(enumerate(list(inputs)))
        return self._parallel_map(
            lambda item: self.predict_stage2_window(
                item[1][0],
                item[1][1],
                full_relation_prompt,
                sample_id=(
                    sample_ids[item[0]] if sample_ids is not None else None
                ),
            ),
            indexed,
        )

    def evaluate_stage1_batch(
        self,
        samples: List[dict],
        full_entity_prompt: str,
        collect_errors: bool = True,
        capture_predictions: bool = False,
    ) -> Tuple[EvaluationResult, List[ErrorExample]]:
        """在样本批次上评估 Stage 1 完整 Prompt，并收集错误样本。"""
        batch_tp, batch_fp, batch_fn = 0, 0, 0
        errors: List[ErrorExample] = []
        type_counts = {t: [0, 0, 0] for t in sorted(EXTRACTION_ENTITY_TYPES)}
        normalization_correct = 0
        normalization_total = 0
        prediction_records: List[dict] = []
        overlap_threshold = 0.5
        config_overlap_counts = [0, 0, 0]

        abbreviation_contexts = None
        if any("document_abbreviations" in sample for sample in samples):
            abbreviation_contexts = [
                str(sample.get("document_abbreviations", "(none detected)"))
                for sample in samples
            ]
        predictions = self.predict_stage1_texts(
            [sample["text"] for sample in samples],
            full_entity_prompt,
            document_abbreviations=abbreviation_contexts,
            sample_ids=[
                sample.get("sample_id") or sample.get("id") or "unknown"
                for sample in samples
            ],
        )
        if self.vulnerability_anchored_backfill:
            predictions = vulnerability_anchored_backfill(
                samples, predictions, enabled=True
            )
        for sample, pred_entities in zip(samples, predictions):
            sample_id = sample.get("sample_id") or sample.get("id") or "unknown"
            text = sample["text"]
            gold_entities = sample.get("gold_entities", sample.get("entities", []))
            tp, fp, fn = calc_strict_entity_sample_counts(pred_entities, gold_entities)

            batch_tp += tp
            batch_fp += fp
            batch_fn += fn

            for entity_type in sorted(EXTRACTION_ENTITY_TYPES):
                type_tp, type_fp, type_fn = calc_strict_entity_sample_counts(
                    pred_entities,
                    gold_entities,
                    allowed_types={entity_type},
                )
                type_counts[entity_type][0] += type_tp
                type_counts[entity_type][1] += type_fp
                type_counts[entity_type][2] += type_fn

            config_strict = calc_strict_entity_sample_counts(
                pred_entities,
                gold_entities,
                allowed_types={"Configuration"},
            )
            config_overlap = calc_same_type_jaccard_overlap_counts(
                pred_entities,
                gold_entities,
                allowed_types={"Configuration"},
                threshold=overlap_threshold,
            )
            for index, count in enumerate(config_overlap):
                config_overlap_counts[index] += count
            config_error_profile = {
                "entity_type": "Configuration",
                "strict_tp": config_strict[0],
                "strict_fp": config_strict[1],
                "strict_fn": config_strict[2],
                "boundary_overlap_pairs": max(0, config_overlap[0] - config_strict[0]),
                "overlap_threshold": overlap_threshold,
            }

            pred_by_span = {
                (e.get("start"), e.get("end"), e.get("type")): e
                for e in pred_entities
            }
            for gold_entity in gold_entities:
                gold_normalized = gold_entity.get("normalized_id")
                if not gold_normalized:
                    continue
                span_key = (
                    gold_entity.get("start"),
                    gold_entity.get("end"),
                    gold_entity.get("type"),
                )
                if span_key in pred_by_span:
                    normalization_total += 1
                    if pred_by_span[span_key].get("normalized_id") == gold_normalized:
                        normalization_correct += 1

            if capture_predictions:
                prediction_records.append({
                    "sample_id": sample_id,
                    "gold_entities": gold_entities,
                    "pred_entities": pred_entities,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

            if collect_errors and (fp > 0 or fn > 0):
                errors.append(
                    ErrorExample(
                        sample_id=sample_id,
                        input_text=text,
                        gold_output={"entities": gold_entities},
                        predicted_output={"entities": pred_entities},
                        error_details={
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "configuration_focus": config_error_profile,
                            "document_abbreviations": sample.get(
                                "document_abbreviations"
                            ),
                        },
                    )
                )

        by_type = {}
        for entity_type, counts in type_counts.items():
            payload = aggregate_micro_f1(*counts).to_dict()
            payload["gold_support"] = counts[0] + counts[2]
            payload["predicted_support"] = counts[0] + counts[1]
            payload["applicable"] = bool(payload["gold_support"] or payload["predicted_support"])
            if not payload["applicable"]:
                payload["precision"] = None
                payload["recall"] = None
                payload["f1"] = None
            by_type[entity_type] = payload
        details = {
            "num_samples": len(samples),
            "by_type": by_type,
            "configuration_overlap_diagnostic": {
                **aggregate_micro_f1(*config_overlap_counts).to_dict(),
                "threshold": overlap_threshold,
                "matching": "one_to_one_same_type_span_jaccard",
                "selection_role": "diagnostic_only",
                "strict_tp": type_counts["Configuration"][0],
                "boundary_only_matches": max(
                    0,
                    config_overlap_counts[0] - type_counts["Configuration"][0],
                ),
            },
            "normalization": {
                "correct": normalization_correct,
                "total": normalization_total,
                "accuracy": (
                    normalization_correct / normalization_total
                    if normalization_total
                    else None
                ),
                "scope": "strict_span_matches_with_non_null_gold_normalized_id",
            },
        }
        if capture_predictions:
            details["predictions"] = prediction_records
        eval_result = aggregate_micro_f1(batch_tp, batch_fp, batch_fn, details=details)
        return eval_result, errors

    def evaluate_stage2_batch(
        self,
        samples: List[dict],
        full_relation_prompt: str,
        collect_errors: bool = True,
        capture_predictions: bool = False,
    ) -> Tuple[EvaluationResult, List[ErrorExample]]:
        """在样本批次上评估 Stage 2 完整 Prompt，输入实体必须来自预先冻结的预测。"""
        batch_tp, batch_fp, batch_fn = 0, 0, 0
        errors: List[ErrorExample] = []
        type_counts = {t: [0, 0, 0] for t in sorted(EXTRACTION_RELATION_TYPES)}
        prediction_records: List[dict] = []

        for sample in samples:
            if "fixed_entities" not in sample or sample["fixed_entities"] is None:
                sample_id = sample.get("sample_id") or sample.get("id") or "unknown"
                raise ValueError(
                    f"Stage 2 样本 {sample_id} 缺少 fixed_entities 字段；"
                    "Stage 2 严禁回退到 Gold entities 或其它未经冻结的实体输入"
                )

        predictions = self.predict_stage2_inputs(
            [
                (
                    sample["text"],
                    sample["fixed_entities"],
                )
                for sample in samples
            ],
            full_relation_prompt,
            sample_ids=[
                sample.get("sample_id") or sample.get("id") or "unknown"
                for sample in samples
            ],
        )
        for sample, pred_relations in zip(samples, predictions):
            sample_id = sample.get("sample_id") or sample.get("id") or "unknown"
            text = sample["text"]
            fixed_entities = sample["fixed_entities"]
            gold_entities = sample.get("gold_entities", sample.get("entities", []))
            gold_relations = sample.get("gold_relations", sample.get("relations", []))
            tp, fp, fn = calc_strict_relation_sample_counts(
                fixed_entities, pred_relations, gold_entities, gold_relations
            )

            batch_tp += tp
            batch_fp += fp
            batch_fn += fn

            for relation_type in sorted(EXTRACTION_RELATION_TYPES):
                type_tp, type_fp, type_fn = calc_strict_relation_sample_counts(
                    fixed_entities,
                    pred_relations,
                    gold_entities,
                    gold_relations,
                    allowed_relation_types={relation_type},
                )
                type_counts[relation_type][0] += type_tp
                type_counts[relation_type][1] += type_fp
                type_counts[relation_type][2] += type_fn

            if capture_predictions:
                prediction_records.append({
                    "sample_id": sample_id,
                    "fixed_entities": fixed_entities,
                    "gold_entities": gold_entities,
                    "gold_relations": gold_relations,
                    "pred_relations": pred_relations,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

            if collect_errors and (fp > 0 or fn > 0):
                errors.append(
                    ErrorExample(
                        sample_id=sample_id,
                        input_text=text,
                        gold_output={"relations": gold_relations},
                        predicted_output={"relations": pred_relations},
                        error_details={
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "fixed_entities": fixed_entities,
                        },
                    )
                )

        by_type = {}
        for relation_type, counts in type_counts.items():
            payload = aggregate_micro_f1(*counts).to_dict()
            payload["gold_support"] = counts[0] + counts[2]
            payload["predicted_support"] = counts[0] + counts[1]
            payload["applicable"] = bool(payload["gold_support"] or payload["predicted_support"])
            if not payload["applicable"]:
                payload["precision"] = None
                payload["recall"] = None
                payload["f1"] = None
            by_type[relation_type] = payload
        details = {
            "num_samples": len(samples),
            "by_type": by_type,
        }
        if capture_predictions:
            details["predictions"] = prediction_records
        eval_result = aggregate_micro_f1(batch_tp, batch_fp, batch_fn, details=details)
        return eval_result, errors
