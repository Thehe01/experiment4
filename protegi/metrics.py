"""ProTeGi 严格评估指标适配器。

直接复用 eval_metrics.py 的严格匹配逻辑：
- Stage 1: Strict Entity Micro-F1 (按 batch 累积 TP/FP/FN 计算)
- Stage 2: Strict Relation Micro-F1 (按 batch 累积 TP/FP/FN 计算)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_metrics import (
    extract_entity_spans,
    extract_relation_tuples,
    calc_entity_metrics,
    calc_relation_metrics,
)
from schema import EXTRACTION_ENTITY_TYPES, EXTRACTION_RELATION_TYPES
from protegi.models import EvaluationResult


def calc_strict_entity_sample_counts(
    pred_entities: List[dict],
    gold_entities: List[dict],
    allowed_types: Set[str] = set(EXTRACTION_ENTITY_TYPES),
) -> Tuple[int, int, int]:
    """计算单个样本上的严格实体匹配 TP, FP, FN。"""
    filtered_pred = [e for e in pred_entities if e.get("type") in allowed_types]
    filtered_gold = [e for e in gold_entities if e.get("type") in allowed_types]

    pred_spans = extract_entity_spans(filtered_pred)
    gold_spans = extract_entity_spans(filtered_gold)

    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    return tp, fp, fn


def calc_same_type_jaccard_overlap_counts(
    pred_entities: List[dict],
    gold_entities: List[dict],
    *,
    allowed_types: Set[str] = set(EXTRACTION_ENTITY_TYPES),
    threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """Return one-to-one same-type span-overlap counts for diagnostics only.

    Candidate pairs must have span Jaccard >= ``threshold``.  Exact matches are
    included.  A maximum-cardinality bipartite matching prevents one broad
    prediction from receiving credit for multiple Gold mentions.  This metric
    must never replace strict matching for search or final model selection.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("overlap threshold must be in (0, 1]")

    pred_spans = sorted({
        (int(e["start"]), int(e["end"]), str(e["type"]))
        for e in pred_entities
        if e.get("type") in allowed_types
        and isinstance(e.get("start"), int)
        and isinstance(e.get("end"), int)
        and e["end"] > e["start"]
    })
    gold_spans = sorted({
        (int(e["start"]), int(e["end"]), str(e["type"]))
        for e in gold_entities
        if e.get("type") in allowed_types
        and isinstance(e.get("start"), int)
        and isinstance(e.get("end"), int)
        and e["end"] > e["start"]
    })

    candidates: List[List[int]] = []
    for p_start, p_end, p_type in pred_spans:
        compatible = []
        for gold_index, (g_start, g_end, g_type) in enumerate(gold_spans):
            if p_type != g_type:
                continue
            intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
            union = max(p_end, g_end) - min(p_start, g_start)
            if union and intersection / union >= threshold:
                compatible.append(gold_index)
        candidates.append(compatible)

    matched_gold_to_pred: Dict[int, int] = {}

    def augment(pred_index: int, visited_gold: Set[int]) -> bool:
        for gold_index in candidates[pred_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous = matched_gold_to_pred.get(gold_index)
            if previous is None or augment(previous, visited_gold):
                matched_gold_to_pred[gold_index] = pred_index
                return True
        return False

    tp = sum(augment(index, set()) for index in range(len(pred_spans)))
    return tp, len(pred_spans) - tp, len(gold_spans) - tp


def calc_strict_relation_sample_counts(
    pred_entities: List[dict],
    pred_relations: List[dict],
    gold_entities: List[dict],
    gold_relations: List[dict],
    allowed_entity_types: Set[str] = set(EXTRACTION_ENTITY_TYPES),
    allowed_relation_types: Set[str] = set(EXTRACTION_RELATION_TYPES),
) -> Tuple[int, int, int]:
    """计算单个样本上的严格关系匹配 TP, FP, FN。"""
    filtered_pred_ent = [e for e in pred_entities if e.get("type") in allowed_entity_types]
    filtered_gold_ent = [e for e in gold_entities if e.get("type") in allowed_entity_types]
    filtered_pred_rel = [r for r in pred_relations if r.get("type") in allowed_relation_types]
    filtered_gold_rel = [r for r in gold_relations if r.get("type") in allowed_relation_types]

    pred_tuples = extract_relation_tuples(filtered_pred_ent, filtered_pred_rel)
    gold_tuples = extract_relation_tuples(filtered_gold_ent, filtered_gold_rel)

    tp = len(pred_tuples & gold_tuples)
    fp = len(pred_tuples - gold_tuples)
    fn = len(gold_tuples - pred_tuples)
    return tp, fp, fn


def aggregate_micro_f1(tp: int, fp: int, fn: int, details: dict = None) -> EvaluationResult:
    """根据累积的 TP, FP, FN 计算严谨的 Micro 指标。"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return EvaluationResult(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        details=details or {},
    )
