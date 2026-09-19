"""
评估指标计算模块

计算实体抽取、关系抽取、SCR、NA 等指标。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from schema import ENTITY_TYPES, RELATION_ARGUMENT_TYPES, RELATION_TYPES


def load_annotation(file_path: Path) -> dict:
    """加载标注文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_entity_spans(entities: List[dict]) -> Set[Tuple[int, int, str]]:
    """提取实体边界集合 (start, end, type)"""
    return {(e['start'], e['end'], e['type']) for e in entities}


def extract_relation_tuples(entities: List[dict], relations: List[dict]) -> Set[Tuple[Tuple[int, int, str], str, Tuple[int, int, str]]]:
    """提取关系三元组集合 ((head_start, head_end, head_type), relation_type, (tail_start, tail_end, tail_type))"""
    entity_map = {e['id']: (e['start'], e['end'], e['type']) for e in entities}

    tuples = set()
    for r in relations:
        head_span = entity_map.get(r['head'])
        tail_span = entity_map.get(r['tail'])
        if head_span and tail_span:
            tuples.add((head_span, r['type'], tail_span))
    return tuples


def _canonical_normalized_id(entity: dict) -> str:
    """Return a comparison-safe normalized identifier.

    Missing normalization must not disappear from the prediction set, because
    silently dropping such a relation would avoid a false-positive penalty.
    The span-scoped sentinel can never match a properly normalized Gold entity.
    """
    value = str(entity.get('normalized_id') or '').strip()
    if value:
        if value.casefold().startswith('cpe:2.3:'):
            return value.casefold()
        return ''.join(value.split()).upper()
    return (
        f"__MISSING__:{entity.get('type', '')}:"
        f"{entity.get('start', '')}:{entity.get('end', '')}"
    )


def extract_normalized_relation_facts(
    entities: List[dict], relations: List[dict]
) -> Set[Tuple[str, str, str, str, str]]:
    """Project mention-level relations to document-level normalized facts.

    The caller evaluates one document at a time, so the document identifier is
    supplied by the surrounding evaluation record. Duplicate mention pairs for
    the same normalized fact collapse deterministically in this set.
    """
    entity_map = {e['id']: e for e in entities}
    facts = set()
    for relation in relations:
        head = entity_map.get(relation['head'])
        tail = entity_map.get(relation['tail'])
        if not head or not tail:
            continue
        facts.add((
            head['type'],
            _canonical_normalized_id(head),
            relation['type'],
            tail['type'],
            _canonical_normalized_id(tail),
        ))
    return facts


def _calc_set_metrics(pred_items: set, gold_items: set) -> dict:
    tp = len(pred_items & gold_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def calc_entity_metrics(pred_entities: List[dict], gold_entities: List[dict]) -> dict:
    """
    计算实体抽取的 P/R/F1

    匹配方式：(start, end, type) 严格匹配
    """
    pred_spans = extract_entity_spans(pred_entities)
    gold_spans = extract_entity_spans(gold_entities)

    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def _maximum_same_type_overlap_matches(
    pred_entities: List[dict], gold_entities: List[dict],
) -> int:
    """Return a one-to-one maximum matching for same-type overlapping spans.

    This deliberately does not use normalized IDs or surface forms.  It is an
    auxiliary boundary-tolerance diagnostic: a prediction can match a Gold
    mention only when both have the same entity type and their character
    intervals overlap.  The augmenting-path matching prevents one long span
    from receiving credit for several Gold mentions.
    """
    pred_spans = sorted(extract_entity_spans(pred_entities))
    gold_spans = sorted(extract_entity_spans(gold_entities))
    candidates: List[List[int]] = []
    for p_start, p_end, p_type in pred_spans:
        compatible = [
            gold_index
            for gold_index, (g_start, g_end, g_type) in enumerate(gold_spans)
            if p_type == g_type and max(p_start, g_start) < min(p_end, g_end)
        ]
        candidates.append(compatible)

    matched_gold_to_pred: Dict[int, int] = {}

    def _augment(pred_index: int, visited_gold: Set[int]) -> bool:
        for gold_index in candidates[pred_index]:
            if gold_index in visited_gold:
                continue
            visited_gold.add(gold_index)
            previous_pred = matched_gold_to_pred.get(gold_index)
            if previous_pred is None or _augment(previous_pred, visited_gold):
                matched_gold_to_pred[gold_index] = pred_index
                return True
        return False

    return sum(
        _augment(pred_index, set())
        for pred_index in range(len(pred_spans))
    )


def calc_overlap_entity_metrics(
    pred_entities: List[dict], gold_entities: List[dict]
) -> dict:
    """Calculate an auxiliary same-type overlap-span entity F1.

    Strict span F1 remains the primary entity metric.  This measure is
    reported alongside it to make boundary-only errors (for example a product
    mention with an extra edition qualifier) distinguishable from a wholly
    incorrect entity prediction.
    """
    pred_count = len(extract_entity_spans(pred_entities))
    gold_count = len(extract_entity_spans(gold_entities))
    tp = _maximum_same_type_overlap_matches(pred_entities, gold_entities)
    fp = pred_count - tp
    fn = gold_count - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


def calc_entity_metrics_by_type(pred_entities: List[dict], gold_entities: List[dict]) -> dict:
    """按实体类型分别计算 P/R/F1"""
    pred_by_type = {}
    for e in pred_entities:
        pred_by_type.setdefault(e['type'], []).append(e)

    gold_by_type = {}
    for e in gold_entities:
        gold_by_type.setdefault(e['type'], []).append(e)

    all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
    results = {}

    for entity_type in sorted(all_types):
        pred_list = pred_by_type.get(entity_type, [])
        gold_list = gold_by_type.get(entity_type, [])
        results[entity_type] = calc_entity_metrics(pred_list, gold_list)

    return results


def calc_overlap_entity_metrics_by_type(
    pred_entities: List[dict], gold_entities: List[dict]
) -> dict:
    """Calculate auxiliary overlap-span entity F1 separately by type."""
    pred_by_type = {}
    for entity in pred_entities:
        pred_by_type.setdefault(entity['type'], []).append(entity)
    gold_by_type = {}
    for entity in gold_entities:
        gold_by_type.setdefault(entity['type'], []).append(entity)
    all_types = set(pred_by_type) | set(gold_by_type)
    return {
        entity_type: calc_overlap_entity_metrics(
            pred_by_type.get(entity_type, []),
            gold_by_type.get(entity_type, []),
        )
        for entity_type in sorted(all_types)
    }


def calc_relation_metrics(pred_entities: List[dict], pred_relations: List[dict],
                          gold_entities: List[dict], gold_relations: List[dict]) -> dict:
    """
    计算关系抽取的 P/R/F1

    匹配方式：(head_entity_span, relation_type, tail_entity_span) 严格匹配
    """
    pred_tuples = extract_relation_tuples(pred_entities, pred_relations)
    gold_tuples = extract_relation_tuples(gold_entities, gold_relations)

    tp = len(pred_tuples & gold_tuples)
    fp = len(pred_tuples - gold_tuples)
    fn = len(gold_tuples - pred_tuples)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calc_normalized_relation_metrics(
    pred_entities: List[dict], pred_relations: List[dict],
    gold_entities: List[dict], gold_relations: List[dict],
) -> dict:
    """Calculate document-level normalized fact P/R/F1.

    This metric complements, but does not replace, strict mention-level relation
    F1. It makes alias and repeated-mention effects visible for KG construction.
    """
    pred_facts = extract_normalized_relation_facts(
        pred_entities, pred_relations
    )
    gold_facts = extract_normalized_relation_facts(
        gold_entities, gold_relations
    )
    return _calc_set_metrics(pred_facts, gold_facts)


def calc_normalized_relation_metrics_by_type(
    pred_entities: List[dict], pred_relations: List[dict],
    gold_entities: List[dict], gold_relations: List[dict],
) -> dict:
    """Calculate normalized fact metrics separately for each relation type."""
    pred_facts = extract_normalized_relation_facts(
        pred_entities, pred_relations
    )
    gold_facts = extract_normalized_relation_facts(
        gold_entities, gold_relations
    )
    relation_types = {
        fact[2] for fact in pred_facts | gold_facts
    }
    return {
        relation_type: _calc_set_metrics(
            {fact for fact in pred_facts if fact[2] == relation_type},
            {fact for fact in gold_facts if fact[2] == relation_type},
        )
        for relation_type in sorted(relation_types)
    }


def calc_evidence_ambiguity_metrics(
    entities: List[dict], relations: List[dict], text: str = ""
) -> dict:
    """统计关系证据区间的歧义率。

    与 Gold 审计保持同一口径：证据区间非法、超过 200 字符，或在区间内
    出现多个端点同类型提及，均记为歧义。该指标是诊断指标，不改变严格 F1。
    """
    entity_map = {entity.get("id"): entity for entity in entities}
    total = 0
    ambiguous = 0
    reasons: Dict[str, int] = {}
    for relation in relations:
        total += 1
        head = entity_map.get(relation.get("head"))
        tail = entity_map.get(relation.get("tail"))
        if not head or not tail:
            ambiguous += 1
            reasons["missing_relation_endpoint"] = (
                reasons.get("missing_relation_endpoint", 0) + 1
            )
            continue
        relation_reasons = []
        start = relation.get("evidence_start")
        end = relation.get("evidence_end")
        evidence = str(relation.get("evidence", ""))
        valid = (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
            and text[start:end] == evidence
            and start <= head.get("start", -1) < head.get("end", -1) <= end
            and start <= tail.get("start", -1) < tail.get("end", -1) <= end
        )
        if not valid:
            relation_reasons.append("invalid_relation_evidence")
        else:
            contained = [
                entity for entity in entities
                if start <= entity.get("start", -1)
                and entity.get("end", -1) <= end
            ]
            type_counts = {}
            for entity in contained:
                entity_type = entity.get("type")
                type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
            if end - start > 200:
                relation_reasons.append("evidence_over_200_chars")
            for endpoint_type in sorted({head.get("type"), tail.get("type")}):
                if type_counts.get(endpoint_type, 0) > 1:
                    relation_reasons.append(
                        f"multiple_{endpoint_type}_mentions"
                    )
        if relation_reasons:
            ambiguous += 1
            for reason in relation_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "ambiguous": ambiguous,
        "total": total,
        "rate": round(ambiguous / total, 4) if total else 0.0,
        "reasons": dict(sorted(reasons.items())),
    }


def calc_relation_metrics_by_type(pred_entities: List[dict], pred_relations: List[dict],
                                   gold_entities: List[dict], gold_relations: List[dict]) -> dict:
    """按关系类型分别计算 P/R/F1"""
    # 建立 entity id -> span 映射
    pred_entity_map = {e['id']: (e['start'], e['end'], e['type']) for e in pred_entities}
    gold_entity_map = {e['id']: (e['start'], e['end'], e['type']) for e in gold_entities}

    # 按关系类型分组
    pred_by_type = {}
    for r in pred_relations:
        pred_by_type.setdefault(r['type'], []).append(r)

    gold_by_type = {}
    for r in gold_relations:
        gold_by_type.setdefault(r['type'], []).append(r)

    all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
    results = {}

    for rel_type in sorted(all_types):
        pred_list = pred_by_type.get(rel_type, [])
        gold_list = gold_by_type.get(rel_type, [])

        # 计算该类型的三元组
        pred_tuples = set()
        for r in pred_list:
            head = pred_entity_map.get(r['head'])
            tail = pred_entity_map.get(r['tail'])
            if head and tail:
                pred_tuples.add((head, r['type'], tail))

        gold_tuples = set()
        for r in gold_list:
            head = gold_entity_map.get(r['head'])
            tail = gold_entity_map.get(r['tail'])
            if head and tail:
                gold_tuples.add((head, r['type'], tail))

        tp = len(pred_tuples & gold_tuples)
        fp = len(pred_tuples - gold_tuples)
        fn = len(gold_tuples - pred_tuples)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[rel_type] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return results


def calc_scr(pred_entities: List[dict], pred_relations: List[dict]) -> dict:
    """
    计算 Schema Consistency Rate (SCR)

    检查：
    1. 实体类型是否属于第三章定义的 6 类实体之一
    2. 关系类型是否属于第三章定义的 5 类关系之一
    3. 关系方向是否符合 schema 定义
    """
    # 实体类型检查
    entity_total = len(pred_entities)
    entity_valid = sum(1 for e in pred_entities if e['type'] in ENTITY_TYPES)
    entity_scr = entity_valid / entity_total if entity_total > 0 else 1.0

    # 关系类型和头尾类型检查
    relation_total = len(pred_relations)
    relation_valid = 0

    # 建立 entity id -> type 映射
    entity_type_map = {e['id']: e['type'] for e in pred_entities}

    for r in pred_relations:
        # 检查关系类型
        if r['type'] not in RELATION_TYPES:
            continue

        # 检查关系方向和尾实体类型
        head_type = entity_type_map.get(r['head'])
        tail_type = entity_type_map.get(r['tail'])

        if head_type and tail_type:
            expected_types = RELATION_ARGUMENT_TYPES.get(r['type'])
            if expected_types is None:
                continue
            expected_head, expected_tail = expected_types
            if head_type == expected_head and tail_type == expected_tail:
                relation_valid += 1

    relation_scr = relation_valid / relation_total if relation_total > 0 else 1.0

    # 整体 SCR
    overall_total = entity_total + relation_total
    overall_valid = entity_valid + relation_valid
    overall_scr = overall_valid / overall_total if overall_total > 0 else 1.0

    return {
        'entity_scr': round(entity_scr, 4),
        'relation_scr': round(relation_scr, 4),
        'overall_scr': round(overall_scr, 4),
        'entity_valid': entity_valid,
        'entity_total': entity_total,
        'relation_valid': relation_valid,
        'relation_total': relation_total
    }


def calc_na(pred_entities: List[dict], gold_entities: List[dict]) -> dict:
    """
    计算 Normalization Accuracy (NA)

    只在带 normalized_id 的实体上计算。
    先按 (start, end, type) 对齐，再判断 normalized_id 是否一致。
    """
    # 建立预测实体的 span -> normalized_id 映射
    pred_map = {}
    for e in pred_entities:
        pred_map[(e['start'], e['end'], e['type'])] = e.get('normalized_id')

    # 只在 gold 中本应规范化的实体上统计；漏预测或未规范化视为错误
    type_stats = {}
    total_correct = 0
    total_count = 0

    for e in gold_entities:
        gold_normalized_id = e.get('normalized_id')
        if not gold_normalized_id:
            continue

        total_count += 1
        entity_type = e['type']
        pred_normalized_id = pred_map.get((e['start'], e['end'], e['type']))

        if entity_type not in type_stats:
            type_stats[entity_type] = {'correct': 0, 'total': 0}

        type_stats[entity_type]['total'] += 1

        if pred_normalized_id == gold_normalized_id:
            type_stats[entity_type]['correct'] += 1
            total_correct += 1

    # 计算各类型的 NA
    results = {}
    for entity_type, stats in type_stats.items():
        results[entity_type] = {
            'na': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0.0,
            'correct': stats['correct'],
            'total': stats['total']
        }

    # 整体 NA
    results['overall'] = {
        'na': round(total_correct / total_count, 4) if total_count > 0 else 0.0,
        'correct': total_correct,
        'total': total_count
    }

    return results


def evaluate_single_doc(pred_file: Path, gold_file: Path) -> dict:
    """评估单个文档"""
    pred = load_annotation(pred_file)
    gold = load_annotation(gold_file)

    return {
        'doc_id': pred.get('doc_id', pred_file.stem),
        'entity_metrics': calc_entity_metrics(pred['entities'], gold['entities']),
        'entity_metrics_by_type': calc_entity_metrics_by_type(pred['entities'], gold['entities']),
        'overlap_entity_metrics': calc_overlap_entity_metrics(
            pred['entities'], gold['entities']
        ),
        'overlap_entity_metrics_by_type': calc_overlap_entity_metrics_by_type(
            pred['entities'], gold['entities']
        ),
        'relation_metrics': calc_relation_metrics(pred['entities'], pred['relations'],
                                                   gold['entities'], gold['relations']),
        'relation_metrics_by_type': calc_relation_metrics_by_type(pred['entities'], pred['relations'],
                                                                   gold['entities'], gold['relations']),
        'normalized_relation_metrics': calc_normalized_relation_metrics(
            pred['entities'], pred['relations'],
            gold['entities'], gold['relations'],
        ),
        'normalized_relation_metrics_by_type': (
            calc_normalized_relation_metrics_by_type(
                pred['entities'], pred['relations'],
                gold['entities'], gold['relations'],
            )
        ),
        'evidence_ambiguity': calc_evidence_ambiguity_metrics(
            pred['entities'], pred['relations'], pred.get('text', '')
        ),
        'gold_evidence_ambiguity': calc_evidence_ambiguity_metrics(
            gold['entities'], gold['relations'], gold.get('text', '')
        ),
        'scr': calc_scr(pred['entities'], pred['relations']),
        'na': calc_na(pred['entities'], gold['entities'])
    }


def evaluate_dataset(pred_dir: Path, gold_dir: Path, doc_ids: List[str], strict: bool = False) -> dict:
    """评估整个数据集"""
    results = []
    missing_prediction_files = []
    missing_gold_files = []

    for doc_id in doc_ids:
        pred_file = pred_dir / f"{doc_id}.json"
        gold_file = gold_dir / f"{doc_id}.json"

        if not pred_file.exists():
            missing_prediction_files.append(str(pred_file))
            continue
        if not gold_file.exists():
            missing_gold_files.append(str(gold_file))
            continue

        result = evaluate_single_doc(pred_file, gold_file)
        results.append(result)

    if strict and (missing_prediction_files or missing_gold_files):
        parts = []
        if missing_prediction_files:
            parts.append(f"missing predictions: {missing_prediction_files}")
        if missing_gold_files:
            parts.append(f"missing gold files: {missing_gold_files}")
        raise FileNotFoundError("; ".join(parts))

    if not results:
        return {}

    # 汇总指标
    overall_entity = aggregate_metrics([r['entity_metrics'] for r in results])
    overall_overlap_entity = aggregate_metrics([
        r['overlap_entity_metrics'] for r in results
    ])
    overall_relation = aggregate_metrics([r['relation_metrics'] for r in results])
    overall_scr = aggregate_scr([r['scr'] for r in results])
    overall_na = aggregate_na([r['na'] for r in results])

    # 按类型汇总
    entity_by_type = aggregate_metrics_by_type([r['entity_metrics_by_type'] for r in results])
    overlap_entity_by_type = aggregate_metrics_by_type([
        r['overlap_entity_metrics_by_type'] for r in results
    ])
    relation_by_type = aggregate_metrics_by_type([r['relation_metrics_by_type'] for r in results])
    normalized_relation = aggregate_metrics([
        r['normalized_relation_metrics'] for r in results
    ])
    normalized_relation_by_type = aggregate_metrics_by_type([
        r['normalized_relation_metrics_by_type'] for r in results
    ])
    evidence_ambiguity = aggregate_evidence_ambiguity(
        [r['evidence_ambiguity'] for r in results]
    )
    gold_evidence_ambiguity = aggregate_evidence_ambiguity(
        [r['gold_evidence_ambiguity'] for r in results]
    )

    return {
        'num_docs': len(results),
        'overall_entity_metrics': overall_entity,
        'overall_overlap_entity_metrics': overall_overlap_entity,
        'overall_relation_metrics': overall_relation,
        'overall_scr': overall_scr,
        'overall_na': overall_na,
        'entity_metrics_by_type': entity_by_type,
        'overlap_entity_metrics_by_type': overlap_entity_by_type,
        'relation_metrics_by_type': relation_by_type,
        'overall_normalized_relation_metrics': normalized_relation,
        'normalized_relation_metrics_by_type': normalized_relation_by_type,
        'overall_evidence_ambiguity': {
            'prediction': evidence_ambiguity,
            'gold': gold_evidence_ambiguity,
        },
        'requested_doc_count': len(doc_ids),
        'per_doc_results': results,
        'missing_prediction_files': missing_prediction_files,
        'missing_gold_files': missing_gold_files
    }


def aggregate_evidence_ambiguity(metrics_list: List[dict]) -> dict:
    """汇总证据歧义计数、比例及原因。"""
    ambiguous = sum(item.get('ambiguous', 0) for item in metrics_list)
    total = sum(item.get('total', 0) for item in metrics_list)
    reasons: Dict[str, int] = {}
    for item in metrics_list:
        for reason, count in item.get('reasons', {}).items():
            reasons[reason] = reasons.get(reason, 0) + count
    return {
        'ambiguous': ambiguous,
        'total': total,
        'rate': round(ambiguous / total, 4) if total else 0.0,
        'reasons': dict(sorted(reasons.items())),
    }


def aggregate_metrics(metrics_list: List[dict]) -> dict:
    """汇总 P/R/F1 指标（micro-average）"""
    total_tp = sum(m['tp'] for m in metrics_list)
    total_fp = sum(m['fp'] for m in metrics_list)
    total_fn = sum(m['fn'] for m in metrics_list)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }


def aggregate_scr(scr_list: List[dict]) -> dict:
    """汇总 SCR 指标"""
    total_entity_valid = sum(s['entity_valid'] for s in scr_list)
    total_entity_total = sum(s['entity_total'] for s in scr_list)
    total_relation_valid = sum(s['relation_valid'] for s in scr_list)
    total_relation_total = sum(s['relation_total'] for s in scr_list)

    entity_scr = total_entity_valid / total_entity_total if total_entity_total > 0 else 1.0
    relation_scr = total_relation_valid / total_relation_total if total_relation_total > 0 else 1.0
    overall_scr = (total_entity_valid + total_relation_valid) / (total_entity_total + total_relation_total) if (total_entity_total + total_relation_total) > 0 else 1.0

    return {
        'entity_scr': round(entity_scr, 4),
        'relation_scr': round(relation_scr, 4),
        'overall_scr': round(overall_scr, 4)
    }


def aggregate_na(na_list: List[dict]) -> dict:
    """汇总 NA 指标"""
    type_stats = {}

    for na in na_list:
        for entity_type, stats in na.items():
            if entity_type == 'overall':
                continue
            if entity_type not in type_stats:
                type_stats[entity_type] = {'correct': 0, 'total': 0}
            type_stats[entity_type]['correct'] += stats['correct']
            type_stats[entity_type]['total'] += stats['total']

    total_correct = sum(stats['correct'] for stats in type_stats.values())
    total_count = sum(stats['total'] for stats in type_stats.values())

    results = {}
    for entity_type, stats in type_stats.items():
        results[entity_type] = {
            'na': round(stats['correct'] / stats['total'], 4) if stats['total'] > 0 else 0.0,
            'correct': stats['correct'],
            'total': stats['total']
        }

    results['overall'] = {
        'na': round(total_correct / total_count, 4) if total_count > 0 else 0.0,
        'correct': total_correct,
        'total': total_count
    }

    return results


def aggregate_metrics_by_type(metrics_by_type_list: List[dict]) -> dict:
    """按类型汇总指标"""
    all_types = set()
    for m in metrics_by_type_list:
        all_types.update(m.keys())

    results = {}
    for entity_type in sorted(all_types):
        metrics_list = [m.get(entity_type, {'tp': 0, 'fp': 0, 'fn': 0}) for m in metrics_by_type_list]
        results[entity_type] = aggregate_metrics(metrics_list)

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='计算评估指标')
    parser.add_argument('--pred_dir', required=True, help='预测结果目录')
    parser.add_argument('--gold_dir', required=True, help='Gold 标注目录')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--doc_ids', nargs='+', help='文档 ID 列表（可选）')

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gold_dir = Path(args.gold_dir)

    # 如果没有指定 doc_ids，使用 gold_dir 中的所有文件
    if args.doc_ids:
        doc_ids = args.doc_ids
    else:
        doc_ids = [f.stem for f in gold_dir.glob('*.json') if f.name != '_manifest.json']

    results = evaluate_dataset(pred_dir, gold_dir, doc_ids)

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"评估完成: {results.get('num_docs', 0)} 篇文档")
    print(f"\n实体抽取指标:")
    print(f"  Precision: {results['overall_entity_metrics']['precision']}")
    print(f"  Recall: {results['overall_entity_metrics']['recall']}")
    print(f"  F1: {results['overall_entity_metrics']['f1']}")
    print(f"\n实体重叠跨度辅助指标:")
    print(
        "  Precision: "
        f"{results['overall_overlap_entity_metrics']['precision']}"
    )
    print(f"  Recall: {results['overall_overlap_entity_metrics']['recall']}")
    print(f"  F1: {results['overall_overlap_entity_metrics']['f1']}")
    print(f"\n关系抽取指标:")
    print(f"  Precision: {results['overall_relation_metrics']['precision']}")
    print(f"  Recall: {results['overall_relation_metrics']['recall']}")
    print(f"  F1: {results['overall_relation_metrics']['f1']}")
    print(f"\n规范化关系事实指标:")
    print(
        "  Precision: "
        f"{results['overall_normalized_relation_metrics']['precision']}"
    )
    print(
        "  Recall: "
        f"{results['overall_normalized_relation_metrics']['recall']}"
    )
    print(
        "  F1: "
        f"{results['overall_normalized_relation_metrics']['f1']}"
    )
    print(f"\nSCR 指标:")
    print(f"  Entity SCR: {results['overall_scr']['entity_scr']}")
    print(f"  Relation SCR: {results['overall_scr']['relation_scr']}")
    print(f"  Overall SCR: {results['overall_scr']['overall_scr']}")
    print(f"\nNA 指标:")
    print(f"  Overall NA: {results['overall_na']['overall']['na']}")
