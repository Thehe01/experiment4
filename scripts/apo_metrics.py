"""APO-only diagnostic metrics that do not alter frozen baseline evaluation."""

from __future__ import annotations


def _entity_spans(entities: list[dict]) -> set[tuple[int, int, str]]:
    return {
        (entity["start"], entity["end"], entity["type"])
        for entity in entities
    }


def _relation_tuples(
    entities: list[dict], relations: list[dict]
) -> set[tuple[tuple[int, int, str], str, tuple[int, int, str]]]:
    entity_map = {
        entity["id"]: (entity["start"], entity["end"], entity["type"])
        for entity in entities
    }
    return {
        (entity_map[relation["head"]], relation["type"], entity_map[relation["tail"]])
        for relation in relations
        if relation.get("head") in entity_map and relation.get("tail") in entity_map
    }


def conditional_relation_recall_by_type(
    pred_entities: list[dict],
    pred_relations: list[dict],
    gold_entities: list[dict],
    gold_relations: list[dict],
    endpoint_entities: list[dict] | None = None,
) -> dict:
    """Measure strict relation recall conditional on both Gold endpoints.

    ``endpoint_entities`` should be the Stage-1 candidate layer.  The metric
    therefore separates endpoint extraction failures from relation decisions
    made after both correct mentions were available.
    """
    endpoint_entities = endpoint_entities or pred_entities
    endpoint_spans = _entity_spans(endpoint_entities)
    pred_tuples = _relation_tuples(pred_entities, pred_relations)
    gold_entity_map = {
        entity["id"]: (entity["start"], entity["end"], entity["type"])
        for entity in gold_entities
    }
    gold_by_type: dict[str, set] = {}
    eligible_by_type: dict[str, set] = {}
    for relation in gold_relations:
        head = gold_entity_map.get(relation.get("head"))
        tail = gold_entity_map.get(relation.get("tail"))
        if not head or not tail:
            continue
        relation_type = relation["type"]
        triple = (head, relation_type, tail)
        gold_by_type.setdefault(relation_type, set()).add(triple)
        if head in endpoint_spans and tail in endpoint_spans:
            eligible_by_type.setdefault(relation_type, set()).add(triple)

    results = {}
    for relation_type in sorted(gold_by_type):
        gold_tuples = gold_by_type[relation_type]
        eligible = eligible_by_type.get(relation_type, set())
        predicted = {
            triple for triple in pred_tuples if triple[1] == relation_type
        }
        tp = len(eligible & predicted)
        eligible_total = len(eligible)
        gold_total = len(gold_tuples)
        results[relation_type] = {
            "conditional_recall": round(tp / eligible_total, 4)
            if eligible_total else 0.0,
            "tp": tp,
            "fn": eligible_total - tp,
            "eligible_gold": eligible_total,
            "endpoint_missing": gold_total - eligible_total,
            "gold_total": gold_total,
        }
    return results


def aggregate_conditional_relation_recall_by_type(
    metrics_list: list[dict],
) -> dict:
    """Micro-average conditional relation recall diagnostics by type."""
    relation_types = set().union(*(set(item) for item in metrics_list))
    results = {}
    for relation_type in sorted(relation_types):
        tp = sum(
            int(item.get(relation_type, {}).get("tp", 0))
            for item in metrics_list
        )
        eligible_gold = sum(
            int(item.get(relation_type, {}).get("eligible_gold", 0))
            for item in metrics_list
        )
        endpoint_missing = sum(
            int(item.get(relation_type, {}).get("endpoint_missing", 0))
            for item in metrics_list
        )
        gold_total = sum(
            int(item.get(relation_type, {}).get("gold_total", 0))
            for item in metrics_list
        )
        results[relation_type] = {
            "conditional_recall": round(tp / eligible_gold, 4)
            if eligible_gold else 0.0,
            "tp": tp,
            "fn": eligible_gold - tp,
            "eligible_gold": eligible_gold,
            "endpoint_missing": endpoint_missing,
            "gold_total": gold_total,
        }
    return results
