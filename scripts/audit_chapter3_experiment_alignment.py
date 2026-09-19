"""审计 v5 是否落实第三章的两层实验边界。

该脚本只读取冻结预测和结果，不改写任何方法输出。它把不满足直接证据
契约的预测关系单独导出为复核候选，因此候选统计不会改变既有 F1。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from schema import (
    BOUNDARY_CONTRACT_VERSION,
    EXTRACTION_ENTITY_TYPES,
    EXTRACTION_RELATION_ARGUMENT_TYPES,
    EXTRACTION_RELATION_TYPES,
    MAPPING_RELATION_TYPES,
    SCHEMA_VERSION,
)


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXP_DIR / "results" / "chapter3_experiment_alignment_v1.json"
DEFAULT_CANDIDATES = EXP_DIR / "results" / "chapter3_contract_review_v1"
METHODS = ("rule", "multipass", "full", "apo", "apo_full")
EXPLOIT_TRIGGER = re.compile(
    r"\b(?:exploit(?:ed|ing|s|ation)?|leverag(?:e|ed|ing)|trigger(?:ed|ing|s)?)\b|"
    r"利用|漏洞利用|触发",
    re.IGNORECASE,
)
POST_EXPLOIT_SEQUENCE = re.compile(
    r"\bafter\s+(?:successfully\s+)?exploit(?:ing|ation)?\b|"
    r"\bfollowing\s+(?:the\s+)?exploit(?:ation)?\b|"
    r"\bexploit(?:ed|ing)\b.{0,120}\b(?:and\s+then|then|subsequently|and\s+ran)\b",
    re.IGNORECASE | re.DOTALL,
)
POST_EXPLOIT_ACTIVITY = re.compile(
    r"\b(?:RAT|C2|command[- ]and[- ]control|persistence|credential(?:s)?|"
    r"lateral\s+movement|arbitrary\s+code\s+execution|indirect\s+command\s+execution)\b|"
    r"持久化|凭据访问|横向移动|命令与控制|任意代码执行",
    re.IGNORECASE,
)
CONFIGURATION_RELEASE_SUFFIX = re.compile(
    r"(?:\s|[-_(]|^)(?:v(?:ersions?)?\s*)?\d+(?:\.\d+)+(?:\b|\))|"
    r"\b(?:build|patch|release|update)\s*[-_:]?\s*\d+|"
    r"\bversions?\s+\d+|"
    r"\b(?:\d{4})\s+(?:update|patch|release)\s+\d+|"
    r"\bColdFusion\s+(?:11|2016|2018|2021)\b",
    re.IGNORECASE,
)


def _allowed_non_wildcard_cpe(surface: str) -> bool:
    if surface.casefold().startswith("cpe:2.3:"):
        return True
    s = surface.strip()
    if re.fullmatch(r"SMB\s+version\s+1", s, re.IGNORECASE):
        return True
    if re.search(
        r"^(?:Windows\s+(?:10|11|7|8|8\.1|2000|XP|Vista)|FortiGate\s+300D|SMA\s*100)(?:\s+.*)?$",
        s,
        re.IGNORECASE,
    ):
        return True
    return False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _canonical_cwe(value: object) -> str | None:
    match = re.fullmatch(r"CWE-?(\d+)", str(value or "").strip(), re.IGNORECASE)
    return f"CWE-{match.group(1)}" if match else None


def _local_fact_block(text: str, start: int, end: int) -> str:
    left = max(text.rfind(mark, 0, start) for mark in ("\n", ".", "!", "?", "。", "！", "？")) + 1
    right_positions = [
        pos
        for mark in ("\n", ".", "!", "?", "。", "！", "？")
        if (pos := text.find(mark, end)) >= 0
    ]
    right = min(right_positions) + 1 if right_positions else len(text)
    return text[left:right]


def _weakness_has_local_explicit_cwe(entity: dict, text: str) -> bool:
    if re.search(r"CWE-\s*\d+", str(entity.get("text") or ""), re.IGNORECASE):
        return True
    normalized_id = _canonical_cwe(entity.get("normalized_id"))
    start, end = entity.get("start"), entity.get("end")
    if not normalized_id or not isinstance(start, int) or not isinstance(end, int):
        return False
    local_ids = {
        _canonical_cwe(re.sub(r"\s+", "", match.group(0)))
        for match in re.finditer(
            r"CWE-\s*\d+", _local_fact_block(text, start, end), re.IGNORECASE
        )
    }
    return normalized_id in local_ids


def _evidence_reasons(
    entities: list[dict], relation: dict, text: str
) -> list[str]:
    entity_by_id = {entity.get("id"): entity for entity in entities}
    head = entity_by_id.get(relation.get("head"))
    tail = entity_by_id.get(relation.get("tail"))
    if not head or not tail:
        return ["missing_relation_endpoint"]

    start = relation.get("evidence_start")
    end = relation.get("evidence_end")
    evidence = relation.get("evidence")
    valid = (
        isinstance(start, int)
        and isinstance(end, int)
        and isinstance(evidence, str)
        and 0 <= start < end <= len(text)
        and text[start:end] == evidence
        and isinstance(head.get("start"), int)
        and isinstance(head.get("end"), int)
        and isinstance(tail.get("start"), int)
        and isinstance(tail.get("end"), int)
        and start <= head["start"] < head["end"] <= end
        and start <= tail["start"] < tail["end"] <= end
    )
    if not valid:
        return ["invalid_relation_evidence"]

    reasons = []
    contained = [
        entity
        for entity in entities
        if isinstance(entity.get("start"), int)
        and isinstance(entity.get("end"), int)
        and start <= entity["start"] < entity["end"] <= end
    ]
    type_counts = Counter(entity.get("type") for entity in contained)
    if end - start > 200:
        reasons.append("evidence_over_200_chars")
    for endpoint_type in sorted(
        {head.get("type"), tail.get("type")}, key=lambda value: str(value)
    ):
        if type_counts[endpoint_type] > 1:
            reasons.append(f"multiple_{endpoint_type}_mentions")
    if relation.get("type") == "exploited_by":
        if POST_EXPLOIT_SEQUENCE.search(evidence) and POST_EXPLOIT_ACTIVITY.search(
            evidence
        ):
            reasons.append("explicit_post_exploitation_sequence")
        elif not EXPLOIT_TRIGGER.search(evidence):
            reasons.append("exploited_by_directness_unresolved")
    return reasons


def _candidate_record(
    relation: dict,
    entities: list[dict],
    reasons: list[str],
) -> dict:
    entity_by_id = {entity.get("id"): entity for entity in entities}
    head = entity_by_id.get(relation.get("head"), {})
    tail = entity_by_id.get(relation.get("tail"), {})
    return {
        "relation_id": relation.get("id"),
        "relation_type": relation.get("type"),
        "head": {
            "id": head.get("id"),
            "type": head.get("type"),
            "text": head.get("text"),
            "normalized_id": head.get("normalized_id"),
            "start": head.get("start"),
            "end": head.get("end"),
        },
        "tail": {
            "id": tail.get("id"),
            "type": tail.get("type"),
            "text": tail.get("text"),
            "normalized_id": tail.get("normalized_id"),
            "start": tail.get("start"),
            "end": tail.get("end"),
        },
        "evidence": relation.get("evidence"),
        "evidence_start": relation.get("evidence_start"),
        "evidence_end": relation.get("evidence_end"),
        "reasons": reasons,
        "status": "manual_review_candidate",
        "included_in_reported_f1": True,
        "graph_ingestible_without_review": False,
    }


def _entity_candidate_record(entity: dict, reasons: list[str]) -> dict:
    return {
        "entity_id": entity.get("id"),
        "entity_type": entity.get("type"),
        "text": entity.get("text"),
        "start": entity.get("start"),
        "end": entity.get("end"),
        "normalized_id": entity.get("normalized_id"),
        "reasons": reasons,
        "status": "normalization_or_span_review_candidate",
        "included_in_reported_f1": True,
        "graph_ingestible_without_review": False,
    }


def _audit_document(
    document: dict,
) -> tuple[Counter, list[dict], list[dict]]:
    counts = Counter()
    text = str(document.get("text", ""))
    entities = document.get("entities", [])
    relations = document.get("relations", [])
    entity_by_id = {entity.get("id"): entity for entity in entities}

    counts["documents"] = 1
    counts["entities"] = len(entities)
    counts["relations"] = len(relations)
    entity_candidates = []
    for entity in entities:
        entity_reasons = []
        entity_type = entity.get("type")
        if entity_type not in EXTRACTION_ENTITY_TYPES:
            counts["illegal_or_mapping_entities"] += 1
        start = entity.get("start")
        end = entity.get("end")
        exact_span = (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
            and text[start:end] == entity.get("text")
        )
        if exact_span:
            counts["entities_with_exact_span"] += 1
        else:
            counts["invalid_entity_spans"] += 1
            entity_reasons.append("invalid_entity_span")
        if str(entity.get("normalized_id") or "").strip():
            counts["entities_with_normalized_id"] += 1
        else:
            counts["entities_without_normalized_id"] += 1
            entity_reasons.append("missing_normalized_id")
        if entity_type == "Weakness" and not _weakness_has_local_explicit_cwe(
            entity, text
        ):
            entity_reasons.append("weakness_without_local_explicit_cwe")
        if (
            entity_type == "Configuration"
            and not str(entity.get("text") or "").casefold().startswith("cpe:2.3:")
            and CONFIGURATION_RELEASE_SUFFIX.search(str(entity.get("text") or ""))
            and not _allowed_non_wildcard_cpe(str(entity.get("text") or ""))
        ):
            entity_reasons.append("configuration_release_bearing_span")
        if entity_reasons:
            entity_candidates.append(
                _entity_candidate_record(entity, entity_reasons)
            )
            counts["review_candidate_entities"] += 1
            for reason in entity_reasons:
                counts[f"entity_candidate_reason:{reason}"] += 1

    candidates = []
    for relation in relations:
        relation_type = relation.get("type")
        head = entity_by_id.get(relation.get("head"))
        tail = entity_by_id.get(relation.get("tail"))
        expected = EXTRACTION_RELATION_ARGUMENT_TYPES.get(relation_type)
        actual = (
            head.get("type") if head else None,
            tail.get("type") if tail else None,
        )
        if relation_type not in EXTRACTION_RELATION_TYPES:
            counts["illegal_or_mapping_relations"] += 1
        elif actual != expected:
            counts["illegal_relation_arguments"] += 1
        else:
            counts["schema_valid_relations"] += 1

        reasons = _evidence_reasons(entities, relation, text)
        if reasons:
            candidates.append(_candidate_record(relation, entities, reasons))
            counts["review_candidate_relations"] += 1
            for reason in reasons:
                counts[f"candidate_reason:{reason}"] += 1
        else:
            counts["structural_evidence_compliant_relations"] += 1

    postprocess = document.get("postprocess") or {}
    rejected = postprocess.get("rejected") or {}
    counts["reported_postprocess_rejections"] += sum(rejected.values())
    counts["reported_deterministic_relation_additions"] += sum(
        value
        for reason, value in rejected.items()
        if reason.startswith("relation_") and reason.endswith("_added")
    )
    return counts, entity_candidates, candidates


def _audit_method(method: str, candidates_root: Path | None) -> dict:
    prediction_dir = EXP_DIR / "results" / "raw_predictions" / f"v5_{method}"
    result_path = EXP_DIR / "results" / f"v5_{method}_test.json"
    if not prediction_dir.is_dir() or not result_path.is_file():
        raise FileNotFoundError(
            f"缺少 {method} 的预测目录或测试结果：{prediction_dir}, {result_path}"
        )

    aggregate = Counter()
    candidate_documents = []
    prediction_paths = sorted(prediction_dir.glob("*.json"))
    for path in prediction_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        counts, entity_candidates, relation_candidates = _audit_document(
            document
        )
        aggregate.update(counts)
        if not entity_candidates and not relation_candidates:
            continue
        candidate_document = {
            "doc_id": document.get("doc_id") or path.stem,
            "method": method,
            "construction_layer": "text_extraction",
            "source_prediction": str(path),
            "source_prediction_sha256": _sha256(path),
            "candidate_count": len(entity_candidates) + len(relation_candidates),
            "entity_candidates": entity_candidates,
            "relation_candidates": relation_candidates,
        }
        candidate_documents.append(candidate_document)
        if candidates_root is not None:
            output_dir = candidates_root / method
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / path.name).write_text(
                json.dumps(candidate_document, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    entities = aggregate["entities"]
    relations = aggregate["relations"]
    schema_contract_passed = not any(
        aggregate[name]
        for name in (
            "illegal_or_mapping_entities",
            "illegal_or_mapping_relations",
            "illegal_relation_arguments",
            "invalid_entity_spans",
        )
    )
    normalization_contract_passed = not aggregate[
        "entities_without_normalized_id"
    ]
    return {
        "method": method,
        "prediction_dir": str(prediction_dir),
        "result_file": str(result_path),
        "result_file_sha256": _sha256(result_path),
        "documents": aggregate["documents"],
        "entities": entities,
        "relations": relations,
        "entity_exact_span_rate": _rate(
            aggregate["entities_with_exact_span"], entities
        ),
        "entity_normalized_id_completeness": _rate(
            aggregate["entities_with_normalized_id"], entities
        ),
        "relation_schema_valid_rate": _rate(
            aggregate["schema_valid_relations"], relations
        ),
        "relation_structural_evidence_compliance_rate": _rate(
            aggregate["structural_evidence_compliant_relations"], relations
        ),
        "review_candidate_relations": aggregate[
            "review_candidate_relations"
        ],
        "review_candidate_entities": aggregate[
            "review_candidate_entities"
        ],
        "review_candidate_documents": len(candidate_documents),
        "candidate_reasons": {
            key.split(":", 1)[1]: value
            for key, value in sorted(aggregate.items())
            if key.startswith("candidate_reason:")
        },
        "entity_candidate_reasons": {
            key.split(":", 1)[1]: value
            for key, value in sorted(aggregate.items())
            if key.startswith("entity_candidate_reason:")
        },
        "reported_postprocess_rejections": aggregate[
            "reported_postprocess_rejections"
        ],
        "reported_deterministic_relation_additions": aggregate[
            "reported_deterministic_relation_additions"
        ],
        "schema_contract_passed": schema_contract_passed,
        "normalization_contract_passed": normalization_contract_passed,
        "mapping_layer_separated": not (
            aggregate["illegal_or_mapping_entities"]
            or aggregate["illegal_or_mapping_relations"]
        ),
        "graph_ingestible_without_review": not candidate_documents,
        "reported_metrics": {
            "entity_f1": result.get("entity", {}).get("f1"),
            "strict_relation_f1": result.get("relation", {}).get("f1"),
            "normalized_fact_f1": result.get("normalized_relation", {}).get(
                "f1"
            ),
            "schema_consistency_rate": result.get("scr", {}).get(
                "overall_scr"
            ),
        },
    }


def build_report(
    methods: list[str], candidates_root: Path | None = None
) -> dict:
    status_path = EXP_DIR / "data" / "review_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    method_reports = [
        _audit_method(method, candidates_root) for method in methods
    ]
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "audit_kind": "chapter3_two_layer_experiment_contract",
        "changes_frozen_predictions": False,
        "changes_reported_f1": False,
        "candidate_output_dir": (
            str(candidates_root) if candidates_root is not None else None
        ),
        "contract": {
            "text_extraction_entities": sorted(EXTRACTION_ENTITY_TYPES),
            "text_extraction_relations": sorted(EXTRACTION_RELATION_TYPES),
            "mapping_relations": sorted(MAPPING_RELATION_TYPES),
            "direct_text_evidence_required": True,
            "offset_and_endpoint_compliance_is_not_semantic_directness": True,
            "mapping_output_evaluated_separately": True,
            "schema_validation_cannot_confirm_unsupported_cooccurrence": True,
        },
        "methods": method_reports,
        "formal_gate": {
            "human_reannotation_complete": status.get(
                "human_reannotation_complete"
            ),
            "adjudication_complete": status.get("adjudication_complete"),
            "human_iaa_complete": status.get("human_iaa_complete"),
            "formal_experiment_ready": status.get("formal_experiment_ready"),
            "blocker": status.get("formal_experiment_blocker"),
        },
        "interpretation": {
            "reported_f1": (
                "保留旧冻结结果用于追溯，不因本审计回写；边界复裁后必须在"
                "新冻结 Gold 上重新评价。"
            ),
            "review_candidates": (
                "证据区间无效、超过200字符或包含多个同类型端点的预测关系；"
                "进入构图前需要人工复核或更窄证据定位。"
            ),
            "deterministic_relation_additions": (
                "Full 后处理若报告 *_added，论文应将其表述为基于直接证据的"
                "确定性关系补取，而不能全部归因于模式校验。"
            ),
            "formal_status": (
                "既有结果属于复裁前的历史受控结果；边界复裁与人工 IAA 均"
                "阻断当前正式实验门禁。"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--candidates-dir",
        default=str(DEFAULT_CANDIDATES),
    )
    args = parser.parse_args()

    output = Path(args.output)
    candidates_root = Path(args.candidates_dir)
    report = build_report(args.methods, candidates_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
