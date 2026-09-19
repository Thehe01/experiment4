"""Audit the v5 package's Gold strategy and semantic gate status.

The audit strictly enforces:
1. hard_errors == 0 (no invalid spans, missing metadata, dangling/invalid relation schema, or orphan configurations).
2. blocking_semantic_errors == 0 (no unsupported descriptive Weakness, no
   release-bearing Configuration span, no unresolved exploited_by directness,
   and no explicit transition from exploitation to a later activity).
3. review_candidates == 0 (unresolved endpoint/evidence ambiguity cannot enter a frozen dataset).
4. gate_status == 'gate_passed' iff all three conditions hold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from schema import (
    ANNOTATION_PROTOCOL_VERSION,
    BOUNDARY_CONTRACT_VERSION,
    EXTRACTION_RELATION_ARGUMENT_TYPES,
    SCHEMA_VERSION,
)


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"
GUIDELINE = EXP_DIR / "ANNOTATION_GUIDELINE.md"
GUIDELINE_ENTRY = GOLD_DIR / "ANNOTATION_GUIDELINE.md"
PROTOCOL_VERSION = ANNOTATION_PROTOCOL_VERSION
ALLOWED_SPLITS = ("train", "dev", "test")

SCOPE_BASIS_MARKERS = (
    "same sentence", "same clause", "self-contained clause", "same table",
    "same row", "same cell", "same block", "explicit verb", "verb binding",
    "scope basis", "list scope", "governing list", "明确", "主谓", "表格", "列表", "支配",
)
EXPLOIT_TRIGGER = re.compile(
    r"\b(?:exploit(?:ed|ing|s|ation)?|leverag(?:e|ed|ing)|trigger(?:ed|ing|s)?)\b|"
    r"\b(?:use(?:d|s)?|using)\b.{0,90}\b(?:CVE(?:s)?|vulnerabilit(?:y|ies))\b|"
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
    r"(?:\s|[-_(])(?:v(?:ersion)?\s*)?\d+(?:\.\d+){1,}(?:\b|\))|"
    r"\b(?:build|patch|release)\s*[-_:]?\s*\d+|"
    r"\bversions?\s+\d+|\bColdFusion\s+(?:11|2016|2018|2021)\b",
    re.IGNORECASE,
)
CONFIGURATION_GENERIC_SUFFIX = re.compile(
    r"(?:\b(?:software\s+)?library|\bemail\s+servers?|\bwebmail\s+clients?|"
    r"\bGateway\s+appliances?|\bSMA\s+100\s+Series\s+Appliances|"
    r"\bSD-WAN\s+WANOP\s+appliance|"
    r"\bweb\s+application\s+delivery\s+control\s*\(ADC\))$",
    re.IGNORECASE,
)


def _cpe_version(value: object) -> str | None:
    parts = str(value or "").split(":")
    if len(parts) != 13 or parts[:2] != ["cpe", "2.3"]:
        return None
    return parts[5]


def _allowed_non_wildcard_cpe(surface: str) -> bool:
    return surface.casefold().startswith("cpe:2.3:") or bool(
        re.fullmatch(r"SMB\s+version\s+1", surface, re.IGNORECASE)
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized(value: object) -> str:
    text = str(value or "").strip()
    if text.casefold().startswith("cpe:2.3:"):
        return text.casefold()
    return "".join(text.split()).upper()


def _relation_fact(relation: dict, entity_by_id: dict[str, dict]) -> tuple:
    head = entity_by_id[relation["head"]]
    tail = entity_by_id[relation["tail"]]
    return (
        relation["type"],
        head["type"],
        _normalized(head.get("normalized_id")),
        tail["type"],
        _normalized(tail.get("normalized_id")),
    )


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
    surface = str(entity.get("text") or "")
    if re.search(r"CWE-\s*\d+", surface, re.IGNORECASE):
        return True
    normalized_id = _canonical_cwe(entity.get("normalized_id"))
    start, end = entity.get("start"), entity.get("end")
    if not normalized_id or not isinstance(start, int) or not isinstance(end, int):
        return False
    block = _local_fact_block(text, start, end)
    local_ids = {
        _canonical_cwe(re.sub(r"\s+", "", match.group(0)))
        for match in re.finditer(r"CWE-\s*\d+", block, re.IGNORECASE)
    }
    return normalized_id in local_ids


def _explicit_post_exploitation_sequence(evidence: str) -> bool:
    return bool(
        POST_EXPLOIT_SEQUENCE.search(evidence)
        and POST_EXPLOIT_ACTIVITY.search(evidence)
    )


def _has_scope_basis(relation: dict) -> bool:
    """Return whether adjudication records an explicit source scope.

    v4.5 permits a long or enumerated evidence block only when a source-
    grounded adjudication basis records the sentence/table/list scope.  An
    empty or generic note therefore remains a blocking ambiguity; the basis
    itself is preserved verbatim in the audit report for manual inspection.
    """
    basis = str(relation.get("adjudication_basis", "")).strip().casefold()
    return bool(basis) and any(marker in basis for marker in SCOPE_BASIS_MARKERS)


def audit_document(document: dict, split_name: str) -> dict:
    text = str(document.get("text", ""))
    entities = list(document.get("entities", []))
    relations = list(document.get("relations", []))
    entity_by_id = {str(entity.get("id")): entity for entity in entities}
    hard_errors: list[dict] = []
    blocking_semantic_errors: list[dict] = []
    review_candidates: list[dict] = []
    resolved_candidates: list[dict] = []
    informational_flags: list[dict] = []
    counters = Counter()

    # 1. Metadata and Protocol version check
    doc_protocol = document.get("annotation_protocol_version")
    if doc_protocol != PROTOCOL_VERSION:
        hard_errors.append({
            "kind": "invalid_protocol_version",
            "expected": PROTOCOL_VERSION,
            "actual": doc_protocol,
        })
    if document.get("schema_version") != SCHEMA_VERSION:
        hard_errors.append({
            "kind": "invalid_schema_version",
            "expected": SCHEMA_VERSION,
            "actual": document.get("schema_version"),
        })
    if not document.get("annotation_status"):
        hard_errors.append({"kind": "missing_annotation_status"})
    if document.get("boundary_contract_version") != BOUNDARY_CONTRACT_VERSION:
        hard_errors.append({
            "kind": "invalid_boundary_contract_version",
            "expected": BOUNDARY_CONTRACT_VERSION,
            "actual": document.get("boundary_contract_version"),
        })

    # 2. Entity audit
    seen_spans: set[tuple] = set()
    for entity in entities:
        counters["entities"] += 1
        start = entity.get("start")
        end = entity.get("end")
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start <= end <= len(text)
            and text[start:end] == entity.get("text")
        ):
            hard_errors.append({
                "kind": "invalid_entity_span",
                "entity_id": entity.get("id"),
            })
        span_key = (entity.get("type"), start, end)
        if span_key in seen_spans:
            hard_errors.append({
                "kind": "duplicate_entity_span",
                "entity_id": entity.get("id"),
            })
        seen_spans.add(span_key)
        if entity.get("type") == "Configuration":
            counters["configuration"] += 1
            if not str(entity.get("normalized_id", "")).casefold().startswith(
                "cpe:2.3:"
            ):
                hard_errors.append({
                    "kind": "configuration_without_cpe",
                    "entity_id": entity.get("id"),
                })
            occurrences = len(re.findall(
                re.escape(str(entity.get("text", ""))), text
            ))
            if occurrences > 1:
                counters["configuration_repeated_surface"] += 1
                informational_flags.append({
                    "kind": "repeated_configuration_surface",
                    "entity_id": entity.get("id"),
                    "surface": entity.get("text"),
                    "surface_occurrences": occurrences,
                    "selected_start": start,
                })
            surface = str(entity.get("text", ""))
            if (
                not surface.casefold().startswith("cpe:2.3:")
                and CONFIGURATION_RELEASE_SUFFIX.search(surface)
                and not _allowed_non_wildcard_cpe(surface)
            ):
                counters["configuration_release_bearing_span"] += 1
                blocking_semantic_errors.append({
                    "kind": "configuration_release_bearing_span",
                    "entity_id": entity.get("id"),
                    "surface": surface,
                    "normalized_id": entity.get("normalized_id"),
                })
            if CONFIGURATION_GENERIC_SUFFIX.search(surface):
                counters["configuration_generic_suffix_span"] += 1
                blocking_semantic_errors.append({
                    "kind": "configuration_generic_suffix_span",
                    "entity_id": entity.get("id"),
                    "surface": surface,
                    "normalized_id": entity.get("normalized_id"),
                })
            cpe_version = _cpe_version(entity.get("normalized_id"))
            if (
                cpe_version not in {None, "*", "-"}
                and not _allowed_non_wildcard_cpe(surface)
            ):
                counters["configuration_unapproved_non_wildcard_cpe"] += 1
                blocking_semantic_errors.append({
                    "kind": "configuration_unapproved_non_wildcard_cpe",
                    "entity_id": entity.get("id"),
                    "surface": surface,
                    "normalized_id": entity.get("normalized_id"),
                })
        if entity.get("type") == "Weakness":
            counters["weakness"] += 1
            if not _weakness_has_local_explicit_cwe(entity, text):
                counters["weakness_without_local_explicit_cwe"] += 1
                blocking_semantic_errors.append({
                    "kind": "weakness_without_local_explicit_cwe",
                    "entity_id": entity.get("id"),
                    "surface": entity.get("text"),
                    "normalized_id": entity.get("normalized_id"),
                    "local_fact_block": (
                        _local_fact_block(text, start, end)[:300]
                        if isinstance(start, int) and isinstance(end, int)
                        else ""
                    ),
                })

    configuration_tails = {
        str(relation.get("tail"))
        for relation in relations
        if relation.get("type") == "affects"
    }
    for entity in entities:
        if (
            entity.get("type") == "Configuration"
            and str(entity.get("id")) not in configuration_tails
        ):
            hard_errors.append({
                "kind": "orphan_configuration",
                "entity_id": entity.get("id"),
            })

    # 3. Relation audit & Strict Semantic Gate checks
    seen_facts: set[tuple] = set()
    seen_relation_mentions: set[tuple] = set()
    for relation in relations:
        counters["relations"] += 1
        relation_type = str(relation.get("type"))
        counters[f"relation:{relation_type}"] += 1
        head = entity_by_id.get(str(relation.get("head")))
        tail = entity_by_id.get(str(relation.get("tail")))
        if head is None or tail is None:
            hard_errors.append({
                "kind": "dangling_relation",
                "relation_id": relation.get("id"),
            })
            continue
        expected = EXTRACTION_RELATION_ARGUMENT_TYPES.get(relation_type)
        if expected != (head.get("type"), tail.get("type")):
            hard_errors.append({
                "kind": "invalid_relation_schema",
                "relation_id": relation.get("id"),
            })
        
        mention_key = (relation_type, str(relation.get("head")), str(relation.get("tail")))
        if mention_key in seen_relation_mentions:
            hard_errors.append({
                "kind": "duplicate_relation_mention",
                "relation_id": relation.get("id"),
            })
        seen_relation_mentions.add(mention_key)

        fact = _relation_fact(relation, entity_by_id)
        if fact in seen_facts:
            counters["duplicate_normalized_fact"] += 1
        seen_facts.add(fact)

        start = relation.get("evidence_start")
        end = relation.get("evidence_end")
        evidence = str(relation.get("evidence", ""))
        valid_evidence = (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end <= len(text)
            and text[start:end] == evidence
            and start <= head["start"] < head["end"] <= end
            and start <= tail["start"] < tail["end"] <= end
        )
        if not valid_evidence:
            hard_errors.append({
                "kind": "invalid_relation_evidence",
                "relation_id": relation.get("id"),
            })
            continue

        contained = [
            entity for entity in entities
            if start <= entity["start"] and entity["end"] <= end
        ]
        type_counts = Counter(entity["type"] for entity in contained)
        evidence_length = end - start
        
        # Strict Semantic Gate checks on exploited_by
        if relation_type == "exploited_by":
            vulns_in_span = [e for e in contained if e["type"] == "Vulnerability"]
            has_scope_basis = _has_scope_basis(relation)
            endpoint_ambiguity = any(
                type_counts[entity_type] > 1
                for entity_type in ("Vulnerability", "AttackTechnique")
            )
            if len(vulns_in_span) > 1 and not has_scope_basis:
                blocking_semantic_errors.append({
                    "kind": "multi_vuln_exploited_by_evidence",
                    "relation_id": relation.get("id"),
                    "vuln_count": len(vulns_in_span),
                    "vulns": [v["text"] for v in vulns_in_span],
                    "evidence_length": evidence_length,
                    "evidence_preview": evidence[:150],
                })
            if evidence_length > 200 and endpoint_ambiguity and not has_scope_basis:
                blocking_semantic_errors.append({
                    "kind": "overlong_exploited_by_evidence",
                    "relation_id": relation.get("id"),
                    "evidence_length": evidence_length,
                    "evidence_preview": evidence[:150],
                })
            if _explicit_post_exploitation_sequence(evidence):
                blocking_semantic_errors.append({
                    "kind": "explicit_post_exploitation_sequence",
                    "relation_id": relation.get("id"),
                    "technique": tail.get("normalized_id"),
                    "evidence_length": evidence_length,
                    "evidence_preview": evidence[:150],
                })
            boundary_basis = str(
                relation.get("adjudication_basis", "")
            ).casefold()
            if (
                BOUNDARY_CONTRACT_VERSION.casefold() not in boundary_basis
                or not has_scope_basis
            ):
                counters["exploited_by_boundary_recertification_required"] += 1
                review_candidates.append({
                    "kind": "exploited_by_boundary_recertification_required",
                    "relation_id": relation.get("id"),
                    "technique": tail.get("normalized_id"),
                    "evidence_length": evidence_length,
                    "previous_basis": relation.get("adjudication_basis", ""),
                    "evidence_preview": evidence[:300],
                })
            if not EXPLOIT_TRIGGER.search(evidence):
                counters["exploited_by_without_direct_trigger_or_scope_basis"] += 1
                review_candidates.append({
                    "kind": "exploited_by_without_direct_trigger_or_scope_basis",
                    "relation_id": relation.get("id"),
                    "technique": tail.get("normalized_id"),
                    "evidence_length": evidence_length,
                    "evidence_preview": evidence[:300],
                })

        reasons = []
        if evidence_length > 200:
            reasons.append("evidence_over_200_chars")
            counters["evidence_over_200_chars"] += 1
        endpoint_types = {head["type"], tail["type"]}
        for entity_type in sorted(endpoint_types):
            if type_counts[entity_type] > 1:
                reasons.append(f"multiple_{entity_type}_mentions")
                counters[f"multiple_{entity_type}_mentions"] += 1
        if reasons:
            counters["ambiguous_relation_evidence"] += 1
            candidate = {
                "kind": "ambiguous_relation_evidence",
                "relation_id": relation.get("id"),
                "relation_type": relation_type,
                "head": head.get("text"),
                "tail": tail.get("text"),
                "evidence_start": start,
                "evidence_end": end,
                "evidence_length": evidence_length,
                "reasons": reasons,
                "basis": relation.get("adjudication_basis", ""),
                "evidence_preview": evidence[:300],
            }
            if str(relation.get("adjudication_basis", "")).strip():
                resolved_candidates.append(candidate)
            else:
                review_candidates.append(candidate)

    return {
        "doc_id": document.get("doc_id"),
        "split": split_name,
        "counts": dict(counters),
        "hard_errors": hard_errors,
        "blocking_semantic_errors": blocking_semantic_errors,
        "review_candidates": review_candidates,
        "resolved_candidates": resolved_candidates,
        "informational_flags": informational_flags,
    }


def build_report(
    split_names: tuple[str, ...],
    split_file: Path = SPLIT_FILE,
) -> dict:
    if any(split_name not in ALLOWED_SPLITS for split_name in split_names):
        raise ValueError(f"Splits must be among {ALLOWED_SPLITS}")
    split_file = Path(split_file).resolve()
    split = json.loads(split_file.read_text(encoding="utf-8"))
    documents = []
    aggregate = Counter()
    hard_error_count = 0
    blocking_semantic_count = 0
    candidate_count = 0
    resolved_candidate_count = 0
    informational_count = 0
    for split_name in split_names:
        for doc_id in split[split_name]:
            path = GOLD_DIR / f"{doc_id}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            result = audit_document(document, split_name)
            documents.append(result)
            aggregate.update(result["counts"])
            hard_error_count += len(result["hard_errors"])
            blocking_semantic_count += len(result["blocking_semantic_errors"])
            candidate_count += len(result["review_candidates"])
            resolved_candidate_count += len(result["resolved_candidates"])
            informational_count += len(result["informational_flags"])

    gate_passed = (
        hard_error_count == 0
        and blocking_semantic_count == 0
        and candidate_count == 0
    )
    gate_status = "gate_passed" if gate_passed else "gate_failed"

    return {
        "gate_status": gate_status,
        "status": "passed" if gate_passed else "failed",
        "schema_version": SCHEMA_VERSION,
        "annotation_protocol_version": PROTOCOL_VERSION,
        "boundary_contract_version": BOUNDARY_CONTRACT_VERSION,
        "scope": list(split_names),
        "split_file": str(split_file.relative_to(EXP_DIR)),
        "split_sha256": _sha256(split_file),
        "guideline": {
            "path": str(GUIDELINE),
            "sha256": _sha256(GUIDELINE),
            "entry_path": str(GUIDELINE_ENTRY),
            "entry_sha256": _sha256(GUIDELINE_ENTRY),
        },
        "summary": {
            "documents": len(documents),
            "gate_status": gate_status,
            "hard_errors": hard_error_count,
            "blocking_semantic_errors": blocking_semantic_count,
            "review_candidates": candidate_count,
            "resolved_review_candidates": resolved_candidate_count,
            "informational_flags": informational_count,
            **dict(aggregate),
        },
        "documents": documents,
        "interpretation": {
            "hard_errors": "must be zero before any experiment",
            "blocking_semantic_errors": "unsupported descriptive Weakness, release-bearing/generic-suffix Configuration surface, unapproved non-wildcard Configuration CPE, unresolved exploited_by scope, or an explicit transition from CVE exploitation to a later activity; strictly zero for gate pass",
            "review_candidates": "unresolved relation evidence candidates; must be zero before formal promotion",
            "resolved_review_candidates": "relation evidence candidates with source-grounded adjudication_basis",
            "informational_flags": "repeated surface mentions retained as non-blocking mention-level information",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits", nargs="+", choices=ALLOWED_SPLITS,
        default=list(ALLOWED_SPLITS),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=SPLIT_FILE,
        help="正式文档切分文件；默认使用 v7 正式切分",
    )
    args = parser.parse_args()
    report = build_report(tuple(args.splits), split_file=args.split_file)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["gate_status"] == "gate_passed" else 1)


if __name__ == "__main__":
    main()

