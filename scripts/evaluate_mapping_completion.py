"""单独评价第四章映射补全层，不把查表边计入文本抽取 F1。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from complete_bron_layer import (
    TACTIC_ID,
    TACTIC_PHASE,
    TECHNIQUE_TACTIC,
    _technique_id,
    complete,
    mapping_metadata,
)


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = EXP_DIR / "results" / "raw_predictions" / "v6_protegi"
DEFAULT_OUTPUT = EXP_DIR / "results" / "v6_protegi_mapping_completion.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(f"{path.name}\0{_sha256(path)}\n".encode("utf-8"))
    return digest.hexdigest()


def _completion_document(
    source_path: Path,
    document: dict,
    new_entities: list[dict],
    new_relations: list[dict],
    technique_ids: set[str],
    mapped_technique_ids: set[str],
) -> dict:
    unmapped = sorted(technique_ids - mapped_technique_ids)
    candidates = [
        {
            "type": "AttackTechnique",
            "normalized_id": technique_id,
            "reason": "not_in_frozen_attack_mapping",
            "status": "mapping_review_candidate",
        }
        for technique_id in unmapped
    ]
    return {
        "doc_id": document.get("doc_id") or source_path.stem,
        "schema_version": document.get("schema_version"),
        "construction_layer": "mapping_completion",
        "text_extraction_metrics_excluded": True,
        "source_extraction": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "entity_count": len(document.get("entities", [])),
            "relation_count": len(document.get("relations", [])),
        },
        "mapping_metadata": mapping_metadata(),
        "entities": new_entities,
        "relations": new_relations,
        "mapping_review_candidates": candidates,
        "review_status": (
            "requires_mapping_review" if candidates else "complete"
        ),
    }


def evaluate(
    input_dir: Path,
    completed_output_dir: Path | None = None,
) -> dict:
    counts = Counter()
    failures = Counter()
    input_paths = sorted(input_dir.glob("*.json"))
    document_summaries = []
    unmapped_techniques = Counter()
    if completed_output_dir is not None:
        completed_output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        entities = document.get("entities", [])
        relations = document.get("relations", [])
        new_entities, new_relations = complete(entities, relations)
        original_entities = {entity["id"]: entity for entity in entities}
        combined_entities = {
            entity["id"]: entity for entity in entities + new_entities
        }
        exploited_technique_ids = {
            technique_id
            for relation in relations
            if relation.get("type") == "exploited_by"
            for technique_id in [
                _technique_id(
                    original_entities.get(relation.get("tail"), {})
                )
            ]
            if technique_id
        }
        technique_ids = {
            _technique_id(entity)
            for entity in entities
            if entity.get("type") == "AttackTechnique"
            and _technique_id(entity)
        }
        counts["documents"] += 1
        counts["normalized_techniques"] += len(technique_ids)

        mapped_technique_ids = {
            _technique_id(combined_entities.get(relation["head"], {}))
            for relation in new_relations
            if relation.get("type") == "implies"
        }
        counts["mapped_techniques"] += len(
            technique_ids & mapped_technique_ids
        )
        missing_techniques = technique_ids - mapped_technique_ids
        unmapped_techniques.update(missing_techniques)
        counts["implies"] += sum(
            relation.get("type") == "implies" for relation in new_relations
        )
        counts["belongs_to_phase"] += sum(
            relation.get("type") == "belongs_to_phase"
            for relation in new_relations
        )

        for entity in new_entities:
            counts["mapping_entities"] += 1
            if (
                entity.get("construction_layer") != "mapping_completion"
                or entity.get("span_applicable") is not False
                or entity.get("start") is not None
                or entity.get("end") is not None
                or not entity.get("provenance")
            ):
                failures["mapping_entity_contract_violation"] += 1

        for relation in new_relations:
            counts["mapping_relations"] += 1
            provenance = relation.get("provenance") or {}
            required = {
                "implies": {
                    "kind",
                    "source",
                    "source_version",
                    "source_snapshot_date",
                    "mapping_table_sha256",
                    "technique_id",
                    "tactic_id",
                    "confidence_status",
                },
                "belongs_to_phase": {
                    "kind",
                    "alignment_rule_version",
                    "alignment_reference",
                    "alignment_context",
                    "tactic_id",
                    "phase_id",
                    "supporting_technique_ids",
                    "confidence_status",
                },
            }.get(relation.get("type"), set())
            if (
                relation.get("construction_layer") != "mapping_completion"
                or relation.get("has_text_evidence") is not False
                or not required.issubset(provenance)
            ):
                failures["mapping_relation_provenance_incomplete"] += 1
            else:
                counts["mapping_relations_with_complete_provenance"] += 1

        for relation in new_relations:
            if relation.get("type") == "implies":
                head = combined_entities.get(relation["head"], {})
                tail = combined_entities.get(relation["tail"], {})
                technique_id = _technique_id(head)
                tactic_id = tail.get("normalized_id")
                allowed_tactics = {
                    TACTIC_ID.get(shortname)
                    for shortname in (
                        TECHNIQUE_TACTIC.get(technique_id, [])
                        or TECHNIQUE_TACTIC.get(
                            (technique_id or "").split(".")[0],
                            [],
                        )
                    )
                }
                counts["source_checks"] += 1
                if tactic_id in allowed_tactics:
                    counts["source_consistent"] += 1
                else:
                    failures["implies_source_mismatch"] += 1
            elif relation.get("type") == "belongs_to_phase":
                tactic = combined_entities.get(relation["head"], {})
                phase = combined_entities.get(relation["tail"], {})
                tactic_id = tactic.get("normalized_id")
                shortname = next(
                    (
                        name
                        for name, identifier in TACTIC_ID.items()
                        if identifier == tactic_id
                    ),
                    None,
                )
                expected_phase = TACTIC_PHASE.get(shortname)
                if shortname == "initial-access":
                    supporting_implies = [
                        item
                        for item in new_relations
                        if item.get("type") == "implies"
                        and item.get("tail") == relation.get("head")
                    ]
                    contextual_exploitation = any(
                        _technique_id(
                            combined_entities.get(item.get("head"), {})
                        )
                        in exploited_technique_ids
                        for item in supporting_implies
                    )
                    allowed_phases = {"KC-DELIVERY"}
                    if contextual_exploitation:
                        allowed_phases.add("KC-EXPLOITATION")
                    expected_phase = sorted(allowed_phases)
                    counts["initial_access_exploitation" if contextual_exploitation else "initial_access_delivery"] += 1
                    counts["initial_access_context_checks"] += 1
                    if phase.get("normalized_id") in allowed_phases:
                        counts["initial_access_context_passed"] += 1
                if (
                    phase.get("normalized_id") not in expected_phase
                    if isinstance(expected_phase, list)
                    else phase.get("normalized_id") != expected_phase
                ):
                    failures["phase_rule_mismatch"] += 1

        completion_document = _completion_document(
            path,
            document,
            new_entities,
            new_relations,
            technique_ids,
            mapped_technique_ids,
        )
        if completed_output_dir is not None:
            (completed_output_dir / path.name).write_text(
                json.dumps(
                    completion_document,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        document_summaries.append({
            "doc_id": completion_document["doc_id"],
            "normalized_techniques": len(technique_ids),
            "mapped_techniques": len(technique_ids & mapped_technique_ids),
            "unmapped_techniques": sorted(missing_techniques),
            "implies": sum(
                relation.get("type") == "implies"
                for relation in new_relations
            ),
            "belongs_to_phase": sum(
                relation.get("type") == "belongs_to_phase"
                for relation in new_relations
            ),
            "review_status": completion_document["review_status"],
        })

    normalized = counts["normalized_techniques"]
    source_checks = counts["source_checks"]
    context_checks = counts["initial_access_context_checks"]
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "input_document_count": len(input_paths),
        "input_aggregate_sha256": _aggregate_sha256(input_paths),
        "completed_output_dir": (
            str(completed_output_dir)
            if completed_output_dir is not None
            else None
        ),
        "evaluation_kind": "generator_self_consistency_and_coverage",
        "independent_oracle": False,
        "interpretation": (
            "attack_source_consistency reuses the generator mapping tables; "
            "it is an implementation consistency check, not independent accuracy"
        ),
        "mapping_metadata": mapping_metadata(),
        "layer_contract": {
            "source_layer": "text_extraction",
            "output_layer": "mapping_completion",
            "separate_outputs": completed_output_dir is not None,
            "included_in_text_extraction_f1": False,
            "mapping_entities_have_text_spans": False,
            "per_relation_provenance_required": True,
        },
        "counts": dict(sorted(counts.items())),
        "technique_tactic_coverage": (
            counts["mapped_techniques"] / normalized
            if normalized
            else 0.0
        ),
        "attack_source_consistency": (
            counts["source_consistent"] / source_checks
            if source_checks
            else 0.0
        ),
        "initial_access_context_evidence_pass_rate": (
            counts["initial_access_context_passed"] / context_checks
            if context_checks
            else 0.0
        ),
        "failures": dict(sorted(failures.items())),
        "unmapped_technique_counts": dict(sorted(unmapped_techniques.items())),
        "documents": document_summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--completed-output-dir",
        help=(
            "可选：把每篇文档的映射补全实体、关系、来源、版本和候选"
            "单独写入此目录；不会改写文本抽取层预测"
        ),
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"缺少Full预测目录：{input_dir}；请先运行正式抽取实验"
        )
    completed_output_dir = (
        Path(args.completed_output_dir)
        if args.completed_output_dir
        else None
    )
    report = evaluate(input_dir, completed_output_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
