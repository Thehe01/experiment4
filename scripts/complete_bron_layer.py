"""第三章上层关系补全。

输入为文本抽取层实体与关系。程序依据 ATT&CK 的技术—战术映射补全
AttackTechnique→AttackTactic，再依据第三章表 3-3 的对齐规则补全
AttackTactic→KillChainPhase。CAPEC 不再参与第四章实验。
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
EXP_DIR = Path(__file__).resolve().parents[1]
PAPER_KG = EXP_DIR.parents[1] / "data" / "paper_kg"
SCHEMA_VERSION = "chapter3-no-capec-v1"
EXTRACTION_LAYER = "text_extraction"
MAPPING_LAYER = "mapping_completion"
ATTACK_VERSION = "Enterprise ATT&CK v19"
ATTACK_SNAPSHOT_DATE = "2026-05-06"
ATTACK_SOURCE = (
    "https://github.com/mitre/cti/tree/master/enterprise-attack"
)
ALIGNMENT_VERSION = "chapter3-tactic-kill-chain-v1"
TECHNIQUE_TACTIC_PATH = PAPER_KG / "technique_tactic_map.json"
TACTIC_ID_PATH = PAPER_KG / "tactic_id_name_map.json"
EXPECTED_TECHNIQUE_TACTIC_SHA256 = (
    "5b7da21efff5e663cf08f12cab87948af4ac2d4432dff3ec3e84eaa8cc856180"
)
EXPECTED_TACTIC_ID_SHA256 = (
    "640f4bda2e9258510e7d4955c49ca0d46311d8949649b9bcb25ad5037e50a88c"
)

TECHNIQUE_TACTIC = json.loads(
    TECHNIQUE_TACTIC_PATH.read_text(encoding="utf-8")
)
TACTIC_ID = json.loads(
    TACTIC_ID_PATH.read_text(encoding="utf-8")
)

TACTIC_PHASE = {
    "reconnaissance": "KC-RECONNAISSANCE",
    "resource-development": "KC-WEAPONIZATION",
    "initial-access": "KC-DELIVERY",
    "execution": "KC-EXPLOITATION",
    "persistence": "KC-INSTALLATION",
    "privilege-escalation": "KC-INSTALLATION",
    "defense-evasion": "KC-INSTALLATION",
    "defense-impairment": "KC-INSTALLATION",
    "stealth": "KC-INSTALLATION",
    "credential-access": "KC-ACTIONS-ON-OBJECTIVES",
    "discovery": "KC-ACTIONS-ON-OBJECTIVES",
    "lateral-movement": "KC-ACTIONS-ON-OBJECTIVES",
    "collection": "KC-ACTIONS-ON-OBJECTIVES",
    "command-and-control": "KC-COMMAND-AND-CONTROL",
    "exfiltration": "KC-ACTIONS-ON-OBJECTIVES",
    "impact": "KC-ACTIONS-ON-OBJECTIVES",
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_mapping_sources():
    """拒绝使用已偏离当前实验快照的外部映射文件。"""
    actual = {
        "technique_tactic_map": _sha256(TECHNIQUE_TACTIC_PATH),
        "tactic_id_name_map": _sha256(TACTIC_ID_PATH),
    }
    expected = {
        "technique_tactic_map": EXPECTED_TECHNIQUE_TACTIC_SHA256,
        "tactic_id_name_map": EXPECTED_TACTIC_ID_SHA256,
    }
    mismatches = {
        name: {"expected": expected[name], "actual": digest}
        for name, digest in actual.items()
        if digest != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"映射来源已偏离 v5 冻结快照：{mismatches}")
    return actual


def mapping_metadata():
    """返回补全结果必须随实验记录保存的版本和来源。"""
    hashes = validate_mapping_sources()
    return {
        "schema_version": SCHEMA_VERSION,
        "output_layer": MAPPING_LAYER,
        "text_extraction_metrics_excluded": True,
        "attack_version": ATTACK_VERSION,
        "attack_snapshot_date": ATTACK_SNAPSHOT_DATE,
        "attack_source": ATTACK_SOURCE,
        "technique_tactic_map_sha256": hashes["technique_tactic_map"],
        "tactic_id_name_map_sha256": hashes["tactic_id_name_map"],
        "tactic_phase_alignment_version": ALIGNMENT_VERSION,
        "tactic_phase_alignment_kind": "study-defined contextual alignment",
        "tactic_phase_alignment_reference": "Chapter 3 Table 3-3",
    }


def _technique_id(entity):
    value = str(entity.get("normalized_id") or entity.get("text") or "").upper()
    match = re.search(r"\bT\d{4}(?:\.\d{3})?\b", value)
    return match.group(0) if match else None


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def complete(entities, relations=None):
    """返回补全产生的新实体和新关系，不改写抽取层输入。

    Initial Access 通常对齐投递阶段；当某技术已作为 exploited_by 的尾实体
    与具体漏洞相连时，说明其在该文档中承担直接利用行为，因而对齐利用阶段。
    """
    validate_mapping_sources()
    relations = relations or []
    entity_by_id = {entity.get("id"): entity for entity in entities}
    exploited_context = collections.defaultdict(
        lambda: {
            "relation_ids": [],
            "vulnerability_entity_ids": [],
            "vulnerability_normalized_ids": [],
        }
    )
    for relation in relations:
        if relation.get("type") != "exploited_by":
            continue
        technique_id = _technique_id(
            entity_by_id.get(relation.get("tail"), {})
        )
        if not technique_id:
            continue
        vulnerability = entity_by_id.get(relation.get("head"), {})
        context = exploited_context[technique_id]
        context["relation_ids"].append(relation.get("id"))
        context["vulnerability_entity_ids"].append(vulnerability.get("id"))
        context["vulnerability_normalized_ids"].append(
            vulnerability.get("normalized_id")
        )

    technique_mentions = collections.defaultdict(list)
    for entity in entities:
        if entity.get("type") != "AttackTechnique":
            continue
        technique_id = _technique_id(entity)
        if technique_id:
            technique_mentions[technique_id].append(entity)
    for mentions in technique_mentions.values():
        mentions.sort(
            key=lambda item: (
                item.get("start") if isinstance(item.get("start"), int) else 10**18,
                str(item.get("id", "")),
            )
        )
    new_entities = []
    new_relations = []
    tactic_entity_ids = {}
    phase_entity_ids = {}
    relation_by_key = {}
    counter = [10000]

    def new_id(prefix):
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def _merge_unique(target, values):
        for value in values:
            if value is not None and value not in target:
                target.append(value)

    def add_relation(relation_type, head, tail, evidence, provenance):
        key = (relation_type, head, tail)
        if key in relation_by_key:
            current = relation_by_key[key]["provenance"]
            for field in (
                "source_attack_technique_entity_ids",
                "supporting_technique_ids",
                "supporting_implies_relation_ids",
                "supporting_extraction_relation_ids",
                "supporting_vulnerability_entity_ids",
                "supporting_vulnerability_normalized_ids",
            ):
                _merge_unique(
                    current.setdefault(field, []),
                    provenance.get(field, []),
                )
            return relation_by_key[key]
        relation = {
            "id": new_id("R"),
            "type": relation_type,
            "head": head,
            "tail": tail,
            "evidence": evidence,
            "construction_layer": MAPPING_LAYER,
            "has_text_evidence": False,
            "confidence_status": provenance["confidence_status"],
            "provenance": provenance,
        }
        relation_by_key[key] = relation
        new_relations.append(relation)
        return relation

    for technique, mentions in sorted(technique_mentions.items()):
        technique_entity_id = mentions[0]["id"]
        technique_entity_ids = [item["id"] for item in mentions]
        tactics = (
            TECHNIQUE_TACTIC.get(technique)
            or TECHNIQUE_TACTIC.get(technique.split(".")[0])
            or []
        )
        for tactic in _as_list(tactics):
            tactic_id = TACTIC_ID.get(tactic)
            if not tactic_id:
                continue
            if tactic_id not in tactic_entity_ids:
                tactic_entity_ids[tactic_id] = new_id("M")
                new_entities.append({
                    "id": tactic_entity_ids[tactic_id],
                    "text": tactic.replace("-", " ").title(),
                    "type": "AttackTactic",
                    "start": None,
                    "end": None,
                    "normalized_id": tactic_id,
                    "construction_layer": MAPPING_LAYER,
                    "span_applicable": False,
                    "confidence_status": "source_confirmed",
                    "provenance": {
                        "kind": "official_attack_mapping",
                        "source": ATTACK_SOURCE,
                        "source_version": ATTACK_VERSION,
                        "source_snapshot_date": ATTACK_SNAPSHOT_DATE,
                        "mapping_table_sha256": EXPECTED_TACTIC_ID_SHA256,
                    },
                    "notes": (
                        f"{ATTACK_VERSION} technique-tactic mapping; "
                        f"snapshot {ATTACK_SNAPSHOT_DATE}"
                    ),
                })
            implies_relation = add_relation(
                "implies",
                technique_entity_id,
                tactic_entity_ids[tactic_id],
                (
                    f"{ATTACK_VERSION} mapping "
                    f"({ATTACK_SNAPSHOT_DATE}): {technique} -> {tactic_id}"
                ),
                {
                    "kind": "official_attack_mapping",
                    "source": ATTACK_SOURCE,
                    "source_version": ATTACK_VERSION,
                    "source_snapshot_date": ATTACK_SNAPSHOT_DATE,
                    "mapping_table_sha256": EXPECTED_TECHNIQUE_TACTIC_SHA256,
                    "technique_id": technique,
                    "tactic_id": tactic_id,
                    "source_attack_technique_entity_ids": technique_entity_ids,
                    "confidence_status": "source_confirmed",
                },
            )

            phase = TACTIC_PHASE.get(tactic)
            alignment_context = "default_tactic_alignment"
            reason = (
                f"{ALIGNMENT_VERSION}: {tactic} -> {phase}; "
                "study-defined semantic alignment"
            )
            if tactic == "initial-access" and technique in exploited_context:
                phase = "KC-EXPLOITATION"
                alignment_context = "initial_access_exploitation"
                reason = (
                    f"{ALIGNMENT_VERSION}: initial-access technique is the "
                    "target of exploited_by -> KC-EXPLOITATION; "
                    "study-defined contextual alignment"
                )
            if not phase:
                continue
            if phase not in phase_entity_ids:
                phase_entity_ids[phase] = new_id("M")
                new_entities.append({
                    "id": phase_entity_ids[phase],
                    "text": phase,
                    "type": "KillChainPhase",
                    "start": None,
                    "end": None,
                    "normalized_id": phase,
                    "construction_layer": MAPPING_LAYER,
                    "span_applicable": False,
                    "confidence_status": "rule_derived",
                    "provenance": {
                        "kind": "study_defined_alignment",
                        "alignment_rule_version": ALIGNMENT_VERSION,
                        "alignment_reference": "Chapter 3 Table 3-3",
                    },
                    "notes": (
                        f"{ALIGNMENT_VERSION}; "
                        "study-defined contextual alignment"
                    ),
                })
            context = exploited_context.get(technique, {})
            add_relation(
                "belongs_to_phase",
                tactic_entity_ids[tactic_id],
                phase_entity_ids[phase],
                reason,
                {
                    "kind": "study_defined_alignment",
                    "alignment_rule_version": ALIGNMENT_VERSION,
                    "alignment_reference": "Chapter 3 Table 3-3",
                    "alignment_context": alignment_context,
                    "tactic_shortname": tactic,
                    "tactic_id": tactic_id,
                    "phase_id": phase,
                    "supporting_technique_ids": [technique],
                    "supporting_implies_relation_ids": [
                        implies_relation["id"]
                    ],
                    "supporting_extraction_relation_ids": context.get(
                        "relation_ids", []
                    ),
                    "supporting_vulnerability_entity_ids": context.get(
                        "vulnerability_entity_ids", []
                    ),
                    "supporting_vulnerability_normalized_ids": context.get(
                        "vulnerability_normalized_ids", []
                    ),
                    "confidence_status": "rule_derived",
                },
            )
    return new_entities, new_relations


def demo(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    new_entities, new_relations = complete(
        data.get("entities", []), data.get("relations", [])
    )
    print(f"文档：{Path(path).stem}")
    print(
        "补全实体：",
        dict(collections.Counter(entity["type"] for entity in new_entities)),
    )
    print(
        "补全关系：",
        dict(collections.Counter(relation["type"] for relation in new_relations)),
    )
    print("映射元数据：", mapping_metadata())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("用法：python complete_bron_layer.py <prediction.json>")
    demo(sys.argv[1])
