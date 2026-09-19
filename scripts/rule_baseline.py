"""第四章无主题泄漏的规则基线。

规则资源仅由冻结划分中的训练集和开发集构建。测试时不读取文档名称、
主题标签或测试集 Gold。文本抽取层严格限定为 4 类实体和 3 类关系。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]
GOLD_DIR = EXP_DIR / "data" / "annotations" / "gold"
SPLIT_FILE = EXP_DIR / "data" / "train_dev_test_split_v7.json"

CVE_PATTERN = re.compile(
    r"\bCVE(?:\s*-\s*|\s+)\d{4}\s*-\s*\d{4,7}\b",
    re.IGNORECASE,
)
CWE_PATTERN = re.compile(r"\bCWE(?:\s*-\s*|\s+)\d+\b", re.IGNORECASE)
TECHNIQUE_PATTERN = re.compile(
    r"\bT\d{4}(?:\s*\.\s*\d{3})?\b",
    re.IGNORECASE,
)
EXPLOIT_TERMS = re.compile(
    r"\b(exploit(?:ed|ing|s|ation)?|leverag(?:e|ed|ing)|"
    r"trigger(?:ed|ing|s)?)\b|利用|漏洞利用|触发",
    re.IGNORECASE,
)
GENERIC_CONFIGURATIONS = {
    "affected system",
    "affected systems",
    "multiple products",
    "various products",
    "software",
    "product",
    "products",
}


def _normalized_identifier(surface: str, prefix: str) -> str:
    digits = re.findall(r"\d+", surface)
    if prefix == "CVE" and len(digits) >= 2:
        return f"CVE-{digits[0]}-{digits[1]}"
    if prefix == "CWE" and digits:
        return f"CWE-{digits[0]}"
    compact = re.sub(r"\s+", "", surface).upper()
    return compact


def _literal_pattern(surface: str) -> re.Pattern[str]:
    escaped = re.escape(surface)
    left = r"(?<!\w)" if surface[:1].isalnum() else ""
    right = r"(?!\w)" if surface[-1:].isalnum() else ""
    return re.compile(left + escaped + right, re.IGNORECASE)


@lru_cache(maxsize=1)
def load_train_dev_lexicon() -> dict[str, list[tuple[str, str | None]]]:
    """只从 train/dev Gold 构建全局词典，不使用测试文档或主题标签。"""
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    values: dict[str, dict[str, Counter]] = {
        "Configuration": defaultdict(Counter),
        "Weakness": defaultdict(Counter),
    }
    surfaces: dict[str, dict[str, str]] = {
        "Configuration": {},
        "Weakness": {},
    }
    for doc_id in split["train"] + split["dev"]:
        annotation = json.loads(
            (GOLD_DIR / f"{doc_id}.json").read_text(encoding="utf-8")
        )
        for entity in annotation.get("entities", []):
            entity_type = entity.get("type")
            if entity_type not in values:
                continue
            surface = str(entity.get("text") or "").strip()
            if len(surface) < 3:
                continue
            if (
                entity_type == "Configuration"
                and surface.lower() in GENERIC_CONFIGURATIONS
            ):
                continue
            key = surface.casefold()
            surfaces[entity_type].setdefault(key, surface)
            values[entity_type][key][entity.get("normalized_id")] += 1

    lexicon: dict[str, list[tuple[str, str | None]]] = {}
    for entity_type, by_surface in values.items():
        entries = []
        for key, normalizations in by_surface.items():
            normalized_id = normalizations.most_common(1)[0][0]
            entries.append((surfaces[entity_type][key], normalized_id))
        lexicon[entity_type] = sorted(
            entries,
            key=lambda item: (-len(item[0]), item[0].casefold()),
        )
    return lexicon


def _identifier_entities(text: str) -> list[dict]:
    entities = []
    specifications = (
        (CVE_PATTERN, "Vulnerability", "CVE"),
        (CWE_PATTERN, "Weakness", "CWE"),
        (TECHNIQUE_PATTERN, "AttackTechnique", "T"),
    )
    for pattern, entity_type, prefix in specifications:
        for match in pattern.finditer(text):
            surface = match.group(0)
            entities.append({
                "text": surface,
                "type": entity_type,
                "start": match.start(),
                "end": match.end(),
                "normalized_id": _normalized_identifier(surface, prefix),
            })
    return entities


def _lexicon_entities(text: str) -> list[dict]:
    entities = []
    lexicon = load_train_dev_lexicon()
    occupied: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for entity_type in ("Configuration", "Weakness"):
        for surface, normalized_id in lexicon[entity_type]:
            for match in _literal_pattern(surface).finditer(text):
                span = (match.start(), match.end())
                if any(
                    not (span[1] <= start or span[0] >= end)
                    for start, end in occupied[entity_type]
                ):
                    continue
                occupied[entity_type].append(span)
                entities.append({
                    "text": match.group(0),
                    "type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "normalized_id": normalized_id,
                })
    return entities


def _sentence_bounds(text: str, start: int) -> tuple[int, int]:
    left = max(
        text.rfind(".", 0, start),
        text.rfind("\n", 0, start),
        text.rfind("。", 0, start),
    )
    endings = [
        position
        for position in (
            text.find(".", start),
            text.find("\n", start),
            text.find("。", start),
        )
        if position >= 0
    ]
    right = min(endings) + 1 if endings else len(text)
    return left + 1, right


def _span_distance(left: dict, right: dict) -> int:
    if left["end"] <= right["start"]:
        return right["start"] - left["end"]
    if right["end"] <= left["start"]:
        return left["start"] - right["end"]
    return 0


def _local_targets(
    source: dict,
    targets: list[dict],
    text: str,
    max_distance: int = 300,
    max_targets: int = 2,
) -> list[dict]:
    sent_start, sent_end = _sentence_bounds(text, source["start"])
    ranked = []
    for target in targets:
        distance = _span_distance(source, target)
        if distance > max_distance:
            continue
        if not (sent_start <= target["start"] < sent_end):
            continue
        ranked.append((distance, target["start"], target))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked[:max_targets]]


def _relation_evidence(text: str, head: dict, tail: dict) -> str:
    start = min(head["start"], tail["start"])
    end = max(head["end"], tail["end"])
    sent_start, _ = _sentence_bounds(text, start)
    _, sent_end = _sentence_bounds(text, end)
    return text[sent_start:sent_end].strip()


def _build_relations(text: str, entities: list[dict]) -> list[dict]:
    vulnerabilities = [
        entity for entity in entities if entity["type"] == "Vulnerability"
    ]
    by_type = {
        entity_type: [
            entity for entity in entities if entity["type"] == entity_type
        ]
        for entity_type in ("Configuration", "Weakness", "AttackTechnique")
    }
    relations = []
    seen = set()

    def add(relation_type: str, head: dict, tail: dict, evidence: str) -> None:
        key = (relation_type, head["id"], tail["id"])
        if key in seen:
            return
        seen.add(key)
        relations.append({
            "id": f"R{len(relations) + 1}",
            "type": relation_type,
            "head": head["id"],
            "tail": tail["id"],
            "evidence": evidence,
        })

    for vulnerability in vulnerabilities:
        for configuration in _local_targets(
            vulnerability, by_type["Configuration"], text
        ):
            add(
                "affects",
                vulnerability,
                configuration,
                _relation_evidence(text, vulnerability, configuration),
            )
        for weakness in _local_targets(
            vulnerability, by_type["Weakness"], text
        ):
            add(
                "instantiates",
                vulnerability,
                weakness,
                _relation_evidence(text, vulnerability, weakness),
            )

        sent_start, sent_end = _sentence_bounds(text, vulnerability["start"])
        sentence = text[sent_start:sent_end]
        if not EXPLOIT_TERMS.search(sentence):
            continue
        for technique in _local_targets(
            vulnerability,
            by_type["AttackTechnique"],
            text,
        ):
            add("exploited_by", vulnerability, technique, sentence.strip())
    return relations


def extract(text: str) -> dict:
    """抽取一篇文档；同一表面文本的每次出现均保留独立字符跨度。"""
    raw_entities = _identifier_entities(text) + _lexicon_entities(text)
    unique = {}
    for entity in raw_entities:
        key = (entity["start"], entity["end"], entity["type"])
        unique.setdefault(key, entity)
    entities = sorted(
        unique.values(),
        key=lambda item: (item["start"], item["end"], item["type"]),
    )
    for index, entity in enumerate(entities, start=1):
        entity["id"] = f"E{index}"
    return {
        "entities": entities,
        "relations": _build_relations(text, entities),
        "_resource": {
            "lexicon_source": "frozen train+dev annotations only",
            "split_file": SPLIT_FILE.name,
        },
    }

