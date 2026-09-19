"""ProTeGi 候选校验器。

``constrained`` 实验臂要求冻结契约与 P0 完全一致，只允许改写
``OPTIMIZABLE_GUIDANCE``。``unconstrained`` 实验臂允许改写完整语义提示，
但仍对评价器必需的输入占位符和 JSON 接口执行 fail-closed 校验。两种校验
结果必须在实验产物中分开记录，不能把接口有效误写为边界契约一致。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from schema import (  # noqa: E402
    EXTRACTION_ENTITY_TYPES,
    EXTRACTION_RELATION_ARGUMENT_TYPES,
    EXTRACTION_RELATION_TYPES,
)

from protegi.prompts_p0 import (  # noqa: E402
    ENTITY_IMMUTABLE_CONTRACT,
    GUIDANCE_END,
    GUIDANCE_START,
    IMMUTABLE_END,
    IMMUTABLE_START,
    RELATION_IMMUTABLE_CONTRACT,
    extract_immutable_contract,
    extract_optimizable_guidance,
)


FORBIDDEN_EXTRACTION_ENTITY_TYPES = {
    "AttackTactic",
    "KillChainPhase",
    "AttackPattern",
    "CAPEC",
}
FORBIDDEN_EXTRACTION_RELATION_TYPES = {
    "implies",
    "belongs_to_phase",
    "leverages",
    "realizes",
}

_BLOCK_PATTERN = re.compile(
    r"<example>\s*<text>\s*(.*?)\s*</text>\s*"
    r"<entities>\s*(.*?)\s*</entities>"
    r"(?:\s*<relations>\s*(.*?)\s*</relations>)?\s*</example>",
    re.DOTALL | re.IGNORECASE,
)

_GUIDANCE_FORBIDDEN_TAGS = (
    "<example",
    "<text",
    "<entities",
    "<relations",
    "<immutable_contract",
    "<optimizable_guidance",
)


class ContractValidationResult:
    """契约校验结果。"""

    def __init__(self, is_valid: bool, reasons: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.reasons = reasons or []

    @property
    def error_message(self) -> str:
        return "; ".join(self.reasons)

    def __bool__(self) -> bool:
        return self.is_valid


def _normalise_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validate_frozen_examples(
    contract: str,
    *,
    require_relations: bool,
) -> List[str]:
    """逐字校验冻结示例的 span、端点、方向与 evidence。"""
    reasons: List[str] = []
    blocks = _BLOCK_PATTERN.findall(contract)
    if not blocks:
        return ["冻结契约未包含可解析的 reference example"]

    allowed_entity_types = set(EXTRACTION_ENTITY_TYPES)
    allowed_relation_types = set(EXTRACTION_RELATION_TYPES)

    for example_idx, (text, entities_json, relations_json) in enumerate(blocks, 1):
        try:
            entities = json.loads(entities_json)
        except json.JSONDecodeError as exc:
            reasons.append(f"冻结示例 {example_idx} 的 entities 不是合法 JSON: {exc.msg}")
            continue
        if not isinstance(entities, list):
            reasons.append(f"冻结示例 {example_idx} 的 entities 必须为列表")
            continue

        by_id: Dict[str, Dict[str, Any]] = {}
        seen_spans = set()
        for entity_idx, entity in enumerate(entities, 1):
            if not isinstance(entity, dict):
                reasons.append(f"冻结示例 {example_idx} 实体 {entity_idx} 不是对象")
                continue
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            start = entity.get("start")
            end = entity.get("end")
            surface = entity.get("text")
            expected_entity_fields = {"id", "text", "type", "start", "end"}
            if not require_relations:
                expected_entity_fields.add("normalized_id")
            if set(entity) != expected_entity_fields:
                reasons.append(
                    f"冻结示例 {example_idx} 实体 {entity_idx} 字段不等于 "
                    f"{sorted(expected_entity_fields)}"
                )
            if not isinstance(entity_id, str) or not entity_id:
                reasons.append(f"冻结示例 {example_idx} 实体 {entity_idx} 缺少合法 id")
                continue
            if entity_id in by_id:
                reasons.append(f"冻结示例 {example_idx} 存在重复实体 id: {entity_id}")
            by_id[entity_id] = entity
            if entity_type not in allowed_entity_types:
                reasons.append(
                    f"冻结示例 {example_idx} 实体 {entity_id} 类型非法: {entity_type}"
                )
            if not isinstance(start, int) or not isinstance(end, int):
                reasons.append(f"冻结示例 {example_idx} 实体 {entity_id} offset 非整数")
                continue
            if start < 0 or end <= start or end > len(text):
                reasons.append(f"冻结示例 {example_idx} 实体 {entity_id} offset 越界")
                continue
            if not isinstance(surface, str) or text[start:end] != surface:
                reasons.append(
                    f"冻结示例 {example_idx} 实体 {entity_id} span 与原文不一致"
                )
            span_key = (entity_type, start, end)
            if span_key in seen_spans:
                reasons.append(f"冻结示例 {example_idx} 存在重复 type+span: {span_key}")
            seen_spans.add(span_key)

        if require_relations and not relations_json.strip():
            reasons.append(f"冻结示例 {example_idx} 缺少 relations")
            continue
        if not require_relations:
            if relations_json.strip():
                reasons.append(f"实体冻结示例 {example_idx} 不得包含 relations")
            continue

        try:
            relations = json.loads(relations_json)
        except json.JSONDecodeError as exc:
            reasons.append(f"冻结示例 {example_idx} 的 relations 不是合法 JSON: {exc.msg}")
            continue
        if not isinstance(relations, list):
            reasons.append(f"冻结示例 {example_idx} 的 relations 必须为列表")
            continue

        for relation_idx, relation in enumerate(relations, 1):
            if not isinstance(relation, dict):
                reasons.append(f"冻结示例 {example_idx} 关系 {relation_idx} 不是对象")
                continue
            relation_type = relation.get("type")
            source = relation.get("source")
            target = relation.get("target")
            expected_relation_fields = {
                "source",
                "target",
                "type",
                "evidence_start",
                "evidence_end",
            }
            if set(relation) != expected_relation_fields:
                reasons.append(
                    f"冻结示例 {example_idx} 关系 {relation_idx} 字段不等于 "
                    f"{sorted(expected_relation_fields)}"
                )
            if relation_type not in allowed_relation_types:
                reasons.append(
                    f"冻结示例 {example_idx} 关系 {relation_idx} 类型非法: {relation_type}"
                )
                continue
            if source not in by_id or target not in by_id:
                reasons.append(f"冻结示例 {example_idx} 关系 {relation_idx} 端点不存在")
                continue
            expected_source, expected_target = EXTRACTION_RELATION_ARGUMENT_TYPES[
                relation_type
            ]
            actual_types = (by_id[source].get("type"), by_id[target].get("type"))
            if actual_types != (expected_source, expected_target):
                reasons.append(
                    f"冻结示例 {example_idx} 关系 {relation_idx} 方向错误: "
                    f"{actual_types[0]} -> {actual_types[1]}"
                )
            evidence_start = relation.get("evidence_start")
            evidence_end = relation.get("evidence_end")
            if not isinstance(evidence_start, int) or not isinstance(evidence_end, int):
                reasons.append(f"冻结示例 {example_idx} 关系 {relation_idx} evidence offset 非整数")
                continue
            if evidence_start < 0 or evidence_end <= evidence_start or evidence_end > len(text):
                reasons.append(f"冻结示例 {example_idx} 关系 {relation_idx} evidence 越界")
                continue
            source_entity = by_id[source]
            target_entity = by_id[target]
            if not (
                evidence_start <= source_entity.get("start", -1)
                and source_entity.get("end", len(text) + 1) <= evidence_end
                and evidence_start <= target_entity.get("start", -1)
                and target_entity.get("end", len(text) + 1) <= evidence_end
            ):
                reasons.append(
                    f"冻结示例 {example_idx} 关系 {relation_idx} evidence 未覆盖两个端点"
                )
    return reasons


def _validate_wrapper(prompt_text: str, expected_contract: str) -> tuple[List[str], str]:
    reasons: List[str] = []
    for marker in (IMMUTABLE_START, IMMUTABLE_END, GUIDANCE_START, GUIDANCE_END):
        if prompt_text.count(marker) != 1:
            reasons.append(f"标记 {marker} 必须且只能出现一次")

    immutable = extract_immutable_contract(prompt_text)
    guidance = extract_optimizable_guidance(prompt_text)
    if immutable is None:
        reasons.append("无法解析冻结契约块")
    elif _normalise_newlines(immutable) != _normalise_newlines(expected_contract):
        reasons.append("冻结契约与 P0 不一致；候选不得修改标签、边界、示例或 schema")

    if guidance is None:
        reasons.append("无法解析可优化 guidance 块")
        return reasons, ""
    if not guidance.strip():
        reasons.append("可优化 guidance 不得为空")
    return reasons, guidance


def _validate_contract_definition(contract: str, *, stage: str) -> List[str]:
    """独立检查 P0 常量本身，避免“与被误改常量一致”造成自证通过。"""
    reasons: List[str] = []
    if "BOUNDARY_CONTRACT_VERSION: chapter3-boundary-sync-v2" not in contract:
        reasons.append("冻结契约缺少 chapter3-boundary-sync-v2 版本标识")
    if "{text}" not in contract:
        reasons.append("冻结契约缺少 {text} 输入占位符")

    if stage == "entity":
        if "TASK_STAGE: entity-only" not in contract:
            reasons.append("实体冻结契约缺少 entity-only 阶段标识")
        for entity_type in sorted(EXTRACTION_ENTITY_TYPES):
            if not re.search(rf"\b{re.escape(entity_type)}\b", contract):
                reasons.append(f"实体冻结契约缺少合法类型: {entity_type}")
        for field in ("id", "text", "type", "start", "end", "normalized_id"):
            if not re.search(rf'"{re.escape(field)}"\s*:', contract):
                reasons.append(f"实体冻结契约缺少 JSON 字段: {field}")
        if "{entities}" in contract:
            reasons.append("实体冻结契约不得包含 {entities} 输入占位符")
    else:
        if "TASK_STAGE: relation-only" not in contract:
            reasons.append("关系冻结契约缺少 relation-only 阶段标识")
        if "{entities}" not in contract:
            reasons.append("关系冻结契约缺少 {entities} 输入占位符")
        for relation_type, (source_type, target_type) in sorted(
            EXTRACTION_RELATION_ARGUMENT_TYPES.items()
        ):
            expected = f"{relation_type}: {source_type} -> {target_type}"
            if expected not in contract:
                reasons.append(f"关系冻结契约缺少标准方向: {expected}")
        for field in ("source", "target", "type", "evidence_start", "evidence_end"):
            if not re.search(rf'"{re.escape(field)}"\s*:', contract):
                reasons.append(f"关系冻结契约缺少 JSON 字段: {field}")
    return reasons


def _validate_guidance(guidance: str, *, stage: str) -> List[str]:
    reasons: List[str] = []
    lower = guidance.lower()
    if len(guidance) > 4000:
        reasons.append("guidance 超过 4000 字符，疑似复写冻结契约或样本")
    for tag in _GUIDANCE_FORBIDDEN_TAGS:
        if tag in lower:
            reasons.append(f"guidance 不得包含契约/示例标签: {tag}")

    schema_fields = (
        "id",
        "text",
        "type",
        "start",
        "end",
        "normalized_id",
        "source",
        "target",
        "evidence_start",
        "evidence_end",
        "entities",
        "relations",
    )
    for field in schema_fields:
        quoted_field = re.search(
            rf'["\']{re.escape(field)}["\']\s*:',
            guidance,
            re.IGNORECASE,
        )
        field_definition = re.search(
            rf"(?im)^\s*[-*]?\s*{re.escape(field)}\s*:",
            guidance,
        )
        if quoted_field or field_definition:
            reasons.append(f"guidance 不得重写 JSON schema 字段: {field}")

    if re.search(
        r"\b(?:CVE-\d{4}-\d+|CWE-\d+|T\d{4}(?:\.\d{3})?|cpe:2\.3:|E\d+)\b",
        guidance,
        re.IGNORECASE,
    ):
        reasons.append("guidance 不得复制样本标识符、实体 ID 或具体规范化答案")

    all_labels = sorted(EXTRACTION_ENTITY_TYPES) + sorted(EXTRACTION_RELATION_TYPES)
    for label in all_labels:
        redefine = re.compile(
            rf"(?im)^\s*[-*]?\s*{re.escape(label)}\s*(?::|=|means\b|is\s+defined\b|->|→|\s+-\s+)"
        )
        if redefine.search(guidance):
            reasons.append(f"guidance 不得重新定义冻结标签或方向: {label}")

    for forbidden in FORBIDDEN_EXTRACTION_ENTITY_TYPES | FORBIDDEN_EXTRACTION_RELATION_TYPES:
        if re.search(rf"\b{re.escape(forbidden)}\b", guidance, re.IGNORECASE):
            reasons.append(f"guidance 引入了非抽取层概念: {forbidden}")

    if stage == "entity":
        for relation_type in sorted(EXTRACTION_RELATION_TYPES):
            if re.search(rf"\b{re.escape(relation_type)}\b", guidance, re.IGNORECASE):
                reasons.append(f"Stage 1 guidance 不得引入关系类型: {relation_type}")
        endpoint_schema = re.compile(
            r"\b(?:source|target|head|tail)\s*(?:(?:id|field|entity\s+id)\b|:)",
            re.IGNORECASE,
        )
        relation_output = re.compile(
            r"\b(?:output|extract|emit|return)\s+(?:an?\s+|the\s+)?(?:relations?|relationships?)\b",
            re.IGNORECASE,
        )
        if endpoint_schema.search(guidance) or relation_output.search(guidance):
            reasons.append("Stage 1 guidance 不得引入关系端点或关系输出")

    return reasons


def _validate_runtime_interface(prompt_text: str, *, stage: str) -> List[str]:
    """校验无约束语义搜索仍可由固定评价程序执行和解析。"""
    reasons: List[str] = []
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return ["完整提示词不得为空"]
    if len(prompt_text) > 50000:
        reasons.append("完整提示词超过 50000 字符，拒绝进入固定评价程序")

    if prompt_text.count("{text}") != 1:
        reasons.append("运行接口要求且只允许一个字面占位符 {text}")

    if stage == "entity":
        if "{entities}" in prompt_text:
            reasons.append("Stage 1 完整提示词不得包含 {entities} 输入占位符")
        top_level_key = "entities"
        required_fields = ("id", "text", "type", "start", "end", "normalized_id")
    elif stage == "relation":
        if prompt_text.count("{entities}") != 1:
            reasons.append("Stage 2 运行接口要求且只允许一个字面占位符 {entities}")
        top_level_key = "relations"
        required_fields = (
            "source",
            "target",
            "type",
            "evidence_start",
            "evidence_end",
        )
    else:
        raise ValueError(f"未知阶段: {stage}")

    if not re.search(rf'["\']{re.escape(top_level_key)}["\']\s*:', prompt_text):
        reasons.append(f"运行接口缺少 JSON 顶层键: {top_level_key}")
    for field in required_fields:
        if not re.search(rf'["\']{re.escape(field)}["\']\s*:', prompt_text):
            reasons.append(f"运行接口缺少 JSON 字段: {field}")
    return reasons


class PromptContractValidator:
    """冻结契约与无约束运行接口的分层校验器。"""

    @classmethod
    def validate_entity_prompt(cls, prompt_text: str) -> ContractValidationResult:
        reasons, guidance = _validate_wrapper(prompt_text, ENTITY_IMMUTABLE_CONTRACT)
        reasons.extend(_validate_contract_definition(ENTITY_IMMUTABLE_CONTRACT, stage="entity"))
        reasons.extend(_validate_frozen_examples(ENTITY_IMMUTABLE_CONTRACT, require_relations=False))
        reasons.extend(_validate_guidance(guidance, stage="entity"))
        return ContractValidationResult(len(reasons) == 0, reasons)

    @classmethod
    def validate_relation_prompt(cls, prompt_text: str) -> ContractValidationResult:
        reasons, guidance = _validate_wrapper(prompt_text, RELATION_IMMUTABLE_CONTRACT)
        reasons.extend(_validate_contract_definition(RELATION_IMMUTABLE_CONTRACT, stage="relation"))
        reasons.extend(_validate_frozen_examples(RELATION_IMMUTABLE_CONTRACT, require_relations=True))
        reasons.extend(_validate_guidance(guidance, stage="relation"))
        return ContractValidationResult(len(reasons) == 0, reasons)

    @classmethod
    def validate_runtime_interface(
        cls,
        stage: str,
        prompt_text: str,
    ) -> ContractValidationResult:
        reasons = _validate_runtime_interface(prompt_text, stage=stage)
        return ContractValidationResult(len(reasons) == 0, reasons)

    @classmethod
    def validate_candidate(
        cls,
        stage: str,
        prompt_text: str,
        prompt_scope: str = "constrained",
    ) -> ContractValidationResult:
        scope = str(prompt_scope).strip().lower()
        if scope == "unconstrained":
            return cls.validate_runtime_interface(stage, prompt_text)
        if scope != "constrained":
            raise ValueError(
                "未知 prompt_scope: "
                f"{prompt_scope!r}；必须为 constrained 或 unconstrained"
            )
        if stage == "entity":
            return cls.validate_entity_prompt(prompt_text)
        if stage == "relation":
            return cls.validate_relation_prompt(prompt_text)
        raise ValueError(f"未知阶段: {stage}")
