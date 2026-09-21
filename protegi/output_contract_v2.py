"""Stage 1 输出范式 v2（诊断专用，未冻结）。

状态：``OUTPUT_CONTRACT_V2_DIAGNOSTIC`` / ``NOT_READY_FOR_RUN9``。

- 诊断假设：poison-window runaway 是枚举语义触发的生成行为问题，
  不是容量问题（A 类正确输出约占 4096 的一半，B 类零 Gold 照样跑飞）。
- v2 只改输出范式，不改 window（3000/400）、不加 max_tokens（4096）、
  不改标签定义与边界语义：
  1. 每个合法 mention 最多一条 record；
  2. 禁止 duplicate / nested-overlap 穷举 / substring-token 枚举；
  3. ``id``/``text`` 移到程序端确定性恢复，模型只输出
     ``type``/``start``/``end``/``normalized_id`` 最小字段；
  4. 缺失偏移不再做表面文本全量扩展（关闭一条程序端放大向量）。
- 冻结红线：本模块不得被 optimizer / evaluator / promotion /
  entity_cache / contract_validator 引用；不得修改 prompts_p0 冻结契约；
  诊断通过、正式冻结之前不得进入 run9。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from schema import EXTRACTION_ENTITY_TYPES  # noqa: E402

from protegi.prompts_p0 import (  # noqa: E402
    ENTITY_GUIDANCE_P0,
    ENTITY_IMMUTABLE_CONTRACT,
    GUIDANCE_END,
    GUIDANCE_START,
    IMMUTABLE_END,
    IMMUTABLE_START,
    build_prompt,
)

OUTPUT_PARADIGM_VERSION = "output-contract-v2-diagnostic-v1"

# v2 记录的必需字段（模型侧）；normalized_id 保留（规范化需要模型判断，
# 程序端无法恢复），id/text 由程序端恢复，模型不得输出。
V2_REQUIRED_FIELDS = ("type", "start", "end")
V2_OPTIONAL_FIELDS = ("normalized_id",)

# 冻结规则 3（逐字，必须恰好出现一次，替换时 fail-closed）。
_V1_RULE3 = (
    "3. Extract every distinct explicit mention; "
    "do not duplicate the same type and span."
)
_V2_RULE3 = (
    "3. Each distinct explicit mention emits at most ONE record. "
    "Never emit duplicate records for the same type and span, "
    "never emit nested or overlapping variants of one mention, "
    "and never enumerate substrings or tokens as candidate records. "
    "When no explicit mention satisfies the frozen definitions, "
    "return {\"entities\": []} immediately without enumerating candidates."
)

# 冻结规则 6（逐字，必须恰好出现一次）。
_V1_RULE6 = (
    "6. Return exactly these entity fields: "
    "id, text, type, start, end, normalized_id."
)
_V2_RULE6 = (
    "6. Return exactly these entity fields: "
    "type, start, end, normalized_id. "
    "Do NOT output id or text; offsets and surface text are recovered "
    "programmatically from start/end."
)

# 冻结返回示例（f-string 已渲染为单花括号，逐字匹配，恰好一次）。
_V1_RETURN_EXAMPLE = (
    '{"entities": [{"id": "E1", "text": "exact substring", '
    '"type": "Vulnerability", "start": 0, "end": 14, '
    '"normalized_id": "CVE-..."}]}'
)
_V2_RETURN_EXAMPLE = (
    '{"entities": [{"type": "Vulnerability", "start": 0, "end": 14, '
    '"normalized_id": "CVE-..."}]}'
)

# 冻结 few-shot 记录形状：{"id": ..., "text": ..., "type": ..., ...}。
# 程序端把 7 条（3+4）逐字转成最小字段，条数不对即 fail-closed。
_FEWSHOT_RECORD_RE = re.compile(
    r'\{"id": "[^"]+", "text": "[^"]*", '
    r'"type": "([^"]+)", "start": (\d+), "end": (\d+), '
    r'"normalized_id": ("(?:[^"\\]|\\.)*"|null)\}'
)
_EXPECTED_FEWSHOT_RECORDS = 7

V2_GUIDANCE_APPENDIX = (
    "Emit at most one record per distinct mention. "
    "Do not duplicate records, do not emit nested or overlapping variants "
    "of the same mention, and do not enumerate substrings or tokens. "
    "When nothing satisfies the frozen definitions, return "
    '{"entities": []} immediately. '
    "Output only type, start, end and normalized_id per record."
)


def _replace_once(haystack: str, needle: str, replacement: str, *, label: str) -> str:
    count = haystack.count(needle)
    if count != 1:
        raise ValueError(
            f"v2 诊断契约构造失败：{label} 期望出现 1 次，实际 {count} 次；"
            "冻结原文可能已漂移，拒绝静默生成诊断 prompt"
        )
    return haystack.replace(needle, replacement)


def build_v2_immutable_contract() -> str:
    """从冻结契约派生 v2 诊断契约：定义段落逐字节保留，只换规则与示例。"""
    contract = ENTITY_IMMUTABLE_CONTRACT
    contract = _replace_once(contract, _V1_RULE3, _V2_RULE3, label="rule3")
    contract = _replace_once(contract, _V1_RULE6, _V2_RULE6, label="rule6")
    contract = _replace_once(
        contract, _V1_RETURN_EXAMPLE, _V2_RETURN_EXAMPLE, label="return_example"
    )
    transformed, n = _FEWSHOT_RECORD_RE.subn(
        lambda m: (
            f'{{"type": "{m.group(1)}", "start": {m.group(2)}, '
            f'"end": {m.group(3)}, "normalized_id": {m.group(4)}}}'
        ),
        contract,
    )
    if n != _EXPECTED_FEWSHOT_RECORDS:
        raise ValueError(
            "v2 诊断契约构造失败：few-shot 记录期望转换 "
            f"{_EXPECTED_FEWSHOT_RECORDS} 条，实际 {n} 条；拒绝静默生成"
        )
    marker = f"OUTPUT_PARADIGM_VERSION: {OUTPUT_PARADIGM_VERSION}\n"
    lines = transformed.splitlines(keepends=True)
    if not lines or not lines[0].startswith("BOUNDARY_CONTRACT_VERSION:"):
        raise ValueError("v2 诊断契约构造失败：首行版本标识缺失")
    return lines[0] + marker + "".join(lines[1:])


def build_v2_entity_prompt() -> str:
    """组装完整 v2 诊断 prompt（contract + guidance），结构与 P0 同构。"""
    guidance = ENTITY_GUIDANCE_P0.strip() + "\n" + V2_GUIDANCE_APPENDIX
    return build_prompt(build_v2_immutable_contract(), guidance)


def _coerce_offset(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_minimal_entity_mentions(
    text: str, raw_entities: Any
) -> Tuple[List[dict], dict]:
    """解析 v2 最小字段记录；id/text 由程序端从 start/end 确定性恢复。

    - 缺失偏移不再做表面文本全量扩展：无合法偏移即 invalid 丢弃并计数
     （关闭 v1 `_entity_spans` 的 fallback 放大向量）。
    - 同一 (type, start, end) 只保留一条，重复计数进 diagnostics。
    - nested/overlap 同类型相交对只计数、不丢弃（由 strict 指标自然惩罚）。
    - 模型若顺带输出 text，以原文切片为准，不一致计数。
    """
    allowed = set(EXTRACTION_ENTITY_TYPES)
    kept: Dict[Tuple[str, int, int], dict] = {}
    diag: Dict[str, Any] = {
        "output_paradigm": OUTPUT_PARADIGM_VERSION,
        "raw_count": 0,
        "kept_count": 0,
        "duplicates_collapsed": 0,
        "invalid_dropped": 0,
        "invalid_reasons": [],
        "nested_overlap_pairs": 0,
        "text_mismatches": 0,
    }
    raw_list = raw_entities if isinstance(raw_entities, list) else []
    for idx, raw in enumerate(raw_list):
        if not isinstance(raw, dict):
            diag["invalid_dropped"] += 1
            diag["invalid_reasons"].append(f"record_{idx}:not_an_object")
            continue
        diag["raw_count"] += 1
        entity_type = raw.get("type", raw.get("label", raw.get("entity_type")))
        start = _coerce_offset(raw.get("start"))
        end = _coerce_offset(raw.get("end"))
        if not isinstance(entity_type, str) or entity_type not in allowed:
            diag["invalid_dropped"] += 1
            diag["invalid_reasons"].append(f"record_{idx}:bad_type")
            continue
        if (
            start is None
            or end is None
            or not (0 <= start < end <= len(text))
        ):
            diag["invalid_dropped"] += 1
            diag["invalid_reasons"].append(f"record_{idx}:bad_offsets")
            continue
        surface = text[start:end]
        supplied = raw.get("text")
        if isinstance(supplied, str) and supplied != surface:
            diag["text_mismatches"] += 1
        normalized = raw.get("normalized_id", raw.get("normalized"))
        if normalized is not None and not isinstance(normalized, str):
            normalized = str(normalized)
        key = (entity_type, start, end)
        if key in kept:
            diag["duplicates_collapsed"] += 1
            continue
        kept[key] = {
            "text": surface,
            "type": entity_type,
            "start": start,
            "end": end,
            "normalized_id": normalized,
        }
    # 同类型相交（非全等）对计数：诊断“nested-overlap 穷举”残留。
    keys = sorted(kept, key=lambda k: (k[1], k[2], k[0]))
    pairs = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            t1, s1, e1 = keys[i][0], keys[i][1], keys[i][2]
            t2, s2, e2 = keys[j][0], keys[j][1], keys[j][2]
            if t1 != t2:
                continue
            if s2 >= e1:
                break
            if s1 < e2 and s2 < e1:
                pairs += 1
    diag["nested_overlap_pairs"] = pairs
    entities = []
    for key in sorted(kept, key=lambda k: (k[1], k[2], k[0])):
        record = dict(kept[key])
        record["id"] = f"E{len(entities) + 1}"
        entities.append(record)
    diag["kept_count"] = len(entities)
    return entities, diag


def estimate_canonical_output_chars(
    text: str,
    gold_entities: List[dict],
    *,
    paradigm: str = "v2",
) -> int:
    """用 Gold 标准答案估算正确输出的紧凑 JSON 长度（诊断容量用）。"""
    records = []
    for idx, entity in enumerate(gold_entities):
        start, end = entity.get("start"), entity.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if paradigm == "v2":
            records.append({
                "type": entity.get("type"),
                "start": start,
                "end": end,
                "normalized_id": entity.get("normalized_id"),
            })
        elif paradigm == "v1":
            records.append({
                "id": f"E{idx + 1}",
                "text": text[start:end]
                if 0 <= start < end <= len(text)
                else "",
                "type": entity.get("type"),
                "start": start,
                "end": end,
                "normalized_id": entity.get("normalized_id"),
            })
        else:
            raise ValueError(f"未知范式: {paradigm}")
    return len(json.dumps({"entities": records}, ensure_ascii=False,
                          separators=(",", ":")))


__all__ = [
    "OUTPUT_PARADIGM_VERSION",
    "V2_REQUIRED_FIELDS",
    "V2_OPTIONAL_FIELDS",
    "V2_GUIDANCE_APPENDIX",
    "build_v2_immutable_contract",
    "build_v2_entity_prompt",
    "parse_minimal_entity_mentions",
    "estimate_canonical_output_chars",
]
