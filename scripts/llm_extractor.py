"""
LLM 抽取模块

调用本地模型进行实体和关系抽取。
支持多种后端：OpenAI-compatible API、本地 transformers。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ModelOutputBudgetExhaustedError(RuntimeError):
    """The provider exhausted the completion budget before a complete reply."""

    def __init__(self, completion_tokens: int, max_tokens: int):
        self.completion_tokens = completion_tokens
        self.max_tokens = max_tokens
        super().__init__(
            "model_output_budget_exhausted: "
            f"completion_tokens={completion_tokens}, max_tokens={max_tokens}"
        )


ALLOWED_ENTITY_TYPES = {
    'Configuration',
    'Vulnerability',
    'Weakness',
    'AttackPattern',
    'AttackTechnique',
    'AttackTactic',
    'KillChainPhase',
}

ALLOWED_RELATION_TYPES = {
    'affects',
    'instantiates',
    'leverages',
    'realizes',
    'implies',
    'belongs_to_phase',
}

RELATION_TYPE_CONSTRAINTS = {
    'affects': ('Vulnerability', 'Configuration'),
    'instantiates': ('Vulnerability', 'Weakness'),
    'leverages': ('AttackPattern', 'Weakness'),
    'realizes': ('AttackPattern', 'AttackTechnique'),
    'implies': ('AttackTechnique', 'AttackTactic'),
    'belongs_to_phase': ('AttackTactic', 'KillChainPhase'),
}

ATTACK_TACTIC_TEXTS = {
    'initial access',
    'execution',
    'persistence',
    'privilege escalation',
    'defense evasion',
    'credential access',
    'lateral movement',
    'collection',
    'exfiltration',
    'impact',
    'reconnaissance',
    'discovery',
    'resource development',
    'command and control',
}

KILL_CHAIN_PHASE_TEXTS = {
    'reconnaissance',
    'weaponization',
    'delivery',
    'exploitation',
    'installation',
    'command and control',
    'actions on objectives',
}

SECURITY_METHOD_PHRASES = [
    'server-side request forgery',
    'remote code execution',
    'arbitrary code execution',
    'insecure deserialization',
    'cross-site scripting',
    'arbitrary file disclosure',
    'arbitrary file write',
    'arbitrary file read',
    'information disclosure',
    'information exposure',
    'authentication bypass',
    'privilege escalation',
    'directory traversal',
    'path traversal',
    'command injection',
    'code injection',
    'sql injection',
    'ognl injection',
    'deserialization',
    'buffer overflow',
    'format string',
    'use-after-free',
    'memory corruption',
    'race condition',
    'ssrf',
    'xss',
    'rce',
]

WEAKNESS_FALLBACK_PATTERN = re.compile(
    r"\b(?:[A-Za-z][\w/+.-]*\s+){0,4}(?:injection|traversal|deserialization|disclosure|exposure|execution|overflow|bypass|forgery|escalation|corruption|spoofing|validation|race condition|use-after-free)\b",
    re.IGNORECASE,
)

GENERIC_CONFIGURATION_TEXTS = {
    'software',
    'system',
    'server',
    'package',
    'application',
    'product',
    'component',
    'service',
}


def _opencode_headers(config: dict) -> dict[str, str]:
    """Return the routing headers currently required by OpenCode Go."""
    base_url = str(config.get("base_url", ""))
    if "opencode.ai/zen/go" not in base_url.casefold():
        return {}
    session_id = str(
        config.get("session_id")
        or os.environ.get("V3_OPENCODE_SESSION_ID", "")
        or os.environ.get("V5_OPENCODE_SESSION_ID", "")
    ).strip()
    if not session_id:
        raise RuntimeError(
            "OpenCode Go requires a stable x-opencode-session header; set "
            "V5_OPENCODE_SESSION_ID for this experiment run"
        )
    user_agent = str(
        config.get("user_agent")
        or os.environ.get("V3_OPENCODE_USER_AGENT", "")
        or os.environ.get("V5_OPENCODE_USER_AGENT", "")
        or "bron-apo-research/1.0"
    ).strip()
    return {
        "x-opencode-session": session_id,
        "User-Agent": user_agent,
    }


def _coerce_scalar_string(value) -> Optional[str]:
    """把 LLM 产出的标量字段尽量收敛成单个字符串。"""
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, list):
        for item in value:
            coerced = _coerce_scalar_string(item)
            if coerced:
                return coerced
        return None
    return None


def _coerce_int(value) -> Optional[int]:
    """把 start/end 之类字段尽量转成整数。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _extract_core_phrase(text: str, phrases: list[str]) -> Optional[str]:
    lower_text = text.lower()
    best_match = None
    best_index = None
    for phrase in sorted(phrases, key=len, reverse=True):
        index = lower_text.find(phrase.lower())
        if index == -1:
            continue
        if best_match is None or index < best_index or (index == best_index and len(phrase) > len(best_match)):
            best_match = text[index:index + len(phrase)]
            best_index = index
    return best_match


def _extract_security_phrase(text: str) -> Optional[str]:
    """从通用安全短语中抽取核心 weakness/attack-pattern 表达。"""
    core = _extract_core_phrase(text, SECURITY_METHOD_PHRASES)
    if core:
        return core
    match = WEAKNESS_FALLBACK_PATTERN.search(text)
    if match:
        return match.group(0)
    return None


def _find_text_span(original_text: str, entity_text: str) -> Optional[tuple[int, int, str]]:
    start = original_text.find(entity_text)
    if start != -1:
        return start, start + len(entity_text), original_text[start:start + len(entity_text)]

    lower_text = original_text.lower()
    lower_entity = entity_text.lower()
    start = lower_text.find(lower_entity)
    if start != -1:
        end = start + len(entity_text)
        return start, end, original_text[start:end]
    return None


def _normalize_entity_text(entity_type: str, entity_text: str) -> Optional[str]:
    text = " ".join(entity_text.strip().split())
    text = text.strip(" \t\r\n-:;,.'\"()[]")
    if not text:
        return None

    if entity_type == 'Vulnerability':
        cve_match = re.search(r"CVE-\d{4}-\d{4,7}", text, re.IGNORECASE)
        if cve_match:
            return cve_match.group(0)
        if re.search(r"\b(vulnerability|flaw|issue)\b", text, re.IGNORECASE) and len(text) > 30:
            return None
        return re.sub(r"\s+\((?:cve-[^)]+)\)$", "", text, flags=re.IGNORECASE).strip()

    if entity_type == 'Configuration':
        text = re.sub(r"^(?:the|affected|vulnerable)\s+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+\b(?:package|packages|application|applications|software|products?|components?)\b$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+\d+(?:\.\d+){1,3}$", "", text)
        text = text.strip(" \t\r\n-:;,.'\"()[]")
        if text.lower() in GENERIC_CONFIGURATION_TEXTS:
            return None
        return text or None

    if entity_type == 'Weakness':
        core = _extract_security_phrase(text)
        if core:
            return core
        text = re.sub(r"\s+\b(?:vulnerability|flaw|weakness)\b$", "", text, flags=re.IGNORECASE)
        return text.strip() or None

    if entity_type == 'AttackPattern':
        core = _extract_security_phrase(text)
        if core:
            return core
        capec_match = re.search(r"CAPEC-\d+", text, re.IGNORECASE)
        if capec_match:
            return capec_match.group(0)
        if len(text) > 40:
            return None
        return text

    if entity_type == 'AttackTactic':
        normalized = text.lower()
        if normalized in ATTACK_TACTIC_TEXTS:
            return normalized
        return None

    if entity_type == 'KillChainPhase':
        normalized = text.lower()
        if normalized in KILL_CHAIN_PHASE_TEXTS:
            return normalized
        return None

    return text


def call_openai_compatible(prompt: str, system_prompt: str, config: dict) -> str:
    """调用 OpenAI-compatible API（适用于本地部署的 vLLM、Ollama、Responses 端点等）"""
    model_name = config.get('model', 'default')
    endpoint = str(config.get('endpoint', '')).strip().lower()
    base_url = config.get('base_url', 'http://localhost:8000/v1')

    is_muse_contributor = (
        'muse-spark' in str(model_name).casefold()
        and 'contributor' in str(model_name).casefold()
    )
    if is_muse_contributor or 'responses' in endpoint or '/responses' in base_url:
        import requests
        clean_base = base_url.rstrip('/')
        if clean_base.endswith('/responses'):
            endpoint_url = clean_base
        else:
            endpoint_url = f"{clean_base}/responses"
        headers = {
            'Authorization': f"Bearer {config.get('api_key', 'not-needed')}",
            'Content-Type': 'application/json',
            **_opencode_headers(config),
        }
        payload = {
            'model': model_name,
            'input': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            'temperature': float(config.get('temperature', 0.1)),
            # Responses uses max_output_tokens; keeping this here ensures the
            # APO manifest's Critic/Editor budgets are effective, not merely
            # descriptive metadata.
            'max_output_tokens': int(
                config.get('max_output_tokens', config.get('max_tokens', 4096))
            ),
            'top_p': float(config.get('top_p', 1.0)),
            'reasoning': {'effort': str(config.get('reasoning_effort') or 'xhigh').strip().lower() or 'xhigh'}
        }
        resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=float(config.get('timeout', 180)))

        if not resp.ok:
            safe_error = {}
            try:
                error_body = resp.json()
            except ValueError:
                error_body = {}
            error_value = (
                error_body.get('error')
                if isinstance(error_body, dict)
                else None
            )
            if isinstance(error_value, dict):
                safe_error = {
                    key: error_value.get(key)
                    for key in ('type', 'code', 'param', 'message')
                    if error_value.get(key) is not None
                }
            elif isinstance(error_value, str):
                safe_error = {'message': error_value[:1000]}
            exc = RuntimeError(
                'responses_api_http_error:'
                + json.dumps(
                    {'status_code': resp.status_code, 'error': safe_error},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            exc.status_code = resp.status_code
            raise exc
        data = resp.json()
        text_parts = []
        top_level_text = data.get('output_text')
        if isinstance(top_level_text, str) and top_level_text.strip():
            text_parts.append(top_level_text.strip())
        elif isinstance(top_level_text, list):
            text_parts.extend(
                item.strip()
                for item in top_level_text
                if isinstance(item, str) and item.strip()
            )

        output_items = data.get('output', [])
        if not isinstance(output_items, list):
            output_items = []
        for item in output_items:
            if isinstance(item, str) and item.strip():
                text_parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            if item.get('role') == 'assistant' or item.get('type') == 'message':
                content_items = item.get('content', [])
                if isinstance(content_items, dict):
                    content_items = [content_items]
                for content in content_items:
                    if isinstance(content, str) and content.strip():
                        text_parts.append(content.strip())
                    elif isinstance(content, dict):
                        value = content.get('text')
                        if isinstance(value, dict):
                            value = value.get('value')
                        if (
                            content.get('type') in {'output_text', 'text'}
                            and isinstance(value, str)
                            and value.strip()
                        ):
                            text_parts.append(value.strip())
        if text_parts:
            return '\n'.join(text_parts)

        max_output_tokens = int(
            config.get('max_output_tokens', config.get('max_tokens', 4096))
        )
        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        output_tokens = usage.get('output_tokens')
        incomplete_details = (
            data.get('incomplete_details')
            if isinstance(data.get('incomplete_details'), dict)
            else {}
        )
        incomplete_reason = str(incomplete_details.get('reason') or '').lower()
        if data.get('status') == 'incomplete' or incomplete_reason in {
            'max_output_tokens',
            'length',
        }:
            raise ModelOutputBudgetExhaustedError(
                output_tokens if isinstance(output_tokens, int) else max_output_tokens,
                max_output_tokens,
            )

        # Never persist prompt or response bodies in diagnostics.  Structural
        # metadata is enough to distinguish a provider schema change from a
        # refusal, content filter, or exhausted response budget.
        response_shape = {
            'status': data.get('status'),
            'top_level_keys': sorted(str(key) for key in data),
            'output_item_types': [
                item.get('type') if isinstance(item, dict) else type(item).__name__
                for item in output_items
            ],
            'content_item_types': [
                content.get('type')
                for item in output_items
                if isinstance(item, dict)
                for content in (
                    item.get('content', [])
                    if isinstance(item.get('content', []), list)
                    else [item.get('content', {})]
                )
                if isinstance(content, dict)
            ],
            'incomplete_details': incomplete_details,
            'usage': {
                key: value
                for key, value in usage.items()
                if key in {'input_tokens', 'output_tokens', 'total_tokens'}
            },
            'error_type': (
                data.get('error', {}).get('type')
                if isinstance(data.get('error'), dict)
                else None
            ),
        }
        raise RuntimeError(
            'responses_api_returned_no_text:'
            + json.dumps(response_shape, ensure_ascii=False, sort_keys=True)
        )

    try:
        import openai
    except ImportError:
        raise ImportError("需要安装 openai 包: pip install openai")

    client = openai.OpenAI(
        api_key=config.get('api_key', 'not-needed'),
        base_url=base_url,
        timeout=float(config.get('timeout', 90)),
        max_retries=int(config.get('max_retries', 0)),
        default_headers=_opencode_headers(config),
    )

    request_kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": config.get('temperature', 0.1),
        "max_tokens": config.get('max_tokens', 4096),
        "top_p": config.get('top_p', 0.95),
    }
    thinking = str(config.get('thinking', '')).strip().lower()
    reasoning_effort = str(config.get('reasoning_effort', '')).strip().lower()
    extra_body = {}
    if thinking in {'enabled', 'disabled'}:
        extra_body['thinking'] = {'type': thinking}
    if reasoning_effort in {'none', 'low', 'medium', 'high', 'xhigh', 'max'}:
        # The OpenAI SDK exposes reasoning_effort as a native request field.
        # Mirroring it into extra_body produces a duplicate JSON key after the
        # SDK merges both dictionaries and can make gateway behaviour unclear.
        request_kwargs['reasoning_effort'] = reasoning_effort
    elif thinking == 'disabled' and not reasoning_effort:
        extra_body['reasoning_effort'] = 'none'

    if extra_body:
        request_kwargs['extra_body'] = extra_body

    response = client.chat.completions.create(**request_kwargs)
    choice = response.choices[0]
    content = choice.message.content
    usage = getattr(response, 'usage', None)
    completion_tokens = getattr(usage, 'completion_tokens', None)
    max_tokens = int(request_kwargs['max_tokens'])
    finish_reason = str(getattr(choice, 'finish_reason', '') or '').lower()
    budget_exhausted = finish_reason in {'length', 'max_tokens'} or (
        isinstance(completion_tokens, int)
        and completion_tokens >= max_tokens - 1
    )
    if budget_exhausted:
        raise ModelOutputBudgetExhaustedError(
            completion_tokens if isinstance(completion_tokens, int) else max_tokens,
            max_tokens,
        )
    return content



def call_ollama(prompt: str, system_prompt: str, config: dict) -> str:
    """调用 Ollama API"""
    try:
        import requests
    except ImportError:
        raise ImportError("需要安装 requests 包: pip install requests")

    url = config.get('base_url', 'http://localhost:11434') + '/api/generate'
    payload = {
        'model': config.get('model', 'llama3'),
        'prompt': prompt,
        'system': system_prompt,
        'stream': False,
        'options': {
            'temperature': config.get('temperature', 0.1),
            'num_predict': config.get('max_tokens', 4096),
            'top_p': config.get('top_p', 0.95)
        }
    }

    response = requests.post(url, json=payload, timeout=config.get('timeout', 300))
    response.raise_for_status()
    return response.json()['response']


def parse_llm_output(output: str, original_text: str = None) -> dict:
    """解析 LLM 输出的 JSON"""
    # 尝试直接解析
    try:
        result = json.loads(output)
        return validate_output(result, original_text)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', output, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(1))
            return validate_output(result, original_text)
        except json.JSONDecodeError:
            pass

    # 尝试找到第一个 { 和最后一个 }
    first_brace = output.find('{')
    last_brace = output.rfind('}')
    if first_brace != -1 and last_brace != -1:
        try:
            result = json.loads(output[first_brace:last_brace + 1])
            return validate_output(result, original_text)
        except json.JSONDecodeError:
            pass

    # 解析失败，返回空结果
    return {'entities': [], 'relations': []}


def relocate_entities(entities: list, original_text: str) -> list:
    """重新定位实体在原始文本中的位置

    LLM 输出的 start/end 可能基于 prompt 文本，需要在原始文本中重新定位。
    """
    relocated = []
    for e in entities:
        entity_type = _coerce_scalar_string(e.get('type', ''))
        entity_text = _coerce_scalar_string(e.get('text', ''))
        if not entity_text:
            continue
        if entity_type:
            entity_text = _normalize_entity_text(entity_type, entity_text)
        if not entity_text:
            continue

        span = _find_text_span(original_text, entity_text)
        if span is None:
            # 找不到实体文本，跳过
            continue
        start, end, matched_text = span
        e['text'] = matched_text
        e['start'] = start
        e['end'] = end

        relocated.append(e)
    return relocated


def validate_output(result: dict, original_text: str = None) -> dict:
    """验证和清洗输出"""
    if not isinstance(result, dict):
        return {'entities': [], 'relations': []}

    entities = result.get('entities', [])
    relations = result.get('relations', [])

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []

    # 如果提供了原始文本，重新定位实体
    if original_text:
        entities = relocate_entities(entities, original_text)

    # 清洗实体
    valid_entity_ids = set()
    cleaned_entities = []
    seen_entity_keys = set()
    for e in entities:
        if not isinstance(e, dict):
            continue
        # 必须字段
        if not all(k in e for k in ['id', 'text', 'type', 'start', 'end']):
            continue
        entity_id = _coerce_scalar_string(e.get('id'))
        entity_text = _coerce_scalar_string(e.get('text'))
        entity_type = _coerce_scalar_string(e.get('type'))
        start = _coerce_int(e.get('start'))
        end = _coerce_int(e.get('end'))
        if not entity_id or not entity_text or not entity_type:
            continue
        # 类型检查
        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue
        entity_text = _normalize_entity_text(entity_type, entity_text)
        if not entity_text:
            continue
        if original_text:
            span = _find_text_span(original_text, entity_text)
            if span is None:
                continue
            start, end, matched_text = span
            entity_text = matched_text
        # 位置检查
        if start is None or end is None:
            continue
        if start >= end:
            continue
        entity_key = (start, end, entity_type)
        if entity_key in seen_entity_keys:
            continue
        cleaned_entity = dict(e)
        cleaned_entity['id'] = entity_id
        cleaned_entity['text'] = entity_text
        cleaned_entity['type'] = entity_type
        cleaned_entity['start'] = start
        cleaned_entity['end'] = end

        cleaned_entities.append(cleaned_entity)
        valid_entity_ids.add(entity_id)
        seen_entity_keys.add(entity_key)

    # 清洗关系
    entity_type_by_id = {entity['id']: entity['type'] for entity in cleaned_entities}
    cleaned_relations = []
    seen_relation_keys = set()
    for r in relations:
        if not isinstance(r, dict):
            continue
        if not all(k in r for k in ['id', 'type', 'head', 'tail']):
            continue
        relation_id = _coerce_scalar_string(r.get('id'))
        relation_type = _coerce_scalar_string(r.get('type'))
        head = _coerce_scalar_string(r.get('head'))
        tail = _coerce_scalar_string(r.get('tail'))
        if not relation_id or not relation_type or not head or not tail:
            continue
        if relation_type not in ALLOWED_RELATION_TYPES:
            continue
        if head not in valid_entity_ids or tail not in valid_entity_ids:
            continue
        if head == tail:
            continue
        expected_head_type, expected_tail_type = RELATION_TYPE_CONSTRAINTS[relation_type]
        if entity_type_by_id.get(head) != expected_head_type or entity_type_by_id.get(tail) != expected_tail_type:
            continue
        relation_key = (relation_type, head, tail)
        if relation_key in seen_relation_keys:
            continue
        cleaned_relation = dict(r)
        cleaned_relation['id'] = relation_id
        cleaned_relation['type'] = relation_type
        cleaned_relation['head'] = head
        cleaned_relation['tail'] = tail
        cleaned_relations.append(cleaned_relation)
        seen_relation_keys.add(relation_key)

    return {'entities': cleaned_entities, 'relations': cleaned_relations}


def _build_focus_text(original_text: str, max_chars: int = 4000) -> str:
    """为长文构建一个聚焦片段，优先保留含 CVE 的关键行及其邻近上下文。"""
    lines = [line.rstrip() for line in original_text.splitlines()]
    if not lines:
        return original_text

    cve_pattern = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
    weakness_pattern = re.compile(
        r"\b(summary|affected|product|vulnerability|weakness|injection|traversal|disclosure|deserialization|code execution|code injection)\b",
        re.IGNORECASE,
    )
    tactic_pattern = re.compile(r"\b(persistence|exfiltration|reconnaissance)\b", re.IGNORECASE)

    anchor_indexes: list[int] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        score = 0
        if cve_pattern.search(line):
            score += 3
        if weakness_pattern.search(line):
            score += 2
        if tactic_pattern.search(line):
            score += 1
        if score > 0:
            anchor_indexes.extend(range(max(0, idx - 2), min(len(lines), idx + 3)))

    if anchor_indexes:
        selected_indexes = [idx for idx in anchor_indexes if lines[idx].strip()]
    else:
        non_empty_indexes = [idx for idx, line in enumerate(lines) if line.strip()]
        selected_indexes = non_empty_indexes[:6]

    seen = set()
    ordered_indexes = []
    for idx in selected_indexes:
        if idx not in seen:
            seen.add(idx)
            ordered_indexes.append(idx)

    chunks: list[str] = []
    total_chars = 0
    for idx in ordered_indexes:
        line = lines[idx]
        if not line:
            continue
        candidate_len = len(line) + (1 if chunks else 0)
        if total_chars + candidate_len > max_chars:
            break
        chunks.append(line)
        total_chars += candidate_len

    focused = "\n".join(chunks).strip()
    return focused or original_text[:max_chars]


def _merge_results(primary: dict, secondary: dict) -> dict:
    """按实体跨度/类型合并两次抽取结果，并重建关系引用。"""
    merged_entities = []
    entity_key_to_new_id = {}
    old_to_key_maps = []

    for result in (primary, secondary):
        source_map = {}
        for entity in result.get('entities', []):
            key = (entity['start'], entity['end'], entity['type'], entity['text'])
            if key not in entity_key_to_new_id:
                new_id = f"E{len(merged_entities) + 1}"
                entity_key_to_new_id[key] = new_id
                merged_entity = dict(entity)
                merged_entity['id'] = new_id
                merged_entities.append(merged_entity)
            source_map[entity['id']] = key
        old_to_key_maps.append(source_map)

    relation_keys = set()
    merged_relations = []
    for result, source_map in zip((primary, secondary), old_to_key_maps):
        for relation in result.get('relations', []):
            head_key = source_map.get(relation['head'])
            tail_key = source_map.get(relation['tail'])
            if not head_key or not tail_key:
                continue
            merged_key = (head_key, relation['type'], tail_key)
            if merged_key in relation_keys:
                continue
            relation_keys.add(merged_key)
            merged_relations.append(
                {
                    'id': f"R{len(merged_relations) + 1}",
                    'type': relation['type'],
                    'head': entity_key_to_new_id[head_key],
                    'tail': entity_key_to_new_id[tail_key],
                }
            )

    return {'entities': merged_entities, 'relations': merged_relations}


def _should_retry_with_focus(text: str, result: dict) -> bool:
    """判断是否需要对长文做聚焦重试，以提升关键实体召回。"""
    if len(text) < 2500:
        return False

    entities = result.get('entities', [])
    entity_types = {entity.get('type') for entity in entities}
    has_explicit_cve = bool(re.search(r"CVE-\d{4}-\d{4,7}", text, re.IGNORECASE))
    needs_priority_tactic = bool(re.search(r"\b(persistence|exfiltration|reconnaissance)\b", text, re.IGNORECASE))

    if not entities:
        return True
    if has_explicit_cve and 'Vulnerability' not in entity_types:
        return True
    if has_explicit_cve and 'Configuration' not in entity_types:
        return True
    if needs_priority_tactic and 'AttackTactic' not in entity_types:
        return True
    return False


def _derive_title_supplements(original_text: str, current_result: dict) -> dict:
    """从标题首行补关键实体，缓解长文中正文噪声导致的漏抽。"""
    first_line = next((line.strip() for line in original_text.splitlines() if line.strip()), "")
    if not first_line:
        return {'entities': [], 'relations': []}

    existing = {(entity['type'], entity['text']) for entity in current_result.get('entities', [])}
    current_types = {entity['type'] for entity in current_result.get('entities', [])}
    supplements = {'entities': [], 'relations': []}

    def add_entity(entity_type: str, entity_text: str) -> None:
        normalized = _normalize_entity_text(entity_type, entity_text)
        if not normalized:
            return
        if (entity_type, normalized) in existing:
            return
        span = _find_text_span(original_text, normalized)
        if span is None:
            return
        start, end, matched_text = span
        supplements['entities'].append(
            {
                'id': f"S{len(supplements['entities']) + 1}",
                'text': matched_text,
                'type': entity_type,
                'start': start,
                'end': end,
            }
        )
        existing.add((entity_type, matched_text))

    if 'Vulnerability' not in current_types:
        first_cve = re.search(r"CVE-\d{4}-\d{4,7}", first_line, re.IGNORECASE)
        if first_cve:
            add_entity('Vulnerability', first_cve.group(0))

    title_weakness = _extract_security_phrase(first_line)
    if title_weakness:
        if ('Weakness', title_weakness) not in existing:
            add_entity('Weakness', title_weakness)
        if ('AttackPattern', title_weakness) not in existing:
            add_entity('AttackPattern', title_weakness)

    if 'Configuration' not in current_types:
        boundary_indexes = []
        if title_weakness:
            boundary_indexes.append(first_line.lower().find(title_weakness.lower()))
        first_cve = re.search(r"CVE-\d{4}-\d{4,7}", first_line, re.IGNORECASE)
        if first_cve:
            boundary_indexes.append(first_cve.start())
        vulnerability_word = re.search(r"\bvulnerability\b", first_line, re.IGNORECASE)
        if vulnerability_word:
            boundary_indexes.append(vulnerability_word.start())
        split_index = min(index for index in boundary_indexes if index > 0) if any(index > 0 for index in boundary_indexes) else -1
        if split_index > 0:
            config_candidate = first_line[:split_index].strip(" -:|()")
            config_candidate = re.sub(r"\b(?:and|or)\b\s*$", "", config_candidate, flags=re.IGNORECASE).strip()
            if 0 < len(config_candidate.split()) <= 6:
                add_entity('Configuration', config_candidate)

    return supplements


class LLMExtractor:
    """LLM 知识抽取器"""

    def __init__(self, backend: str = 'openai_compatible', config: dict = None):
        self.backend = backend
        self.config = config or {}

        if backend == 'openai_compatible':
            self.call_fn = call_openai_compatible
        elif backend == 'ollama':
            self.call_fn = call_ollama
        else:
            raise ValueError(f"不支持的后端: {backend}")

    def extract(self, text: str, prompt: str, system_prompt: str = None, max_retries: int = 3) -> dict:
        """调用 LLM 抽取实体和关系"""
        if system_prompt is None:
            from prompts.manual_prompt import SYSTEM_PROMPT
            system_prompt = SYSTEM_PROMPT

        for attempt in range(max_retries):
            try:
                output = self.call_fn(prompt, system_prompt, self.config)
                result = parse_llm_output(output, original_text=text)

                # 如果结果为空且还有重试机会，继续
                if not result['entities'] and attempt < max_retries - 1:
                    continue

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠ LLM 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    print(f"✗ LLM 调用最终失败: {e}")
                    return {'entities': [], 'relations': []}

        return {'entities': [], 'relations': []}

    def extract_for_doc(self, cleaned_file: Path, prompt_builder, system_prompt: str = None) -> dict:
        """对单个文档进行 LLM 抽取"""
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            text = f.read()

        prompt = prompt_builder(text)
        result = self.extract(text, prompt, system_prompt)

        if _should_retry_with_focus(text, result):
            focus_text = _build_focus_text(text)
            if focus_text and focus_text != text:
                focus_prompt = prompt_builder(focus_text)
                focus_result = self.extract(text, focus_prompt, system_prompt)
                result = _merge_results(result, focus_result)

        title_supplements = _derive_title_supplements(text, result)
        if title_supplements['entities']:
            result = _merge_results(result, title_supplements)

        return result

    def extract_for_dataset(self, cleaned_dir: Path, doc_topics: Dict[str, str],
                            doc_cleaned_files: Dict[str, str], output_dir: Path,
                            prompt_builder=None, system_prompt: str = None,
                            prompt_name: str = 'manual',
                            split_name: Optional[str] = None,
                            extra_metadata: Optional[dict] = None) -> dict:
        """对整个数据集进行 LLM 抽取"""
        from prompts.manual_prompt import get_manual_prompt, SYSTEM_PROMPT

        if prompt_builder is None:
            prompt_builder = get_manual_prompt
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        output_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        skipped = []
        expected_doc_count = len(doc_topics)
        run_metadata = {
            'backend': self.backend,
            'model': self.config.get('model'),
            'prompt_type': prompt_name,
            'split': split_name,
            'doc_count_expected': expected_doc_count
        }
        if extra_metadata:
            run_metadata.update(extra_metadata)

        for doc_id, topic in doc_topics.items():
            # 找到 cleaned 文件
            if doc_id in doc_cleaned_files:
                cleaned_file = ROOT / doc_cleaned_files[doc_id]
            else:
                cleaned_file = cleaned_dir / topic / f"{doc_id}.txt"
                if not cleaned_file.exists():
                    cleaned_file = cleaned_dir / f"{doc_id}.txt"

            if not cleaned_file.exists():
                print(f"⚠ 清洗文本不存在: {cleaned_file}")
                skipped.append(doc_id)
                continue

            print(f"  处理: {doc_id}...", end=' ')

            # 读取文本
            with open(cleaned_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # 调用统一抽取流程，确保数据集与单文档共享相同的后处理/补全逻辑
            result = self.extract_for_doc(cleaned_file, prompt_builder, system_prompt)

            # 构建完整标注
            annotation = {
                'doc_id': doc_id,
                'title': '',
                'source_type': '',
                'source_name': '',
                'language': 'en',
                'cve_ids': [],
                'text': text,
                'entities': result['entities'],
                'relations': result['relations'],
                'quality': {
                    'annotation_stage': 'prediction',
                    'annotator': f'llm_{self.backend}',
                    'reviewer': None,
                    'review_status': 'auto_generated'
                },
                'experiment': {
                    'backend': self.backend,
                    'model': self.config.get('model'),
                    'prompt_type': prompt_name,
                    'split': split_name
                }
            }

            # 提取 CVE IDs
            annotation['cve_ids'] = [e['normalized_id'] for e in result['entities']
                                      if e['type'] == 'Vulnerability' and e.get('normalized_id')]

            # 保存
            output_file = output_dir / f"{doc_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)

            print(f"{len(result['entities'])}E/{len(result['relations'])}R")
            processed += 1

        run_metadata['doc_count_actual'] = processed
        run_metadata['skipped_doc_ids'] = skipped
        run_metadata_file = output_dir / '_run_metadata.json'
        with open(run_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(run_metadata, f, ensure_ascii=False, indent=2)

        return {
            'processed': processed,
            'skipped': skipped,
            'doc_count_expected': expected_doc_count,
            'run_metadata_file': str(run_metadata_file)
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LLM 知识抽取')
    parser.add_argument('--backend', choices=['openai_compatible', 'ollama'],
                        default='ollama', help='LLM 后端')
    parser.add_argument('--model', default='qwen2.5:latest', help='模型名称')
    parser.add_argument('--base_url', default=None, help='API 基础 URL')
    parser.add_argument('--cleaned_dir', default=str(ROOT / 'data' / 'cti_corpus' / 'cleaned'))
    parser.add_argument('--gold_dir', default=str(ROOT / 'data' / 'annotations' / 'gold'))
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--split', default='apo_dev', help='数据划分')
    parser.add_argument('--prompt', choices=['manual', 'manual_no_fewshot', 'apo'],
                        default='manual', help='Prompt 类型')
    parser.add_argument('--doc_ids', nargs='+', help='文档 ID 列表')

    args = parser.parse_args()

    # 配置
    config = {
        'model': args.model,
        'temperature': 0.1,
        'max_tokens': 4096,
        'top_p': 0.95
    }
    if args.base_url:
        config['base_url'] = args.base_url
    elif args.backend == 'ollama':
        config['base_url'] = 'http://localhost:11434'

    # 加载数据划分
    from run_experiments import load_split, load_doc_topics, load_doc_cleaned_files

    if args.doc_ids:
        doc_ids = args.doc_ids
    else:
        doc_ids = load_split(args.split)

    doc_topics = load_doc_topics(doc_ids)
    doc_cleaned_files = load_doc_cleaned_files(doc_ids)

    # 选择 prompt
    from prompts.manual_prompt import get_manual_prompt, get_manual_prompt_no_fewshot, SYSTEM_PROMPT as MANUAL_SYSTEM_PROMPT

    if args.prompt == 'manual':
        prompt_builder = get_manual_prompt
        system_prompt = MANUAL_SYSTEM_PROMPT
    elif args.prompt == 'manual_no_fewshot':
        prompt_builder = get_manual_prompt_no_fewshot
        system_prompt = MANUAL_SYSTEM_PROMPT
    elif args.prompt == 'apo':
        from prompts.apo_prompt import get_apo_prompt, SYSTEM_PROMPT as APO_SYSTEM_PROMPT
        prompt_builder = get_apo_prompt
        system_prompt = APO_SYSTEM_PROMPT
    else:
        prompt_builder = get_manual_prompt
        system_prompt = MANUAL_SYSTEM_PROMPT

    # 运行抽取
    print(f"\n{'='*60}")
    print(f"LLM 抽取: {args.split}")
    print(f"后端: {args.backend}, 模型: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"文档数: {len(doc_topics)}")
    print('='*60)

    extractor = LLMExtractor(backend=args.backend, config=config)
    summary = extractor.extract_for_dataset(
        Path(args.cleaned_dir), doc_topics, doc_cleaned_files,
        Path(args.output_dir), prompt_builder, system_prompt,
        prompt_name=args.prompt,
        split_name=args.split,
        extra_metadata={
            'temperature': config['temperature'],
            'top_p': config['top_p'],
            'max_tokens': config['max_tokens']
        }
    )

    print(f"\n完成: {summary['processed']} 篇处理, {len(summary['skipped'])} 篇跳过")
