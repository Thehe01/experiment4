"""ProTeGi Task Runtime 单一来源契约。

11 个冻结字段在以下链路中必须完全一致：
YAML → run_protegi effective runtime → config["effective_task_runtime"]
→ optimizer summary → promotion 比较 → protegi_final_artifact.task_runtime
→ predict_llm_protegi / entity cache manifest。

本模块是唯一的字段集合与校验实现；
entity_cache / promote / llm_methods 应复用此处常量与校验，
禁止各自维护一套列表后漂移。
"""

from __future__ import annotations

from typing import Any, Dict

TASK_RUNTIME_FIELDS = (
    "model",
    "max_workers",
    "temperature",
    "thinking",
    "reasoning_effort",
    "top_p",
    "max_tokens",
    "window_chars",
    "window_overlap",
    "document_abbreviation_context",
    "vulnerability_anchored_backfill",
)


def validate_task_runtime(runtime: Any, *, label: str = "task_runtime") -> Dict[str, Any]:
    """校验 11 字段齐全且类型/范围合法，返回原 dict（不做类型强制转换）。

    缺字段、类型非法、范围非法一律抛 ValueError，禁止 fallback 到环境默认值。
    """
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} 必须为 dict，当前={type(runtime).__name__}")
    for field in TASK_RUNTIME_FIELDS:
        if field not in runtime or runtime[field] is None:
            # top_p/max_tokens 在旧逻辑允许 None？正式 runtime 不允许。
            # effective runtime 必须 11 字段齐全；None 视为缺失。
            raise ValueError(f"{label} 缺少必需字段 {field}")
    model = runtime["model"]
    max_workers = runtime["max_workers"]
    temperature = runtime["temperature"]
    thinking = runtime["thinking"]
    reasoning_effort = runtime["reasoning_effort"]
    top_p = runtime["top_p"]
    max_tokens = runtime["max_tokens"]
    window_chars = runtime["window_chars"]
    window_overlap = runtime["window_overlap"]
    abbrev = runtime["document_abbreviation_context"]
    backfill = runtime["vulnerability_anchored_backfill"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{label}.model 必须为非空字符串")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool):
        raise ValueError(f"{label}.max_workers 必须为 int")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError(f"{label}.temperature 类型非法")
    if not isinstance(thinking, str):
        raise ValueError(f"{label}.thinking 必须为字符串")
    if not isinstance(reasoning_effort, str):
        raise ValueError(f"{label}.reasoning_effort 必须为字符串")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValueError(f"{label}.top_p 类型非法")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError(f"{label}.max_tokens 必须为 int")
    if not isinstance(window_chars, int) or isinstance(window_chars, bool):
        raise ValueError(f"{label}.window_chars 必须为 int")
    if not isinstance(window_overlap, int) or isinstance(window_overlap, bool):
        raise ValueError(f"{label}.window_overlap 必须为 int")
    if not isinstance(abbrev, bool):
        raise ValueError(f"{label}.document_abbreviation_context 必须为 bool")
    if not isinstance(backfill, bool):
        raise ValueError(f"{label}.vulnerability_anchored_backfill 必须为 bool")
    if not max_workers > 0:
        raise ValueError(f"{label}.max_workers 必须 > 0")
    if not window_chars > 0:
        raise ValueError(f"{label}.window_chars 必须 > 0")
    if not 0 <= window_overlap < window_chars:
        raise ValueError(
            f"{label} 窗口参数非法: "
            f"window_chars={window_chars}, window_overlap={window_overlap}"
        )
    return runtime


def build_effective_task_runtime(
    *,
    model: str,
    max_workers: int,
    temperature: float,
    thinking: str,
    reasoning_effort: str,
    top_p: float,
    max_tokens: int,
    window_chars: int,
    window_overlap: int,
    document_abbreviation_context: bool,
    vulnerability_anchored_backfill: bool,
) -> Dict[str, Any]:
    """用已规范化的 effective 值构造 11 字段 runtime 并校验。"""
    runtime = {
        "model": str(model),
        "max_workers": int(max_workers),
        "temperature": float(temperature),
        "thinking": str(thinking).strip().lower(),
        "reasoning_effort": str(reasoning_effort).strip().lower(),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "window_chars": int(window_chars),
        "window_overlap": int(window_overlap),
        "document_abbreviation_context": bool(document_abbreviation_context),
        "vulnerability_anchored_backfill": bool(vulnerability_anchored_backfill),
    }
    return validate_task_runtime(runtime, label="effective_task_runtime")
