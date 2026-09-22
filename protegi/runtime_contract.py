"""ProTeGi Task Runtime 单一来源契约。

全部冻结字段在以下链路中必须完全一致：
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
    "provider",
    "base_url",
    "endpoint",
    "max_workers",
    "temperature",
    "thinking",
    "reasoning_effort",
    "top_p",
    "max_tokens",
    "max_escalated_tokens",
    "request_timeout_seconds",
    "transport_max_retries",
    "transient_retry_max_attempts",
    "budget_exhaustion_retries",
    "window_chars",
    "window_overlap",
    "dense_run_split",
    "dense_min_ids",
    "dense_min_span",
    "dense_gap",
    "dense_max_ids",
    "dense_seam",
    "document_abbreviation_context",
    "vulnerability_anchored_backfill",
)

# 重叠窗口中的实体/关系只计入一个优化评价样本。
SELECTION_WINDOW_OWNERSHIP = "midpoint-partition-v1"


def validate_task_runtime(runtime: Any, *, label: str = "task_runtime") -> Dict[str, Any]:
    """校验全部字段齐全且类型/范围合法，返回原 dict（不做类型强制转换）。

    缺字段、类型非法、范围非法一律抛 ValueError，禁止 fallback 到环境默认值。
    """
    if not isinstance(runtime, dict):
        raise ValueError(f"{label} 必须为 dict，当前={type(runtime).__name__}")
    for field in TASK_RUNTIME_FIELDS:
        if field not in runtime or runtime[field] is None:
            # top_p/max_tokens 在旧逻辑允许 None？正式 runtime 不允许。
            # effective runtime 必须全部字段齐全；None 视为缺失。
            raise ValueError(f"{label} 缺少必需字段 {field}")
    model = runtime["model"]
    provider = runtime["provider"]
    base_url = runtime["base_url"]
    endpoint = runtime["endpoint"]
    max_workers = runtime["max_workers"]
    temperature = runtime["temperature"]
    thinking = runtime["thinking"]
    reasoning_effort = runtime["reasoning_effort"]
    top_p = runtime["top_p"]
    max_tokens = runtime["max_tokens"]
    max_escalated_tokens = runtime["max_escalated_tokens"]
    request_timeout_seconds = runtime["request_timeout_seconds"]
    transport_max_retries = runtime["transport_max_retries"]
    transient_retry_max_attempts = runtime["transient_retry_max_attempts"]
    budget_exhaustion_retries = runtime["budget_exhaustion_retries"]
    window_chars = runtime["window_chars"]
    window_overlap = runtime["window_overlap"]
    dense_run_split = runtime["dense_run_split"]
    dense_min_ids = runtime["dense_min_ids"]
    dense_min_span = runtime["dense_min_span"]
    dense_gap = runtime["dense_gap"]
    dense_max_ids = runtime["dense_max_ids"]
    dense_seam = runtime["dense_seam"]
    abbrev = runtime["document_abbreviation_context"]
    backfill = runtime["vulnerability_anchored_backfill"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{label}.model 必须为非空字符串")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"{label}.provider 必须为非空字符串")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError(f"{label}.base_url 必须为非空字符串")
    if not isinstance(endpoint, str) or not endpoint.startswith("/v1/"):
        raise ValueError(f"{label}.endpoint 必须为 /v1/ 开头的字符串")
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
    if not isinstance(max_escalated_tokens, int) or isinstance(
        max_escalated_tokens, bool
    ):
        raise ValueError(f"{label}.max_escalated_tokens 必须为 int")
    if not isinstance(request_timeout_seconds, (int, float)) or isinstance(
        request_timeout_seconds, bool
    ):
        raise ValueError(f"{label}.request_timeout_seconds 类型非法")
    for field, value in (
        ("transport_max_retries", transport_max_retries),
        ("transient_retry_max_attempts", transient_retry_max_attempts),
        ("budget_exhaustion_retries", budget_exhaustion_retries),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{field} 必须为 int")
    if not isinstance(window_chars, int) or isinstance(window_chars, bool):
        raise ValueError(f"{label}.window_chars 必须为 int")
    if not isinstance(window_overlap, int) or isinstance(window_overlap, bool):
        raise ValueError(f"{label}.window_overlap 必须为 int")
    if not isinstance(abbrev, bool):
        raise ValueError(f"{label}.document_abbreviation_context 必须为 bool")
    if not isinstance(backfill, bool):
        raise ValueError(f"{label}.vulnerability_anchored_backfill 必须为 bool")
    if not isinstance(dense_run_split, bool):
        raise ValueError(f"{label}.dense_run_split 必须为 bool")
    for field, value in (
        ("dense_min_ids", dense_min_ids),
        ("dense_min_span", dense_min_span),
        ("dense_gap", dense_gap),
        ("dense_max_ids", dense_max_ids),
        ("dense_seam", dense_seam),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{field} 必须为 int")
    if not max_workers > 0:
        raise ValueError(f"{label}.max_workers 必须 > 0")
    if not window_chars > 0:
        raise ValueError(f"{label}.window_chars 必须 > 0")
    if max_tokens <= 0 or max_escalated_tokens < max_tokens:
        raise ValueError(
            f"{label} token 预算非法: max_tokens={max_tokens}, "
            f"max_escalated_tokens={max_escalated_tokens}"
        )
    if request_timeout_seconds <= 0:
        raise ValueError(f"{label}.request_timeout_seconds 必须 > 0")
    if transport_max_retries < 0 or budget_exhaustion_retries < 0:
        raise ValueError(f"{label} retry 次数必须非负")
    if transient_retry_max_attempts <= 0:
        raise ValueError(f"{label}.transient_retry_max_attempts 必须 > 0")
    if dense_min_ids <= 0 or dense_min_span <= 0 or dense_max_ids <= 0:
        raise ValueError(f"{label} dense 正整数参数非法")
    if dense_gap < 0 or dense_seam < 0:
        raise ValueError(f"{label} dense gap/seam 必须非负")
    if not 0 <= window_overlap < window_chars:
        raise ValueError(
            f"{label} 窗口参数非法: "
            f"window_chars={window_chars}, window_overlap={window_overlap}"
        )
    return runtime


def build_effective_task_runtime(
    *,
    model: str,
    provider: str,
    base_url: str,
    endpoint: str,
    max_workers: int,
    temperature: float,
    thinking: str,
    reasoning_effort: str,
    top_p: float,
    max_tokens: int,
    max_escalated_tokens: int,
    request_timeout_seconds: float,
    transport_max_retries: int,
    transient_retry_max_attempts: int,
    budget_exhaustion_retries: int,
    window_chars: int,
    window_overlap: int,
    dense_run_split: bool,
    dense_min_ids: int,
    dense_min_span: int,
    dense_gap: int,
    dense_max_ids: int,
    dense_seam: int,
    document_abbreviation_context: bool,
    vulnerability_anchored_backfill: bool,
) -> Dict[str, Any]:
    """用已规范化的 effective 值构造完整 runtime 并校验。"""
    runtime = {
        "model": str(model),
        "provider": str(provider),
        "base_url": str(base_url).rstrip("/"),
        "endpoint": str(endpoint),
        "max_workers": int(max_workers),
        "temperature": float(temperature),
        "thinking": str(thinking).strip().lower(),
        "reasoning_effort": str(reasoning_effort).strip().lower(),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "max_escalated_tokens": int(max_escalated_tokens),
        "request_timeout_seconds": float(request_timeout_seconds),
        "transport_max_retries": int(transport_max_retries),
        "transient_retry_max_attempts": int(transient_retry_max_attempts),
        "budget_exhaustion_retries": int(budget_exhaustion_retries),
        "window_chars": int(window_chars),
        "window_overlap": int(window_overlap),
        "dense_run_split": bool(dense_run_split),
        "dense_min_ids": int(dense_min_ids),
        "dense_min_span": int(dense_min_span),
        "dense_gap": int(dense_gap),
        "dense_max_ids": int(dense_max_ids),
        "dense_seam": int(dense_seam),
        "document_abbreviation_context": bool(document_abbreviation_context),
        "vulnerability_anchored_backfill": bool(vulnerability_anchored_backfill),
    }
    return validate_task_runtime(runtime, label="effective_task_runtime")
