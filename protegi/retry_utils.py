"""ProTeGi API 调用容错与重试工具。

针对长时间、大吞吐量的 LLM 调用场景，提供指数退避重试，
自动容忍 SSL EOF、连接中断、超时、限流等临时性网络抖动。
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

RETRYABLE_EXCEPTIONS = (
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "InternalServerError",
    "ConnectError",
    "TimeoutException",
    "RemoteDisconnected",
    "ConnectionResetError",
    "ProtocolError",
    "BadStatusLine",
)

RETRYABLE_MESSAGES = (
    "unexpected_eof_while_reading",
    "eof occurred in violation of protocol",
    "connection error",
    "connection reset",
    "timeout",
    "rate limit",
    "502",
    "503",
    "504",
    "broken pipe",
)


def is_retryable_error(exc: Exception) -> bool:
    """判断异常是否属于可重试的瞬态网络/服务错误。"""
    exc_type_name = type(exc).__name__
    exc_msg = str(exc).lower()

    if any(err_name in exc_type_name for err_name in RETRYABLE_EXCEPTIONS):
        return True
    if any(msg in exc_msg for msg in RETRYABLE_MESSAGES):
        return True
    # 检查直接引起该异常的 __cause__ 与 __context__
    if getattr(exc, "__cause__", None) and is_retryable_error(exc.__cause__):  # type: ignore[arg-type]
        return True
    if getattr(exc, "__context__", None) and is_retryable_error(exc.__context__):  # type: ignore[arg-type]
        return True
    return False


def retry_api_call(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 8,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    **kwargs: Any,
) -> Any:
    """使用指数退避重试执行 API 调用。"""
    delay = initial_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if not is_retryable_error(e) or attempt == max_retries:
                raise
            err_desc = f"{type(e).__name__}: {str(e)[:120]}"
            print(
                f"[ProTeGi Retry] 捕获网络抖动异常 ({err_desc})，第 {attempt}/{max_retries} 次重试，等待 {delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(max_delay, delay * backoff_factor)

    if last_exc:
        raise last_exc
