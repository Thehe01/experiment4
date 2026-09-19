"""Provider profiles for the v5 OpenAI-compatible clients.

Secrets stay in files or environment variables. Select a profile with
``V5_PROVIDER=official``, ``opencode`` or ``relay``. The implementation keeps
the old ``V3_*`` names as a compatibility bridge so existing launch scripts do
not silently change the model configuration.
"""

from __future__ import annotations

import os
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parents[1]


def apply_v5_environment_aliases() -> None:
    """Expose V5_* settings to the legacy implementation without overwrites."""
    for name, value in list(os.environ.items()):
        if name.startswith("V5_"):
            os.environ.setdefault("V3_" + name[3:], value)


FORMAL_APO_RUNTIME_DEFAULTS = {
    "V3_PROVIDER": "opencode",
    "V3_OPENCODE_API_MODEL": "hy3",
    "V3_LLM_TEMPERATURE": "0.0",
    "V3_LLM_THINKING": "disabled",
    "V3_LLM_MAX_TOKENS": "4096",
    "V3_LLM_MAX_ESCALATED_TOKENS": "4096",
    "V3_API_TIMEOUT_SECONDS": "180",
    "V3_API_TRANSPORT_MAX_RETRIES": "0",
    "V3_LLM_WINDOW_CHARS": "3000",
    "V3_LLM_WINDOW_OVERLAP": "400",
    "V3_LLM_MAX_WORKERS": "16",
    "V3_APO_OPTIMIZER_MODEL": "muse-spark-1.3-contributor",
    "V3_APO_OPTIMIZER_THINKING": "disabled",
    "V3_APO_OPTIMIZER_REASONING_EFFORT": "xhigh",
    "V3_APO_OPTIMIZER_TOP_P": "1.0",
    "V3_APO_OPTIMIZER_MAX_TOKENS": "16384",
    "V3_APO_EDITOR_MODEL": "muse-spark-1.3-contributor",
    "V3_APO_EDITOR_THINKING": "disabled",
    "V3_APO_EDITOR_REASONING_EFFORT": "xhigh",
    "V3_APO_EDITOR_TOP_P": "1.0",
    "V3_APO_EDITOR_MAX_TOKENS": "8192",
    "V3_APO_CRITIC_TEMPERATURE": "0.1",
    "V3_APO_EDITOR_TEMPERATURE": "0.1",
    "V3_APO_CRITIC_FALLBACK_REASONING_EFFORT": "none",
}


def apply_formal_apo_runtime_defaults() -> None:
    """Apply the formal APO profile without overriding explicit shell values."""
    apply_v5_environment_aliases()
    for name, value in FORMAL_APO_RUNTIME_DEFAULTS.items():
        os.environ.setdefault(name, value)
    apply_v5_environment_aliases()


def _read_key(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = EXP_DIR / path
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def _profile_from_name(name: str) -> dict[str, str | None]:
    normalized = name.strip().casefold()
    if normalized in {"official", "deepseek", "deepseek-official"}:
        return {
            "name": "official",
            "base_url": "https://api.deepseek.com",
            "key": _read_key(os.environ.get("V3_OFFICIAL_API_KEY_FILE"))
            or _read_key(".deepseek_api_key"),
            "model": os.environ.get("V3_OFFICIAL_API_MODEL", "deepseek-v4-flash"),
        }
    if normalized in {"opencode", "go", "opencode-go"}:
        return {
            "name": "opencode",
            "base_url": "https://opencode.ai/zen/go/v1",
            "key": _read_key(os.environ.get("V3_OPENCODE_API_KEY_FILE"))
            or _read_key(".opencode_key")
            or _read_key(".api_key"),
            "model": os.environ.get("V3_OPENCODE_API_MODEL", "hy3"),
        }
    if normalized in {"relay", "proxy", "gateway", "transit"}:
        base_url = os.environ.get(
            "V3_RELAY_BASE_URL", "https://ai.kdysite.cloud/v1"
        ).strip()
        key = _read_key(
            os.environ.get("V3_RELAY_API_KEY_FILE", ".relay_api_key")
        )
        if not key:
            key = os.environ.get("V3_RELAY_API_KEY", "").strip() or None
        return {
            "name": "relay",
            "base_url": base_url,
            "key": key,
            "model": os.environ.get("V3_RELAY_API_MODEL", "deepseek-v4-flash"),
        }
    raise ValueError(
        f"未知 V5_PROVIDER={name!r}；可选值：official、opencode、relay"
    )


def resolve_provider() -> dict[str, str | None]:
    """Resolve the selected provider without exposing the API key."""
    selected = os.environ.get("V3_PROVIDER", "").strip()
    if selected:
        profile = _profile_from_name(selected)
        # Explicit V3_API_* overrides remain useful for one-off experiments.
        profile["base_url"] = os.environ.get("V3_API_BASE_URL", profile["base_url"])
        profile["model"] = os.environ.get("V3_API_MODEL", profile["model"])
        if os.environ.get("V3_API_KEY"):
            profile["key"] = os.environ["V3_API_KEY"].strip()
        if not profile["base_url"]:
            raise ValueError(
                "relay 未配置地址；请设置 V5_RELAY_BASE_URL=https://.../v1"
            )
        # A provider profile is also used for offline rule/evaluation runs.
        # Require the key only when make_extractor creates a network client.
        return profile

    # Legacy mode: preserve existing V3_API_BASE_URL/V3_API_MODEL behavior.
    base_url = os.environ.get(
        "V3_API_BASE_URL",
        os.environ.get("V2_API_BASE_URL", "https://api.deepseek.com"),
    )
    is_official = "api.deepseek.com" in base_url.casefold()
    key = (
        os.environ.get("V3_API_KEY", "").strip()
        or _read_key(".deepseek_api_key" if is_official else ".api_key")
    )
    return {
        "name": "official" if is_official else "custom",
        "base_url": base_url,
        "key": key,
        "model": os.environ.get("V3_API_MODEL", "deepseek-v4-flash"),
    }


def public_provider_config() -> dict[str, str | None]:
    profile = resolve_provider()
    return {key: value for key, value in profile.items() if key != "key"}

