"""Minimal OpenCode parameter probe for the APO-v2 runtime.

The probe makes small, non-experimental calls.  It never loads Gold data,
does not call the test runner, and never writes or prints the API key.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from provider_config import _profile_from_name


EXP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = EXP_DIR / "results" / "opencode_model_parameter_probe_v3.json"
TASK_MODEL = "hy3"
OPTIMIZER_MODEL = "muse-spark-1.3-contributor"
SESSION_ID = "bron-v5-apo-v2-parameter-probe-v1"
USER_AGENT = "bron-apo-research/1.0"


def _response_text(payload: dict) -> str:
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text.strip()
    return ""


def _json_object(text: str) -> bool:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else ""
    try:
        return isinstance(json.loads(candidate), dict)
    except (TypeError, ValueError):
        return False


def _request(
    *,
    base_url: str,
    key: str,
    name: str,
    endpoint: str,
    body: dict,
    timeout: float,
) -> dict:
    url = base_url.rstrip("/") + endpoint
    started = time.perf_counter()
    result = {
        "name": name,
        "model": body.get("model"),
        "endpoint": endpoint,
        "parameters": {
            key: value
            for key, value in body.items()
            if key not in {"messages", "input", "instructions"}
        },
    }
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "x-opencode-session": SESSION_ID,
                "User-Agent": USER_AGENT,
            },
            json=body,
            timeout=timeout,
        )
        result["status_code"] = response.status_code
        result["latency_seconds"] = round(time.perf_counter() - started, 3)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        text = _response_text(payload)
        result.update(
            {
                "request_succeeded": response.ok,
                "nonempty_output": bool(text),
                "json_object_output": _json_object(text),
                "output_excerpt": text[:500],
                "response_model": payload.get("model"),
                "finish_status": (
                    (payload.get("choices") or [{}])[0].get("finish_reason")
                    if payload.get("choices")
                    else payload.get("status")
                ),
                "usage": payload.get("usage"),
                "error_excerpt": None if response.ok else response.text[:500],
            }
        )
    except Exception as exc:  # network failures belong in the audit artifact
        result.update(
            {
                "latency_seconds": round(time.perf_counter() - started, 3),
                "request_succeeded": False,
                "nonempty_output": False,
                "json_object_output": False,
                "error_type": type(exc).__name__,
                "error_excerpt": str(exc)[:500],
            }
        )
    return result


def run_probe(output: Path, timeout: float = 90.0) -> dict:
    profile = _profile_from_name("opencode")
    key = profile.get("key")
    if not key:
        raise RuntimeError("missing OpenCode key; expected experiments/v5/.opencode_key")
    base_url = str(profile["base_url"])

    model_response = requests.get(
        base_url.rstrip("/") + "/models",
        headers={
            "Authorization": f"Bearer {key}",
            "x-opencode-session": SESSION_ID,
            "User-Agent": USER_AGENT,
        },
        timeout=30,
    )
    model_response.raise_for_status()
    available_models = {
        str(item.get("id"))
        for item in model_response.json().get("data", [])
        if isinstance(item, dict) and item.get("id")
    }

    task_messages = [
        {
            "role": "system",
            "content": "Return only one valid JSON object. Do not use markdown.",
        },
        {
            "role": "user",
            "content": (
                "Return exactly this semantic content as JSON: "
                '{"ok":true,"role":"task-model"}'
            ),
        },
    ]
    optimizer_input = [
        {
            "role": "system",
            "content": "Return only one valid JSON object. Do not use markdown.",
        },
        {
            "role": "user",
            "content": (
                "Return a concise critic result with keys ok and role, where "
                'ok is true and role is "critic-editor".'
            ),
        },
    ]

    cases = [
        _request(
            base_url=base_url,
            key=key,
            name="hy3_chat_reasoning_none",
            endpoint="/chat/completions",
            body={
                "model": TASK_MODEL,
                "messages": task_messages,
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 256,
                "thinking": {"type": "disabled"},
                "reasoning_effort": "none",
            },
            timeout=timeout,
        ),
        _request(
            base_url=base_url,
            key=key,
            name="hy3_chat_reasoning_low",
            endpoint="/chat/completions",
            body={
                "model": TASK_MODEL,
                "messages": task_messages,
                "temperature": 0.0,
                "top_p": 0.95,
                "max_tokens": 256,
                "thinking": {"type": "disabled"},
                "reasoning_effort": "low",
            },
            timeout=timeout,
        ),
        _request(
            base_url=base_url,
            key=key,
            name="muse_1_3_responses_high",
            endpoint="/responses",
            body={
                "model": OPTIMIZER_MODEL,
                "input": optimizer_input,
                "temperature": 0.1,
                "top_p": 1.0,
                "max_output_tokens": 512,
                "reasoning": {"effort": "high"},
            },
            timeout=timeout,
        ),
        _request(
            base_url=base_url,
            key=key,
            name="muse_1_3_chat_high",
            endpoint="/chat/completions",
            body={
                "model": OPTIMIZER_MODEL,
                "messages": optimizer_input,
                "temperature": 0.1,
                "top_p": 1.0,
                "max_tokens": 512,
                "thinking": {"type": "disabled"},
                "reasoning_effort": "high",
            },
            timeout=timeout,
        ),
    ]

    required = {
        "hy3_chat_reasoning_none",
        "muse_1_3_responses_high",
    }
    passed_cases = {
        item["name"]
        for item in cases
        if item.get("request_succeeded")
        and item.get("nonempty_output")
        and item.get("json_object_output")
    }
    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": "opencode_apo_v2_parameter_probe",
        "provider": "opencode",
        "base_url": base_url,
        "session_header_present": True,
        "user_agent": USER_AGENT,
        "task_model": TASK_MODEL,
        "optimizer_model": OPTIMIZER_MODEL,
        "models_advertised": {
            TASK_MODEL: TASK_MODEL in available_models,
            OPTIMIZER_MODEL: OPTIMIZER_MODEL in available_models,
        },
        "gold_loaded": False,
        "dev_loaded": False,
        "test_loaded": False,
        "cases": cases,
        "required_cases": sorted(required),
        "passed_cases": sorted(passed_cases),
        "passed": required <= passed_cases,
        "selected_parameters": (
            {
                "task": {
                    "endpoint": "/chat/completions",
                    "model": TASK_MODEL,
                    "temperature": 0.0,
                    "top_p": 0.95,
                    "thinking": "disabled",
                    "reasoning_effort": "none",
                    "max_tokens": 4096,
                },
                "critic_editor": {
                    "endpoint": "/responses",
                    "model": OPTIMIZER_MODEL,
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "reasoning_effort": "high",
                    "critic_max_tokens": 16384,
                    "editor_max_tokens": 8192,
                },
            }
            if required <= passed_cases
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()
    report = run_probe(args.output, args.timeout)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "models_advertised": report["models_advertised"],
                "cases": [
                    {
                        "name": item["name"],
                        "status_code": item.get("status_code"),
                        "request_succeeded": item.get("request_succeeded"),
                        "nonempty_output": item.get("nonempty_output"),
                        "json_object_output": item.get("json_object_output"),
                        "latency_seconds": item.get("latency_seconds"),
                        "error_excerpt": item.get("error_excerpt"),
                    }
                    for item in report["cases"]
                ],
                "selected_parameters": report["selected_parameters"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
