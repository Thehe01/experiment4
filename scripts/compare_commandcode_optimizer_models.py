"""Compare two Command Code models as APO Critic and Editor.

This invokes only the train-derived Configuration capability preflight.  It
does not call the task extraction model and does not open dev or test Gold.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
BASE_URL = "https://api.commandcode.ai/provider/v1"
MODELS = (
    "deepseek/deepseek-v4-pro",
    "meta/muse-spark-1.2-contributor",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _key_path() -> Path:
    for path in (
        EXP_DIR / "commandcode" / "_key",
        EXP_DIR / "commandcode_key",
    ):
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return path.resolve()
    raise FileNotFoundError(
        "Command Code API key not found at commandcode/_key or commandcode_key"
    )


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(".", "_")


def _environment(model: str, key_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "V5_PROVIDER": "relay",
            "V3_PROVIDER": "relay",
            "V5_RELAY_BASE_URL": BASE_URL,
            "V3_RELAY_BASE_URL": BASE_URL,
            "V5_RELAY_API_KEY_FILE": str(key_path),
            "V3_RELAY_API_KEY_FILE": str(key_path),
            # The task model remains frozen for the later APO experiment, but
            # this preflight makes zero task-model calls.
            "V5_RELAY_API_MODEL": "deepseek/deepseek-v4-flash-fast",
            "V3_RELAY_API_MODEL": "deepseek/deepseek-v4-flash-fast",
            "V5_LLM_THINKING": " ",
            "V3_LLM_THINKING": " ",
            "V5_LLM_REASONING_EFFORT": "low",
            "V3_LLM_REASONING_EFFORT": "low",
            "V5_LLM_TEMPERATURE": "0.1",
            "V3_LLM_TEMPERATURE": "0.1",
            "V5_LLM_MAX_TOKENS": "2048",
            "V3_LLM_MAX_TOKENS": "2048",
            "V5_APO_OPTIMIZER_MODEL": model,
            "V3_APO_OPTIMIZER_MODEL": model,
            "V5_APO_EDITOR_MODEL": model,
            "V3_APO_EDITOR_MODEL": model,
            # High reasoning is explicitly sent; the incompatible OpenCode
            # thinking object is omitted on the Command Code Chat wire.
            "V5_APO_OPTIMIZER_THINKING": " ",
            "V3_APO_OPTIMIZER_THINKING": " ",
            "V5_APO_EDITOR_THINKING": " ",
            "V3_APO_EDITOR_THINKING": " ",
            "V5_APO_CRITIC_FALLBACK_THINKING": " ",
            "V3_APO_CRITIC_FALLBACK_THINKING": " ",
        }
    )
    for name in ("V3_API_BASE_URL", "V3_API_MODEL", "V3_API_KEY"):
        env.pop(name, None)
    return env


def _seconds(artifact: dict) -> float | None:
    try:
        started = datetime.fromisoformat(artifact["started_at_utc"])
        completed = datetime.fromisoformat(artifact["completed_at_utc"])
    except (KeyError, TypeError, ValueError):
        return None
    return round((completed - started).total_seconds(), 3)


def _summary(artifact: dict) -> dict:
    checks = artifact.get("checks") or {}
    round1 = artifact.get("round_1") or {}
    round2 = artifact.get("round_2") or {}
    round1_candidates = (round1.get("editor") or {}).get("candidates") or []
    round2_candidates = (round2.get("editor") or {}).get("candidates") or []
    return {
        "passed": artifact.get("passed") is True,
        "checks_passed": sum(value is True for value in checks.values()),
        "checks_total": len(checks),
        "round1_candidate_count": len(round1_candidates),
        "round2_candidate_count": len(round2_candidates),
        "round1_strategies": [
            item.get("strategy") for item in round1_candidates
        ],
        "round2_strategies": [
            item.get("strategy") for item in round2_candidates
        ],
        "round1_max_pairwise_jaccard": (
            ((round1.get("audit") or {}).get("diversity") or {}).get(
                "maximum_pairwise_token_jaccard"
            )
        ),
        "round2_max_pairwise_jaccard": (
            ((round2.get("audit") or {}).get("diversity") or {}).get(
                "maximum_pairwise_token_jaccard"
            )
        ),
        "optimizer_semantic_calls": artifact.get("optimizer_semantic_calls", 0),
        "optimizer_transport_attempts": artifact.get(
            "optimizer_transport_attempts", 0
        ),
        "elapsed_seconds": _seconds(artifact),
        "error_type": artifact.get("error_type"),
        "error": artifact.get("error"),
    }


def main() -> int:
    key_path = _key_path()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = EXP_DIR / "results" / f"optimizer_model_probe_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_script = SCRIPT_DIR / "audit_apo_optimizer_capability.py"
    artifacts = {}
    processes = {}
    for model in MODELS:
        output_path = output_dir / f"{_slug(model)}.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(audit_script),
                "--output",
                str(output_path),
            ],
            cwd=EXP_DIR.parents[1],
            env=_environment(model, key_path),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        processes[model] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        }
        if output_path.exists():
            artifacts[model] = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            artifacts[model] = {
                "passed": False,
                "error_type": "MissingAuditArtifact",
                "error": f"preflight exited {completed.returncode}",
            }

    comparison = {
        "run_kind": "commandcode_optimizer_model_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "models": list(MODELS),
        "reasoning_effort": "high",
        "thinking": "omitted",
        "task_model": "deepseek/deepseek-v4-flash-fast",
        "task_extraction_model_calls": 0,
        "dev_gold_loaded": False,
        "test_gold_loaded": False,
        "results": {
            model: _summary(artifacts[model]) for model in MODELS
        },
        "processes": processes,
    }
    _write_json(output_dir / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"RESULT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
