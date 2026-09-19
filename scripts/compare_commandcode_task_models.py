"""Compare Command Code task models on the frozen Configuration dev screen.

The parent process launches one isolated worker per model so provider settings
are resolved before importing the v5 extraction code.  Only the fixed v7 dev
partition is scored; test Gold is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
DEFAULT_MODELS = (
    "tencent/hy3-paid",
    "deepseek/deepseek-v4-flash-fast",
)
BASE_URL = "https://api.commandcode.ai/provider/v1"
SEED = 20260730
MAX_WORKERS = 32


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _key_path() -> Path:
    candidates = (
        EXP_DIR / "commandcode" / "_key",
        EXP_DIR / "commandcode_key",
    )
    for path in candidates:
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return path.resolve()
    raise FileNotFoundError(
        "Command Code API key not found at commandcode/_key or commandcode_key"
    )


def _slug(model: str) -> str:
    return model.replace("/", "__").replace(".", "_")


def _configuration_view(metrics: dict) -> dict:
    strict = metrics.get("candidate_entity_by_type", {}).get(
        "Configuration", {}
    )
    overlap = metrics.get("candidate_overlap_entity_by_type", {}).get(
        "Configuration", {}
    )
    strict_tp = float(strict.get("tp", 0.0))
    overlap_tp = float(overlap.get("tp", 0.0))
    return {
        "strict": {
            field: float(strict.get(field, 0.0))
            for field in ("precision", "recall", "f1", "tp", "fp", "fn")
        },
        "overlap": {
            field: float(overlap.get(field, 0.0))
            for field in ("precision", "recall", "f1", "tp", "fp", "fn")
        },
        "overlap_tp_minus_strict_tp": round(overlap_tp - strict_tp, 6),
        "failed_stage_calls": int(metrics.get("failed_stage_calls", 0)),
        "api_call_attempts": int(metrics.get("api_call_attempts", 0)),
        "elapsed_seconds": float(metrics.get("elapsed_seconds", 0.0)),
        "candidate_entity_macro_f1": float(
            metrics.get("candidate_entity_macro_f1", 0.0)
        ),
    }


def _worker(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(SCRIPT_DIR))
    import apo_optimizer as optimizer

    _, dev_paths, split = optimizer._load_split()
    selected = optimizer.select_target_stratified_evaluation_batch(
        dev_paths,
        batch_size=min(args.selection_dev_size, len(dev_paths)),
        seed=SEED + 100_000,
        target_labels=("Configuration",),
    )
    test_ids = set(split.get("test", []))
    selected_ids = [path.stem for path in selected]
    if test_ids.intersection(selected_ids):
        raise RuntimeError("task_model_probe_selected_test_document")

    prompt = optimizer._initial_pair_from_artifact(None)
    prompt["stage1_only"] = True
    prompt["configuration_policy_override"] = True
    metrics = optimizer._evaluate_repeated_with_failure_recovery(
        prompt,
        selected,
        args.output_dir / "predictions" / _slug(args.worker_model),
        args.repeats,
        cache_namespace_prefix=(
            f"commandcode_task_model_probe_{args.run_id}_{_slug(args.worker_model)}"
        ),
    )
    split_hash = hashlib.sha256(optimizer.SPLIT_FILE.read_bytes()).hexdigest()
    result = {
        "probe_kind": "commandcode_task_model_configuration_p0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.worker_model,
        "runtime_config": optimizer.runtime_config(),
        "schema_version": optimizer.SCHEMA_VERSION,
        "split_sha256": split_hash,
        "selection_dev_documents": len(selected),
        "selection_dev_ids": selected_ids,
        "repeats": args.repeats,
        "stage1_only": True,
        "prompt_id": prompt["id"],
        "test_documents_loaded": 0,
        "configuration": _configuration_view(metrics),
        "metrics": metrics,
    }
    _write_json(args.worker_output, result)
    print(json.dumps(result["configuration"], ensure_ascii=False))
    return 0


def _worker_environment(model: str, key_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "V5_PROVIDER": "relay",
            "V3_PROVIDER": "relay",
            "V5_RELAY_BASE_URL": BASE_URL,
            "V3_RELAY_BASE_URL": BASE_URL,
            "V5_RELAY_API_KEY_FILE": str(key_path),
            "V3_RELAY_API_KEY_FILE": str(key_path),
            "V5_RELAY_API_MODEL": model,
            "V3_RELAY_API_MODEL": model,
            # Omit the old OpenCode-specific request fields while preserving
            # all frozen P0 task parameters.
            "V5_LLM_THINKING": " ",
            "V3_LLM_THINKING": " ",
            "V5_LLM_REASONING_EFFORT": "low",
            "V3_LLM_REASONING_EFFORT": "low",
            "V5_LLM_TEMPERATURE": "0.1",
            "V3_LLM_TEMPERATURE": "0.1",
            "V5_LLM_MAX_TOKENS": "2048",
            "V3_LLM_MAX_TOKENS": "2048",
            "V5_LLM_MAX_ESCALATED_TOKENS": "2048",
            "V3_LLM_MAX_ESCALATED_TOKENS": "2048",
            "V5_LLM_MAX_WORKERS": str(MAX_WORKERS),
            "V3_LLM_MAX_WORKERS": str(MAX_WORKERS),
        }
    )
    for name in ("V3_API_BASE_URL", "V3_API_MODEL", "V3_API_KEY"):
        env.pop(name, None)
    return env


def _main(args: argparse.Namespace) -> int:
    key_path = _key_path()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = EXP_DIR / "results" / f"task_model_probe_commandcode_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    result_paths = []
    for model in DEFAULT_MODELS:
        result_path = output_dir / f"{_slug(model)}.json"
        command = [
            sys.executable,
            "-X",
            "utf8",
            str(Path(__file__).resolve()),
            "--worker-model",
            model,
            "--worker-output",
            str(result_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            run_id,
            "--selection-dev-size",
            str(args.selection_dev_size),
            "--repeats",
            str(args.repeats),
        ]
        completed = subprocess.run(
            command,
            cwd=EXP_DIR.parents[1],
            env=_worker_environment(model, key_path),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            failure = {
                "model": model,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
            _write_json(output_dir / "failure.json", failure)
            raise RuntimeError(f"task model probe failed for {model}")
        result_paths.append(result_path)

    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    ids = [result["selection_dev_ids"] for result in results]
    if any(item != ids[0] for item in ids[1:]):
        raise RuntimeError("model_probe_dev_document_mismatch")
    comparison = {
        "probe_kind": "commandcode_task_model_configuration_p0_comparison",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "thinking": "omitted",
        "reasoning_effort": "low",
        "max_tokens": 2048,
        "temperature": 0.1,
        "max_workers": MAX_WORKERS,
        "models": [result["model"] for result in results],
        "selection_dev_documents": len(ids[0]),
        "selection_dev_ids": ids[0],
        "repeats": args.repeats,
        "test_documents_loaded": 0,
        "results": {
            result["model"]: result["configuration"] for result in results
        },
    }
    _write_json(output_dir / "comparison.json", comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"RESULT_DIR={output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dev-size", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--worker-model")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.selection_dev_size < 1 or args.repeats < 1:
        parser.error("selection-dev-size and repeats must be positive")
    if args.worker_model:
        required = (args.worker_output, args.output_dir, args.run_id)
        if not all(required):
            parser.error("worker mode requires output paths and run id")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    raise SystemExit(_worker(parsed) if parsed.worker_model else _main(parsed))
