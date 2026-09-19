"""Dev/test-blind Critic/Editor gate for Stage-1 + relation APO-v2.

The audit replays bounded TRAIN error summaries from prior audited runs.  It
calls only the Muse Critic and Editor, never the hy3 task model, and never loads
development or test Gold.  The Configuration/CPE Stage-1 round and each of the
three registered relation rounds must produce valid in-scope edits without a
transport or budget error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import apo_optimizer as AO  # noqa: E402


DEFAULT_OUTPUT = (
    EXP_DIR / "results" / "opencode_apo_v2_stage1_relation_preflight_v1.json"
)
CURRENT_RUN = (
    EXP_DIR
    / "results"
    / "apo_optimization_v5"
    / "apo_v2_relation_tuning_small_20260914T172245Z"
)
STAGE1_SOURCE = CURRENT_RUN / "round_01_relation.json"
SOURCE_CASES = (
    (
        "affects",
        CURRENT_RUN / "round_01_relation.json",
    ),
    (
        "instantiates",
        CURRENT_RUN / "round_02_relation.json",
    ),
    (
        "exploited_by",
        CURRENT_RUN / "round_03_relation.json",
    ),
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _train_error_summary(path: Path, expected_label: str) -> dict:
    manifest = _read_json(path.parent / "run_manifest.json")
    if manifest.get("feedback_split") != "train":
        raise ValueError(f"source_feedback_is_not_train:{path}")
    if manifest.get("test_gold_loaded") is not False:
        raise ValueError(f"source_loaded_test_gold:{path}")
    round_artifact = _read_json(path)
    if round_artifact.get("phase") != "relation":
        raise ValueError(f"source_is_not_relation_round:{path}")
    if round_artifact.get("atomic_target_labels") != [expected_label]:
        raise ValueError(f"source_target_mismatch:{path}")
    parents = round_artifact.get("parents") or []
    if not parents or not isinstance(parents[0], dict):
        raise ValueError(f"source_missing_parent:{path}")
    summary = parents[0].get("error_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"source_missing_error_summary:{path}")
    boundary_note = str(summary.get("data_boundary_note", ""))
    if "TRAIN" not in boundary_note or "dev and test" not in boundary_note:
        raise ValueError(f"source_is_not_train_only:{path}")
    return summary


def _current_train_error_summary(path: Path) -> dict:
    """Recompute Stage-1 diagnostics from audited TRAIN predictions only."""
    manifest = _read_json(path.parent / "run_manifest.json")
    if manifest.get("feedback_split") != "train":
        raise ValueError(f"source_feedback_is_not_train:{path}")
    if manifest.get("test_gold_loaded") is not False:
        raise ValueError(f"source_loaded_test_gold:{path}")
    round_artifact = _read_json(path)
    parents = round_artifact.get("parents") or []
    if not parents or not isinstance(parents[0], dict):
        raise ValueError(f"source_missing_parent:{path}")
    parent = parents[0]
    train_ids = list(parent.get("training_batch_ids") or [])
    prediction_dir = Path(
        str((parent.get("training_metrics") or {}).get("prediction_dir", ""))
    )
    split = _read_json(AO.SPLIT_FILE)
    registered_train = set(split.get("train") or [])
    if not train_ids or any(item not in registered_train for item in train_ids):
        raise ValueError(f"source_batch_is_not_registered_train:{path}")
    if not prediction_dir.is_dir():
        raise ValueError(f"source_prediction_dir_missing:{prediction_dir}")
    gold_paths = [AO.GOLD_DIR / f"{doc_id}.json" for doc_id in train_ids]
    if any(not gold_path.is_file() for gold_path in gold_paths):
        raise ValueError(f"source_train_gold_missing:{path}")
    summary = AO.build_error_summary(gold_paths, prediction_dir)
    boundary_note = str(summary.get("data_boundary_note", ""))
    if "TRAIN" not in boundary_note or "dev and test" not in boundary_note:
        raise ValueError(f"source_is_not_train_only:{path}")
    return summary


def _make_clients():
    critic = AO.make_apo_extractor(
        temperature=AO.CRITIC_TEMPERATURE,
        model=AO.OPTIMIZER_MODEL,
        thinking=AO.OPTIMIZER_THINKING,
        reasoning_effort=AO.OPTIMIZER_REASONING_EFFORT,
        max_tokens=AO.OPTIMIZER_MAX_TOKENS,
        top_p=AO.OPTIMIZER_TOP_P,
    )
    fallback = AO.make_apo_extractor(
        temperature=AO.CRITIC_TEMPERATURE,
        model=AO.OPTIMIZER_MODEL,
        thinking=AO.CRITIC_FALLBACK_THINKING,
        reasoning_effort=AO.CRITIC_FALLBACK_REASONING_EFFORT,
        max_tokens=AO.OPTIMIZER_MAX_TOKENS,
        top_p=AO.OPTIMIZER_TOP_P,
    )
    editor = AO.make_apo_extractor(
        temperature=AO.EDITOR_TEMPERATURE,
        model=AO.EDITOR_MODEL,
        thinking=AO.EDITOR_THINKING,
        reasoning_effort=AO.EDITOR_REASONING_EFFORT,
        max_tokens=AO.EDITOR_MAX_TOKENS,
        top_p=AO.EDITOR_TOP_P,
    )
    return critic, fallback, editor


def run_preflight(output_path: Path = DEFAULT_OUTPUT) -> dict:
    started_at = datetime.now(timezone.utc)
    output_path = output_path.resolve()
    optimizer_script = Path(AO.__file__).resolve()
    preflight_script = Path(__file__).resolve()
    artifact = {
        "schema_version": AO.SCHEMA_VERSION,
        "run_kind": "apo_v2_stage1_relation_optimizer_capability_preflight",
        "started_at_utc": started_at.isoformat(),
        "model": AO.OPTIMIZER_MODEL,
        "endpoint": AO._model_endpoint(AO.OPTIMIZER_MODEL),
        "runtime": {
            "critic_reasoning_effort": AO.OPTIMIZER_REASONING_EFFORT,
            "critic_max_output_tokens": AO.OPTIMIZER_MAX_TOKENS,
            "editor_reasoning_effort": AO.EDITOR_REASONING_EFFORT,
            "editor_max_output_tokens": AO.EDITOR_MAX_TOKENS,
        },
        "task_extraction_model_calls": 0,
        "dev_gold_loaded": False,
        "test_gold_loaded": False,
        "optimizer_script": str(optimizer_script),
        "optimizer_script_sha256": _sha256(optimizer_script),
        "preflight_script": str(preflight_script),
        "preflight_script_sha256": _sha256(preflight_script),
        "registered_schedule": {
            "entity": ["Configuration"],
            "relation": ["affects", "instantiates", "exploited_by"],
        },
        "source_artifacts": {
            "Configuration": {
                "path": str(STAGE1_SOURCE),
                "sha256": _sha256(STAGE1_SOURCE),
            },
            **{
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in SOURCE_CASES
            },
        },
        "output_path": str(output_path),
    }
    try:
        if AO.OPTIMIZER_MODEL != "muse-spark-1.3-contributor":
            raise ValueError("preflight_requires_muse_spark_1_3_contributor")
        if AO.EDITOR_MODEL != AO.OPTIMIZER_MODEL:
            raise ValueError("critic_editor_model_mismatch")
        if AO.OPTIMIZER_REASONING_EFFORT != "high":
            raise ValueError("critic_requires_high_reasoning")
        if AO.EDITOR_REASONING_EFFORT != "high":
            raise ValueError("editor_requires_high_reasoning")
        if AO.OPTIMIZER_MAX_TOKENS < 16384:
            raise ValueError("critic_requires_16384_output_tokens")
        if AO.EDITOR_MAX_TOKENS < 8192:
            raise ValueError("editor_requires_8192_output_tokens")

        critic, fallback, editor = _make_clients()
        prompt = AO._pair(
            "",
            "",
            id="stage1_relation_preflight_p0",
            phase="entity",
            round_index=0,
        )
        cases = {}
        all_checks = {}
        semantic_calls = 0
        transport_attempts = 0
        stage1_summary = _current_train_error_summary(STAGE1_SOURCE)
        stage1_gradient = AO.generate_textual_gradient(
            critic,
            prompt,
            stage1_summary,
            "entity",
            ("Configuration",),
            fallback,
            objective_focus="configuration_cpe_then_relations",
            phase_round=1,
        )
        stage1_edited = AO.generate_candidates(
            editor,
            prompt,
            stage1_gradient["textual_gradient"],
            "entity",
            len(AO.CONFIGURATION_CPE_STRATEGIES),
            1,
            ("Configuration",),
            objective_focus="configuration_cpe_then_relations",
            phase_round=1,
        )
        stage1_candidates = stage1_edited["candidates"]
        stage1_checks = {
            "candidate_count_is_four": len(stage1_candidates)
            == len(AO.CONFIGURATION_CPE_STRATEGIES),
            "strategies_match_registration": [
                candidate.get("strategy") for candidate in stage1_candidates
            ] == list(AO.CONFIGURATION_CPE_STRATEGIES),
            "stage1_deltas_are_nonempty": bool(stage1_candidates)
            and all(candidate.get("stage1_delta") for candidate in stage1_candidates),
            "stage2_is_unchanged": bool(stage1_candidates)
            and all(not candidate.get("stage2_delta") for candidate in stage1_candidates),
            "target_is_program_attached": bool(stage1_candidates)
            and all(
                candidate.get("target_labels") == ["Configuration"]
                for candidate in stage1_candidates
            ),
            "critic_has_no_transport_error": not stage1_gradient["trace"].get(
                "errors"
            ),
            "editor_has_no_transport_error": not stage1_edited["trace"].get(
                "errors"
            ),
        }
        all_checks.update(
            {
                f"Configuration_{name}": value
                for name, value in stage1_checks.items()
            }
        )
        cases["Configuration"] = {
            "textual_gradient": stage1_gradient["textual_gradient"],
            "candidates": stage1_candidates,
            "checks": stage1_checks,
            "critic_trace": stage1_gradient["trace"],
            "editor_trace": stage1_edited["trace"],
            "critic_input_chars": stage1_gradient["input_chars"],
            "critic_output_chars": stage1_gradient["output_chars"],
            "editor_input_chars": stage1_edited["input_chars"],
            "editor_output_chars": stage1_edited["output_chars"],
        }
        semantic_calls += len(stage1_gradient["raw_responses"]) + len(
            stage1_edited["raw_responses"]
        )
        transport_attempts += int(stage1_gradient["trace"].get("attempts", 0))
        transport_attempts += int(stage1_edited["trace"].get("attempts", 0))
        if stage1_candidates:
            prompt = stage1_candidates[0]

        for round_index, (label, source_path) in enumerate(
            SOURCE_CASES, start=1
        ):
            summary = _train_error_summary(source_path, label)
            gradient = AO.generate_textual_gradient(
                critic,
                prompt,
                summary,
                "relation",
                (label,),
                fallback,
                objective_focus="configuration_cpe_then_relations",
                phase_round=round_index,
            )
            edited = AO.generate_candidates(
                editor,
                prompt,
                gradient["textual_gradient"],
                "relation",
                1,
                round_index,
                (label,),
                id_factory=lambda _s1, _s2, target=label: f"preflight_{target}",
                objective_focus="configuration_cpe_then_relations",
                phase_round=round_index,
            )
            candidates = edited["candidates"]
            candidate = candidates[0] if candidates else {}
            checks = {
                "candidate_count_is_one": len(candidates) == 1,
                "stage1_is_unchanged": candidate.get("stage1_guidance")
                == prompt.get("stage1_guidance"),
                "stage2_delta_is_nonempty": bool(candidate.get("stage2_delta")),
                "target_is_program_attached": candidate.get("target_labels")
                == [label],
                "critic_has_no_transport_error": not gradient["trace"].get(
                    "errors"
                ),
                "editor_has_no_transport_error": not edited["trace"].get(
                    "errors"
                ),
            }
            all_checks.update(
                {f"{label}_{name}": value for name, value in checks.items()}
            )
            cases[label] = {
                "textual_gradient": gradient["textual_gradient"],
                "candidate": candidate,
                "checks": checks,
                "critic_trace": gradient["trace"],
                "editor_trace": edited["trace"],
                "critic_input_chars": gradient["input_chars"],
                "critic_output_chars": gradient["output_chars"],
                "editor_input_chars": edited["input_chars"],
                "editor_output_chars": edited["output_chars"],
            }
            semantic_calls += len(gradient["raw_responses"]) + len(
                edited["raw_responses"]
            )
            transport_attempts += int(gradient["trace"].get("attempts", 0))
            transport_attempts += int(edited["trace"].get("attempts", 0))
        artifact.update(
            {
                "cases": cases,
                "checks": all_checks,
                "optimizer_semantic_calls": semantic_calls,
                "optimizer_transport_attempts": transport_attempts,
                "passed": all(all_checks.values()),
            }
        )
    except Exception as exc:  # noqa: BLE001
        artifact.update(
            {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    artifact["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    AO._write_json(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Muse 1.3 for Stage-1 + relation APO-v2"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_preflight(args.output)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "passed",
                    "model",
                    "endpoint",
                    "optimizer_semantic_calls",
                    "optimizer_transport_attempts",
                    "task_extraction_model_calls",
                    "checks",
                    "error_type",
                    "error",
                    "output_path",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.get("passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
