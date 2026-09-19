"""Low-cost Critic/Editor capability preflight for Configuration APO.

The audit reuses bounded TRAIN error summaries from an earlier run.  It calls
only the APO Critic and Editor; it never invokes the extraction model and never
loads development or test Gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import apo_optimizer as AO  # noqa: E402


DEFAULT_SOURCE_RUN = (
    EXP_DIR
    / "results"
    / "apo_optimization_v5"
    / "configuration_recall_only_p0_20260831T125442Z"
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected_json_object:{path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round_error_summary(source_run: Path, round_index: int) -> tuple[dict, str]:
    path = source_run / f"round_{round_index:02d}_entity.json"
    value = _read_json(path)
    parents = value.get("parents") or []
    if not parents or not isinstance(parents[0], dict):
        raise ValueError(f"preflight_source_missing_parent_audit:{path}")
    summary = parents[0].get("error_summary")
    if not isinstance(summary, dict):
        raise ValueError(f"preflight_source_missing_error_summary:{path}")
    boundary_note = str(summary.get("data_boundary_note", ""))
    if "TRAIN" not in boundary_note or "dev and test" not in boundary_note:
        raise ValueError(f"preflight_source_is_not_train_only:{path}")
    return summary, _sha256(path)


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-z0-9]+", left.casefold()))
    right_tokens = set(re.findall(r"[a-z0-9]+", right.casefold()))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return round(len(left_tokens & right_tokens) / len(union), 6)


def _diversity_audit(candidates: list[dict]) -> dict:
    pairs = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            pairs.append(
                {
                    "pair": [left["id"], right["id"]],
                    "token_jaccard": _token_jaccard(
                        left.get("stage1_delta", ""),
                        right.get("stage1_delta", ""),
                    ),
                }
            )
    values = [item["token_jaccard"] for item in pairs]
    return {
        "pairs": pairs,
        "maximum_pairwise_token_jaccard": max(values, default=0.0),
        "mean_pairwise_token_jaccard": (
            round(sum(values) / len(values), 6) if values else 0.0
        ),
        "threshold": 0.80,
    }


def _candidate_checks(
    candidates: list[dict],
    phase_round: int,
    parent: dict,
) -> dict:
    expected = set(AO.CONFIGURATION_RECALL_STRATEGIES_BY_ROUND[phase_round])
    operation = "replace" if phase_round == 1 else "append"
    diversity = _diversity_audit(candidates)
    checks = {
        "candidate_count_is_four": len(candidates) == 4,
        "all_required_strategies_present": (
            {item.get("strategy") for item in candidates} == expected
        ),
        "stage1_operation_is_correct": all(
            item.get("stage1_edit_operation") == operation
            for item in candidates
        ),
        "stage2_is_empty_append": all(
            item.get("stage2_edit_operation") == "append"
            and not item.get("stage2_delta")
            for item in candidates
        ),
        "candidate_diversity_passed": (
            diversity["maximum_pairwise_token_jaccard"]
            <= diversity["threshold"] + 1e-12
        ),
    }
    if phase_round == 2:
        parent_guidance = parent["stage1_guidance"].strip()
        checks["round1_parent_is_preserved"] = all(
            item.get("stage1_guidance", "").startswith(parent_guidance + "\n")
            and item.get("stage1_delta", "").strip()
            and item.get("stage1_delta", "").strip() not in parent_guidance
            for item in candidates
        )
    return {"checks": checks, "diversity": diversity}


def _make_optimizer_clients():
    critic = AO.make_apo_extractor(
        temperature=AO.CRITIC_TEMPERATURE,
        model=AO.OPTIMIZER_MODEL,
        thinking=AO.OPTIMIZER_THINKING,
        reasoning_effort=AO.OPTIMIZER_REASONING_EFFORT,
        max_tokens=AO.OPTIMIZER_MAX_TOKENS,
        top_p=AO.OPTIMIZER_TOP_P,
    )
    critic_fallback = AO.make_apo_extractor(
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
    return critic, critic_fallback, editor


def run_preflight(source_run: Path, output_path: Path | None = None) -> dict:
    source_run = source_run.resolve()
    source_manifest_path = source_run / "run_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("feedback_split") != "train":
        raise ValueError("preflight_requires_train_feedback_source")
    if source_manifest.get("test_gold_loaded") is not False:
        raise ValueError("preflight_source_must_not_load_test_gold")

    required_efforts = {
        "critic": AO.OPTIMIZER_REASONING_EFFORT,
        "critic_fallback": AO.CRITIC_FALLBACK_REASONING_EFFORT,
        "editor": AO.EDITOR_REASONING_EFFORT,
    }
    if any(value != "high" for value in required_efforts.values()):
        raise ValueError(
            f"optimizer_preflight_requires_high_reasoning:{required_efforts}"
        )
    if not AO.OPTIMIZER_MODEL:
        raise ValueError("optimizer_preflight_requires_nonempty_model")
    if AO.EDITOR_MODEL != AO.OPTIMIZER_MODEL:
        raise ValueError("optimizer_preflight_requires_same_critic_editor_model")

    round1_errors, round1_sha256 = _round_error_summary(source_run, 1)
    round2_errors, round2_sha256 = _round_error_summary(source_run, 2)
    started_at = datetime.now(timezone.utc)
    if output_path is None:
        timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
        output_path = (
            AO.OUTPUT_ROOT / f"optimizer_preflight_{timestamp}" / "preflight.json"
        )
    output_path = output_path.resolve()
    artifact = {
        "schema_version": AO.SCHEMA_VERSION,
        "run_kind": "configuration_optimizer_capability_preflight",
        "started_at_utc": started_at.isoformat(),
        "source_run": str(source_run),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_round_sha256": {
            "round_1": round1_sha256,
            "round_2": round2_sha256,
        },
        "model": AO.OPTIMIZER_MODEL,
        "endpoint": AO._model_endpoint(AO.OPTIMIZER_MODEL),
        "reasoning": {
            "critic_thinking": AO.OPTIMIZER_THINKING,
            "critic_effort": AO.OPTIMIZER_REASONING_EFFORT,
            "critic_fallback_thinking": AO.CRITIC_FALLBACK_THINKING,
            "critic_fallback_effort": AO.CRITIC_FALLBACK_REASONING_EFFORT,
            "editor_thinking": AO.EDITOR_THINKING,
            "editor_effort": AO.EDITOR_REASONING_EFFORT,
        },
        "task_extraction_model_calls": 0,
        "dev_gold_loaded": False,
        "test_gold_loaded": False,
        "output_path": str(output_path),
    }
    try:
        critic, critic_fallback, editor = _make_optimizer_clients()
        prompt0 = AO._pair(
            "",
            "",
            id="preflight_p0",
            phase="entity",
            round_index=0,
            stage1_only=True,
            configuration_policy_override=True,
        )
        candidate_counter = 0

        def candidate_id(stage1: str, stage2: str) -> str:
            nonlocal candidate_counter
            candidate_counter += 1
            return f"preflight_p{candidate_counter}"

        gradient1 = AO.generate_textual_gradient(
            critic,
            prompt0,
            round1_errors,
            "entity",
            ("Configuration",),
            critic_fallback,
            objective_focus="configuration_recall_only",
            phase_round=1,
        )
        edited1 = AO.generate_candidates(
            editor,
            prompt0,
            gradient1["textual_gradient"],
            "entity",
            4,
            1,
            ("Configuration",),
            id_factory=candidate_id,
            objective_focus="configuration_recall_only",
            phase_round=1,
        )
        round1_candidates = edited1["candidates"]
        parent = next(
            (
                item
                for item in round1_candidates
                if item.get("strategy") == "repeated_alias_occurrence"
            ),
            round1_candidates[0] if round1_candidates else None,
        )
        if parent is None:
            raise RuntimeError("preflight_round1_produced_no_parent_candidate")

        gradient2 = AO.generate_textual_gradient(
            critic,
            parent,
            round2_errors,
            "entity",
            ("Configuration",),
            critic_fallback,
            objective_focus="configuration_recall_only",
            phase_round=2,
        )
        edited2 = AO.generate_candidates(
            editor,
            parent,
            gradient2["textual_gradient"],
            "entity",
            4,
            2,
            ("Configuration",),
            id_factory=candidate_id,
            objective_focus="configuration_recall_only",
            phase_round=2,
        )
        round2_candidates = edited2["candidates"]
        audit1 = _candidate_checks(round1_candidates, 1, prompt0)
        audit2 = _candidate_checks(round2_candidates, 2, parent)
        all_checks = {
            "high_reasoning_configured": all(
                value == "high" for value in required_efforts.values()
            ),
            "round1_gradient_route_passed": True,
            "round2_gradient_route_passed": True,
            **{f"round1_{key}": value for key, value in audit1["checks"].items()},
            **{f"round2_{key}": value for key, value in audit2["checks"].items()},
        }
        artifact.update(
            {
                "round_1": {
                    "gradient": gradient1,
                    "editor": edited1,
                    "audit": audit1,
                },
                "round_2": {
                    "parent_id": parent["id"],
                    "gradient": gradient2,
                    "editor": edited2,
                    "audit": audit2,
                },
                "checks": all_checks,
                "passed": all(all_checks.values()),
                "optimizer_semantic_calls": (
                    len(gradient1["raw_responses"])
                    + len(edited1["raw_responses"])
                    + len(gradient2["raw_responses"])
                    + len(edited2["raw_responses"])
                ),
                "optimizer_transport_attempts": sum(
                    int(item["trace"].get("attempts", 0))
                    for item in (gradient1, edited1, gradient2, edited2)
                ),
            }
        )
    except Exception as exc:
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
        description="Audit high-reasoning APO Critic/Editor without task extraction"
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(args.source_run, args.output)
    print(
        json.dumps(
            {
                "passed": result.get("passed"),
                "model": result.get("model"),
                "reasoning": result.get("reasoning"),
                "optimizer_semantic_calls": result.get(
                    "optimizer_semantic_calls", 0
                ),
                "optimizer_transport_attempts": result.get(
                    "optimizer_transport_attempts", 0
                ),
                "task_extraction_model_calls": result.get(
                    "task_extraction_model_calls", 0
                ),
                "checks": result.get("checks", {}),
                "error_type": result.get("error_type"),
                "error": result.get("error"),
                "output_path": result.get("output_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result.get("passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
